"""Train/export alphaow's entity-transformer value net.

Input NPZ must contain:
  tokens: [N, 77, 24] float32
  mask:   [N, 77] bool/uint8 where True means valid token
  labels: [N] float32 target value in [-1, 1]

Exports AOWV version 3, consumed by Rust `value_net::TransformerWeights`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_TOKENS = 77
TOKEN_DIM = 24
SUMMARY_V2_DIM = 46
ROUTE_DIM = 20
SUMMARY_DIM = SUMMARY_V2_DIM + ROUTE_DIM
MAGIC = 0x564F4157
EPISODE_STEPS = 500
HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE_WEIGHTS = [
    HERE / "weights" / "v2_replays.bin",
    HERE / "weights" / "v2_h64_preliminary.bin",
]


class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"


def log(msg: str):
    print(msg, flush=True)


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def launch_dashboard(metrics_path: Path, dashboard_path: Path, current_proc):
    if current_proc is not None and current_proc.poll() is None:
        return current_proc
    try:
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("value_dashboard.py")), str(metrics_path), str(dashboard_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def save_training_state(
    path: Path,
    *,
    epoch_completed: int,
    model,
    opt,
    best_state,
    best_val: float,
    best_sign: float,
    summary_mean: np.ndarray,
    summary_std: np.ndarray,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch_completed": epoch_completed,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "best_state": best_state,
            "best_val": best_val,
            "best_sign": best_sign,
            "summary_mean": summary_mean,
            "summary_std": summary_std,
            "config": {
                "d_model": args.d_model,
                "layers": args.layers,
                "heads": args.heads,
                "ff_dim": args.ff_dim,
                "summary_hidden": args.summary_hidden,
                "summary_dim": SUMMARY_DIM,
                "target_mode": args.target_mode,
                "time_coef": args.time_coef,
                "episode_steps": args.episode_steps,
                "seed": args.seed,
            },
        },
        tmp,
    )
    tmp.replace(path)


def tag(name: str, color: str = C.CYAN) -> str:
    return f"{color}{name:>8s}{C.RESET}"


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def human_int(n: int) -> str:
    return f"{n:,}"


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.1f} {unit}" if unit != "B" else f"{int(x)} B"
        x /= 1024.0


def metric_key_for_path(path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem).strip("_")
    return f"baseline_{name or 'value_net'}"


def read_u32(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def take_f32(buf: bytes, cursor: int, n: int) -> tuple[np.ndarray, int]:
    size = 4 * n
    if cursor + size > len(buf):
        raise ValueError("weights file ended early")
    arr = np.frombuffer(buf, dtype="<f4", count=n, offset=cursor).astype(np.float32).copy()
    return arr, cursor + size


class BaselineMlp:
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w3: np.ndarray | None,
        b3: float,
    ):
        self.input_dim = input_dim
        self.hidden = hidden
        self.w1 = torch.from_numpy(w1.reshape(hidden, input_dim))
        self.b1 = torch.from_numpy(b1)
        if w3 is None:
            self.w2 = torch.from_numpy(w2.reshape(1, hidden))
            self.b2 = torch.tensor([b3], dtype=torch.float32)
            self.w3 = None
            self.b3 = None
        else:
            hidden2 = b2.shape[0]
            self.w2 = torch.from_numpy(w2.reshape(hidden2, hidden))
            self.b2 = torch.from_numpy(b2)
            self.w3 = torch.from_numpy(w3.reshape(1, hidden2))
            self.b3 = torch.tensor([b3], dtype=torch.float32)

    def predict(self, x: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32, copy=False))
                h = torch.relu(F.linear(xb, self.w1, self.b1))
                if self.w3 is None:
                    y = F.linear(h, self.w2, self.b2)
                else:
                    h = torch.relu(F.linear(h, self.w2, self.b2))
                    y = F.linear(h, self.w3, self.b3)
                outs.append(torch.tanh(y.squeeze(-1)).numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)


def load_baseline_mlp(path: Path) -> tuple[BaselineMlp, str]:
    buf = path.read_bytes()
    magic = read_u32(buf, 0)
    version = read_u32(buf, 4)
    if magic != MAGIC or version not in (1, 2):
        raise ValueError(f"{path.name} is not an AOWV MLP v1/v2 file")
    input_dim = read_u32(buf, 8)
    hidden = read_u32(buf, 12)
    cursor = 16
    hidden2 = 0
    if version == 2:
        hidden2 = read_u32(buf, 16)
        cursor = 20
    if input_dim not in (SUMMARY_V2_DIM, SUMMARY_DIM):
        raise ValueError(f"{path.name} input_dim={input_dim} is not available from summary features")
    w1, cursor = take_f32(buf, cursor, hidden * input_dim)
    b1, cursor = take_f32(buf, cursor, hidden)
    if version == 2:
        w2, cursor = take_f32(buf, cursor, hidden2 * hidden)
        b2, cursor = take_f32(buf, cursor, hidden2)
        w3, cursor = take_f32(buf, cursor, hidden2)
        b3, cursor = take_f32(buf, cursor, 1)
        return BaselineMlp(input_dim, hidden, w1, b1, w2, b2, w3, float(b3[0])), f"mlp-v2 {input_dim}->{hidden}->{hidden2}->1"
    w2, cursor = take_f32(buf, cursor, hidden)
    b2, cursor = take_f32(buf, cursor, 1)
    return BaselineMlp(input_dim, hidden, w1, b1, w2, np.zeros(0, dtype=np.float32), None, float(b2[0])), f"mlp-v1 {input_dim}->{hidden}->1"


def bucket_sign_metrics(
    prefix: str,
    score: np.ndarray,
    target: np.ndarray,
    steps: np.ndarray,
    episode_steps: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, mask in bucket_masks(steps, episode_steps):
        key = f"bucket_{name}_{prefix}"
        metrics[key] = float(np.mean((score[mask] > 0) == (target[mask] > 0))) if np.any(mask) else 0.0
    return metrics


def score_value_baselines(
    paths: list[Path],
    summary_val: np.ndarray,
    labels_val: np.ndarray,
    steps_val: np.ndarray,
    episode_steps: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for path in paths:
        if not path.exists():
            log(f"{tag('base', C.YELLOW)} skip missing value baseline {path}")
            continue
        try:
            model, desc = load_baseline_mlp(path)
            pred = model.predict(summary_val[:, : model.input_dim])
        except Exception as exc:
            log(f"{tag('base', C.YELLOW)} skip {path.name}: {exc}")
            continue
        key = metric_key_for_path(path)
        scores[key] = float(np.mean((pred > 0) == (labels_val > 0)))
        scores.update(bucket_sign_metrics(key, pred, labels_val, steps_val, episode_steps))
        log(f"{tag('base')} val sign {key}={scores[key]:.3f} ({desc})")
    return scores


def sanitize_value(x: torch.Tensor) -> torch.Tensor:
    # The value target is final reward in [-1, 1]. Keeping the prediction in
    # that range also avoids rare MPS loss-kernel nonsense from poisoning logs.
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = sanitize_value(pred)
    target = sanitize_value(target)
    diff = (pred - target).abs()
    return torch.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5).mean()


def sign_accuracy_from_score(score: np.ndarray, labels: np.ndarray) -> float:
    pred = np.where(score > 0.0, 1.0, -1.0)
    return float((pred == np.where(labels > 0.0, 1.0, -1.0)).mean())


def reward_sign(labels: np.ndarray) -> np.ndarray:
    return np.where(labels > 0.0, 1.0, np.where(labels < 0.0, -1.0, 0.0)).astype(np.float32)


def game_keys_from_meta(meta: np.ndarray, data_paths: list[str] | None = None, arrays: list[np.ndarray] | None = None) -> np.ndarray:
    games = meta[:, 0].astype(np.int64).copy()
    if data_paths is not None and arrays is not None and len(data_paths) > 1:
        cursor = 0
        for path, arr in zip(data_paths, arrays):
            n = arr.shape[0]
            games[cursor : cursor + n] += (hash(path) & 0xFFFF) * 100_000
            cursor += n
    return games


def infer_finish_steps(meta: np.ndarray, games: np.ndarray) -> np.ndarray:
    steps = meta[:, 1].astype(np.int32)
    finish_by_game: dict[int, int] = {}
    for game, step in zip(games, steps):
        g = int(game)
        finish_by_game[g] = max(finish_by_game.get(g, 0), int(step))
    return np.array([finish_by_game[int(g)] for g in games], dtype=np.int32)


def make_targets(
    labels: np.ndarray,
    meta: np.ndarray,
    games: np.ndarray,
    target_mode: str,
    time_coef: float,
    episode_steps: int,
    finish_steps: np.ndarray | None = None,
) -> np.ndarray:
    raw = reward_sign(labels)
    if target_mode == "outcome":
        return raw
    if finish_steps is None:
        finish_steps = infer_finish_steps(meta, games)
    finish_frac = np.clip(finish_steps.astype(np.float32) / max(float(episode_steps), 1.0), 0.0, 1.0)
    coef = np.clip(float(time_coef), 0.0, 0.95)
    magnitude = 1.0 - coef * finish_frac
    return (raw * magnitude).astype(np.float32)


def bucket_masks(steps: np.ndarray, episode_steps: int) -> list[tuple[str, np.ndarray]]:
    s = steps.astype(np.float32)
    a = episode_steps * 0.2
    b = episode_steps * 0.6
    return [
        ("opener", s < a),
        ("mid", (s >= a) & (s < b)),
        ("end", s >= b),
    ]


def bucket_metrics(pred: np.ndarray, target: np.ndarray, steps: np.ndarray, episode_steps: int) -> tuple[str, dict[str, float]]:
    parts = []
    metrics: dict[str, float] = {}
    for name, mask in bucket_masks(steps, episode_steps):
        if not np.any(mask):
            parts.append(f"{name}=n/a")
            metrics[f"bucket_{name}_sign"] = 0.0
            metrics[f"bucket_{name}_loss"] = 0.0
            continue
        p = pred[mask]
        y = target[mask]
        sign = float(((p > 0) == (y > 0)).mean())
        diff = np.abs(p - y)
        sl1 = float(np.mean(np.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)))
        metrics[f"bucket_{name}_sign"] = sign
        metrics[f"bucket_{name}_loss"] = sl1
        parts.append(f"{name}={sign:.3f}/{sl1:.3f}")
    return " ".join(parts), metrics


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ships_from_token(v: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(v, 0.0, 2.0) * np.log(1001.0)).astype(np.float32)


def fleet_speed_np(ships: np.ndarray) -> np.ndarray:
    ships = np.maximum(ships.astype(np.float32), 1.0)
    scaled = np.power(np.log(ships) / np.log(1000.0), 1.5)
    return 1.0 + 5.0 * np.clip(scaled, 0.0, 1.0)


def augment_summary_routes(tokens: np.ndarray, summary: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    if summary.shape[1] >= SUMMARY_DIM:
        return summary[:, :SUMMARY_DIM].astype(np.float32, copy=False)
    out = np.zeros((tokens.shape[0], SUMMARY_DIM), dtype=np.float32)
    out[:, : summary.shape[1]] = summary.astype(np.float32, copy=False)
    for start in range(0, tokens.shape[0], chunk_size):
        end = min(tokens.shape[0], start + chunk_size)
        p = tokens[start:end, 1 : 1 + 44, :]
        exists = p[..., 0] > 0.5
        is_me = (p[..., 2] > 0.5) & exists
        is_enemy = (p[..., 3] > 0.5) & exists
        is_neutral = (p[..., 4] > 0.5) & exists
        ships = ships_from_token(p[..., 8])
        x = p[..., 5] * 50.0 + 50.0
        y = p[..., 6] * 50.0 + 50.0
        dx = x[:, :, None] - x[:, None, :]
        dy = y[:, :, None] - y[:, None, :]
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        same = np.eye(p.shape[1], dtype=bool)[None, :, :]
        sx = x[:, :, None]
        sy = y[:, :, None]
        tx = x[:, None, :]
        ty = y[:, None, :]
        vx = tx - sx
        vy = ty - sy
        wx = 50.0 - sx
        wy = 50.0 - sy
        vv = np.maximum(vx * vx + vy * vy, 1e-6)
        t = np.clip((wx * vx + wy * vy) / vv, 0.0, 1.0)
        cx = sx + t * vx
        cy = sy + t * vy
        sun_block = ((cx - 50.0) ** 2 + (cy - 50.0) ** 2) < 100.0
        src_ships = ships[:, :, None]
        dst_ships = ships[:, None, :]
        send = np.minimum(np.maximum(src_ships, 1.0), np.maximum(dst_ships + 1.0, 1.0))
        travel = np.minimum(dist / np.maximum(fleet_speed_np(send), 0.01), 500.0)
        pair_valid = exists[:, :, None] & exists[:, None, :] & (~same)
        rels = [
            (is_me[:, :, None] & is_neutral[:, None, :]),
            (is_me[:, :, None] & is_enemy[:, None, :]),
            (is_enemy[:, :, None] & is_me[:, None, :]),
            (is_enemy[:, :, None] & is_neutral[:, None, :]),
        ]
        route = np.zeros((end - start, ROUTE_DIM), dtype=np.float32)
        for g, rel in enumerate(rels):
            mask = rel & pair_valid
            count = mask.sum(axis=(1, 2)).astype(np.float32)
            n = np.maximum(count, 1.0)
            masked_t = np.where(mask, travel, 1e9)
            min_t = masked_t.min(axis=(1, 2))
            mean_t = np.where(count > 0, np.where(mask, travel, 0.0).sum(axis=(1, 2)) / n, 500.0)
            clear = np.where(count > 0, (mask & (~sun_block)).sum(axis=(1, 2)).astype(np.float32) / n, 0.0)
            feasible = np.where(count > 0, (mask & (src_ships > dst_ships)).sum(axis=(1, 2)).astype(np.float32) / n, 0.0)
            base = g * 5
            route[:, base] = np.clip(count / 100.0, 0.0, 2.0)
            route[:, base + 1] = np.clip(np.where(count > 0, min_t, 500.0) / 100.0, 0.0, 5.0)
            route[:, base + 2] = np.clip(mean_t / 100.0, 0.0, 5.0)
            route[:, base + 3] = clear
            route[:, base + 4] = feasible
        out[start:end, SUMMARY_V2_DIM:] = route
    return out


def derive_summary_from_tokens(tokens: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
    """Approximate summary_v2 from token NPZs made before exact summary export.

    Newer datasets should include `summary_v2` directly. This fallback lets
    existing token files stay usable.
    """
    out = np.zeros((tokens.shape[0], SUMMARY_V2_DIM), dtype=np.float32)
    for start in range(0, tokens.shape[0], chunk_size):
        end = min(tokens.shape[0], start + chunk_size)
        p = tokens[start:end, 1 : 1 + 44, :]
        f = tokens[start:end, 1 + 44 :, :]
        exists = (p[..., 0] > 0.5).astype(np.float32)
        is_me = p[..., 2] * exists
        is_enemy = p[..., 3] * exists
        is_neutral = p[..., 4] * exists
        is_static = p[..., 10] * exists
        is_orbit = p[..., 11] * exists
        is_comet = p[..., 12] * exists
        prod = p[..., 9] * 5.0
        ships = ships_from_token(p[..., 8])
        ext_me = p[..., 17] * exists
        ext_enemy = p[..., 18] * exists
        ext_neutral = p[..., 19] * exists
        ext_ships = ships_from_token(p[..., 20])
        fleet_exists = f[..., 1] > 0.5
        fleet_me = f[..., 2] * fleet_exists
        fleet_enemy = f[..., 3] * fleet_exists
        fleet_ships = ships_from_token(f[..., 8])

        x = p[..., 5] * 50.0 + 50.0
        y = p[..., 6] * 50.0 + 50.0
        dx = x[:, :, None] - x[:, None, :]
        dy = y[:, :, None] - y[:, None, :]
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        big = np.float32(1e6)

        def closer_counts(owner_me, owner_enemy, target_mask):
            d_me = np.where(owner_me[:, None, :] > 0.5, dist, big).min(axis=2)
            d_enemy = np.where(owner_enemy[:, None, :] > 0.5, dist, big).min(axis=2)
            return (((d_me < d_enemy) & (target_mask > 0.5))).sum(axis=1).astype(np.float32)

        def player_block(owner, other):
            neutral_closer = closer_counts(owner, other, is_neutral)
            enemy_closer = closer_counts(owner, other, other)
            return np.stack(
                [
                    (owner * ships).sum(axis=1),
                    np.zeros(end - start, dtype=np.float32),
                    (owner * is_static).sum(axis=1),
                    (owner * is_orbit).sum(axis=1),
                    (owner * is_comet).sum(axis=1),
                    (owner * is_static * prod).sum(axis=1),
                    (owner * is_orbit * prod).sum(axis=1),
                    (owner * is_comet * prod).sum(axis=1),
                    neutral_closer,
                    enemy_closer,
                ],
                axis=1,
            )

        cur_me = player_block(is_me, is_enemy)
        cur_enemy = player_block(is_enemy, is_me)
        cur_me[:, 1] = (fleet_me * fleet_ships).sum(axis=1)
        cur_enemy[:, 1] = (fleet_enemy * fleet_ships).sum(axis=1)

        def extrap_block(owner, other):
            neutral_closer = closer_counts(owner, other, ext_neutral)
            enemy_closer = closer_counts(owner, other, other)
            return np.stack(
                [
                    (owner * ext_ships).sum(axis=1),
                    (owner * is_static).sum(axis=1),
                    (owner * is_orbit).sum(axis=1),
                    (owner * is_comet).sum(axis=1),
                    (owner * is_static * prod).sum(axis=1),
                    (owner * is_orbit * prod).sum(axis=1),
                    (owner * is_comet * prod).sum(axis=1),
                    neutral_closer,
                    enemy_closer,
                ],
                axis=1,
            )

        neutral = np.stack(
            [
                (is_neutral * ships).sum(axis=1),
                (is_neutral * is_static).sum(axis=1),
                (is_neutral * is_orbit).sum(axis=1),
                (is_neutral * is_comet).sum(axis=1),
                (is_neutral * is_static * prod).sum(axis=1),
                (is_neutral * is_orbit * prod).sum(axis=1),
                (is_neutral * is_comet * prod).sum(axis=1),
                (p[..., 16] * 500.0 * is_comet).sum(axis=1),
            ],
            axis=1,
        )
        out[start:end] = np.concatenate(
            [cur_me, cur_enemy, extrap_block(ext_me, ext_enemy), extrap_block(ext_enemy, ext_me), neutral],
            axis=1,
        )
    return augment_summary_routes(tokens, out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True, dropout=0.0)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x, key_padding_mask):
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x


class EntityTransformer(nn.Module):
    def __init__(self, d_model: int, layers: int, heads: int, ff_dim: int, summary_hidden: int = 64, summary_dim: int = SUMMARY_DIM):
        super().__init__()
        self.d_model = d_model
        self.layers_n = layers
        self.heads = heads
        self.ff_dim = ff_dim
        self.summary_hidden = summary_hidden
        self.summary_dim = summary_dim
        self.cls = nn.Parameter(torch.zeros(d_model))
        self.embed = nn.Linear(TOKEN_DIM, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, heads, ff_dim) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        if summary_hidden > 0:
            self.summary_fc = nn.Linear(summary_dim, summary_hidden)
            self.head = nn.Linear(d_model + summary_hidden, 1)
        else:
            self.summary_fc = None
            self.head = nn.Linear(d_model, 1)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, tokens, mask, summary=None):
        # tokens already contains a blank CLS slot at [:, 0]; Rust uses the
        # learned cls vector directly, so do the same here.
        b = tokens.shape[0]
        x = self.embed(tokens)
        x[:, 0, :] = self.cls.view(1, -1).expand(b, -1)
        mask = mask.clone()
        mask[:, 0] = True
        key_padding_mask = ~mask.bool()
        for block in self.blocks:
            x = block(x, key_padding_mask)
        cls = self.ln_f(x[:, 0, :])
        if self.summary_hidden > 0:
            if summary is None:
                raise ValueError("summary features required when summary_hidden > 0")
            s = torch.relu(self.summary_fc(summary))
            cls = torch.cat([cls, s], dim=-1)
        y = self.head(cls).squeeze(-1)
        return torch.tanh(y)


def write_aowv_transformer(out_path: Path, model: EntityTransformer, summary_mean=None, summary_std=None):
    buf = bytearray()
    buf.extend(struct.pack("<I", MAGIC))
    buf.extend(struct.pack("<I", 4 if model.summary_hidden > 0 else 3))
    buf.extend(struct.pack("<I", TOKEN_DIM))
    buf.extend(struct.pack("<I", model.d_model))
    buf.extend(struct.pack("<I", model.layers_n))
    buf.extend(struct.pack("<I", model.heads))
    buf.extend(struct.pack("<I", model.ff_dim))
    buf.extend(struct.pack("<I", MAX_TOKENS))
    if model.summary_hidden > 0:
        buf.extend(struct.pack("<I", model.summary_dim))
        buf.extend(struct.pack("<I", model.summary_hidden))

    def add(arr):
        a = arr.detach().cpu().numpy().astype(np.float32)
        buf.extend(a.tobytes(order="C"))

    add(model.cls)
    add(model.embed.weight)
    add(model.embed.bias)
    for block in model.blocks:
        add(block.ln1.weight)
        add(block.ln1.bias)
        # PyTorch stores q/k/v as one packed [3*d_model, d_model] matrix,
        # exactly the layout Rust expects.
        add(block.attn.in_proj_weight)
        add(block.attn.in_proj_bias)
        add(block.attn.out_proj.weight)
        add(block.attn.out_proj.bias)
        add(block.ln2.weight)
        add(block.ln2.bias)
        add(block.ff[0].weight)
        add(block.ff[0].bias)
        add(block.ff[2].weight)
        add(block.ff[2].bias)
    add(model.ln_f.weight)
    add(model.ln_f.bias)
    if model.summary_hidden > 0:
        if summary_mean is None or summary_std is None:
            raise ValueError("summary normalization stats required for hybrid export")
        mean = np.asarray(summary_mean, dtype=np.float32)
        std = np.asarray(summary_std, dtype=np.float32)
        w = model.summary_fc.weight.detach().cpu().numpy().astype(np.float32)
        b = model.summary_fc.bias.detach().cpu().numpy().astype(np.float32)
        w_fold = w / std.reshape(1, -1)
        b_fold = b - w_fold @ mean
        buf.extend(w_fold.astype(np.float32).tobytes(order="C"))
        buf.extend(b_fold.astype(np.float32).tobytes(order="C"))
    add(model.head.weight.reshape(-1))
    buf.extend(struct.pack("<f", float(model.head.bias.detach().cpu().numpy().reshape(-1)[0])))

    out_path.write_bytes(bytes(buf))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=128)
    p.add_argument("--summary-hidden", type=int, default=64, help="0 disables the summary-v2 branch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--target-mode", choices=["time", "outcome", "stored"], default="time")
    p.add_argument("--time-coef", type=float, default=0.10)
    p.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    p.add_argument("--metrics-path", default=None, help="epoch metrics JSONL path; default is next to --out")
    p.add_argument("--dashboard-path", default=None, help="HTML dashboard path; default is next to --out")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--state-path", default=None, help="epoch checkpoint path; default is next to --out")
    p.add_argument("--resume-state", action="store_true", help="resume model/optimizer from --state-path if it exists")
    p.add_argument(
        "--baseline-weights",
        nargs="*",
        default=None,
        help="AOWV MLP baselines to score on val sign accuracy; default uses v2_replays and v2_h64_preliminary when present",
    )
    p.add_argument("--log-interval", type=float, default=10.0, help="seconds between progress updates")
    p.add_argument(
        "--keep-last-batch",
        action="store_true",
        help="train on the final partial batch; default drops it to avoid small-batch MPS attention NaNs",
    )
    args = p.parse_args()

    dev = device()
    out_path = Path(args.out)
    metrics_path = Path(args.metrics_path) if args.metrics_path else out_path.with_suffix(".metrics.jsonl")
    dashboard_path = Path(args.dashboard_path) if args.dashboard_path else out_path.with_suffix(".dashboard.html")
    state_path = Path(args.state_path) if args.state_path else out_path.with_suffix(".state.pt")
    run_name = out_path.stem
    dashboard_proc = None
    log(f"{tag('device')} {dev}")
    if not args.no_dashboard:
        log(f"{tag('dash')} {dashboard_path}")
    xs, ms, ss, ys, finish_steps_all, metas = [], [], [], [], [], []
    early_sampled = False
    for path in args.data:
        d = np.load(path)
        if "tokens" not in d.files or "mask" not in d.files:
            log(f"{tag('data', C.YELLOW)} skip {path}: missing tokens/mask")
            continue
        n = d["tokens"].shape[0]
        bytes_est = d["tokens"].nbytes + d["mask"].nbytes + d["labels"].nbytes + d["meta"].nbytes
        log(f"{tag('data')} {Path(path).name}: {human_int(n)} samples, {human_bytes(bytes_est)} raw")
        idx = None
        if args.max_samples and len(args.data) == 1 and n > args.max_samples:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(n, args.max_samples, replace=False)
            early_sampled = True
        tokens_arr = d["tokens"][idx].astype(np.float32) if idx is not None else d["tokens"].astype(np.float32)
        xs.append(tokens_arr)
        ms.append(d["mask"][idx].astype(bool) if idx is not None else d["mask"].astype(bool))
        if "summary_v2" in d.files:
            summary = d["summary_v2"][idx].astype(np.float32) if idx is not None else d["summary_v2"].astype(np.float32)
            if summary.shape[1] < SUMMARY_DIM:
                log(f"{tag('summary', C.YELLOW)} {Path(path).name}: augmenting summary_v2 with route features")
                summary = augment_summary_routes(tokens_arr, summary)
            ss.append(summary)
        else:
            log(f"{tag('summary', C.YELLOW)} {Path(path).name}: deriving approximate summary_v2 from tokens")
            ss.append(derive_summary_from_tokens(tokens_arr))
        if args.target_mode != "stored" and "labels_raw" in d.files:
            ys.append(d["labels_raw"][idx].astype(np.float32) if idx is not None else d["labels_raw"].astype(np.float32))
        else:
            ys.append(d["labels"][idx].astype(np.float32) if idx is not None else d["labels"].astype(np.float32))
        if "finish_step" in d.files:
            finish_steps_all.append(d["finish_step"][idx].astype(np.int32) if idx is not None else d["finish_step"].astype(np.int32))
        else:
            finish_steps_all.append(None)
        metas.append(d["meta"][idx].astype(np.int64) if idx is not None else d["meta"].astype(np.int64))
    if not xs:
        raise SystemExit("no usable token datasets")
    X = np.concatenate(xs)
    M = np.concatenate(ms)
    S = np.concatenate(ss)
    y = np.concatenate(ys)
    meta = np.concatenate(metas)
    games = game_keys_from_meta(meta, args.data, xs)
    finish_steps = None
    if all(x is not None for x in finish_steps_all):
        finish_steps = np.concatenate(finish_steps_all)  # type: ignore[arg-type]
    if args.max_samples and not early_sampled and X.shape[0] > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X.shape[0], args.max_samples, replace=False)
        X, M, S, y, meta, games = X[idx], M[idx], S[idx], y[idx], meta[idx], games[idx]
        if finish_steps is not None:
            finish_steps = finish_steps[idx]
    log(
        f"{tag('loaded', C.BOLD)} samples={human_int(X.shape[0])} "
        f"tokens={X.shape[1]} token_dim={X.shape[2]} memory={human_bytes(X.nbytes + M.nbytes + S.nbytes + y.nbytes + meta.nbytes)}"
    )

    if args.target_mode != "stored":
        y = make_targets(y, meta, games, args.target_mode, args.time_coef, args.episode_steps, finish_steps)
    log(
        f"{tag('target')} mode={args.target_mode} time_coef={args.time_coef:g} "
        f"range=[{float(y.min()):+.3f},{float(y.max()):+.3f}] mean={float(y.mean()):+.3f}"
    )
    unique = np.unique(games)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_games = set(unique[:n_val].tolist())
    val_mask_np = np.array([g in val_games for g in games])
    train_samples = int((~val_mask_np).sum())
    val_samples = int(val_mask_np.sum())
    log(
        f"{tag('split')} games total={human_int(len(unique))} train={human_int(len(unique)-n_val)} "
        f"val={human_int(n_val)}"
    )
    log(f"{tag('split')} samples train={human_int(train_samples)} val={human_int(val_samples)}")
    log(
        f"{tag('model')} transformer d={args.d_model} layers={args.layers} "
        f"heads={args.heads} ff={args.ff_dim} summary_hidden={args.summary_hidden} batch={args.batch_size} epochs={args.epochs}"
    )

    model = EntityTransformer(args.d_model, args.layers, args.heads, args.ff_dim, args.summary_hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    train_idx_all = np.flatnonzero(~val_mask_np)
    val_idx_all = np.flatnonzero(val_mask_np)
    summary_mean = np.zeros((SUMMARY_DIM,), dtype=np.float32)
    summary_std = np.ones((SUMMARY_DIM,), dtype=np.float32)
    if args.summary_hidden > 0:
        summary_mean = S[train_idx_all].mean(axis=0).astype(np.float32)
        summary_std = S[train_idx_all].std(axis=0).astype(np.float32)
        summary_std = np.maximum(summary_std, 1e-3).astype(np.float32)
        log(
            f"{tag('summary')} standardized train summary "
            f"mean_abs={float(np.abs(summary_mean).mean()):.3g} std_min={float(summary_std.min()):.3g} std_max={float(summary_std.max()):.3g}"
        )
    val_labels = y[val_idx_all]
    val_steps = meta[val_idx_all, 1].astype(np.int32)
    val_ship_score = S[val_idx_all, 0] - S[val_idx_all, 10]
    val_ext_ship_score = S[val_idx_all, 20] - S[val_idx_all, 29]
    val_prod_score = S[val_idx_all, 5:8].sum(axis=1) - S[val_idx_all, 15:18].sum(axis=1)
    ship_baseline_sign = sign_accuracy_from_score(val_ship_score, val_labels)
    ext_ship_baseline_sign = sign_accuracy_from_score(val_ext_ship_score, val_labels)
    prod_baseline_sign = sign_accuracy_from_score(val_prod_score, val_labels)
    baseline_bucket_signs = {
        **bucket_sign_metrics("base_ship", val_ship_score, val_labels, val_steps, args.episode_steps),
        **bucket_sign_metrics("base_ext", val_ext_ship_score, val_labels, val_steps, args.episode_steps),
        **bucket_sign_metrics("base_prod", val_prod_score, val_labels, val_steps, args.episode_steps),
    }
    if args.baseline_weights is None:
        baseline_paths = DEFAULT_BASELINE_WEIGHTS
    else:
        baseline_paths = [Path(x).expanduser() for x in args.baseline_weights]
    value_baseline_signs = score_value_baselines(
        baseline_paths,
        S[val_idx_all],
        val_labels,
        val_steps,
        args.episode_steps,
    )
    log(
        f"{tag('base')} val sign ship_diff={ship_baseline_sign:.3f} "
        f"ext_ship_diff={ext_ship_baseline_sign:.3f} prod_diff={prod_baseline_sign:.3f} "
        + " ".join(f"{k}={v:.3f}" for k, v in value_baseline_signs.items())
    )
    if not args.keep_last_batch:
        train_full = (train_idx_all.shape[0] // args.batch_size) * args.batch_size
        dropped = train_idx_all.shape[0] - train_full
        if dropped:
            log(f"{tag('batch', C.YELLOW)} dropping final partial train batch: {dropped} sample(s)")
        train_idx_all = train_idx_all[:train_full]
    best_val = float("inf")
    best_state = None
    best_sign = 0.0
    start_epoch = 0
    if args.resume_state and state_path.exists():
        ckpt = torch.load(state_path, map_location=dev, weights_only=False)
        cfg = ckpt.get("config", {})
        expected = {
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "ff_dim": args.ff_dim,
            "summary_hidden": args.summary_hidden,
            "summary_dim": SUMMARY_DIM,
        }
        mismatches = [k for k, v in expected.items() if cfg.get(k) != v]
        if mismatches:
            raise SystemExit(f"resume checkpoint config mismatch for {mismatches}; refusing to load {state_path}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        best_state = ckpt.get("best_state")
        best_val = float(ckpt.get("best_val", best_val))
        best_sign = float(ckpt.get("best_sign", best_sign))
        start_epoch = int(ckpt.get("epoch_completed", 0))
        if "summary_mean" in ckpt and "summary_std" in ckpt:
            summary_mean = np.asarray(ckpt["summary_mean"], dtype=np.float32)
            summary_std = np.asarray(ckpt["summary_std"], dtype=np.float32)
        log(f"{tag('resume', C.GREEN)} {state_path} epoch={start_epoch} best_val={best_val:.4f} best_sign={best_sign:.3f}")
        if start_epoch >= args.epochs:
            log(f"{tag('resume', C.YELLOW)} checkpoint already reached epochs={args.epochs}")
    t0 = time.time()
    for ep in range(start_epoch, args.epochs):
        ep_t0 = time.time()
        model.train()
        perm = rng.permutation(train_idx_all)
        total = 0.0
        seen = 0
        last_log = time.time()
        n_batches = (perm.shape[0] + args.batch_size - 1) // args.batch_size
        skipped_batches = 0
        consecutive_skips = 0
        for j in range(0, perm.shape[0], args.batch_size):
            batch_i = j // args.batch_size + 1
            sel = perm[j : j + args.batch_size]
            xb = torch.from_numpy(X[sel]).to(dev, non_blocking=True)
            mb = torch.from_numpy(M[sel]).to(dev, non_blocking=True)
            sb_np = (S[sel] - summary_mean) / summary_std if args.summary_hidden > 0 else S[sel]
            sb = torch.from_numpy(sb_np.astype(np.float32, copy=False)).to(dev, non_blocking=True)
            yb = torch.from_numpy(y[sel]).to(dev, non_blocking=True)
            raw_pred = model(xb, mb, sb)
            pred = sanitize_value(raw_pred)
            loss = value_loss(pred, yb)
            loss_finite = bool(torch.isfinite(loss).item())
            pred_finite = bool(torch.isfinite(raw_pred).all().item())
            loss_absurd = loss_finite and float(loss.detach().cpu()) > 10.0
            if not loss_finite or not pred_finite or loss_absurd:
                skipped_batches += 1
                log(
                    f"{tag('nan', C.YELLOW)} ep {ep+1:02d}/{args.epochs} "
                    f"batch {batch_i:,}/{n_batches:,} skipped "
                    f"pred_finite={pred_finite} "
                    f"loss={float(loss.detach().cpu()) if loss_finite else 'nan'} "
                    f"pred=[{float(raw_pred.min().detach().cpu()):+.3g},{float(raw_pred.max().detach().cpu()):+.3g}] "
                    f"labels=[{float(yb.min().detach().cpu()):+.1f},{float(yb.max().detach().cpu()):+.1f}] "
                    f"valid_tokens=[{int(mb.sum(dim=1).min().detach().cpu())},{int(mb.sum(dim=1).max().detach().cpu())}]"
                )
                opt.zero_grad(set_to_none=True)
                consecutive_skips += 1
                if consecutive_skips >= 20:
                    raise SystemExit(
                        "20 consecutive non-finite train batches. Stop this run and try "
                        "`--summary-hidden 0`, lower `--lr`, or CPU/CUDA instead of MPS."
                    )
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            bad_param = False
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    bad_param = True
                    break
            if bad_param:
                skipped_batches += 1
                log(f"{tag('nan', C.YELLOW)} ep {ep+1:02d}/{args.epochs} batch {batch_i:,}/{n_batches:,} skipped non-finite gradient")
                opt.zero_grad(set_to_none=True)
                consecutive_skips += 1
                if consecutive_skips >= 20:
                    raise SystemExit(
                        "20 consecutive non-finite gradient batches. Stop this run and try "
                        "`--summary-hidden 0`, lower `--lr`, or CPU/CUDA instead of MPS."
                    )
                continue
            opt.step()
            consecutive_skips = 0
            total += loss.item() * len(sel)
            seen += len(sel)
            now = time.time()
            if args.log_interval > 0 and (now - last_log >= args.log_interval or batch_i == n_batches):
                rate = seen / max(now - ep_t0, 1e-6)
                pct = 100.0 * (j + len(sel)) / max(perm.shape[0], 1)
                eta = (perm.shape[0] - seen) / max(rate, 1e-6)
                skip_s = f" skipped={skipped_batches}" if skipped_batches else ""
                log(
                    f"{tag('train', C.MAGENTA)} ep {ep+1:02d}/{args.epochs} "
                    f"{pct:5.1f}% batch {batch_i:,}/{n_batches:,} "
                    f"loss={total/max(seen,1):.4f} {rate:,.0f} samp/s eta={fmt_time(eta)}{skip_s}"
                )
                last_log = now
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_seen = 0
            sign_hits = 0
            val_pred_chunks = []
            val_target_chunks = []
            val_step_chunks = []
            val_t0 = time.time()
            last_val_log = val_t0
            n_val_batches = (val_idx_all.shape[0] + args.batch_size - 1) // args.batch_size
            for j in range(0, val_idx_all.shape[0], args.batch_size):
                batch_i = j // args.batch_size + 1
                sel = val_idx_all[j : j + args.batch_size]
                xb = torch.from_numpy(X[sel]).to(dev, non_blocking=True)
                mb = torch.from_numpy(M[sel]).to(dev, non_blocking=True)
                sb_np = (S[sel] - summary_mean) / summary_std if args.summary_hidden > 0 else S[sel]
                sb = torch.from_numpy(sb_np.astype(np.float32, copy=False)).to(dev, non_blocking=True)
                yb = torch.from_numpy(y[sel]).to(dev, non_blocking=True)
                pv = sanitize_value(model(xb, mb, sb))
                val_loss_sum += value_loss(pv, yb).item() * len(sel)
                sign_hits += int(((pv > 0) == (yb > 0)).sum().item())
                val_seen += len(sel)
                val_pred_chunks.append(pv.detach().cpu().numpy().astype(np.float32))
                val_target_chunks.append(yb.detach().cpu().numpy().astype(np.float32))
                val_step_chunks.append(meta[sel, 1].astype(np.int32))
                now = time.time()
                if args.log_interval > 0 and (now - last_val_log >= args.log_interval or batch_i == n_val_batches):
                    pct = 100.0 * val_seen / max(val_idx_all.shape[0], 1)
                    log(
                        f"{tag('val', C.YELLOW)} ep {ep+1:02d}/{args.epochs} "
                        f"{pct:5.1f}% batch {batch_i:,}/{n_val_batches:,}"
                    )
                    last_val_log = now
            val_loss = val_loss_sum / max(val_seen, 1)
            sign = sign_hits / max(val_seen, 1)
            val_bucket, val_bucket_metrics = bucket_metrics(
                np.concatenate(val_pred_chunks),
                np.concatenate(val_target_chunks),
                np.concatenate(val_step_chunks),
                args.episode_steps,
            )
        improved = val_loss < best_val
        if val_loss < best_val:
            best_val = val_loss
            best_sign = sign
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        marker = f"{C.GREEN}best{C.RESET}" if improved else f"{C.DIM}    {C.RESET}"
        log(
            f"{tag('epoch', C.BOLD)} {ep+1:02d}/{args.epochs} "
            f"train={total/max(seen,1):.4f} val={val_loss:.4f} sign={sign:.3f} "
            f"base_ship={ship_baseline_sign:.3f} base_ext={ext_ship_baseline_sign:.3f} base_prod={prod_baseline_sign:.3f} "
            f"buckets[{val_bucket}] "
            f"{marker} elapsed={fmt_time(time.time()-ep_t0)} total={fmt_time(time.time()-t0)}"
        )
        row = {
            "run_name": run_name,
            "epoch": ep + 1,
            "epochs": args.epochs,
            "train_loss": total / max(seen, 1),
            "val_loss": val_loss,
            "sign": sign,
            "best_val": best_val,
            "best_sign": best_sign,
            "base_ship": ship_baseline_sign,
            "base_ext": ext_ship_baseline_sign,
            "base_prod": prod_baseline_sign,
            **baseline_bucket_signs,
            **value_baseline_signs,
            "samples_per_sec": seen / max(time.time() - ep_t0, 1e-6),
            "epoch_seconds": time.time() - ep_t0,
            "total_seconds": time.time() - t0,
            "target_mode": args.target_mode,
            "time_coef": args.time_coef,
            "summary_dim": SUMMARY_DIM,
            "train_samples": train_samples,
            "val_samples": val_samples,
            **val_bucket_metrics,
        }
        append_jsonl(metrics_path, row)
        save_training_state(
            state_path,
            epoch_completed=ep + 1,
            model=model,
            opt=opt,
            best_state=best_state,
            best_val=best_val,
            best_sign=best_sign,
            summary_mean=summary_mean,
            summary_std=summary_std,
            args=args,
        )
        if not args.no_dashboard:
            dashboard_proc = launch_dashboard(metrics_path, dashboard_path, dashboard_proc)

    if best_state is None:
        if args.resume_state and state_path.exists():
            ckpt = torch.load(state_path, map_location=dev, weights_only=False)
            best_state = ckpt.get("best_state") or ckpt.get("model")
            best_val = float(ckpt.get("best_val", best_val))
            best_sign = float(ckpt.get("best_sign", best_sign))
        if best_state is None:
            raise SystemExit("no trained state available to export")
    model.load_state_dict(best_state)
    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    write_aowv_transformer(out, model, summary_mean=summary_mean, summary_std=summary_std)
    log(
        f"{tag('saved', C.GREEN)} best_val={best_val:.4f} sign={best_sign:.3f} "
        f"elapsed={fmt_time(time.time()-t0)}"
    )
    log(
        f"{tag('weights', C.GREEN)} {out} ({human_bytes(out.stat().st_size)}, "
        f"arch=tokens{MAX_TOKENS}x{TOKEN_DIM}+summary{SUMMARY_DIM}->{args.layers}x d={args.d_model} h={args.heads} ff={args.ff_dim})"
    )
    if not args.no_dashboard:
        dashboard_proc = launch_dashboard(metrics_path, dashboard_path, dashboard_proc)
        if dashboard_proc is not None:
            try:
                dashboard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


if __name__ == "__main__":
    main()
