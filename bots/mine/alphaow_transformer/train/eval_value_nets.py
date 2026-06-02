"""Offline evaluator for alphaow AOWV value nets.

No games are simulated. This scores exported value-net files on held-out replay
frames from token NPZs, comparing predicted final outcome against labels.
Supported exactly from token NPZs:
  - AOWV v3/v4 entity transformers: uses tokens/mask/(summary_v2)
  - AOWV v1/v2 MLPs with input_dim=46: uses summary_v2

Older 23-d summary MLPs need the original 23 pressure features and are skipped
for token-only replay datasets.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE / "weights"
DEFAULT_DATA = HERE / "data" / "tokens_by_day" / "2026-05-28_orbit-wars-episodes-2026-05-28_tokens.npz"
MAGIC = 0x564F4157
SUMMARY_V2_DIM = 46
ROUTE_DIM = 20
SUMMARY_DIM = SUMMARY_V2_DIM + ROUTE_DIM
COMPARABLE_OLD = ["v2_replays", "v2_h64_preliminary"]
EPISODE_STEPS = 500
SUMMARY_NAMES = [
    "me_ships",
    "me_flying",
    "me_static",
    "me_orbit",
    "me_comet",
    "me_prod_static",
    "me_prod_orbit",
    "me_prod_comet",
    "me_neutrals_closer",
    "me_enemies_closer",
    "enemy_ships",
    "enemy_flying",
    "enemy_static",
    "enemy_orbit",
    "enemy_comet",
    "enemy_prod_static",
    "enemy_prod_orbit",
    "enemy_prod_comet",
    "enemy_neutrals_closer",
    "enemy_enemies_closer",
    "me_ext_ships",
    "me_ext_static",
    "me_ext_orbit",
    "me_ext_comet",
    "me_ext_prod_static",
    "me_ext_prod_orbit",
    "me_ext_prod_comet",
    "me_ext_neutrals_closer",
    "me_ext_enemies_closer",
    "enemy_ext_ships",
    "enemy_ext_static",
    "enemy_ext_orbit",
    "enemy_ext_comet",
    "enemy_ext_prod_static",
    "enemy_ext_prod_orbit",
    "enemy_ext_prod_comet",
    "enemy_ext_neutrals_closer",
    "enemy_ext_enemies_closer",
    "neutral_ships",
    "neutral_static",
    "neutral_orbit",
    "neutral_comet",
    "neutral_prod_static",
    "neutral_prod_orbit",
    "neutral_prod_comet",
    "comet_time_left",
    "my_to_neutral_count",
    "my_to_neutral_min_time",
    "my_to_neutral_mean_time",
    "my_to_neutral_clear_frac",
    "my_to_neutral_feasible_frac",
    "my_to_enemy_count",
    "my_to_enemy_min_time",
    "my_to_enemy_mean_time",
    "my_to_enemy_clear_frac",
    "my_to_enemy_feasible_frac",
    "enemy_to_my_count",
    "enemy_to_my_min_time",
    "enemy_to_my_mean_time",
    "enemy_to_my_clear_frac",
    "enemy_to_my_feasible_frac",
    "enemy_to_neutral_count",
    "enemy_to_neutral_min_time",
    "enemy_to_neutral_mean_time",
    "enemy_to_neutral_clear_frac",
    "enemy_to_neutral_feasible_frac",
]

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_transformer import EntityTransformer  # type: ignore
from train_transformer import augment_summary_routes  # type: ignore


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


def paint(text: str, color: str, enabled: bool = True) -> str:
    return f"{color}{text}{C.RESET}" if enabled else text


def human_int(n: int) -> str:
    return f"{n:,}"


def reward_sign(labels: np.ndarray) -> np.ndarray:
    return np.where(labels > 0.0, 1.0, np.where(labels < 0.0, -1.0, 0.0)).astype(np.float32)


def infer_finish_steps(meta: np.ndarray) -> np.ndarray:
    games = meta[:, 0].astype(np.int64)
    steps = meta[:, 1].astype(np.int32)
    finish_by_game: dict[int, int] = {}
    for game, step in zip(games, steps):
        g = int(game)
        finish_by_game[g] = max(finish_by_game.get(g, 0), int(step))
    return np.array([finish_by_game[int(g)] for g in games], dtype=np.int32)


def make_targets(labels: np.ndarray, meta: np.ndarray, finish_steps: np.ndarray | None, mode: str, time_coef: float, episode_steps: int):
    if mode == "stored":
        return labels.astype(np.float32)
    raw = reward_sign(labels)
    if mode == "outcome":
        return raw
    if finish_steps is None:
        finish_steps = infer_finish_steps(meta)
    finish_frac = np.clip(finish_steps.astype(np.float32) / max(float(episode_steps), 1.0), 0.0, 1.0)
    magnitude = 1.0 - np.clip(float(time_coef), 0.0, 0.95) * finish_frac
    return (raw * magnitude).astype(np.float32)


def bucket_masks(steps: np.ndarray, episode_steps: int):
    s = steps.astype(np.float32)
    return [
        ("opener", s < episode_steps * 0.2),
        ("midgame", (s >= episode_steps * 0.2) & (s < episode_steps * 0.6)),
        ("endgame", s >= episode_steps * 0.6),
    ]


def resolve_weight(name: str) -> Path:
    if name == "latest":
        bins = sorted(WEIGHTS_DIR.glob("*.bin"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not bins:
            raise SystemExit(f"no .bin weights found in {WEIGHTS_DIR}")
        return bins[0].resolve()
    p = Path(name).expanduser()
    candidates = []
    if p.is_absolute() or p.parent != Path("."):
        candidates.append(p)
    candidates.append(WEIGHTS_DIR / name)
    if p.suffix != ".bin":
        candidates.append(WEIGHTS_DIR / f"{name}.bin")
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise SystemExit(f"unknown weights file: {name}")


def expand_weight_args(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name == "comparable-old":
            out.extend(COMPARABLE_OLD)
        elif name == "all-old":
            out.extend(p.stem for p in sorted(WEIGHTS_DIR.glob("*.bin"), key=lambda x: x.stat().st_mtime))
        else:
            out.append(name)
    seen: set[str] = set()
    unique: list[str] = []
    for name in out:
        try:
            key = str(resolve_weight(name))
        except SystemExit:
            key = name
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


def read_u32(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def take_f32(buf: bytes, cursor: int, n: int) -> tuple[np.ndarray, int]:
    size = 4 * n
    if cursor + size > len(buf):
        raise ValueError("weights file ended early")
    arr = np.frombuffer(buf, dtype="<f4", count=n, offset=cursor).astype(np.float32).copy()
    return arr, cursor + size


@dataclass
class LoadedNet:
    path: Path
    kind: str
    input_dim: int
    model: object
    summary_dim: int = 0


class MlpNet:
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

    def to(self, device: torch.device):
        for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, val.to(device))
        return self

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(F.linear(x, self.w1, self.b1))
        if self.w3 is None:
            y = F.linear(h, self.w2, self.b2)
        else:
            h = torch.relu(F.linear(h, self.w2, self.b2))
            y = F.linear(h, self.w3, self.b3)
        return torch.tanh(y.squeeze(-1))


def load_mlp(path: Path, buf: bytes, version: int) -> LoadedNet:
    input_dim = read_u32(buf, 8)
    hidden = read_u32(buf, 12)
    cursor = 16
    hidden2 = 0
    if version == 2:
        hidden2 = read_u32(buf, 16)
        cursor = 20
    w1, cursor = take_f32(buf, cursor, hidden * input_dim)
    b1, cursor = take_f32(buf, cursor, hidden)
    if version == 2:
        w2, cursor = take_f32(buf, cursor, hidden2 * hidden)
        b2, cursor = take_f32(buf, cursor, hidden2)
        w3, cursor = take_f32(buf, cursor, hidden2)
        b3, cursor = take_f32(buf, cursor, 1)
        model = MlpNet(input_dim, hidden, w1, b1, w2, b2, w3, float(b3[0]))
        kind = f"mlp-v2 {input_dim}->{hidden}->{hidden2}->1"
    else:
        w2, cursor = take_f32(buf, cursor, hidden)
        b2, cursor = take_f32(buf, cursor, 1)
        model = MlpNet(input_dim, hidden, w1, b1, w2, np.zeros(0, dtype=np.float32), None, float(b2[0]))
        kind = f"mlp-v1 {input_dim}->{hidden}->1"
    return LoadedNet(path=path, kind=kind, input_dim=input_dim, model=model)


def load_transformer(path: Path, buf: bytes, version: int) -> LoadedNet:
    token_dim = read_u32(buf, 8)
    d_model = read_u32(buf, 12)
    layers = read_u32(buf, 16)
    heads = read_u32(buf, 20)
    ff_dim = read_u32(buf, 24)
    max_tokens = read_u32(buf, 28)
    cursor = 32
    summary_hidden = 0
    summary_dim = 0
    if version == 4:
        summary_dim = read_u32(buf, 32)
        summary_hidden = read_u32(buf, 36)
        cursor = 40
        if summary_dim not in (SUMMARY_V2_DIM, SUMMARY_DIM):
            raise ValueError(f"unsupported summary_dim={summary_dim}")
    model = EntityTransformer(
        d_model,
        layers,
        heads,
        ff_dim,
        summary_hidden,
        summary_dim=summary_dim if summary_dim > 0 else SUMMARY_DIM,
    )
    state = model.state_dict()

    def assign(name: str, shape: tuple[int, ...]):
        nonlocal cursor
        n = math.prod(shape)
        arr, cursor = take_f32(buf, cursor, n)
        state[name] = torch.from_numpy(arr.reshape(shape))

    assign("cls", (d_model,))
    assign("embed.weight", (d_model, token_dim))
    assign("embed.bias", (d_model,))
    for i in range(layers):
        assign(f"blocks.{i}.ln1.weight", (d_model,))
        assign(f"blocks.{i}.ln1.bias", (d_model,))
        assign(f"blocks.{i}.attn.in_proj_weight", (3 * d_model, d_model))
        assign(f"blocks.{i}.attn.in_proj_bias", (3 * d_model,))
        assign(f"blocks.{i}.attn.out_proj.weight", (d_model, d_model))
        assign(f"blocks.{i}.attn.out_proj.bias", (d_model,))
        assign(f"blocks.{i}.ln2.weight", (d_model,))
        assign(f"blocks.{i}.ln2.bias", (d_model,))
        assign(f"blocks.{i}.ff.0.weight", (ff_dim, d_model))
        assign(f"blocks.{i}.ff.0.bias", (ff_dim,))
        assign(f"blocks.{i}.ff.2.weight", (d_model, ff_dim))
        assign(f"blocks.{i}.ff.2.bias", (d_model,))
    assign("ln_f.weight", (d_model,))
    assign("ln_f.bias", (d_model,))
    if version == 4:
        assign("summary_fc.weight", (summary_hidden, summary_dim))
        assign("summary_fc.bias", (summary_hidden,))
        assign("head.weight", (1, d_model + summary_hidden))
    else:
        assign("head.weight", (1, d_model))
    assign("head.bias", (1,))
    model.load_state_dict(state)
    return LoadedNet(
        path=path,
        kind=f"transformer-v{version} tokens={max_tokens}x{token_dim} d={d_model} l={layers} h={heads} summary={summary_hidden}x{summary_dim}",
        input_dim=token_dim,
        model=model,
        summary_dim=summary_dim,
    )


def load_net(path: Path) -> LoadedNet:
    buf = path.read_bytes()
    if len(buf) < 16 or read_u32(buf, 0) != MAGIC:
        raise ValueError(f"{path} is not an AOWV file")
    version = read_u32(buf, 4)
    if version in (1, 2):
        return load_mlp(path, buf, version)
    if version in (3, 4):
        return load_transformer(path, buf, version)
    raise ValueError(f"unsupported AOWV version {version}")


def load_data(paths: list[Path], max_samples: int, seed: int, target_mode: str, time_coef: float, episode_steps: int):
    tokens, masks, summaries, labels, finish_steps_all, metas = [], [], [], [], [], []
    early_sampled = False
    per_file_cap = int(math.ceil(max_samples / max(len(paths), 1))) if max_samples else 0
    for path in paths:
        d = np.load(path, mmap_mode="r")
        for key in ("tokens", "mask", "summary_v2", "labels", "meta"):
            if key not in d.files:
                raise SystemExit(f"{path} missing {key}; regenerate tokens with current extractor")
        n = d["tokens"].shape[0]
        idx = None
        if per_file_cap and n > per_file_cap:
            rng = np.random.default_rng(seed + len(tokens) * 9973)
            idx = rng.choice(n, size=per_file_cap, replace=False)
            early_sampled = True
        tok = np.asarray(d["tokens"][idx], dtype=np.float32) if idx is not None else np.asarray(d["tokens"], dtype=np.float32)
        tokens.append(tok)
        masks.append(np.asarray(d["mask"][idx], dtype=np.bool_) if idx is not None else np.asarray(d["mask"], dtype=np.bool_))
        summary = np.asarray(d["summary_v2"][idx], dtype=np.float32) if idx is not None else np.asarray(d["summary_v2"], dtype=np.float32)
        if summary.shape[1] < SUMMARY_DIM:
            summary = augment_summary_routes(tok, summary)
        summaries.append(summary)
        if target_mode != "stored" and "labels_raw" in d.files:
            labels.append(np.asarray(d["labels_raw"][idx], dtype=np.float32) if idx is not None else np.asarray(d["labels_raw"], dtype=np.float32))
        else:
            labels.append(np.asarray(d["labels"][idx], dtype=np.float32) if idx is not None else np.asarray(d["labels"], dtype=np.float32))
        if "finish_step" in d.files:
            finish_steps_all.append(np.asarray(d["finish_step"][idx], dtype=np.int32) if idx is not None else np.asarray(d["finish_step"], dtype=np.int32))
        else:
            finish_steps_all.append(None)
        metas.append(np.asarray(d["meta"][idx], dtype=np.int64) if idx is not None else np.asarray(d["meta"], dtype=np.int64))
    X = np.concatenate(tokens, axis=0)
    M = np.concatenate(masks, axis=0)
    S = np.concatenate(summaries, axis=0)
    y = np.concatenate(labels, axis=0)
    meta = np.concatenate(metas, axis=0)
    finish_steps = None
    if all(x is not None for x in finish_steps_all):
        finish_steps = np.concatenate(finish_steps_all)  # type: ignore[arg-type]
    y = make_targets(y, meta, finish_steps, target_mode, time_coef, episode_steps)
    if max_samples and not early_sampled and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, M, S, y, meta = X[idx], M[idx], S[idx], y[idx], meta[idx]
    return X, M, S, y, meta


def val_mask_from_meta(meta: np.ndarray, seed: int, frac: float):
    # Match train_transformer.py's game-level split closely.
    game_ids = meta[:, 0].astype(np.int64)
    game_key = game_ids
    games = np.unique(game_key)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(frac * len(games)))
    val_games = set(games[:n_val].tolist())
    return np.array([g in val_games for g in game_key])


def predict(net: LoadedNet, X, M, S, batch_size: int, device: torch.device):
    model = net.model
    model.to(device)
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            if net.kind.startswith("transformer"):
                xb = torch.from_numpy(X[start:end]).to(device)
                mb = torch.from_numpy(M[start:end]).to(device)
                sdim = net.summary_dim or S.shape[1]
                sb = torch.from_numpy(S[start:end, :sdim]).to(device)
                out = model(xb, mb, sb if "summary=0" not in net.kind else None)
            elif net.input_dim in (SUMMARY_V2_DIM, SUMMARY_DIM):
                sb = torch.from_numpy(S[start:end, : net.input_dim]).to(device)
                out = model(sb)
            else:
                raise ValueError(f"cannot evaluate {net.path.name}: input_dim={net.input_dim} is not available in token NPZ")
            outs.append(out.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sign = float(np.mean((pred > 0) == (y > 0)))
    mse = float(np.mean((pred - y) ** 2))
    abs_err = float(np.mean(np.abs(pred - y)))
    smooth_l1 = float(np.mean(np.where(np.abs(pred - y) < 1.0, 0.5 * (pred - y) ** 2, np.abs(pred - y) - 0.5)))
    corr = float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 1e-6 and y.std() > 1e-6 else 0.0
    margin = float(np.mean(pred[y > 0]) - np.mean(pred[y < 0])) if np.any(y > 0) and np.any(y < 0) else 0.0
    return {
        "sign": sign,
        "smooth_l1": smooth_l1,
        "mse": mse,
        "mae": abs_err,
        "corr": corr,
        "margin": margin,
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
    }


def bucket_metrics(pred: np.ndarray, y: np.ndarray, steps: np.ndarray, episode_steps: int) -> str:
    parts = []
    for name, mask in bucket_masks(steps, episode_steps):
        if not np.any(mask):
            parts.append(f"{name}=n/a")
            continue
        m = metrics(pred[mask], y[mask])
        parts.append(f"{name}={m['sign']:.3f}/{m['smooth_l1']:.3f}/n{int(mask.sum())}")
    return "  ".join(parts)


def feature_importance(
    net: LoadedNet,
    X: np.ndarray,
    M: np.ndarray,
    S: np.ndarray,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    batch_size: int,
    device: torch.device,
    samples: int,
    top_k: int,
    seed: int,
    use_color: bool,
):
    if samples and X.shape[0] > samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=samples, replace=False)
        X, M, S, y = X[idx], M[idx], S[idx], y[idx]
        baseline_pred = baseline_pred[idx]
    base = metrics(baseline_pred, y)
    rng = np.random.default_rng(seed + 17)
    rows = []
    for col, name in enumerate(SUMMARY_NAMES):
        Sp = S.copy()
        Sp[:, col] = Sp[rng.permutation(Sp.shape[0]), col]
        pred = predict(net, X, M, Sp, batch_size, device)
        m = metrics(pred, y)
        rows.append((name, base["sign"] - m["sign"], m["smooth_l1"] - base["smooth_l1"], m["sign"], m["smooth_l1"]))
    rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
    max_drop = max([r[1] for r in rows[:top_k]] + [1e-6])
    print(paint("\nsummary feature importance", C.BOLD, use_color))
    print("feature                         sign_drop  sL1_delta  perm_sign  graph")
    for name, sign_drop, loss_delta, perm_sign, _perm_loss in rows[:top_k]:
        bar_n = int(round(28 * max(sign_drop, 0.0) / max_drop))
        bar = "#" * bar_n
        print(f"{name:<31} {sign_drop:+.4f}    {loss_delta:+.4f}    {perm_sign:.3f}   {bar}")


def main() -> int:
    p = argparse.ArgumentParser(description="Offline compare alphaow value nets on held-out replay frames.")
    p.add_argument("--data", nargs="+", default=[str(DEFAULT_DATA)])
    p.add_argument(
        "--weights",
        nargs="+",
        default=["latest", "comparable-old"],
        help="Weight names/paths. Special values: latest, comparable-old, all-old.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--max-samples", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--target-mode", choices=["time", "outcome", "stored"], default="time")
    p.add_argument("--time-coef", type=float, default=0.10)
    p.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    p.add_argument("--importance", action="store_true", help="run summary_v2 permutation importance for the best scored net")
    p.add_argument("--importance-samples", type=int, default=30000)
    p.add_argument("--importance-top-k", type=int, default=12)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()
    use_color = not args.no_color and sys.stdout.isatty()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t0 = time.time()
    data_paths = [Path(x).expanduser().resolve() for x in args.data]
    X, M, S, y, meta = load_data(data_paths, args.max_samples, args.seed, args.target_mode, args.time_coef, args.episode_steps)
    mask = val_mask_from_meta(meta, args.seed, args.val_frac)
    if mask.sum() == 0:
        raise SystemExit("empty validation split")
    Xv, Mv, Sv, yv = X[mask], M[mask], S[mask], y[mask]
    steps_v = meta[mask, 1].astype(np.int32)
    print(
        paint("data ", C.BOLD, use_color)
        + f"samples={human_int(X.shape[0])} val={human_int(Xv.shape[0])} "
        + f"target={args.target_mode} range=[{float(yv.min()):+.3f},{float(yv.max()):+.3f}] "
        + f"pos={np.mean(yv > 0):.1%} device={device}"
    )
    for label, score in [
        ("ship_diff", Sv[:, 0] - Sv[:, 10]),
        ("ext_ship_diff", Sv[:, 20] - Sv[:, 29]),
        ("prod_diff", Sv[:, 5:8].sum(axis=1) - Sv[:, 15:18].sum(axis=1)),
    ]:
        m = metrics(score, yv)
        print(f"baseline {label:<14} sign={m['sign']:.3f} corr={m['corr']:.3f}")

    rows = []
    for name in expand_weight_args(args.weights):
        path = resolve_weight(name)
        try:
            net = load_net(path)
            pred = predict(net, Xv, Mv, Sv, args.batch_size, device)
            m = metrics(pred, yv)
            b = bucket_metrics(pred, yv, steps_v, args.episode_steps)
            rows.append((path.name, net.kind, m, b, net, pred, None))
        except Exception as exc:
            rows.append((path.name, "", {}, "", None, None, str(exc)))

    print(paint("\nvalue net results", C.BOLD, use_color))
    print("weight                                      sign   sL1    corr  margin  pred_std  kind")
    sorted_rows = sorted(rows, key=lambda r: r[2].get("sign", -1.0), reverse=True)
    for name, kind, m, bucket, _net, _pred, err in sorted_rows:
        if err:
            print(f"{name:<43} {paint('skip', C.YELLOW, use_color):>6}  {err}")
            continue
        sign = m["sign"]
        sign_s = f"{sign:5.3f}"
        sign_s = paint(sign_s, C.GREEN if sign >= 0.80 else C.YELLOW if sign >= 0.70 else C.RED, use_color)
        print(
            f"{name:<43} {sign_s}  {m['smooth_l1']:.4f}  {m['corr']:.3f} "
            f"{m['margin']:+.3f}   {m['pred_std']:.3f}    {kind}"
        )
        print(f"{'':<43} buckets  {bucket}")
    if args.importance:
        for _name, _kind, _m, _bucket, net, pred, err in sorted_rows:
            if err is None and net is not None and pred is not None:
                feature_importance(
                    net,
                    Xv,
                    Mv,
                    Sv,
                    yv,
                    pred,
                    args.batch_size,
                    device,
                    args.importance_samples,
                    args.importance_top_k,
                    args.seed,
                    use_color,
                )
                break
    print(paint("done ", C.GREEN, use_color) + f"{time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
