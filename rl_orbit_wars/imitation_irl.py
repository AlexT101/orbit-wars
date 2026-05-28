from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_orbit_wars.orbit_wars_rl.features import (
    ACTION_DIM,
    GLOBAL_FEATURES,
    PLANET_FEATURES,
    SEND_FRACTIONS,
    decode_action_index,
    encode_move_as_action_index,
    encode_obs,
    encode_teacher_moves_as_action_index,
)
from rl_orbit_wars.orbit_wars_rl.model import build_policy
from rl_orbit_wars.orbit_wars_rl.ppo import _stack_encoded
from rl_orbit_wars.orbit_wars_rl.visualization import append_jsonl, write_training_report

ALPHAOW_MANIFEST_DIR = ROOT / "bots" / "mine" / "alphaow_transformer" / "train" / "manifests"
ALPHAOW_DATASET_DIR = ROOT / "bots" / "mine" / "alphaow_transformer" / "train" / "datasets"
ALPHAOW_CSV = ROOT / "bots" / "alphaow_experimental" / "prometheus" / "train" / "manifest.csv"

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


@dataclass(frozen=True)
class AgentStats:
    games: int
    wins: int

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games)


@dataclass(frozen=True)
class ReplaySample:
    encoded: object
    anchor_label: int
    labels: tuple[int, ...]
    agent_name: str
    episode_id: int
    step: int
    reward: float


class ActionRewardModel(nn.Module):
    """Scores state-action pairs for contrastive inverse RL."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        input_dim = GLOBAL_FEATURES + PLANET_FEATURES * 4 + 4 + len(SEND_FRACTIONS) + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("rb") as f:
            return json.load(f)
    except Exception:
        return None


def _as_path_list(values: Iterable[str]) -> list[Path]:
    return [Path(v).expanduser() for v in values if str(v).strip()]


def _manifest_json_files(manifest_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not manifest_dir.exists():
        return files
    for manifest in sorted(manifest_dir.glob("*_files.json")):
        data = _read_json(manifest) or {}
        files.extend(_as_path_list(data.get("files", []) or []))
    return files


def _marker_dataset_files(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not dataset_dir.exists():
        return files
    for marker in sorted(dataset_dir.glob("*/_kagglehub_path.txt")):
        cached = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
        if cached.exists():
            files.extend(sorted(cached.rglob("*.json")))
    return files


def _download_alphaow_manifest(
    csv_path: Path,
    download_root: Path,
    start_date: str | None,
    end_date: str | None,
    limit_days: int | None,
) -> list[Path]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date = row.get("date", "")
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            slug = (row.get("daily_dataset_slug") or "").strip()
            if not slug:
                continue
            rows.append((date, slug))
    rows.sort(key=lambda x: x[0])
    if limit_days:
        rows = rows[:limit_days]

    try:
        try:
            import kagglesdk.kaggle_env as kaggle_env

            if not hasattr(kaggle_env, "get_web_endpoint") and hasattr(kaggle_env, "get_endpoint"):
                kaggle_env.get_web_endpoint = kaggle_env.get_endpoint
        except ImportError:
            pass
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not available in this Python environment.\n"
            f"Python: {sys.executable}\n"
            f"Import error: {exc}\n"
            "Try: python3 -m pip install --upgrade kagglehub kagglesdk"
        ) from exc

    files: list[Path] = []
    for date, slug in rows:
        marker = download_root / slug / ".kagglehub_path"
        if marker.exists():
            path = Path(marker.read_text(encoding="utf-8").strip())
        else:
            ref = slug if "/" in slug else f"kaggle/{slug}"
            print(f"{_c('download', BOLD + CYAN)} {date} {ref}", flush=True)
            path = Path(kagglehub.dataset_download(ref))
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(path) + "\n", encoding="utf-8")
        if path.exists():
            files.extend(sorted(path.rglob("*.json")))
    return files


def discover_replay_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for manifest in args.files_json:
        data = _read_json(Path(manifest).expanduser()) or {}
        files.extend(_as_path_list(data.get("files", []) or []))
    for replay_dir in args.replay_dir:
        files.extend(sorted(Path(replay_dir).expanduser().rglob("*.json")))
    if not args.no_alphaow_manifests:
        files.extend(_manifest_json_files(Path(args.alphaow_manifest_dir).expanduser()))
        files.extend(_marker_dataset_files(Path(args.alphaow_dataset_dir).expanduser()))
    if args.download_from_alphaow_manifest:
        files.extend(
            _download_alphaow_manifest(
                Path(args.alphaow_csv).expanduser(),
                Path(args.download_root).expanduser(),
                args.start_date,
                args.end_date,
                args.limit_days,
            )
        )

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() or not args.skip_missing:
            deduped.append(path)
    return deduped


def _agent_names(data: dict) -> list[str]:
    info = data.get("info") or {}
    agents = info.get("Agents") or []
    if agents:
        return [str(agent.get("Name", f"p{i}")) for i, agent in enumerate(agents)]
    teams = info.get("TeamNames") or []
    if teams:
        return [str(name) for name in teams]
    rewards = data.get("rewards") or []
    return [f"p{i}" for i in range(len(rewards))]


def _scan_agent_stats(files: list[Path], limit: int | None = None) -> dict[str, AgentStats]:
    games: dict[str, int] = {}
    wins: dict[str, int] = {}
    scanned = 0
    for path in files:
        if limit is not None and scanned >= limit:
            break
        data = _read_json(path)
        if not data:
            continue
        rewards = data.get("rewards") or []
        names = _agent_names(data)
        if len(rewards) != len(names) or len(rewards) < 2 or any(r is None for r in rewards):
            continue
        scanned += 1
        reward_values = [float(r) for r in rewards]
        best = max(reward_values)
        winners = {i for i, value in enumerate(reward_values) if value == best}
        if len(winners) != 1:
            continue
        winner = next(iter(winners))
        for i, name in enumerate(names):
            games[name] = games.get(name, 0) + 1
            wins[name] = wins.get(name, 0) + (1 if i == winner else 0)
    return {name: AgentStats(games=count, wins=wins.get(name, 0)) for name, count in games.items()}


def choose_target_name(files: list[Path], args: argparse.Namespace) -> str | None:
    if args.target_mode != "top-agent":
        return args.target_name
    stats = _scan_agent_stats(files, args.scan_limit)
    rows = [
        (stat.win_rate, stat.games, stat.wins, name)
        for name, stat in stats.items()
        if stat.games >= args.min_agent_games
    ]
    if not rows:
        if args.target_name:
            return args.target_name
        raise RuntimeError(
            "No agent had enough cached games for --target-mode top-agent. "
            "Try --target-mode winner or lower --min-agent-games."
        )
    rows.sort(reverse=True)
    print(_c("top cached agents", BOLD + CYAN), flush=True)
    for win_rate, games, wins, name in rows[: min(10, len(rows))]:
        print(f"  {win_rate:6.1%} {wins:>5}/{games:<5} {name}", flush=True)
    return args.target_name or rows[0][3]


def _normalize_obs(obs: dict, step_index: int) -> dict:
    out = dict(obs)
    out["step"] = int(out.get("step", step_index) or step_index)
    return out


def _normalize_moves(action) -> list[list[float]]:
    if action is None:
        return []
    if isinstance(action, tuple):
        action = list(action)
    if not isinstance(action, list):
        return []
    if len(action) == 3 and not isinstance(action[0], (list, tuple)):
        return [action]
    moves = []
    for move in action:
        if isinstance(move, (list, tuple)) and len(move) >= 3:
            moves.append([move[0], move[1], move[2]])
    return moves


def _slot_selected(
    mode: str,
    slot: int,
    names: list[str],
    rewards: list[float],
    target_name: str | None,
    target_index: int | None,
) -> bool:
    if mode in {"top-agent", "agent-name"}:
        return target_name is not None and slot < len(names) and names[slot] == target_name
    if mode == "player-index":
        return target_index is not None and slot == target_index
    if mode == "all":
        return True
    if mode == "winner":
        if not rewards:
            return False
        best = max(rewards)
        return rewards[slot] == best and sum(1 for r in rewards if r == best) == 1
    raise ValueError(f"unknown target mode: {mode}")


def _labels_from_moves(obs: dict, moves: list[list[float]], encoded) -> tuple[int, tuple[int, ...], int]:
    labels = []
    invalid = 0
    seen = set()
    for move in moves:
        idx = encode_move_as_action_index(obs, move)
        if idx <= 0 or idx >= len(encoded.action_mask) or not bool(encoded.action_mask[idx]):
            invalid += 1
            continue
        if idx not in seen:
            labels.append(idx)
            seen.add(idx)
    if not labels:
        anchor = encode_teacher_moves_as_action_index(obs, moves)
        if anchor > 0:
            invalid += 1
        return 0, (0,), invalid
    by_ships = []
    for move in moves:
        idx = encode_move_as_action_index(obs, move)
        if idx in seen:
            try:
                ships = int(float(move[2]))
            except (TypeError, ValueError):
                ships = 0
            by_ships.append((ships, idx))
    anchor = max(by_ships, key=lambda item: item[0])[1] if by_ships else labels[0]
    return anchor, tuple(labels), invalid


def collect_samples(
    files: list[Path],
    args: argparse.Namespace,
    target_name: str | None,
) -> tuple[list[ReplaySample], dict]:
    rng = random.Random(args.seed)
    files = list(files)
    rng.shuffle(files)
    if args.max_replays:
        files = files[: args.max_replays]

    samples: list[ReplaySample] = []
    seen_steps = 0
    matched_steps = 0
    skipped_noops = 0
    skipped_invalid = 0
    skipped_bad_obs = 0
    matched_games = set()
    label_counts: dict[int, int] = {}

    for replay_idx, path in enumerate(files):
        if len(samples) >= args.samples:
            break
        data = _read_json(path)
        if not data:
            continue
        names = _agent_names(data)
        rewards_raw = data.get("rewards") or []
        try:
            rewards = [float(r) for r in rewards_raw]
        except (TypeError, ValueError):
            rewards = []
        steps = data.get("steps") or []
        episode_id = int((data.get("info") or {}).get("EpisodeId") or data.get("id") or replay_idx)
        for step_index, step in enumerate(steps):
            if len(samples) >= args.samples:
                break
            if not isinstance(step, list):
                continue
            for slot, entry in enumerate(step):
                if slot >= len(names):
                    continue
                if not _slot_selected(args.target_mode, slot, names, rewards, target_name, args.target_index):
                    continue
                seen_steps += 1
                if not isinstance(entry, dict) or not isinstance(entry.get("observation"), dict):
                    skipped_bad_obs += 1
                    continue
                obs = _normalize_obs(entry["observation"], step_index)
                moves = _normalize_moves(entry.get("action"))
                encoded = encode_obs(obs)
                anchor, labels, invalid = _labels_from_moves(obs, moves, encoded)
                skipped_invalid += invalid
                if anchor == 0:
                    allowed_noops = int(args.max_noop_fraction * max(1, len(samples)))
                    if label_counts.get(0, 0) >= allowed_noops:
                        skipped_noops += 1
                        continue
                samples.append(
                    ReplaySample(
                        encoded=encoded,
                        anchor_label=anchor,
                        labels=labels,
                        agent_name=names[slot],
                        episode_id=episode_id,
                        step=step_index,
                        reward=rewards[slot] if slot < len(rewards) else 0.0,
                    )
                )
                matched_steps += 1
                matched_games.add(episode_id)
                label_counts[anchor] = label_counts.get(anchor, 0) + 1
                if len(samples) >= args.samples:
                    break

    stats = {
        "samples": len(samples),
        "seen_steps": seen_steps,
        "matched_steps": matched_steps,
        "matched_games": len(matched_games),
        "skipped_noops": skipped_noops,
        "skipped_invalid_moves": skipped_invalid,
        "skipped_bad_obs": skipped_bad_obs,
        "noop_fraction": label_counts.get(0, 0) / max(1, len(samples)),
        "unique_anchor_labels": len(label_counts),
        "target_mode": args.target_mode,
        "target_name": target_name,
    }
    if not samples:
        raise RuntimeError(
            "No replay samples were collected. Check --target-name/--target-mode and the replay cache paths."
        )
    return samples, stats


def _valid_indices(sample: ReplaySample) -> np.ndarray:
    return np.flatnonzero(sample.encoded.action_mask).astype(np.int64)


def _sample_negatives(
    valid: np.ndarray,
    positives: set[int],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    pool = valid[~np.isin(valid, list(positives), assume_unique=False)]
    if pool.size == 0:
        return np.zeros((0,), dtype=np.int64)
    size = min(count, int(pool.size))
    return rng.choice(pool, size=size, replace=False).astype(np.int64)


def _sampled_multilabel_loss(
    logits: torch.Tensor,
    samples: list[ReplaySample],
    sample_indices: np.ndarray,
    valid_by_sample: list[np.ndarray],
    negatives: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    terms = []
    for row, sample_idx in enumerate(sample_indices):
        labels = tuple(int(x) for x in samples[int(sample_idx)].labels)
        positives = set(labels or (0,))
        pos_t = torch.as_tensor(sorted(positives), dtype=torch.long, device=device)
        terms.append(F.softplus(-logits[row, pos_t]).mean())
        neg = _sample_negatives(valid_by_sample[int(sample_idx)], positives, negatives, rng)
        if neg.size:
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            terms.append(F.softplus(logits[row, neg_t]).mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _action_features_np(encoded, action_index: int) -> np.ndarray:
    decoded = decode_action_index(int(action_index))
    noop = 1.0 if decoded is None else 0.0
    if decoded is None:
        src = np.zeros(PLANET_FEATURES, dtype=np.float32)
        tgt = np.zeros(PLANET_FEATURES, dtype=np.float32)
        geom = np.zeros(4, dtype=np.float32)
        send = np.zeros(len(SEND_FRACTIONS), dtype=np.float32)
    else:
        source_slot, target_slot, send_bin = decoded
        src = encoded.planets[source_slot].astype(np.float32)
        tgt = encoded.planets[target_slot].astype(np.float32)
        delta = tgt[:2] - src[:2]
        dist = float(np.linalg.norm(delta))
        geom = np.asarray([delta[0], delta[1], dist, min(20.0, 1.0 / max(1e-4, dist))], dtype=np.float32)
        send = np.zeros(len(SEND_FRACTIONS), dtype=np.float32)
        if 0 <= send_bin < len(send):
            send[send_bin] = 1.0
    return np.concatenate(
        [
            encoded.globals.astype(np.float32),
            src,
            tgt,
            src - tgt,
            src * tgt,
            geom,
            send,
            np.asarray([noop], dtype=np.float32),
        ],
        axis=0,
    )


def _reward_batch(
    samples: list[ReplaySample],
    sample_indices: np.ndarray,
    valid_by_sample: list[np.ndarray],
    negatives: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    targets = []
    for sample_idx in sample_indices:
        sample = samples[int(sample_idx)]
        positives = set(int(x) for x in sample.labels)
        for label in sorted(positives):
            features.append(_action_features_np(sample.encoded, label))
            targets.append(1.0)
        neg = _sample_negatives(valid_by_sample[int(sample_idx)], positives, negatives, rng)
        for label in neg:
            features.append(_action_features_np(sample.encoded, int(label)))
            targets.append(0.0)
    x = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(targets, dtype=np.float32), dtype=torch.float32, device=device)
    return x, y


def train_models(
    samples: list[ReplaySample],
    args: argparse.Namespace,
    stats: dict,
) -> tuple[dict, dict]:
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = build_policy(
        args.model,
        args.hidden,
        args.transformer_layers,
        args.transformer_heads,
    ).to(device)
    reward_model = ActionRewardModel(hidden=args.reward_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    reward_optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=args.reward_learning_rate,
        weight_decay=args.weight_decay,
    )

    anchors = torch.as_tensor([sample.anchor_label for sample in samples], dtype=torch.long, device=device)
    valid_by_sample = [_valid_indices(sample) for sample in samples]
    indices = np.arange(len(samples))
    metrics_path = Path(args.log_dir) / "imitation_irl_metrics.jsonl"
    metrics_path.write_text("")
    append_jsonl(metrics_path, {"phase": "collect", "epoch": 0, "step": 0, **stats})

    final_policy_loss = None
    final_anchor_accuracy = None
    final_reward_loss = None
    for epoch in range(args.epochs):
        rng.shuffle(indices)
        policy_losses = []
        ce_losses = []
        ml_losses = []
        accuracies = []
        reward_losses = []
        reward_accs = []
        for start in range(0, len(samples), args.batch_size):
            mb = indices[start : start + args.batch_size]
            batch = _stack_encoded([samples[int(i)].encoded for i in mb], device)
            logits, _value = model(**batch)
            ce = F.cross_entropy(logits, anchors[torch.as_tensor(mb, dtype=torch.long, device=device)])
            ml = _sampled_multilabel_loss(
                logits,
                samples,
                mb,
                valid_by_sample,
                args.policy_negatives,
                rng,
                device,
            )
            if args.policy_loss == "ce":
                policy_loss = ce
            elif args.policy_loss == "multilabel":
                policy_loss = ml
            else:
                policy_loss = ce + args.multilabel_weight * ml
            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                target = anchors[torch.as_tensor(mb, dtype=torch.long, device=device)]
                acc = (torch.argmax(logits, dim=-1) == target).float().mean().item()
            policy_losses.append(float(policy_loss.item()))
            ce_losses.append(float(ce.item()))
            ml_losses.append(float(ml.item()))
            accuracies.append(acc)

            if not args.skip_reward_model:
                rx, ry = _reward_batch(
                    samples,
                    mb,
                    valid_by_sample,
                    args.reward_negatives,
                    rng,
                    device,
                )
                scores = reward_model(rx)
                reward_loss = F.binary_cross_entropy_with_logits(scores, ry)
                reward_optimizer.zero_grad(set_to_none=True)
                reward_loss.backward()
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
                reward_optimizer.step()
                with torch.no_grad():
                    pred = (torch.sigmoid(scores) >= 0.5).float()
                    reward_acc = (pred == ry).float().mean().item()
                reward_losses.append(float(reward_loss.item()))
                reward_accs.append(reward_acc)

        row = {
            "phase": "imitation_irl",
            "epoch": epoch + 1,
            "step": epoch + 1,
            "policy_loss": float(np.mean(policy_losses)),
            "anchor_ce_loss": float(np.mean(ce_losses)),
            "multilabel_loss": float(np.mean(ml_losses)),
            "anchor_accuracy": float(np.mean(accuracies)),
            "reward_loss": float(np.mean(reward_losses)) if reward_losses else None,
            "reward_accuracy": float(np.mean(reward_accs)) if reward_accs else None,
            **stats,
        }
        append_jsonl(metrics_path, row)
        print(
            f"{_c('imitate', BOLD + CYAN)} epoch={row['epoch']:>3}/{args.epochs} "
            f"policy={row['policy_loss']:.4f} acc={_c(f'{row['anchor_accuracy']:.1%}', GREEN)} "
            f"reward={row['reward_loss'] if row['reward_loss'] is not None else 'skip'}",
            flush=True,
        )
        write_training_report(Path(args.log_dir))
        final_policy_loss = row["policy_loss"]
        final_anchor_accuracy = row["anchor_accuracy"]
        final_reward_loss = row["reward_loss"]

    policy_config = {
        "teacher": stats.get("target_name") or stats.get("target_mode"),
        "bc_samples": len(samples),
        "bc_epochs": args.epochs,
        "bc_final_loss": final_policy_loss,
        "bc_final_accuracy": final_anchor_accuracy,
        "model": args.model,
        "hidden": args.hidden,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "max_launches_per_turn": args.max_launches_per_turn,
        "multi_launch_logit_margin": args.multi_launch_logit_margin,
        "policy_loss": args.policy_loss,
        "multilabel_weight": args.multilabel_weight,
        **stats,
    }
    reward_config = {
        "reward_hidden": args.reward_hidden,
        "reward_negatives": args.reward_negatives,
        "reward_final_loss": final_reward_loss,
        "input_dim": GLOBAL_FEATURES + PLANET_FEATURES * 4 + 4 + len(SEND_FRACTIONS) + 1,
        **stats,
    }
    return (
        {"model": model.state_dict(), "config": policy_config, "action_dim": ACTION_DIM},
        {"model": reward_model.state_dict(), "config": reward_config},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Imitation + contrastive inverse-RL training from Kaggle Orbit Wars replays."
    )
    parser.add_argument("--files-json", action="append", default=[], help="Manifest JSON with a 'files' list.")
    parser.add_argument("--replay-dir", action="append", default=[], help="Directory containing replay JSON files.")
    parser.add_argument("--alphaow-manifest-dir", default=str(ALPHAOW_MANIFEST_DIR))
    parser.add_argument("--alphaow-dataset-dir", default=str(ALPHAOW_DATASET_DIR))
    parser.add_argument("--alphaow-csv", default=str(ALPHAOW_CSV))
    parser.add_argument("--no-alphaow-manifests", action="store_true")
    parser.add_argument("--download-from-alphaow-manifest", action="store_true")
    parser.add_argument("--download-root", default="rl_orbit_wars/data/kagglehub")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--include-missing", dest="skip_missing", action="store_false")
    parser.add_argument("--max-replays", type=int, default=None)
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument(
        "--target-mode",
        choices=["top-agent", "agent-name", "winner", "all", "player-index"],
        default="top-agent",
    )
    parser.add_argument("--target-name", default=None, help="Exact replay agent name. Overrides top-agent choice.")
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--min-agent-games", type=int, default=25)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--max-noop-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model", choices=["mlp", "entity_transformer"], default="mlp")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--policy-loss", choices=["ce", "multilabel", "both"], default="both")
    parser.add_argument("--multilabel-weight", type=float, default=0.5)
    parser.add_argument("--policy-negatives", type=int, default=64)
    parser.add_argument("--reward-hidden", type=int, default=256)
    parser.add_argument("--reward-learning-rate", type=float, default=3e-4)
    parser.add_argument("--reward-negatives", type=int, default=8)
    parser.add_argument("--skip-reward-model", action="store_true")
    parser.add_argument("--max-launches-per-turn", type=int, default=4)
    parser.add_argument("--multi-launch-logit-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="rl_orbit_wars/checkpoints/irl_policy.pt")
    parser.add_argument("--reward-out", default="rl_orbit_wars/checkpoints/irl_reward.pt")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for imitation_irl_metrics.jsonl and training_report.html. Defaults to policy checkpoint parent.",
    )
    args = parser.parse_args()

    args.max_noop_fraction = max(0.0, min(1.0, args.max_noop_fraction))
    out = Path(args.out)
    reward_out = Path(args.reward_out)
    log_dir = Path(args.log_dir) if args.log_dir else out.parent
    args.log_dir = str(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    files = discover_replay_files(args)
    print(f"{_c('replays', BOLD + CYAN)} discovered={len(files)}", flush=True)
    if not files:
        raise RuntimeError(
            "No replay files found. Try --download-from-alphaow-manifest or pass --replay-dir/--files-json."
        )
    target_name = choose_target_name(files, args)
    print(f"{_c('target', BOLD + CYAN)} {target_name or args.target_mode}", flush=True)
    samples, stats = collect_samples(files, args, target_name)
    print(
        f"{_c('collect', BOLD + CYAN)} samples={stats['samples']} games={stats['matched_games']} "
        f"noop={stats['noop_fraction']:.1%} invalid_moves={stats['skipped_invalid_moves']}",
        flush=True,
    )

    (policy_ckpt, reward_ckpt) = train_models(samples, args, stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_ckpt, out)
    if not args.skip_reward_model:
        reward_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reward_ckpt, reward_out)
    stats_path = log_dir / "imitation_irl_dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_training_report(log_dir)
    print(f"wrote {out}")
    if not args.skip_reward_model:
        print(f"wrote {reward_out}")
    print(f"wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
