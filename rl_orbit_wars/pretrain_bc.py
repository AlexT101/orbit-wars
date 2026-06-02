from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_orbit_wars.orbit_wars_rl.env import OrbitWarsDuelEnv
from rl_orbit_wars.orbit_wars_rl.features import (
    ACTION_DIM,
    decode_move,
    encode_obs,
    encode_teacher_moves_as_action_index,
)
from rl_orbit_wars.orbit_wars_rl.heuristics import nearest_capture_action_index
from rl_orbit_wars.orbit_wars_rl.model import build_policy
from rl_orbit_wars.orbit_wars_rl.ppo import _stack_encoded
from rl_orbit_wars.orbit_wars_rl.visualization import append_jsonl, write_training_report

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _bot_entry(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path
    direct = ROOT / "bots" / name_or_path / "main.py"
    if direct.is_file():
        return direct
    for group in (ROOT / "bots").iterdir():
        candidate = group / name_or_path / "main.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find bot '{name_or_path}'")


def _load_agent(name_or_path: str):
    if name_or_path == "nearest":
        return None
    path = _bot_entry(name_or_path)
    module_name = f"bc_teacher_{path.parent.name}"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.agent
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load teacher bot '{name_or_path}' from {path}. "
            "For apollo, build its native module first with `maturin develop --release` "
            "inside bots/mine/apollo."
        ) from exc


def _teacher_label_and_moves(obs: dict, teacher_name: str, teacher_agent):
    if teacher_name == "nearest":
        label = nearest_capture_action_index(obs)
        return label, decode_move(obs, label)
    moves = teacher_agent(obs)
    label = encode_teacher_moves_as_action_index(obs, moves)
    return label, moves or []


def collect(
    samples: int,
    seed: int,
    opponents: list[str],
    teacher_name: str,
    advance_with_teacher: bool,
    max_noop_fraction: float,
):
    observations = []
    labels = []
    label_counts: dict[int, int] = {}
    seen_steps = 0
    skipped_noops = 0
    skipped_invalid = 0
    teacher_agent = _load_agent(teacher_name)
    episode = 0
    opponent_idx = 0
    env = OrbitWarsDuelEnv(seed=seed, opponent=opponents[opponent_idx])
    obs = env.reset(seed)
    max_seen_steps = max(10_000, samples * 200)

    while len(labels) < samples:
        if seen_steps >= max_seen_steps:
            raise RuntimeError(
                f"BC collection stalled after {seen_steps} teacher steps with only {len(labels)} "
                f"accepted labels. The teacher may be returning mostly noop; try a larger "
                "--max-noop-fraction or check that the teacher bot runs."
            )
        label, teacher_moves = _teacher_label_and_moves(obs, teacher_name, teacher_agent)
        seen_steps += 1
        keep = True
        if label == 0:
            allowed_noops = int(max_noop_fraction * max(1, len(labels)))
            keep = label_counts.get(0, 0) < allowed_noops
            if not keep:
                skipped_noops += 1
        if keep:
            encoded = encode_obs(obs)
            if label < 0 or label >= len(encoded.action_mask) or not bool(encoded.action_mask[label]):
                skipped_invalid += 1
                keep = False
        if keep:
            observations.append(encoded)
            labels.append(label)
            label_counts[label] = label_counts.get(label, 0) + 1

        if advance_with_teacher:
            result = env.step_moves(teacher_moves)
        else:
            result = env.step(label)
        obs = result.obs

        if result.done:
            episode += 1
            opponent_idx = episode % len(opponents)
            env = OrbitWarsDuelEnv(seed=seed + episode, opponent=opponents[opponent_idx])
            obs = env.reset(seed + episode)

    stats = {
        "samples": len(labels),
        "seen_steps": seen_steps,
        "skipped_noops": skipped_noops,
        "skipped_invalid": skipped_invalid,
        "max_noop_fraction": max_noop_fraction,
        "noop_fraction": label_counts.get(0, 0) / max(1, len(labels)),
        "unique_labels": len(label_counts),
    }
    return observations, np.asarray(labels, dtype=np.int64), stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Behavior-clone a teacher bot into the small PPO policy.")
    parser.add_argument("--teacher", default="nearest", help="Bot name, bot main.py path, or 'nearest'.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model", choices=["mlp", "entity_transformer"], default="mlp")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--opponents",
        default="random,nearest,baselines/starter",
        help="Comma-separated scripted opponents for data collection.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="rl_orbit_wars/checkpoints/bc_teacher.pt")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for bc_metrics.jsonl and training_report.html. Defaults to checkpoint parent.",
    )
    parser.add_argument(
        "--advance-with-policy",
        action="store_true",
        help="Advance games with the discretized policy action instead of the teacher's full moves.",
    )
    parser.add_argument(
        "--max-noop-fraction",
        type=float,
        default=0.10,
        help="Cap accepted noop labels during BC collection. Noop states are still used to advance the game.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    log_dir = Path(args.log_dir) if args.log_dir else out.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    bc_metrics_path = log_dir / "bc_metrics.jsonl"
    bc_metrics_path.write_text("")

    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    observations, labels_np, collection_stats = collect(
        args.samples,
        args.seed,
        opponents,
        args.teacher,
        advance_with_teacher=not args.advance_with_policy,
        max_noop_fraction=max(0.0, min(1.0, args.max_noop_fraction)),
    )
    collection_row = {
        "phase": "bc_collection",
        "epoch": 0,
        "step": 0,
        "teacher": args.teacher,
        "opponents": ",".join(opponents),
        **collection_stats,
    }
    append_jsonl(bc_metrics_path, collection_row)
    print(
        f"{_c('bc collect', BOLD + CYAN)} teacher={_c(args.teacher, BLUE)} "
        f"samples={collection_stats['samples']} seen={collection_stats['seen_steps']} "
        f"noop={collection_stats['noop_fraction']:.1%} "
        f"skipped_noop={collection_stats['skipped_noops']} "
        f"skipped_invalid={collection_stats.get('skipped_invalid', 0)}",
        flush=True,
    )

    model = build_policy(
        args.model,
        args.hidden,
        args.transformer_layers,
        args.transformer_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    inds = np.arange(args.samples)
    final_loss = None
    final_accuracy = None

    for epoch in range(args.epochs):
        np.random.shuffle(inds)
        losses = []
        accs = []
        for start in range(0, args.samples, args.batch_size):
            mb = inds[start : start + args.batch_size]
            mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)
            batch = _stack_encoded([observations[i] for i in mb], device)
            logits, _value = model(**batch)
            loss = F.cross_entropy(logits, labels[mb_t])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                acc = (torch.argmax(logits, dim=-1) == labels[mb_t]).float().mean().item()
            losses.append(float(loss.item()))
            accs.append(acc)
        row = {
            "phase": "bc",
            "epoch": epoch + 1,
            "step": epoch + 1,
            "teacher": args.teacher,
            "samples": args.samples,
            "loss": float(np.mean(losses)),
            "accuracy": float(np.mean(accs)),
            **collection_stats,
        }
        append_jsonl(bc_metrics_path, row)
        print(
            f"{_c('bc', BOLD + CYAN)} epoch={row['epoch']:>3}/{args.epochs} "
            f"loss={row['loss']:.4f} acc={_c(f'{row['accuracy']:.1%}', GREEN)}",
            flush=True,
        )
        write_training_report(log_dir)
        final_loss = row["loss"]
        final_accuracy = row["accuracy"]

    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_config = {
        "bc_samples": args.samples,
        "teacher": args.teacher,
        "opponents": opponents,
        "bc_epochs": args.epochs,
        "bc_final_loss": final_loss,
        "bc_final_accuracy": final_accuracy,
        "model": args.model,
        "hidden": args.hidden,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        **collection_stats,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "config": checkpoint_config,
            "action_dim": ACTION_DIM,
        },
        out,
    )
    append_jsonl(
        bc_metrics_path,
        {
            "phase": "bc_done",
            "epoch": args.epochs + 1,
            "step": args.epochs + 1,
            "checkpoint": str(out),
            **checkpoint_config,
        },
    )
    write_training_report(log_dir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
