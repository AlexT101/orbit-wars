from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_orbit_wars.imitation_irl import (
    ALPHAOW_CSV,
    ALPHAOW_DATASET_DIR,
    ALPHAOW_MANIFEST_DIR,
    _agent_names,
    _normalize_moves,
    _normalize_obs,
    _read_json,
    _slot_selected,
    choose_target_name,
    discover_replay_files,
)
from rl_orbit_wars.orbit_wars_rl.features import (
    GLOBAL_FEATURES,
    MAX_PLANETS,
    PLANET_FEATURES,
    encode_move_as_source_target_slots,
    encode_obs,
)
from rl_orbit_wars.orbit_wars_rl.visualization import append_jsonl, write_training_report


RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


@dataclass(frozen=True)
class PairLabel:
    source: int
    target: int
    ships: int

    @property
    def pair(self) -> int:
        return self.source * MAX_PLANETS + self.target


@dataclass(frozen=True)
class ImitationSample:
    encoded: object
    source_label: int
    target_label: int
    pair_label: int
    pair_labels: tuple[int, ...]
    agent_name: str
    episode_id: int
    step: int
    reward: float


class SourceTargetTransformer(nn.Module):
    """Entity transformer that predicts launch source and target, not ship count."""

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout

        self.planet_encoder = nn.Sequential(
            nn.Linear(PLANET_FEATURES, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
                category=UserWarning,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(hidden)
        self.source_head = nn.Sequential(
            nn.Linear(hidden + GLOBAL_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden + GLOBAL_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        pair_features = hidden * 4 + 4 + GLOBAL_FEATURES
        self.pair_head = nn.Sequential(
            nn.Linear(pair_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        planets: torch.Tensor,
        planet_mask: torch.Tensor,
        globals_: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = planets.shape[0]
        planet_tokens = self.planet_encoder(planets)
        global_token = self.global_encoder(globals_).unsqueeze(1)
        tokens = torch.cat([global_token, planet_tokens], dim=1)
        global_valid = torch.ones(batch, 1, dtype=torch.bool, device=planet_mask.device)
        valid = torch.cat([global_valid, planet_mask.bool()], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        encoded = self.final_norm(encoded)
        planet_encoded = encoded[:, 1:]

        g_planet = globals_.view(batch, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, -1)
        source_logits = self.source_head(torch.cat([planet_encoded, g_planet], dim=-1)).squeeze(-1)
        target_logits = self.target_head(torch.cat([planet_encoded, g_planet], dim=-1)).squeeze(-1)

        src = planet_encoded.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        tgt = planet_encoded.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)

        xy = planets[..., :2]
        src_xy = xy.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        tgt_xy = xy.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, 2)
        delta = tgt_xy - src_xy
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g_pair = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair_logits = self.pair_head(torch.cat([pair, pair_geom, g_pair], dim=-1)).squeeze(-1)

        planet_valid = planet_mask.bool()
        if source_mask is None:
            source_mask = planets[..., 25] > 0.5
        if pair_mask is None:
            not_same = ~torch.eye(MAX_PLANETS, dtype=torch.bool, device=planets.device).unsqueeze(0)
            pair_mask = source_mask.unsqueeze(2) & planet_valid.unsqueeze(1) & not_same
        source_logits = source_logits.masked_fill(~source_mask.bool(), -1e9)
        target_logits = target_logits.masked_fill(~planet_valid, -1e9)
        pair_logits = pair_logits.masked_fill(~pair_mask.bool(), -1e9)
        return {
            "source_logits": source_logits,
            "target_logits": target_logits,
            "pair_logits": pair_logits.reshape(batch, MAX_PLANETS * MAX_PLANETS),
        }


def _labels_from_moves(obs: dict, moves: list[list[float]]) -> tuple[PairLabel | None, tuple[int, ...], int]:
    labels: list[PairLabel] = []
    invalid = 0
    seen: set[int] = set()
    for move in moves:
        slots = encode_move_as_source_target_slots(obs, move)
        if slots is None:
            invalid += 1
            continue
        source, target = slots
        pair = source * MAX_PLANETS + target
        if pair in seen:
            continue
        try:
            ships = max(1, int(float(move[2])))
        except (TypeError, ValueError):
            ships = 0
        labels.append(PairLabel(source=source, target=target, ships=ships))
        seen.add(pair)
    if not labels:
        return None, (), invalid
    anchor = max(labels, key=lambda label: label.ships)
    return anchor, tuple(label.pair for label in labels), invalid


def collect_samples(
    files: list[Path],
    args: argparse.Namespace,
    target_name: str | None,
) -> tuple[list[ImitationSample], dict]:
    rng = random.Random(args.seed)
    files = list(files)
    rng.shuffle(files)
    if args.max_replays:
        files = files[: args.max_replays]

    samples: list[ImitationSample] = []
    seen_steps = 0
    matched_steps = 0
    skipped_noops = 0
    skipped_invalid = 0
    skipped_bad_obs = 0
    matched_games: set[int] = set()
    source_counts: dict[int, int] = {}
    target_counts: dict[int, int] = {}
    pair_counts: dict[int, int] = {}

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
                anchor, pair_labels, invalid = _labels_from_moves(obs, moves)
                skipped_invalid += invalid
                if anchor is None:
                    skipped_noops += 1
                    continue
                encoded = encode_obs(obs)
                samples.append(
                    ImitationSample(
                        encoded=encoded,
                        source_label=anchor.source,
                        target_label=anchor.target,
                        pair_label=anchor.pair,
                        pair_labels=pair_labels,
                        agent_name=names[slot],
                        episode_id=episode_id,
                        step=step_index,
                        reward=rewards[slot] if slot < len(rewards) else 0.0,
                    )
                )
                matched_steps += 1
                matched_games.add(episode_id)
                source_counts[anchor.source] = source_counts.get(anchor.source, 0) + 1
                target_counts[anchor.target] = target_counts.get(anchor.target, 0) + 1
                pair_counts[anchor.pair] = pair_counts.get(anchor.pair, 0) + 1
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
        "launch_fraction": matched_steps / max(1, seen_steps),
        "unique_sources": len(source_counts),
        "unique_targets": len(target_counts),
        "unique_pairs": len(pair_counts),
        "target_mode": args.target_mode,
        "target_name": target_name,
    }
    if not samples:
        raise RuntimeError(
            "No source-target imitation samples were collected. Check --target-name/--target-mode "
            "and replay cache paths."
        )
    return samples, stats


def _stack_encoded(items: list[object], device: torch.device) -> dict[str, torch.Tensor]:
    planets = torch.as_tensor(np.stack([x.planets for x in items]), dtype=torch.float32, device=device)
    planet_mask = torch.as_tensor(np.stack([x.planet_mask for x in items]), dtype=torch.float32, device=device)
    globals_ = torch.as_tensor(np.stack([x.globals for x in items]), dtype=torch.float32, device=device)
    source_mask = planets[..., 25] > 0.5
    not_same = ~torch.eye(MAX_PLANETS, dtype=torch.bool, device=device).unsqueeze(0)
    pair_mask = source_mask.unsqueeze(2) & planet_mask.bool().unsqueeze(1) & not_same
    return {
        "planets": planets,
        "planet_mask": planet_mask,
        "globals_": globals_,
        "source_mask": source_mask,
        "pair_mask": pair_mask,
    }


def _valid_pair_indices(sample: ImitationSample) -> np.ndarray:
    encoded = sample.encoded
    planet_mask = np.asarray(encoded.planet_mask, dtype=bool)
    source_mask = np.asarray(encoded.planets[:, 25] > 0.5, dtype=bool)
    valid = source_mask[:, None] & planet_mask[None, :]
    np.fill_diagonal(valid, False)
    return np.flatnonzero(valid.reshape(-1)).astype(np.int64)


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
    return rng.choice(pool, size=min(count, int(pool.size)), replace=False).astype(np.int64)


def _sampled_multilabel_loss(
    pair_logits: torch.Tensor,
    samples: list[ImitationSample],
    sample_indices: np.ndarray,
    valid_by_sample: list[np.ndarray],
    negatives: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    terms = []
    for row, sample_idx in enumerate(sample_indices):
        sample = samples[int(sample_idx)]
        positives = set(int(x) for x in sample.pair_labels)
        pos_t = torch.as_tensor(sorted(positives), dtype=torch.long, device=device)
        terms.append(F.softplus(-pair_logits[row, pos_t]).mean())
        neg = _sample_negatives(valid_by_sample[int(sample_idx)], positives, negatives, rng)
        if neg.size:
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            terms.append(F.softplus(pair_logits[row, neg_t]).mean())
    if not terms:
        return pair_logits.sum() * 0.0
    return torch.stack(terms).mean()


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.shape[-1])
    pred = torch.topk(logits, k=k, dim=-1).indices
    return (pred == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _run_epoch(
    *,
    model: SourceTargetTransformer,
    samples: list[ImitationSample],
    indices: np.ndarray,
    valid_by_sample: list[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    pair_losses: list[float] = []
    source_losses: list[float] = []
    target_losses: list[float] = []
    multilabel_losses: list[float] = []
    pair_accs: list[float] = []
    pair_top3: list[float] = []
    pair_top5: list[float] = []
    source_accs: list[float] = []
    target_accs: list[float] = []
    grad_norms: list[float] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(indices), args.batch_size):
            mb = indices[start : start + args.batch_size]
            batch = _stack_encoded([samples[int(i)].encoded for i in mb], device)
            source_labels = torch.as_tensor(
                [samples[int(i)].source_label for i in mb],
                dtype=torch.long,
                device=device,
            )
            target_labels = torch.as_tensor(
                [samples[int(i)].target_label for i in mb],
                dtype=torch.long,
                device=device,
            )
            pair_labels = torch.as_tensor(
                [samples[int(i)].pair_label for i in mb],
                dtype=torch.long,
                device=device,
            )
            out = model(**batch)
            source_loss = F.cross_entropy(out["source_logits"], source_labels)
            target_loss = F.cross_entropy(out["target_logits"], target_labels)
            pair_loss = F.cross_entropy(out["pair_logits"], pair_labels)
            multilabel_loss = (
                _sampled_multilabel_loss(
                    out["pair_logits"],
                    samples,
                    mb,
                    valid_by_sample,
                    args.pair_negatives,
                    rng,
                    device,
                )
                if args.multilabel_weight > 0
                else out["pair_logits"].sum() * 0.0
            )
            loss = (
                pair_loss
                + args.source_weight * source_loss
                + args.target_weight * target_loss
                + args.multilabel_weight * multilabel_loss
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                grad_norms.append(float(grad_norm.item()))

            with torch.no_grad():
                pair_accs.append(_topk_accuracy(out["pair_logits"], pair_labels, 1))
                pair_top3.append(_topk_accuracy(out["pair_logits"], pair_labels, 3))
                pair_top5.append(_topk_accuracy(out["pair_logits"], pair_labels, 5))
                source_accs.append(_topk_accuracy(out["source_logits"], source_labels, 1))
                target_accs.append(_topk_accuracy(out["target_logits"], target_labels, 1))
            losses.append(float(loss.item()))
            pair_losses.append(float(pair_loss.item()))
            source_losses.append(float(source_loss.item()))
            target_losses.append(float(target_loss.item()))
            multilabel_losses.append(float(multilabel_loss.item()))

    return {
        "loss": _mean(losses),
        "pair_loss": _mean(pair_losses),
        "source_loss": _mean(source_losses),
        "target_loss": _mean(target_losses),
        "multilabel_loss": _mean(multilabel_losses),
        "pair_accuracy": _mean(pair_accs),
        "pair_top3": _mean(pair_top3),
        "pair_top5": _mean(pair_top5),
        "source_accuracy": _mean(source_accs),
        "target_accuracy": _mean(target_accs),
        "grad_norm": _mean(grad_norms),
    }


def _prefixed(prefix: str, row: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def _save_checkpoint(
    path: Path,
    *,
    model: SourceTargetTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    stats: dict,
    best_val_pair_accuracy: float,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "dataset_stats": stats,
            "best_val_pair_accuracy": best_val_pair_accuracy,
            "best_val_loss": best_val_loss,
            "model_class": "SourceTargetTransformer",
            "label_schema": "source_slot,target_slot,pair_slot_no_count",
            "pair_dim": MAX_PLANETS * MAX_PLANETS,
        },
        path,
    )


def train(
    samples: list[ImitationSample],
    args: argparse.Namespace,
    stats: dict,
) -> dict:
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = SourceTargetTransformer(
        hidden=args.hidden,
        layers=args.transformer_layers,
        heads=args.transformer_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    best_val_pair_accuracy = -1.0
    best_val_loss = float("inf")

    log_dir = Path(args.log_dir)
    state_path = Path(args.state_path) if args.state_path else log_dir / "imitation_state.pt"
    if args.resume and state_path.exists():
        ckpt = torch.load(state_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_pair_accuracy = float(ckpt.get("best_val_pair_accuracy", -1.0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    indices = np.arange(len(samples))
    rng.shuffle(indices)
    val_count = int(round(len(samples) * args.val_fraction))
    if len(samples) > 1:
        val_count = min(max(1, val_count), len(samples) - 1)
    else:
        val_count = 0
    val_indices = indices[:val_count]
    train_indices = indices[val_count:] if val_count else indices
    valid_by_sample = [_valid_pair_indices(sample) for sample in samples]

    metrics_path = log_dir / "imitation_metrics.jsonl"
    if not args.resume or not metrics_path.exists():
        metrics_path.write_text("", encoding="utf-8")
    config = {
        "framework": "source_target_imitation",
        "target_name": stats.get("target_name"),
        "target_mode": stats.get("target_mode"),
        "samples": len(samples),
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "hidden": args.hidden,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "source_weight": args.source_weight,
        "target_weight": args.target_weight,
        "multilabel_weight": args.multilabel_weight,
        "pair_negatives": args.pair_negatives,
        "seed": args.seed,
    }
    (log_dir / "imitation_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    append_jsonl(
        metrics_path,
        {
            "phase": "collect",
            "epoch": start_epoch,
            "step": start_epoch,
            "train_samples": int(len(train_indices)),
            "val_samples": int(len(val_indices)),
            **stats,
        },
    )
    write_training_report(log_dir)

    latest_path = log_dir / "latest_source_target.pt"
    out = Path(args.out)
    started = time.perf_counter()
    final_row: dict = {}
    for epoch in range(start_epoch, args.epochs):
        rng.shuffle(train_indices)
        t0 = time.perf_counter()
        train_row = _run_epoch(
            model=model,
            samples=samples,
            indices=train_indices,
            valid_by_sample=valid_by_sample,
            args=args,
            device=device,
            rng=rng,
            optimizer=optimizer,
        )
        val_row = (
            _run_epoch(
                model=model,
                samples=samples,
                indices=val_indices,
                valid_by_sample=valid_by_sample,
                args=args,
                device=device,
                rng=rng,
                optimizer=None,
            )
            if len(val_indices)
            else train_row
        )
        elapsed = time.perf_counter() - t0
        total_elapsed = time.perf_counter() - started
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "phase": "imitation",
            "epoch": epoch + 1,
            "step": epoch + 1,
            "samples": len(samples),
            "train_samples": int(len(train_indices)),
            "val_samples": int(len(val_indices)),
            "samples_per_sec": len(train_indices) / max(1e-9, elapsed),
            "epoch_seconds": elapsed,
            "total_seconds": total_elapsed,
            "lr": lr,
            **_prefixed("train", train_row),
            **_prefixed("val", val_row),
            **stats,
        }
        append_jsonl(metrics_path, row)
        improved = (
            row["val_pair_accuracy"] > best_val_pair_accuracy
            or (
                abs(row["val_pair_accuracy"] - best_val_pair_accuracy) < 1e-12
                and row["val_loss"] < best_val_loss
            )
        )
        if improved:
            best_val_pair_accuracy = row["val_pair_accuracy"]
            best_val_loss = row["val_loss"]
        _save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            config=config,
            stats=stats,
            best_val_pair_accuracy=best_val_pair_accuracy,
            best_val_loss=best_val_loss,
        )
        _save_checkpoint(
            state_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            config=config,
            stats=stats,
            best_val_pair_accuracy=best_val_pair_accuracy,
            best_val_loss=best_val_loss,
        )
        if improved:
            _save_checkpoint(
                out,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                config=config,
                stats=stats,
                best_val_pair_accuracy=best_val_pair_accuracy,
                best_val_loss=best_val_loss,
            )
        write_training_report(log_dir)
        pair_text = f"{row['val_pair_accuracy']:.1%}"
        src_text = f"{row['val_source_accuracy']:.1%}"
        tgt_text = f"{row['val_target_accuracy']:.1%}"
        print(
            f"{_c('imitate', BOLD + CYAN)} epoch={epoch + 1:>3}/{args.epochs} "
            f"loss={row['val_loss']:.4f} pair={_c(pair_text, GREEN)} "
            f"src={_c(src_text, BLUE)} tgt={_c(tgt_text, BLUE)} "
            f"{_c('best', YELLOW) if improved else ''}",
            flush=True,
        )
        final_row = row

    append_jsonl(
        metrics_path,
        {
            "phase": "imitation_done",
            "epoch": args.epochs,
            "step": args.epochs,
            "checkpoint": str(out),
            "latest_checkpoint": str(latest_path),
            "best_val_pair_accuracy": best_val_pair_accuracy,
            "best_val_loss": best_val_loss,
            **stats,
        },
    )
    return {
        "checkpoint": str(out),
        "latest_checkpoint": str(latest_path),
        "metrics_path": str(metrics_path),
        "best_val_pair_accuracy": best_val_pair_accuracy,
        "best_val_loss": best_val_loss,
        "final": final_row,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train source-target imitation from Orbit Wars replay actions. Ship count is not modeled."
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
        default="agent-name",
    )
    parser.add_argument("--target-name", default="Isaiah @ Tufa Labs", help="Exact replay agent name.")
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--min-agent-games", type=int, default=25)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--source-weight", type=float, default=0.35)
    parser.add_argument("--target-weight", type=float, default=0.35)
    parser.add_argument("--multilabel-weight", type=float, default=0.25)
    parser.add_argument("--pair-negatives", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="rl_orbit_wars/checkpoints/imitation_source_target.pt")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.val_fraction = max(0.0, min(0.5, args.val_fraction))
    out = Path(args.out)
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
        f"launch={stats['launch_fraction']:.1%} unique_pairs={stats['unique_pairs']} "
        f"invalid_moves={stats['skipped_invalid_moves']}",
        flush=True,
    )
    (log_dir / "imitation_dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    result = train(samples, args, stats)
    write_training_report(log_dir)
    print(f"wrote {result['checkpoint']}")
    print(f"wrote {result['latest_checkpoint']}")
    print(f"wrote {result['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
