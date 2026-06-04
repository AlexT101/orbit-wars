from __future__ import annotations

import json
import os
import random
import sys
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import asdict, dataclass
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional for local smoke tests.
    wandb = None

IL_DIR = Path(__file__).resolve().parent
EXPERIMENTAL_ARCH_DIR = IL_DIR.parent
TRAIN_DIR = EXPERIMENTAL_ARCH_DIR / "train_transformer"
REPO_ROOT = EXPERIMENTAL_ARCH_DIR.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from features import ACTION_DIM  # noqa: E402
from model import build_policy  # noqa: E402


DATASET_PATH = IL_DIR / "data" / "isaiah_tufa_labs_2p_wins_bc_manifest.json"
OUT_DIR = IL_DIR / "checkpoints" / "isaiah_bc_transformer"
LATEST_CHECKPOINT = OUT_DIR / "latest.pt"
BEST_CHECKPOINT = OUT_DIR / "best.pt"
METRICS_JSONL = OUT_DIR / "metrics.jsonl"
DATASET_STATS_JSON = OUT_DIR / "dataset_stats.json"

SEED = 123
DEVICE = "cuda"
MODEL = "entity_transformer"
HIDDEN = 128
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 4
EPOCHS = 12
BATCH_SIZE = 64
DATALOADER_WORKERS = min(8, os.cpu_count() or 1)
DATALOADER_PREFETCH_FACTOR = 4
USE_AMP = True
AMP_DTYPE = "bfloat16"
LOG_EVERY_STEPS = 200
CHECKPOINT_EVERY_STEPS = 1000
CHECKPOINT_EVERY_EPOCHS = 1
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
MAX_GRAD_NORM = 1.0
VAL_FRACTION = 0.10
USE_WANDB = True
WANDB_PROJECT = "orbit-wars"
WANDB_RUN_NAME = "isaiah-bc-transformer"
ISAIAH_NAME = "Isaiah @ Tufa Labs"


def rebase_legacy_path(path: str | Path, *, manifest_dir: Path) -> Path:
    candidate = Path(path)

    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        parts = candidate.parts
        if "experimental_arch" in parts:
            suffix = parts[parts.index("experimental_arch") + 1 :]
            return EXPERIMENTAL_ARCH_DIR.joinpath(*suffix)
        return candidate

    for base in (manifest_dir, IL_DIR, REPO_ROOT, Path.cwd()):
        rebased = base / candidate
        if rebased.exists():
            return rebased
    return IL_DIR / candidate


@dataclass(frozen=True)
class ILConfig:
    dataset_path: str
    player_name: str
    action_dim: int
    seed: int
    device: str
    model: str
    hidden: int
    transformer_layers: int
    transformer_heads: int
    epochs: int
    batch_size: int
    dataloader_workers: int
    dataloader_prefetch_factor: int
    use_amp: bool
    amp_dtype: str
    log_every_steps: int
    checkpoint_every_steps: int
    checkpoint_every_epochs: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str


class ChunkedILDataset(Dataset):
    def __init__(self, chunks: list[dict], manifest_dir: Path, cache_size: int = 2) -> None:
        self.chunks = chunks
        self.paths = [rebase_legacy_path(chunk["path"], manifest_dir=manifest_dir) for chunk in chunks]
        self.lengths = [int(chunk["rows"]) for chunk in chunks]
        self.game_ids_by_chunk = [str(chunk["game_id"]) for chunk in chunks]
        self.offsets = [0]
        for length in self.lengths:
            self.offsets.append(self.offsets[-1] + length)
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.offsets[-1]

    def chunk_rows(self, chunk_index: int) -> range:
        start = self.offsets[chunk_index]
        return range(start, start + self.lengths[chunk_index])

    def _load_chunk(self, chunk_index: int) -> dict[str, torch.Tensor]:
        if chunk_index in self._cache:
            self._cache.move_to_end(chunk_index)
            return self._cache[chunk_index]
        payload = torch.load(self.paths[chunk_index], map_location="cpu", weights_only=False)
        tensors = payload["tensors"]
        self._cache[chunk_index] = tensors
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tensors

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk_index = bisect_right(self.offsets, idx) - 1
        local_idx = idx - self.offsets[chunk_index]
        tensors = self._load_chunk(chunk_index)
        return {
            "planets": tensors["planets"][local_idx].float(),
            "planet_mask": tensors["planet_mask"][local_idx].float(),
            "globals_": tensors["globals_"][local_idx].float(),
            "action_mask": tensors["action_mask"][local_idx].bool(),
            "label": tensors["labels"][local_idx].long(),
            "value": tensors["values"][local_idx].float(),
            "weight": tensors["weights"][local_idx].float(),
        }


class ChunkBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: ChunkedILDataset,
        chunk_indices: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.chunk_indices = list(chunk_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        chunks = list(self.chunk_indices)
        if self.shuffle:
            rng.shuffle(chunks)
        for chunk_index in chunks:
            rows = list(self.dataset.chunk_rows(chunk_index))
            if self.shuffle:
                rng.shuffle(rows)
            for start in range(0, len(rows), self.batch_size):
                yield rows[start : start + self.batch_size]

    def __len__(self) -> int:
        rows = sum(self.dataset.lengths[i] for i in self.chunk_indices)
        return (rows + self.batch_size - 1) // self.batch_size


class WeightedChunkBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: ChunkedILDataset,
        chunk_indices: list[int],
        batch_size: int,
        num_samples: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.chunk_indices = list(chunk_indices)
        self.batch_size = batch_size
        self.num_samples = int(num_samples)
        self.seed = seed
        self.epoch = 0
        sums = []
        for chunk_index in self.chunk_indices:
            chunk = self.dataset.chunks[chunk_index]
            weight_sum = chunk.get("weight_sum")
            if weight_sum is None:
                weight_sum = float(self.dataset._load_chunk(chunk_index)["weights"].float().sum().item())
            sums.append(max(float(weight_sum), 1.0e-8))
        self.chunk_weights = torch.tensor(sums, dtype=torch.float32)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        remaining = self.num_samples
        while remaining > 0:
            batch_n = min(self.batch_size, remaining)
            picked = int(torch.multinomial(self.chunk_weights, 1, replacement=True, generator=generator).item())
            chunk_index = self.chunk_indices[picked]
            chunk = self.dataset._load_chunk(chunk_index)
            weights = chunk["weights"].float().clamp_min(0.0)
            if float(weights.sum().item()) <= 0.0:
                weights = torch.ones_like(weights)
            local_rows = torch.multinomial(weights, batch_n, replacement=True, generator=generator)
            base = self.dataset.offsets[chunk_index]
            yield [base + int(i) for i in local_rows.tolist()]
            remaining -= batch_n

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ILConfig,
    epoch: int,
    global_step: int,
    metrics: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ILConfig,
    epoch: int,
    global_step: int,
    metrics: dict,
    run,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, optimizer, cfg, epoch, global_step, metrics), path)
    if run is not None:
        wandb.save(str(path), policy="now")


def make_config() -> ILConfig:
    return ILConfig(
        dataset_path=str(DATASET_PATH),
        player_name=ISAIAH_NAME,
        action_dim=ACTION_DIM,
        seed=SEED,
        device=DEVICE,
        model=MODEL,
        hidden=HIDDEN,
        transformer_layers=TRANSFORMER_LAYERS,
        transformer_heads=TRANSFORMER_HEADS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        dataloader_workers=DATALOADER_WORKERS,
        dataloader_prefetch_factor=DATALOADER_PREFETCH_FACTOR,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        log_every_steps=LOG_EVERY_STEPS,
        checkpoint_every_steps=CHECKPOINT_EVERY_STEPS,
        checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        val_fraction=VAL_FRACTION,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
    )


def load_prebuilt_dataset(path: Path, expected_player_name: str) -> tuple[ChunkedILDataset, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"prebuilt IL manifest not found: {path}\n"
            f"Run: python {IL_DIR / 'build_dataset.py'}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 2:
        raise ValueError(f"unsupported IL manifest format_version={payload.get('format_version')!r}")
    player_name = payload.get("player_name")
    if player_name != expected_player_name:
        raise ValueError(f"dataset player name mismatch: expected {expected_player_name!r}, got {player_name!r}")
    chunks = list(payload.get("chunks") or [])
    if not chunks:
        raise ValueError(f"IL manifest has no chunks: {path}")
    return ChunkedILDataset(chunks, manifest_dir=path.parent), dict(payload.get("stats") or {})


def split_chunks_by_game(dataset: ChunkedILDataset, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    games = sorted(set(dataset.game_ids_by_chunk))
    if len(games) < 2:
        chunks = list(range(len(dataset.chunks)))
        return chunks, []
    rng = random.Random(seed)
    rng.shuffle(games)
    n_val = max(1, int(round(len(games) * val_fraction)))
    val_games = set(games[:n_val])
    train_chunks = [i for i, game_id in enumerate(dataset.game_ids_by_chunk) if game_id not in val_games]
    val_chunks = [i for i, game_id in enumerate(dataset.game_ids_by_chunk) if game_id in val_games]
    return train_chunks, val_chunks


def make_loaders(
    dataset: ChunkedILDataset,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    train_chunks, val_chunks = split_chunks_by_game(dataset, val_fraction, seed)
    if not val_chunks:
        val_chunks = train_chunks[:1]
        train_chunks = train_chunks[1:] or val_chunks
    train_rows = sum(dataset.lengths[i] for i in train_chunks)
    val_rows = sum(dataset.lengths[i] for i in val_chunks)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        dataset,
        batch_sampler=WeightedChunkBatchSampler(
            dataset,
            train_chunks,
            batch_size,
            num_samples=train_rows,
            seed=seed,
        ),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        dataset,
        batch_sampler=ChunkBatchSampler(dataset, val_chunks, batch_size, shuffle=False, seed=seed),
        **loader_kwargs,
    )
    return train_loader, val_loader, train_rows, val_rows


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def can_use_dataloader_workers() -> bool:
    try:
        listener = Listener(authkey=b"torch-dataloader-probe", backlog=128)
        listener.close()
    except OSError:
        return False
    return True


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE='cuda' but PyTorch cannot see CUDA. Check the NVIDIA driver/container GPU access; "
                "training will not silently fall back to CPU."
            )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError("AMP_DTYPE='bfloat16' but this CUDA device does not support bfloat16")
        return torch.bfloat16
    raise ValueError(f"unsupported AMP_DTYPE={dtype_name!r}; expected 'bfloat16' or 'float16'")


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    launch_total = 0
    launch_correct = 0
    noop_total = 0
    noop_correct = 0
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _value = model(batch["planets"], batch["planet_mask"], batch["globals_"], batch["action_mask"])
        loss_rows = F.cross_entropy(logits, batch["label"], reduction="none")
        pred = torch.argmax(logits, dim=-1)
        total_loss += float(loss_rows.sum().item())
        total += int(batch["label"].numel())
        correct += int((pred == batch["label"]).sum().item())
        launch_mask = batch["label"] != 0
        noop_mask = ~launch_mask
        launch_total += int(launch_mask.sum().item())
        noop_total += int(noop_mask.sum().item())
        launch_correct += int(((pred == batch["label"]) & launch_mask).sum().item())
        noop_correct += int(((pred == batch["label"]) & noop_mask).sum().item())
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "launch_accuracy": launch_correct / max(1, launch_total),
        "noop_accuracy": noop_correct / max(1, noop_total),
        "rows": float(total),
    }


def main() -> int:
    cfg = make_config()
    run = None
    device = resolve_device(cfg.device)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    amp_dtype = resolve_amp_dtype(cfg.amp_dtype) if amp_enabled else torch.float32
    actual_workers = cfg.dataloader_workers if can_use_dataloader_workers() else 0
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSONL.write_text("", encoding="utf-8")

    dataset, stats_payload = load_prebuilt_dataset(Path(cfg.dataset_path), cfg.player_name)
    DATASET_STATS_JSON.write_text(json.dumps(stats_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_stats = stats_payload.get("dataset", {})

    if cfg.use_wandb:
        if wandb is None:
            raise RuntimeError("USE_WANDB=True but wandb is not installed")
        wandb.login()
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={
                **asdict(cfg),
                "actual_dataloader_workers": actual_workers,
                "out_dir": str(OUT_DIR),
                "latest_checkpoint": str(LATEST_CHECKPOINT),
                "best_checkpoint": str(BEST_CHECKPOINT),
            },
            resume="allow",
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    if run is not None:
        wandb.log(
            {
                "dataset/replays_seen": dataset_stats.get("replays_seen", 0),
                "dataset/replays_kept": dataset_stats.get("replays_kept", 0),
                "dataset/rows": dataset_stats.get("rows", len(dataset)),
                "dataset/noop_rows": dataset_stats.get("noop_rows", 0),
                "dataset/launch_rows": dataset_stats.get("launch_rows", 0),
                "dataset/skipped_invalid": dataset_stats.get("skipped_invalid", 0),
                "dataset/skipped_bad_ship_fraction": dataset_stats.get("skipped_bad_ship_fraction", 0),
                "dataset/multi_launch_steps": dataset_stats.get("multi_launch_steps", 0),
                "dataset/noop_fraction": stats_payload["noop_fraction"],
                "dataset/launch_fraction": stats_payload["launch_fraction"],
                "value/mean": stats_payload["value_mean"],
                "value/p50": stats_payload["value_p50"],
                "value/p90": stats_payload["value_p90"],
                "sample_weight/mean": stats_payload["sample_weight_mean"],
                "sample_weight/p10": stats_payload["sample_weight_p10"],
                "sampling/value_weighted": 1,
                "loss/value_weighted": 0,
                "epoch": 0,
            }
        )
    print(
        f"loaded {cfg.dataset_path}\n"
        f"rows={len(dataset)} games={stats_payload.get('unique_games', len(set(dataset.game_ids_by_chunk)))} "
        f"launch={dataset_stats.get('launch_rows', 0)} noop={dataset_stats.get('noop_rows', 0)} "
        f"invalid={dataset_stats.get('skipped_invalid', 0)} "
        f"bad_frac={dataset_stats.get('skipped_bad_ship_fraction', 0)} "
        f"weight_mean={stats_payload.get('sample_weight_mean', 0.0):.3f} "
        f"device={device} amp={int(amp_enabled)} amp_dtype={cfg.amp_dtype} workers={actual_workers}"
    )

    train_loader, val_loader, train_rows, val_rows = make_loaders(
        dataset,
        cfg.batch_size,
        cfg.val_fraction,
        cfg.seed,
        actual_workers,
        cfg.device.startswith("cuda"),
        cfg.dataloader_prefetch_factor,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    model = build_policy(cfg.model, cfg.hidden, cfg.transformer_layers, cfg.transformer_heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_accs: list[float] = []
        recent_losses: list[float] = []
        recent_accs: list[float] = []
        epoch_rows = 0
        epoch_t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        log_t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        wall_t0 = None
        if device.type == "cuda":
            assert epoch_t0 is not None and log_t0 is not None
            epoch_t0.record()
            log_t0.record()
        else:
            import time

            wall_t0 = time.perf_counter()
        for raw_batch in train_loader:
            global_step += 1
            batch = batch_to_device(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, _value = model(batch["planets"], batch["planet_mask"], batch["globals_"], batch["action_mask"])
                loss = ce_loss(logits, batch["label"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = float((pred == batch["label"]).float().mean().item())
                loss_value = float(loss.item())
                batch_rows = int(batch["label"].numel())
                epoch_rows += batch_rows
                train_accs.append(acc)
                train_losses.append(loss_value)
                recent_accs.append(acc)
                recent_losses.append(loss_value)

            if cfg.log_every_steps and global_step % cfg.log_every_steps == 0:
                if device.type == "cuda":
                    now = torch.cuda.Event(enable_timing=True)
                    now.record()
                    torch.cuda.synchronize()
                    assert log_t0 is not None
                    dt = log_t0.elapsed_time(now) / 1000.0
                    log_t0 = now
                else:
                    import time

                    assert wall_t0 is not None
                    now_wall = time.perf_counter()
                    dt = now_wall - wall_t0
                    wall_t0 = now_wall
                rows_per_sec = cfg.log_every_steps * cfg.batch_size / max(dt, 1.0e-9)
                log_row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train/loss_recent": float(np.mean(recent_losses)),
                    "train/accuracy_recent": float(np.mean(recent_accs)),
                    "train/rows_per_sec": rows_per_sec,
                }
                print(
                    f"step {global_step} epoch={epoch:02d} "
                    f"loss={log_row['train/loss_recent']:.4f} "
                    f"acc={log_row['train/accuracy_recent']:.3f} "
                    f"rows/s={rows_per_sec:.0f}",
                    flush=True,
                )
                if run is not None:
                    wandb.log(log_row, step=global_step)
                recent_losses.clear()
                recent_accs.clear()

            if cfg.checkpoint_every_steps and global_step % cfg.checkpoint_every_steps == 0:
                save_checkpoint(
                    LATEST_CHECKPOINT,
                    model,
                    optimizer,
                    cfg,
                    epoch,
                    global_step,
                    {"train_loss_recent": float(np.mean(train_losses[-100:])), "epoch_rows": epoch_rows},
                    run,
                )

        val = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
        if device.type == "cuda":
            epoch_end = torch.cuda.Event(enable_timing=True)
            epoch_end.record()
            torch.cuda.synchronize()
            assert epoch_t0 is not None
            epoch_seconds = epoch_t0.elapsed_time(epoch_end) / 1000.0
        else:
            import time

            assert wall_t0 is not None
            epoch_seconds = time.perf_counter() - wall_t0
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(np.mean(train_accs)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_launch_accuracy": val["launch_accuracy"],
            "val_noop_accuracy": val["noop_accuracy"],
            "train_rows": int(train_rows),
            "val_rows": int(val_rows),
            "train_rows_per_sec": epoch_rows / max(epoch_seconds, 1.0e-9),
        }
        append_jsonl(METRICS_JSONL, row)
        if run is not None:
            wandb.log(
                {
                    "train/loss": row["train_loss"],
                    "train/accuracy": row["train_accuracy"],
                    "val/loss": row["val_loss"],
                    "val/accuracy": row["val_accuracy"],
                    "val/launch_accuracy": row["val_launch_accuracy"],
                    "val/noop_accuracy": row["val_noop_accuracy"],
                    "rows/train": row["train_rows"],
                    "rows/val": row["val_rows"],
                    "train/rows_per_sec_epoch": row["train_rows_per_sec"],
                    "epoch": epoch,
                }
            )
        print(
            f"epoch {epoch:02d} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.3f} val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.3f} launch={row['val_launch_accuracy']:.3f} "
            f"rows/s={row['train_rows_per_sec']:.0f}"
        )

        save_checkpoint(LATEST_CHECKPOINT, model, optimizer, cfg, epoch, global_step, row, run)
        if cfg.checkpoint_every_epochs and epoch % cfg.checkpoint_every_epochs == 0:
            save_checkpoint(OUT_DIR / f"epoch_{epoch:03d}.pt", model, optimizer, cfg, epoch, global_step, row, run)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            save_checkpoint(BEST_CHECKPOINT, model, optimizer, cfg, epoch, global_step, row, run)

    print(f"wrote {LATEST_CHECKPOINT}")
    print(f"best {BEST_CHECKPOINT}")
    if run is not None:
        wandb.save(str(LATEST_CHECKPOINT))
        if BEST_CHECKPOINT.exists():
            wandb.save(str(BEST_CHECKPOINT))
        wandb.save(str(METRICS_JSONL))
        wandb.save(str(DATASET_STATS_JSON))
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
