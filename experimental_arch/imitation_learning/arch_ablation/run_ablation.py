from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

IL_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = IL_DIR.parent / "train_transformer"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(IL_DIR) not in sys.path:
    sys.path.insert(0, str(IL_DIR))

from constants import (  # noqa: E402
    ACTIONS_DIM,
    GLOBAL_DIM,
    PAIR_OUTCOME_DIM,
    PAIR_OUTCOME_SHAPE,
    PAIR_TURN_SHAPE,
    PLANET_SLOTS,
    PLANET_TIMELINE_DIM,
    PLANET_TIMELINE_SHAPE,
    PRESENCE_SHAPE,
    SEND_ALL_ACTION,
    SEND_HALF_ACTION,
    TOKEN_DIM,
    TOKEN_SHAPE,
)
from train import DATASET_PATH, rebase_legacy_path  # noqa: E402

SEND_ACTIONS = (SEND_HALF_ACTION, SEND_ALL_ACTION)
SEND_FRACTIONS = (0.5, 1.0)
ACTION_DIM = 1 + PLANET_SLOTS * PLANET_SLOTS * len(SEND_ACTIONS)
CURRENT_XY_SLICE = slice(11, 13)


@dataclass(frozen=True)
class Variant:
    name: str
    hidden: int
    layers: int
    heads: int
    action_features: bool = False
    temporal_frames: bool = False
    cross_attention: bool = False
    factorized_cross_attention: bool = False
    ngpt_backbone: bool = False


VARIANTS = {
    "baseline": Variant("baseline", hidden=128, layers=3, heads=4),
    "action_features": Variant("action_features", hidden=128, layers=3, heads=4, action_features=True),
    "scaled": Variant("scaled", hidden=192, layers=4, heads=6, action_features=True),
    "temporal": Variant("temporal", hidden=192, layers=4, heads=6, action_features=True, temporal_frames=True),
    "cross_attention": Variant(
        "cross_attention",
        hidden=192,
        layers=4,
        heads=6,
        action_features=True,
        temporal_frames=True,
        cross_attention=True,
    ),
    "factorized_ca": Variant(
        "factorized_ca",
        hidden=192,
        layers=4,
        heads=6,
        action_features=True,
        temporal_frames=True,
        factorized_cross_attention=True,
    ),
    "ngpt_action_features": Variant(
        "ngpt_action_features",
        hidden=128,
        layers=3,
        heads=4,
        action_features=True,
        ngpt_backbone=True,
    ),
    "ngpt_scaled": Variant(
        "ngpt_scaled",
        hidden=192,
        layers=4,
        heads=6,
        action_features=True,
        ngpt_backbone=True,
    ),
}


class ChunkedAblationDataset(Dataset):
    def __init__(self, chunks: list[dict], manifest_dir: Path, chunk_indices: list[int], cache_size: int = 2) -> None:
        self.all_chunks = chunks
        self.chunk_indices = list(chunk_indices)
        self.chunks = [chunks[i] for i in self.chunk_indices]
        self.paths = [rebase_legacy_path(chunk["path"], manifest_dir=manifest_dir) for chunk in self.chunks]
        self.lengths = [int(chunk["rows"]) for chunk in self.chunks]
        self.offsets = [0]
        for length in self.lengths:
            self.offsets.append(self.offsets[-1] + length)
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.offsets[-1]

    def chunk_rows(self, local_chunk_index: int) -> range:
        start = self.offsets[local_chunk_index]
        return range(start, start + self.lengths[local_chunk_index])

    def _load_chunk(self, local_chunk_index: int) -> dict[str, torch.Tensor]:
        if local_chunk_index in self._cache:
            self._cache.move_to_end(local_chunk_index)
            return self._cache[local_chunk_index]
        path = self.paths[local_chunk_index]
        with np.load(path, allow_pickle=False) as payload:
            our_ship_fraction = payload["our_ship_fraction"].astype(np.float32, copy=False)
            player_rank = payload["player_rank"].astype(np.float32, copy=False)
            weights = ((1.0 - our_ship_fraction) ** 2) * (1.0 - player_rank / 30.0)
            weights = np.clip(weights, 0.0, None).astype(np.float32, copy=False)
            tensors = {
                "tokens": torch.from_numpy(payload["tokens"]),
                "presence": torch.from_numpy(payload["presence"]),
                "globals_": torch.from_numpy(payload["globals_"]),
                "action_mask": torch.from_numpy(payload["action_mask"]),
                "pair_turns": torch.from_numpy(payload["pair_turns"]),
                "pair_reachable_mask": torch.from_numpy(payload["pair_reachable_mask"]),
                "pair_outcome_features": torch.from_numpy(payload["pair_outcome_features"]),
                "planet_timeline_features": torch.from_numpy(payload["planet_timeline_features"]),
                "labels": torch.from_numpy(payload["labels"]),
                "weights": torch.from_numpy(weights),
            }
        self._cache[local_chunk_index] = tensors
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tensors

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        local_chunk_index = bisect_right(self.offsets, idx) - 1
        local_idx = idx - self.offsets[local_chunk_index]
        tensors = self._load_chunk(local_chunk_index)
        return {
            "tokens": tensors["tokens"][local_idx].float(),
            "presence": tensors["presence"][local_idx].float(),
            "globals_": tensors["globals_"][local_idx].float(),
            "action_mask": tensors["action_mask"][local_idx].bool(),
            "pair_turns": tensors["pair_turns"][local_idx].float(),
            "pair_reachable_mask": tensors["pair_reachable_mask"][local_idx].float(),
            "pair_outcome_features": tensors["pair_outcome_features"][local_idx].float(),
            "planet_timeline_features": tensors["planet_timeline_features"][local_idx].float(),
            "label": tensors["labels"][local_idx].long(),
            "weight": tensors["weights"][local_idx].float(),
        }


class ChunkBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: ChunkedAblationDataset, batch_size: int, shuffle: bool, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        chunks = list(range(len(self.dataset.chunks)))
        if self.shuffle:
            rng.shuffle(chunks)
        for chunk_index in chunks:
            rows = list(self.dataset.chunk_rows(chunk_index))
            if self.shuffle:
                rng.shuffle(rows)
            for start in range(0, len(rows), self.batch_size):
                yield rows[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)


def hypersphere_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1.0e-6) * math.sqrt(x.shape[-1])


class NormalizedEncoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.attn_alpha = nn.Parameter(torch.tensor(0.05))
        self.mlp_alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = hypersphere_norm(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = hypersphere_norm(x + self.attn_alpha * hypersphere_norm(attn_out))
        mlp_out = self.mlp(x)
        x = hypersphere_norm(x + self.mlp_alpha * hypersphere_norm(mlp_out))
        return x


class NormalizedEncoder(nn.Module):
    def __init__(self, hidden: int, heads: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([NormalizedEncoderLayer(hidden, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        return x


class ILPolicyAblation(nn.Module):
    def __init__(self, variant: Variant) -> None:
        super().__init__()
        self.variant = variant
        hidden = variant.hidden
        planet_in = TOKEN_DIM if variant.temporal_frames else TOKEN_DIM + PLANET_TIMELINE_DIM
        self.planet_encoder = nn.Sequential(
            nn.Linear(planet_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.frame_embedding = nn.Parameter(torch.zeros(4, hidden)) if variant.temporal_frames else None
        if variant.ngpt_backbone:
            self.transformer = NormalizedEncoder(hidden, variant.heads, variant.layers)
            self.final_norm = nn.Identity()
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=variant.heads,
                dim_feedforward=hidden * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=variant.layers)
            self.final_norm = nn.LayerNorm(hidden)

        action_extra = 0
        if variant.action_features:
            action_extra = 8
        pair_features = hidden * 4 + 4 + GLOBAL_DIM + PAIR_OUTCOME_DIM + action_extra
        if variant.factorized_cross_attention:
            self.source_cross_attention = nn.MultiheadAttention(hidden, variant.heads, batch_first=True, dropout=0.0)
            self.target_cross_attention = nn.MultiheadAttention(hidden, variant.heads, batch_first=True, dropout=0.0)
        if variant.cross_attention:
            self.pair_query = nn.Sequential(nn.Linear(pair_features, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.cross_attention = nn.MultiheadAttention(hidden, variant.heads, batch_first=True, dropout=0.0)
            self.pair_head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        else:
            self.pair_head = nn.Sequential(
                nn.Linear(pair_features, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        self.noop_head = nn.Sequential(nn.Linear(hidden + GLOBAL_DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden + GLOBAL_DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def encode_entities(self, tokens, presence, globals_, timeline):
        batch = tokens.shape[0]
        global_token = self.global_encoder(globals_).unsqueeze(1)
        if self.variant.temporal_frames:
            frame_tokens = self.planet_encoder(tokens)
            assert self.frame_embedding is not None
            frame_tokens = frame_tokens + self.frame_embedding.view(1, 4, 1, -1)
            planet_seq = frame_tokens.reshape(batch, 4 * PLANET_SLOTS, -1)
            valid_planets = presence.bool().reshape(batch, 4 * PLANET_SLOTS)
            seq = torch.cat([global_token, planet_seq], dim=1)
            valid = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=tokens.device), valid_planets], dim=1)
            encoded = self.final_norm(self.transformer(seq, src_key_padding_mask=~valid))
            global_encoded = encoded[:, 0]
            planet_encoded = encoded[:, 1 : 1 + PLANET_SLOTS]
            memory = torch.cat([global_encoded.unsqueeze(1), planet_encoded], dim=1)
            memory_valid = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=tokens.device), presence[:, 0].bool()], dim=1)
            return global_encoded, planet_encoded, memory, memory_valid

        current = torch.cat([tokens[:, 0], timeline], dim=-1)
        planet_tokens = self.planet_encoder(current)
        seq = torch.cat([global_token, planet_tokens], dim=1)
        valid = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=tokens.device), presence[:, 0].bool()], dim=1)
        encoded = self.final_norm(self.transformer(seq, src_key_padding_mask=~valid))
        return encoded[:, 0], encoded[:, 1:], encoded, valid

    def action_feature_tensor(self, tokens, pair_turns, pair_reachable_mask):
        batch = tokens.shape[0]
        device = tokens.device
        dtype = tokens.dtype
        frac = torch.tensor(SEND_FRACTIONS, dtype=dtype, device=device).view(1, 1, 1, 2, 1)
        is_half = torch.tensor([1.0, 0.0], dtype=dtype, device=device).view(1, 1, 1, 2, 1)
        is_all = torch.tensor([0.0, 1.0], dtype=dtype, device=device).view(1, 1, 1, 2, 1)
        turns = pair_turns[:, :, :, SEND_ACTIONS].unsqueeze(-1) / 20.0
        reachable = pair_reachable_mask[:, :, :, SEND_ACTIONS].unsqueeze(-1)
        ships = tokens[:, 0, :, 10]
        src_ships = ships.unsqueeze(2).unsqueeze(3).expand(batch, PLANET_SLOTS, PLANET_SLOTS, 2).unsqueeze(-1)
        tgt_ships = ships.unsqueeze(1).unsqueeze(3).expand(batch, PLANET_SLOTS, PLANET_SLOTS, 2).unsqueeze(-1)
        keep_frac = 1.0 - frac
        sent_proxy = src_ships * frac
        return torch.cat(
            [
                frac.expand(batch, PLANET_SLOTS, PLANET_SLOTS, 2, 1),
                is_half.expand(batch, PLANET_SLOTS, PLANET_SLOTS, 2, 1),
                is_all.expand(batch, PLANET_SLOTS, PLANET_SLOTS, 2, 1),
                keep_frac.expand(batch, PLANET_SLOTS, PLANET_SLOTS, 2, 1),
                turns,
                reachable,
                sent_proxy,
                tgt_ships,
            ],
            dim=-1,
        )

    def forward(self, batch):
        tokens = batch["tokens"]
        presence = batch["presence"]
        globals_ = batch["globals_"]
        batch_size = tokens.shape[0]
        global_encoded, planet_encoded, memory, memory_valid = self.encode_entities(
            tokens, presence, globals_, batch["planet_timeline_features"]
        )
        source_encoded = planet_encoded
        target_encoded = planet_encoded
        if self.variant.factorized_cross_attention:
            src_context, _ = self.source_cross_attention(
                planet_encoded,
                memory,
                memory,
                key_padding_mask=~memory_valid,
                need_weights=False,
            )
            tgt_context, _ = self.target_cross_attention(
                planet_encoded,
                memory,
                memory,
                key_padding_mask=~memory_valid,
                need_weights=False,
            )
            source_encoded = hypersphere_norm(planet_encoded + src_context)
            target_encoded = hypersphere_norm(planet_encoded + tgt_context)

        src = source_encoded.unsqueeze(2).expand(batch_size, PLANET_SLOTS, PLANET_SLOTS, -1)
        tgt = target_encoded.unsqueeze(1).expand(batch_size, PLANET_SLOTS, PLANET_SLOTS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)
        xy = tokens[:, 0, :, CURRENT_XY_SLICE]
        src_xy = xy.unsqueeze(2).expand(batch_size, PLANET_SLOTS, PLANET_SLOTS, 2)
        tgt_xy = xy.unsqueeze(1).expand(batch_size, PLANET_SLOTS, PLANET_SLOTS, 2)
        delta = tgt_xy - src_xy
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch_size, 1, 1, GLOBAL_DIM).expand(batch_size, PLANET_SLOTS, PLANET_SLOTS, -1)
        pair_base = torch.cat([pair, pair_geom, g], dim=-1)
        pair_base = pair_base.unsqueeze(3).expand(batch_size, PLANET_SLOTS, PLANET_SLOTS, len(SEND_ACTIONS), -1)
        outcome = batch["pair_outcome_features"][:, :, :, SEND_ACTIONS, :].to(dtype=pair_base.dtype)
        pair_features = [pair_base, outcome]
        if self.variant.action_features:
            pair_features.append(self.action_feature_tensor(tokens, batch["pair_turns"], batch["pair_reachable_mask"]))
        pair_input = torch.cat(pair_features, dim=-1)

        if self.variant.cross_attention:
            query_base = self.pair_query(pair_input).reshape(batch_size, PLANET_SLOTS * PLANET_SLOTS * len(SEND_ACTIONS), -1)
            attended, _ = self.cross_attention(query_base, memory, memory, key_padding_mask=~memory_valid, need_weights=False)
            pair_logits = self.pair_head(torch.cat([query_base, attended], dim=-1)).squeeze(-1)
        else:
            pair_logits = self.pair_head(pair_input).reshape(batch_size, -1)
        state = torch.cat([global_encoded, globals_], dim=-1)
        logits = torch.cat([self.noop_head(state), pair_logits], dim=-1)
        logits = logits.masked_fill(~batch["action_mask"], -1e9)
        value = self.value_head(state).squeeze(-1)
        return logits, value


def load_manifest(path: Path) -> tuple[list[dict], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["chunks"]), path.parent


def choose_chunks(chunks: list[dict], train_chunks: int, val_chunks: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(len(chunks)))
    rng.shuffle(indices)
    return indices[:train_chunks], indices[train_chunks : train_chunks + val_chunks]


def count_games(dataset: ChunkedAblationDataset) -> int:
    total = 0
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as payload:
            if "games_json" in payload:
                total += len(json.loads(str(payload["games_json"])))
    return total


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def weighted_loss(logits, labels, weights):
    rows = F.cross_entropy(logits, labels, reduction="none")
    weights = weights.to(dtype=rows.dtype).clamp_min(0.0)
    return (rows * weights).sum() / weights.sum().clamp_min(1.0e-8)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    total = correct = launch_total = launch_correct = noop_total = noop_correct = 0
    loss_sum = weighted_loss_sum = weight_sum = 0.0
    for raw in loader:
        batch = batch_to_device(raw, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits, _ = model(batch)
        loss_rows = F.cross_entropy(logits, batch["label"], reduction="none")
        weights = batch["weight"].to(dtype=loss_rows.dtype).clamp_min(0.0)
        pred = logits.argmax(dim=-1)
        labels = batch["label"]
        launch = labels != 0
        noop = ~launch
        total += labels.numel()
        correct += int((pred == labels).sum().item())
        launch_total += int(launch.sum().item())
        noop_total += int(noop.sum().item())
        launch_correct += int(((pred == labels) & launch).sum().item())
        noop_correct += int(((pred == labels) & noop).sum().item())
        loss_sum += float(loss_rows.sum().item())
        weighted_loss_sum += float((loss_rows * weights).sum().item())
        weight_sum += float(weights.sum().item())
    return {
        "val_loss": loss_sum / max(1, total),
        "val_weighted_loss": weighted_loss_sum / max(1.0e-8, weight_sum),
        "val_accuracy": correct / max(1, total),
        "val_launch_accuracy": launch_correct / max(1, launch_total),
        "val_noop_accuracy": noop_correct / max(1, noop_total),
        "val_rows": total,
    }


def train_variant(variant: Variant, train_loader, val_loader, args, device, out_dir: Path) -> dict:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    amp_dtype = torch.bfloat16
    model = ILPolicyAblation(variant).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    t0 = time.perf_counter()
    step = 0
    train_losses: list[float] = []
    train_accs: list[float] = []
    rows_seen = 0
    for epoch in range(1, args.epochs + 1):
        for raw in train_loader:
            step += 1
            batch = batch_to_device(raw, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits, _ = model(batch)
                loss = weighted_loss(logits, batch["label"], batch["weight"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                train_losses.append(float(loss.item()))
                train_accs.append(float((pred == batch["label"]).float().mean().item()))
                rows_seen += int(batch["label"].numel())
            if args.steps > 0 and step >= args.steps:
                break
        if args.steps > 0 and step >= args.steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t0
    val = evaluate(model, val_loader, device, amp_dtype)
    result = {
        "variant": asdict(variant),
        "params": params,
        "train_steps": step,
        "train_rows_seen": rows_seen,
        "train_rows_per_sec": rows_seen / max(train_seconds, 1.0e-9),
        "train_loss": float(np.mean(train_losses)),
        "train_accuracy": float(np.mean(train_accs)),
        **val,
    }
    (out_dir / f"{variant.name}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--train-chunks", type=int, default=10)
    parser.add_argument("--val-chunks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cross-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunks, manifest_dir = load_manifest(args.dataset)
    train_indices, val_indices = choose_chunks(chunks, args.train_chunks, args.val_chunks, args.seed)
    train_ds = ChunkedAblationDataset(chunks, manifest_dir, train_indices)
    val_ds = ChunkedAblationDataset(chunks, manifest_dir, val_indices)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "dataset": str(args.dataset),
        "train_chunk_indices": train_indices,
        "val_chunk_indices": val_indices,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "train_games": count_games(train_ds),
        "val_games": count_games(val_ds),
        "device": str(device),
        "args": vars(args) | {"dataset": str(args.dataset), "out_dir": str(args.out_dir)},
    }
    (args.out_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(split, sort_keys=True), flush=True)

    results = []
    results_path = args.out_dir / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    for name in args.variants:
        variant = VARIANTS[name]
        batch_size = args.cross_batch_size if variant.cross_attention else args.batch_size
        train_loader = DataLoader(
            train_ds,
            batch_sampler=ChunkBatchSampler(train_ds, batch_size, shuffle=True, seed=args.seed),
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=ChunkBatchSampler(val_ds, batch_size, shuffle=False, seed=args.seed),
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
        )
        print(f"running {name} batch={batch_size}", flush=True)
        result = train_variant(variant, train_loader, val_loader, args, device, args.out_dir)
        results.append(result)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    summary = sorted(results, key=lambda row: row["val_weighted_loss"])
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ranked:")
    for row in summary:
        print(
            f"{row['variant']['name']}: val_weighted={row['val_weighted_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.3f} rows/s={row['train_rows_per_sec']:.0f} params={row['params']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
