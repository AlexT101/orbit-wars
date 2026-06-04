"""Set-encoder for per-planet target prediction.

Architectures:
- deepsets:     per-planet MLP + mean/max pool. Cheapest floor.
- transformer:  N planets + 1 CLS token, vanilla nn.TransformerEncoder.
- distbias:     transformer + learnable per-head distance bias on attention scores.
- graphsage:    k-NN GraphSAGE (mean aggregation from nearest neighbors).
- gat:          k-NN graph attention (attention over nearest neighbors only).

Training options added 2026-06-03:
- --max-step S: drop rows with step >= S from both train and val.
- --opening-weight: opening-emphasis schedule. "smooth" applies 1/(1+step/30) to
  the row loss, "linear" 1.0->0.2 by turn 150, "step" 3x for turns 0-30, or "off".
- best-checkpoint criterion is val recall@8 (was AUC).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch not importable; pip install torch") from exc


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_npz(path: Path):
    # Avoid .astype() copies for arrays already in the target dtype; large
    # planet_feats (~9 GB at 786K rows × 50 × 58 × 4B) needlessly doubles RAM.
    z = np.load(path, allow_pickle=True)
    def _as(arr, dtype):
        return arr if arr.dtype == dtype else arr.astype(dtype)
    return dict(
        planet_feats=_as(z["planet_feats"], np.float32),
        globals=_as(z["globals"], np.float32),
        mask=_as(z["mask"], np.bool_),
        labels=_as(z["labels"], np.float32),
        meta=_as(z["meta"], np.int64),
        feat_names=list(z["feat_names"]) if "feat_names" in z.files else None,
        global_names=list(z["global_names"]) if "global_names" in z.files else None,
    )


def game_level_split(meta: np.ndarray, seed: int = 42, val_frac: float = 0.12):
    games = meta[:, 0]
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(val_frac * len(unique)))
    val_games = set(unique[:n_val].tolist())
    val_mask = np.array([g in val_games for g in games])
    return ~val_mask, val_mask


def fit_norm(planet_feats, globals_, train_mask, real_mask):
    # Avoid the 8 GB train_planets intermediate by combining masks directly.
    combined = train_mask[:, None] & real_mask    # (N, N_max) bool
    flat = planet_feats[combined]                  # (R_train_real, F)  ~110 MB
    p_mean = flat.mean(axis=0)
    p_std = flat.std(axis=0) + 1e-6
    del flat
    g_mean = globals_[train_mask].mean(axis=0)
    g_std = globals_[train_mask].std(axis=0) + 1e-6
    return p_mean.astype(np.float32), p_std.astype(np.float32), \
           g_mean.astype(np.float32), g_std.astype(np.float32)


def apply_norm(planet_feats, globals_, p_mean, p_std, g_mean, g_std):
    # MUTATES inputs in place. Callers must drop their references to the
    # originals if they don't want them normalized. Avoids 9 GB temporaries.
    p_mean32 = p_mean.astype(np.float32, copy=False)
    p_std32 = p_std.astype(np.float32, copy=False)
    g_mean32 = g_mean.astype(np.float32, copy=False)
    g_std32 = g_std.astype(np.float32, copy=False)
    planet_feats -= p_mean32
    planet_feats /= p_std32
    globals_ -= g_mean32
    globals_ /= g_std32
    return planet_feats, globals_


def opening_weight(step: np.ndarray, schedule: str) -> np.ndarray:
    """Return per-row loss weight in [0, ~3] given an array of step indices."""
    s = step.astype(np.float32)
    if schedule == "off":
        return np.ones_like(s)
    if schedule == "smooth":
        return 1.0 / (1.0 + s / 30.0)
    if schedule == "linear":
        # 1.0 at turn 0, 0.2 at turn 150, flat after
        w = 1.0 - 0.8 * np.clip(s / 150.0, 0.0, 1.0)
        return w
    if schedule == "step":
        return np.where(s < 30, 3.0, 1.0)
    raise ValueError(f"unknown opening-weight schedule: {schedule}")


# ---------------------------------------------------------------------------
# Position / k-NN helpers (re-used by GNN variants)
# ---------------------------------------------------------------------------


# After apply_norm the original column indices are preserved; x_now=11, y_now=12.
POS_X_IDX = 11
POS_Y_IDX = 12


def pairwise_dist(planet_feats: torch.Tensor) -> torch.Tensor:
    """L2 distance between every planet pair using the normalized x_now/y_now
    columns. Shape: (B, N, N)."""
    xy = planet_feats[:, :, [POS_X_IDX, POS_Y_IDX]]
    return torch.cdist(xy, xy, p=2)


def knn_indices(dist: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """For each planet, return indices of its k nearest real neighbors.
    Padding columns and self-edges get inf distance so they're never picked.
    Shape: (B, N, k)."""
    B, N, _ = dist.shape
    not_mask_col = ~(mask.unsqueeze(1).expand(-1, N, -1))
    d = dist.masked_fill(not_mask_col, float("inf"))
    eye = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
    d = d.masked_fill(eye, float("inf"))
    k_eff = min(k, N - 1)
    _, idx = d.topk(k_eff, largest=False)
    return idx


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class DeepSets(nn.Module):
    def __init__(self, f_planet, f_global, hidden=64):
        super().__init__()
        self.planet_enc = nn.Sequential(
            nn.Linear(f_planet, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.global_enc = nn.Sequential(nn.Linear(f_global, hidden), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, planet_feats, globals_, mask):
        h = self.planet_enc(planet_feats)
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (h * mask_f).sum(dim=1) / denom
        h_for_max = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)
        max_pool = h_for_max.max(dim=1).values
        g = self.global_enc(globals_)
        context = torch.cat([mean_pool, max_pool, g], dim=-1)
        context_b = context.unsqueeze(1).expand(-1, h.size(1), -1)
        return self.head(torch.cat([h, context_b], dim=-1)).squeeze(-1)


class PlanetTransformer(nn.Module):
    """Vanilla transformer encoder over N planet tokens + 1 CLS token (globals)."""

    def __init__(self, f_planet, f_global, d_model=64, n_heads=4, n_layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.planet_proj = nn.Linear(f_planet, d_model)
        self.global_proj = nn.Linear(f_global, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        # enable_nested_tensor=False because the nested-tensor fast path uses
        # _nested_tensor_from_mask_left_aligned which is not implemented on MPS.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, planet_feats, globals_, mask):
        B = planet_feats.size(0)
        p_tok = self.planet_proj(planet_feats)
        g_tok = self.global_proj(globals_).unsqueeze(1)
        tokens = torch.cat([g_tok, p_tok], dim=1)
        kpm = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
            ~mask,
        ], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=kpm)
        return self.head(out[:, 1:, :]).squeeze(-1)


class DistanceBiasedAttention(nn.Module):
    """Multi-head self-attention with an additive per-head distance bias."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dist_alpha = nn.Parameter(torch.full((n_heads,), 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dist_bias, kpm):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # subtract |alpha| * distance — far planets get attenuated
        scores = scores - dist_bias.unsqueeze(1) * self.dist_alpha.view(1, -1, 1, 1).abs()
        if kpm is not None:
            scores = scores.masked_fill(kpm.view(B, 1, 1, L), float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out)


class DistBiasedEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff, dropout=0.1):
        super().__init__()
        self.attn = DistanceBiasedAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dist_bias, kpm):
        x = x + self.dropout(self.attn(self.norm1(x), dist_bias, kpm))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DistanceBiasedTransformer(nn.Module):
    """Transformer where each attention head learns its own distance attenuation."""

    def __init__(self, f_planet, f_global, d_model=64, n_heads=4, n_layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.planet_proj = nn.Linear(f_planet, d_model)
        self.global_proj = nn.Linear(f_global, d_model)
        self.layers = nn.ModuleList([
            DistBiasedEncoderLayer(d_model, n_heads, ff, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, planet_feats, globals_, mask):
        B, N, _ = planet_feats.shape
        dist = pairwise_dist(planet_feats)              # (B, N, N)
        # extend to include the CLS token at position 0 with zero distance (no bias)
        dist_full = torch.zeros(B, N + 1, N + 1, device=dist.device, dtype=dist.dtype)
        dist_full[:, 1:, 1:] = dist

        p_tok = self.planet_proj(planet_feats)
        g_tok = self.global_proj(globals_).unsqueeze(1)
        tokens = torch.cat([g_tok, p_tok], dim=1)
        kpm = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
            ~mask,
        ], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, dist_full, kpm)
        return self.head(tokens[:, 1:, :]).squeeze(-1)


class PlanetGraphSAGE(nn.Module):
    """k-NN GraphSAGE: each layer mixes a planet's feature with the mean of its
    k nearest real neighbors. Concatenates with a global token at the head."""

    def __init__(self, f_planet, f_global, d_model=64, n_layers=2, k=8, dropout=0.1):
        super().__init__()
        self.k = k
        self.planet_proj = nn.Linear(f_planet, d_model)
        self.global_proj = nn.Linear(f_global, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, planet_feats, globals_, mask):
        B, N, _ = planet_feats.shape
        dist = pairwise_dist(planet_feats)
        idx = knn_indices(dist, mask, self.k)  # (B, N, k)

        h = self.planet_proj(planet_feats)
        for layer, norm in zip(self.layers, self.norms):
            D = h.size(-1)
            # gather neighbors: result (B, N, k, D)
            h_expand = h.unsqueeze(1).expand(-1, N, -1, -1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, D)
            neigh = torch.gather(h_expand, 2, gather_idx)
            agg = neigh.mean(dim=2)
            h = norm(h + layer(torch.cat([h, agg], dim=-1)))

        g = self.global_proj(globals_).unsqueeze(1).expand(-1, N, -1)
        return self.head(torch.cat([h, g], dim=-1)).squeeze(-1)


class PlanetGAT(nn.Module):
    """k-NN graph attention: per layer, attend over k nearest real neighbors
    with multi-head additive attention. No fully-connected attention."""

    def __init__(self, f_planet, f_global, d_model=64, n_heads=4, n_layers=2, k=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.k = k
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.planet_proj = nn.Linear(f_planet, d_model)
        self.global_proj = nn.Linear(f_global, d_model)
        self.attn_scorer = nn.ModuleList([
            nn.Linear(2 * d_model, n_heads) for _ in range(n_layers)
        ])
        self.value_proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, planet_feats, globals_, mask):
        B, N, _ = planet_feats.shape
        dist = pairwise_dist(planet_feats)
        idx = knn_indices(dist, mask, self.k)  # (B, N, k)
        k_eff = idx.size(-1)

        h = self.planet_proj(planet_feats)
        for attn_layer, val_layer, norm in zip(self.attn_scorer, self.value_proj, self.norms):
            D = h.size(-1)
            # neighbor features
            h_expand = h.unsqueeze(1).expand(-1, N, -1, -1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, D)
            neigh = torch.gather(h_expand, 2, gather_idx)         # (B, N, k, D)
            self_h = h.unsqueeze(2).expand(-1, -1, k_eff, -1)     # (B, N, k, D)
            scores = attn_layer(torch.cat([self_h, neigh], dim=-1))  # (B, N, k, H)
            attn_w = scores.softmax(dim=2)                         # softmax across neighbors
            v = val_layer(neigh).reshape(B, N, k_eff, self.n_heads, self.head_dim)
            out = (attn_w.unsqueeze(-1) * v).sum(dim=2)           # (B, N, H, head_dim)
            out = out.reshape(B, N, -1)
            h = norm(h + self.dropout(out))

        g = self.global_proj(globals_).unsqueeze(1).expand(-1, N, -1)
        return self.head(torch.cat([h, g], dim=-1)).squeeze(-1)


def build_model(arch: str, f_planet: int, f_global: int,
                d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                hidden: int = 64, ff: int = 128, dropout: float = 0.1, k: int = 8):
    if arch == "deepsets":
        return DeepSets(f_planet, f_global, hidden=hidden)
    if arch == "transformer":
        return PlanetTransformer(f_planet, f_global,
                                 d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                                 ff=ff, dropout=dropout)
    if arch == "distbias":
        return DistanceBiasedTransformer(f_planet, f_global,
                                         d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                                         ff=ff, dropout=dropout)
    if arch == "graphsage":
        return PlanetGraphSAGE(f_planet, f_global, d_model=d_model,
                               n_layers=n_layers, k=k, dropout=dropout)
    if arch == "gat":
        return PlanetGAT(f_planet, f_global, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, k=k, dropout=dropout)
    raise ValueError(f"unknown arch: {arch}")


# ---------------------------------------------------------------------------
# Loss / eval
# ---------------------------------------------------------------------------


def masked_weighted_bce(logits, labels, mask, row_w, pos_weight=None):
    """BCE per planet slot, masked by padding and multiplied by per-row weights.
    row_w: (B,) tensor in [0, ~3]."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=pos_weight,
    )
    bce = bce * mask.float() * row_w.unsqueeze(-1)
    denom = (mask.float() * row_w.unsqueeze(-1)).sum().clamp_min(1.0)
    return bce.sum() / denom


def _approx_auc(y, p, max_n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    if len(y) > max_n:
        idx = rng.choice(len(y), size=max_n, replace=False)
        y = y[idx]; p = p[idx]
    pos = p[y > 0.5]; neg = p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_pos_sum = ranks[:len(pos)].sum()
    return float((rank_pos_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def predict_all(model, pf, gl, mk, device, batch=512):
    out = []
    for i in range(0, len(pf), batch):
        out.append(model(
            torch.from_numpy(pf[i:i+batch]).to(device),
            torch.from_numpy(gl[i:i+batch]).to(device),
            torch.from_numpy(mk[i:i+batch]).to(device),
        ).cpu().numpy())
    return np.concatenate(out, axis=0)


def eval_metrics(logits, labels, mask, recall_ks=(1, 3, 5, 8, 10)):
    flat_l = logits[mask]; flat_y = labels[mask]
    probs = 1.0 / (1.0 + np.exp(-flat_l))
    acc = float(((probs >= 0.5) == (flat_y >= 0.5)).mean())
    auc = _approx_auc(flat_y, probs, max_n=200_000)

    logits_masked = np.where(mask, logits, -1e9)
    order_full = np.argsort(-logits_masked, axis=1)
    n_pos_per_row = (labels * mask).sum(axis=1)
    has_pos = n_pos_per_row > 0

    metrics = {"acc": acc, "auc": auc,
               "pos_rate": float(flat_y.mean()),
               "rows_with_pos": int(has_pos.sum())}
    for K in recall_ks:
        top = order_full[:, :K]
        hits = np.take_along_axis(labels, top, axis=1).sum(axis=1)
        hit_any = (hits >= 1).astype(np.float32)
        # top-K hit: ≥1 positive captured in top K (out of all rows w/ ≥1 positive)
        metrics[f"top{K}"] = float(hit_any[has_pos].mean()) if has_pos.any() else 0.0
        # recall@K: fraction of true positives captured in top K, averaged over rows w/ ≥1 positive
        recall = np.where(has_pos, hits / np.maximum(n_pos_per_row, 1), 0.0)
        metrics[f"recall{K}"] = float(recall[has_pos].mean()) if has_pos.any() else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arch", choices=["deepsets", "transformer", "distbias", "graphsage", "gat"],
                    default="transformer")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--k", type=int, default=8, help="k for k-NN in graphsage/gat")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--max-step", type=int, default=None,
                    help="drop rows with step >= this (both train and val)")
    ap.add_argument("--max-ship-ratio", type=float, default=None,
                    help="drop rows where max(my,enemy)/min(my,enemy) ship-counts exceeds this "
                         "(detects one-sided games). 2.0 is a reasonable starting point.")
    ap.add_argument("--use-offensive-labels", action="store_true",
                    help="(legacy) shortcut for --label-key offensive_labels")
    ap.add_argument("--label-key", default="labels",
                    help="which key in the NPZ to use as the per-planet binary label "
                         "(e.g. labels, offensive_labels, support_target_labels, support_source_labels)")
    ap.add_argument("--opening-weight", choices=["off", "smooth", "linear", "step"], default="smooth")
    ap.add_argument("--select-by", default="recall8",
                    help="metric name to select the best checkpoint by (default recall8)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "weights" / "setnet_latest.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"loading {args.data}")
    data = load_npz(args.data)
    pf = data["planet_feats"]; gl = data["globals"]
    mk = data["mask"]; lb = data["labels"]; meta = data["meta"]

    if args.max_step is not None:
        keep = meta[:, 1] < args.max_step
        pf = pf[keep]; gl = gl[keep]; mk = mk[keep]; lb = lb[keep]; meta = meta[keep]
        print(f"  step filter <{args.max_step}: kept {keep.sum()}/{len(keep)} rows")
    if args.max_ship_ratio is not None:
        # globals[2]=my_ships_total, globals[3]=enemy_ships_total per row
        my_s = gl[:, 2]; en_s = gl[:, 3]
        hi = np.maximum(my_s, en_s); lo = np.maximum(np.minimum(my_s, en_s), 1.0)
        keep = (hi / lo) <= args.max_ship_ratio
        pf = pf[keep]; gl = gl[keep]; mk = mk[keep]; lb = lb[keep]; meta = meta[keep]
        print(f"  ship-ratio filter ≤{args.max_ship_ratio}: kept {keep.sum()}/{len(keep)} rows")
    label_key = "offensive_labels" if args.use_offensive_labels else args.label_key
    if label_key != "labels":
        z = np.load(args.data, allow_pickle=True)
        if label_key not in z.files:
            raise SystemExit(f"--label-key {label_key!r} not found in NPZ. "
                             f"available keys: {list(z.files)}")
        olb = z[label_key].astype(np.float32)
        full = load_npz(args.data)
        sel = np.ones(len(full["meta"]), dtype=bool)
        if args.max_step is not None:
            sel &= full["meta"][:, 1] < args.max_step
        if args.max_ship_ratio is not None:
            my_s = full["globals"][:, 2]; en_s = full["globals"][:, 3]
            hi = np.maximum(my_s, en_s); lo = np.maximum(np.minimum(my_s, en_s), 1.0)
            sel &= (hi / lo) <= args.max_ship_ratio
        lb = olb[sel].astype(np.float32)
        print(f"  label key: {label_key}  "
              f"positives: {int((lb * mk).sum())} ({(lb * mk).sum() / mk.sum() * 100:.2f}% of slots)")
    print(f"  shapes planet={pf.shape} globals={gl.shape}")

    train_mask, val_mask = game_level_split(meta, seed=args.seed, val_frac=args.val_frac)
    p_mean, p_std, g_mean, g_std = fit_norm(pf, gl, train_mask, mk)
    pf_n, gl_n = apply_norm(pf, gl, p_mean, p_std, g_mean, g_std)

    row_w_all = opening_weight(meta[:, 1], args.opening_weight).astype(np.float32)
    print(f"  opening weight schedule: {args.opening_weight}  "
          f"(min={row_w_all.min():.3f}  mean={row_w_all.mean():.3f}  max={row_w_all.max():.3f})")

    def to_ds(idx_mask):
        return TensorDataset(
            torch.from_numpy(pf_n[idx_mask]),
            torch.from_numpy(gl_n[idx_mask]),
            torch.from_numpy(mk[idx_mask]),
            torch.from_numpy(lb[idx_mask]),
            torch.from_numpy(row_w_all[idx_mask]),
        )

    tr_ds = to_ds(train_mask)
    va_ds = to_ds(val_mask)
    print(f"  train rows: {len(tr_ds)}  val rows: {len(va_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos = float(lb[train_mask][mk[train_mask]].sum())
    neg = float((1 - lb[train_mask])[mk[train_mask]].sum())
    pos_w = (neg / max(pos, 1.0)) if args.pos_weight is None else args.pos_weight
    print(f"  pos rate: {pos / (pos + neg):.4f}  pos_weight={pos_w:.2f}")

    device = torch.device(args.device)
    model = build_model(
        args.arch, pf.shape[-1], gl.shape[-1],
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        hidden=args.hidden, ff=args.ff, dropout=args.dropout, k=args.k,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model={args.arch} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_w_t = torch.tensor([pos_w], device=device)

    best_metric = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for pf_b, gl_b, mk_b, lb_b, w_b in tr_loader:
            pf_b = pf_b.to(device); gl_b = gl_b.to(device)
            mk_b = mk_b.to(device); lb_b = lb_b.to(device); w_b = w_b.to(device)
            logits = model(pf_b, gl_b, mk_b)
            loss = masked_weighted_bce(logits, lb_b, mk_b, w_b, pos_weight=pos_w_t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses))

        # val inference
        model.eval()
        logits_all = predict_all(model, pf_n[val_mask], gl_n[val_mask], mk[val_mask], device)
        m = eval_metrics(logits_all, lb[val_mask], mk[val_mask])
        dt = time.time() - t0
        sel = m[args.select_by]
        print(f"  epoch {epoch:2d} | tr_loss={tr_loss:.4f} | "
              f"acc={m['acc']:.4f} auc={m['auc']:.4f} "
              f"r1={m['recall1']:.3f} r3={m['recall3']:.3f} "
              f"r5={m['recall5']:.3f} r8={m['recall8']:.3f} r10={m['recall10']:.3f} "
              f"| pos%={m['pos_rate']*100:.2f}  ({dt:.1f}s)")
        if sel > best_metric:
            best_metric = sel
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "f_planet": pf.shape[-1],
                "f_global": gl.shape[-1],
                "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
                "hidden": args.hidden, "ff": args.ff, "dropout": args.dropout, "k": args.k,
                "p_mean": p_mean, "p_std": p_std,
                "g_mean": g_mean, "g_std": g_std,
                "feat_names": data["feat_names"],
                "global_names": data["global_names"],
                "val_metrics": m,
                "selected_by": args.select_by,
                "opening_weight_schedule": args.opening_weight,
                "max_step": args.max_step,
                "label_key": label_key,
                "max_ship_ratio": args.max_ship_ratio,
            }, args.out)
            print(f"    saved (best {args.select_by}={sel:.4f}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
