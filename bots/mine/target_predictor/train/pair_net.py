"""Pair-prediction trainer: predicts P(player launches a fleet from planet i to
planet j) for every (i, j) pair in one transformer forward pass.

Loss: BCE on the dense N×N pair-label matrix, masked to (real_i & real_j & i!=j).

By default the bot wants well-calibrated per-pair probabilities (it samples
each pair with `rand < prob`), so this trainer defaults to:
  - --pos-weight 1.0  (no class re-weighting; keeps probabilities calibrated)
  - --select-by pair_logloss  (lowest val log-loss = best calibration)
You can flip back to ranking-style selection with --select-by pair_recall8.

Usage:
  python3 train/pair_net.py --data train/data/targets_2k_v5.npz \\
      --epochs 25 --device mps --batch-size 512 \\
      --out train/weights/transformer_pair.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit("torch not importable; pip install torch") from exc

from set_net import (
    apply_norm, fit_norm, game_level_split, opening_weight, load_npz,
)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


F_PAIR = 15  # distance (1)
             # + 3 ETAs with n=src.ships+prod·t at h=0/1/2  (near-term decision)
             # + 4 ETAs with n=20 at h=5/10/20/30           (longer-range reach)
             # + 3 argmin bits at h=0/1/2 (on the n=src series)
             # + cos/sin of board angle src→tgt
             # + cos/sin of orbital-phase difference Δθ
PAIR_HORIZONS_ALL = (0, 1, 2, 5, 10, 20, 30)
SRC_HORIZONS = (0, 1, 2)         # indices into raw_xy
N20_HORIZONS = (5, 10, 20, 30)
SRC_H_IDXS = [PAIR_HORIZONS_ALL.index(h) for h in SRC_HORIZONS]
N20_H_IDXS = [PAIR_HORIZONS_ALL.index(h) for h in N20_HORIZONS]


def _fleet_speed_torch(n_ships: torch.Tensor) -> torch.Tensor:
    """Mirrors build_dataset.fleet_speed but as a vectorized torch op."""
    n = n_ships.clamp_min(1.0)
    speed = 1.0 + 5.0 * (torch.log(n) / math.log(1000.0)).clamp_min(0.0).pow(1.5)
    return speed.clamp(1.0, 6.0)


def compute_pair_features(raw_xy: torch.Tensor,
                          raw_ships: torch.Tensor,
                          raw_prod: torch.Tensor) -> torch.Tensor:
    """Returns (B, N, N, F_PAIR=16) pair-feature tensor.

    raw_xy: (B, N, 5, 2) positions at horizons (0, 1, 10, 20, 30).
    raw_ships, raw_prod: (B, N)

    Features (per ordered (i=src, j=tgt) pair):
      0: distance now (i at t0 → j at t0)
      1–5: ETA with n = src.ships + t·src.prod at horizons (0,1,10,20,30)
      6–10: ETA with n = 20 at the same horizons
      11: argmin_eta_now (1 if j is closest target from i right now)
      12–13: cos/sin of board direction src→tgt
      14–15: cos/sin of orbital phase difference Δθ = tgt.θ − src.θ
    """
    B, N, H, _ = raw_xy.shape
    assert H == len(PAIR_HORIZONS_ALL), f"expected raw_xy with H={len(PAIR_HORIZONS_ALL)} horizons, got {H}"

    xy_at = [raw_xy[:, :, k, :] for k in range(H)]   # list of (B, N, 2)
    xy_now = xy_at[0]

    # Distance NOW (src@0 → tgt@0)
    dist_t0 = torch.cdist(xy_now, xy_now)

    # ETA at horizon h: both src AND tgt have moved to their t=h positions,
    # AND the launcher's ship count has grown by h·prod (fleet_speed is higher).
    # Approximation: ignores further motion during the eta-length flight.
    eta_src_list = []
    for h, k in zip(SRC_HORIZONS, SRC_H_IDXS):
        n_at_h = raw_ships + h * raw_prod
        spd = _fleet_speed_torch(n_at_h).unsqueeze(2).clamp_min(1e-3)
        d = torch.cdist(xy_at[k], xy_at[k])
        eta_src_list.append(d / spd)
    # ETAs with constant n=20 at each horizon (target+src moved)
    speed_n20 = _fleet_speed_torch(torch.full_like(raw_ships, 20.0)).unsqueeze(2).clamp_min(1e-3)
    eta_n20_list = []
    for k in N20_H_IDXS:
        d = torch.cdist(xy_at[k], xy_at[k])
        eta_n20_list.append(d / speed_n20)

    # argmin bits per source row, one per n=src horizon (h=0/1/2)
    eye = torch.eye(N, device=raw_xy.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    argmin_bits = []
    for eta in eta_src_list:
        eta_masked = eta.masked_fill(eye, float("inf"))
        idx = eta_masked.argmin(dim=2, keepdim=True)
        bit = torch.zeros_like(eta)
        bit.scatter_(2, idx, 1.0)
        argmin_bits.append(bit)

    # Board direction src→tgt as (cos, sin); zero on diagonal.
    dx = xy_now[:, None, :, 0] - xy_now[:, :, None, 0]
    dy = xy_now[:, None, :, 1] - xy_now[:, :, None, 1]
    safe_dist = dist_t0.clamp_min(1e-3)
    cos_ang = (dx / safe_dist).masked_fill(eye, 0.0)
    sin_ang = (dy / safe_dist).masked_fill(eye, 0.0)

    # Orbital phase difference Δθ
    sx = xy_now[:, :, 0] - 50.0
    sy = xy_now[:, :, 1] - 50.0
    r = (sx * sx + sy * sy).sqrt().clamp_min(0.1)
    cs = sx / r
    sn = sy / r
    cs_src = cs.unsqueeze(2); sn_src = sn.unsqueeze(2)
    cs_tgt = cs.unsqueeze(1); sn_tgt = sn.unsqueeze(1)
    cos_delta = (cs_tgt * cs_src + sn_tgt * sn_src).masked_fill(eye, 0.0)
    sin_delta = (sn_tgt * cs_src - cs_tgt * sn_src).masked_fill(eye, 0.0)

    return torch.stack([
        dist_t0,
        *eta_src_list,         # 3
        *eta_n20_list,         # 4
        *argmin_bits,          # 3
        cos_ang, sin_ang,      # 2
        cos_delta, sin_delta,  # 2
    ], dim=-1)  # (B, N, N, 15)


class PlanetTransformerPair(nn.Module):
    """Encoder + bilinear pair head + value head + noop head.

    Pair logits:  score[i,j] = (h_i W_q) · (h_j W_k)^T / sqrt(D) + pair_bias[i,j]
                  where pair_bias = linear(pair_features[i,j])
    Value:        scalar in [-1, 1] via tanh on a pooled embedding
    Noop:         per-source scalar logit (sigmoid → P(source does not launch))
    """

    def __init__(self, f_planet, f_global, d_model=64, n_heads=4, n_layers=2,
                 ff=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.planet_proj = nn.Linear(f_planet, d_model)
        self.global_proj = nn.Linear(f_global, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.bias = nn.Parameter(torch.zeros(1))
        # Pair-feature head: bias added to bilinear scores.
        # Initialize weights to zero so the head starts neutral; otherwise the
        # raw distance/ETA features (scale ~10–100) trample the bilinear scores
        # (scale ~1) at random init and the model never recovers.
        self.pair_feat_proj = nn.Linear(F_PAIR, 1)
        nn.init.zeros_(self.pair_feat_proj.weight)
        nn.init.zeros_(self.pair_feat_proj.bias)
        # Value head: pooled (CLS + mean + max) planet embeddings -> tanh scalar.
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # Noop head: per-source scalar logit.
        self.noop_head = nn.Linear(d_model, 1)

    def forward(self, planet_feats, globals_, mask,
                raw_xy=None, raw_ships=None, raw_prod=None,
                return_value=True, return_noop=True):
        B, N, _ = planet_feats.shape
        p_tok = self.planet_proj(planet_feats)
        g_tok = self.global_proj(globals_).unsqueeze(1)
        tokens = torch.cat([g_tok, p_tok], dim=1)
        kpm = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
            ~mask,
        ], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=kpm)
        cls_h = out[:, 0, :]                            # (B, D)
        h = out[:, 1:, :]                               # (B, N, D)
        q = self.q_proj(h); k = self.k_proj(h)
        pair_logits = torch.bmm(q, k.transpose(-2, -1)) / (self.d_model ** 0.5) + self.bias
        if raw_xy is not None:
            pair_feats = compute_pair_features(raw_xy, raw_ships, raw_prod)  # (B, N, N, F_PAIR)
            pair_logits = pair_logits + self.pair_feat_proj(pair_feats).squeeze(-1)
        result = (pair_logits,)
        if return_value:
            mask_f = mask.unsqueeze(-1).float()
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            mean_pool = (h * mask_f).sum(dim=1) / denom
            h_for_max = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)
            max_pool = h_for_max.max(dim=1).values
            value = torch.tanh(self.value_head(torch.cat([cls_h, mean_pool, max_pool], dim=-1))).squeeze(-1)
            result = result + (value,)
        if return_noop:
            noop_logits = self.noop_head(h).squeeze(-1)  # (B, N)
            result = result + (noop_logits,)
        if len(result) == 1:
            return result[0]
        return result


# ---------------------------------------------------------------------------
# Sparse pair labels -> dense per-batch
# ---------------------------------------------------------------------------


def dense_pair_labels(pair_src: torch.Tensor, pair_tgt: torch.Tensor, N: int):
    """sparse (B, K) int indices (with -1 = pad) -> dense (B, N, N) float."""
    B, K = pair_src.shape
    out = torch.zeros(B, N, N, device=pair_src.device, dtype=torch.float32)
    valid = pair_src >= 0
    if not valid.any():
        return out
    b_idx = torch.arange(B, device=pair_src.device).unsqueeze(1).expand(-1, K)
    b = b_idx[valid]
    s = pair_src[valid].long()
    t = pair_tgt[valid].long()
    out[b, s, t] = 1.0
    return out


def pair_mask(planet_mask: torch.Tensor) -> torch.Tensor:
    """(B, N) -> (B, N, N) bool: real_i & real_j & i!=j."""
    B, N = planet_mask.shape
    m2d = planet_mask.unsqueeze(2) & planet_mask.unsqueeze(1)
    eye = torch.eye(N, device=planet_mask.device, dtype=torch.bool).unsqueeze(0)
    return m2d & ~eye


def masked_pair_bce(logits, labels, mask, row_w, pos_weight=None):
    """BCE over dense N×N pair matrix, masked, with per-row weight."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=pos_weight,
    )
    m = mask.float()
    bce = bce * m * row_w.view(-1, 1, 1)
    denom = (m * row_w.view(-1, 1, 1)).sum().clamp_min(1.0)
    return bce.sum() / denom


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_pair_all(model, pf, gl, mk, device,
                     raw_xy=None, raw_ships=None, raw_prod=None, batch=256):
    """Returns (N_rows, N, N) numpy pair logits."""
    out = []
    for i in range(0, len(pf), batch):
        kwargs = dict(return_value=False, return_noop=False)
        if raw_xy is not None:
            kwargs["raw_xy"] = torch.from_numpy(raw_xy[i:i+batch]).to(device)
            kwargs["raw_ships"] = torch.from_numpy(raw_ships[i:i+batch]).to(device)
            kwargs["raw_prod"] = torch.from_numpy(raw_prod[i:i+batch]).to(device)
        logits = model(
            torch.from_numpy(pf[i:i+batch]).to(device),
            torch.from_numpy(gl[i:i+batch]).to(device),
            torch.from_numpy(mk[i:i+batch]).to(device),
            **kwargs,
        )
        out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_value_all(model, pf, gl, mk, device,
                     raw_xy=None, raw_ships=None, raw_prod=None, batch=256):
    """Returns (N_rows,) numpy value predictions in [-1, 1]."""
    out = []
    for i in range(0, len(pf), batch):
        kwargs = dict(return_value=True, return_noop=False)
        if raw_xy is not None:
            kwargs["raw_xy"] = torch.from_numpy(raw_xy[i:i+batch]).to(device)
            kwargs["raw_ships"] = torch.from_numpy(raw_ships[i:i+batch]).to(device)
            kwargs["raw_prod"] = torch.from_numpy(raw_prod[i:i+batch]).to(device)
        _, v = model(
            torch.from_numpy(pf[i:i+batch]).to(device),
            torch.from_numpy(gl[i:i+batch]).to(device),
            torch.from_numpy(mk[i:i+batch]).to(device),
            **kwargs,
        )
        out.append(v.cpu().numpy())
    return np.concatenate(out, axis=0)


def eval_pair_metrics(logits, pair_src, pair_tgt, mask, ks=(1, 3, 5, 8, 10)):
    """Per-row pair metrics. logits: (R, N, N), pair_*: (R, K), mask: (R, N)."""
    R, N, _ = logits.shape
    # apply pair mask -> set illegal pairs to -inf so they never enter top-K
    pmask = (mask[:, :, None] & mask[:, None, :])
    eye = np.eye(N, dtype=bool)[None]
    pmask = pmask & ~eye
    logits_masked = np.where(pmask, logits, -1e9)
    # log-loss + BCE-style acc (computed on masked pair slots)
    # dense labels for metrics
    dense = np.zeros((R, N, N), dtype=np.float32)
    n_pairs_per_row = np.zeros(R, dtype=np.int32)
    for r in range(R):
        for k in range(pair_src.shape[1]):
            s = pair_src[r, k]; t = pair_tgt[r, k]
            if s < 0: break
            dense[r, s, t] = 1.0
            n_pairs_per_row[r] += 1
    flat_l = logits_masked[pmask]
    flat_y = dense[pmask]
    probs = 1.0 / (1.0 + np.exp(-np.clip(flat_l, -30, 30)))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logloss = float(-(flat_y * np.log(probs) + (1 - flat_y) * np.log(1 - probs)).mean())
    acc = float(((probs >= 0.5) == (flat_y >= 0.5)).mean())
    pos_rate = float(flat_y.mean())

    # top-K and recall@K over rows with ≥1 launch
    has_pair = n_pairs_per_row > 0
    metrics = {"pair_acc": acc, "pair_logloss": logloss,
               "pair_pos_rate": pos_rate, "rows_with_pair": int(has_pair.sum()),
               "mean_pairs_per_row": float(n_pairs_per_row[has_pair].mean()) if has_pair.any() else 0.0}
    if not has_pair.any():
        for k in ks:
            metrics[f"pair_top{k}"] = 0.0
            metrics[f"pair_recall{k}"] = 0.0
        return metrics

    flat_logits_per_row = logits_masked.reshape(R, N * N)
    # only consider rows with at least 1 pair for top-K stats
    sub_logits = flat_logits_per_row[has_pair]
    sub_dense = dense.reshape(R, N * N)[has_pair]
    sub_npairs = n_pairs_per_row[has_pair]
    order = np.argsort(-sub_logits, axis=1)
    for K in ks:
        topk = order[:, :K]
        hits = np.take_along_axis(sub_dense, topk, axis=1).sum(axis=1)
        metrics[f"pair_top{K}"] = float((hits >= 1).mean())
        metrics[f"pair_recall{K}"] = float((hits / np.maximum(sub_npairs, 1)).mean())
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="positive class weight for BCE. 1.0 keeps probs calibrated; "
                         "set higher if you want rank-style training.")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--max-step", type=int, default=200)
    ap.add_argument("--max-ship-ratio", type=float, default=None)
    ap.add_argument("--opening-weight", choices=["off", "smooth", "linear", "step"],
                    default="smooth")
    ap.add_argument("--value-loss-weight", type=float, default=0.5,
                    help="multiplier on the value-head MSE loss in multi-task training. "
                         "0.0 = disable value training entirely.")
    ap.add_argument("--noop-loss-weight", type=float, default=0.1,
                    help="multiplier on the per-source noop BCE in multi-task training.")
    ap.add_argument("--select-by", default="pair_logloss",
                    help="metric to pick best epoch by. Lower is better for *logloss; "
                         "higher is better for *recall* / *top*. "
                         "Common choices: pair_logloss, pair_recall8, pair_top1.")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "weights" / "transformer_pair.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"loading {args.data}")
    data = load_npz(args.data)
    pf = data["planet_feats"]; gl = data["globals"]
    mk = data["mask"]; meta = data["meta"]
    z = np.load(args.data, allow_pickle=True)
    if "pair_source_idx" not in z.files:
        raise SystemExit("NPZ lacks pair_source_idx; rebuild with the v5+ build_dataset.py")
    pair_src = z["pair_source_idx"].astype(np.int64)
    pair_tgt = z["pair_target_idx"].astype(np.int64)
    value_labels = z["value_labels"].astype(np.float32) if "value_labels" in z.files else None
    raw_xy = z["raw_xy"].astype(np.float32) if "raw_xy" in z.files else None
    raw_ships = z["raw_ships"].astype(np.float32) if "raw_ships" in z.files else None
    raw_prod = z["raw_prod"].astype(np.float32) if "raw_prod" in z.files else None
    noop_labels = z["noop_labels"].astype(np.float32) if "noop_labels" in z.files else None
    if value_labels is None:
        print("  warning: NPZ has no value_labels — value head will not be trained")
    if raw_xy is None:
        print("  warning: NPZ has no raw_xy — pair-feature head will be inactive")
    if noop_labels is None:
        print("  warning: NPZ has no noop_labels — noop head will not be trained")

    # Cache the small metadata fields before we drop the big arrays.
    _feat_names_cache = data.get("feat_names")
    _global_names_cache = data.get("global_names")
    def _apply_keep(keep):
        nonlocal pf, gl, mk, meta, pair_src, pair_tgt, value_labels, raw_xy, raw_ships, raw_prod, noop_labels
        pf = pf[keep]; gl = gl[keep]; mk = mk[keep]; meta = meta[keep]
        pair_src = pair_src[keep]; pair_tgt = pair_tgt[keep]
        if value_labels is not None: value_labels = value_labels[keep]
        if raw_xy is not None: raw_xy = raw_xy[keep]
        if raw_ships is not None: raw_ships = raw_ships[keep]
        if raw_prod is not None: raw_prod = raw_prod[keep]
        if noop_labels is not None: noop_labels = noop_labels[keep]

    if args.max_step is not None:
        keep = meta[:, 1] < args.max_step
        _apply_keep(keep)
        print(f"  step filter <{args.max_step}: kept {keep.sum()} rows")
        data = None
        del z
        import gc; gc.collect()
    if args.max_ship_ratio is not None:
        my_s = gl[:, 2]; en_s = gl[:, 3]
        hi = np.maximum(my_s, en_s); lo = np.maximum(np.minimum(my_s, en_s), 1.0)
        keep = (hi / lo) <= args.max_ship_ratio
        _apply_keep(keep)
        print(f"  ship-ratio filter ≤{args.max_ship_ratio}: kept {keep.sum()} rows")

    train_mask, val_mask = game_level_split(meta, seed=args.seed, val_frac=args.val_frac)
    p_mean, p_std, g_mean, g_std = fit_norm(pf, gl, train_mask, mk)
    pf_n, gl_n = apply_norm(pf, gl, p_mean, p_std, g_mean, g_std)
    row_w_all = opening_weight(meta[:, 1], args.opening_weight).astype(np.float32)
    print(f"  opening weight: {args.opening_weight}  mean={row_w_all.mean():.3f}")

    def to_ds(idx_mask):
        n = int(idx_mask.sum())
        vlabels = value_labels[idx_mask] if value_labels is not None else np.zeros(n, dtype=np.float32)
        rxy = raw_xy[idx_mask] if raw_xy is not None else np.zeros((n, mk.shape[1], 3, 2), dtype=np.float32)
        rsh = raw_ships[idx_mask] if raw_ships is not None else np.zeros((n, mk.shape[1]), dtype=np.float32)
        rpd = raw_prod[idx_mask] if raw_prod is not None else np.zeros((n, mk.shape[1]), dtype=np.float32)
        nlb = noop_labels[idx_mask] if noop_labels is not None else np.zeros((n, mk.shape[1]), dtype=np.float32)
        return TensorDataset(
            torch.from_numpy(pf_n[idx_mask]),
            torch.from_numpy(gl_n[idx_mask]),
            torch.from_numpy(mk[idx_mask]),
            torch.from_numpy(pair_src[idx_mask]),
            torch.from_numpy(pair_tgt[idx_mask]),
            torch.from_numpy(row_w_all[idx_mask]),
            torch.from_numpy(vlabels),
            torch.from_numpy(rxy),
            torch.from_numpy(rsh),
            torch.from_numpy(rpd),
            torch.from_numpy(nlb),
        )

    tr_ds = to_ds(train_mask)
    print(f"  train rows: {len(tr_ds)}  val rows: {int(val_mask.sum())}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device(args.device)
    model = PlanetTransformerPair(
        pf.shape[-1], gl.shape[-1],
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ff=args.ff, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: PlanetTransformerPair  params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_w_t = torch.tensor([args.pos_weight], device=device)
    N = pf.shape[1]

    # selection direction
    higher_better = not args.select_by.endswith("logloss")
    best = -math.inf if higher_better else math.inf

    VALUE_LOSS_W = args.value_loss_weight if value_labels is not None else 0.0
    NOOP_LOSS_W = args.noop_loss_weight if noop_labels is not None else 0.0
    use_pair_feats = raw_xy is not None
    print(f"  loss weights: value={VALUE_LOSS_W}  noop={NOOP_LOSS_W}  "
          f"pair-feat-head={'on' if use_pair_feats else 'off'}")

    # Used to mask noop loss to my planets only (label is set to 0 for non-mine, but
    # mining real planets is the natural restriction). Use planet-feats column for
    # is_mine (index 0 of PLANET_FEAT_NAMES is "is_mine"). For pure noop labels we
    # multiply by (is_mine after un-normalization). Easiest: derive from noop_labels
    # being >0 only on my planets at build time — so just use mk as the mask and let
    # the noop_labels distinguish 0/1.

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses_p = []; losses_v = []; losses_n = []
        for pf_b, gl_b, mk_b, ps_b, pt_b, w_b, v_b, rxy_b, rsh_b, rpd_b, nlb_b in tr_loader:
            pf_b = pf_b.to(device); gl_b = gl_b.to(device); mk_b = mk_b.to(device)
            ps_b = ps_b.to(device); pt_b = pt_b.to(device); w_b = w_b.to(device); v_b = v_b.to(device)
            rxy_b = rxy_b.to(device); rsh_b = rsh_b.to(device); rpd_b = rpd_b.to(device); nlb_b = nlb_b.to(device)
            pair_logits, value_pred, noop_logits = model(
                pf_b, gl_b, mk_b,
                raw_xy=rxy_b if use_pair_feats else None,
                raw_ships=rsh_b if use_pair_feats else None,
                raw_prod=rpd_b if use_pair_feats else None,
                return_value=True, return_noop=True,
            )
            dense = dense_pair_labels(ps_b, pt_b, N)
            pm = pair_mask(mk_b)
            loss_p = masked_pair_bce(pair_logits, dense, pm, w_b, pos_weight=pos_w_t)
            loss_v = nn.functional.mse_loss(value_pred, v_b) if VALUE_LOSS_W > 0 else torch.tensor(0.0, device=device)
            if NOOP_LOSS_W > 0:
                noop_bce = nn.functional.binary_cross_entropy_with_logits(noop_logits, nlb_b, reduction="none")
                # Mask noop loss to MY planets only. After fit_norm the is_mine
                # column (planet_feats index 0) has two distinct values; the
                # > 0 threshold cleanly separates raw is_mine == 1 from == 0.
                is_mine_mask = pf_b[..., 0] > 0
                noop_mask = mk_b & is_mine_mask
                loss_n = (noop_bce * noop_mask.float()).sum() / noop_mask.float().sum().clamp_min(1.0)
            else:
                loss_n = torch.tensor(0.0, device=device)
            loss = loss_p + VALUE_LOSS_W * loss_v + NOOP_LOSS_W * loss_n
            opt.zero_grad(); loss.backward(); opt.step()
            losses_p.append(loss_p.item())
            losses_v.append(loss_v.item() if VALUE_LOSS_W > 0 else 0.0)
            losses_n.append(loss_n.item() if NOOP_LOSS_W > 0 else 0.0)
        tr_loss_p = float(np.mean(losses_p))
        tr_loss_v = float(np.mean(losses_v))
        tr_loss_n = float(np.mean(losses_n))

        # val
        model.eval()
        val_kwargs = {}
        if use_pair_feats:
            val_kwargs.update(raw_xy=raw_xy[val_mask], raw_ships=raw_ships[val_mask], raw_prod=raw_prod[val_mask])
        logits_all = predict_pair_all(model, pf_n[val_mask], gl_n[val_mask], mk[val_mask], device, **val_kwargs)
        m = eval_pair_metrics(logits_all, pair_src[val_mask], pair_tgt[val_mask], mk[val_mask])
        if value_labels is not None:
            v_pred = predict_value_all(model, pf_n[val_mask], gl_n[val_mask], mk[val_mask], device, **val_kwargs)
            v_true = value_labels[val_mask]
            m["value_mse"] = float(((v_pred - v_true) ** 2).mean())
            m["value_signacc"] = float(((np.sign(v_pred) == np.sign(v_true)) & (v_true != 0)).sum() / max((v_true != 0).sum(), 1))
        dt = time.time() - t0
        sel = m[args.select_by]
        v_suffix = (f"  v_mse={m.get('value_mse', 0.0):.3f}  v_signacc={m.get('value_signacc', 0.0):.3f}"
                    if value_labels is not None else "")
        print(f"  epoch {epoch:2d} | tr_p={tr_loss_p:.4f} tr_v={tr_loss_v:.4f} tr_n={tr_loss_n:.4f} | "
              f"acc={m['pair_acc']:.4f} ll={m['pair_logloss']:.4f} "
              f"r1={m['pair_recall1']:.3f} r3={m['pair_recall3']:.3f} "
              f"r5={m['pair_recall5']:.3f} r8={m['pair_recall8']:.3f} "
              f"r10={m['pair_recall10']:.3f}{v_suffix}  ({dt:.1f}s)")

        improved = sel > best if higher_better else sel < best
        if improved:
            best = sel
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "arch": "pair",
                "f_planet": pf.shape[-1],
                "f_global": gl.shape[-1],
                "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
                "ff": args.ff, "dropout": args.dropout,
                "p_mean": p_mean, "p_std": p_std,
                "g_mean": g_mean, "g_std": g_std,
                "feat_names": _feat_names_cache,
                "global_names": _global_names_cache,
                "val_metrics": m,
                "selected_by": args.select_by,
                "pos_weight": args.pos_weight,
                "opening_weight_schedule": args.opening_weight,
                "max_step": args.max_step,
                "max_ship_ratio": args.max_ship_ratio,
            }, args.out)
            print(f"    saved (best {args.select_by}={sel:.4f}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
