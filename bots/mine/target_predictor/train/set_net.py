"""Set-encoder for per-planet target prediction.

Two backbones:
- DeepSets:    cheap floor. Per-planet MLP → mean+max pool → re-broadcast → head.
- Transformer: primary. 2 layers, multi-head attention across planet tokens.

Both consume the NPZ produced by build_dataset.py:
  planet_feats[N, N_max, F_planet]
  globals[N, F_global]
  mask[N, N_max]
  labels[N, N_max]

Loss: masked BCE-with-logits. Eval: per-planet AUC + accuracy + top-1 accuracy
(did the highest-logit real planet match a true positive?).

Game-level split (same pattern as prometheus): hash on `meta[:, 0]` (game id),
seed=42, val_frac=0.12.

Usage:
  python3 set_net.py --data data/targets.npz --arch transformer --epochs 8
  python3 set_net.py --data data/targets.npz --arch deepsets --epochs 8
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
    raise SystemExit(
        "torch not importable. Try: python3 -m pip install torch\n"
        f"Import error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    return dict(
        planet_feats=z["planet_feats"].astype(np.float32),
        globals=z["globals"].astype(np.float32),
        mask=z["mask"].astype(np.bool_),
        labels=z["labels"].astype(np.float32),
        meta=z["meta"].astype(np.int64),
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


# Per-feature normalization stats are fit on training rows only and applied to
# both splits. Saved alongside the model so inference matches.
def fit_norm(planet_feats, globals_, train_mask, real_mask):
    """Compute (mean, std) over real planet slots for planet_feats, and over
    rows for globals."""
    train_planets = planet_feats[train_mask]              # (N_tr, N_max, F)
    train_real = real_mask[train_mask]                    # (N_tr, N_max)
    flat = train_planets[train_real]                      # (R, F)
    p_mean = flat.mean(axis=0)
    p_std = flat.std(axis=0) + 1e-6
    g_mean = globals_[train_mask].mean(axis=0)
    g_std = globals_[train_mask].std(axis=0) + 1e-6
    return p_mean.astype(np.float32), p_std.astype(np.float32), \
           g_mean.astype(np.float32), g_std.astype(np.float32)


def apply_norm(planet_feats, globals_, p_mean, p_std, g_mean, g_std):
    pf = (planet_feats - p_mean) / p_std
    gl = (globals_ - g_mean) / g_std
    return pf.astype(np.float32), gl.astype(np.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DeepSets(nn.Module):
    """Per-planet MLP → mean+max pool → broadcast back → head. Includes globals.

    No attention; treats planets as exchangeable. Cheap floor.
    """

    def __init__(self, f_planet, f_global, hidden=64):
        super().__init__()
        self.planet_enc = nn.Sequential(
            nn.Linear(f_planet, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.global_enc = nn.Sequential(
            nn.Linear(f_global, hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, planet_feats, globals_, mask):
        # planet_feats: (B, N, F_p); globals: (B, F_g); mask: (B, N) bool
        h = self.planet_enc(planet_feats)                       # (B, N, H)
        mask_f = mask.unsqueeze(-1).float()
        h_masked = h * mask_f
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = h_masked.sum(dim=1) / denom                  # (B, H)
        neg_inf = torch.finfo(h.dtype).min
        h_for_max = h.masked_fill(~mask.unsqueeze(-1), neg_inf)
        max_pool = h_for_max.max(dim=1).values                   # (B, H)
        g = self.global_enc(globals_)                            # (B, H)
        context = torch.cat([mean_pool, max_pool, g], dim=-1)    # (B, 3H)
        context_b = context.unsqueeze(1).expand(-1, h.size(1), -1)  # (B, N, 3H)
        full = torch.cat([h, context_b], dim=-1)                 # (B, N, 4H)
        return self.head(full).squeeze(-1)                       # (B, N)


class PlanetTransformer(nn.Module):
    """Token-per-planet + 1 CLS token carrying globals. 2 layers self-attention."""

    def __init__(self, f_planet, f_global, d_model=64, n_heads=4, n_layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.planet_proj = nn.Linear(f_planet, d_model)
        self.global_proj = nn.Linear(f_global, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, planet_feats, globals_, mask):
        B, N, _ = planet_feats.shape
        p_tok = self.planet_proj(planet_feats)                   # (B, N, D)
        g_tok = self.global_proj(globals_).unsqueeze(1)          # (B, 1, D)
        tokens = torch.cat([g_tok, p_tok], dim=1)                # (B, N+1, D)
        # nn.Transformer expects key-padding-mask True = ignore
        kpm = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
            ~mask,
        ], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=kpm)     # (B, N+1, D)
        planet_out = out[:, 1:, :]                                # (B, N, D)
        return self.head(planet_out).squeeze(-1)                  # (B, N)


def build_model(arch: str, f_planet: int, f_global: int,
                d_model: int = 64, n_heads: int = 4, n_layers: int = 2, hidden: int = 64,
                ff: int = 128, dropout: float = 0.1):
    if arch == "deepsets":
        return DeepSets(f_planet, f_global, hidden=hidden)
    if arch == "transformer":
        return PlanetTransformer(f_planet, f_global, d_model=d_model, n_heads=n_heads,
                                 n_layers=n_layers, ff=ff, dropout=dropout)
    raise ValueError(f"unknown arch: {arch}")


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------


def masked_bce(logits, labels, mask, pos_weight=None):
    # logits: (B, N), labels: (B, N), mask: (B, N) bool
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=pos_weight,
    )
    bce = bce * mask.float()
    return bce.sum() / mask.float().sum().clamp_min(1.0)


def eval_split(model, loader, device):
    model.eval()
    all_logits, all_labels, all_mask = [], [], []
    with torch.no_grad():
        for pf, gl, mk, lb in loader:
            pf = pf.to(device); gl = gl.to(device); mk = mk.to(device); lb = lb.to(device)
            logits = model(pf, gl, mk)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(lb.cpu().numpy())
            all_mask.append(mk.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    mask = np.concatenate(all_mask, axis=0)

    flat_l = logits[mask]; flat_y = labels[mask]
    probs = 1.0 / (1.0 + np.exp(-flat_l))
    pred = (probs >= 0.5).astype(np.float32)
    acc = float((pred == flat_y).mean())
    pos_rate = float(flat_y.mean())
    # naive AUC (Mann-Whitney) on a sample for speed
    auc = _approx_auc(flat_y, probs, max_n=200_000)

    # top-1 / top-2 accuracy: among real planets in a row, did the top-k logit
    # land on at least one positive?
    top1_hits = 0; top2_hits = 0; rows_with_pos = 0
    for i in range(logits.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        y = labels[i][m]
        if y.sum() == 0:
            continue
        l = logits[i][m]
        order = np.argsort(-l)
        if y[order[0]] == 1:
            top1_hits += 1
        if y[order[:2]].sum() >= 1:
            top2_hits += 1
        rows_with_pos += 1
    top1 = top1_hits / max(rows_with_pos, 1)
    top2 = top2_hits / max(rows_with_pos, 1)
    return dict(acc=acc, auc=auc, pos_rate=pos_rate,
                top1=top1, top2=top2, rows_with_pos=rows_with_pos, n_slots=int(mask.sum()))


def _approx_auc(y, p, max_n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    if len(y) > max_n:
        idx = rng.choice(len(y), size=max_n, replace=False)
        y = y[idx]; p = p[idx]
    pos = p[y > 0.5]; neg = p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_pos_sum = ranks[: len(pos)].sum()
    return float((rank_pos_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arch", choices=["deepsets", "transformer"], default="transformer")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="weight for positive class in BCE (default: auto = neg/pos)")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64, help="DeepSets hidden width")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "weights" / "setnet_latest.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"loading {args.data}")
    data = load_npz(args.data)
    pf = data["planet_feats"]; gl = data["globals"]
    mk = data["mask"]; lb = data["labels"]; meta = data["meta"]
    print(f"  shapes  planet={pf.shape}  globals={gl.shape}  mask={mk.shape}  labels={lb.shape}")
    print(f"  unique games: {len(np.unique(meta[:, 0]))}  rows: {pf.shape[0]}")

    train_mask, val_mask = game_level_split(meta, seed=args.seed, val_frac=args.val_frac)
    p_mean, p_std, g_mean, g_std = fit_norm(pf, gl, train_mask, mk)
    pf_n, gl_n = apply_norm(pf, gl, p_mean, p_std, g_mean, g_std)

    def to_ds(idx_mask):
        return TensorDataset(
            torch.from_numpy(pf_n[idx_mask]),
            torch.from_numpy(gl_n[idx_mask]),
            torch.from_numpy(mk[idx_mask]),
            torch.from_numpy(lb[idx_mask]),
        )

    tr_ds = to_ds(train_mask)
    va_ds = to_ds(val_mask)
    print(f"  train rows: {len(tr_ds)}  val rows: {len(va_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos = float(lb[train_mask][mk[train_mask]].sum())
    neg = float((1 - lb[train_mask])[mk[train_mask]].sum())
    pos_w = (neg / max(pos, 1.0)) if args.pos_weight is None else args.pos_weight
    print(f"  pos rate (train, masked): {pos / (pos + neg):.4f}  pos_weight={pos_w:.2f}")

    device = torch.device(args.device)
    model = build_model(
        args.arch, pf.shape[-1], gl.shape[-1],
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        hidden=args.hidden, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model={args.arch} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_w_t = torch.tensor([pos_w], device=device)

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for pf_b, gl_b, mk_b, lb_b in tr_loader:
            pf_b = pf_b.to(device); gl_b = gl_b.to(device)
            mk_b = mk_b.to(device); lb_b = lb_b.to(device)
            logits = model(pf_b, gl_b, mk_b)
            loss = masked_bce(logits, lb_b, mk_b, pos_weight=pos_w_t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses))
        m = eval_split(model, va_loader, device)
        dt = time.time() - t0
        print(f"  epoch {epoch:2d} | tr_loss={tr_loss:.4f} | "
              f"val acc={m['acc']:.4f} auc={m['auc']:.4f} "
              f"top1={m['top1']:.4f} top2={m['top2']:.4f} | "
              f"pos%={m['pos_rate'] * 100:.2f}  rows_w_pos={m['rows_with_pos']}  "
              f"({dt:.1f}s)")
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "f_planet": pf.shape[-1],
                "f_global": gl.shape[-1],
                "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
                "hidden": args.hidden, "dropout": args.dropout,
                "p_mean": p_mean, "p_std": p_std,
                "g_mean": g_mean, "g_std": g_std,
                "feat_names": data["feat_names"],
                "global_names": data["global_names"],
                "val_metrics": m,
            }, args.out)
            print(f"    saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
