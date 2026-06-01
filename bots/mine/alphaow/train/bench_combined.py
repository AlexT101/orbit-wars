"""Benchmark different feature sets and models on the combined NPZ.

Compares:
  * baseline 46-d summary_v2
  * 46-d + 16 extras (62-d)
  * extras alone (sanity)

Models tested per feature set:
  * MLP h64 (80 epochs, seed=42)
  * XGBoost (depth 6, lr 0.08, n_est 600, early stop)
  * MLP h128 dropout=0.2 (more capacity)
  * Ensemble of 5 MLPs (avg of probabilities)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
# torch imported lazily inside train_mlp to avoid OMP conflict with xgboost


def split_mask(meta, seed=42, frac=0.12):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games])


def train_mlp(X, y, val_mask, hidden=64, epochs=40, bs=2048, lr=2e-3, wd=5e-4, seed=42, dropout=0.0, return_pred=False, max_train=2_000_000):
    import torch
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden=64, dropout=0.0):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc2 = nn.Linear(hidden, 1)
        def forward(self, x):
            h = torch.relu(self.fc1(x))
            h = self.dp(h)
            return torch.tanh(self.fc2(h)).squeeze(-1)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    Xtr_full, ytr_full = Xt[~val_mask], yt[~val_mask]
    if max_train and Xtr_full.shape[0] > max_train:
        rng = torch.Generator(device=dev).manual_seed(seed)
        idx = torch.randperm(Xtr_full.shape[0], generator=rng, device=dev)[:max_train]
        Xtr, ytr = Xtr_full[idx], ytr_full[idx]
    else:
        Xtr, ytr = Xtr_full, ytr_full
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp(min=1e-3)
    model = MLP(X.shape[1], hidden, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss()
    n_train = Xtr.shape[0]
    best_sign = 0.0; best_pred = None
    for ep in range(epochs):
        idx = torch.randperm(n_train, device=dev)
        model.train()
        for j in range(0, n_train, bs):
            sel = idx[j:j+bs]
            xb = (Xtr[sel] - mean) / std
            pred = model(xb)
            loss = loss_fn(pred, ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = model((Xv - mean) / std)
            sign = ((v > 0) == (yv > 0)).float().mean().item()
        if sign > best_sign:
            best_sign = sign
            best_pred = v.cpu().numpy().copy()
    return (best_sign, best_pred) if return_pred else best_sign


def train_xgb(X, y, val_mask, n_est=800, depth=6, lr=0.08):
    import xgboost as xgb
    yb_tr = (y[~val_mask] > 0).astype(np.float32)
    yb_va = (y[val_mask] > 0).astype(np.float32)
    dtr = xgb.DMatrix(X[~val_mask], label=yb_tr)
    dva = xgb.DMatrix(X[val_mask], label=yb_va)
    bst = xgb.train(
        dict(objective="binary:logistic", eval_metric="logloss",
             max_depth=depth, learning_rate=lr,
             subsample=0.85, colsample_bytree=0.85, tree_method="hist",
             verbosity=0),
        dtr, num_boost_round=n_est,
        evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=False,
    )
    pred = bst.predict(dva)
    return ((pred > 0.5) == (yb_va > 0.5)).mean(), bst.best_iteration


def bench(name, X, y, val_mask, do_ensemble=True):
    print(f"\n========== {name}  (in_dim={X.shape[1]}, rows={X.shape[0]}) ==========", flush=True)
    yv_pos = (y[val_mask] > 0)
    results = {}
    # XGB first (fast + best)
    t0 = time.time(); s, it = train_xgb(X, y, val_mask)
    print(f"  XGBoost d=6 lr=0.08      val sign-acc = {100*s:.2f}%  (best_iter={it}, {time.time()-t0:.0f}s)", flush=True)
    results["xgb_d6"] = s
    t0 = time.time(); s, it = train_xgb(X, y, val_mask, depth=8, lr=0.05, n_est=1500)
    print(f"  XGBoost d=8 lr=0.05      val sign-acc = {100*s:.2f}%  (best_iter={it}, {time.time()-t0:.0f}s)", flush=True)
    results["xgb_d8"] = s
    # MLP (sub-sampled for speed; results indicative)
    t0 = time.time(); s = train_mlp(X, y, val_mask)
    print(f"  MLP h64  (2M sub)        val sign-acc = {100*s:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    results["mlp_h64"] = s
    if do_ensemble:
        t0 = time.time()
        preds = []
        for seed in [42, 7, 123, 2026, 999]:
            _, p = train_mlp(X, y, val_mask, seed=seed, return_pred=True)
            preds.append(p)
        avg = np.mean(preds, axis=0)
        s = ((avg > 0) == yv_pos).mean()
        print(f"  MLP h64 ensemble (5x)    val sign-acc = {100*s:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
        results["mlp_ens"] = s
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--extras", default=None, help="optional NPZ with extras array aligned to meta")
    p.add_argument("--skip-extras-only", action="store_true")
    p.add_argument("--no-ensemble", action="store_true")
    args = p.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    val_mask = split_mask(meta)
    print(f"loaded {args.data}  X={X.shape}  rows train={(~val_mask).sum()} val={val_mask.sum()}")

    bench("baseline 46-d", X, y, val_mask, do_ensemble=not args.no_ensemble)

    if args.extras:
        e = np.load(args.extras)["extras"].astype(np.float32)
        assert e.shape[0] == X.shape[0], f"extras rows {e.shape[0]} != X rows {X.shape[0]}"
        Xe = np.concatenate([X, e], axis=1)
        bench(f"46-d + extras ({e.shape[1]}d)", Xe, y, val_mask, do_ensemble=not args.no_ensemble)
        if not args.skip_extras_only:
            bench(f"extras only ({e.shape[1]}d)", e, y, val_mask, do_ensemble=False)


if __name__ == "__main__":
    main()
