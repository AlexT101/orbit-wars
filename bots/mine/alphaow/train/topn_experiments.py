"""Top-N skill filter + retrain experiments.

Loads a combined NPZ with `game_names`, computes per-player win rates,
and for each top-N threshold trains MLP + XGBoost on (a) all data and
(b) games where both players are in the top-N. Reports val sign-accuracy
so we can see whether tightening the skill band reduces label noise.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
# Import xgboost FIRST (before torch) so the libomp linked by xgboost wins
# the initialization race. Torch is imported lazily inside train_mlp.
import xgboost  # noqa: F401


def split_mask(meta, seed=42, frac=0.12):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games])


def train_mlp(X, y, val_mask, hidden=64, epochs=40, bs=2048, lr=2e-3, wd=5e-4, seed=42, verbose=False):
    import torch
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden=64):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)
        def forward(self, x):
            h = torch.relu(self.fc1(x))
            return torch.tanh(self.fc2(h)).squeeze(-1)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    Xtr, ytr = Xt[~val_mask], yt[~val_mask]
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp(min=1e-3)
    model = MLP(X.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss()
    n_train = Xtr.shape[0]
    best_sign = 0.0
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
        if verbose and (ep + 1) % 20 == 0:
            print(f"    ep {ep+1}/{epochs} sign={sign:.3f} best={best_sign:.3f}")
    return best_sign


def train_xgb(X, y, val_mask, n_est=600, depth=6, lr=0.08):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--extras", default=None)
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--top-n", nargs="+", type=int, default=[0, 100, 50, 20, 10])
    p.add_argument("--skip-mlp", action="store_true")
    p.add_argument("--skip-xgb", action="store_true")
    args = p.parse_args()

    print(f"loading {args.data} ...")
    d = np.load(args.data, allow_pickle=False)
    X = d["summary_v2"].astype(np.float32)
    if args.extras:
        e = np.load(args.extras)["extras"].astype(np.float32)
        assert e.shape[0] == X.shape[0]
        X = np.concatenate([X, e], axis=1)
        print(f"  with extras: X={X.shape}")
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    game_names = d["game_names"]   # (n_games, 2) of <U64
    n_games = game_names.shape[0]
    print(f"X={X.shape}  n_games={n_games}  labels={np.unique(y).tolist()}")

    # per-player stats
    pg = defaultdict(int); pw = defaultdict(int)
    for gid in range(n_games):
        n0 = str(game_names[gid, 0]); n1 = str(game_names[gid, 1])
        pg[n0] += 1; pg[n1] += 1
        # need rewards — we have labels per row; first row of each game's slot=0 is +1 if p0 won
        # easier: scan meta. but cheaper: use first occurrence.
    # actually compute wins by scanning meta + labels: meta col0=gid, col2=slot. label = +1 if that slot won.
    # for each (gid, slot=0), grab label.
    p0_mask = (meta[:, 2] == 0)
    # one row per game-tick at slot 0; take the first row per gid to get reward[0]
    # since all rows of same gid+slot have same reward, just pick uniques.
    seen = set(); r0_by_gid = {}
    for i in range(0, meta.shape[0], 2):  # every other row is slot 0 (matches build order)
        gid = int(meta[i, 0])
        if gid in seen: continue
        seen.add(gid); r0_by_gid[gid] = float(y[i])
    for gid in range(n_games):
        r0 = r0_by_gid.get(gid, 0.0)
        n0 = str(game_names[gid, 0]); n1 = str(game_names[gid, 1])
        if r0 > 0: pw[n0] += 1
        elif r0 < 0: pw[n1] += 1
    rates = {pl: pw[pl]/pg[pl] for pl in pg if pg[pl] >= args.min_games}
    print(f"\nplayers with >= {args.min_games} games: {len(rates)}")
    if rates:
        sorted_rates = sorted(rates.items(), key=lambda kv: -kv[1])
        print("  top 15:")
        for pl, r in sorted_rates[:15]:
            print(f"    {r:.3f} ({pw[pl]:>4}/{pg[pl]:<4})  {pl[:50]}")
        print("  bottom 5:")
        for pl, r in sorted_rates[-5:]:
            print(f"    {r:.3f} ({pw[pl]:>4}/{pg[pl]:<4})  {pl[:50]}")

    val_mask = split_mask(meta, seed=42)
    print(f"\ngame-level val split: rows train={(~val_mask).sum()} val={val_mask.sum()}\n")

    for n_top in args.top_n:
        if n_top <= 0:
            sub_mask = np.ones(meta.shape[0], dtype=bool)
            tag = "all"
        else:
            sorted_rates = sorted(rates.items(), key=lambda kv: -kv[1])
            top_set = {pl for pl, _ in sorted_rates[:n_top]}
            game_in_top = np.array([
                (str(game_names[g, 0]) in top_set and str(game_names[g, 1]) in top_set)
                for g in range(n_games)
            ])
            sub_mask = game_in_top[meta[:, 0].astype(np.int64)]
            tag = f"top{n_top}"
        ng_kept = int(np.unique(meta[sub_mask, 0]).size) if sub_mask.any() else 0
        nrows = int(sub_mask.sum())
        if ng_kept < 8:
            print(f"=== {tag}: only {ng_kept} games / {nrows} rows -> skip ===\n")
            continue
        print(f"=== {tag}: {ng_kept} games, {nrows} rows ===")
        Xs, ys, ms = X[sub_mask], y[sub_mask], meta[sub_mask]
        vm = split_mask(ms, seed=42)
        if vm.sum() == 0 or (~vm).sum() == 0:
            print("  empty train or val split  skip\n"); continue
        if not args.skip_mlp:
            t0 = time.time()
            sign = train_mlp(Xs, ys, vm, hidden=64, epochs=80)
            print(f"  MLP h64        val sign-acc = {100*sign:.2f}%   ({time.time()-t0:.0f}s)")
        if not args.skip_xgb:
            t0 = time.time()
            sign, it = train_xgb(Xs, ys, vm)
            print(f"  XGBoost d6     val sign-acc = {100*sign:.2f}%   (best_iter={it}, {time.time()-t0:.0f}s)")
        print()


if __name__ == "__main__":
    main()
