"""XGBoost + LightGBM baselines on summary_v2.

Same game-level 12% val split as the MLP trainer (seed=42) so the numbers
are directly comparable to the 85.7% MLP/85.04% linear ceilings.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def split_mask(meta, seed=42, frac=0.12):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games]), len(unique), n_val


def run_xgb(X, y, val_mask, n_est=600, depth=6, lr=0.08, subsample=0.85, colsample=0.85):
    import xgboost as xgb
    dtr = xgb.DMatrix(X[~val_mask], label=y[~val_mask])
    dva = xgb.DMatrix(X[val_mask], label=y[val_mask])
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=depth, learning_rate=lr,
        subsample=subsample, colsample_bytree=colsample,
        tree_method="hist",
        verbosity=0,
    )
    yb_tr = (y[~val_mask] > 0).astype(np.float32)
    yb_va = (y[val_mask] > 0).astype(np.float32)
    dtr.set_label(yb_tr)
    dva.set_label(yb_va)
    t0 = time.time()
    bst = xgb.train(
        params, dtr, num_boost_round=n_est,
        evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=False,
    )
    pred = bst.predict(dva)
    sign = ((pred > 0.5) == (yb_va > 0.5)).mean()
    return sign, time.time() - t0, bst.best_iteration

def run_lgbm(X, y, val_mask, n_est=800, leaves=63, lr=0.05):
    import lightgbm as lgb
    yb_tr = (y[~val_mask] > 0).astype(np.float32)
    yb_va = (y[val_mask] > 0).astype(np.float32)
    dtr = lgb.Dataset(X[~val_mask], label=yb_tr)
    dva = lgb.Dataset(X[val_mask], label=yb_va, reference=dtr)
    params = dict(
        objective="binary", metric="binary_logloss",
        num_leaves=leaves, learning_rate=lr,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
        min_data_in_leaf=100, verbose=-1,
    )
    t0 = time.time()
    model = lgb.train(
        params, dtr, num_boost_round=n_est,
        valid_sets=[dva], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )
    pred = model.predict(X[val_mask])
    sign = ((pred > 0.5) == (yb_va > 0.5)).mean()
    return sign, time.time() - t0, model.best_iteration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--filter-strong", action="store_true")
    args = p.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    if args.filter_strong and "is_strong" in d.files:
        m = d["is_strong"].astype(bool)
        X, y, meta = X[m], y[m], meta[m]
        print(f"strong filter: kept {m.sum()} / {m.shape[0]} rows")
    print(f"X={X.shape}  labels unique={np.unique(y).tolist()}")
    val_mask, ng, nv = split_mask(meta)
    print(f"games={ng} val={nv}  rows train={(~val_mask).sum()} val={val_mask.sum()}")
    print()

    print("=== XGBoost (depth=6, lr=0.08, n_est=600) ===")
    s, sec, it = run_xgb(X, y, val_mask)
    print(f"  val sign-acc = {100*s:.2f}%   best_iter={it}   ({sec:.1f}s)")

    print("\n=== XGBoost (depth=8, lr=0.05, n_est=1000) ===")
    s, sec, it = run_xgb(X, y, val_mask, n_est=1000, depth=8, lr=0.05)
    print(f"  val sign-acc = {100*s:.2f}%   best_iter={it}   ({sec:.1f}s)")

    print("\n=== LightGBM (leaves=63, lr=0.05, n_est=800) ===")
    s, sec, it = run_lgbm(X, y, val_mask)
    print(f"  val sign-acc = {100*s:.2f}%   best_iter={it}   ({sec:.1f}s)")

    print("\n=== LightGBM (leaves=127, lr=0.03, n_est=1500) ===")
    s, sec, it = run_lgbm(X, y, val_mask, n_est=1500, leaves=127, lr=0.03)
    print(f"  val sign-acc = {100*s:.2f}%   best_iter={it}   ({sec:.1f}s)")


if __name__ == "__main__":
    main()
