"""Push XGBoost further on 46d + extras (62d) -- bigger trees, more rounds,
different reg, log-loss vs hinge. Reports val sign-acc per config to find
the offline ceiling."""

from __future__ import annotations

import argparse
import time
import numpy as np
import xgboost as xgb


def split_mask(meta, seed=42, frac=0.12):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed); rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games])


def run(X, y, vm, **params):
    yb_tr = (y[~vm] > 0).astype(np.float32)
    yb_va = (y[vm] > 0).astype(np.float32)
    dtr = xgb.DMatrix(X[~vm], label=yb_tr)
    dva = xgb.DMatrix(X[vm], label=yb_va)
    n_est = params.pop("n_est", 2000)
    base = dict(objective="binary:logistic", eval_metric="logloss",
                subsample=0.85, colsample_bytree=0.85, tree_method="hist",
                verbosity=0)
    base.update(params)
    t0 = time.time()
    bst = xgb.train(base, dtr, num_boost_round=n_est,
                    evals=[(dva, "val")], early_stopping_rounds=60,
                    verbose_eval=False)
    pred = bst.predict(dva)
    acc = ((pred > 0.5) == (yb_va > 0.5)).mean()
    return acc, bst.best_iteration, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--extras", default=None)
    args = p.parse_args()
    d = np.load(args.data, allow_pickle=False)
    X = d["summary_v2"].astype(np.float32)
    if args.extras:
        e = np.load(args.extras)["extras"].astype(np.float32)
        X = np.concatenate([X, e], axis=1)
        print(f"with extras: X={X.shape}")
    else:
        print(f"baseline: X={X.shape}")
    y = d["labels"].astype(np.float32); meta = d["meta"]
    vm = split_mask(meta)
    print(f"train={(~vm).sum()} val={vm.sum()}", flush=True)

    grid = [
        dict(max_depth=6,  learning_rate=0.10, n_est=600),
        dict(max_depth=6,  learning_rate=0.05, n_est=1500),
        dict(max_depth=8,  learning_rate=0.05, n_est=2000),
        dict(max_depth=10, learning_rate=0.05, n_est=2000),
        dict(max_depth=8,  learning_rate=0.03, n_est=3000),
        dict(max_depth=8,  learning_rate=0.05, n_est=2000, min_child_weight=5, gamma=0.1),
        dict(max_depth=8,  learning_rate=0.05, n_est=2000, reg_alpha=1.0, reg_lambda=2.0),
        dict(max_depth=6,  learning_rate=0.05, n_est=2000, max_bin=512),
    ]
    best = (0.0, "")
    for cfg in grid:
        acc, it, sec = run(X, y, vm, **dict(cfg))
        tag = "  ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"  {acc*100:.2f}%  iter={it:4d}  {sec:5.0f}s  | {tag}", flush=True)
        if acc > best[0]:
            best = (acc, tag)
    print(f"\nBEST: {best[0]*100:.2f}% | {best[1]}")


if __name__ == "__main__":
    main()
