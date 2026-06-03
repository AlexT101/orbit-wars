"""Train the final 157-d XGBoost value net on the full dataset and save it
as a drop-in replacement for the production weights.

Uses a single game-grouped holdout purely for early stopping (so the round
count is chosen honestly), then reports that holdout's accuracy. The saved
booster is trained on train+the early-stop rounds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from cv_eval import load_features, game_folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n-est", type=int, default=2000)
    p.add_argument("--subsample", type=float, default=0.85)
    p.add_argument("--colsample", type=float, default=0.85)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--val-folds", type=int, default=8, help="hold out 1/val_folds of games for early stop")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backup", action="store_true", help="back up an existing --out before overwriting")
    args = p.parse_args()

    X, y, meta, names, is_strong = load_features(args.npz)
    yb = (y > 0).astype(np.float32)
    games = meta[:, 0].astype(np.int64)
    n_games = len(np.unique(games))
    folds = game_folds(games, args.val_folds, args.seed)
    va = folds == 0
    tr = ~va
    print(f"dataset: {X.shape[0]:,} rows  {X.shape[1]} feats  {n_games:,} games  val_rows={int(va.sum()):,}")

    params = dict(
        objective="binary:logistic",
        eval_metric=["error", "logloss"],
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        tree_method="hist",
        max_bin=args.max_bin,
        verbosity=0,
        nthread=0,
    )
    dtr = xgb.DMatrix(X[tr], label=yb[tr])
    dva = xgb.DMatrix(X[va], label=yb[va])
    t0 = time.time()
    bst = xgb.train(
        params, dtr, num_boost_round=args.n_est,
        evals=[(dva, "val")], early_stopping_rounds=80,
        verbose_eval=False,
    )
    pred = bst.predict(dva)
    acc = float(((pred > 0.5) == (yb[va] > 0.5)).mean())
    best_it = bst.best_iteration
    print(f"holdout acc={100*acc:.2f}%  best_iter={best_it}  ({time.time()-t0:.0f}s)")

    if args.out.exists() and args.backup:
        bak = args.out.with_suffix(args.out.suffix + ".bak")
        shutil.copy2(args.out, bak)
        print(f"backed up existing model -> {bak}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(args.out))
    meta_out = args.out.with_suffix(".meta.json")
    meta_out.write_text(json.dumps({
        "dataset": str(args.npz),
        "features": "summary_v2[46] + extras_v4[12] + engineered[88] + tempo[11]",
        "feature_dim": int(X.shape[1]),
        "objective": "binary",
        "accuracy": acc,
        "config": {
            "max_depth": args.max_depth, "learning_rate": args.lr, "n_est": int(best_it),
            "min_child_weight": args.min_child_weight, "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha, "gamma": args.gamma, "max_bin": args.max_bin,
        },
        "n_games": int(n_games),
        "n_rows": int(X.shape[0]),
        "seed": args.seed,
        "val_folds": args.val_folds,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)  meta -> {meta_out}")


if __name__ == "__main__":
    main()
