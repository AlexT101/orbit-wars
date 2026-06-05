"""Train an Aphrodite SummaryV2 XGBoost value model from an NPZ.

This is the simple train-only path for already-selected or mixed datasets. For
top-N replay filtering, use ``filter_top10_and_train_xgb.py``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def game_level_split_mask(meta: np.ndarray, frac: float, seed: int) -> np.ndarray:
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([int(g) in val_set for g in games])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--model-out", required=True, type=Path)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--rounds", type=int, default=600)
    p.add_argument("--early-stopping", type=int, default=40)
    p.add_argument(
        "--zero-cols",
        type=str,
        default="",
        help=(
            "Comma-separated SummaryV2 column indices to neutralize (set to a "
            "constant 0) before training. A constant column has zero split gain, "
            "so XGBoost ignores it — equivalent to dropping the feature while "
            "keeping the model 65-d so the Rust runtime loads it unchanged. "
            "E.g. '41,61,62,63,64' drops step + the 4p-standing block."
        ),
    )
    args = p.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"].astype(np.int32)
    if args.zero_cols:
        cols = [int(c) for c in args.zero_cols.split(",") if c.strip() != ""]
        X = X.copy()
        X[:, cols] = 0.0
        print(f"zeroed columns {cols} (treated as dropped; model stays {X.shape[1]}-d)")
    yb = (y > 0).astype(np.float32)
    val_mask = game_level_split_mask(meta, args.val_frac, args.seed)

    import xgboost as xgb

    dtr = xgb.DMatrix(X[~val_mask], label=yb[~val_mask])
    dva = xgb.DMatrix(X[val_mask], label=yb[val_mask])
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        verbosity=0,
    )

    print(
        f"data={args.data} rows={X.shape[0]:,} games={len(np.unique(meta[:, 0])):,} "
        f"train={(~val_mask).sum():,} val={val_mask.sum():,}"
    )
    t0 = time.time()
    bst = xgb.train(
        params,
        dtr,
        num_boost_round=args.rounds,
        evals=[(dva, "val")],
        early_stopping_rounds=args.early_stopping,
        verbose_eval=False,
    )
    pred = bst.predict(dva)
    sign_acc = float(((pred > 0.5) == (yb[val_mask] > 0.5)).mean())
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(args.model_out))
    print(
        f"saved {args.model_out} sign_acc={100 * sign_acc:.3f}% "
        f"best_iter={bst.best_iteration} t={time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
