"""Game-grouped cross-validation harness for the XGBoost value function.

The pipeline's built-in split holds out only ~12% of games (2 of 16), which
gives a very noisy accuracy estimate. This harness does K-fold CV grouped by
game so every game is held out exactly once, and reports mean +/- std win-sign
accuracy. Validation is always game-grouped (no row leakage between folds).

Usage:
  python3 cv_eval.py --npz data/pipeline/combined_46p12e88t11.npz
  python3 cv_eval.py --npz <big>.npz --folds 5 --max-depth 6 --lr 0.08 --n-est 900
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from engineered_features import append_tempo_features


def load_features(npz_path: Path, with_tempo: bool = True):
    d = np.load(npz_path, allow_pickle=False)
    y = d["labels"].astype(np.float32)
    meta = d["meta"].astype(np.int32)
    if "features" in d.files:
        X = d["features"].astype(np.float32)
        # `features` in combined npz already includes tempo; nothing to add.
    else:
        # base-only npz (summary+extras): build engineered+tempo on the fly.
        from engineered_features import append_engineered_features

        base = d["base_features"].astype(np.float32)
        X = append_tempo_features(append_engineered_features(base), meta)
    names = d["feature_names"].astype(str).tolist() if "feature_names" in d.files else [f"f{i}" for i in range(X.shape[1])]
    is_strong = d["is_strong"].astype(bool) if "is_strong" in d.files else np.ones(len(y), bool)
    return X, y, meta, names, is_strong


def game_folds(games: np.ndarray, k: int, seed: int):
    uniq = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq.tolist())}
    return np.array([fold_of[g] for g in games])


def run_cv(X, y, meta, k, seed, max_depth, lr, n_est, subsample, colsample, min_child_weight, reg_lambda, reg_alpha, gamma, max_bin, verbose=True):
    games = meta[:, 0].astype(np.int64)
    folds = game_folds(games, k, seed)
    yb = (y > 0).astype(np.float32)
    accs, lls = [], []
    best_iters = []
    for f in range(k):
        va = folds == f
        tr = ~va
        params = dict(
            objective="binary:logistic",
            eval_metric=["error", "logloss"],
            max_depth=max_depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            gamma=gamma,
            tree_method="hist",
            max_bin=max_bin,
            verbosity=0,
            nthread=0,
        )
        dtr = xgb.DMatrix(X[tr], label=yb[tr])
        dva = xgb.DMatrix(X[va], label=yb[va])
        evals_result: dict = {}
        bst = xgb.train(
            params, dtr, num_boost_round=n_est,
            evals=[(dva, "val")], early_stopping_rounds=60,
            verbose_eval=False, evals_result=evals_result,
        )
        pred = bst.predict(dva)
        acc = float(((pred > 0.5) == (yb[va] > 0.5)).mean())
        eps = 1e-7
        ll = float(-np.mean(yb[va] * np.log(pred + eps) + (1 - yb[va]) * np.log(1 - pred + eps)))
        accs.append(acc)
        lls.append(ll)
        best_iters.append(bst.best_iteration)
        if verbose:
            print(f"  fold {f}: acc={100*acc:.2f}% logloss={ll:.4f} iter={bst.best_iteration} (val_rows={va.sum()})")
    accs = np.array(accs)
    lls = np.array(lls)
    if verbose:
        print(f"CV mean acc={100*accs.mean():.2f}% +/- {100*accs.std():.2f}  logloss={lls.mean():.4f}  median_iter={int(np.median(best_iters))}")
    return accs.mean(), accs.std(), lls.mean(), int(np.median(best_iters))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--n-est", type=int, default=900)
    p.add_argument("--subsample", type=float, default=0.85)
    p.add_argument("--colsample", type=float, default=0.85)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--filter-strong", action="store_true")
    args = p.parse_args()

    X, y, meta, names, is_strong = load_features(args.npz)
    if args.filter_strong:
        X, y, meta = X[is_strong], y[is_strong], meta[is_strong]
    n_games = len(np.unique(meta[:, 0]))
    print(f"dataset: {X.shape[0]:,} rows  {X.shape[1]} feats  {n_games:,} games  pos_rate={float((y>0).mean()):.3f}")
    t0 = time.time()
    run_cv(
        X, y, meta, args.folds, args.seed, args.max_depth, args.lr, args.n_est,
        args.subsample, args.colsample, args.min_child_weight, args.reg_lambda,
        args.reg_alpha, args.gamma, args.max_bin,
    )
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
