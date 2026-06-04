"""Run several XGBoost configs through game-grouped K-fold CV, loading the
dataset once. Reports mean +/- std win-sign accuracy and logloss per config.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cv_eval import load_features, run_cv

# (name, dict-of-params)
GRID = [
    ("d6_lr08", dict(max_depth=6, lr=0.08, n_est=1200)),
    ("d8_lr05", dict(max_depth=8, lr=0.05, n_est=1200)),
    ("d8_lr03", dict(max_depth=8, lr=0.03, n_est=1600)),
    ("d10_lr05", dict(max_depth=10, lr=0.05, n_est=1200)),
    ("d10_lr03_mcw20", dict(max_depth=10, lr=0.03, n_est=1600, min_child_weight=20)),
    ("d8_lr05_mcw10_l2", dict(max_depth=8, lr=0.05, n_est=1200, min_child_weight=10, reg_lambda=3.0)),
    ("d12_lr03_mcw30", dict(max_depth=12, lr=0.03, n_est=1600, min_child_weight=30, reg_lambda=3.0)),
]

DEFAULTS = dict(
    max_depth=8, lr=0.05, n_est=1200, subsample=0.85, colsample=0.85,
    min_child_weight=1.0, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, max_bin=256,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filter-strong", action="store_true")
    args = p.parse_args()

    X, y, meta, names, is_strong = load_features(args.npz)
    if args.filter_strong:
        X, y, meta = X[is_strong], y[is_strong], meta[is_strong]
    n_games = len(np.unique(meta[:, 0]))
    print(f"dataset: {X.shape[0]:,} rows  {X.shape[1]} feats  {n_games:,} games  strong_filter={args.filter_strong}")

    results = []
    for name, override in GRID:
        cfg = dict(DEFAULTS)
        cfg.update(override)
        t0 = time.time()
        acc, std, ll, it = run_cv(
            X, y, meta, args.folds, args.seed,
            cfg["max_depth"], cfg["lr"], cfg["n_est"], cfg["subsample"], cfg["colsample"],
            cfg["min_child_weight"], cfg["reg_lambda"], cfg["reg_alpha"], cfg["gamma"], cfg["max_bin"],
            verbose=False,
        )
        dt = time.time() - t0
        print(f"  {name:22s} acc={100*acc:.2f}% +/- {100*std:.2f}  logloss={ll:.4f}  iter={it}  ({dt:.0f}s)", flush=True)
        results.append((acc, ll, name))

    results.sort(reverse=True)
    print("\n=== ranked by accuracy ===")
    for acc, ll, name in results:
        print(f"  {100*acc:.2f}%  ll={ll:.4f}  {name}")


if __name__ == "__main__":
    main()
