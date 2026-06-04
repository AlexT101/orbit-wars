"""Game-grouped CV comparison: base 157-d features vs 157 + spatial.

Trains both on identical folds so the delta is attributable to the spatial
columns alone.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cv_eval import load_features, run_cv
from spatial_features import SPATIAL_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--spatial", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n-est", type=int, default=1200)
    args = p.parse_args()

    X, y, meta, names, _ = load_features(args.npz)
    sp = np.load(args.spatial, allow_pickle=False)["spatial"].astype(np.float32)
    assert sp.shape[0] == X.shape[0], f"{sp.shape} vs {X.shape}"
    Xa = np.concatenate([X, sp], axis=1)
    print(f"base dim={X.shape[1]}  +spatial dim={Xa.shape[1]}  rows={X.shape[0]:,}  spatial cols={len(SPATIAL_NAMES)}")

    common = dict(subsample=0.85, colsample=0.85, min_child_weight=1.0,
                  reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, max_bin=256)

    for tag, Xuse in [("base157", X), ("base+spatial", Xa)]:
        t0 = time.time()
        acc, std, ll, it = run_cv(
            Xuse, y, meta, args.folds, args.seed, args.max_depth, args.lr, args.n_est,
            common["subsample"], common["colsample"], common["min_child_weight"],
            common["reg_lambda"], common["reg_alpha"], common["gamma"], common["max_bin"],
            verbose=False,
        )
        print(f"  {tag:14s} acc={100*acc:.3f}% +/- {100*std:.2f}  logloss={ll:.4f}  iter={it}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
