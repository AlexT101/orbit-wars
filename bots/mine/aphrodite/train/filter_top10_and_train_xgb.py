"""Take a `build_from_zip.py` output NPZ, apply the top-10 player filter,
and train XGBoost with the same params the deployed `xgb_top10_d6.json`
used (binary:logistic, max_depth=6, lr=0.08, n_est=600, early stop=40).
Saves the booster as `weights/<out_stem>.json`.

This collapses filtering and XGBoost training into one script for the
"rebuild + retrain" workflow.

Usage:
    python filter_top10_and_train_xgb.py \\
        --input data/fixed/combined_fixed.npz \\
        --top10-out data/fixed/combined_top10_fixed.npz \\
        --model-out weights/xgb_top10_d6_fixed.json
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_top_n(d, n_top=10, min_games=5):
    """Compute per-player win rates and return the set of top-N player names."""
    meta = d["meta"]
    y = d["labels"]
    game_names = d["game_names"]
    n_games = game_names.shape[0]

    # First row of each game-slot has the same reward; just dedupe by gid.
    pg = defaultdict(int)
    pw = defaultdict(int)
    for gid in range(n_games):
        n0 = str(game_names[gid, 0])
        n1 = str(game_names[gid, 1])
        pg[n0] += 1
        pg[n1] += 1

    # Recover reward[0] per game from labels (slot 0's label).
    r0_by_gid = {}
    seen = set()
    for i in range(0, meta.shape[0], 2):  # every other row is slot 0 in build order
        gid = int(meta[i, 0])
        if gid in seen:
            continue
        seen.add(gid)
        r0_by_gid[gid] = float(y[i])

    for gid in range(n_games):
        r0 = r0_by_gid.get(gid, 0.0)
        n0 = str(game_names[gid, 0])
        n1 = str(game_names[gid, 1])
        if r0 > 0:
            pw[n0] += 1
        elif r0 < 0:
            pw[n1] += 1

    rates = {pl: pw[pl] / pg[pl] for pl in pg if pg[pl] >= min_games}
    sorted_rates = sorted(rates.items(), key=lambda kv: -kv[1])
    print(f"players with >= {min_games} games: {len(rates)}")
    print(f"top 15:")
    for pl, r in sorted_rates[:15]:
        print(f"  {r:.3f} ({pw[pl]:>4}/{pg[pl]:<4})  {pl[:60]}")
    return {pl for pl, _ in sorted_rates[:n_top]}


def filter_top_n(d, top_set, out_path: Path):
    meta = d["meta"]
    game_names = d["game_names"]
    n_games = game_names.shape[0]
    game_in_top = np.array([
        (str(game_names[g, 0]) in top_set and str(game_names[g, 1]) in top_set)
        for g in range(n_games)
    ])
    sub = game_in_top[meta[:, 0].astype(np.int64)]
    Xs = d["summary_v2"][sub]
    ys = d["labels"][sub]
    ms = meta[sub]
    n_kept = int(np.unique(ms[:, 0]).size)
    print(f"  top-N filter kept {n_kept} games / {len(Xs):,} rows / "
          f"{len(Xs) * 196 / 1e6:.1f} MB raw")
    np.savez_compressed(
        out_path,
        summary_v2=Xs.astype(np.float32),
        labels=ys.astype(np.float32),
        meta=ms.astype(np.int32),
        game_names=game_names,
        game_files=d["game_files"],
    )
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return Xs, ys, ms


def game_level_split_mask(meta, frac=0.12, seed=42):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games])


def train_xgb(X, y, val_mask, out_json: Path,
              max_depth=6, learning_rate=0.08, n_est=600):
    import xgboost as xgb
    yb = (y > 0).astype(np.float32)
    dtr = xgb.DMatrix(X[~val_mask], label=yb[~val_mask])
    dva = xgb.DMatrix(X[val_mask], label=yb[val_mask])
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        verbosity=0,
    )
    t0 = time.time()
    bst = xgb.train(
        params, dtr, num_boost_round=n_est,
        evals=[(dva, "val")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )
    pred = bst.predict(dva)
    sign_acc = float(((pred > 0.5) == (yb[val_mask] > 0.5)).mean())
    elapsed = time.time() - t0
    print(f"  XGB val sign-acc = {100*sign_acc:.3f}%  "
          f"best_iter={bst.best_iteration}  t={elapsed:.1f}s")
    bst.save_model(str(out_json))
    print(f"  saved {out_json} ({out_json.stat().st_size / 1e6:.2f} MB)")
    return sign_acc, bst.best_iteration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="combined NPZ from build_from_zip.py")
    p.add_argument("--top10-out", required=True, type=Path)
    p.add_argument("--model-out", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-games", type=int, default=5)
    args = p.parse_args()

    print(f"Loading {args.input}...")
    d = np.load(args.input, allow_pickle=False)
    n_games = d["game_names"].shape[0]
    n_rows = d["summary_v2"].shape[0]
    print(f"  {n_games} games / {n_rows:,} rows")

    print(f"\n=== STEP 1: top-{args.top_n} filter (min {args.min_games} games) ===")
    top_set = compute_top_n(d, n_top=args.top_n, min_games=args.min_games)
    Xs, ys, ms = filter_top_n(d, top_set, args.top10_out)

    print(f"\n=== STEP 2: train XGB (binary:logistic d=6 lr=0.08 n_est=600) ===")
    val_mask = game_level_split_mask(ms, frac=0.12, seed=42)
    n_train_games = len(np.unique(ms[~val_mask, 0]))
    n_val_games = len(np.unique(ms[val_mask, 0]))
    print(f"  split: train games={n_train_games}, val games={n_val_games}, "
          f"train rows={(~val_mask).sum():,}, val rows={val_mask.sum():,}")
    train_xgb(Xs, ys, val_mask, args.model_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
