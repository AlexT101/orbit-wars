"""Experiment: replace the 9-d ext block with the (ext - cur_matching) delta
on both me and opp sides, leaving everything else (cur, neut) untouched.
Train XGB with the same params as xgb_top10_d6_fixed, compare val accuracy.

If delta features improve val acc by >=0.3pp on the fixed top-10 NPZ, port
the change into Rust `summary_features_v2_delta` and rebuild NPZ + model.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xgboost as xgb


# Layout of summary_v2 (46-d):
#   [0:10]  me_cur:  ships_on_planets, ships_flying, n_static, n_orbit, n_comet,
#                    prod_static, prod_orbit, prod_comet, n_neutrals_closer, n_enemies_closer
#   [10:20] opp_cur: same 10 fields, for the dominant enemy
#   [20:29] me_ext:  9 fields (omits ships_flying since extrap resolves all in-flight)
#                    ships_on_planets, n_static, n_orbit, n_comet,
#                    prod_static, prod_orbit, prod_comet, n_neutrals_closer, n_enemies_closer
#   [29:38] opp_ext: same 9 fields
#   [38:46] neutral: 8 fields, owner-agnostic
#
# ext block omits ships_flying — so ext[0] corresponds to cur[0] but
# the subsequent indices shift by 1 (skipping cur[1]=ships_flying).
EXT_TO_CUR_IDX = [0, 2, 3, 4, 5, 6, 7, 8, 9]


def to_delta(X: np.ndarray) -> np.ndarray:
    """Transform raw summary_v2 (N, 46) → (N, 46) with ext block replaced
    by ext - cur_matching for both me (cols 20:29) and opp (cols 29:38).
    """
    out = X.copy()
    me_cur_match = X[:, EXT_TO_CUR_IDX]            # (N, 9)
    opp_cur_match = X[:, [10 + i for i in EXT_TO_CUR_IDX]]  # (N, 9)
    out[:, 20:29] = X[:, 20:29] - me_cur_match
    out[:, 29:38] = X[:, 29:38] - opp_cur_match
    return out


def game_level_split(meta, frac=0.12, seed=42):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games])


def train_xgb(X, y, val_mask, label):
    yb = (y > 0).astype(np.float32)
    dtr = xgb.DMatrix(X[~val_mask], label=yb[~val_mask])
    dva = xgb.DMatrix(X[val_mask], label=yb[val_mask])
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        verbosity=0,
    )
    t0 = time.time()
    bst = xgb.train(
        params, dtr, num_boost_round=600,
        evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=False,
    )
    pred = bst.predict(dva)
    acc = float(((pred > 0.5) == (yb[val_mask] > 0.5)).mean())
    print(f"  [{label}]  XGB val sign-acc = {acc:.4f}  best_iter={bst.best_iteration}  t={time.time()-t0:.1f}s")
    return acc, bst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/fixed/combined_top10_fixed.npz", type=Path)
    p.add_argument("--save-delta-model", type=Path, default=None,
                   help="if set, save the delta-trained XGB to this path")
    args = p.parse_args()

    print(f"Loading {args.input}...")
    d = np.load(args.input, allow_pickle=False)
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    val_mask = game_level_split(meta, frac=0.12, seed=42)
    print(f"  X={X.shape}  train={(~val_mask).sum():,}  val={val_mask.sum():,}")

    print("\n=== A: raw ext features (baseline) ===")
    acc_raw, _ = train_xgb(X, y, val_mask, "raw")

    print("\n=== B: delta ext features (ext - cur_match) ===")
    Xd = to_delta(X)
    acc_delta, bst_delta = train_xgb(Xd, y, val_mask, "delta")

    print(f"\nΔ val sign-acc:  delta - raw = {acc_delta - acc_raw:+.4f}")
    if args.save_delta_model is not None:
        bst_delta.save_model(str(args.save_delta_model))
        print(f"saved delta model: {args.save_delta_model}")


if __name__ == "__main__":
    main()
