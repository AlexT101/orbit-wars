"""End-to-end "what last time looked like, but with the fix" — local-only.

Last-time pipeline:
  kaggle_rebuild_v2.py  (on Kaggle, ~3.6k raw replays at /kaggle/input/) →
    combined_top10.npz  →  xgb_tune.py  →  xgb_top10_d6.json

We can't run the Kaggle step locally (5 raw JSONs vs 3,616 games), so this
script does the SAME pipeline on the 5 local replays for BOTH the BUGGY and
FIXED extrapolation. The result is a directional/A-B comparison, not a
deployment-quality model. Saves both NPZs + both XGB JSONs to disk.

Outputs (in data/ and weights/):
  data/local5_buggy.npz       — 1,210 rows × 46 features, BUGGY extrap
  data/local5_fixed.npz       — same shape, FIXED extrap (production accrual etc.)
  weights/xgb_local5_buggy.json
  weights/xgb_local5_fixed.json

Plus a printed Ridge-regression + XGBoost summary on each, with feature-coef
deltas and val accuracy.
"""

import json
import pathlib
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kaggle_rebuild_v2 as kr  # noqa: E402
import rebuild_fixed_extrap as rfe  # noqa: E402

DATA = HERE / "data"
WEIGHTS = HERE / "weights"
DATA.mkdir(exist_ok=True)
WEIGHTS.mkdir(exist_ok=True)


def build_local_npz(extrap_fn, out_path: pathlib.Path):
    replays_dir = HERE.parent.parent.parent.parent / "replays"
    json_files = sorted(replays_dir.glob("*.json"))
    feats, labels, meta = [], [], []
    game_idx = 0
    for jf in json_files:
        res = rfe.process_game(jf, extrap_fn)
        if res is None:
            continue
        f, y = res
        feats.append(f)
        labels.append(y)
        # meta col0 = game id (for game-level val split), rest = (step, player, 1-player)
        meta.append(np.stack([np.full(len(f), game_idx, dtype=np.int32),
                              np.arange(len(f), dtype=np.int32),
                              np.zeros(len(f), dtype=np.int32),
                              np.zeros(len(f), dtype=np.int32)], axis=1))
        game_idx += 1
    if not feats:
        raise SystemExit("no replays processed")
    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.float32)
    m = np.concatenate(meta, axis=0).astype(np.int32)
    np.savez_compressed(out_path, summary_v2=X, labels=y, meta=m)
    print(f"  wrote {out_path.name}: X={X.shape} y={y.shape}")
    return X, y, m


def game_level_split(meta, frac=0.20, seed=42):
    games = meta[:, 0]
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_games = set(unique[:n_val].tolist())
    return np.array([g in val_games for g in games])


def train_ridge(X, y, val_mask, label):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = Ridge(alpha=1.0)
    m.fit(Xs[~val_mask], y[~val_mask])
    yhat_va = m.predict(Xs[val_mask])
    acc = float(np.mean(np.sign(yhat_va) == np.sign(y[val_mask])))
    mse = float(np.mean((yhat_va - y[val_mask]) ** 2))
    print(f"  [{label}] Ridge val_acc={acc:.4f}  val_mse={mse:.4f}")
    return m.coef_, acc


def train_xgb(X, y, val_mask, label, out_json):
    # Match xgb_tune.py / train_gbm.py shape: binary:logistic on (y > 0)
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
    elapsed = time.time() - t0
    print(f"  [{label}] XGB val_acc={acc:.4f}  best_iter={bst.best_iteration}  t={elapsed:.1f}s")
    bst.save_model(str(out_json))
    print(f"      saved {out_json.name}")
    return acc


def main():
    print("=== STEP 1: rebuild local NPZs ===")
    print("BUGGY (original extrapolate_fleets — no production accrual):")
    Xb, yb_arr, mb = build_local_npz(kr.extrapolate_fleets, DATA / "local5_buggy.npz")
    print("FIXED (extrapolate_fixed — adds production accrual + same-tick agg + ties):")
    Xf, yf_arr, mf = build_local_npz(rfe.extrapolate_fixed, DATA / "local5_fixed.npz")

    # The two NPZs share the same labels and meta, just different features.
    assert np.array_equal(yb_arr, yf_arr)
    assert np.array_equal(mb, mf)
    y = yb_arr
    meta = mb
    val_mask = game_level_split(meta, frac=0.25, seed=42)
    print(f"\nval split: train={(~val_mask).sum()}  val={val_mask.sum()}  "
          f"({len(np.unique(meta[val_mask, 0]))} val games of {len(np.unique(meta[:,0]))})")

    print("\n=== STEP 2: Ridge regression (linear sanity) ===")
    cb, ridge_acc_b = train_ridge(Xb, y, val_mask, "buggy")
    cf, ridge_acc_f = train_ridge(Xf, y, val_mask, "fixed")

    print("\n=== STEP 3: XGBoost (same params as xgb_tune baseline: d=6, lr=0.08) ===")
    xgb_acc_b = train_xgb(Xb, y, val_mask, "buggy", WEIGHTS / "xgb_local5_buggy.json")
    xgb_acc_f = train_xgb(Xf, y, val_mask, "fixed", WEIGHTS / "xgb_local5_fixed.json")

    print("\n=== STEP 4: COEF DELTAS (Ridge, top-12 by max |coef|) ===")
    pairs = sorted(range(46), key=lambda i: -max(abs(cb[i]), abs(cf[i])))[:12]
    print(f"{'idx':>4} {'feature':<32}  {'buggy':>9}  {'fixed':>9}  {'Δ':>9}")
    for i in pairs:
        d = cf[i] - cb[i]
        tag = " (SIGN FLIP)" if cb[i] * cf[i] < -1e-3 else ""
        print(f"{i:>4} {rfe.NAMES[i]:<32}  {cb[i]:+9.4f}  {cf[i]:+9.4f}  {d:+9.4f}{tag}")

    print("\n=== SUMMARY ===")
    print(f"  Ridge:    buggy={ridge_acc_b:.4f}  fixed={ridge_acc_f:.4f}  Δ={ridge_acc_f-ridge_acc_b:+.4f}")
    print(f"  XGBoost:  buggy={xgb_acc_b:.4f}  fixed={xgb_acc_f:.4f}  Δ={xgb_acc_f-xgb_acc_b:+.4f}")
    print()
    print("NOTE: 5 local replays = ~1,210 rows = 1400x less data than the deployed")
    print("xgb_top10_d6.json (1.7M rows / 3,616 games). Absolute accuracy is noisy at")
    print("this scale; treat the Δ values as DIRECTIONAL evidence, not the new ceiling.")
    print("To actually replace the deployed XGB JSON, run kaggle_rebuild_v2.py with")
    print("extrapolate_fixed patched in on Kaggle (where /kaggle/input/ has the 3,616")
    print("raw replays), then re-run xgb_tune.py on the resulting NPZ.")


if __name__ == "__main__":
    main()
