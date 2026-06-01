"""Evaluate a saved XGB model on a fresh test split — handles older
models trained with fewer features by slicing X to the first N columns
that match the model's `num_features()`."""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from train_first_owned_xgb import extract_game  # noqa: E402


def chunk(args):
    zip_path, names = args
    out_X, out_y, out_w = [], [], []
    out_meta = []
    group_map = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in names:
            try:
                with z.open(name) as f:
                    import json
                    data = json.load(f)
                rows = extract_game(data, name)
                if not rows:
                    continue
                for feats, label, weight, gid, persp, pid, r in rows:
                    key = (gid, persp)
                    if key not in group_map:
                        group_map[key] = len(group_map)
                    out_X.append(feats)
                    out_y.append(label)
                    out_w.append(weight)
                    out_meta.append((group_map[key], pid, r))
            except Exception:
                continue
    return (
        np.asarray(out_X, dtype=np.float32),
        np.asarray(out_y, dtype=np.int8),
        np.asarray(out_w, dtype=np.float32),
        np.asarray(out_meta, dtype=np.int64),
        len(group_map),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--zip", nargs="+", required=True)
    ap.add_argument("--rank1-only-label", action="store_true",
                    help="Use rank==1 as positive label (must match training)")
    ap.add_argument("--winners-only", action="store_true")
    args = ap.parse_args()

    # Toggle global flags read inside extract_game.
    import builtins
    builtins.OW4_NO_WEIGHTS = False
    builtins.OW4_NO_SYM_BONUS = False
    builtins.OW4_RANK1_ONLY = args.rank1_only_label
    builtins.OW4_WINNERS_ONLY = args.winners_only

    bst = xgb.Booster()
    bst.load_model(args.model)
    n_feat_model = bst.num_features()
    print(f"model expects {n_feat_model} features")

    chunks_in = []
    for zp in args.zip:
        with zipfile.ZipFile(zp) as z:
            names = sorted(z.namelist())
        cs = 100
        for i in range(0, len(names), cs):
            chunks_in.append((zp, names[i:i + cs]))
    print(f"{sum(len(c[1]) for c in chunks_in)} games over {len(chunks_in)} chunks")

    Xs, ys, ws, metas = [], [], [], []
    offset = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=9) as ex:
        futs = [ex.submit(chunk, c) for c in chunks_in]
        for k, f in enumerate(as_completed(futs)):
            X, y, w, meta, ngroup = f.result()
            if len(X):
                meta[:, 0] += offset
                offset += ngroup
                Xs.append(X)
                ys.append(y)
                ws.append(w)
                metas.append(meta)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(chunks_in)} elapsed={time.time()-t0:.0f}s")

    X_full = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    meta = np.concatenate(metas, axis=0)
    print(f"got {X_full.shape[0]} rows x {X_full.shape[1]} cols; model wants {n_feat_model}")

    # Slice to match model: take the first n_feat_model columns (works
    # for older models trained without the newer reciprocal features
    # since those were appended at the end).
    if X_full.shape[1] < n_feat_model:
        raise SystemExit(
            f"current extraction emits fewer cols ({X_full.shape[1]}) than "
            f"the model expects ({n_feat_model})"
        )
    X = X_full[:, :n_feat_model]
    feat_names = [f"f{i}" for i in range(n_feat_model)]

    # Group-aware split (same seed as training script).
    rng = np.random.default_rng(0)
    n_groups = int(meta[:, 0].max()) + 1
    perm = rng.permutation(n_groups)
    n_test_groups = max(200, n_groups // 10)
    test_groups = set(perm[:n_test_groups].tolist())
    test_mask = np.fromiter(
        (g in test_groups for g in meta[:, 0]), dtype=bool, count=len(meta)
    )
    test_i = np.where(test_mask)[0]
    Xte, yte = X[test_i], y[test_i]
    meta_te = meta[test_i]

    pred = bst.predict(xgb.DMatrix(Xte, feature_names=feat_names))
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    bin_pred = (pred >= 0.5).astype(int)
    print(f"\nholdout: acc={accuracy_score(yte, bin_pred):.4f}  "
          f"ll={log_loss(yte, pred):.4f}  auc={roc_auc_score(yte, pred):.4f}")

    # Rank-1 top-1 metric.
    groups_te = meta_te[:, 0]
    ranks_te = meta_te[:, 2]
    order = np.argsort(groups_te, kind="stable")
    groups_sorted = groups_te[order]
    ranks_sorted = ranks_te[order]
    pred_sorted = pred[order]
    n_correct, n_total = 0, 0
    i, n = 0, len(order)
    while i < n:
        j = i
        while j < n and groups_sorted[j] == groups_sorted[i]:
            j += 1
        g_ranks = ranks_sorted[i:j]
        g_pred = pred_sorted[i:j]
        actual = np.where(g_ranks == 1)[0]
        nh_mask = g_ranks != 0
        if len(actual) == 1 and nh_mask.any():
            n_total += 1
            cand_idx = np.where(nh_mask)[0]
            top = cand_idx[int(np.argmax(g_pred[cand_idx]))]
            if top == actual[0]:
                n_correct += 1
        i = j
    print(f"rank-1 top-1 acc: {n_correct}/{n_total} = {n_correct/n_total:.4f}")


if __name__ == "__main__":
    main()
