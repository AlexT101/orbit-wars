"""Merge several combined 157-d NPZ datasets into one, offsetting game ids in
meta[:,0] so games stay distinct across sources. Concatenates features/labels/
meta/is_strong and the per-game metadata arrays where present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    feats, labels, metas, strong, dates = [], [], [], [], []
    game_offset = 0
    feature_names = None
    for path in args.inputs:
        d = np.load(path, allow_pickle=False)
        X = d["features"].astype(np.float32)
        y = d["labels"].astype(np.float32)
        meta = d["meta"].astype(np.int32).copy()
        n_games = int(meta[:, 0].max()) + 1
        meta[:, 0] += game_offset
        game_offset += n_games
        feats.append(X)
        labels.append(y)
        metas.append(meta)
        strong.append(d["is_strong"].astype(np.uint8) if "is_strong" in d.files else np.ones(len(y), np.uint8))
        dates.append(d["game_dates"] if "game_dates" in d.files else np.full(n_games, "unknown", dtype="<U10"))
        if feature_names is None and "feature_names" in d.files:
            feature_names = d["feature_names"]
        print(f"  {path.name}: rows={X.shape[0]:,} games={n_games}")

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    meta = np.concatenate(metas, axis=0)
    is_strong = np.concatenate(strong, axis=0)
    game_dates = np.concatenate(dates, axis=0)
    payload = dict(
        features=X, labels=y, meta=meta, is_strong=is_strong, game_dates=game_dates,
        feature_set=np.array("summary_v2", dtype="<U16"),
        label_mode=np.array("native", dtype="<U16"),
        feature_layout=np.array("summary_v2[46] + extras_v4[12] + engineered[88] + tempo[11]", dtype="<U160"),
    )
    if feature_names is not None:
        payload["feature_names"] = feature_names
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"merged rows={X.shape[0]:,} games={int(meta[:,0].max())+1:,} dim={X.shape[1]} -> {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
