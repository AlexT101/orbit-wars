"""Concatenate spatial[13] onto a combined 157-d dataset -> 170-d, preserving
all metadata. Column order: [features 157][spatial 13], matching the Rust
`summary_features_v10` layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spatial_features import SPATIAL_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined", type=Path, required=True)
    p.add_argument("--spatial", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = dict(np.load(args.combined, allow_pickle=False))
    sp = np.load(args.spatial, allow_pickle=False)["spatial"].astype(np.float32)
    X = d["features"].astype(np.float32)
    if sp.shape[0] != X.shape[0]:
        raise SystemExit(f"row mismatch {sp.shape} vs {X.shape}")
    d["features"] = np.concatenate([X, sp], axis=1).astype(np.float32)
    if "feature_names" in d:
        names = d["feature_names"].astype(str).tolist() + list(SPATIAL_NAMES)
        d["feature_names"] = np.array(names, dtype="<U96")
    d["feature_layout"] = np.array(
        "summary_v2[46] + extras_v4[12] + engineered[88] + tempo[11] + spatial[13]", dtype="<U200"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **d)
    print(f"wrote {args.out} dim={d['features'].shape[1]} rows={d['features'].shape[0]:,} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
