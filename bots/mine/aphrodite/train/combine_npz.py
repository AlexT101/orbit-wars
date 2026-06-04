"""Combine Aphrodite training NPZs.

Offsets ``meta[:, 0]`` game ids from each input so game-level validation splits
do not accidentally merge unrelated games from different files.

Example:
    python combine_npz.py \
        --out data/2p/train_2p_mixed.npz \
        data/2p/replays_top10_2p.npz \
        data/2p/selfplay_2p.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("inputs", nargs="+", type=Path)
    args = p.parse_args()

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metas: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    game_offset = 0

    for source_idx, path in enumerate(args.inputs):
        d = np.load(path, allow_pickle=False)
        if "summary_v2" not in d.files or "labels" not in d.files or "meta" not in d.files:
            raise SystemExit(f"{path} is missing one of: summary_v2, labels, meta")
        x = d["summary_v2"].astype(np.float32)
        y = d["labels"].astype(np.float32)
        meta = d["meta"].astype(np.int32).copy()
        if x.shape[0] != y.shape[0] or x.shape[0] != meta.shape[0]:
            raise SystemExit(f"{path} has mismatched row counts")

        old_games = meta[:, 0].astype(np.int64)
        unique_games = np.unique(old_games)
        remap = {int(g): game_offset + i for i, g in enumerate(unique_games)}
        meta[:, 0] = np.fromiter((remap[int(g)] for g in old_games), dtype=np.int32, count=old_games.shape[0])
        game_offset += unique_games.shape[0]

        xs.append(x)
        ys.append(y)
        metas.append(meta)
        sources.append(np.full(x.shape[0], source_idx, dtype=np.int16))
        print(f"{path}: rows={x.shape[0]:,} games={unique_games.shape[0]:,}")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        summary_v2=np.concatenate(xs).astype(np.float32),
        labels=np.concatenate(ys).astype(np.float32),
        meta=np.concatenate(metas).astype(np.int32),
        source=np.concatenate(sources).astype(np.int16),
        source_files=np.array([str(p) for p in args.inputs], dtype="<U260"),
    )
    print(f"wrote {out} rows={sum(x.shape[0] for x in xs):,} games={game_offset:,}")


if __name__ == "__main__":
    main()
