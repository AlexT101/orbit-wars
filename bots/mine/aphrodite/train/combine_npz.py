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
    win_rates: list[np.ndarray] = []
    auxes: list[np.ndarray] = []          # v3 decisiveness_aux, if present
    have_aux = True
    feat_key = None                       # "summary_v3" (4p) or "summary_v2"
    # Per-game (names, rewards) rows accumulated in global-gid order, so a
    # downstream rating fit (Bradley-Terry) can see who played whom and who won.
    game_names_all: list[np.ndarray] = []
    game_rewards_all: list[np.ndarray] = []
    have_win_rate = True
    have_games = True
    game_offset = 0

    for source_idx, path in enumerate(args.inputs):
        d = np.load(path, allow_pickle=False)
        key = "summary_v3" if "summary_v3" in d.files else "summary_v2"
        if key not in d.files or "labels" not in d.files or "meta" not in d.files:
            raise SystemExit(f"{path} is missing one of: summary_v2/summary_v3, labels, meta")
        if feat_key is None:
            feat_key = key
        elif key != feat_key:
            raise SystemExit(f"{path} has {key} but earlier inputs had {feat_key}; don't mix feature versions")
        x = d[key].astype(np.float32)
        y = d["labels"].astype(np.float32)
        meta = d["meta"].astype(np.int32).copy()
        if x.shape[0] != y.shape[0] or x.shape[0] != meta.shape[0]:
            raise SystemExit(f"{path} has mismatched row counts")

        old_games = meta[:, 0].astype(np.int64)
        unique_games = np.unique(old_games)
        remap = {int(g): game_offset + i for i, g in enumerate(unique_games)}
        meta[:, 0] = np.fromiter((remap[int(g)] for g in old_games), dtype=np.int32, count=old_games.shape[0])

        # Carry per-game names/rewards aligned to the new global gids: global gid
        # == game_offset + i for the i-th entry of sorted `unique_games`, so
        # appending in that order keeps the combined arrays gid-indexed. Note the
        # per-file arrays are indexed by the file's ORIGINAL (full) gid, of which
        # `unique_games` selects exactly the rows referenced by `meta`.
        if "game_names" in d.files and "game_rewards" in d.files:
            gn = d["game_names"]
            gr = d["game_rewards"]
            for g in unique_games:
                game_names_all.append(gn[int(g)])
                game_rewards_all.append(gr[int(g)])
        else:
            have_games = False

        game_offset += unique_games.shape[0]

        xs.append(x)
        ys.append(y)
        metas.append(meta)
        sources.append(np.full(x.shape[0], source_idx, dtype=np.int16))
        # Per-row strength carried through only if every input has it.
        if "win_rate" in d.files:
            win_rates.append(d["win_rate"].astype(np.float32))
        else:
            have_win_rate = False
        if "decisiveness_aux" in d.files:
            auxes.append(d["decisiveness_aux"].astype(np.float32))
        else:
            have_aux = False
        print(f"{path}: rows={x.shape[0]:,} games={unique_games.shape[0]:,}")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if have_win_rate:
        extra["win_rate"] = np.concatenate(win_rates).astype(np.float32)
    else:
        print("note: not all inputs had `win_rate`; dropping it from the combined NPZ")
    if have_games and game_names_all:
        extra["game_names"] = np.array(game_names_all, dtype="<U64")
        extra["game_rewards"] = np.array(game_rewards_all, dtype=np.float32)
    else:
        print("note: not all inputs had game_names/game_rewards; dropping them "
              "(rating-based quality weighting will be unavailable)")
    if have_aux and auxes:
        extra["decisiveness_aux"] = np.concatenate(auxes).astype(np.float32)
    extra[feat_key] = np.concatenate(xs).astype(np.float32)
    np.savez_compressed(
        out,
        labels=np.concatenate(ys).astype(np.float32),
        meta=np.concatenate(metas).astype(np.int32),
        source=np.concatenate(sources).astype(np.int16),
        source_files=np.array([str(p) for p in args.inputs], dtype="<U260"),
        **extra,
    )
    print(f"wrote {out} rows={sum(x.shape[0] for x in xs):,} games={game_offset:,}")


if __name__ == "__main__":
    main()
