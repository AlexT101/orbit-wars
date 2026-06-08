"""Post-filter a combined Elo-gated NPZ down to games where BOTH players are in
that day's top-N rating list (a stricter gate than the top-15 extraction gate,
which keeps a game if EITHER player is top-15).

No re-extraction needed: a both-top-10 game has both players in top-10 ⊂ the
top-15 that was extracted, so all its rows are already present in the combined
NPZ. We just drop rows whose game isn't both-top-N, then compact gids so
`game_names`/`game_rewards` stay gid-aligned with `meta[:, 0]` (train_xgb indexes
game_names by gid via meta).

The per-day top-N comes from the rating-ORDERED topn_<day>.json keep-lists
(first N entries = top-N). Each row's day is its combine `source` index, mapped
to a day string via `source_files`.

Usage (from train/):
    python filter_both_topn.py \
        --in data/2p/_ladder_work/combined_7day.npz \
        --topn-dir data/2p/_ladder_work --top-n 10 \
        --out data/2p/_ladder_work/combined_7day_both10.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DAY_RE = re.compile(r"gated_replays_(.+?)\.npz$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--topn-dir", required=True, type=Path,
                   help="directory holding topn_<day>.json keep-lists")
    p.add_argument("--top-n", type=int, default=10,
                   help="keep games where both players are in the day's top-N (default 10)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    d = np.load(args.inp, allow_pickle=False)
    X = d["summary_v2"]
    y = d["labels"]
    meta = d["meta"].astype(np.int64).copy()
    source = d["source"].astype(np.int64)
    game_names = d["game_names"]
    game_rewards = d["game_rewards"]
    source_files = [str(s) for s in d["source_files"]]
    n_games = game_names.shape[0]

    # source index -> set of that day's top-N player names
    top_sets: dict[int, set[str]] = {}
    for si, sf in enumerate(source_files):
        m = DAY_RE.search(sf)
        if not m:
            raise SystemExit(f"could not parse day from source_file: {sf}")
        day = m.group(1)
        tj = args.topn_dir / f"topn_{day}.json"
        names = json.loads(tj.read_text(encoding="utf-8"))
        top_sets[si] = set(names[: args.top_n])
        print(f"  source {si} ({day}): top-{args.top_n} = {len(top_sets[si])} players")

    # gid -> source (every row of a game shares one source/day)
    game_source = np.full(n_games, -1, dtype=np.int64)
    game_source[meta[:, 0]] = source  # last write wins; all rows of a gid agree

    # keep games where BOTH player names are in that day's top-N
    keep_game = np.zeros(n_games, dtype=bool)
    for gid in range(n_games):
        si = int(game_source[gid])
        if si < 0:
            continue  # game has no rows in this NPZ (shouldn't happen)
        ts = top_sets[si]
        a, b = str(game_names[gid, 0]), str(game_names[gid, 1])
        keep_game[gid] = (a in ts) and (b in ts)

    row_mask = keep_game[meta[:, 0]]
    kept_gids = np.nonzero(keep_game)[0]
    print(f"\n  games: {int(keep_game.sum()):,} / {n_games:,} kept "
          f"({100 * keep_game.mean():.1f}%)")
    print(f"  rows : {int(row_mask.sum()):,} / {X.shape[0]:,} kept "
          f"({100 * row_mask.mean():.1f}%)")
    if kept_gids.size == 0:
        raise SystemExit("no games survived the both-top-N filter")

    # compact gids -> 0..M-1 so game_names/game_rewards stay gid-indexed
    remap = np.full(n_games, -1, dtype=np.int64)
    remap[kept_gids] = np.arange(kept_gids.shape[0])
    Xf = X[row_mask]
    yf = y[row_mask]
    metaf = meta[row_mask]
    metaf[:, 0] = remap[metaf[:, 0]]
    sourcef = source[row_mask]
    gnf = game_names[kept_gids]
    grf = game_rewards[kept_gids]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        summary_v2=Xf.astype(np.float32),
        labels=yf.astype(np.float32),
        meta=metaf.astype(np.int32),
        source=sourcef.astype(np.int16),
        source_files=np.array(source_files, dtype="<U260"),
        game_names=gnf.astype("<U64"),
        game_rewards=grf.astype(np.float32),
    )
    print(f"\nwrote {args.out} rows={Xf.shape[0]:,} games={kept_gids.shape[0]:,}")


if __name__ == "__main__":
    main()
