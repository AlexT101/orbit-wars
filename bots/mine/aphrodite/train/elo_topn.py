"""Phase 1 of the Elo-gated training pipeline.

Cheaply scans replay zips for game OUTCOMES ONLY — each game's agent names and
rewards — without running the (expensive) Rust feature extraction, fits a single
GLOBAL Bradley-Terry rating across every game, and writes the top-N players by
rating to a JSON file.

`build_from_zip.py --keep-players <that file>` then extracts SummaryV2 features
only for those players' rows, so the costly extraction never touches rows the
Elo gate would have discarded. This replaces the old win-rate gate (which
required extracting everything first, then dropping most of it).

Ratings are global (pooled over all zips), so unlike the old per-day win-rate
gate a strong player having an unlucky day is still kept.

Usage:
    python elo_topn.py --zip ../../../../ladder_replays/*.zip \
        --players 2 --top-n 15 --out data/2p/_ladder_work/topn_players.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_xgb import bradley_terry_ratings  # noqa: E402  (shared BT fitter)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _scan_chunk(args):
    """Read agent names + rewards for each assigned replay (no extraction)."""
    zip_path, entries, n_players = args
    zf = zipfile.ZipFile(zip_path)
    out: list[tuple] = []  # (names_tuple, rewards_tuple)
    for entry in entries:
        try:
            d = json.loads(zf.read(entry))
        except Exception:
            continue
        rewards = d.get("rewards") or []
        if len(rewards) != n_players or any(r is None for r in rewards[:n_players]):
            continue
        agents = (d.get("info") or {}).get("Agents") or []
        if len(agents) != n_players:
            continue
        names = tuple(str(a.get("Name", f"p{i}")) for i, a in enumerate(agents[:n_players]))
        rews = tuple(float(r) for r in rewards[:n_players])
        out.append((names, rews))
    zf.close()
    return out


def scan_outcomes(zip_paths, n_players: int, workers: int):
    """Return (game_names, game_rewards) arrays over every valid game in the zips."""
    names_all: list[tuple] = []
    rewards_all: list[tuple] = []
    for zi, zip_path in enumerate(zip_paths):
        zf = zipfile.ZipFile(zip_path)
        entries = [n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/")]
        zf.close()
        entries.sort()
        chunks = [entries[i::workers] for i in range(workers)]
        print(f">>> [{zi + 1}/{len(zip_paths)}] {Path(zip_path).name}: {len(entries)} entries", flush=True)
        with mp.Pool(workers) as pool:
            for res in pool.map(_scan_chunk, [(zip_path, c, n_players) for c in chunks]):
                for nm, rw in res:
                    names_all.append(nm)
                    rewards_all.append(rw)
    if not names_all:
        raise SystemExit("no valid games found in the provided zips")
    return (
        np.array(names_all, dtype="<U64"),
        np.array(rewards_all, dtype=np.float32),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, nargs="+", help="one or more replay zips (globbed)")
    p.add_argument("--players", type=int, choices=(2, 4), default=2)
    p.add_argument("--top-n", type=int, default=15, help="how many top-rated players to keep")
    p.add_argument("--min-games", type=int, default=5, help="min appearances for a rating")
    p.add_argument("--prior-strength", type=float, default=3.0,
                   help="Bradley-Terry Gamma-prior strength: shrinks low-game players toward the "
                        "field average so a few-game lucky run can't top the ranking (higher = more shrinkage)")
    p.add_argument("--out", required=True, type=Path, help="JSON file of kept player names")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    game_names, game_rewards = scan_outcomes(args.zip, args.players, args.workers)
    print(f"\nscanned {game_names.shape[0]:,} games; fitting global Bradley-Terry rating...")
    ratings = bradley_terry_ratings(game_names, game_rewards, args.min_games,
                                    prior_strength=args.prior_strength)
    if not ratings:
        raise SystemExit(f"no players reached --min-games={args.min_games}")

    ordered = sorted(ratings.items(), key=lambda kv: -kv[1])
    keep = [name for name, _ in ordered[: args.top_n]]
    print(f"\ntop-{args.top_n} of {len(ratings)} rated players (by global Elo):")
    for name, r in ordered[: args.top_n]:
        print(f"   {r:+.3f}  {name[:50]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len(keep)} player names -> {args.out}")


if __name__ == "__main__":
    if os.name != "nt":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
