"""Two-pass replay filter:
  1. Scan all replay JSONs in a directory for player names + per-game
     rewards (cheap — only reads `info.Agents` + `rewards`).
  2. Compute per-player win rate. Filter to games where BOTH players'
     win rate is above the median, then sample `--limit` games.

Outputs a manifest JSON listing the selected replay file paths so
from_replays_fast.py can process just those.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def scan(path: Path):
    """Return (player_names, rewards) for a 2P game, or None."""
    try:
        with path.open("rb") as f:
            data = json.load(f)
    except Exception:
        return None
    rewards = data.get("rewards") or []
    info = data.get("info") or {}
    agents = info.get("Agents") or []
    if len(rewards) != 2 or len(agents) != 2:
        return None
    if any(r is None for r in rewards):
        return None
    names = [a.get("Name", f"p{i}") for i, a in enumerate(agents)]
    return names, [float(r) for r in rewards]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--min-games", type=int, default=3, help="min games a player must have to count")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    replay_dir = Path(args.replays)
    files = sorted(replay_dir.glob("*.json"))
    print(f"scanning {len(files)} replays for player metadata...")

    games = []  # list of (path, name0, name1, reward0, reward1)
    for i, fp in enumerate(files):
        r = scan(fp)
        if r is None:
            continue
        names, rew = r
        games.append((fp, names[0], names[1], rew[0], rew[1]))
        if (i + 1) % 500 == 0:
            print(f"  scanned {i + 1}/{len(files)}")

    print(f"got {len(games)} valid 2P games")

    # Per-player win rate.
    p_games = defaultdict(int)
    p_wins = defaultdict(int)
    for _, n0, n1, r0, r1 in games:
        p_games[n0] += 1
        p_games[n1] += 1
        if r0 > r1:
            p_wins[n0] += 1
        elif r1 > r0:
            p_wins[n1] += 1
    rates = {p: p_wins[p] / p_games[p] for p in p_games if p_games[p] >= args.min_games}
    if not rates:
        print("no players with enough games")
        return
    sorted_rates = sorted(rates.values())
    median = sorted_rates[len(sorted_rates) // 2]
    print(f"player win-rate stats: n={len(rates)} median={median:.3f}")
    above = {p for p, r in rates.items() if r > median}
    print(f"{len(above)} players above median win rate")

    # Filter games where both players are above median.
    strong = [g for g in games if g[1] in above and g[2] in above]
    print(f"{len(strong)} games where BOTH players are above-median")

    rng = random.Random(args.seed)
    rng.shuffle(strong)
    selected = strong[: args.limit]
    print(f"selected {len(selected)} games")

    manifest = {
        "median_win_rate": median,
        "n_above_median_players": len(above),
        "files": [str(g[0]) for g in selected],
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
