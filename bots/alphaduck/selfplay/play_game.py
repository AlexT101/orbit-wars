"""Run one self-play orbit_wars game and dump the kaggle-format JSON.

Usage:
  python3 play_game.py --out games/g000123.json --seed 42 \
      [--bot-a bots/alphaduck/main.py] [--bot-b bots/alphaduck/main.py]

Designed to be called by batch_play.py as a subprocess, one game per process,
so a Python-level crash in alphaduck does not poison sibling games. The bot
binaries are spawned by kaggle_environments under separate child PIDs.

Peak RSS per game: ~3 GB (parent + 2 alphaduck children with model+MCTS).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bot-a", default=None,
                    help="path to bot-a main.py (default: bots/alphaduck/main.py)")
    ap.add_argument("--bot-b", default=None,
                    help="path to bot-b main.py (default: bots/alphaduck/main.py)")
    ap.add_argument("--act-timeout", type=float, default=1.0,
                    help="seconds per turn per agent (default 1.0)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    bot_a = Path(args.bot_a) if args.bot_a else repo / "bots" / "alphaduck" / "main.py"
    bot_b = Path(args.bot_b) if args.bot_b else repo / "bots" / "alphaduck" / "main.py"
    if not bot_a.exists() or not bot_b.exists():
        sys.exit(f"bot path missing: {bot_a} or {bot_b}")

    from kaggle_environments import make
    env = make(
        "orbit_wars",
        debug=False,
        configuration={"seed": int(args.seed), "actTimeout": float(args.act_timeout)},
    )
    t0 = time.time()
    env.run([str(bot_a), str(bot_b)])
    dt = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(env.toJSON(), fh)
    print(f"seed={args.seed} rewards={env.toJSON().get('rewards')} took={dt:.1f}s -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
