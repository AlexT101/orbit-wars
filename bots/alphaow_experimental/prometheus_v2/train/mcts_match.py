"""Run N games of alphaow (current) vs a baseline bot, with optional env
overrides for the MCTS variant under test. Reports win/draw/loss + per-game
seed so failures are reproducible.

Usage:
  python mcts_match.py --me alphaow --baseline alphaow_base --n 10 \\
      --env OW_K_ROOT=3 OW_ROLLOUT_DEPTH=12 OW_PUCT_C=0.5
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]   # bots/mine/alphaow/train/<this> -> orbit-wars


def run_one(me_bot: str, opp_bot: str, seed: int, env_overrides: dict, swap: bool):
    bot1, bot2 = (opp_bot, me_bot) if swap else (me_bot, opp_bot)
    env = os.environ.copy()
    env.update(env_overrides)
    # Apply env overrides only to alphaow's slot. The match runner loads both
    # bots in the same Python process, so overrides apply to both -- that's
    # OK for our experiment since opp is the *baseline* bot whose code path
    # doesn't read these env vars (it has its own settings).
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, str(ROOT / "run_match.py"), bot1, bot2, "--seed", str(seed)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    dur = time.time() - t0
    out = p.stdout.decode(errors="replace")
    # Parse "Player N (name): reward=R" lines from run_match.py
    import re
    rewards = {}
    for line in out.splitlines():
        m = re.match(r'^Player (\d+) \([^)]+\):\s*reward=([-\d.]+)', line.strip())
        if m:
            rewards[int(m.group(1))] = float(m.group(2))
    winner_slot = None
    if 0 in rewards and 1 in rewards:
        if rewards[0] > rewards[1]: winner_slot = 0
        elif rewards[1] > rewards[0]: winner_slot = 1
        else: winner_slot = -1   # draw
    # Translate back: did "me" win?
    if winner_slot is None:
        return None, dur, out
    me_slot = 1 if swap else 0
    won = winner_slot == me_slot
    return won, dur, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--me", default="alphaow")
    p.add_argument("--baseline", required=True, help="bot folder name under bots/mine/")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--env", nargs="*", default=[], help="K=V overrides for me's bot")
    p.add_argument("--swap", action="store_true", help="play both sides (n games each side)")
    p.add_argument("--save-log", default=None)
    args = p.parse_args()

    env_over = {}
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1); env_over[k] = v
    print(f"me={args.me}  baseline={args.baseline}  n={args.n}  env={env_over}")
    print(f"swap={args.swap}  total games = {args.n * (2 if args.swap else 1)}")

    rng = random.Random(2026)
    wins = losses = draws = unknown = 0
    rows = []
    sides = [False, True] if args.swap else [False]
    for swap in sides:
        for i in range(args.n):
            seed = rng.randint(0, 2**31 - 1)
            won, dur, out = run_one(args.me, args.baseline, seed, env_over, swap)
            tag = "W" if won is True else ("L" if won is False else "?")
            print(f"  game seed={seed:>10} swap={swap}  {tag}  ({dur:.1f}s)", flush=True)
            rows.append((seed, swap, tag, dur, out))
            if won is True: wins += 1
            elif won is False: losses += 1
            else: unknown += 1
    n = wins + losses + draws + unknown
    print()
    print(f"== {wins}W - {losses}L - {draws}D - {unknown}? (n={n})  win-rate (decided) = {100*wins/max(1,wins+losses):.1f}% ==")

    if args.save_log:
        with open(args.save_log, "w") as f:
            for seed, swap, tag, dur, out in rows:
                f.write(f"### seed={seed} swap={swap} result={tag} dur={dur:.1f}\n{out}\n\n")


if __name__ == "__main__":
    main()
