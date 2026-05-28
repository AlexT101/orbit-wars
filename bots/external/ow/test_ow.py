"""Reliable tester for the ow bot. Use `debug=True` because kaggle's
`debug=False` codepath consumes Python's global `random` state differently
than `debug=True`, making `random_agent`'s behavior nondeterministic between
runs. Pass `--swap` to put ow as player 1.
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault(
    "OW_BOT_DIR", os.path.dirname(os.path.abspath(__file__))
)

from kaggle_environments import make  # noqa: E402

SEEDS = [42, 7, 123, 2024, 9999, 555, 777, 333]
OW = "ow/main.py"


def run_match(opp, seed, ow_first=True):
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    random.seed(seed)  # Stabilizes random_agent
    agents = [OW, opp] if ow_first else [opp, OW]
    env.run(agents)
    rewards = [s.reward for s in env.steps[-1]]
    ow_idx = 0 if ow_first else 1
    return rewards[ow_idx] == 1, len(env.steps), env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="random", help="opponent agent (path or name)")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--swap", action="store_true", help="ow as player 1")
    args = ap.parse_args()

    wins = 0
    t0 = time.time()
    for seed in args.seeds:
        won, steps, _ = run_match(args.opp, seed, ow_first=not args.swap)
        wins += int(won)
        print(f"seed={seed:>5} {'WIN' if won else 'LOSS':4} steps={steps}")
    print(f"\nvs {args.opp} ({'p1' if args.swap else 'p0'}): "
          f"{wins}-{len(args.seeds)-wins} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
