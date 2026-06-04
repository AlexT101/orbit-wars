"""Play target_predictor against a list of opponents over multiple seeds and
seat assignments; print per-opponent win/loss/draw and overall record.

Usage:
  python3 bots/mine/target_predictor/gauntlet.py \\
    --opponents apollo hellburner nearest-sniper random \\
    --seeds 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOTS_DIR = ROOT / "bots"
sys.path.insert(0, str(ROOT))


def bot_entry(name: str) -> Path:
    direct = BOTS_DIR / name / "main.py"
    if direct.is_file():
        return direct
    for subdir in BOTS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        cand = subdir / name / "main.py"
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no main.py for bot {name}")


def play(bot_a: Path, bot_b: Path, seed: int):
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    env.run([str(bot_a), str(bot_b)])
    return [s.reward for s in env.state]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default="target_predictor")
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-seed", type=int, default=42)
    args = ap.parse_args()

    me_path = bot_entry(args.me)
    print(f"  me = {args.me} ({me_path})")
    for opp in args.opponents:
        opp_path = bot_entry(opp)
        print(f"  vs {opp} ({opp_path})")

    overall_w = overall_l = overall_d = 0
    summary = []
    for opp in args.opponents:
        opp_path = bot_entry(opp)
        w = l = d = 0
        details = []
        for seed_i in range(args.seeds):
            seed = args.start_seed + seed_i * 7919
            for seat in (0, 1):  # 0 = me first, 1 = me second
                if seat == 0:
                    a, b = me_path, opp_path
                else:
                    a, b = opp_path, me_path
                t0 = time.time()
                rewards = play(a, b, seed)
                dt = time.time() - t0
                my_r = rewards[seat]
                opp_r = rewards[1 - seat]
                outcome = "W" if my_r > opp_r else "L" if my_r < opp_r else "D"
                if outcome == "W": w += 1
                elif outcome == "L": l += 1
                else: d += 1
                details.append(f"seed={seed:>10d} seat={'P0' if seat == 0 else 'P1'} "
                               f"r={my_r}:{opp_r} {outcome} ({dt:.0f}s)")
                print(f"  vs {opp}  {details[-1]}", flush=True)
        overall_w += w; overall_l += l; overall_d += d
        n = w + l + d
        wr = 100.0 * (w + 0.5 * d) / max(n, 1)
        summary.append((opp, w, l, d, wr))
        print(f"  ==> vs {opp}: {w}W-{l}L-{d}D  ({wr:.1f}% score over {n} games)", flush=True)
        print()

    print("=== GAUNTLET SUMMARY ===")
    print(f"  {'opponent':<20} {'W':>3} {'L':>3} {'D':>3}   score%")
    for opp, w, l, d, wr in summary:
        print(f"  {opp:<20} {w:>3} {l:>3} {d:>3}   {wr:5.1f}%")
    total = overall_w + overall_l + overall_d
    overall_wr = 100.0 * (overall_w + 0.5 * overall_d) / max(total, 1)
    print(f"  {'OVERALL':<20} {overall_w:>3} {overall_l:>3} {overall_d:>3}   {overall_wr:5.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
