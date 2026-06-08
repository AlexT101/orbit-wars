"""Play alphaduck vs a list of opponents over N seeds × 2 seats; report results.

Usage:
  python3 bots/alphaduck/eval_alphaduck.py \
    --opponents random nearest-sniper alphaow \
    --seeds 3 \
    --budget-ms 400
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOTS_DIR = ROOT / "bots"


def bot_path(name: str) -> Path:
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
    ap.add_argument("--me", default="alphaduck")
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-seed", type=int, default=42)
    ap.add_argument("--budget-ms", type=int, default=400)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    # Set env vars before any import of bots (so kaggle env passes them).
    os.environ["ALPHADUCK_BUDGET_MS"] = str(args.budget_ms)
    os.environ["ALPHADUCK_USE_VALUE"] = "1"
    if args.ckpt:
        os.environ["PAIR_NET_CKPT"] = args.ckpt
    # alphaow needs to know where its binary lives
    os.environ.setdefault("ALPHAOW_BOT_DIR", str(ROOT / "bots" / "mine" / "alphaow"))

    me = bot_path(args.me)
    print(f"me = {args.me}  budget_ms={args.budget_ms}  ckpt={args.ckpt or 'default'}")
    overall_w = overall_l = overall_d = 0

    for opp_name in args.opponents:
        try:
            opp = bot_path(opp_name)
        except FileNotFoundError as e:
            print(f"  SKIP {opp_name}: {e}")
            continue
        w = l = d = 0
        for s in range(args.seeds):
            seed = args.start_seed + s * 1000
            for seat in (0, 1):
                t0 = time.time()
                bots = (me, opp) if seat == 0 else (opp, me)
                try:
                    r = play(bots[0], bots[1], seed)
                except Exception as exc:
                    print(f"  vs {opp_name} seed={seed} seat=P{seat} ERROR {exc!r}")
                    continue
                my_r = r[seat]
                opp_r = r[1 - seat]
                if my_r > opp_r: w += 1; tag = "W"
                elif my_r < opp_r: l += 1; tag = "L"
                else: d += 1; tag = "D"
                dt = time.time() - t0
                print(f"  vs {opp_name:18s} seed={seed:6d} seat=P{seat} "
                      f"r={my_r}:{opp_r} {tag} ({dt:.0f}s)")
        score = (w + 0.5 * d) / max(1, (w + l + d))
        print(f"  ==> vs {opp_name}: {w}W-{l}L-{d}D  ({score * 100:.1f}% over {w+l+d})")
        overall_w += w; overall_l += l; overall_d += d

    print("\n=== OVERALL ===")
    score = (overall_w + 0.5 * overall_d) / max(1, (overall_w + overall_l + overall_d))
    print(f"  {overall_w}W-{overall_l}L-{overall_d}D  ({score * 100:.1f}%)")


if __name__ == "__main__":
    main()
