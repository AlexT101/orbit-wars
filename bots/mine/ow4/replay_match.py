"""Run ow4 vs an opponent on the Kaggle engine and save a viewable HTML replay.

Usage:
    python bots/mine/ow4/replay_match.py [opponent] [--seed N] [--out FILE]

Open the resulting file in a browser to scrub through the game tick-by-tick.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent


def bot_entry(name: str) -> Path:
    for cand in [
        REPO / "bots" / name / "main.py",
        REPO / "bots" / "mine" / name / "main.py",
        REPO / "bots" / "baselines" / name / "main.py",
        REPO / "bots" / "external" / name / "main.py",
    ]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find bot {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("opponent", nargs="?", default="apollo_fast")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "replay.html"))
    args = ap.parse_args()

    try:
        from kaggle_environments import make
    except ModuleNotFoundError:
        print('Install: pip install "kaggle-environments>=1.28.0"')
        return 1

    ow4 = bot_entry("ow4")
    opp = bot_entry(args.opponent)
    print(f"Running ow4 vs {args.opponent} (seed={args.seed}) on Kaggle engine...")

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=True)
    env.run([str(ow4), str(opp)])

    html = env.render(mode="html")
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    rewards = [step[0].get("reward") for step in env.steps[-1:]]
    print(f"Saved replay: {out_path}")
    print(f"Final state: {env.state[0]['reward']} vs {env.state[1]['reward']}")
    print(f"Open in browser: file://{out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
