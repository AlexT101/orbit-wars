"""Run a match between two bots via the Kaggle engine and open the HTML replay."""

import argparse
import os
import random
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"


def bot_entry(bot_name: str) -> Path:
    direct = BOTS_DIR / bot_name / "main.py"
    if direct.is_file():
        return direct
    for subdir in BOTS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / bot_name / "main.py"
        if candidate.is_file():
            return candidate
    return direct


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bot1")
    parser.add_argument("bot2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="replay.html")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    print(f"Seed: {seed}")

    bot_paths = [bot_entry(args.bot1), bot_entry(args.bot2)]
    missing = [str(p) for p in bot_paths if not p.is_file()]
    if missing:
        print("Missing bot file(s):", missing)
        return 1

    from kaggle_environments import make

    print(f"Match (kaggle engine): {args.bot1} vs {args.bot2}")
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run([str(p) for p in bot_paths])

    final = env.steps[-1]
    for i, (name, state) in enumerate(zip([args.bot1, args.bot2], final)):
        print(f"Player {i} ({name}): reward={state.reward}, status={state.status}")

    html = env.render(mode="html")
    out = Path(args.out).resolve()
    out.write_text(html, encoding="utf-8")
    print(f"Replay: {out}")

    if not args.no_open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
