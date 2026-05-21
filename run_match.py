import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"
OPEN_SOURCE_BOTS_DIR = BOTS_DIR / "_open_source"


def bot_entry(bot_name: str) -> Path:
    candidates = [
        BOTS_DIR / bot_name / "main.py",
        OPEN_SOURCE_BOTS_DIR / bot_name / "main.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python run_match.py <bot1> <bot2>")
        return 1

    bot_names = sys.argv[1:3]
    bot_paths = [bot_entry(name) for name in bot_names]

    missing = [str(path.relative_to(ROOT)) for path in bot_paths if not path.is_file()]
    if missing:
        print("Missing bot file(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    try:
        from kaggle_environments import make
    except ModuleNotFoundError:
        print('Missing dependency: install with `pip install "kaggle-environments>=1.28.0"`')
        return 1

    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([str(path) for path in bot_paths])

    print(f"Match: {bot_names[0]} vs {bot_names[1]}")
    final = env.steps[-1]
    for i, state in enumerate(final):
        print(f"Player {i}: reward={state.reward}, status={state.status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
