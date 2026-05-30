"""Batch tournament runner. Runs N matches between two bots over a deterministic
seed range and reports win rate / average step time.

Usage:
    python run_batched.py <bot1> <bot2> <n_matches> [--start-seed S]

Each match runs in the same Python process; agents are re-imported per match
under a fresh module name so any module-level singletons (e.g. apollo's
`_BOT = apollo_native.Bot()`) are re-created cleanly per match.
"""

import argparse
import contextlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"
MAX_STEPS = 500


@contextlib.contextmanager
def _silence_imports():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    logging.disable(logging.CRITICAL)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)
        logging.disable(logging.NOTSET)


with _silence_imports():
    try:
        import kaggle_environments  # noqa: F401
    except ModuleNotFoundError:
        pass


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


def load_agent(main_path: Path, mod_name: str):
    # Drop any cached module with this name so module-level state (singletons
    # like `_BOT = apollo_native.Bot()`) is rebuilt each match.
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module.agent


def run_one_match(bot_paths, seed, match_idx):
    from engine_parity_checker.candidates.rust import RustEngine

    agents = [
        load_agent(path, f"bot_{i}_main")
        for i, path in enumerate(bot_paths)
    ]
    engine = RustEngine()
    obs = engine.reset(seed, len(agents))

    total_time = [0.0] * len(agents)
    call_counts = [0] * len(agents)

    done = False
    steps_run = 0
    for _ in range(MAX_STEPS):
        actions = []
        for i in range(len(agents)):
            start = perf_counter()
            action = agents[i](obs[i].as_dict())
            total_time[i] += perf_counter() - start
            call_counts[i] += 1
            actions.append(action)
        obs, done = engine.step(actions)
        steps_run += 1
        if done:
            break

    snap = engine.snapshot()
    rewards = snap.rewards if snap.rewards is not None else [None] * len(agents)
    avg_ms = [
        (total_time[i] / call_counts[i] * 1000.0) if call_counts[i] else 0.0
        for i in range(len(agents))
    ]
    return rewards, steps_run, avg_ms


def main():
    parser = argparse.ArgumentParser(description="Batch tournament between two bots.")
    parser.add_argument("bot1")
    parser.add_argument("bot2")
    parser.add_argument("n_matches", type=int)
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="First seed to use; subsequent matches use start_seed+1, +2, ...",
    )
    args = parser.parse_args()

    bot_names = [args.bot1, args.bot2]
    bot_paths = [bot_entry(name) for name in bot_names]
    missing = [str(p.relative_to(ROOT)) for p in bot_paths if not p.is_file()]
    if missing:
        print("Missing bot file(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    n = args.n_matches
    wins = [0, 0]
    draws = 0
    sum_steps = 0
    sum_avg_ms = [0.0, 0.0]

    print(
        f"Running {n} matches: {bot_names[0]} vs {bot_names[1]} "
        f"(seeds {args.start_seed}..{args.start_seed + n - 1})",
        flush=True,
    )
    for k in range(n):
        seed = args.start_seed + k
        rewards, steps, avg_ms = run_one_match(bot_paths, seed, k)
        r0, r1 = rewards[0], rewards[1]
        if r0 is None or r1 is None:
            winner = "?"
        elif r0 > r1:
            wins[0] += 1
            winner = "P0"
        elif r1 > r0:
            wins[1] += 1
            winner = "P1"
        else:
            draws += 1
            winner = "draw"
        sum_steps += steps
        sum_avg_ms[0] += avg_ms[0]
        sum_avg_ms[1] += avg_ms[1]
        print(
            f"  seed={seed:>6} steps={steps:>3} r=({r0},{r1}) -> {winner}  "
            f"ms=({avg_ms[0]:.2f},{avg_ms[1]:.2f})",
            flush=True,
        )

    decided = wins[0] + wins[1]
    p0_rate = (wins[0] / decided * 100.0) if decided else 0.0
    print()
    print(f"Summary ({n} matches):")
    print(f"  {bot_names[0]}: {wins[0]} wins ({p0_rate:.1f}% of decided)  avg_ms={sum_avg_ms[0]/n:.2f}")
    print(f"  {bot_names[1]}: {wins[1]} wins  avg_ms={sum_avg_ms[1]/n:.2f}")
    print(f"  draws: {draws}")
    print(f"  avg steps/match: {sum_steps / n:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
