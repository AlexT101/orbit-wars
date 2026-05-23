import argparse
import contextlib
import importlib.util
import logging
import os
import random
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"
OPEN_SOURCE_BOTS_DIR = BOTS_DIR / "_open_source"

MAX_STEPS = 500


@contextlib.contextmanager
def _silence_imports():
    """Suppress import-time chatter from `kaggle_environments`.

    Two sources need silencing: pyspiel's C++ `load_game` writes
    "OpenSpiel exception: ..." straight to the native stderr fd (Python-level
    redirects don't catch it), and `open_spiel_env` logs an INFO summary
    through the standard `logging` module. We redirect fds 1/2 to devnull
    and disable logging for the duration.
    """
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


# Pre-import kaggle_environments under the silencer so any later bot import
# (e.g. `from kaggle_environments.envs.orbit_wars.orbit_wars import Planet`)
# hits the module cache and stays quiet.
with _silence_imports():
    try:
        import kaggle_environments  # noqa: F401
    except ModuleNotFoundError:
        pass


def bot_entry(bot_name: str) -> Path:
    candidates = [
        BOTS_DIR / bot_name / "main.py",
        OPEN_SOURCE_BOTS_DIR / bot_name / "main.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_agent(main_path: Path, mod_name: str):
    """Import a bot's main.py in-process and return its `agent` callable."""
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module.agent


def run_rust_match(bot_paths: list[Path], bot_names: list[str], seed: int) -> int:
    """Drive a match through the native Rust engine, calling each bot's
    `agent` in-process. Reports the final rewards from the engine snapshot."""

    print(f"Match (rust engine): {bot_names[0]} vs {bot_names[1]}")
        
    from parity.candidates.rust import RustEngine

    agents = [load_agent(path, f"bot_{i}_main") for i, path in enumerate(bot_paths)]
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
    print(f"Finished: done={done} steps={steps_run}")
    rewards = snap.rewards if snap.rewards is not None else [None] * len(agents)
    for i, reward in enumerate(rewards):
        avg_ms = (total_time[i] / call_counts[i] * 1000.0) if call_counts[i] else 0.0
        print(f"Player {i} ({bot_names[i]}): reward={reward}, avg_step_ms={avg_ms:.2f}")
    return 0


def run_kaggle_match(bot_paths: list[Path], bot_names: list[str], seed: int) -> int:
    """Run a match with the Kaggle reference engine via env.run()."""
    try:
        from kaggle_environments import make
    except ModuleNotFoundError:
        print('Missing dependency: install with `pip install "kaggle-environments>=1.28.0"`')
        return 1
    
    print(f"Match (kaggle engine): {bot_names[0]} vs {bot_names[1]}")

    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run([str(path) for path in bot_paths])

    n_players = len(bot_names)
    total_time = [0.0] * n_players
    call_counts = [0] * n_players
    over_budget = [0] * n_players
    act_timeout = env.configuration.actTimeout
    for step_logs in env.logs:
        if not step_logs:
            continue
        for i in range(n_players):
            if i < len(step_logs) and "duration" in step_logs[i]:
                duration = step_logs[i]["duration"]
                total_time[i] += duration
                call_counts[i] += 1
                if duration > act_timeout:
                    over_budget[i] += 1

    final = env.steps[-1]
    for i, state in enumerate(final):
        avg_ms = (total_time[i] / call_counts[i] * 1000.0) if call_counts[i] else 0.0
        print(
            f"Player {i} ({bot_names[i]}): reward={state.reward}, "
            f"status={state.status}, avg_step_ms={avg_ms:.2f}"
        )

    timed_out = any(over_budget) or any(s.status == "TIMEOUT" for s in final)
    if timed_out:
        parts = [
            f"P{i} ({bot_names[i]})={over_budget[i]}{' [KILLED]' if final[i].status == 'TIMEOUT' else ''}"
            for i in range(n_players)
        ]
        print(f"Turns over actTimeout ({act_timeout}s): " + ", ".join(parts))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single Orbit Wars match between two bots.")
    parser.add_argument("bot1")
    parser.add_argument("bot2")
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use the Kaggle reference engine (env.run). Default: native Rust engine.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the match. If omitted, a random seed is chosen and printed.",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    print(f"Seed: {seed}")

    bot_names = [args.bot1, args.bot2]
    bot_paths = [bot_entry(name) for name in bot_names]

    missing = [str(path.relative_to(ROOT)) for path in bot_paths if not path.is_file()]
    if missing:
        print("Missing bot file(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    if args.kaggle:
        return run_kaggle_match(bot_paths, bot_names, seed)
    return run_rust_match(bot_paths, bot_names, seed)


if __name__ == "__main__":
    sys.exit(main())
