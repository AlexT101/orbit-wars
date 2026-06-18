"""Batch tournament runner. Runs N matches between four bots over a deterministic
seed range and reports win rate / average step time.

Usage:
    python run_batched_4p.py <bot1> <bot2> <bot3> <bot4> <n_matches> [--start-seed S] [--threads T]

Matches can run concurrently in worker processes. Within each worker process,
agents are re-imported per match under a fresh module name so any module-level
singletons (e.g. apollo's `_BOT = apollo_native.Bot()`) are re-created cleanly
per match.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import itertools
import importlib.util
import logging
import os
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"
MAX_STEPS = 500
N_PLAYERS = 4
SLOT_ORDERS = list(itertools.permutations(range(N_PLAYERS)))


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
    return module


def cleanup_agent_module(module):
    executor = getattr(module, "_IL_EXEC", None)
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass
    owners = [module]
    nested = getattr(module, "_aph", None)
    if nested is not None:
        owners.append(nested)
    for owner in owners:
        proc = getattr(owner, "_PROC", None)
        if proc is None:
            continue
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=1.0)
        except Exception:
            pass
        try:
            owner._PROC = None
        except Exception:
            pass


def run_one_match(bot_paths, seed, match_idx):
    from engine_parity_checker.candidates.rust import RustEngine

    modules = [
        load_agent(path, f"bot_{i}_match_{match_idx}_main")
        for i, path in enumerate(bot_paths)
    ]
    agents = [module.agent for module in modules]
    try:
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
    finally:
        for module in modules:
            cleanup_agent_module(module)


def match_winner(rewards):
    if any(reward is None for reward in rewards):
        return None

    best = max(rewards)
    winners = [i for i, reward in enumerate(rewards) if reward == best]
    if len(winners) == 1:
        return winners[0]
    return "draw"


def slot_order_for_seed(seed):
    # Seven is coprime with 24, so consecutive seeds cycle through every
    # permutation while varying the first slot immediately.
    return SLOT_ORDERS[(seed * 7) % len(SLOT_ORDERS)]


def reorder_by_input(values, slot_order):
    by_input = [None] * len(values)
    for slot_idx, bot_idx in enumerate(slot_order):
        by_input[bot_idx] = values[slot_idx]
    return by_input


def run_match_job(job):
    bot_paths, seed, match_idx, slot_order = job
    slotted_bot_paths = [bot_paths[i] for i in slot_order]
    rewards, steps, avg_ms = run_one_match(slotted_bot_paths, seed, match_idx)
    return match_idx, seed, slot_order, rewards, steps, avg_ms


def main():
    parser = argparse.ArgumentParser(description="Batch tournament between four bots.")
    parser.add_argument("bot1")
    parser.add_argument("bot2")
    parser.add_argument("bot3")
    parser.add_argument("bot4")
    parser.add_argument("n_matches", type=int)
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="First seed to use; subsequent matches use start_seed+1, +2, ...",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of matches to run concurrently. Use 1 for sequential execution.",
    )
    args = parser.parse_args()

    if args.n_matches < 1:
        parser.error("n_matches must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    bot_names = [args.bot1, args.bot2, args.bot3, args.bot4]
    bot_paths = [bot_entry(name) for name in bot_names]
    missing = [str(p.relative_to(ROOT)) for p in bot_paths if not p.is_file()]
    if missing:
        print("Missing bot file(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    n = args.n_matches
    threads = min(args.threads, n)
    wins = [0] * N_PLAYERS
    draws = 0
    unknown = 0
    sum_steps = 0
    sum_avg_ms = [0.0] * N_PLAYERS

    def record_result(result):
        nonlocal draws, unknown, sum_steps

        _match_idx, seed, slot_order, rewards, steps, avg_ms = result
        rewards_by_input = reorder_by_input(rewards, slot_order)
        avg_ms_by_input = reorder_by_input(avg_ms, slot_order)

        winner = match_winner(rewards_by_input)
        if winner is None:
            winner_label = "?"
            unknown += 1
        elif winner == "draw":
            winner_label = "draw"
            draws += 1
        else:
            wins[winner] += 1
            winner_label = f"P{winner}"

        sum_steps += steps
        for i, ms in enumerate(avg_ms_by_input):
            sum_avg_ms[i] += ms

        rewards_str = ",".join(str(reward) for reward in rewards_by_input)
        avg_ms_str = ",".join(f"{ms:.2f}" for ms in avg_ms_by_input)
        print(
            f"  seed={seed:>6} steps={steps:>3} r=({rewards_str}) -> {winner_label}  "
            f"ms=({avg_ms_str})"
        )

    matchup = " vs ".join(bot_names)
    print(
        f"Running {n} matches: {matchup} "
        f"(seeds {args.start_seed}..{args.start_seed + n - 1}, threads={threads})"
    )

    jobs = [
        (bot_paths, args.start_seed + k, k, slot_order_for_seed(args.start_seed + k))
        for k in range(n)
    ]
    if threads == 1:
        for job in jobs:
            record_result(run_match_job(job))
    else:
        results_by_idx = {}
        next_to_print = 0
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(run_match_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                match_idx = result[0]
                results_by_idx[match_idx] = result
                while next_to_print in results_by_idx:
                    record_result(results_by_idx.pop(next_to_print))
                    next_to_print += 1

    decided = sum(wins)
    print()
    print(f"Summary ({n} matches):")
    for i, name in enumerate(bot_names):
        win_rate = (wins[i] / decided * 100.0) if decided else 0.0
        print(
            f"  {name}: {wins[i]} wins ({win_rate:.1f}% of decided)  "
            f"avg_ms={sum_avg_ms[i] / n:.2f}"
        )
    print(f"  draws: {draws}")
    if unknown:
        print(f"  unknown: {unknown}")
    print(f"  avg steps/match: {sum_steps / n:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
