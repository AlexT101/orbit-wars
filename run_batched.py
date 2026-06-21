"""Batch tournament runner. Runs N matches between two bots over a deterministic
seed range and reports win rate / average step time.

Usage:
    python run_batched.py <bot1> <bot2> <n_matches> [--start-seed S] [--threads T]

Matches can run concurrently in worker processes. Within each worker process,
agents are re-imported per match under a fresh module name so any module-level
singletons (e.g. apollo's `_BOT = apollo_native.Bot()`) are re-created cleanly
per match.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"
MAX_STEPS = 500
SLOT_ORDERS = ((0, 1), (1, 0))


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


def run_one_match(bot_paths, seed, match_idx, timing_config=None):
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
        timing_records = []
        track_timing = bool(timing_config)
        act_timeout_s = float((timing_config or {}).get("act_timeout_s", 1.0))
        overage_s = [
            float((timing_config or {}).get("initial_overage_s", 60.0))
            for _ in agents
        ]
        config = {
            "episodeSteps": MAX_STEPS,
            "actTimeout": act_timeout_s,
        }

        done = False
        steps_run = 0
        for _ in range(MAX_STEPS):
            actions = []
            for i in range(len(agents)):
                obs_dict = obs[i].as_dict()
                if track_timing:
                    obs_dict["remainingOverageTime"] = overage_s[i]
                    obs_dict["config"] = dict(config)
                before_overage_s = overage_s[i]
                start = perf_counter()
                action = agents[i](obs_dict)
                duration_s = perf_counter() - start
                total_time[i] += duration_s
                call_counts[i] += 1
                if track_timing:
                    overage_s[i] = max(0.0, overage_s[i] - max(0.0, duration_s - act_timeout_s))
                    timing_records.append(
                        {
                            "match_idx": match_idx,
                            "seed": seed,
                            "step": int(obs[i].step),
                            "slot": i,
                            "duration_ms": duration_s * 1000.0,
                            "act_timeout_ms": act_timeout_s * 1000.0,
                            "remaining_overage_before_s": before_overage_s,
                            "remaining_overage_after_s": overage_s[i],
                            "over_budget": duration_s > act_timeout_s,
                            "moves": len(action or []),
                        }
                    )
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
        return rewards, steps_run, avg_ms, timing_records
    finally:
        for module in modules:
            cleanup_agent_module(module)


def slot_order_for_seed(seed):
    return SLOT_ORDERS[seed % len(SLOT_ORDERS)]


def reorder_by_input(values, slot_order):
    by_input = [None] * len(values)
    for slot_idx, bot_idx in enumerate(slot_order):
        by_input[bot_idx] = values[slot_idx]
    return by_input


def run_match_job(job):
    bot_paths, seed, match_idx, slot_order, timing_config = job
    slotted_bot_paths = [bot_paths[i] for i in slot_order]
    rewards, steps, avg_ms, timing_records = run_one_match(
        slotted_bot_paths, seed, match_idx, timing_config
    )
    return match_idx, seed, slot_order, rewards, steps, avg_ms, timing_records


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
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of matches to run concurrently. Use 1 for sequential execution.",
    )
    parser.add_argument(
        "--timing-jsonl",
        type=Path,
        default=None,
        help="Write per-agent call duration/overage records as JSONL.",
    )
    parser.add_argument(
        "--initial-overage",
        type=float,
        default=60.0,
        help="Initial synthetic remainingOverageTime in seconds for --timing-jsonl.",
    )
    parser.add_argument(
        "--act-timeout",
        type=float,
        default=1.0,
        help="Synthetic actTimeout in seconds for --timing-jsonl.",
    )
    args = parser.parse_args()

    if args.n_matches < 1:
        parser.error("n_matches must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    bot_names = [args.bot1, args.bot2]
    bot_paths = [bot_entry(name) for name in bot_names]
    missing = [str(p.relative_to(ROOT)) for p in bot_paths if not p.is_file()]
    if missing:
        print("Missing bot file(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    n = args.n_matches
    threads = min(args.threads, n)
    wins = [0, 0]
    draws = 0
    sum_steps = 0
    sum_avg_ms = [0.0, 0.0]
    timing_out = None
    if args.timing_jsonl is not None:
        args.timing_jsonl.parent.mkdir(parents=True, exist_ok=True)
        timing_out = args.timing_jsonl.open("w", encoding="utf-8")
    timing_config = None
    if args.timing_jsonl is not None:
        timing_config = {
            "initial_overage_s": args.initial_overage,
            "act_timeout_s": args.act_timeout,
        }

    def record_result(result):
        nonlocal draws, sum_steps

        _match_idx, seed, slot_order, rewards, steps, avg_ms, timing_records = result
        rewards = reorder_by_input(rewards, slot_order)
        avg_ms = reorder_by_input(avg_ms, slot_order)
        if timing_out is not None:
            for record in timing_records:
                record["input_bot"] = slot_order[record["slot"]]
                timing_out.write(json.dumps(record, separators=(",", ":")) + "\n")
            timing_out.flush()
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
            f"ms=({avg_ms[0]:.2f},{avg_ms[1]:.2f})"
        )

    print(
        f"Running {n} matches: {bot_names[0]} vs {bot_names[1]} "
        f"(seeds {args.start_seed}..{args.start_seed + n - 1}, threads={threads})"
    )

    jobs = [
        (
            bot_paths,
            args.start_seed + k,
            k,
            slot_order_for_seed(args.start_seed + k),
            timing_config,
        )
        for k in range(n)
    ]
    try:
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
    finally:
        if timing_out is not None:
            timing_out.close()

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
