"""TEMP: run a real apollo-vs-apollo match and dump aim stage counters.

Usage: python bench_match_counters.py [--seed N] [--steps N]
Reports the global aim hot-path counters (accumulated across both bots over
the whole match) plus per-bot-step averages, alongside avg_step_ms.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from time import perf_counter

import apollo_native

ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"
MAX_STEPS = 500


def load_agent(main_path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module.agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    from engine_parity_checker.candidates.rust import RustEngine

    main_path = BOTS_DIR / "mine" / "apollo" / "main.py"
    agents = [load_agent(main_path, f"bot_{i}_main") for i in range(2)]
    engine = RustEngine()
    obs = engine.reset(args.seed, len(agents))

    apollo_native.aim_counters_reset()
    total_time = [0.0, 0.0]
    call_counts = [0, 0]

    steps_run = 0
    done = False
    for _ in range(args.steps):
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

    bot_steps = sum(call_counts)
    print(f"seed={args.seed} steps={steps_run} done={done} bot_steps={bot_steps}")
    for i in range(2):
        avg_ms = (total_time[i] / call_counts[i] * 1000.0) if call_counts[i] else 0.0
        print(f"  Player {i}: avg_step_ms={avg_ms:.2f}")
    print("COUNTERS (both bots, whole match):")
    print("  " + apollo_native.aim_counters_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
