"""Benchmark: Rust bot vs Python bot, both driven through the Rust engine.

Runs N full games for two self-matchups, all stepped by the native Rust
simulator (`parity.candidates.rust.RustEngine`):

  * rust   : both players use bots/nearest-sniper-rust   (rust_bot_native)
  * python : both players use bots/_open_source/nearest-sniper (pure Python)

Because the simulator is identical in both runs, the wall-clock difference
between the two matchups is the cost of agent decision-making alone. We also
time the agent calls separately so the engine cost can be factored out.

Usage:
    python bench_bots.py [--games 10] [--max-steps 500] [--seed 42]
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

from parity.candidates.rust import RustEngine

ROOT = Path(__file__).resolve().parent
RUST_BOT_MAIN = ROOT / "bots" / "nearest-sniper-rust" / "main.py"
PYTHON_BOT_MAIN = ROOT / "bots" / "_open_source" / "nearest-sniper" / "main.py"


def load_agent(main_path: Path, mod_name: str):
    """Import a bot's main.py and return its `agent` callable."""
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def run_game(engine: RustEngine, agent, seed: int, num_players: int, max_steps: int):
    """Play one game; return (steps_run, total_agent_seconds)."""
    obs = engine.reset(seed, num_players)
    agent_seconds = 0.0
    steps = 0
    for _ in range(max_steps):
        t0 = time.perf_counter()
        actions = [agent(obs[i].as_dict()) for i in range(num_players)]
        agent_seconds += time.perf_counter() - t0

        obs, done = engine.step(actions)
        steps += 1
        if done:
            break
    return steps, agent_seconds


def bench_matchup(label: str, agent, games: int, max_steps: int, base_seed: int):
    engine = RustEngine()
    # Warm up once so first-call import/JIT costs don't skew the timing.
    run_game(engine, agent, base_seed, 2, max_steps)

    total_steps = 0
    total_agent = 0.0
    t0 = time.perf_counter()
    for g in range(games):
        steps, agent_s = run_game(engine, agent, base_seed + g, 2, max_steps)
        total_steps += steps
        total_agent += agent_s
    wall = time.perf_counter() - t0

    print(f"\n=== {label} bot vs itself ({games} games) ===")
    print(f"  total wall time : {wall:.4f} s")
    print(f"  total steps     : {total_steps}")
    print(f"  agent time      : {total_agent:.4f} s  ({100 * total_agent / wall:.1f}% of wall)")
    print(f"  engine time     : {wall - total_agent:.4f} s")
    print(f"  per-game wall    : {1000 * wall / games:.2f} ms")
    print(f"  per-step agent   : {1e6 * total_agent / total_steps:.2f} us")
    return wall, total_agent, total_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rust_agent = load_agent(RUST_BOT_MAIN, "rust_bot_main")
    python_agent = load_agent(PYTHON_BOT_MAIN, "python_bot_main")

    print(f"Engine: Rust simulator (RustEngine)  |  games={args.games}  "
          f"max_steps={args.max_steps}  seed={args.seed}")

    rust_wall, rust_agent_s, rust_steps = bench_matchup(
        "RUST", rust_agent, args.games, args.max_steps, args.seed)
    py_wall, py_agent_s, py_steps = bench_matchup(
        "PYTHON", python_agent, args.games, args.max_steps, args.seed)

    print("\n=== comparison ===")
    if rust_steps != py_steps:
        print(f"  NOTE: step counts differ (rust={rust_steps}, python={py_steps}); "
              f"comparing per-step rates.")
    print(f"  wall time  : python {py_wall:.4f}s  vs  rust {rust_wall:.4f}s "
          f"->  {py_wall / rust_wall:.2f}x")
    print(f"  agent time : python {py_agent_s:.4f}s  vs  rust {rust_agent_s:.4f}s "
          f"->  {py_agent_s / rust_agent_s:.2f}x faster decisions in Rust")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
