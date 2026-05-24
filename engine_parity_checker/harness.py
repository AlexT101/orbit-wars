"""Lockstep parity loop.

Drives two engines with identical (seed, action) sequences and reports the
first turn at which their canonical state diverges. Actions are computed
from engine A's observation and applied to *both* engines so that the
action stream itself never diverges; only true engine differences show up.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass

from parity.agents import AGENTS, AgentFn
from parity.diff import Diff, diff_snapshots, format_diff
from parity.engine import Engine, Snapshot
from parity.kaggle_engine import KaggleEngine


@dataclass
class ParityResult:
    converged: bool
    steps_run: int
    first_divergence_step: int | None
    first_divergence: list[Diff]
    final_snapshot_a: Snapshot
    final_snapshot_b: Snapshot
    elapsed_seconds: float

    def summary(self) -> str:
        head = (
            f"steps={self.steps_run}  elapsed={self.elapsed_seconds:.2f}s  "
            f"rate={self.steps_run / max(self.elapsed_seconds, 1e-9):.1f} steps/s"
        )
        if self.converged:
            return f"PARITY OK — {head}"
        return (
            f"DIVERGED at step {self.first_divergence_step} — {head}\n"
            + format_diff(self.first_divergence)
        )


def run_parity(
    engine_a: Engine,
    engine_b: Engine,
    seed: int = 42,
    num_players: int = 2,
    agents: list[AgentFn] | None = None,
    max_steps: int = 500,
    atol: float = 0.0,
    agent_seed: int = 0,
    stop_on_divergence: bool = True,
    verbose: bool = False,
) -> ParityResult:
    """Run both engines in lockstep. Returns the first divergence (if any)
    and the final snapshots from each engine."""

    if agents is None:
        agents = [AGENTS["random"]] * num_players
    assert len(agents) == num_players, (
        f"need {num_players} agents, got {len(agents)}"
    )

    obs_a = engine_a.reset(seed, num_players)
    obs_b = engine_b.reset(seed, num_players)

    # Sanity-diff right after reset — if seeds disagree here, the engines
    # have different RNG semantics and there's no point stepping further.
    snap_a, snap_b = engine_a.snapshot(), engine_b.snapshot()
    diffs = diff_snapshots(snap_a, snap_b, atol=atol)
    if diffs and stop_on_divergence:
        return ParityResult(
            converged=False,
            steps_run=0,
            first_divergence_step=0,
            first_divergence=diffs,
            final_snapshot_a=snap_a,
            final_snapshot_b=snap_b,
            elapsed_seconds=0.0,
        )

    # Per-player agent RNGs so different players don't share streams.
    rngs = [random.Random(agent_seed + 1000 * i) for i in range(num_players)]

    first_div_step: int | None = None
    first_div: list[Diff] = []
    t0 = time.perf_counter()
    steps_run = 0

    for step_idx in range(1, max_steps + 1):
        # Actions are computed from engine A's view, then sent to both
        # engines. This isolates engine differences from action drift.
        actions = [
            agents[i](obs_a[i].as_dict(), rngs[i]) for i in range(num_players)
        ]

        obs_a, done_a = engine_a.step(actions)
        obs_b, done_b = engine_b.step(actions)
        steps_run = step_idx

        snap_a, snap_b = engine_a.snapshot(), engine_b.snapshot()
        diffs = diff_snapshots(snap_a, snap_b, atol=atol)
        if diffs and first_div_step is None:
            first_div_step = step_idx
            first_div = diffs
            if verbose:
                print(f"[step {step_idx}] diverged: {format_diff(diffs, 5)}")
            if stop_on_divergence:
                break

        if done_a or done_b:
            if done_a != done_b and first_div_step is None:
                first_div_step = step_idx
                first_div = [
                    Diff("done", done_a, done_b, "one engine ended early")
                ]
            break

        if verbose and step_idx % 50 == 0:
            print(f"[step {step_idx}] ok  planets={len(snap_a.planets)}  fleets={len(snap_a.fleets)}")

    elapsed = time.perf_counter() - t0
    return ParityResult(
        converged=(first_div_step is None),
        steps_run=steps_run,
        first_divergence_step=first_div_step,
        first_divergence=first_div,
        final_snapshot_a=snap_a,
        final_snapshot_b=snap_b,
        elapsed_seconds=elapsed,
    )


# --- CLI ------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Run parity check between Kaggle and a candidate engine."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--players", type=int, default=2, choices=[2, 4])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--agent",
        choices=list(AGENTS),
        default="random",
        help="Scripted agent used for every player.",
    )
    parser.add_argument(
        "--candidate",
        choices=["kaggle", "rust"],
        default="kaggle",
        help="Candidate engine. 'kaggle' = self-parity sanity check; "
        "'rust' = PyO3-backed native candidate.",
    )
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    engine_a: Engine = KaggleEngine()
    if args.candidate == "kaggle":
        engine_b: Engine = KaggleEngine()
    elif args.candidate == "rust":
        from parity.candidates.rust import RustEngine

        engine_b = RustEngine()
    else:
        raise ValueError(args.candidate)

    result = run_parity(
        engine_a=engine_a,
        engine_b=engine_b,
        seed=args.seed,
        num_players=args.players,
        agents=[AGENTS[args.agent]] * args.players,
        max_steps=args.steps,
        atol=args.atol,
        agent_seed=args.agent_seed,
        verbose=args.verbose,
    )
    print(result.summary())
    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(_cli())
