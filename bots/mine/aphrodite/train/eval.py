"""Evaluate aphrodite (optionally with trained value-net weights) vs a fixed opponent set.

Runs both bot sides on each seed and reports per-pairing win/loss/tie tallies
and average step-ms.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BOTS_DIR = ROOT / "bots"
APHRODITE_DIR = ROOT / "bots" / "mine" / "aphrodite"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import collector to reuse the per-process daemon driver.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # type: ignore


def run_one(bot0: str, bot1: str, seed: int, budget_ms: int, weights_path: Path | None):
    from engine_parity_checker.candidates.rust import RustEngine

    agent_funcs: list = [None, None]
    closers: list = []
    for i, name in enumerate([bot0, bot1]):
        if name == "aphrodite":
            d = collect.AphroditeDaemon(
                dump_path=None,
                budget_ms=budget_ms,
                weights_path=weights_path,
            )
            agent_funcs[i] = d
            closers.append(d.close)
        else:
            fn, mod = collect.load_other_agent(name)
            agent_funcs[i] = fn
            closers.append(lambda m=mod: collect.teardown_other(m))

    engine = RustEngine()
    obs = engine.reset(seed, 2)
    done = False
    import time

    times = [0.0, 0.0]
    calls = [0, 0]
    for _ in range(collect.MAX_STEPS):
        acts = []
        for i in range(2):
            t0 = time.perf_counter()
            a = agent_funcs[i](obs[i].as_dict())
            times[i] += time.perf_counter() - t0
            calls[i] += 1
            acts.append(a)
        obs, done = engine.step(acts)
        if done:
            break
    snap = engine.snapshot()
    rewards = snap.rewards or [0.0, 0.0]
    for c in closers:
        try:
            c()
        except Exception:
            pass
    avg_ms = [(times[i] / max(calls[i], 1)) * 1000.0 for i in range(2)]
    return rewards, avg_ms


def run_job(job):
    """One match worker (top-level so it is picklable for ProcessPoolExecutor).

    job = (opp, seed, as_p0, budget_ms, weights). Returns the result framed
    from aphrodite's perspective so the parent can tally regardless of side.
    """
    opp, seed, as_p0, budget_ms, weights = job
    if as_p0:
        rewards, avg_ms = run_one("aphrodite", opp, seed, budget_ms, weights)
        r_alpha, r_opp, ms_alpha, ms_opp = rewards[0], rewards[1], avg_ms[0], avg_ms[1]
    else:
        rewards, avg_ms = run_one(opp, "aphrodite", seed, budget_ms, weights)
        r_alpha, r_opp, ms_alpha, ms_opp = rewards[1], rewards[0], avg_ms[1], avg_ms[0]
    outcome = "W" if r_alpha > r_opp else ("L" if r_alpha < r_opp else "T")
    return (opp, seed, as_p0, outcome, r_alpha, r_opp, ms_alpha, ms_opp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--opponents", nargs="+", default=["heuristic", "apollo_fast"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42, 100, 2025])
    p.add_argument("--budget-ms", type=int, default=500)
    p.add_argument("--swap", action=argparse.BooleanOptionalAction, default=True,
                   help="play both sides of each seed (--no-swap = aphrodite as p0 only, unique seeds)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 4),
                   help="matches to run concurrently in worker processes. Each match uses "
                        "~2 cores (aphrodite + opponent), so keep <= cores/2. NOTE: under "
                        "contention the per-turn ms (and thus aphrodite's search depth at a "
                        "fixed budget) degrade — use --threads 1 for accurate timing/strength.")
    args = p.parse_args()

    # Pass weights as a resolved string: picklable and unambiguous across workers.
    weights = str(Path(args.weights).resolve()) if args.weights else None
    sides = [True, False] if args.swap else [True]
    jobs = [
        (opp, seed, as_p0, args.budget_ms, weights)
        for opp in args.opponents
        for seed in args.seeds
        for as_p0 in sides
    ]
    threads = max(1, min(args.threads, len(jobs)))
    print(f"aphrodite budget={args.budget_ms}ms weights={weights} "
          f"jobs={len(jobs)} threads={threads}"
          + ("  [parallel: ms/strength may degrade under contention]" if threads > 1 else ""),
          flush=True)

    results = []
    if threads == 1:
        for job in jobs:
            results.append(run_job(job))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futures = [ex.submit(run_job, j) for j in jobs]
            for fut in as_completed(futures):
                results.append(fut.result())

    # Stable per-seed printout, grouped by opponent in the requested order.
    results.sort(key=lambda r: (args.opponents.index(r[0]), r[1], not r[2]))
    for opp in args.opponents:
        rs = [r for r in results if r[0] == opp]
        wins = sum(r[3] == "W" for r in rs)
        losses = sum(r[3] == "L" for r in rs)
        ties = sum(r[3] == "T" for r in rs)
        ms_alpha = [r[6] for r in rs]
        ms_opp = [r[7] for r in rs]
        for opp_, seed, as_p0, outcome, r_alpha, r_opp, ma, mo in rs:
            print(f"   seed={seed} as_p0={as_p0} {outcome} "
                  f"rewards=[{r_alpha},{r_opp}] ms=[{ma:.1f},{mo:.1f}]", flush=True)
        n = wins + losses + ties
        print(
            f"vs {opp}: {wins}W/{losses}L/{ties}T (n={n}) | "
            f"aphrodite avg {np.mean(ms_alpha):.1f}ms, opp avg {np.mean(ms_opp):.2f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
