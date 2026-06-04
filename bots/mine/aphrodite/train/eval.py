"""Evaluate aphrodite (optionally with trained value-net weights) vs a fixed opponent set.

2p: aphrodite vs one opponent; --swap plays both sides of each seed.
4p: aphrodite vs three copies of the opponent. Seat order matters in 4p (it
decides who you spawn next to), so seats are shuffled by a deterministic,
seed-derived permutation and results are normalized back to aphrodite's
perspective before tallying (internal only). Reports per-opponent W/L/T and
average step-ms.
"""

from __future__ import annotations

import argparse
import itertools
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


# Deterministic seat permutations for 4p. 7 is coprime with 24, so consecutive
# seeds cycle through every permutation while varying the first slot immediately
# (mirrors run_batched_4p.slot_order_for_seed).
_PERMS_4P = list(itertools.permutations(range(4)))


def slot_order_for_seed(seed: int) -> tuple[int, ...]:
    return _PERMS_4P[(seed * 7) % len(_PERMS_4P)]


def run_match(slot_bots, seed: int, budget_ms: int, weights_path):
    """Run one match with bots placed in the given engine-slot order.

    Returns (rewards, avg_ms) indexed by ENGINE SLOT. The caller is
    responsible for mapping back to input/perspective order.
    """
    from engine_parity_checker.candidates.rust import RustEngine
    import time

    n = len(slot_bots)
    agent_funcs: list = [None] * n
    closers: list = []
    for i, name in enumerate(slot_bots):
        if name == "aphrodite":
            d = collect.AphroditeDaemon(dump_path=None, budget_ms=budget_ms, weights_path=weights_path)
            agent_funcs[i] = d
            closers.append(d.close)
        else:
            fn, mod = collect.load_other_agent(name)
            agent_funcs[i] = fn
            closers.append(lambda m=mod: collect.teardown_other(m))

    engine = RustEngine()
    obs = engine.reset(seed, n)
    times = [0.0] * n
    calls = [0] * n
    for _ in range(collect.MAX_STEPS):
        acts = []
        for i in range(n):
            t0 = time.perf_counter()
            a = agent_funcs[i](obs[i].as_dict())
            times[i] += time.perf_counter() - t0
            calls[i] += 1
            acts.append(a)
        obs, done = engine.step(acts)
        if done:
            break
    snap = engine.snapshot()
    rewards = snap.rewards or [0.0] * n
    for c in closers:
        try:
            c()
        except Exception:
            pass
    avg_ms = [(times[i] / max(calls[i], 1)) * 1000.0 for i in range(n)]
    return rewards, avg_ms


def run_job(job):
    """One match worker (top-level so it is picklable for ProcessPoolExecutor).

    job = (opp, seed, variant, budget_ms, weights, n_players). aphrodite is
    input bot 0; the other (n_players - 1) seats are the opponent. Seats are
    permuted (2p: swap; 4p: seed-derived) then results normalized back to
    aphrodite's perspective. Returns (opp, seed, outcome, r_alpha, ms_alpha,
    ms_opp, side) where outcome is W (sole top) / T (tied top) / L.
    """
    opp, seed, variant, budget_ms, weights, n_players = job
    input_bots = ["aphrodite"] + [opp] * (n_players - 1)
    if n_players == 2:
        slot_order = (0, 1) if variant else (1, 0)
    else:
        slot_order = slot_order_for_seed(seed)
    slotted = [input_bots[i] for i in slot_order]

    rewards, avg_ms = run_match(slotted, seed, budget_ms, weights)

    aphro_slot = list(slot_order).index(0)  # engine slot aphrodite played
    rvals = [float(x) for x in rewards]
    r_alpha = rvals[aphro_slot]
    best = max(rvals)
    n_best = sum(1 for x in rvals if x == best)
    if r_alpha == best:
        outcome = "W" if n_best == 1 else "T"
    else:
        outcome = "L"
    ms_alpha = avg_ms[aphro_slot]
    others = [avg_ms[s] for s in range(n_players) if s != aphro_slot]
    ms_opp = sum(others) / len(others) if others else 0.0
    return (opp, seed, outcome, r_alpha, ms_alpha, ms_opp, f"p{aphro_slot}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--players", type=int, choices=(2, 4), default=2,
                   help="2p: aphrodite vs opponent. 4p: aphrodite vs 3x opponent, seats seed-shuffled.")
    p.add_argument("--opponents", nargs="+", default=["heuristic", "apollo_fast"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42, 100, 2025])
    p.add_argument("--budget-ms", type=int, default=500)
    p.add_argument("--swap", action=argparse.BooleanOptionalAction, default=True,
                   help="2p only: play both sides of each seed (--no-swap = aphrodite as p0 only). "
                        "Ignored in 4p, where seats are always shuffled by seed.")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 4),
                   help="matches to run concurrently in worker processes. Each match uses "
                        "~2 cores (aphrodite + opponent), so keep <= cores/2. NOTE: under "
                        "contention the per-turn ms (and thus aphrodite's search depth at a "
                        "fixed budget) degrade — use --threads 1 for accurate timing/strength.")
    args = p.parse_args()

    # Pass weights as a resolved string: picklable and unambiguous across workers.
    weights = str(Path(args.weights).resolve()) if args.weights else None
    if args.players == 2:
        variants = [True, False] if args.swap else [True]
    else:
        variants = [None]  # 4p seats come from the seed permutation
    jobs = [
        (opp, seed, variant, args.budget_ms, weights, args.players)
        for opp in args.opponents
        for seed in args.seeds
        for variant in variants
    ]
    threads = max(1, min(args.threads, len(jobs)))
    matchup = "vs opp" if args.players == 2 else "vs 3x opp (seats seed-shuffled)"
    print(f"aphrodite {args.players}p {matchup} budget={args.budget_ms}ms weights={weights} "
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
    results.sort(key=lambda r: (args.opponents.index(r[0]), r[1], r[6]))
    for opp in args.opponents:
        rs = [r for r in results if r[0] == opp]
        wins = sum(r[2] == "W" for r in rs)
        losses = sum(r[2] == "L" for r in rs)
        ties = sum(r[2] == "T" for r in rs)
        ms_alpha = [r[4] for r in rs]
        ms_opp = [r[5] for r in rs]
        for opp_, seed, outcome, r_alpha, ma, mo, side in rs:
            print(f"   seed={seed} {side} {outcome} "
                  f"aphro_r={r_alpha} ms=[{ma:.1f},{mo:.1f}]", flush=True)
        n = wins + losses + ties
        print(
            f"vs {opp}: {wins}W/{losses}L/{ties}T (n={n}) | "
            f"aphrodite avg {np.mean(ms_alpha):.1f}ms, opp avg {np.mean(ms_opp):.2f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
