"""Evaluate alphaow (optionally with trained value-net weights) vs a fixed opponent set.

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
ALPHAOW_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import collector to reuse the per-process daemon driver.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # type: ignore
from model_dashboard import append_eval_dashboard


def run_one(bot0: str, bot1: str, seed: int, budget_ms: int, weights_path: Path | None):
    from engine_parity_checker.candidates.rust import RustEngine

    agent_funcs: list = [None, None]
    closers: list = []
    for i, name in enumerate([bot0, bot1]):
        if name == "alphaow":
            d = collect.AlphaowDaemon(
                dump_path=None,
                budget_ms=budget_ms,
                weights_path=weights_path,
                value_net_off=(weights_path is None),
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--opponents", nargs="+", default=["heuristic", "apollo_fast"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42, 100, 2025])
    p.add_argument("--budget-ms", type=int, default=500)
    p.add_argument("--swap", action="store_true", default=True, help="play both sides")
    p.add_argument("--dashboard-html", default=None, help="update a self-contained HTML dashboard")
    args = p.parse_args()

    weights = Path(args.weights) if args.weights else None
    print(f"alphaow budget={args.budget_ms}ms weights={weights}")
    summaries = []
    for opp in args.opponents:
        wins = losses = ties = 0
        ms_alpha = []
        ms_opp = []
        reward_diffs = []
        side_counts = {"p0": [0, 0, 0], "p1": [0, 0, 0]}  # W/L/T
        for seed in args.seeds:
            for as_p0 in ([True, False] if args.swap else [True]):
                if as_p0:
                    rewards, avg_ms = run_one("alphaow", opp, seed, args.budget_ms, weights)
                    r_alpha, r_opp = rewards[0], rewards[1]
                    ms_alpha.append(avg_ms[0])
                    ms_opp.append(avg_ms[1])
                else:
                    rewards, avg_ms = run_one(opp, "alphaow", seed, args.budget_ms, weights)
                    r_alpha, r_opp = rewards[1], rewards[0]
                    ms_alpha.append(avg_ms[1])
                    ms_opp.append(avg_ms[0])
                outcome = "W" if r_alpha > r_opp else ("L" if r_alpha < r_opp else "T")
                reward_diffs.append(float(r_alpha - r_opp))
                side_key = "p0" if as_p0 else "p1"
                if outcome == "W":
                    wins += 1
                    side_counts[side_key][0] += 1
                elif outcome == "L":
                    losses += 1
                    side_counts[side_key][1] += 1
                else:
                    ties += 1
                    side_counts[side_key][2] += 1
                print(
                    f"   seed={seed} as_p0={as_p0} {outcome} rewards={rewards} ms=[{avg_ms[0]:.1f},{avg_ms[1]:.1f}]",
                    flush=True,
                )
        n = wins + losses + ties
        wr = (wins + 0.5 * ties) / max(n, 1)
        summaries.append(
            dict(
                opponent=opp,
                n=n,
                wins=wins,
                losses=losses,
                ties=ties,
                score=wr,
                reward_diff=float(np.mean(reward_diffs)) if reward_diffs else 0.0,
                reward_diff_std=float(np.std(reward_diffs)) if reward_diffs else 0.0,
                alpha_ms=float(np.mean(ms_alpha)) if ms_alpha else 0.0,
                alpha_ms_p95=float(np.percentile(ms_alpha, 95)) if ms_alpha else 0.0,
                opp_ms=float(np.mean(ms_opp)) if ms_opp else 0.0,
                side_counts=side_counts,
            )
        )
        print(
            f"vs {opp}: {wins}W/{losses}L/{ties}T (n={n}) | alphaow avg {np.mean(ms_alpha):.1f}ms, opp avg {np.mean(ms_opp):.2f}ms"
        )
    if summaries:
        print("\n=== eval dashboard ===")
        print("opponent        n   score    W-L-T     dReward±sd       ms_avg/p95   side p0 W-L-T | p1 W-L-T")
        for s in sorted(summaries, key=lambda x: -x["score"]):
            p0 = s["side_counts"]["p0"]
            p1 = s["side_counts"]["p1"]
            print(
                f"{s['opponent'][:14]:14s} {s['n']:3d} {100*s['score']:6.1f}% "
                f"{s['wins']:2d}-{s['losses']:2d}-{s['ties']:<2d} "
                f"{s['reward_diff']:+8.3f}±{s['reward_diff_std']:<6.3f} "
                f"{s['alpha_ms']:7.1f}/{s['alpha_ms_p95']:<6.1f} "
                f"{p0[0]:2d}-{p0[1]:2d}-{p0[2]:<2d} | {p1[0]:2d}-{p1[1]:2d}-{p1[2]:<2d}"
            )
        if args.dashboard_html:
            append_eval_dashboard(
                args.dashboard_html,
                title=f"eval budget={args.budget_ms}ms weights={weights}",
                summaries=summaries,
            )
            print(f"dashboard_html={args.dashboard_html}")


if __name__ == "__main__":
    main()
