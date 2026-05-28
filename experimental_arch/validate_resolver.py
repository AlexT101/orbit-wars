"""Validate the resolver paths used by experimental_arch.

Three resolution paths exist:
  1. env-rollout (resolve_via_env_rollout)     — ground truth, used during
     training via OrbitWarsDuelEnv's per-step `_resolved_cache`.
  2. interpreter-from-obs (_resolve_via_interpreter) — obs-only fallback
     used by the exported Kaggle bot; exact except for comet-spawn RNG.
  3. cached path (obs["_resolved"]) — shortcut that reads pre-computed
     ground truth set by OrbitWarsDuelEnv.

This validator asserts that the "cached" path matches ground truth (it
must — they're the same data) and reports how often the interpreter
fallback differs (it should differ only when the rollout crosses a comet
spawn boundary at step 50/150/250/350/450).

Usage:
    python experimental_arch/validate_resolver.py --seeds 4 --states-per-seed 5
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kaggle_environments as ke  # noqa: E402

from experimental_arch.orbit_wars_rl.features import (  # noqa: E402
    resolve_all_planets,
    resolve_via_env_rollout,
    _resolve_via_interpreter,
)


def random_moves_for(obs, player: int) -> list[list[float]]:
    """Tiny random-launcher to generate fleets-in-flight for validation."""
    planets = obs.get("planets", []) or []
    own = [p for p in planets if int(p[1]) == player and int(p[5]) >= 8]
    others = [p for p in planets if int(p[1]) != player]
    if not own or not others:
        return []
    moves = []
    for src in own[:2]:
        tgt = random.choice(others)
        ang = math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
        send = max(1, int(0.5 * float(src[5])))
        moves.append([int(src[0]), float(ang), int(send)])
    return moves


def make_env_with_fleets(seed: int, warmup_steps: int = 30):
    env = ke.make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(2)
    for _ in range(warmup_steps):
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation
        env.step([random_moves_for(obs0, 0), random_moves_for(obs1, 1)])
        if env.done:
            break
    return env


def compare_one_state(env) -> dict:
    """Compare the three resolution paths at a single env state."""
    obs = dict(env.state[0].observation)
    obs.setdefault("step", env.steps and len(env.steps) - 1)

    truth = resolve_via_env_rollout(env)
    obs_with_cache = dict(obs)
    obs_with_cache["_resolved"] = truth
    cached = resolve_all_planets(obs_with_cache)
    interp = _resolve_via_interpreter(obs)

    n = 0
    cache_mm_owner = cache_mm_ship = 0
    interp_mm_owner = interp_mm_ship = 0
    interp_errs = []
    cache_errs = []
    crossed_spawn = False
    starting_step = int(obs.get("step", 0) or 0)
    for pid, (o_true, s_true, t_resolve) in truth.items():
        n += 1
        if (starting_step + t_resolve) // 50 > starting_step // 50:
            # Rollout crossed a step%50 boundary.
            for k in (50, 150, 250, 350, 450):
                if starting_step < k <= starting_step + t_resolve:
                    crossed_spawn = True
        if pid in cached:
            o_c, s_c, _ = cached[pid]
            if o_c != o_true:
                cache_mm_owner += 1
            err = abs(s_c - s_true)
            cache_errs.append(err)
            if err > 1.5:
                cache_mm_ship += 1
        if pid in interp:
            o_i, s_i, _ = interp[pid]
            if o_i != o_true:
                interp_mm_owner += 1
            err = abs(s_i - s_true)
            interp_errs.append(err)
            if err > 1.5:
                interp_mm_ship += 1
    return {
        "n_planets": n,
        "n_in_flight_fleets": len(obs.get("fleets", []) or []),
        "rollout_steps": max((t for (_, _, t) in truth.values()), default=0),
        "crossed_spawn": crossed_spawn,
        "cache_owner_mm": cache_mm_owner,
        "cache_ship_mm": cache_mm_ship,
        "cache_max_err": max(cache_errs) if cache_errs else 0.0,
        "interp_owner_mm": interp_mm_owner,
        "interp_ship_mm": interp_mm_ship,
        "interp_max_err": max(interp_errs) if interp_errs else 0.0,
        "interp_mean_err": float(np.mean(interp_errs)) if interp_errs else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--states-per-seed", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    random.seed(0)

    totals = {
        "states": 0, "planets": 0,
        "cache_owner_mm": 0, "cache_ship_mm": 0,
        "interp_owner_mm": 0, "interp_ship_mm": 0,
        "spawn_states": 0,
    }
    interp_max = cache_max = 0.0
    interp_errs = []
    for sidx in range(args.seeds):
        seed = 1000 + sidx
        env = make_env_with_fleets(seed, warmup_steps=args.warmup)
        for state_idx in range(args.states_per_seed):
            if env.done:
                break
            for _ in range(3):
                obs0 = env.state[0].observation
                obs1 = env.state[1].observation
                env.step([random_moves_for(obs0, 0), random_moves_for(obs1, 1)])
                if env.done:
                    break
            if env.done:
                break
            if not (env.state[0].observation.get("fleets") or []):
                continue
            r = compare_one_state(env)
            totals["states"] += 1
            totals["planets"] += r["n_planets"]
            totals["cache_owner_mm"] += r["cache_owner_mm"]
            totals["cache_ship_mm"] += r["cache_ship_mm"]
            totals["interp_owner_mm"] += r["interp_owner_mm"]
            totals["interp_ship_mm"] += r["interp_ship_mm"]
            if r["crossed_spawn"]:
                totals["spawn_states"] += 1
            interp_max = max(interp_max, r["interp_max_err"])
            cache_max = max(cache_max, r["cache_max_err"])
            interp_errs.append(r["interp_mean_err"])
            print(
                f"[seed={seed} state={state_idx}] fleets={r['n_in_flight_fleets']:2d} "
                f"steps={r['rollout_steps']:3d} "
                f"crossed_spawn={'Y' if r['crossed_spawn'] else 'n'}  "
                f"cache_mm=(owner={r['cache_owner_mm']},ship={r['cache_ship_mm']},max={r['cache_max_err']:.1f})  "
                f"interp_mm=(owner={r['interp_owner_mm']},ship={r['interp_ship_mm']},max={r['interp_max_err']:.1f})"
            )

    p = totals["planets"]
    print("\n=== Summary ===")
    print(f"  states checked                   : {totals['states']}")
    print(f"  planet-checks                    : {p}")
    print(f"  states that crossed comet spawn  : {totals['spawn_states']}")
    print()
    print(f"  CACHED path (training/eval path):")
    print(f"    owner mismatches               : {totals['cache_owner_mm']}  ({100*totals['cache_owner_mm']/max(1,p):.2f}%)")
    print(f"    ship mismatches > 1.5          : {totals['cache_ship_mm']}  ({100*totals['cache_ship_mm']/max(1,p):.2f}%)")
    print(f"    max abs ship err               : {cache_max:.2f}")
    print("    (should all be ZERO — this is what the model sees)")
    print()
    print(f"  INTERPRETER fallback (exported Kaggle bot only):")
    print(f"    owner mismatches               : {totals['interp_owner_mm']}  ({100*totals['interp_owner_mm']/max(1,p):.2f}%)")
    print(f"    ship mismatches > 1.5          : {totals['interp_ship_mm']}  ({100*totals['interp_ship_mm']/max(1,p):.2f}%)")
    print(f"    max abs ship err               : {interp_max:.2f}")
    print(f"    mean of per-state means        : {np.mean(interp_errs) if interp_errs else 0.0:.2f}")
    print("    (errors only when rollout crosses step%100∈{50,150,250,350,450})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
