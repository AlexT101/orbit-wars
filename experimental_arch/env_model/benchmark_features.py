"""Benchmark `encode_obs` (the full 4-frame feature build incl. the aim solver).

Generates realistic mid-game states with env_engine (fleets in flight, so the
t_resolved frame and all-in action projection are exercised), then times encoding.

Run (from experimental_arch/):
    python env_model/benchmark_features.py
"""

from __future__ import annotations

import random
import time

from orbit_wars_engine import OrbitWarsEngine
from orbit_wars_model import encode_obs


def sample_states(seed: int, n_states: int, warmup: int) -> list[dict]:
    """Play a noisy game with env_engine, snapshotting obs along the way."""
    rng = random.Random(seed)
    eng = OrbitWarsEngine(num_players=2)
    obs = eng.reset(seed=seed)["observations"]
    states = []
    step = 0
    while len(states) < n_states:
        acts = [[], []]
        for pid in (0, 1):
            for p in obs[pid]["planets"]:
                _id, owner, *_rest, ships, _prod = p
                if owner == pid and ships >= 2 and rng.random() < 0.25:
                    acts[pid].append([int(_id), rng.uniform(-3.14, 3.14), max(1, int(ships * 0.5))])
        out = eng.step(acts)
        obs = out["observations"]
        step += 1
        if out["done"]:
            obs = eng.reset(seed=seed + step)["observations"]
        if step >= warmup and step % 5 == 0:
            states.append(obs[0])
    return states


def main() -> int:
    states = sample_states(seed=1, n_states=40, warmup=10)
    n_planets = [len(s["planets"]) for s in states]
    print(f"{len(states)} states, planets/state: min={min(n_planets)} "
          f"max={max(n_planets)} mean={sum(n_planets)/len(n_planets):.1f}")

    # Warm up (first call pays allocation costs).
    for s in states[:3]:
        encode_obs(s, 0)

    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        for s in states:
            encode_obs(s, 0)
    dt = time.perf_counter() - t0
    n = reps * len(states)
    per = dt / n * 1000.0
    print(f"encode_obs: {per:.2f} ms/call  ({1000.0 / per:,.0f} calls/s)  over {n} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
