"""Validate the Rust feature encoder across the pyo3 boundary.

The Rust unit/property tests (`cargo test`) cover the core `encode` on
in-memory states. This script covers the *Python-facing* surface that training
will actually call:

  - `orbit_wars_model.encode_obs(obs, player)`         (obs-dict path, training)
  - `OrbitWarsModel.set_state(obs).features(player)`   (model path, bot)

For each random observation it checks the distance matrix against an
independent NumPy brute-force reference and asserts the two code paths agree.
Prints `N / N checks pass` in the repo's usual validator style.

Run:
    python env_model/validate_features.py --states 200
"""

import argparse
import math
import random

import numpy as np

from orbit_wars_model import OrbitWarsModel, encode_obs

BOARD = 100.0


def random_obs(rng: random.Random) -> dict:
    n = rng.randint(0, 12)
    planets = []
    for pid in range(n):
        planets.append([
            pid,                              # id
            rng.choice([-1, 0, 1]),           # owner
            rng.uniform(0.0, BOARD),          # x
            rng.uniform(0.0, BOARD),          # y
            rng.uniform(1.0, 3.0),            # radius
            rng.randint(0, 50),               # ships
            rng.randint(0, 3),                # production
        ])
    return {
        "step": rng.randint(0, 499),
        "angular_velocity": rng.uniform(-0.05, 0.05),
        "planets": planets,
        "initial_planets": [list(p) for p in planets],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "next_fleet_id": 0,
    }


def numpy_reference(planets: list) -> np.ndarray:
    """Independent brute-force L2 distance matrix (the oracle)."""
    xy = np.array([[p[2], p[3]] for p in planets], dtype=np.float64)
    n = len(planets)
    ref = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            ref[i, j] = math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
    return ref


def check_one(obs: dict, eps: float) -> list[str]:
    """Return a list of failure messages (empty == all checks passed)."""
    fails: list[str] = []
    planets = obs["planets"]
    n = len(planets)

    feat = encode_obs(obs, 0)
    mat = np.asarray(feat["distance_matrix"], dtype=np.float64).reshape(n, n) if n else np.zeros((0, 0))

    # planet_ids preserved, in order.
    if feat["planet_ids"] != [p[0] for p in planets]:
        fails.append("planet_ids mismatch")
    if feat["n"] != n:
        fails.append(f"n mismatch: {feat['n']} != {n}")

    if n:
        ref = numpy_reference(planets)
        if not np.allclose(mat, ref, atol=eps):
            fails.append(f"matrix != numpy reference (max diff {np.max(np.abs(mat - ref)):.2e})")
        if not np.allclose(mat, mat.T, atol=eps):
            fails.append("matrix not symmetric")
        if np.any(np.abs(np.diag(mat)) > eps):
            fails.append("diagonal not zero")

    # The model path (set_state + .features) must agree with encode_obs exactly.
    model = OrbitWarsModel(num_players=2)
    model.set_state(obs)
    feat2 = model.features(0)
    if feat2["distance_matrix"] != feat["distance_matrix"] or feat2["planet_ids"] != feat["planet_ids"]:
        fails.append("model.features() disagrees with encode_obs()")

    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total = 0
    failed = 0
    for _ in range(args.states):
        total += 1
        fails = check_one(random_obs(rng), args.eps)
        if fails:
            failed += 1
            print("  FAIL:", "; ".join(fails))

    print(f"{total - failed} / {total} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
