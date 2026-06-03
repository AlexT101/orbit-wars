"""Standalone Orbit Wars aiming benchmark — no dependency on the orbit-wars repo.

Public API:
  iter_samples(path=None) -> Iterator[Sample]   # kaggle obs + (source,target,fleet_size)
  validate(angles, path=None) -> list[bool]     # scored with the real kaggle engine
  score(angles, path=None)   -> dict            # {n, correct, accuracy}

Deps: numpy, kaggle_environments. Data file `aim_samples.npz` ships alongside.
"""
from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

_DEFAULT_NPZ = Path(__file__).with_name("aim_samples.npz")
MAX_STEPS = 40  # resolution cap (dataset arrival <= 20)


@dataclass
class Sample:
    obs: dict
    source: int
    target: int
    fleet_size: int
    meta: dict = field(default_factory=dict)


def _load_npz(path):
    p = Path(path) if path is not None else _DEFAULT_NPZ
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Build it with scripts/build_standalone_aim_benchmark.py")
    return np.load(p, allow_pickle=False)


def _rows(arr2d):
    """Planet rows with id (col 0) >= 0, as python lists [id,owner,x,y,r,ships,prod]."""
    out = []
    for row in arr2d:
        if int(row[0]) < 0:
            continue
        out.append([int(row[0]), int(row[1]), float(row[2]), float(row[3]),
                    float(row[4]), float(row[5]), float(row[6])])
    return out


def _comets(grid_ids, path_index, paths):
    """Reconstruct kaggle obs comet groups from padded arrays.

    grid_ids [E,C] (-1 pad), path_index [E], paths [E,C,L,2] (NaN pad).
    A group is one spawn event with >=1 live comet id.
    """
    groups = []
    E = grid_ids.shape[0]
    for e in range(E):
        pids = [int(x) for x in grid_ids[e] if int(x) >= 0]
        if not pids:
            continue
        comet_paths = []
        for c in range(len(pids)):
            pts = paths[e, c]
            valid = pts[~np.isnan(pts[:, 0])]
            comet_paths.append([[float(x), float(y)] for x, y in valid])
        groups.append({"planet_ids": pids, "paths": comet_paths,
                       "path_index": int(path_index[e])})
    return groups


def iter_samples(path=None) -> Iterator[Sample]:
    # Materialise each array once. Indexing an NpzFile (`d[key]`) re-reads and
    # decompresses the whole array on every access, so reading inside the loop
    # would decompress every array N times.
    d = {k: v for k, v in _load_npz(path).items()}
    n = int(d["ships"].shape[0])
    for i in range(n):
        planets = _rows(d["planets"][i])
        initial = _rows(d["initial_planets"][i])
        next_fleet_id = (max((p[0] for p in planets), default=0) + 1) + 1000
        obs = {
            "planets": planets,
            "initial_planets": initial,
            "fleets": [],
            "comets": _comets(d["comet_grid_ids"][i], d["comet_path_index"][i],
                              d["comet_paths"][i]),
            "comet_planet_ids": [int(x) for x in d["comet_flat_ids"][i] if int(x) >= 0],
            "angular_velocity": float(d["angular_velocity"][i]),
            "step": int(d["step"][i]),
            "player": int(d["seat"][i]),
            "next_fleet_id": int(next_fleet_id),
            "ship_speed": float(d["ship_speed"][i]),
        }
        yield Sample(
            obs=obs,
            source=int(d["source_id"][i]),
            target=int(d["target_id"][i]),
            fleet_size=int(d["ships"][i]),
            meta={
                "episode_id": int(d["episode_id"][i]),
                "step": int(d["step"][i]),
                "seat": int(d["seat"][i]),
                "witness_angle": float(d["witness_angle"][i]),
                "example_aimer_fail": (bool(d["example_aimer_fail"][i])
                                       if "example_aimer_fail" in d else False),
                "reachable": (bool(d["reachable"][i]) if "reachable" in d else True),
            },
        )


_OW = None  # lazy import so iter_samples works without kaggle_environments


def _ow():
    global _OW
    if _OW is None:
        from kaggle_environments.envs.orbit_wars import orbit_wars as ow
        _OW = ow
    return _OW


def _hit_planet(sample: Sample, angle: float):
    """Return the planet id the launched fleet first contacts, or None (sun/oob/none).

    Drives the real kaggle interpreter on a one-fleet state. Planet positions are
    recomputed by the engine from initial_planets + angular_velocity*step. The
    recorded launch-frame positions equal orbit(step-1) (verified to 0.0 error), so
    obs.planets already sit one orbit-step behind `step`; the first interpreter call
    must advance them to orbit(step), i.e. obs.step = base_step + tick (NOT +1). With
    exactly one in-flight fleet, the hit is the lone non-source planet whose
    (owner, ships) deviates from pure production on the resolving tick.
    """
    ow = _ow()
    obs_d = sample.obs
    seat = int(obs_d["player"])
    base_step = int(obs_d["step"])
    fleet_id = int(obs_d["next_fleet_id"])
    src = int(sample.source)

    obs = types.SimpleNamespace(
        planets=[list(r) for r in obs_d["planets"]],
        initial_planets=[list(r) for r in obs_d["initial_planets"]],
        fleets=[],
        comets=copy.deepcopy(obs_d["comets"]),
        comet_planet_ids=list(obs_d["comet_planet_ids"]),
        next_fleet_id=fleet_id,
        angular_velocity=float(obs_d["angular_velocity"]),
        step=base_step,
        player=seat,
    )
    num_agents = max(seat + 1, 2)
    state = [types.SimpleNamespace(
        observation=(obs if i == 0 else types.SimpleNamespace()),
        action=None, status="ACTIVE", reward=0) for i in range(num_agents)]
    state[seat].action = [[src, float(angle), int(sample.fleet_size)]]

    cfg = types.SimpleNamespace(shipSpeed=float(obs_d.get("ship_speed", 6.0)),
                                cometSpeed=4.0, episodeSteps=100000,
                                seed=None, randomSeed=None)
    env = types.SimpleNamespace(configuration=cfg, done=False, info={"seed": 0})

    for tick in range(MAX_STEPS):
        obs.step = base_step + tick
        pre = {int(p[0]): (int(p[1]), int(p[5]), int(p[6])) for p in obs.planets}
        ow.interpreter(state, env)
        state[seat].action = None  # only the launch tick carries an action
        if not any(int(f[0]) == fleet_id for f in obs.fleets):
            # Resolved this tick: find the lone deviating non-source planet.
            for p in obs.planets:
                pid = int(p[0])
                if pid == src or pid not in pre:
                    continue
                owner0, ships0, prod0 = pre[pid]
                expected = ships0 + (prod0 if owner0 != -1 else 0)
                if int(p[1]) != owner0 or int(p[5]) != expected:
                    return pid
            return None
    return None


def _validate_one(sample: Sample, angle) -> bool:
    """Score one sample. ``angle`` is a float launch angle, or ``None`` to decline.

    Reachable target: correct iff a (non-None) angle first-contacts the target.
    Impossible target (``meta['reachable']`` False): correct iff the aimer declined
    (``angle is None``) — no angle can hit, so shooting is always wrong.
    """
    if not sample.meta.get("reachable", True):
        return angle is None
    if angle is None:
        return False
    return _hit_planet(sample, float(angle)) == int(sample.target)


def validate(angles: Sequence, path=None) -> list:
    """Score angles (one per sample, ``iter_samples()`` order). Use ``None`` to decline.

    Returns ``list[bool]``. Raises if the count doesn't match.
    """
    samples = list(iter_samples(path))
    if len(angles) != len(samples):
        raise ValueError(f"expected {len(samples)} angles, got {len(angles)}")
    return [_validate_one(s, a) for s, a in zip(samples, angles)]


def score(angles: Sequence[float], path=None) -> dict:
    res = validate(angles, path)
    correct = sum(res)
    return {"n": len(res), "correct": correct,
            "accuracy": correct / len(res) if res else 0.0}
