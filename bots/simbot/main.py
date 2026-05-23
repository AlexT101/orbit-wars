"""
Orbit Wars - simbot

A thin Python wrapper that delegates the entire turn to a native Rust extension
(`simbot_native`). The Rust side owns all bot state — turn counter,
pre-computed entity position trajectories, the vendored simulator — across
calls; Python just normalizes the observation into a dict and forwards it.

Build the native extension first:

    cd bots/simbot
    maturin develop --release
"""

import os
import sys

# Ensure the bundled native module (simbot_native.*.so) sitting next to this
# file is importable, regardless of the harness's cwd or sys.path. This is what
# lets the Kaggle submission tarball work: main.py + the .so live side by side.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import simbot_native

# Module-level so it survives across every agent(obs) call within a match.
# Kaggle imports main.py once per episode, giving this exactly the right
# lifetime: persists turn-to-turn, dies with the process at episode end.
_BOT = simbot_native.Bot()

_OBS_FIELDS = (
    "player",
    "planets",
    "fleets",
    "angular_velocity",
    "initial_planets",
    "comets",
    "comet_planet_ids",
)


def _as_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {k: getattr(obs, k) for k in _OBS_FIELDS}


def agent(obs):
    moves = _BOT.compute_moves(_as_dict(obs))
    # Match the baseline's [from_planet_id, angle, num_ships] list-of-lists shape.
    return [list(move) for move in moves]
