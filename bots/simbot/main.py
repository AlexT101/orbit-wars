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

import inspect
import os
import sys
import traceback

# Locate the directory containing simbot_native.*.so. Both Kaggle's runner and
# kaggle_environments' local runner exec() this file without setting __file__,
# but both compile the source with the real path baked into the code object
# (that's how tracebacks still report the right file). Read it back from the
# current frame's co_filename — works under exec(), normal import, and Kaggle.
_frame = inspect.currentframe()
if _frame is not None:
    _HERE = os.path.dirname(os.path.abspath(_frame.f_code.co_filename))
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

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
    try:
        moves = _BOT.compute_moves_with_search(_as_dict(obs))
    except Exception:
        traceback.print_exc()
        raise
    # Match the baseline's [from_planet_id, angle, num_ships] list-of-lists shape.
    return [list(move) for move in moves]
