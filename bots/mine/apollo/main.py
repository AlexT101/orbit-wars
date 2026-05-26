"""
A thin Python wrapper that delegates the entire turn to a native Rust extension.

Build the native extension first:
    maturin develop --release
    
Note you MUST rebuild the Rust module after changing the code or you will be running an old version of the bot.
"""

import inspect
import os
import sys
import traceback

_frame = inspect.currentframe()
if _frame is not None:
    _HERE = os.path.dirname(os.path.abspath(_frame.f_code.co_filename))
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import apollo_native

_BOT = apollo_native.Bot()

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
    return [list(move) for move in moves]
