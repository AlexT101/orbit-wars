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

# The native module loads its tunable constants from config.json / config_4p.json
# at runtime. Its built-in fallback path is CARGO_MANIFEST_DIR, which on Kaggle is
# the (nonexistent) Docker build dir — so point it explicitly at the JSON files
# bundled next to this main.py. `setdefault` keeps any explicit override (e.g. the
# tuner sets APOLLO_CONFIG) authoritative locally.
os.environ.setdefault("APOLLO_CONFIG", os.path.join(_HERE, "config.json"))
os.environ.setdefault("APOLLO_CONFIG_4P", os.path.join(_HERE, "config_4p.json"))

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
    "remainingOverageTime"
)


def _as_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {k: getattr(obs, k) for k in _OBS_FIELDS}


def agent(obs):
    try:
        moves = _BOT.compute_moves(_as_dict(obs))
    except Exception:
        traceback.print_exc()
        raise
    return [list(move) for move in moves]
