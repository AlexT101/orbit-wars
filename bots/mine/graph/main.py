"""
A thin Python wrapper that delegates the entire turn to a native Rust extension.

Unlike apollo2 (which vendors its own engine clone), this bot's model of the
environment is `env_model` (the `orbit_wars_model` crate), depended on as a Rust
library. Each turn it also draws the planet adjacency graph as `[LINE]` debug
overlays colored by distance, and can hand the same distance matrix to an RL
pipeline via `Bot.distances_matrix(obs)`.

Build the native extension first:
    maturin develop --release

Note you MUST rebuild the Rust module after changing the code (or env_model) or
you will be running an old version of the bot.
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

import graph_native

_BOT = graph_native.Bot()

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
        moves = _BOT.compute_moves(_as_dict(obs))
    except Exception:
        traceback.print_exc()
        raise
    # Drain debug strings emitted by the Rust planner (`[LINE]`/`[DOT]`/`[TEXT]`
    # markers describing the planet graph). Guard via getattr so a stale .so
    # without `take_debug` doesn't crash every turn.
    take_debug = getattr(_BOT, "take_debug", None)
    if take_debug is not None:
        for line in take_debug():
            print(line, flush=True)
    return [list(move) for move in moves]
