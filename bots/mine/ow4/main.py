"""ow4 — heuristic 2p Orbit Wars bot.

Build:
    cd bots/mine/ow4 && maturin develop --release

Pure heuristic. No value net, no MCTS.
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

import ow4_native

_BOT = ow4_native.Bot()

_OBS_FIELDS = (
    "player",
    "step",
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
    out = {}
    for k in _OBS_FIELDS:
        if hasattr(obs, k):
            out[k] = getattr(obs, k)
    return out


_DEBUG = os.environ.get("OW4_DEBUG") == "1"
_TURN = [0]


def agent(obs):
    try:
        d = _as_dict(obs)
        moves = _BOT.compute_moves(d)
    except Exception:
        traceback.print_exc()
        raise
    if _DEBUG:
        planets = d.get("planets", []) if isinstance(d, dict) else []
        me = d.get("player")
        mine = [p for p in planets if p[1] == me]
        enemies = [p for p in planets if p[1] not in (-1, me)]
        my_fleets = [f for f in d.get("fleets", []) if f[1] == me]
        print(
            f"[ow4 t{_TURN[0]}] my_planets={len(mine)} my_ships={sum(p[5] for p in mine)}"
            f" enemy_planets={len(enemies)} enemy_ships={sum(p[5] for p in enemies)}"
            f" my_fleet_ships={sum(f[6] for f in my_fleets)} actions={len(moves)}",
            file=sys.stderr,
        )
        for m in moves[:5]:
            print(f"   action: from={m[0]} angle={m[1]:.2f} ships={m[2]}", file=sys.stderr)
        if len(moves) > 5:
            print(f"   ... +{len(moves)-5} more", file=sys.stderr)
    _TURN[0] += 1
    return [list(move) for move in moves]
