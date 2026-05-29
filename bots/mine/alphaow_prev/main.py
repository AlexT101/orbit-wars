"""Self-play wrapper: PREVIOUS alphaow built from commit 8cdd5e8.

This is the original "apollo + duck = alphaow" binary, before today's apollo
refactor and this session's ballistic-rollout / N-layer-loader changes. It runs
its own worktree binary + v2_replays.bin net via subprocess so it can play
head-to-head against the current alphaow_ball in the same Python process.
Apollo rollout is left at its default (ON); no ballistic flag exists in this
build, so the rollout replans every tick.
"""

import json
import os
import subprocess
import sys
import threading

_PROC = None
_LOCK = threading.Lock()

_WT = "/Users/derekwang/Documents/GitHub/orbit-wars/.wt-prev/bots/mine/alphaow"
_BIN = os.environ.get("ALPHAOW_PREV_BIN", _WT + "/target/release/alphaow-bot")
_NET = os.environ.get("ALPHAOW_PREV_NET", _WT + "/train/weights/v2_replays.bin")


def _norm(o):
    g = o.get if isinstance(o, dict) else (lambda k, d=None: getattr(o, k, d))
    return {
        "player": g("player", 0),
        "step": g("step", 0),
        "planets": list(g("planets", []) or []),
        "fleets": list(g("fleets", []) or []),
        "angular_velocity": g("angular_velocity", 0.0),
        "initial_planets": list(g("initial_planets", []) or []),
        "comets": list(g("comets", []) or []),
        "comet_planet_ids": list(g("comet_planet_ids", []) or []),
    }


def _ensure():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    env = dict(os.environ)
    env.pop("OW_ROLLOUT_REACTIVE", None)
    env["ALPHAOW_VALUE_NET_PATH"] = _NET
    _PROC = subprocess.Popen(
        [_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
        bufsize=0,
    )
    return _PROC


def agent(obs, config=None):
    p = _norm(obs)
    if config is not None:
        cfg = {}
        for k in ("episodeSteps", "actTimeout", "shipSpeed", "sunRadius", "boardSize", "cometSpeed"):
            v = config.get(k) if isinstance(config, dict) else getattr(config, k, None)
            if v is not None:
                cfg[k] = v
        if cfg:
            p["config"] = cfg
    with _LOCK:
        proc = _ensure()
        try:
            proc.stdin.write((json.dumps(p, separators=(",", ":")) + "\n").encode())
            proc.stdin.flush()
            r = proc.stdout.readline()
        except (BrokenPipeError, OSError):
            global _PROC
            _PROC = None
            return []
        if not r:
            return []
        try:
            return json.loads(r.decode())
        except json.JSONDecodeError:
            return []
