"""Self-play wrapper: alphaow with apollo rollout OFF (OW_APOLLO_ROLLOUT=0).

Pins its own subprocess env so it can run head-to-head against alphaow_noroll
in the same Python process without racing on global os.environ.
"""

import json
import os
import subprocess
import sys
import threading

_PROC = None
_LOCK = threading.Lock()

_ROLLOUT = "0"
_BIN = os.environ.get(
    "ALPHAOW_BOT_BIN",
    "/Users/derekwang/Documents/GitHub/orbit-wars/bots/mine/alphaow/target/release/alphaow-bot",
)
_NET = os.environ.get(
    "ALPHAOW_VALUE_NET_PATH",
    "/Users/derekwang/Documents/GitHub/orbit-wars/bots/mine/alphaow/train/weights/v2_replays.bin",
)


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
    env["OW_APOLLO_ROLLOUT"] = _ROLLOUT
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
