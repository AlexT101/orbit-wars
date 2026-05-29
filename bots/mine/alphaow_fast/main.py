"""Self-play wrapper: alphaow with the PORTED faster apollo (vendored from the
refactored standalone apollo crate).

Points _BIN at the .wt-fast worktree binary (HEAD 7fabf46 + refactored apollo
re-vendored into src/apollo/). Baseline rollout (OW_ROLLOUT_REACTIVE unset) so
this is an apples-to-apples A/B against alphaow_base — the ONLY difference is the
ported apollo engine. Pins its own subprocess env to avoid racing on os.environ.
"""

import json
import os
import subprocess
import sys
import threading

_PROC = None
_LOCK = threading.Lock()

_BIN = os.environ.get(
    "ALPHAOW_BOT_BIN",
    "/Users/derekwang/Documents/GitHub/orbit-wars/.wt-fast/bots/mine/alphaow/target/release/alphaow-bot",
)
_NET = os.environ.get(
    "ALPHAOW_VALUE_NET_PATH",
    "/Users/derekwang/Documents/GitHub/orbit-wars/.wt-fast/bots/mine/alphaow/train/weights/v2_replays.bin",
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
    env.pop("OW_ROLLOUT_REACTIVE", None)  # baseline rollout = replan every tick
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
