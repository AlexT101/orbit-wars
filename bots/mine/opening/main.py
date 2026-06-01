"""Kaggle entry point for opening. Spawns the Rust binary once and pipes
one JSON observation per turn.
"""

import json
import os
import shutil
import subprocess
import sys
import threading

_PROC = None
_LOCK = threading.Lock()
_DEFAULT_MODEL = "weights/first_owned_v23.json"


def _here():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    env = os.environ.get("OPENING_BOT_DIR")
    if env:
        return env
    cwd = os.getcwd()
    for cand in (
        cwd,
        os.path.join(cwd, "opening"),
        os.path.join(cwd, "bots", "mine", "opening"),
        os.path.join(cwd, "bots", "opening"),
    ):
        if os.path.isfile(os.path.join(cand, "Cargo.toml")):
            return cand
    return cwd


def _binary_path():
    env = os.environ.get("OPENING_BOT_BIN")
    if env and os.path.isfile(env):
        return env
    return os.path.join(_here(), "target/release/opening-bot")


def _build_if_needed(path):
    if os.path.isfile(path):
        return
    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError(f"binary not found at {path} and cargo not on PATH")
    subprocess.check_call([cargo, "build", "--release"], cwd=_here())


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
    bin_path = _binary_path()
    _build_if_needed(bin_path)
    env = dict(os.environ)
    default_model = os.path.join(_here(), _DEFAULT_MODEL)
    if os.path.isfile(default_model):
        env.setdefault("OPENING_MODEL_PATH", default_model)
    _PROC = subprocess.Popen(
        [bin_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=_here(),
        env=env,
        bufsize=0,
    )
    return _PROC


def agent(obs, config=None):
    p = _norm(obs)
    if config is not None:
        cfg = {}
        for k in ("episodeSteps", "actTimeout", "shipSpeed", "sunRadius",
                  "boardSize", "cometSpeed"):
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
