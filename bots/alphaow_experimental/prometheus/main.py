"""Kaggle entry point for alphaow. Spawns the Rust binary once and pipes
one JSON observation per turn.

Binary path resolution: $ALPHAOW_BOT_BIN, then <this dir>/target/release/alphaow-bot.
If missing, attempts `cargo build --release` once.
"""

import json
import os
import shutil
import subprocess
import threading

_PROC = None
_LOCK = threading.Lock()

# Value nets bundled with the bot. main.py points the Rust runtime at both
# when the caller hasn't set them, so mixed 2P/4P runs route automatically.
_DEFAULT_VALUE_NET_2P = "train/weights/xgb_46p12e88t11_latest.json"
_DEFAULT_VALUE_NET_4P_CANDIDATES = (
    "train/weights/xgb_4p_v2_rank4_latest.json",
    "train/weights/xgb_4p_v1_latest.json",
)


def _here():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        env = os.environ.get("ALPHAOW_BOT_DIR")
        if env:
            return env
        for cand in (os.getcwd(), os.path.join(os.getcwd(), "alphaow")):
            if os.path.isfile(os.path.join(cand, "Cargo.toml")):
                return cand
        return os.getcwd()


def _binary_path():
    env = os.environ.get("ALPHAOW_BOT_BIN")
    if env and os.path.isfile(env):
        return env
    return os.path.join(_here(), "target/release/alphaow-bot")


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
    # Match alphaow_newest's stable baseline: parent experiment knobs should not
    # leak into this bot unless explicitly represented by this wrapper.
    for k in (
        "OW_PLANNER", "OW_PUCT_C",
        "OW_ROLLOUT", "OW_ROLLOUT_DEPTH", "OW_ROLLOUT_REACTIVE",
        "OW_ROLLOUT_NOISE", "OW_DUCT_ENUMERATE", "OW_NO_COOP", "OW_NO_REUSE",
        "OW_FOCUSED_CANDIDATES", "OW_SELECTION", "OW_EXP3_ETA", "OW_EXP3_GAMMA",
    ):
        env.pop(k, None)
    env["OW_ROLLOUT"] = "none"
    env["OW_ROLLOUT_DEPTH"] = "0"
    default_net_2p = os.path.join(_here(), _DEFAULT_VALUE_NET_2P)
    if os.path.isfile(default_net_2p):
        env.setdefault("ALPHAOW_VALUE_NET_PATH_2P", default_net_2p)
        env.setdefault("ALPHAOW_VALUE_NET_PATH", default_net_2p)
    for rel in _DEFAULT_VALUE_NET_4P_CANDIDATES:
        default_net_4p = os.path.join(_here(), rel)
        if os.path.isfile(default_net_4p):
            env.setdefault("ALPHAOW_VALUE_NET_PATH_4P", default_net_4p)
            break
    _PROC = subprocess.Popen(
        [bin_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=_here(),
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
