"""alphaow_xgb_improved — the alphaow planner (identical to alphaow_newest) with
the improved 170-d XGBoost value net swapped in.

The ONLY difference vs alphaow_newest is the value function:

  * 2P: weights/xgb_46p12e88t11sp13_latest.json — 170-d
        summary_v2[46] + extras_v4[12] + engineered[88] + tempo[11] + spatial[13].
        Rebuilt on 964 rank-1 replays; 96.3% game-grouped CV (vs ~88% for the
        46-d xgb_top10_d6 used by alphaow_newest).
  * 4P: weights/xgb_4p_v2_rank4_latest.json (278-d), v1 fallback.

This crate is a copy of the alphaow crate with src/value_net.rs + src/xgb.rs
replaced by the 170-d extractor (and a one-line `observe_root_state` tempo hook
in main.rs). The planner (duct/mcts/apollo/...) is byte-identical to alphaow, so
any play difference is attributable to the value net alone.

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

_DEFAULT_VALUE_NET_2P = "weights/xgb_46p12e88t11sp13_latest.json"
_DEFAULT_VALUE_NET_4P_CANDIDATES = (
    "weights/xgb_4p_v2_rank4_latest.json",
    "weights/xgb_4p_v1_latest.json",
)


def _here():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        env = os.environ.get("ALPHAOW_BOT_DIR")
        if env:
            return env
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
    subprocess.check_call([cargo, "build", "--release", "--bin", "alphaow-bot"], cwd=_here())


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
    env.setdefault("ALPHAOW_BUDGET_MS", "500")
    # Isolated experiment hooks (only THIS bot reads them, so a sweep harness can
    # tune the improved bot without touching the alphaow_newest baseline).
    _sv = os.environ.get("XGB_IMPROVED_VALUE_SCALE")
    if _sv:
        env["OW_VALUE_SCALE"] = _sv
    _vb = os.environ.get("XGB_IMPROVED_VALUE_BLEND")
    if _vb:
        env["OW_VALUE_BLEND"] = _vb
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
