"""alphaow_newest — the alphaow Rust bot with all 2026-05-29 improvements
plugged in by default:

  1. **XGBoost-in-Rust** value net (src/xgb.rs) — pure-Rust gbtree
     inference, bit-exact parity with Python xgboost. Loaded
     automatically when ALPHAOW_VALUE_NET_PATH points at a .json file
     (vs the legacy AOWV binary).

  2. **summary_features_v3** (58-d) — the 46-d summary_v2 + 12 user-
     requested extras (tick, 8 split distances incl. before/after
     extrapolation, n_static, n_orbit, angular_velocity).

  3. **Default model**: weights/xgb_top10_d6.json — XGBoost gbtree,
     46-d (no extras), trained on the top-10 player subset (3,616
     games). Offline val 88.28%, in-play **8W-2L vs production MLP**
     at n=10 — currently the strongest deployable candidate.
     The all-data variant (xgb_all_46p12d6.json, 87.37% offline) was
     6W-4L in-play; top-10 won more states the bot itself produces
     in self-play.

  4. **OW_PLANNER** env switch (defaults to DUCT, the correct planner
     for simultaneous-move games; "mcts" loses 0-10 vs default).

Strips any caller-set OW_* tuning env so this wrapper is a stable
deploy regardless of parent process experiments.

To override the model, set ALPHAOW_VALUE_NET_PATH to another file.
The XGB JSON loader auto-detects vs AOWV magic, so e.g.
  ALPHAOW_VALUE_NET_PATH=.../xgb_top10_d6.json     (proven 8-2 in-play, 46-d)
  ALPHAOW_VALUE_NET_PATH=.../xgb_all_46p12d8.json  (87.49% offline, 4.5x slower inference)
  ALPHAOW_VALUE_NET_PATH=.../v2_replays.bin        (legacy MLP, ~85% offline)
all work.
"""

import json
import os
import shutil
import subprocess
import threading

_PROC = None
_LOCK = threading.Lock()

# Points absolute to the alphaow build tree (we share its binary + weights
# directory). _here() in this wrapper would return alphaow_newest/, which
# does not have target/ or train/ — so we override.
try:
    _WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Some sandboxes (kaggle_environments) exec this file without setting
    # __file__. Fall back to the known absolute path.
    _WRAPPER_DIR = "/Users/derekwang/Documents/GitHub/orbit-wars/bots/mine/alphaow_newest"
_ALPHAOW_DIR = os.path.abspath(os.path.join(_WRAPPER_DIR, "..", "alphaow"))

# Default to the v4 XGB model (46 + 12 extras, trained on all combined data).
_DEFAULT_VALUE_NET = os.path.join(_ALPHAOW_DIR, "train/weights/xgb_top10_d6.json")


def _here():
    """Returns the alphaow build dir (where the cargo project lives).
    The Rust binary expects to be invoked with cwd pointing somewhere it
    can resolve its (cargo-local) defaults — alphaow/, not alphaow_newest/.
    """
    return _ALPHAOW_DIR


def _binary_path():
    env = os.environ.get("ALPHAOW_BOT_BIN")
    if env and os.path.isfile(env):
        return env
    return os.path.join(_ALPHAOW_DIR, "target/release/alphaow-bot")


def _build_if_needed(path):
    if os.path.isfile(path):
        return
    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError(f"binary not found at {path} and cargo not on PATH")
    subprocess.check_call([cargo, "build", "--release"], cwd=_ALPHAOW_DIR)


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
    # Strip OW_* tuning overrides so this wrapper is a stable baseline
    # regardless of parent env used by experiments.
    for k in (
        "OW_PLANNER", "OW_PUCT_C", "OW_K_ROOT", "OW_K_NON_ROOT",
        "OW_ROLLOUT", "OW_ROLLOUT_DEPTH", "OW_ROLLOUT_REACTIVE",
        "OW_ROLLOUT_NOISE", "OW_DUCT_ENUMERATE", "OW_NO_COOP", "OW_NO_REUSE",
        "OW_FOCUSED_CANDIDATES", "OW_SELECTION", "OW_EXP3_ETA", "OW_EXP3_GAMMA",
    ):
        env.pop(k, None)
    # **No rollouts** — leaf-eval only. The XGB value net is strong enough
    # that the (very slow) depth-8 apollo replan rollout doesn't pay for
    # itself; skipping it buys many more MCTS iterations per turn.
    env["OW_ROLLOUT"] = "none"
    env["OW_ROLLOUT_DEPTH"] = "0"
    # Default to the v4 XGB model unless the caller set their own.
    if os.path.isfile(_DEFAULT_VALUE_NET):
        env.setdefault("ALPHAOW_VALUE_NET_PATH", _DEFAULT_VALUE_NET)
    _PROC = subprocess.Popen(
        [bin_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=_ALPHAOW_DIR,
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
