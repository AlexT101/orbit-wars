"""aphrodite - DUCT / MCTS based Rust bot with eval, using Apollo heuristics.
This bot needs to be rebuilt on any Rust change.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import threading

# Update these contants if training new weights
_WEIGHTS_2P_NAME = "xgb_2p_6_01_6_07.json"
_WEIGHTS_4P_NAME = "xgb_4p_6_01_6_07.json"

# Local testing uses lower limits for speed
_DEV_BUDGET_MS = "700"
_SUBMISSION_BUDGET_MS = "1000"

# Set to true if testing higher time limits (also lets bot use remaining overage time)
_USE_PROD_LIMITS = False


_PROC = None
_LOCK = threading.Lock()
_BIN_NAME = "aphrodite"

def _pump_stderr(pipe):
    try:
        for line in iter(pipe.readline, b""):
            try:
                sys.stderr.write(line.decode("utf-8", "replace"))
                sys.stderr.flush()
            except Exception:
                pass
    except Exception:
        pass


def _bin_in(d):
    """Return the aphrodite path inside dir `d`, or None. Prefers the
    platform-native name — cargo emits `aphrodite.exe` on Windows and a
    plain `aphrodite` on Linux/macOS (and in the Kaggle bundle)."""
    names = (_BIN_NAME + ".exe", _BIN_NAME) if sys.platform == "win32" else (_BIN_NAME,)
    for n in names:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return p
    return None


def _wrapper_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return None


def _locate():
    """Resolve (binary, weights, run_cwd, build_cwd) for whichever layout
    we're running under.

    Returns:
      binary    — path to the aphrodite executable.
      weights   — path to the value-net file, or None if not found (the
                  binary then uses its own cargo-local default).
      run_cwd   — cwd to spawn the binary in. Dev uses the aphrodite crate dir
                  so the binary can resolve cargo-local defaults; the flat
                  bundle uses the bundle dir.
      build_cwd — crate dir to `cargo build` in when the binary is missing
                  (dev only), or None when building isn't possible/needed.
    """
    # Explicit binary override always wins (dev experiments).
    env_bin = os.environ.get("APHRODITE_BIN")
    if env_bin and os.path.isfile(env_bin):
        d = os.path.dirname(env_bin)
        w = os.path.join(d, _WEIGHTS_2P_NAME)
        return env_bin, (w if os.path.isfile(w) else None), d, None

    wd = _wrapper_dir()

    # Flat-bundle layout (Kaggle): the binary sits next to main.py. Try the
    # wrapper dir, the Kaggle agent mount, then cwd.
    for d in (wd, "/kaggle_simulations/agent", os.getcwd()):
        if d:
            b = _bin_in(d)
            if b:
                w = os.path.join(d, _WEIGHTS_2P_NAME)
                return b, (w if os.path.isfile(w) else None), d, None

    # Dev layout: this wrapper lives at the aphrodite crate root and uses its
    # own build tree.
    crate_dir = os.path.abspath(wd or os.getcwd())
    release = os.path.join(crate_dir, "target", "release")
    # Use the existing binary if built; otherwise point at the name cargo will
    # produce on this platform so _build_if_needed writes there.
    binary = _bin_in(release) or os.path.join(
        release, _BIN_NAME + (".exe" if sys.platform == "win32" else "")
    )
    weights = os.path.join(crate_dir, "train", "weights", _WEIGHTS_2P_NAME)
    return binary, (weights if os.path.isfile(weights) else None), crate_dir, crate_dir


def _weight_candidates(run_cwd, build_cwd, n_players):
    # 4p prefers the 4p net but falls back to the 2p net; 2p uses the 2p net.
    names = (
        (_WEIGHTS_4P_NAME, _WEIGHTS_2P_NAME)
        if n_players >= 4
        else (_WEIGHTS_2P_NAME,)
    )
    dirs = []
    for d in (run_cwd, build_cwd):
        if d and d not in dirs:
            dirs.append(d)
            tw = os.path.join(d, "train", "weights")
            if tw not in dirs:
                dirs.append(tw)
    for d in dirs:
        for name in names:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def _infer_num_players(payload):
    seen = set()
    for planet in payload.get("planets", []) or []:
        try:
            owner = int(planet[1])
        except Exception:
            continue
        if owner >= 0:
            seen.add(owner)
    for fleet in payload.get("fleets", []) or []:
        try:
            owner = int(fleet[1])
        except Exception:
            continue
        if owner >= 0:
            seen.add(owner)
    if len(seen) > 2 or any(p >= 2 for p in seen):
        return 4
    return 2


def _build_if_needed(path, build_cwd):
    if os.path.isfile(path):
        return
    if not build_cwd:
        raise RuntimeError(f"binary not found at {path} and no build tree to build it from")
    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError(f"binary not found at {path} and cargo not on PATH")
    subprocess.check_call([cargo, "build", "--release"], cwd=build_cwd)


def _ensure_executable(path):
    try:
        st = os.stat(path).st_mode
        os.chmod(path, st | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass


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
        # Seconds of shared overage budget left (engine-reported). The Rust
        # planner only consults this when APHRODITE_USE_OVERAGE is enabled.
        "remainingOverageTime": g("remainingOverageTime", 0.0),
    }


def _ensure(payload=None):
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    binary, weights, run_cwd, build_cwd = _locate()
    _build_if_needed(binary, build_cwd)
    _ensure_executable(binary)
    env = dict(os.environ)
    # Production limits (submission budget + overage pool) when _USE_PROD_LIMITS
    # is on, else the faster dev budget with no overage. See duct::best_move's
    # overage extension for how APHRODITE_USE_OVERAGE is consumed.
    env.setdefault(
        "APHRODITE_BUDGET_MS",
        _SUBMISSION_BUDGET_MS if _USE_PROD_LIMITS else _DEV_BUDGET_MS,
    )
    env.setdefault("APHRODITE_USE_OVERAGE", "1" if _USE_PROD_LIMITS else "0")
    # Default to format-specific XGB weights unless the caller set their own.
    if "APHRODITE_VALUE_NET_PATH" not in env:
        n_players = _infer_num_players(payload or {})
        weights = _weight_candidates(run_cwd, build_cwd, n_players) or weights
        if weights:
            env["APHRODITE_VALUE_NET_PATH"] = weights
    # Expose the 2p net as the "two-players-left" model so a 4p game that
    # collapses to a 1v1 is scored by the (stronger) 2p net. Harmless in 2p
    # games — it just matches the primary.
    if "APHRODITE_VALUE_NET_PATH_2P" not in env:
        w2p = _weight_candidates(run_cwd, build_cwd, 2)
        if w2p:
            env["APHRODITE_VALUE_NET_PATH_2P"] = w2p
    _PROC = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=run_cwd,
        env=env,
        bufsize=0,
    )
    threading.Thread(
        target=_pump_stderr, args=(_PROC.stderr,), daemon=True
    ).start()
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
        proc = _ensure(p)
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
