"""aphrodite - the aphrodite Rust bot paired with the *fixed-extrapolation*
XGB value net, retrained 2026-05-30:

  1. **Default model**: weights/xgb_top10_d6_fixed.json — XGBoost gbtree
     retrained against the corrected extrapolate_fleets combat. It MUST be
     paired with APHRODITE_EXTRAP_FIX=1 (set below) so the deployed bot's
     extrapolation matches the training feature extraction. WITHOUT that
     env the bot's extrapolate_fleets uses the buggy combat the model was
     NOT trained on — silently mispredicting.

  2. **XGBoost-in-Rust** value net (src/xgb.rs) — pure-Rust gbtree
     inference, bit-exact parity with Python xgboost. Loaded automatically
     when APHRODITE_VALUE_NET_PATH points at a .json file.

  3. **No rollouts** — leaf-eval only. The XGB value net is strong enough
     that the (very slow) depth-8 apollo replan rollout doesn't pay for
     itself; skipping it buys many more MCTS iterations per turn.

To override the model, set APHRODITE_VALUE_NET_PATH to another file.

This single file serves BOTH layouts — `_locate()` auto-detects which:

  * **dev**: this wrapper sits in `aphrodite/` and uses this directory's own
    build tree (binary at `target/release/aphrodite`, weights at
    `train/weights/xgb_top10_d6_fixed.json`). Builds the binary on demand if
    missing.

  * **Kaggle submission**: `main.py`, `aphrodite`, and
    `xgb_top10_d6_fixed.json` are bundled flat in one dir.
    `build_submission.py` copies THIS file into the archive verbatim — do
    not fork a second copy.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import threading

_PROC = None
_LOCK = threading.Lock()
_BIN_NAME = "aphrodite"
_WEIGHTS_NAME = "xgb_top10_d6_fixed.json"


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
        w = os.path.join(d, _WEIGHTS_NAME)
        return env_bin, (w if os.path.isfile(w) else None), d, None

    wd = _wrapper_dir()

    # Flat-bundle layout (Kaggle): the binary sits next to main.py. Try the
    # wrapper dir, the Kaggle agent mount, then cwd.
    for d in (wd, "/kaggle_simulations/agent", os.getcwd()):
        if d:
            b = _bin_in(d)
            if b:
                w = os.path.join(d, _WEIGHTS_NAME)
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
    weights = os.path.join(crate_dir, "train", "weights", _WEIGHTS_NAME)
    return binary, (weights if os.path.isfile(weights) else None), crate_dir, crate_dir


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
    }


def _ensure():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    binary, weights, run_cwd, build_cwd = _locate()
    _build_if_needed(binary, build_cwd)
    _ensure_executable(binary)
    env = dict(os.environ)
    env.setdefault("APHRODITE_BUDGET_MS", "500")
    # Pair the fixed model with the fixed extrapolation. WITHOUT this env,
    # the bot's extrapolate_fleets uses the buggy combat that the model was
    # NOT trained on — silently mispredicting.
    env["APHRODITE_EXTRAP_FIX"] = "1"
    # Default to the fixed XGB model unless the caller set their own.
    if weights:
        env.setdefault("APHRODITE_VALUE_NET_PATH", weights)
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
