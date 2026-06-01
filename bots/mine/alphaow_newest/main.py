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

This single file serves BOTH layouts — `_locate()` auto-detects which:

  * **dev**: this wrapper sits in `alphaow_newest/` and shares the sibling
    `alphaow/` build tree (binary at `../alphaow/target/release/alphaow-bot`,
    weights at `../alphaow/train/weights/xgb_top10_d6.json`). Builds the
    binary on demand if missing.

  * **Kaggle submission**: `main.py`, `alphaow-bot`, and `xgb_top10_d6.json`
    are bundled flat in one dir. `build_submission.py` copies THIS file into
    the archive verbatim — do not fork a second copy.
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
_BIN_NAME = "alphaow-bot"
_WEIGHTS_NAME = "xgb_top10_d6.json"


def _pump_stderr(pipe):
    # Forward the binary's stderr into our own stderr line by line. Kaggle
    # captures the agent process's stderr (that's how panics show up in the
    # logs), but it wraps sys.stderr in an object with no real file
    # descriptor, so we can't hand the fd to Popen directly. Pump it here.
    # Harmless in dev: normal play emits nothing (OW_DEBUG/OW_PROFILE are
    # stripped below), so only genuine panics surface.
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
    """Return the alphaow-bot path inside dir `d`, or None. Prefers the
    platform-native name — cargo emits `alphaow-bot.exe` on Windows and a
    plain `alphaow-bot` on Linux/macOS (and in the Kaggle bundle)."""
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
        # Some sandboxes (kaggle_environments) exec this file without setting
        # __file__. Caller falls back to other candidates.
        return None


def _locate():
    """Resolve (binary, weights, run_cwd, build_cwd) for whichever layout
    we're running under.

    Returns:
      binary    — path to the alphaow-bot executable.
      weights   — path to the value-net file, or None if not found (the
                  binary then uses its own cargo-local default).
      run_cwd   — cwd to spawn the binary in. Dev uses the alphaow crate dir
                  so the binary can resolve cargo-local defaults; the flat
                  bundle uses the bundle dir.
      build_cwd — crate dir to `cargo build` in when the binary is missing
                  (dev only), or None when building isn't possible/needed.
    """
    # Explicit binary override always wins (dev experiments).
    env_bin = os.environ.get("ALPHAOW_BOT_BIN")
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

    # Dev layout: this wrapper lives in alphaow_newest/ and shares the sibling
    # alphaow/ build tree. _wrapper_dir() would return alphaow_newest/, which
    # has no target/ or train/ — so resolve the sibling crate.
    base = wd or "/Users/derekwang/Documents/GitHub/orbit-wars/bots/mine/alphaow_newest"
    alphaow = os.path.abspath(os.path.join(base, "..", "alphaow"))
    release = os.path.join(alphaow, "target", "release")
    # Use the existing binary if built; otherwise point at the name cargo will
    # produce on this platform so _build_if_needed writes there.
    binary = _bin_in(release) or os.path.join(
        release, _BIN_NAME + (".exe" if sys.platform == "win32" else "")
    )
    weights = os.path.join(alphaow, "train", "weights", _WEIGHTS_NAME)
    return binary, (weights if os.path.isfile(weights) else None), alphaow, alphaow


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
    # Strip OW_* tuning overrides so this wrapper is a stable baseline
    # regardless of parent env used by experiments.
    for k in (
        "OW_PLANNER", "OW_PUCT_C", "OW_K_ROOT", "OW_K_NON_ROOT",
        "OW_ROLLOUT", "OW_ROLLOUT_DEPTH", "OW_ROLLOUT_REACTIVE",
        "OW_ROLLOUT_NOISE", "OW_DUCT_ENUMERATE", "OW_NO_COOP", "OW_NO_REUSE",
        "OW_FOCUSED_CANDIDATES", "OW_SELECTION", "OW_EXP3_ETA",
        "OW_EXP3_GAMMA", "OW_DEBUG", "OW_PROFILE",
    ):
        env.pop(k, None)
    # **No rollouts** — leaf-eval only. The XGB value net is strong enough
    # that the (very slow) depth-8 apollo replan rollout doesn't pay for
    # itself; skipping it buys many more MCTS iterations per turn. Proven
    # 8W-2L vs production MLP.
    env["OW_ROLLOUT"] = "none"
    env["OW_ROLLOUT_DEPTH"] = "0"
    # Use the full per-turn think budget. The Rust binary defaults to 500ms
    # (main.rs); the harness allows ~1000ms with extra buffer on top, so spend
    # it — DUCT is anytime, so more wall time = strictly more search.
    env.setdefault("ALPHAOW_BUDGET_MS", "1000")
    # Default to the v4 XGB model unless the caller set their own.
    if weights:
        env.setdefault("ALPHAOW_VALUE_NET_PATH", weights)
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
