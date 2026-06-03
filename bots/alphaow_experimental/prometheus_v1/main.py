"""prometheus — Rust bot

Two value nets ship with the bot and main.py routes them automatically:

  * **2P**: train/weights/xgb_46p12e88t11_latest.json
  * **4P**: train/weights/xgb_4p_v2_rank4_latest.json

The Rust runtime picks the 2P or 4P net per game; main.py points it at
both via ALPHAOW_VALUE_NET_PATH_2P / ALPHAOW_VALUE_NET_PATH_4P (and sets
the legacy ALPHAOW_VALUE_NET_PATH to the 2P net) when the caller hasn't.

This single file serves BOTH layouts — `_locate()` auto-detects which:

  * **dev**: this wrapper sits at the crate root, so the binary is at
    `target/release/alphaow-bot` and the weights under `train/weights/`.
    Builds the binary on demand if missing.

  * **Kaggle submission**: `main.py`, `alphaow-bot`, and the two value-net
    JSONs are bundled flat in one dir.
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
# Value-net filenames, identical in the flat bundle and under train/weights/.
_NET_2P_NAME = "xgb_46p12e88t11_latest.json"
_NET_4P_NAME = "xgb_4p_v2_rank4_latest.json"


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
        return None


def _opt(path):
    """Return path if it's an existing file, else None."""
    return path if path and os.path.isfile(path) else None


def _locate():
    """Resolve (binary, net_2p, net_4p, run_cwd, build_cwd) for whichever
    layout we're running under.

    Returns:
      binary    — path to the alphaow-bot executable.
      net_2p    — path to the 2-player value-net JSON, or None if not found.
      net_4p    — path to the 4-player value-net JSON, or None if not found.
      run_cwd   — cwd to spawn the binary in. Dev uses the crate dir so the
                  binary can resolve cargo-local defaults; the flat bundle
                  uses the bundle dir.
      build_cwd — crate dir to `cargo build` in when the binary is missing
                  (dev only), or None when building isn't possible/needed.
    """
    # Explicit binary override always wins (dev experiments).
    env_bin = os.environ.get("ALPHAOW_BOT_BIN")
    if env_bin and os.path.isfile(env_bin):
        d = os.path.dirname(env_bin)
        return (env_bin, _opt(os.path.join(d, _NET_2P_NAME)),
                _opt(os.path.join(d, _NET_4P_NAME)), d, None)

    wd = _wrapper_dir()

    # Flat-bundle layout (Kaggle): the binary and both JSONs sit next to
    # main.py. Try the wrapper dir, the Kaggle agent mount, then cwd.
    for d in (wd, "/kaggle_simulations/agent", os.getcwd()):
        if d:
            b = _bin_in(d)
            if b:
                return (b, _opt(os.path.join(d, _NET_2P_NAME)),
                        _opt(os.path.join(d, _NET_4P_NAME)), d, None)

    # Dev layout: prometheus is self-contained, so this wrapper sits at the
    # crate root — binary under target/release, weights under train/weights/.
    crate = wd or os.getcwd()
    release = os.path.join(crate, "target", "release")
    # Use the existing binary if built; otherwise point at the name cargo will
    # produce on this platform so _build_if_needed writes there.
    binary = _bin_in(release) or os.path.join(
        release, _BIN_NAME + (".exe" if sys.platform == "win32" else "")
    )
    wdir = os.path.join(crate, "train", "weights")
    return (binary, _opt(os.path.join(wdir, _NET_2P_NAME)),
            _opt(os.path.join(wdir, _NET_4P_NAME)), crate, crate)


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
    binary, net_2p, net_4p, run_cwd, build_cwd = _locate()
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
    env["OW_ROLLOUT"] = "none"
    env["OW_ROLLOUT_DEPTH"] = "0"
    env.setdefault("ALPHAOW_BUDGET_MS", "900")

    if net_2p:
        env.setdefault("ALPHAOW_VALUE_NET_PATH_2P", net_2p)
        env.setdefault("ALPHAOW_VALUE_NET_PATH", net_2p)
    if net_4p:
        env.setdefault("ALPHAOW_VALUE_NET_PATH_4P", net_4p)
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
