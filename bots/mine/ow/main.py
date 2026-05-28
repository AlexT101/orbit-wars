"""Kaggle entry point. Spawns the Rust binary once, then for each turn pipes
the obs as one JSON line and reads one JSON line of moves back.

The binary path resolution order:
  1. $OW_BOT_BIN
  2. <this dir>/target/release/ow-bot
  3. <this dir>/target/debug/ow-bot

If the binary is missing we attempt to build it once with `cargo build --release`.
"""

import json
import os
import shutil
import subprocess
import sys
import threading

_PROC = None
_LOCK = threading.Lock()


def _here():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        env = os.environ.get("OW_BOT_DIR")
        if env:
            return env
        # Search a couple of likely spots.
        for cand in (os.getcwd(), os.path.join(os.getcwd(), "ow")):
            if os.path.isfile(os.path.join(cand, "Cargo.toml")):
                return cand
        return os.getcwd()


def _binary_path():
    env = os.environ.get("OW_BOT_BIN")
    if env and os.path.isfile(env):
        try:
            os.chmod(env, 0o755)
        except OSError:
            pass
        if os.access(env, os.X_OK):
            return env
    # Kaggle's submission unpacker doesn't always preserve subdirectory
    # structure or executable bits, so check the same dir first and chmod
    # +x anything that looks right.
    for sub in ("ow-bot", "target/release/ow-bot", "target/debug/ow-bot"):
        p = os.path.join(_here(), sub)
        if os.path.isfile(p):
            try:
                os.chmod(p, 0o755)
            except OSError:
                pass
            if os.access(p, os.X_OK):
                return p
    return None


def _build_binary():
    if shutil.which("cargo") is None:
        return None
    try:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=_here(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr.decode("utf-8", errors="replace"))
        return None
    return _binary_path()


def _ensure_proc():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    bin_path = _binary_path() or _build_binary()
    if bin_path is None:
        # Diagnostic dump so we can see what Kaggle actually extracted.
        try:
            d = _here()
            listing = []
            for root, dirs, files in os.walk(d):
                for f in files:
                    full = os.path.join(root, f)
                    try:
                        st = os.stat(full)
                        listing.append(f"{full} size={st.st_size} mode={oct(st.st_mode)}")
                    except OSError:
                        listing.append(f"{full} (stat failed)")
            sys.stderr.write(
                f"[ow2 wrapper] binary not found. _here={d}\n"
                + "\n".join(listing[:50])
                + "\n"
            )
        except Exception as e:
            sys.stderr.write(f"[ow2 wrapper] diagnostic dump failed: {e}\n")
        raise RuntimeError("ow-bot binary not found and `cargo build --release` failed")
    _PROC = subprocess.Popen(
        [bin_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=_here(),
        bufsize=0,
    )
    return _PROC


def _normalize_obs(obs):
    """Kaggle hands us a SimpleNamespace-ish object; flatten to plain dict."""
    if isinstance(obs, dict):
        get = obs.get
    else:
        def get(k, default=None):
            return getattr(obs, k, default)
    return {
        "player": get("player", 0),
        "step": get("step", 0),
        "planets": list(get("planets", []) or []),
        "fleets": list(get("fleets", []) or []),
        "angular_velocity": get("angular_velocity", 0.0),
        "initial_planets": list(get("initial_planets", []) or []),
        "comets": list(get("comets", []) or []),
        "comet_planet_ids": list(get("comet_planet_ids", []) or []),
    }


def agent(obs, config=None):
    payload = _normalize_obs(obs)
    if config is not None:
        cfg = {}
        for k in ("episodeSteps", "actTimeout", "shipSpeed", "sunRadius", "boardSize", "cometSpeed"):
            v = config.get(k) if isinstance(config, dict) else getattr(config, k, None)
            if v is not None:
                cfg[k] = v
        if cfg:
            payload["config"] = cfg

    with _LOCK:
        proc = _ensure_proc()
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        try:
            proc.stdin.write(line.encode("utf-8"))
            proc.stdin.flush()
            reply = proc.stdout.readline()
        except (BrokenPipeError, OSError):
            global _PROC
            _PROC = None
            return []
        if not reply:
            return []
        try:
            return json.loads(reply.decode("utf-8"))
        except json.JSONDecodeError:
            return []
