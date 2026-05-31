"""Build a Kaggle-ready submission bundle for alphaow_newest.

alphaow_newest is a subprocess-style bot: a Python wrapper (main.py)
spawns the `alphaow-bot` Rust binary and pipes one JSON observation per
turn. The bundle therefore needs three files at the archive root:

  - main.py          (Kaggle-flavored, looks for siblings in its own dir)
  - alphaow-bot      (Linux x86_64 glibc binary, built in Kaggle image)
  - xgb_top10_d6.json (default value-net weights)

Build process:

  1. Compile the Rust binary *inside Kaggle's own runtime image*
     (gcr.io/kaggle-images/python) so it links against the exact
     glibc/libstdc++ the submission worker will run it with. We used to
     musl-cross-compile with cargo-zigbuild on the host, but those
     binaries died silently on exec in the Kaggle sandbox (no stderr at
     all). Building in the runtime image is the same approach apollo uses
     and it Just Works.
  2. Stage the three files into a temp dir, tar them up.

Requires Docker.

Usage:
    python bots/mine/alphaow_newest/build_submission.py

Output: bots/mine/alphaow_newest/submission.tar.gz
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALPHAOW = (HERE / ".." / "alphaow").resolve()
WEIGHTS = ALPHAOW / "train" / "weights" / "xgb_top10_d6.json"

# Pin a digest in production for reproducibility; `latest` keeps the
# scripts simple while we iterate.
KAGGLE_IMAGE = "gcr.io/kaggle-images/python:latest"

# A target dir separate from the host's `target/` so Linux artifacts
# built in Docker never collide with Windows host builds in the same
# crate. The glibc binary lands at target-docker/release/alphaow-bot.
BIN_OUT = ALPHAOW / "target-docker" / "release" / "alphaow-bot"

# Installs Rust inside the Kaggle image (CARGO_HOME/RUSTUP_HOME under the
# mounted dir so they're writable and cached between runs), then does a
# plain glibc release build of just the alphaow-bot binary.
BUILD_SCRIPT = r"""
set -euo pipefail
export CARGO_HOME=/io/.cargo-home
export RUSTUP_HOME=/io/.rustup-home
export CARGO_TARGET_DIR=/io/target-docker
export PATH=$CARGO_HOME/bin:$PATH
if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
cargo build --release --bin alphaow-bot
"""

# Kaggle-flavored main.py — strips OW_* tuning env (so the wrapper is a
# stable baseline regardless of caller env), spawns the binary at
# ./alphaow-bot, points the value-net path at ./xgb_top10_d6.json. Same
# protocol as the dev wrapper (one JSON observation per line on stdin,
# one JSON moves array per line on stdout).
KAGGLE_MAIN_PY = r'''"""alphaow_newest — Kaggle submission entry point.

Spawns the bundled `alphaow-bot` Rust binary once and pipes one JSON
observation per turn. The binary, weights file (xgb_top10_d6.json), and
this main.py all live at the same directory inside the submission
archive.
"""

import json
import os
import stat
import subprocess
import sys
import threading

_PROC = None
_LOCK = threading.Lock()


def _pump_stderr(pipe):
    # Forward the binary's stderr into our own stderr line by line. Kaggle
    # captures the agent process's stderr (that's how panics show up in the
    # logs), but it wraps sys.stderr in an object with no real file
    # descriptor, so we can't hand the fd to Popen directly. Pump it here.
    try:
        for line in iter(pipe.readline, b""):
            try:
                sys.stderr.write(line.decode("utf-8", "replace"))
                sys.stderr.flush()
            except Exception:
                pass
    except Exception:
        pass

def _find_bundle_dir():
    # The submission archive (main.py, alphaow-bot, xgb_top10_d6.json) is
    # unpacked into a single dir, but which dir depends on the Kaggle
    # runtime. __file__ is the most reliable when set, but some exec
    # contexts don't define it, so search known candidates for the binary.
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    candidates.append("/kaggle_simulations/agent")
    candidates.append(os.getcwd())
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "alphaow-bot")):
            return d
    # Nothing matched; fall back to the first candidate so error messages
    # point somewhere sensible.
    return candidates[0] if candidates else os.getcwd()


_HERE = _find_bundle_dir()
_BIN = os.path.join(_HERE, "alphaow-bot")
_WEIGHTS = os.path.join(_HERE, "xgb_top10_d6.json")


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
    _ensure_executable(_BIN)
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
    # No rollouts — leaf-eval with XGB. Proven 8W-2L vs production MLP.
    env["OW_ROLLOUT"] = "none"
    env["OW_ROLLOUT_DEPTH"] = "0"
    if os.path.isfile(_WEIGHTS):
        env.setdefault("ALPHAOW_VALUE_NET_PATH", _WEIGHTS)
    _PROC = subprocess.Popen(
        [_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=_HERE,
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
        for k in ("episodeSteps", "actTimeout", "shipSpeed",
                  "sunRadius", "boardSize", "cometSpeed"):
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
'''


def main() -> int:
    if not WEIGHTS.is_file():
        sys.exit(f"weights file missing: {WEIGHTS}")

    print(f"Building alphaow-bot inside {KAGGLE_IMAGE}...")
    rc = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{ALPHAOW}:/io", "-w", "/io",
         KAGGLE_IMAGE, "bash", "-c", BUILD_SCRIPT],
    ).returncode
    if rc != 0:
        sys.exit(f"docker build failed (rc={rc})")
    if not BIN_OUT.is_file():
        sys.exit(f"build succeeded but binary missing: {BIN_OUT}")
    print(f"  binary: {BIN_OUT} ({BIN_OUT.stat().st_size:,} bytes)")

    bundle = HERE / "submission.tar.gz"
    # Remove any stale bundle first: rewriting in place has been observed to
    # leave trailing bytes past the gzip stream when the previous file was
    # larger, producing a "trailing garbage" tarball.
    bundle.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "main.py").write_text(KAGGLE_MAIN_PY, encoding="utf-8")
        shutil.copy(BIN_OUT, td / "alphaow-bot")
        os.chmod(td / "alphaow-bot", 0o755)
        shutil.copy(WEIGHTS, td / "xgb_top10_d6.json")
        with tarfile.open(bundle, "w:gz") as tar:
            for name in ("main.py", "alphaow-bot", "xgb_top10_d6.json"):
                tar.add(td / name, arcname=name)
    print(f"Wrote {bundle} ({bundle.stat().st_size:,} bytes)")
    print()
    print("Contents:")
    with tarfile.open(bundle) as tar:
        for m in tar.getmembers():
            print(f"  {m.name:<24}  {m.size:>10,} bytes  mode=0o{m.mode:o}")
    print()
    print("Submit (when ready):")
    print(f"  kaggle competitions submit orbit-wars -f {bundle} -m 'alphaow_newest'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
