"""Build a Kaggle-ready submission bundle for alphaow_newest.

alphaow_newest is a subprocess-style bot: a Python wrapper (main.py)
spawns the `alphaow-bot` Rust binary and pipes one JSON observation per
turn. The bundle therefore needs three files at the archive root:

  - main.py          (Kaggle-flavored, looks for siblings in its own dir)
  - alphaow-bot      (Linux x86_64 static binary)
  - xgb_top10_d6.json (default value-net weights)

Build process:

  1. Cross-compile the Rust binary for x86_64-unknown-linux-musl using
     cargo-zigbuild (zig as the linker, musl libc for static linking).
     We chose musl over glibc because Kaggle's worker image's glibc
     version is variable; a musl static binary runs anywhere.
  2. Stage the three files into a temp dir, tar them up.

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
TARGET_TRIPLE = "x86_64-unknown-linux-musl"
BIN_OUT = ALPHAOW / "target" / TARGET_TRIPLE / "release" / "alphaow-bot"

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
import threading

_PROC = None
_LOCK = threading.Lock()

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Some Kaggle environments exec main.py without setting __file__.
    # The submission archive is unpacked into the agent's cwd, so fall
    # back to that.
    _HERE = os.getcwd()

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
        stderr=subprocess.DEVNULL,
        cwd=_HERE,
        env=env,
        bufsize=0,
    )
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
            proc.stdin.write((json.dumps(p, separators=(",", ":")) + "\\n").encode())
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

    print(f"Cross-compiling alphaow-bot for {TARGET_TRIPLE}...")
    rc = subprocess.run(
        ["cargo", "zigbuild", "--release", "--target", TARGET_TRIPLE,
         "--bin", "alphaow-bot"],
        cwd=ALPHAOW,
    ).returncode
    if rc != 0:
        sys.exit(f"cargo zigbuild failed (rc={rc})")
    if not BIN_OUT.is_file():
        sys.exit(f"build succeeded but binary missing: {BIN_OUT}")
    print(f"  binary: {BIN_OUT} ({BIN_OUT.stat().st_size:,} bytes)")

    bundle = HERE / "submission.tar.gz"
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
