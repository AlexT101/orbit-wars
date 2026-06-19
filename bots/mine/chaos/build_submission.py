"""Build a Kaggle-ready submission bundle for chaos.

Bundle layout (flat, everything at the archive root):

  main.py                    chaos wrapper (verbatim — detects the flat layout)
  aphrodite_wrapper.py       copy of bots/mine/aphrodite/main.py (subprocess
                             management; its _locate finds the binary/weights
                             next to itself)
  aphrodite                  Linux glibc binary built inside Kaggle's own image
  xgb_*.json                 value-net weights (names from aphrodite's main.py)
  config*.json               apollo runtime configs read by the aphrodite binary
  osteo_il_2p_latest.pt      2p IL checkpoint
  features.py, model.py, constants.py
                             IL support (from experimental_arch/train_transformer)
  orbit_wars_model.abi3.so   Rust feature encoder (host-built, like triplepoint1)
  orbit_wars_engine.abi3.so  Rust engine (for chaos's schema validation + warmup)

torch and gymnasium are NOT bundled — Kaggle's python image ships them.

Build steps:
  1. cargo-build env_model + env_engine on the host, take the cdylibs.
  2. docker-build the aphrodite binary inside gcr.io/kaggle-images/python
     (reuses aphrodite/target-docker; pass --skip-docker to reuse an existing
     binary there).
  3. Stage everything flat in a temp dir.
  4. SMOKE TEST: import the staged main.py and play one real turn through the
     staged binary + staged .so's. The build fails loudly if anything is off.
  5. tar.gz it.

Usage:
    python bots/mine/chaos/build_submission.py [--skip-docker]

Output: bots/mine/chaos/submission.tar.gz
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
APHRODITE = REPO_ROOT / "bots" / "mine" / "aphrodite"
TRAIN_DIR = REPO_ROOT / "experimental_arch" / "train_transformer"
ENV_MODEL = REPO_ROOT / "experimental_arch" / "env_model"
ENV_ENGINE = REPO_ROOT / "experimental_arch" / "env_engine"
CHECKPOINT_2P = (
    REPO_ROOT
    / "experimental_arch"
    / "imitation_learning"
    / "checkpoints"
    / "osteo_bc_transformer"
    / "latest.pt"
)
BUNDLE = HERE / "submission.tar.gz"

# Weight filenames come from aphrodite's main.py (single source of truth).
sys.path.insert(0, str(APHRODITE))
import main as aphrodite_main  # noqa: E402

sys.path.pop(0)
WEIGHTS = [
    APHRODITE / "train" / "weights" / aphrodite_main._WEIGHTS_2P_NAME,
    APHRODITE / "train" / "weights" / aphrodite_main._WEIGHTS_4P_NAME,
]
CONFIGS = [
    APHRODITE / "config.json",
    APHRODITE / "config_4p.json",
]

# Build the binary against glibc 2.31 (bullseye): older than any plausible
# Kaggle runtime glibc, and binaries built against an older glibc run on newer
# ones. aphrodite's script builds inside the full Kaggle image instead, but
# that image is a tens-of-GB pull; rust:slim-bullseye is ~250MB and gives the
# same compatibility guarantee. (musl cross-builds are NOT safe — they died
# silently on the Kaggle worker; see aphrodite/build_submission.py.)
BUILD_IMAGE = "rust:1-slim-bullseye"
BIN_OUT = APHRODITE / "target-docker" / "release" / "aphrodite"
DOCKER_BUILD_SCRIPT = r"""
set -euo pipefail
export CARGO_HOME=/io/.cargo-home-bullseye
export CARGO_TARGET_DIR=/io/target-docker
cargo build --release --features submission --bin aphrodite
"""


def _run(cmd: list[str], **kw) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)


def _build_cdylibs() -> dict[str, Path]:
    cargo = shutil.which("cargo")
    if not cargo:
        sys.exit("cargo not found; cannot build orbit_wars_model/engine")
    out = {}
    for crate, lib, name in (
        (ENV_MODEL, "liborbit_wars_model.so", "orbit_wars_model.abi3.so"),
        (ENV_ENGINE, "liborbit_wars_engine.so", "orbit_wars_engine.abi3.so"),
    ):
        _run([cargo, "build", "--release"], cwd=crate)
        src = crate / "target" / "release" / lib
        if not src.is_file():
            sys.exit(f"build succeeded but cdylib missing: {src}")
        out[name] = src
    return out


def _build_binary(skip_docker: bool) -> None:
    if skip_docker:
        if not BIN_OUT.is_file():
            sys.exit(f"--skip-docker but no binary at {BIN_OUT}")
        print(f"  reusing existing docker-built binary: {BIN_OUT}")
        return
    print(f"Building aphrodite inside {BUILD_IMAGE}...")
    _run(
        ["docker", "run", "--rm", "-v", f"{APHRODITE}:/io", "-w", "/io",
         BUILD_IMAGE, "bash", "-c", DOCKER_BUILD_SCRIPT],
    )
    if not BIN_OUT.is_file():
        sys.exit(f"docker build succeeded but binary missing: {BIN_OUT}")


def _with_prod_limits(src: Path, label: str) -> str:
    text = src.read_text(encoding="utf-8")
    old_line = "_USE_PROD_LIMITS = False"
    new_line = "_USE_PROD_LIMITS = True"
    if text.count(old_line) != 1:
        sys.exit(f"expected exactly one {old_line!r} in {src}")
    print(f"  submission tweak: {label} _USE_PROD_LIMITS -> True")
    return text.replace(old_line, new_line)


def _stage(td: Path, cdylibs: dict[str, Path]) -> list[str]:
    staged = []
    (td / "main.py").write_text(
        _with_prod_limits(HERE / "main.py", "chaos"),
        encoding="utf-8",
    )
    staged.append("main.py")
    (td / "aphrodite_wrapper.py").write_text(
        _with_prod_limits(APHRODITE / "main.py", "aphrodite wrapper"),
        encoding="utf-8",
    )
    staged.append("aphrodite_wrapper.py")

    files: list[tuple[Path, str]] = [
        (BIN_OUT, "aphrodite"),
        (CHECKPOINT_2P, "osteo_il_2p_latest.pt"),
        (TRAIN_DIR / "features.py", "features.py"),
        (TRAIN_DIR / "model.py", "model.py"),
        (TRAIN_DIR / "constants.py", "constants.py"),
    ]
    files += [(cfg, cfg.name) for cfg in CONFIGS]
    files += [(w, w.name) for w in WEIGHTS if w.is_file()]
    files += [(src, name) for name, src in cdylibs.items()]
    for src, name in files:
        if not src.is_file():
            sys.exit(f"missing source file: {src}")
        shutil.copy2(src, td / name)
        staged.append(name)
    os.chmod(td / "aphrodite", 0o755)
    return staged


SMOKE_SCRIPT = r"""
import json, sys, time
sys.path.insert(0, ".")
t0 = time.perf_counter()
import main  # eager init: torch + schema validation + warmup; checkpoint loads lazily
print(f"[smoke] import+init: {(time.perf_counter()-t0)*1000:.0f}ms", file=sys.stderr)
assert main._BUNDLE, "staged main.py did not detect the flat bundle layout"
from orbit_wars_engine import OrbitWarsEngine
eng = OrbitWarsEngine(num_players=2)
state = eng.reset(seed=123)
# Turn 0 includes the one-time binary spawn + XGB weight parse; it may exceed
# 1s and land on Kaggle's 60s overage pool (by design — same as aphrodite).
# Steady-state turns should stay close to the 1s act timeout. Allow a little
# smoke-test jitter because prod Chaos intentionally targets the full turn and
# Kaggle has an overage pool for occasional spills.
for turn in range(3):
    obs = state["observations"][0]
    obs.setdefault("player", 0)
    t0 = time.perf_counter()
    moves = main.agent(obs, {"actTimeout": 1, "episodeSteps": 500})
    dt = (time.perf_counter() - t0) * 1000
    print(f"[smoke] turn{turn}: {dt:.0f}ms moves={json.dumps(moves)}", file=sys.stderr)
    assert isinstance(moves, list), f"agent returned {type(moves)}"
    if turn > 0:
        assert dt < 1100, f"steady-state turn took {dt:.0f}ms — unexpectedly slow"
    state = eng.step([moves, []])
proc = getattr(main._aph, "_PROC", None)
if proc is not None:
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        proc.kill()
    main._aph._PROC = None
eng4 = OrbitWarsEngine(num_players=4)
state4 = eng4.reset(seed=456)
obs4 = state4["observations"][0]
obs4.setdefault("player", 0)
t0 = time.perf_counter()
moves4 = main.agent(obs4, {"actTimeout": 1, "episodeSteps": 500})
dt4 = (time.perf_counter() - t0) * 1000
print(f"[smoke] 4p turn0: {dt4:.0f}ms moves={json.dumps(moves4)}", file=sys.stderr)
assert isinstance(moves4, list), f"4p agent returned {type(moves4)}"
print("[smoke] OK", file=sys.stderr)
"""


def _smoke_test(td: Path) -> None:
    print("Smoke-testing staged bundle (import + one real turn)...")
    env = dict(os.environ)
    env.pop("OSTEO_ORBIT_WARS_ROOT", None)
    env["OW_DEBUG"] = "1"
    subprocess.run(
        [sys.executable, "-c", SMOKE_SCRIPT], cwd=td, env=env, check=True
    )


def main() -> int:
    skip_docker = "--skip-docker" in sys.argv
    print("Building orbit_wars_model + orbit_wars_engine cdylibs...")
    cdylibs = _build_cdylibs()
    _build_binary(skip_docker)
    BUNDLE.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        staged = _stage(td, cdylibs)
        _smoke_test(td)
        with tarfile.open(BUNDLE, "w:gz") as tar:
            for name in staged:
                tar.add(td / name, arcname=name)
    print(f"\nWrote {BUNDLE} ({BUNDLE.stat().st_size:,} bytes)\nContents:")
    with tarfile.open(BUNDLE) as tar:
        for m in tar.getmembers():
            print(f"  {m.name:<28} {m.size:>12,} bytes  mode=0o{m.mode:o}")
    print(f"\nSubmit (when ready):\n  kaggle competitions submit orbit-wars -f {BUNDLE} -m 'chaos'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
