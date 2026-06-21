"""Build a flat Kaggle submission bundle for chaos.

The bundle contains the Chaos wrapper, Aphrodite wrapper/binary, value-net
weights, Apollo configs, IL support modules, runtime IL checkpoints, and native
Orbit Wars extension modules. Torch and gymnasium are expected from the Kaggle
Python image.

Usage:
    python bots/mine/chaos/build_submission_docker.py [--skip-docker]

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
_IL_CHECKPOINT_DIR = REPO_ROOT / "experimental_arch" / "imitation_learning" / "checkpoints"
CHECKPOINT_2P = _IL_CHECKPOINT_DIR / "osteo_bc_transformer" / "latest.pt"
CHECKPOINT_4P = _IL_CHECKPOINT_DIR / "osteo_il_4p_latest.pt"
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

# Build the Rust daemon in a Linux container for submission compatibility.
BUILD_IMAGE = "rust:1-slim-bullseye"
KAGGLE_IMAGE = "gcr.io/kaggle-images/python:latest"
BIN_OUT = APHRODITE / "target-docker" / "release" / "aphrodite"
DOCKER_BUILD_SCRIPT = r"""
set -euo pipefail
export CARGO_HOME=/io/.cargo-home-bullseye
export CARGO_TARGET_DIR=/io/target-docker
cargo build --release --features submission --bin aphrodite
"""
DOCKER_CDYLIB_SCRIPT = r"""
set -euo pipefail
export CARGO_HOME=/io/.cargo-home-kaggle
export RUSTUP_HOME=/io/.rustup-home-kaggle
export PATH=$CARGO_HOME/bin:$PATH
if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
cd /io/experimental_arch/env_model
export CARGO_TARGET_DIR=/io/experimental_arch/env_model/target-docker
cargo build --release
cd /io/experimental_arch/env_engine
export CARGO_TARGET_DIR=/io/experimental_arch/env_engine/target-docker
cargo build --release
"""


def _run(cmd: list[str], **kw) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)


def _cdylib_outputs() -> dict[str, Path]:
    return {
        "orbit_wars_model.abi3.so": ENV_MODEL
        / "target-docker"
        / "release"
        / "liborbit_wars_model.so",
        "orbit_wars_engine.abi3.so": ENV_ENGINE
        / "target-docker"
        / "release"
        / "liborbit_wars_engine.so",
    }


def _build_cdylibs(skip_docker: bool) -> dict[str, Path]:
    out = _cdylib_outputs()
    if skip_docker:
        missing = [str(path) for path in out.values() if not path.is_file()]
        if missing:
            sys.exit("--skip-docker but cdylib(s) missing:\n  " + "\n  ".join(missing))
        for name, src in out.items():
            print(f"  reusing existing docker-built cdylib: {src} -> {name}")
        return out

    print(f"Building orbit_wars_model + orbit_wars_engine inside {KAGGLE_IMAGE}...")
    _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:/io",
            "-w",
            "/io",
            KAGGLE_IMAGE,
            "bash",
            "-c",
            DOCKER_CDYLIB_SCRIPT,
        ],
    )
    missing = [str(path) for path in out.values() if not path.is_file()]
    if missing:
        sys.exit("docker build succeeded but cdylib(s) missing:\n  " + "\n  ".join(missing))
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


def _write_runtime_checkpoint(src: Path, dst: Path) -> None:
    import torch

    if not src.is_file():
        sys.exit(f"missing source file: {src}")
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    runtime_checkpoint = {
        "format": "chaos_il_runtime_v1",
        "model": checkpoint["model"],
        "config": checkpoint.get("config", {}),
    }
    torch.save(runtime_checkpoint, dst)


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

    _write_runtime_checkpoint(CHECKPOINT_2P, td / "osteo_il_2p_latest.pt")
    staged.append("osteo_il_2p_latest.pt")
    _write_runtime_checkpoint(CHECKPOINT_4P, td / "osteo_il_4p_latest.pt")
    staged.append("osteo_il_4p_latest.pt")

    files: list[tuple[Path, str]] = [
        (BIN_OUT, "aphrodite"),
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
import main
print(f"[smoke] import+init: {(time.perf_counter()-t0)*1000:.0f}ms", file=sys.stderr)
assert main._BUNDLE, "staged main.py did not detect the flat bundle layout"
from orbit_wars_engine import OrbitWarsEngine
eng = OrbitWarsEngine(num_players=2)
state = eng.reset(seed=123)
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
    print(f"Smoke-testing staged Linux bundle inside {KAGGLE_IMAGE}...")
    _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{td}:/tmp/chaos_bundle",
            "-w",
            "/tmp/chaos_bundle",
            "-e",
            "OW_DEBUG=1",
            KAGGLE_IMAGE,
            "python",
            "-c",
            SMOKE_SCRIPT,
        ],
    )


def main() -> int:
    skip_docker = "--skip-docker" in sys.argv
    cdylibs = _build_cdylibs(skip_docker)
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
