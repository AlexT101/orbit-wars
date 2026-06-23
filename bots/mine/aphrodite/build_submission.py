"""Build a Kaggle-ready submission bundle for aphrodite.

aphrodite needs these files at the archive root:

  - main.py      (the dev wrapper with submission-only tweaks applied;
                  auto-detects the flat bundle layout at runtime — see
                  main.py's `_locate`)
  - aphrodite    (Linux x86_64 glibc binary, built in Kaggle image)
  - <2p weights> (required; filename taken from main.py's _WEIGHTS_2P_NAME)
  - <4p weights> (optional; filename taken from main.py's _WEIGHTS_4P_NAME)

Build process:

  1. Compile the Rust binary *inside Kaggle's own runtime image*
     (gcr.io/kaggle-images/python) so it links against the exact
     glibc/libstdc++ the submission worker will run it with. We used to
     musl-cross-compile with cargo-zigbuild on the host, but those
     binaries died silently on exec in the Kaggle sandbox (no stderr at
     all). Building in the runtime image is the same approach apollo uses
     and it Just Works.
  2. Stage the files into a temp dir, tar them up.

Requires Docker.

Usage:
    python bots/mine/aphrodite/build_submission.py

Output: bots/mine/aphrodite/submission.tar.gz
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APHRODITE = HERE
MAIN_PY = HERE / "main.py"

# main.py is the single source of truth for the value-net weight filenames and
# all timing limits; import it so we never duplicate those literals here.
sys.path.insert(0, str(HERE))
import main as aphrodite_main  # noqa: E402

# Weight filenames come straight from main.py's `_locate` names, so the bundle
# always ships exactly what the wrapper looks for next to itself. 2p is the
# required default (also the 4p fallback); 4p is optional.
WEIGHTS_2P_NAME = aphrodite_main._WEIGHTS_2P_NAME
WEIGHTS_4P_NAME = aphrodite_main._WEIGHTS_4P_NAME
WEIGHTS_2P = APHRODITE / "train" / "weights" / WEIGHTS_2P_NAME
WEIGHTS_4P = APHRODITE / "train" / "weights" / WEIGHTS_4P_NAME

# Apollo's tunable constants are read at runtime from these JSON files (2p +
# 4p). The binary's built-in fallback path is CARGO_MANIFEST_DIR, which on
# Kaggle is the (nonexistent) Docker build dir — so they ship next to main.py
# and main.py points APOLLO_CONFIG / APOLLO_CONFIG_4P at them. Both required.
CONFIG_2P_NAME = "config.json"
CONFIG_4P_NAME = "config_4p.json"
CONFIG_2P = APHRODITE / CONFIG_2P_NAME
CONFIG_4P = APHRODITE / CONFIG_4P_NAME

# The submission's main.py is the dev wrapper with a single literal flipped:
# `_USE_PROD_LIMITS = True`. main.py then applies the submission budget and
# overage-pool usage itself (so those values live only in main.py). Otherwise
# the wrapper is verbatim and auto-detects the flat-bundle layout at runtime
# (see main.py's `_locate`).

# Pin a digest in production for reproducibility; `latest` keeps the
# scripts simple while we iterate.
KAGGLE_IMAGE = "gcr.io/kaggle-images/python:latest"

# A target dir separate from the host's `target/` so Linux artifacts
# built in Docker never collide with Windows host builds in the same
# crate. The glibc binary lands at target-docker/release/aphrodite.
BIN_OUT = APHRODITE / "target-docker" / "release" / "aphrodite"

# Installs Rust inside the Kaggle image (CARGO_HOME/RUSTUP_HOME under the
# mounted dir so they're writable and cached between runs), then does a
# plain glibc release build of just the aphrodite binary.
BUILD_SCRIPT = r"""
set -euo pipefail
export CARGO_HOME=/io/.cargo-home
export RUSTUP_HOME=/io/.rustup-home
export CARGO_TARGET_DIR=/io/target-docker
export PATH=$CARGO_HOME/bin:$PATH
# Target the x86-64-v3 microarch level (AVX2/FMA/BMI2; Haswell 2013+ / Zen+),
# which Kaggle's GCP workers support, to vectorize the float-heavy aim/pathing/
# value-net math. Behavior-preserving (Rust keeps IEEE semantics; no implicit
# FMA contraction). RISK: if a worker lacks AVX2 the binary dies with SIGILL on
# the first such instruction — the bot then no-ops every turn and loses, visible
# in the episode logs. Drop to "x86-64-v2" (SSE4.2; ~universal) if that happens.
export RUSTFLAGS="-C target-cpu=x86-64-v3"
if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
cargo build --release --features submission --bin aphrodite
"""


def main() -> int:
    if not WEIGHTS_2P.is_file():
        sys.exit(f"2p weights missing (required): expected {WEIGHTS_2P}")
    for cfg in (CONFIG_2P, CONFIG_4P):
        if not cfg.is_file():
            sys.exit(f"runtime config missing (required): expected {cfg}")

    print(f"Building aphrodite inside {KAGGLE_IMAGE}...")
    rc = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{APHRODITE}:/io", "-w", "/io",
         KAGGLE_IMAGE, "bash", "-c", BUILD_SCRIPT],
    ).returncode
    if rc != 0:
        sys.exit(f"docker build failed (rc={rc})")
    if not BIN_OUT.is_file():
        sys.exit(f"build succeeded but binary missing: {BIN_OUT}")
    print(f"  binary: {BIN_OUT} ({BIN_OUT.stat().st_size:,} bytes)")

    if not MAIN_PY.is_file():
        sys.exit(f"wrapper missing: {MAIN_PY}")

    bundle = HERE / "submission.tar.gz"
    # Remove any stale bundle first: rewriting in place has been observed to
    # leave trailing bytes past the gzip stream when the previous file was
    # larger, producing a "trailing garbage" tarball.
    bundle.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Enable production limits in the bundled wrapper by flipping the single
        # _USE_PROD_LIMITS toggle (main.py applies the submission budget +
        # overage itself). Fail loudly if the expected line is missing so we
        # never silently ship dev limits.
        wrapper_src = MAIN_PY.read_text(encoding="utf-8")
        old_line = "_USE_PROD_LIMITS = False"
        new_line = "_USE_PROD_LIMITS = True"
        if wrapper_src.count(old_line) != 1:
            sys.exit(f"expected exactly one {old_line!r} in main.py to enable prod limits")
        wrapper_src = wrapper_src.replace(old_line, new_line)
        print("  submission tweak: _USE_PROD_LIMITS -> True (prod budget + overage)")
        (td / "main.py").write_text(wrapper_src, encoding="utf-8")
        shutil.copy(BIN_OUT, td / "aphrodite")
        os.chmod(td / "aphrodite", 0o755)
        staged = ["main.py", "aphrodite"]
        for src, name in (
            (WEIGHTS_2P, WEIGHTS_2P_NAME),
            (WEIGHTS_4P, WEIGHTS_4P_NAME),
            (CONFIG_2P, CONFIG_2P_NAME),
            (CONFIG_4P, CONFIG_4P_NAME),
        ):
            if src.is_file():
                shutil.copy(src, td / name)
                staged.append(name)
        with tarfile.open(bundle, "w:gz") as tar:
            for name in staged:
                tar.add(td / name, arcname=name)
    print(f"Wrote {bundle} ({bundle.stat().st_size:,} bytes)")
    print()
    print("Contents:")
    with tarfile.open(bundle) as tar:
        for m in tar.getmembers():
            print(f"  {m.name:<24}  {m.size:>10,} bytes  mode=0o{m.mode:o}")
    print()
    print("Submit (when ready):")
    print(f"  kaggle competitions submit orbit-wars -f {bundle} -m 'aphrodite'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
