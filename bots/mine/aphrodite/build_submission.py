"""Build a Kaggle-ready submission bundle for aphrodite.

aphrodite is the aphrodite Rust bot paired with the *fixed-extrapolation*
XGB model (xgb_top10_d6_fixed.json, retrained 2026-05-30). It is a
subprocess-style bot: a Python wrapper (main.py) spawns the `aphrodite`
Rust binary and pipes one JSON observation per turn. The bundle therefore
needs three files at the archive root:

  - main.py                  (the dev wrapper verbatim; auto-detects the flat
                              bundle layout at runtime — see main.py's `_locate`)
  - aphrodite              (Linux x86_64 glibc binary, built in Kaggle image)
  - xgb_top10_d6_fixed.json  (fixed-extrapolation value-net weights)

The only differences from aphrodite_newest's bundle are the weights file
(the *_fixed.json variant) and that main.py sets APHRODITE_EXTRAP_FIX=1 so
the bot's extrapolate_fleets matches the training feature extraction.

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
WEIGHTS = APHRODITE / "train" / "weights" / "xgb_top10_d6_fixed.json"
# The submission's main.py is the dev wrapper verbatim — it auto-detects the
# flat-bundle layout at runtime, so there is no second Kaggle-only copy to keep
# in sync. (See main.py's `_locate`.)
MAIN_PY = HERE / "main.py"
# Name the weights land under inside the flat bundle. main.py's _locate looks
# for exactly this name next to itself.
WEIGHTS_NAME = "xgb_top10_d6_fixed.json"

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
if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
cargo build --release --bin aphrodite
"""


def main() -> int:
    if not WEIGHTS.is_file():
        sys.exit(f"weights file missing: {WEIGHTS}")

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
        shutil.copy(MAIN_PY, td / "main.py")
        shutil.copy(BIN_OUT, td / "aphrodite")
        os.chmod(td / "aphrodite", 0o755)
        shutil.copy(WEIGHTS, td / WEIGHTS_NAME)
        with tarfile.open(bundle, "w:gz") as tar:
            for name in ("main.py", "aphrodite", WEIGHTS_NAME):
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
