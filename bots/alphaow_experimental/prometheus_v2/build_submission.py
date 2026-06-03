"""Build a Kaggle-ready submission bundle for prometheus.

The bundle has four files at the archive root:

  - main.py                       (the dev wrapper verbatim; auto-detects the
                                   flat bundle layout at runtime — see
                                   main.py's `_locate`)
  - alphaow-bot                   (Linux x86_64 glibc binary, built in Kaggle image)
  - xgb_46p12e88t11_latest.json   (2-player value-net weights)
  - xgb_4p_v2_rank4_latest.json   (4-player value-net weights)

Build process:

  1. Compile the Rust binary *inside Kaggle's own runtime image*
     (gcr.io/kaggle-images/python) so it links against the exact
     glibc/libstdc++ the submission worker will run it with. Building
     in the runtime image is the approach apollo and alphaow_newest use
     and it Just Works; host musl cross-builds died silently on exec in
     the Kaggle sandbox.
  2. Stage the four files into a temp dir, tar them up.

Requires Docker.

Usage:
    python bots/alphaow_experimental/prometheus/build_submission.py

Output: bots/alphaow_experimental/prometheus/submission.tar.gz
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# prometheus is self-contained: the crate root is this directory.
HERE = Path(__file__).resolve().parent
CRATE = HERE
WEIGHTS_DIR = CRATE / "train" / "weights"

# Value nets to bundle. main.py routes the 2P net by default and the 4P net
# for 4-player games. These names mirror the defaults baked into main.py.
WEIGHTS_2P = WEIGHTS_DIR / "xgb_46p12e88t11_latest.json"
WEIGHTS_4P = WEIGHTS_DIR / "xgb_4p_v2_rank4_latest.json"

# The submission's main.py is the dev wrapper verbatim — it auto-detects the
# flat-bundle layout at runtime, so there is no second Kaggle-only copy to keep
# in sync. (See main.py's `_locate`.)
MAIN_PY = HERE / "main.py"

# Pin a digest in production for reproducibility; `latest` keeps the
# scripts simple while we iterate.
KAGGLE_IMAGE = "gcr.io/kaggle-images/python:latest"

# A target dir separate from the host's `target/` so Linux artifacts
# built in Docker never collide with Windows host builds in the same
# crate. The glibc binary lands at target-docker/release/alphaow-bot.
BIN_OUT = CRATE / "target-docker" / "release" / "alphaow-bot"

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


def main() -> int:
    for w in (WEIGHTS_2P, WEIGHTS_4P):
        if not w.is_file():
            sys.exit(f"weights file missing: {w}")

    print(f"Building alphaow-bot inside {KAGGLE_IMAGE}...")
    rc = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{CRATE}:/io", "-w", "/io",
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
    members = ("main.py", "alphaow-bot", WEIGHTS_2P.name, WEIGHTS_4P.name)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(MAIN_PY, td / "main.py")
        shutil.copy(BIN_OUT, td / "alphaow-bot")
        os.chmod(td / "alphaow-bot", 0o755)
        shutil.copy(WEIGHTS_2P, td / WEIGHTS_2P.name)
        shutil.copy(WEIGHTS_4P, td / WEIGHTS_4P.name)
        with tarfile.open(bundle, "w:gz") as tar:
            for name in members:
                tar.add(td / name, arcname=name)
    print(f"Wrote {bundle} ({bundle.stat().st_size:,} bytes)")
    print()
    print("Contents:")
    with tarfile.open(bundle) as tar:
        for m in tar.getmembers():
            print(f"  {m.name:<30}  {m.size:>10,} bytes  mode=0o{m.mode:o}")
    print()
    print("Submit (when ready):")
    print(f"  kaggle competitions submit orbit-wars -f {bundle} -m 'prometheus'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
