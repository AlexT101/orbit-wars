"""Build a Kaggle-ready submission bundle for alphaow_newest.

alphaow_newest is a subprocess-style bot: a Python wrapper (main.py)
spawns the `alphaow-bot` Rust binary and pipes one JSON observation per
turn. The bundle therefore needs three files at the archive root:

  - main.py          (the dev wrapper verbatim; auto-detects the flat
                      bundle layout at runtime — see main.py's `_locate`)
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
