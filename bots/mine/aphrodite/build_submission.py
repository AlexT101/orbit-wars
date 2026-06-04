"""Build a Kaggle-ready submission bundle for aphrodite.

aphrodite needs these files at the archive root:

  - main.py                  (the dev wrapper verbatim; auto-detects the flat
                              bundle layout at runtime — see main.py's `_locate`)
  - aphrodite              (Linux x86_64 glibc binary, built in Kaggle image)
  - xgb_2p_old_top10.json  (fallback fixed-extrapolation value-net weights)
  - xgb_2p.json              (optional 2-player value-net weights)
  - xgb_4p.json              (optional 4-player value-net weights)

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
WEIGHTS = APHRODITE / "train" / "weights" / "xgb_2p_old_top10.json"
WEIGHTS_2P = APHRODITE / "train" / "weights" / "xgb_2p.json"
WEIGHTS_4P = APHRODITE / "train" / "weights" / "xgb_4p.json"
# The submission's main.py is the dev wrapper, copied with ONE tweak: the
# per-turn budget default is raised to SUBMISSION_BUDGET_MS (the Kaggle worker
# allows more time per turn than local dev, where main.py defaults to 500ms).
# Otherwise it is verbatim and auto-detects the flat-bundle layout at runtime
# (see main.py's `_locate`).
MAIN_PY = HERE / "main.py"
SUBMISSION_BUDGET_MS = "1000"
# Name the weights land under inside the flat bundle. main.py's _locate looks
# for exactly this name next to itself.
WEIGHTS_NAME = "xgb_2p_old_top10.json"
WEIGHTS_2P_NAME = "xgb_2p.json"
WEIGHTS_4P_NAME = "xgb_4p.json"

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
    if not any(p.is_file() for p in (WEIGHTS, WEIGHTS_2P, WEIGHTS_4P)):
        sys.exit(
            "weights missing: expected at least one of "
            f"{WEIGHTS}, {WEIGHTS_2P}, or {WEIGHTS_4P}"
        )

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
        # Copy the wrapper, raising the per-turn budget default for the
        # submission. Fail loudly if the expected line is missing so we never
        # silently ship the dev budget.
        wrapper_src = MAIN_PY.read_text(encoding="utf-8")
        old_line = 'env.setdefault("APHRODITE_BUDGET_MS", "500")'
        new_line = f'env.setdefault("APHRODITE_BUDGET_MS", "{SUBMISSION_BUDGET_MS}")'
        if wrapper_src.count(old_line) != 1:
            sys.exit(f"expected exactly one {old_line!r} in main.py to bump for the submission")
        (td / "main.py").write_text(wrapper_src.replace(old_line, new_line), encoding="utf-8")
        print(f"  bumped submission budget to {SUBMISSION_BUDGET_MS}ms (dev main.py stays 500ms)")
        shutil.copy(BIN_OUT, td / "aphrodite")
        os.chmod(td / "aphrodite", 0o755)
        staged = ["main.py", "aphrodite"]
        for src, name in (
            (WEIGHTS, WEIGHTS_NAME),
            (WEIGHTS_2P, WEIGHTS_2P_NAME),
            (WEIGHTS_4P, WEIGHTS_4P_NAME),
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
