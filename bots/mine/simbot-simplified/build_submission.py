"""
Build a Kaggle-ready submission bundle for simbot-simplified.

Kaggle runs agents on Linux x86_64 and will not compile Rust, so we build the
native module *inside Kaggle's own runtime image* (gcr.io/kaggle-images/python)
to guarantee an exact glibc/libstdc++/ABI match. Earlier versions built in a
manylinux container; the resulting wheel imported and ran one turn fine, then
mysteriously stopped being called — a near-textbook symptom of subtle
runtime-mismatch breakage. Building in the runtime image rules that out.

Usage:
    python bots/mine/simbot-simplified/build_submission.py

Requires Docker. Output: bots/mine/simbot-simplified/submission.tar.gz
"""

import os
import subprocess
import sys
import tarfile
import zipfile
from glob import glob
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Pin a digest in production for reproducibility; `latest` here keeps the
# scripts simple while we iterate.
KAGGLE_IMAGE = "gcr.io/kaggle-images/python:latest"

# Installs Rust + maturin inside the Kaggle image, then builds an abi3 wheel
# tied to *this* image's libc/libstdc++. `--compatibility off` skips the
# manylinux audit and emits a `linux_x86_64` wheel — non-portable to other
# distros, but a perfect match for Kaggle's submission runtime.
BUILD_SCRIPT = r"""
set -euo pipefail
export CARGO_HOME=/io/.cargo-home
export RUSTUP_HOME=/io/.rustup-home
export PATH=$CARGO_HOME/bin:$PATH
if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
pip install --quiet --upgrade maturin
maturin build --release --compatibility off
"""


def main() -> int:
    # 1. Build the wheel inside Kaggle's runtime image so the .so links against
    #    the exact libc/libstdc++ that the submission worker will load it with.
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{HERE}:/io", "-w", "/io",
         KAGGLE_IMAGE, "bash", "-c", BUILD_SCRIPT],
        check=True,
    )

    # 2. Pick the newest wheel maturin produced. `--compatibility off` emits
    #    `linux_x86_64`; older manylinux wheels left over from prior builds
    #    are also matched so a clean target/ isn't required to upgrade.
    wheels = sorted(
        glob(str(HERE / "target" / "wheels" / "simbot_simplified_native-*-abi3-*linux*.whl")),
        key=os.path.getmtime,
    )
    if not wheels:
        sys.exit("No manylinux wheel was produced under target/wheels/")
    wheel = wheels[-1]
    print(f"Wheel: {os.path.basename(wheel)}")

    # 3. Extract the .so from the wheel (a wheel is just a zip archive).
    with zipfile.ZipFile(wheel) as zf:
        so_name = next(n for n in zf.namelist() if n.endswith(".so"))
        so_bytes = zf.read(so_name)
    so_out = HERE / os.path.basename(so_name)
    so_out.write_bytes(so_bytes)
    print(f"Extracted: {so_out.name}")

    # 4. Bundle main.py + the .so into a tar.gz, both at the archive root.
    bundle = HERE / "submission.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(HERE / "main.py", arcname="main.py")
        tar.add(so_out, arcname=so_out.name)
    print(f"Wrote {bundle}")
    print(
        "Submit with:\n"
        f"  kaggle competitions submit orbit-wars -f {bundle} -m 'simbot-simplified v1'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
