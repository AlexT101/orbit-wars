"""
Build a Kaggle-ready submission bundle for simbot.

Kaggle runs agents on Linux x86_64 and will not compile Rust, so we cross-build
the native module in the official manylinux container (via Docker), extract the
resulting Linux `.so`, and bundle it next to `main.py` in a tar.gz.

Usage:
    python bots/simbot/build_submission.py

Requires Docker. Output: bots/simbot/submission.tar.gz
"""

import os
import subprocess
import sys
import tarfile
import zipfile
from glob import glob
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    # 1. Cross-build a manylinux abi3 wheel in the official maturin container.
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{HERE}:/io", "-w", "/io",
         "ghcr.io/pyo3/maturin", "build", "--release"],
        check=True,
    )

    # 2. Pick the newest manylinux wheel maturin produced.
    wheels = sorted(
        glob(str(HERE / "target" / "wheels" / "simbot_native-*-abi3-manylinux*.whl")),
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
        f"  kaggle competitions submit orbit-wars -f {bundle} -m 'simbot v1'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
