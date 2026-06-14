from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TRAIN_DIR = REPO_ROOT / "experimental_arch" / "train_transformer"
IL_DIR = REPO_ROOT / "experimental_arch" / "imitation_learning"
ENV_MODEL_DIR = REPO_ROOT / "experimental_arch" / "env_model"
APHRODITE_DIR = REPO_ROOT / "bots" / "mine" / "aphrodite"
FRANKENSTEIN_DIR = REPO_ROOT / "bots" / "mine" / "frankenstein1"

LATEST_CHECKPOINT = IL_DIR / "checkpoints" / "osteo_bc_transformer" / "latest.pt"
ZIP_OUT = HERE / "submission.zip"
TAR_OUT = HERE / "submission.tar.gz"

FILES = [
    "main.py",
    "aphrodite",
    "xgb_2p_old_top10.json",
    "xgb_2p.json",
    "xgb_4p.json",
    "osteo_il_latest.pt",
    "il_support/constants.py",
    "il_support/features.py",
    "il_support/model.py",
    "orbit_wars_model/__init__.py",
    "orbit_wars_model/orbit_wars_model.abi3.so",
]

FLAT_COPIES = [
    ("il_support/constants.py", "constants.py"),
    ("il_support/features.py", "features.py"),
    ("il_support/model.py", "model.py"),
    ("orbit_wars_model/orbit_wars_model.abi3.so", "orbit_wars_model.abi3.so"),
]


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def refresh_sources() -> None:
    _copy(LATEST_CHECKPOINT, HERE / "osteo_il_latest.pt")
    for name in ("constants.py", "features.py", "model.py"):
        _copy(TRAIN_DIR / name, HERE / "il_support" / name)

    cargo = shutil.which("cargo")
    if not cargo:
        raise SystemExit("cargo not found; cannot build orbit_wars_model")
    subprocess.run([cargo, "build", "--release"], cwd=ENV_MODEL_DIR, check=True)
    _copy(ENV_MODEL_DIR / "target" / "release" / "liborbit_wars_model.so", HERE / "orbit_wars_model" / "orbit_wars_model.abi3.so")

    aphrodite_bin = FRANKENSTEIN_DIR / "aphrodite"
    if not aphrodite_bin.is_file():
        aphrodite_bin = APHRODITE_DIR / "target-docker" / "release" / "aphrodite"
    _copy(aphrodite_bin, HERE / "aphrodite")
    os.chmod(HERE / "aphrodite", os.stat(HERE / "aphrodite").st_mode | 0o755)

    weights_dir = APHRODITE_DIR / "train" / "weights"
    _copy(weights_dir / "xgb_2p_old_top10.json", HERE / "xgb_2p_old_top10.json")
    _copy(weights_dir / "xgb_2p.json", HERE / "xgb_2p.json")
    _copy(weights_dir / "xgb_4p.json", HERE / "xgb_4p.json")


def _check() -> None:
    missing = [rel for rel in FILES if not (HERE / rel).is_file()]
    if missing:
        raise SystemExit("missing submission files:\n  " + "\n  ".join(missing))


def _stage(td: Path) -> list[str]:
    staged: list[str] = []
    for rel in FILES:
        dst = td / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / rel, dst)
        staged.append(rel)
    for src_rel, dst_rel in FLAT_COPIES:
        shutil.copy2(HERE / src_rel, td / dst_rel)
        staged.append(dst_rel)
    os.chmod(td / "aphrodite", os.stat(td / "aphrodite").st_mode | 0o755)
    return staged


def _write_zip(staged_dir: Path, staged: list[str]) -> None:
    ZIP_OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in staged:
            zf.write(staged_dir / rel, arcname=rel)


def _write_tar(staged_dir: Path, staged: list[str]) -> None:
    TAR_OUT.unlink(missing_ok=True)
    with tarfile.open(TAR_OUT, "w:gz") as tf:
        for rel in staged:
            tf.add(staged_dir / rel, arcname=rel)


def _print_contents() -> None:
    print(f"Wrote {ZIP_OUT} ({ZIP_OUT.stat().st_size:,} bytes)")
    print(f"Wrote {TAR_OUT} ({TAR_OUT.stat().st_size:,} bytes)")
    print()
    print("Zip contents:")
    with zipfile.ZipFile(ZIP_OUT) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:<42} {info.file_size:>10,} bytes")
    print()
    print("Submit:")
    print(f"  kaggle competitions submit orbit-wars -f {ZIP_OUT} -m 'triplepoint1'")


def main() -> int:
    refresh_sources()
    _check()
    with tempfile.TemporaryDirectory() as tmp:
        staged_dir = Path(tmp)
        staged = _stage(staged_dir)
        _write_zip(staged_dir, staged)
        _write_tar(staged_dir, staged)
        _print_contents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
