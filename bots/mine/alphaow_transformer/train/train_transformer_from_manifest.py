"""Download daily Kaggle episode datasets, extract transformer tokens, train evaluator.

The manifest CSV is expected to contain at least:
  date,daily_dataset_slug,daily_dataset_url

Datasets are cached under `--dataset-cache-dir` by slug. A day is only
downloaded when that slug is not already present in the local cache.
Per-day token NPZs are also cached, so interrupted runs resume at the
next missing step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ALPHAOW_DIR = REPO / "bots" / "mine" / "alphaow"
DEFAULT_MANIFEST = HERE / "manifest.csv"


class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"


@dataclass(frozen=True)
class ManifestRow:
    day: str
    slug: str
    url: str


def log(msg: str):
    print(msg, flush=True)


def tag(name: str, color: str = C.CYAN) -> str:
    return f"{color}{name:>9s}{C.RESET}"


def short_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.1f} {unit}" if unit != "B" else f"{int(x)} B"
        x /= 1024.0


def parse_day(s: str) -> date:
    return date.fromisoformat(s)


def read_manifest(path: Path, start: str | None, end: str | None, limit_days: int | None) -> list[ManifestRow]:
    start_d = parse_day(start) if start else None
    end_d = parse_day(end) if end else None
    rows: list[ManifestRow] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            day = r.get("date", "").strip()
            slug = r.get("daily_dataset_slug", "").strip()
            if not day or not slug:
                continue
            d = parse_day(day)
            if start_d and d < start_d:
                continue
            if end_d and d > end_d:
                continue
            rows.append(ManifestRow(day=day, slug=slug, url=r.get("daily_dataset_url", "").strip()))
    rows.sort(key=lambda r: r.day)
    if limit_days:
        rows = rows[:limit_days]
    return rows


def cached_dataset_path(cache_dir: Path, slug: str) -> Path | None:
    local = cache_dir / slug
    if local.is_dir() and any(local.iterdir()):
        marker = local / "_kagglehub_path.txt"
        if marker.exists():
            p = Path(marker.read_text().strip())
            if p.exists():
                return p
        return local
    if local.is_symlink() and local.exists():
        return local.resolve()
    return None


def remember_dataset_path(cache_dir: Path, slug: str, path: Path) -> Path:
    local = cache_dir / slug
    local.mkdir(parents=True, exist_ok=True)
    (local / "_kagglehub_path.txt").write_text(str(path.resolve()) + "\n")
    return path


def patch_kagglehub_sdk_compat():
    """Work around kagglehub/kagglesdk releases that are slightly out of sync."""
    try:
        import kagglesdk.kaggle_env as kaggle_env  # type: ignore
    except Exception:
        return
    if hasattr(kaggle_env, "get_web_endpoint"):
        return

    def get_web_endpoint(env):
        endpoint = kaggle_env.get_endpoint(env)
        return endpoint.replace("https://api.kaggle.com", "https://www.kaggle.com")

    kaggle_env.get_web_endpoint = get_web_endpoint


def download_dataset(row: ManifestRow, cache_dir: Path) -> Path:
    existing = cached_dataset_path(cache_dir, row.slug)
    if existing is not None:
        log(f"{tag('dataset', C.DIM)} {row.day} cached {C.DIM}{short_path(existing)}{C.RESET}")
        return existing

    patch_kagglehub_sdk_compat()
    try:
        import kagglehub  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Could not import kagglehub cleanly. Try reinstalling the paired Kaggle packages:\n"
            "  python3 -m pip install --force-reinstall kagglehub kagglesdk\n"
            f"Original import error: {e!r}"
        ) from e

    handle = f"kaggle/{row.slug}"
    log(f"{tag('download')} {row.day} {handle}")
    path = Path(kagglehub.dataset_download(handle))
    return remember_dataset_path(cache_dir, row.slug, path)


def ensure_extracted(dataset_path: Path, extracted_root: Path, slug: str) -> Path:
    zip_files = sorted(dataset_path.rglob("*.zip")) if dataset_path.is_dir() else []
    if not zip_files:
        return dataset_path

    out = extracted_root / slug
    done = out / ".complete"
    if done.exists():
        log(f"{tag('extract', C.DIM)} cached {slug} -> {C.DIM}{short_path(out)}{C.RESET}")
        return out

    out.mkdir(parents=True, exist_ok=True)
    log(f"{tag('extract')} {slug}: {len(zip_files)} zip file(s)")
    for zpath in zip_files:
        sub = out / zpath.stem
        sub.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(sub)
    done.write_text("ok\n")
    return out


def replay_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if root.is_file():
        return [root] if root.suffix in {".json", ".gz"} else []
    files.extend(root.rglob("*.json"))
    files.extend(root.rglob("*.json.gz"))
    return sorted(files)


def write_file_manifest(files: list[Path], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"files": [str(p) for p in files]}, indent=2) + "\n")


def run(cmd: list[str], *, cwd: Path = REPO, label: str = "run"):
    display = " ".join(shlex.quote(str(c)) for c in cmd)
    if len(display) > 220:
        display = display[:217] + "..."
    log(f"{tag(label, C.MAGENTA)} {display}")
    subprocess.check_call(cmd, cwd=str(cwd))


def build_extractors(skip_build: bool):
    if skip_build:
        return
    # The local target dir has occasionally contained future-dated stale
    # extractor artifacts, which makes Cargo incorrectly skip rebuilding after
    # the binary record layout changes. Remove only the extractor/lib products
    # before building so the Python reader and Rust writer stay in lockstep.
    for pattern in (
        "target/release/extract_tokens",
        "target/release/deps/extract_tokens-*",
        "target/release/libalphaow_bot*",
        "target/release/deps/libalphaow_bot*",
        "target/release/deps/alphaow_bot-*",
    ):
        for path in ALPHAOW_DIR.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    run(["cargo", "build", "--release", "--quiet", "--bin", "extract_tokens", "--bin", "alphaow-bot"], cwd=ALPHAOW_DIR, label="build")


def extract_day(row: ManifestRow, dataset_path: Path, args) -> Path | None:
    token_out = Path(args.token_dir) / f"{row.day}_{row.slug}_tokens.npz"
    if token_out.exists() and token_out.stat().st_size > 0 and not args.force_extract:
        log(
            f"{tag('tokens', C.DIM)} {row.day} cached "
            f"{human_bytes(token_out.stat().st_size)} {C.DIM}{short_path(token_out)}{C.RESET}"
        )
        return token_out

    source = ensure_extracted(dataset_path, Path(args.extracted_dir), row.slug)
    files = replay_files(source)
    if args.limit_replays_per_day:
        files = files[: args.limit_replays_per_day]
    if not files:
        log(f"{tag('tokens', C.RED)} {row.day}: no replay JSON files found under {source}")
        return None

    manifest = Path(args.manifest_dir) / f"{row.day}_{row.slug}_files.json"
    write_file_manifest(files, manifest)
    log(f"{tag('frames')} {row.day}: extracting {len(files):,} replay(s)")
    run(
        [
            sys.executable,
            str(HERE / "from_replays_tokens.py"),
            "--manifest",
            str(manifest),
            "--out",
            str(token_out),
            "--workers",
            str(args.workers),
            "--target-mode",
            args.target_mode if args.target_mode != "stored" else "outcome",
            "--time-coef",
            str(args.time_coef),
            "--episode-steps",
            str(args.episode_steps),
        ],
        label="extract",
    )
    return token_out


def train(token_files: list[Path], args):
    if args.skip_train:
        return
    if not token_files:
        raise SystemExit("no token NPZs available for training")
    cmd = [
        sys.executable,
        str(HERE / "train_transformer.py"),
        "--data",
        *[str(p) for p in token_files],
        "--out",
        str(args.out),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--ff-dim",
        str(args.ff_dim),
        "--summary-hidden",
        str(args.summary_hidden),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--wd",
        str(args.wd),
        "--seed",
        str(args.seed),
        "--target-mode",
        args.target_mode,
        "--time-coef",
        str(args.time_coef),
        "--episode-steps",
        str(args.episode_steps),
    ]
    if args.metrics_path:
        cmd += ["--metrics-path", str(args.metrics_path)]
    if args.dashboard_path:
        cmd += ["--dashboard-path", str(args.dashboard_path)]
    if args.no_dashboard:
        cmd += ["--no-dashboard"]
    if args.state_path:
        cmd += ["--state-path", str(args.state_path)]
    if args.resume_state:
        cmd += ["--resume-state"]
    if args.baseline_weights is not None:
        cmd += ["--baseline-weights", *[str(x) for x in args.baseline_weights]]
    if args.max_samples:
        cmd += ["--max-samples", str(args.max_samples)]
    run(cmd, label="train")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST))
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--limit-days", type=int, default=None)
    p.add_argument("--dataset-cache-dir", default=str(HERE / "datasets"))
    p.add_argument("--extracted-dir", default=str(HERE / "datasets_extracted"))
    p.add_argument("--manifest-dir", default=str(HERE / "manifests"))
    p.add_argument("--token-dir", default=str(HERE / "data" / "tokens_by_day"))
    p.add_argument("--out", default=str(HERE / "weights" / "transformer_manifest.bin"))
    p.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    p.add_argument("--limit-replays-per-day", type=int, default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=128)
    p.add_argument("--summary-hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--target-mode", choices=["time", "outcome", "stored"], default="time")
    p.add_argument("--time-coef", type=float, default=0.10)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--metrics-path", default=None)
    p.add_argument("--dashboard-path", default=None)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--state-path", default=None)
    p.add_argument("--resume-state", action="store_true")
    p.add_argument(
        "--baseline-weights",
        nargs="*",
        default=None,
        help="AOWV MLP baselines to pass through to train_transformer.py; omit for default comparable old baselines",
    )
    args = p.parse_args()

    rows = read_manifest(Path(args.manifest_csv), args.start_date, args.end_date, args.limit_days)
    if not rows:
        raise SystemExit("manifest selection is empty")
    log(f"{tag('manifest', C.BOLD)} {len(rows)} day(s): {rows[0].day} .. {rows[-1].day}")

    build_extractors(args.skip_build)

    token_files: list[Path] = []
    for row in rows:
        token_out = Path(args.token_dir) / f"{row.day}_{row.slug}_tokens.npz"
        if args.skip_extract and token_out.exists():
            token_files.append(token_out)
            continue
        if args.skip_download:
            dataset_path = cached_dataset_path(Path(args.dataset_cache_dir), row.slug)
            if dataset_path is None:
                log(f"{tag('dataset', C.RED)} {row.day}: missing cache for {row.slug}; skipping")
                continue
        else:
            dataset_path = download_dataset(row, Path(args.dataset_cache_dir))
        if args.skip_extract:
            continue
        extracted = extract_day(row, dataset_path, args)
        if extracted is not None:
            token_files.append(extracted)

    # If extraction was skipped, train on every selected token file that already exists.
    if args.skip_extract:
        token_files = [Path(args.token_dir) / f"{r.day}_{r.slug}_tokens.npz" for r in rows]
        token_files = [p for p in token_files if p.exists() and p.stat().st_size > 0]

    total_bytes = sum(p.stat().st_size for p in token_files if p.exists())
    log(f"{tag('tokens', C.BOLD)} training files={len(token_files)} size={human_bytes(total_bytes)}")
    train(token_files, args)
    log(f"{tag('done', C.GREEN)}")


if __name__ == "__main__":
    main()
