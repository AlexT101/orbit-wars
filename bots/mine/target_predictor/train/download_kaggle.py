"""Download Orbit Wars episode datasets from Kaggle.

Reads manifest.csv (date,slug,...), calls kagglehub.dataset_download for each
day, and stages the result as `<out>/<slug>.zip` (or `<slug>/` directory if the
upstream dataset is unzipped). Idempotent: if the destination already exists it
is skipped.

By default, mirrors the prometheus layout and writes to /tmp/orbit_days/. The
build_dataset.py step then iterates those zips and filters to 2p games.

Usage:
  python3 download_kaggle.py --start-date 2026-05-27 --end-date 2026-05-30
  python3 download_kaggle.py --limit-days 2 --out /tmp/orbit_days
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUT = Path("/tmp/orbit_days")
MANIFEST = Path(__file__).resolve().parent / "manifest.csv"


@dataclass(frozen=True)
class Row:
    date: str
    slug: str

    @property
    def kagglehub_ref(self) -> str:
        return self.slug if "/" in self.slug else f"kaggle/{self.slug}"


def read_manifest(start: str | None, end: str | None, limit: int | None) -> list[Row]:
    rows: list[Row] = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            date = r.get("date", "")
            if start and date < start:
                continue
            if end and date > end:
                continue
            slug = (r.get("daily_dataset_slug") or "").strip()
            if not slug:
                continue
            rows.append(Row(date=date, slug=slug))
    rows.sort(key=lambda x: x.date)
    return rows[:limit] if limit else rows


def import_kagglehub():
    # Same shim as prometheus/pipeline.py: kagglehub 1.0.1 imports
    # get_web_endpoint, newer kagglesdk wheels expose get_endpoint.
    try:
        import kagglesdk.kaggle_env as kaggle_env

        if not hasattr(kaggle_env, "get_web_endpoint") and hasattr(kaggle_env, "get_endpoint"):
            kaggle_env.get_web_endpoint = kaggle_env.get_endpoint
    except ImportError:
        pass
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub not importable. Try:\n"
            f"  {sys.executable} -m pip install --upgrade kagglehub kagglesdk"
        ) from exc
    return kagglehub


def stage(src: Path, dest_zip: Path, dest_dir: Path) -> Path:
    """Stage a kagglehub download into a stable location under `out`.

    Order of preference, to keep disk usage bounded:
      1. If a single .zip > 1 MB exists in src → copy as <slug>.zip (cheap).
      2. Else if src has .json files (kagglehub already extracted) → re-zip
         them as <slug>.zip with ZIP_DEFLATED so we don't blow up disk by
         15× (1.2 GB zip vs 21 GB of raw json).
      3. Else (other formats) → copy directory tree to <slug>/.
    """
    zips = sorted(src.rglob("*.zip"))
    if len(zips) == 1 and zips[0].stat().st_size > 1024 * 1024:
        shutil.copy2(zips[0], dest_zip)
        return dest_zip
    json_files = sorted(p for p in src.rglob("*.json") if p.is_file())
    if json_files:
        import zipfile
        # compresslevel=9 makes ~1.4 GB per day instead of ~2.5 GB at level 1.
        # Worth the few extra seconds per day to fit ~45 days in 100 GB disk.
        with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in json_files:
                zf.write(p, arcname=p.name)
        return dest_zip
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        out = dest_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
    return dest_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date")
    ap.add_argument("--end-date")
    ap.add_argument("--limit-days", type=int)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = read_manifest(args.start_date, args.end_date, args.limit_days)
    if not rows:
        print("no manifest rows matched the date range", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    kagglehub = import_kagglehub()

    for row in rows:
        dest_zip = args.out / f"{row.slug}.zip"
        dest_dir = args.out / row.slug
        if dest_zip.exists() or dest_dir.exists():
            print(f"{row.date}: already staged ({dest_zip if dest_zip.exists() else dest_dir})")
            continue
        print(f"{row.date}: downloading {row.kagglehub_ref}", flush=True)
        src = Path(kagglehub.dataset_download(row.kagglehub_ref))
        staged = stage(src, dest_zip, dest_dir)
        size_mb = staged.stat().st_size // (1024 * 1024) if staged.is_file() else 0
        print(f"{row.date}: staged {staged}  ({size_mb} MB)", flush=True)
        # Aggressively reclaim kagglehub cache (~21 GB per extracted day) so
        # we don't fill the disk while iterating many days.
        import shutil as _sh
        cache = Path.home() / ".cache" / "kagglehub"
        if cache.exists():
            for child in cache.iterdir():
                _sh.rmtree(child, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
