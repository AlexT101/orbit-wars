"""Train an Aphrodite value net from a folder of dated ladder-replay zips,
with optional recency decay.

Each zip in --replays-dir is one calendar day (e.g. `replays_5_14.zip`).
Extracting every day at once is too much to hold in memory, so this driver
processes one day per subprocess (memory is freed when each exits):

  for each zip (oldest -> newest):
      build_from_zip.py        zip            -> raw_<day>.npz
      filter_top10_..(--filter-only)          -> top_<day>.npz   (small)
      delete raw_<day>.npz
  combine_npz.py  top_*.npz (chronological)   -> combined.npz   (`source` = day rank)
  filter_top10_..(--no-filter --recency-halflife H)  -> model.json

Because the per-day files are combined oldest->newest, combine_npz's `source`
column is the day rank, and --recency-halflife turns it into a per-sample
weight (newest day = 1.0, halving every H days).

Example (2p, top-10 per day, 7-day half-life):

    python train/build_ladder.py \
        --replays-dir ladder_replays --players 2 \
        --recency-halflife 7 \
        --model-out train/weights/xgb_2p.json
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

DATE_RE = re.compile(r"(\d{1,2})_(\d{1,2})")


def zip_date(path: Path, year: int) -> date:
    """Parse the month_day out of a `replays_M_DD.zip` stem for sorting."""
    m = DATE_RE.search(path.stem)
    if not m:
        raise SystemExit(f"cannot parse date from zip name: {path.name}")
    return date(year, int(m.group(1)), int(m.group(2)))


def ramp_top_n(day: date, oldest: date, newest: date, n_start: int, n_end: int) -> int:
    """Linearly interpolate top-N by calendar date: oldest day -> n_start,
    newest day -> n_end (rounded). Gaps between zips are respected because the
    interpolation is by actual days, not file index."""
    span = (newest - oldest).days
    if span <= 0:
        return n_end
    frac = (day - oldest).days / span
    return int(round(n_start + frac * (n_end - n_start)))


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replays-dir", type=Path, default=REPO / "ladder_replays")
    p.add_argument("--players", type=int, choices=(2, 4), default=2)
    p.add_argument("--model-out", required=True, type=Path)
    p.add_argument("--recency-halflife", type=float, default=0.0,
                   help="down-weight older days by 0.5 per this many days (0 = uniform)")
    p.add_argument("--gate", choices=("both-topn", "strong-topn", "strong-median", "none"),
                   default="strong-topn",
                   help="per-day quality gate. strong-topn (default): keep each game's side if that "
                        "player is in the day's top-N by win rate; strong-median: bar is the median; "
                        "both-topn: keep games where both players are top-N; none: keep all rows.")
    p.add_argument("--quality-weight", action="store_true", default=True,
                   help="soft-weight kept rows by player strength (default on; only meaningful with a strong-* gate)")
    p.add_argument("--no-quality-weight", dest="quality_weight", action="store_false")
    p.add_argument("--quality-floor", type=float, default=0.25)
    p.add_argument("--top-n-start", type=int, default=10,
                   help="top-N for the OLDEST day (stricter on stale data)")
    p.add_argument("--top-n-end", type=int, default=15,
                   help="top-N for the NEWEST day; linearly ramped by date between the two")
    p.add_argument("--top-n", type=int, default=None,
                   help="force a constant top-N for every day (overrides the start/end ramp)")
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--year", type=int, default=2026, help="year for parsing zip dates (sorting only)")
    p.add_argument("--workdir", type=Path, default=None,
                   help="scratch dir for per-day NPZs (default train/data/<P>p/_ladder_work)")
    p.add_argument("--workers", type=int, default=None, help="passed to build_from_zip.py")
    p.add_argument("--limit", type=int, default=None, help="cap games per zip (debug)")
    p.add_argument("--resume", action="store_true",
                   help="skip days whose per-day NPZ already exists")
    p.add_argument("--keep-temp", action="store_true", help="keep per-day + combined NPZs")
    args = p.parse_args()

    zips = sorted(
        (Path(z) for z in glob.glob(str(args.replays_dir / "*.zip"))),
        key=lambda z: zip_date(z, args.year),
    )
    if not zips:
        raise SystemExit(f"no zips found in {args.replays_dir}")

    workdir = args.workdir or (HERE / "data" / f"{args.players}p" / "_ladder_work")
    workdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    oldest, newest = zip_date(zips[0], args.year), zip_date(zips[-1], args.year)
    if args.top_n is not None:
        n_start = n_end = args.top_n
    else:
        n_start, n_end = args.top_n_start, args.top_n_end
    top_n_desc = f"{n_start}" if n_start == n_end else f"{n_start}->{n_end} (oldest->newest)"
    print(f"ladder: {len(zips)} day(s) {zips[0].stem} .. {zips[-1].stem}  "
          f"players={args.players}  top_n={top_n_desc}  halflife={args.recency_halflife}d")

    filtering = args.gate != "none"
    per_day: list[Path] = []  # in chronological order -> source == day rank
    for z in zips:
        out = workdir / (("gated_" if filtering else "raw_") + z.stem + ".npz")
        if args.resume and out.exists():
            print(f"[resume] {out.name} exists, skipping {z.name}")
            per_day.append(out)
            continue

        n_day = ramp_top_n(zip_date(z, args.year), oldest, newest, n_start, n_end)
        raw = workdir / ("raw_" + z.stem + ".npz")
        build = [py, HERE / "build_from_zip.py", "--players", args.players,
                 "--zip", z, "--out", raw]
        if args.workers:
            build += ["--workers", args.workers]
        if args.limit:
            build += ["--limit", args.limit]
        run(build)

        if filtering:
            print(f"[gate] {z.stem}: {args.gate} top-{n_day}")
            run([py, HERE / "filter_top10_and_train_xgb.py",
                 "--input", raw, "--top10-out", out, "--filter-only",
                 "--gate", args.gate, "--top-n", n_day, "--min-games", args.min_games])
            raw.unlink(missing_ok=True)
        else:
            raw.replace(out)
        per_day.append(out)

    combined = workdir / "combined.npz"
    run([py, HERE / "combine_npz.py", "--out", combined, *per_day])

    train = [py, HERE / "filter_top10_and_train_xgb.py",
             "--data", combined, "--no-filter", "--model-out", args.model_out]
    if args.recency_halflife > 0:
        train += ["--recency-halflife", args.recency_halflife]
    if args.quality_weight:
        train += ["--quality-weight", "--quality-floor", args.quality_floor]
    run(train)

    if not args.keep_temp:
        for f in per_day:
            f.unlink(missing_ok=True)
        combined.unlink(missing_ok=True)
        print(f"\ncleaned temp NPZs in {workdir} (pass --keep-temp to retain)")

    print(f"\nDone. model -> {args.model_out}")


if __name__ == "__main__":
    main()
