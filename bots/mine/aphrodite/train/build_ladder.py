"""Train an Aphrodite value net from a folder of dated ladder-replay zips,
with optional recency decay.

Each zip in --replays-dir is one calendar day (e.g. `replays_5_14.zip`).
Extracting every day at once is too much to hold in memory, so this driver
processes one day per subprocess (memory is freed when each exits):

  for each zip (oldest -> newest, within --from-day/--to-day):
      elo_topn.py      zip (outcomes only)     -> topn_<day>.json  (top-N by Elo, THIS day)
      build_from_zip.py --keep-players topn     -> gated_<day>.npz  (only those players)
  combine_npz.py  gated_*.npz (chronological)   -> combined.npz   (`source` = day rank)
  train_xgb.py (--no-filter, Elo-weighted)      -> model.json

The default gate is **elo-topn**: per day, keep only the top --top-n players by
that day's Bradley-Terry Elo, then extract just their rows (so the costly Rust
extraction never touches discarded rows). `--gate none` keeps every player and
relies on the train-time Elo weight instead. (There is no win-rate gate.)

Because the per-day files are combined oldest->newest, combine_npz's `source`
column is the day rank, and --recency-halflife turns it into a per-sample
weight (newest day = 1.0, halving every H days).

Example (2p, top-20 by Elo per day, last week, 7-day half-life):

    python train/build_ladder.py \
        --replays-dir ladder_replays --players 2 \
        --gate elo-topn --top-n 20 --from-day 6_07 --to-day 6_13 \
        --recency-halflife 7 --keep-temp \
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

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

DATE_RE = re.compile(r"(\d{1,2})_(\d{1,2})")


def zip_date(path: Path, year: int) -> date:
    """Parse the month_day out of a `replays_M_DD.zip` stem for sorting."""
    m = DATE_RE.search(path.stem)
    if not m:
        raise SystemExit(f"cannot parse date from zip name: {path.name}")
    return date(year, int(m.group(1)), int(m.group(2)))


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    env = dict(os.environ, PYTHONUTF8="1")  # UTF-8 stdout in every child (emoji player names)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replays-dir", type=Path, default=REPO / "ladder_replays")
    p.add_argument("--players", type=int, choices=(2, 4), default=2)
    p.add_argument("--model-out", required=True, type=Path)
    p.add_argument("--recency-halflife", type=float, default=0.0,
                   help="down-weight older days by 0.5 per this many days (0 = uniform)")
    p.add_argument("--gate", choices=("elo-topn", "none"), default="elo-topn",
                   help="per-day extraction gate. elo-topn (default): per day, keep only the "
                        "top --top-n players by Bradley-Terry Elo (computed on THAT day alone via "
                        "elo_topn.py), then extract just their rows (build_from_zip --keep-players). "
                        "none: keep ALL players' rows and rely on the train-time Elo weight instead.")
    p.add_argument("--top-n", type=int, default=20,
                   help="elo-topn gate: top-N players by Elo to keep PER DAY (default 20).")
    p.add_argument("--prior-strength", type=float, default=3.0,
                   help="elo_topn Bradley-Terry Gamma-prior strength (shrinks few-game players).")
    p.add_argument("--features", choices=("v2", "v3", "auto"), default="auto",
                   help="feature extractor. auto (default): v2 for 2p, v3 for 4p. "
                        "(v3 = 145-d summary_v3 + decisiveness aux; v2 = 65-d summary_v2.)")
    p.add_argument("--from-day", default=None, metavar="M_DD",
                   help="only process zips on/after this day (e.g. 6_07). Default: oldest available.")
    p.add_argument("--to-day", default=None, metavar="M_DD",
                   help="only process zips on/before this day (e.g. 6_13). Default: newest available.")
    p.add_argument("--quality-weight", action="store_true", default=True,
                   help="soft-weight rows by player Elo at train time (default on); uses a "
                        "Bradley-Terry rating fit (--quality-metric rating).")
    p.add_argument("--no-quality-weight", dest="quality_weight", action="store_false")
    p.add_argument("--quality-floor", type=float, default=0.05,
                   help="weakest kept player's Elo weight (strongest = 1.0); exponential-decay shape.")
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--rounds", type=int, default=2000, help="max XGBoost boosting rounds (early stopping picks the real count)")
    p.add_argument("--early-stopping", type=int, default=50,
                   help="stop if val logloss hasn't improved in this many rounds")
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

    # Optional inclusive date-range filter (e.g. --from-day 6_07 --to-day 6_13).
    def parse_day(s: str) -> date:
        m = DATE_RE.search(s)
        if not m:
            raise SystemExit(f"cannot parse --from/--to-day '{s}' (expected M_DD)")
        return date(args.year, int(m.group(1)), int(m.group(2)))

    if args.from_day:
        lo = parse_day(args.from_day)
        zips = [z for z in zips if zip_date(z, args.year) >= lo]
    if args.to_day:
        hi = parse_day(args.to_day)
        zips = [z for z in zips if zip_date(z, args.year) <= hi]
    if not zips:
        raise SystemExit("no zips left after the --from-day/--to-day filter")

    workdir = args.workdir or (HERE / "data" / f"{args.players}p" / "_ladder_work")
    workdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    gate_desc = f"elo-topn top-{args.top_n}/day" if args.gate == "elo-topn" else "none (all players)"
    print(f"ladder: {len(zips)} day(s) {zips[0].stem} .. {zips[-1].stem}  "
          f"players={args.players}  gate={gate_desc}  halflife={args.recency_halflife}d")

    elo_gate = args.gate == "elo-topn"
    feats = args.features if args.features != "auto" else ("v3" if args.players == 4 else "v2")
    print(f"  features={feats}  (relational decay always on)")
    per_day: list[Path] = []  # in chronological order -> source == day rank
    for z in zips:
        out = workdir / (("gated_" if elo_gate else "raw_") + z.stem + ".npz")
        if args.resume and out.exists():
            print(f"[resume] {out.name} exists, skipping {z.name}")
            per_day.append(out)
            continue

        if elo_gate:
            # Phase 1: cheap outcome-only scan of THIS day's zip -> top-N by Elo.
            topn_json = workdir / f"topn_{z.stem}.json"
            elo_cmd = [py, HERE / "elo_topn.py", "--zip", z, "--players", args.players,
                       "--top-n", args.top_n, "--min-games", args.min_games,
                       "--prior-strength", args.prior_strength, "--out", topn_json]
            if args.workers:
                elo_cmd += ["--workers", args.workers]
            run(elo_cmd)
            # Phase 2: extract features for those players only.
            build = [py, HERE / "build_from_zip.py", "--players", args.players,
                     "--zip", z, "--out", out, "--keep-players", topn_json]
        else:
            build = [py, HERE / "build_from_zip.py", "--players", args.players,
                     "--zip", z, "--out", out]
        build += ["--features", feats]
        if args.workers:
            build += ["--workers", args.workers]
        if args.limit:
            build += ["--limit", args.limit]
        run(build)
        per_day.append(out)

    combined = workdir / "combined.npz"
    run([py, HERE / "combine_npz.py", "--out", combined, *per_day])

    train = [py, HERE / "train_xgb.py",
             "--data", combined, "--no-filter", "--model-out", args.model_out,
             "--rounds", args.rounds, "--early-stopping", args.early_stopping]
    if args.recency_halflife > 0:
        train += ["--recency-halflife", args.recency_halflife]
    if args.quality_weight:
        # Neither flow records a win_rate column anymore, so weight by a
        # Bradley-Terry rating fit (over the kept rows' games) in all cases.
        train += ["--quality-weight", "--quality-floor", args.quality_floor,
                  "--quality-metric", "rating"]
    run(train)

    if not args.keep_temp:
        for f in per_day:
            f.unlink(missing_ok=True)
        combined.unlink(missing_ok=True)
        print(f"\ncleaned temp NPZs in {workdir} (pass --keep-temp to retain)")

    print(f"\nDone. model -> {args.model_out}")


if __name__ == "__main__":
    main()
