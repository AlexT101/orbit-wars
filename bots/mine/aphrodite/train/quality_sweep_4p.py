"""Train the 4p top20/floor050/drop-decided value net from existing data.

This mirrors the selected 2p quality-sweep recipe without re-extracting
features: keep rows whose perspective player is in the daily top 20, combine
the existing SummaryV3 daily NPZs, then train with rating quality weighting,
quality floor 0.50, and player-count-correct decided-row dropping.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PY = REPO / "venv" / "Scripts" / "python.exe"
BASE = HERE / "data" / "4p" / "_ladder_work"
WORK = HERE / "data" / "4p" / "_quality_sweep"
LOGS = WORK / "logs"
WEIGHTS = HERE / "weights"

VARIANT = "top20_floor050_dropdec"
MODEL = WEIGHTS / f"xgb_4p_qsweep_{VARIANT}.json"
COMBINED = WORK / "top20" / "combined.npz"


def run_logged(cmd: list[str | Path], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    env = dict(os.environ, PYTHONUTF8="1")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, [str(c) for c in cmd])


def daily_sources() -> list[tuple[str, Path, Path]]:
    out = []
    for npz in sorted(BASE.glob("gated_replays_6_*.npz")):
        m = re.search(r"(\d+_\d+)", npz.stem)
        if not m:
            continue
        day = m.group(1)
        if day < "6_08" or day > "6_14":
            continue
        top_json = BASE / f"topn_replays_{day}.json"
        if not top_json.is_file():
            raise FileNotFoundError(top_json)
        out.append((day, npz, top_json))
    if len(out) != 7:
        raise SystemExit(f"expected 7 daily 4p NPZs for 6_08..6_14, found {len(out)}")
    return out


def filter_daily(src: Path, top_json: Path, out: Path, force: bool) -> None:
    if out.is_file() and not force:
        print(f"[skip] {out.relative_to(REPO)} exists")
        return
    keep = set(json.loads(top_json.read_text(encoding="utf-8"))[:20])
    d = np.load(src, allow_pickle=False)
    if "summary_v3" not in d.files:
        raise SystemExit(f"{src} is not SummaryV3")
    meta = d["meta"].astype(np.int32)
    game_names = d["game_names"]
    row_names = np.array(
        [str(game_names[int(gid), int(slot)]) for gid, _step, slot, _players in meta],
        dtype=object,
    )
    mask = np.fromiter((name in keep for name in row_names), dtype=bool, count=len(row_names))
    if not bool(mask.any()):
        raise SystemExit(f"{src} produced no top20 rows")

    extra = {}
    for key in ("game_names", "game_rewards", "game_files"):
        if key in d.files:
            extra[key] = d[key]
    for key in ("decisiveness_aux", "win_rate", "is_strong"):
        if key in d.files:
            extra[key] = d[key][mask]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        summary_v3=d["summary_v3"][mask].astype(np.float32),
        labels=d["labels"][mask].astype(np.float32),
        meta=meta[mask],
        **extra,
    )
    print(f"wrote {out.relative_to(REPO)} rows={int(mask.sum()):,}/{len(mask):,}")


def prepare(force: bool) -> None:
    daily = []
    for day, src, top_json in daily_sources():
        out = WORK / "top20" / f"gated_replays_{day}_top20.npz"
        filter_daily(src, top_json, out, force)
        daily.append(out)
    if COMBINED.is_file() and not force:
        print(f"[skip] {COMBINED.relative_to(REPO)} exists")
        return
    run_logged([PY, HERE / "combine_npz.py", "--out", COMBINED, *daily], LOGS / "combine_top20.log")


def train(force: bool) -> None:
    if MODEL.is_file() and not force:
        print(f"[skip] {MODEL.relative_to(REPO)} exists")
        return
    run_logged(
        [
            PY,
            HERE / "train_xgb.py",
            "--data",
            COMBINED,
            "--no-filter",
            "--recency-halflife",
            "7",
            "--rounds",
            "2000",
            "--early-stopping",
            "50",
            "--model-out",
            MODEL,
            "--quality-weight",
            "--quality-metric",
            "rating",
            "--quality-shape",
            "decay",
            "--quality-floor",
            "0.5",
            "--drop-decided",
        ],
        LOGS / f"train_{VARIANT}.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "all"), nargs="?", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        prepare(args.force)
    if args.stage in ("train", "all"):
        train(args.force)


if __name__ == "__main__":
    main()
