"""Build day-specific top-N player filters from ladder replay zips.

For each replay day, this scans outcome metadata only, fits a Bradley-Terry
rating over a small trailing window, and writes a JSON mapping:

    {"6_10": ["player A", "player B", ...], ...}

The extractor can then imitate each day's strong players on that same day while
keeping older weak versions of currently strong bots out of the dataset.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

IL_DIR = Path(__file__).resolve().parent
REPO_ROOT = IL_DIR.parents[1]
APHRODITE_TRAIN_DIR = REPO_ROOT / "bots" / "mine" / "aphrodite" / "train"
if str(APHRODITE_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(APHRODITE_TRAIN_DIR))

from train_xgb import bradley_terry_ratings  # noqa: E402

REPLAY_HEAD_BYTES = 262_144
_NAME_RE = re.compile(rb'"Name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_REWARDS_RE = re.compile(rb'"rewards"\s*:\s*\[([^\]]*)\]')
_STEPS_RE = re.compile(rb'"steps"')


def day_from_zip(path: Path) -> str:
    match = re.search(r"replays_(\d+_\d+)", path.stem)
    return match.group(1) if match else path.stem


def day_ordinal(day: str) -> int:
    month_s, day_s = day.split("_", 1)
    return date(2026, int(month_s), int(day_s)).toordinal()


def replay_meta_from_head(head: bytes, players: int) -> tuple[tuple[str, ...], tuple[float, ...]] | None:
    steps_match = _STEPS_RE.search(head)
    if steps_match is not None:
        head = head[: steps_match.start()]
    names: list[str] = []
    for raw in _NAME_RE.findall(head):
        try:
            names.append(str(json.loads(b'"' + raw + b'"')))
        except Exception:
            names.append(raw.decode("utf-8", "replace"))
    rewards_match = _REWARDS_RE.search(head)
    if len(names) != players or rewards_match is None:
        return None
    try:
        rewards = tuple(
            float(x.strip())
            for x in rewards_match.group(1).decode("ascii", "replace").split(",")
            if x.strip()
        )
    except ValueError:
        return None
    if len(rewards) != players:
        return None
    return tuple(names[:players]), rewards


def _scan_chunk(args: tuple[str, list[str], int]) -> list[tuple[tuple[str, ...], tuple[float, ...]]]:
    zip_path, entries, players = args
    out: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for entry in entries:
            try:
                with zf.open(entry) as f:
                    meta = replay_meta_from_head(f.read(REPLAY_HEAD_BYTES), players)
            except Exception:
                continue
            if meta is not None:
                out.append(meta)
    return out


def iter_zip_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        out.extend(matches if matches else [Path(pattern)])
    paths = sorted({p.resolve() for p in out}, key=lambda p: p.name)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("zip path(s) not found:\n  " + "\n  ".join(missing))
    return paths


def scan_by_day(zip_paths: list[Path], players: int, workers: int) -> dict[str, list[tuple[tuple[str, ...], tuple[float, ...]]]]:
    out: dict[str, list[tuple[tuple[str, ...], tuple[float, ...]]]] = defaultdict(list)
    for zi, zip_path in enumerate(zip_paths, start=1):
        day = day_from_zip(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            entries = sorted(n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/"))
        chunks = [entries[i::workers] for i in range(workers)]
        print(f">>> [{zi}/{len(zip_paths)}] {zip_path.name}: entries={len(entries):,} day={day}", flush=True)
        with mp.Pool(workers) as pool:
            for chunk in pool.map(_scan_chunk, [(str(zip_path), c, players) for c in chunks]):
                out[day].extend(chunk)
        print(f"    kept outcome rows={len(out[day]):,}", flush=True)
    return dict(out)


def fit_window_topn(
    games_by_day: dict[str, list[tuple[tuple[str, ...], tuple[float, ...]]]],
    *,
    top_n: int,
    min_games: int,
    prior_strength: float,
    window_days: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    days = sorted(games_by_day, key=day_ordinal)
    ords = {day: day_ordinal(day) for day in days}
    top_by_day: dict[str, list[str]] = {}
    diagnostics: dict[str, Any] = {}
    for day in days:
        lo = ords[day] - max(0, window_days - 1)
        window_games = [
            game
            for other_day in days
            if lo <= ords[other_day] <= ords[day]
            for game in games_by_day[other_day]
        ]
        if not window_games:
            top_by_day[day] = []
            diagnostics[day] = {"window_games": 0, "rated_players": 0, "top": []}
            continue
        names = np.array([game[0] for game in window_games], dtype="<U128")
        rewards = np.array([game[1] for game in window_games], dtype=np.float32)
        ratings = bradley_terry_ratings(
            names,
            rewards,
            min_games=min_games,
            prior_strength=prior_strength,
        )
        ordered = sorted(ratings.items(), key=lambda kv: -kv[1])
        top = [name for name, _rating in ordered[:top_n]]
        top_by_day[day] = top
        diagnostics[day] = {
            "window_games": len(window_games),
            "rated_players": len(ratings),
            "top": [{"name": name, "rating": float(rating)} for name, rating in ordered[:top_n]],
        }
        print(f"\n{day}: window_games={len(window_games):,} rated={len(ratings):,}", flush=True)
        for rank, (name, rating) in enumerate(ordered[:top_n], start=1):
            print(f"  {rank:2d}. {rating:+.3f}  {name[:64]}", flush=True)
    return top_by_day, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Build day-specific top-N filters from replay zips.")
    parser.add_argument("--zip", required=True, nargs="+", help="replay zip path(s); globs are supported")
    parser.add_argument("--players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--window-days", type=int, default=2, help="trailing inclusive day window")
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--prior-strength", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--diagnostics-out", type=Path, default=None)
    args = parser.parse_args()

    if args.window_days < 1:
        parser.error("--window-days must be >= 1")
    if args.top_n < 1:
        parser.error("--top-n must be >= 1")

    zip_paths = iter_zip_paths(args.zip)
    games_by_day = scan_by_day(zip_paths, args.players, max(1, args.workers))
    top_by_day, diagnostics = fit_window_topn(
        games_by_day,
        top_n=args.top_n,
        min_games=args.min_games,
        prior_strength=args.prior_strength,
        window_days=args.window_days,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(top_by_day, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    diagnostics_payload = {
        "players": args.players,
        "top_n": args.top_n,
        "window_days": args.window_days,
        "min_games": args.min_games,
        "prior_strength": args.prior_strength,
        "zip_paths": [str(path) for path in zip_paths],
        "days": diagnostics,
    }
    diagnostics_out = args.diagnostics_out or args.out.with_name(args.out.stem + "_diagnostics.json")
    diagnostics_out.write_text(
        json.dumps(diagnostics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {diagnostics_out}")
    return 0


if __name__ == "__main__":
    if os.name != "nt":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    raise SystemExit(main())
