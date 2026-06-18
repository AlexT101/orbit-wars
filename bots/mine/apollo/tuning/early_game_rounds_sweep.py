"""Sweep Apollo's compile-time EARLY_GAME_END constant.

Only the single `EARLY_GAME_END` line in src/constants.rs is rewritten. For each
value, Apollo is rebuilt into the repo venv and run_batched.py is run serially
against the configured opponents.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


HERE = Path(__file__).resolve().parent
APOLLO = HERE.parent
REPO = APOLLO.parents[2]
VENV = REPO / "venv"
PY = VENV / "Scripts" / "python.exe"
MATURIN = VENV / "Scripts" / "maturin.exe"
CONSTANTS = APOLLO / "src" / "constants.rs"
OUT = HERE / "early_game_rounds_results"
LOGS = OUT / "logs"

VALUES = [0, 2, 4, 6, 8, 10]
OPPONENTS = [
    ("producer_v2", 750, 50000),
    ("apollo_baseline", 750, 60000),
    ("producer", 500, 70000),
    ("simpleagent", 200, 80000),
    ("owheuristic", 200, 90000),
    ("apollo_tuned", 750, 100000),
]
THREAD_CAPS = {
    "apollo_baseline": 4,
    "apollo_tuned": 4,
}
CHUNK_SIZES = {
    "apollo_baseline": 150,
    "apollo_tuned": 150,
}

CONST_RE = re.compile(r"pub const EARLY_GAME_END: i64 = \d+;")
SEED_RE = re.compile(
    r"seed=\s*(?P<seed>\d+)\s+steps=\s*(?P<steps>\d+)\s+"
    r"r=\((?P<r0>[^,]+),(?P<r1>[^)]+)\)\s+->\s+(?P<winner>\S+)\s+"
    r"ms=\((?P<ms0>[^,]+),(?P<ms1>[^)]+)\)"
)


def set_early_game_end(value: int) -> None:
    text = CONSTANTS.read_text(encoding="utf-8")
    new_text, n = CONST_RE.subn(f"pub const EARLY_GAME_END: i64 = {value};", text, count=1)
    if n != 1:
        raise SystemExit(f"expected exactly one EARLY_GAME_END line in {CONSTANTS}, replaced {n}")
    CONSTANTS.write_text(new_text, encoding="utf-8")
    print(f"set EARLY_GAME_END={value}", flush=True)


def run_logged(cmd: list[str | Path], log_path: Path, cwd: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    merged_env = dict(os.environ, PYTHONUTF8="1")
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line.lstrip().startswith("seed="):
                print(line, end="")
            log.write(line)
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, [str(c) for c in cmd])


def build(value: int) -> None:
    run_logged(
        [MATURIN, "develop", "--release"],
        LOGS / f"build_eg{value}.log",
        cwd=APOLLO,
        env={"VIRTUAL_ENV": str(VENV)},
    )


def eval_log_path(value: int, opponent: str, matches: int, start_seed: int) -> Path:
    return LOGS / f"eg{value}_vs_{opponent}_s{start_seed}_n{matches}.log"


def chunk_log_path(value: int, opponent: str, start_seed: int, matches: int) -> Path:
    return LOGS / f"eg{value}_vs_{opponent}_s{start_seed}_n{matches}_chunk.log"


def chunks(matches: int, start_seed: int, chunk_size: int) -> list[tuple[int, int]]:
    specs = []
    offset = 0
    while offset < matches:
        chunk_matches = min(chunk_size, matches - offset)
        specs.append((start_seed + offset, chunk_matches))
        offset += chunk_matches
    return specs


def log_complete(path: Path, matches: int) -> bool:
    if not path.is_file():
        return False
    rows = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if SEED_RE.search(line):
            rows += 1
    return rows == matches


def combine_logs(paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", errors="replace") as out:
        for path in paths:
            out.write(f"===== {path.name} =====\n")
            out.write(path.read_text(encoding="utf-8", errors="replace"))
            if not out.tell():
                continue
            out.write("\n")


def run_eval(value: int, opponent: str, matches: int, start_seed: int, threads: int, force: bool) -> None:
    log_path = eval_log_path(value, opponent, matches, start_seed)
    if log_complete(log_path, matches) and not force:
        print(f"[skip] {log_path.relative_to(REPO)} exists")
        return

    effective_threads = min(threads, THREAD_CAPS.get(opponent, threads))
    chunk_size = CHUNK_SIZES.get(opponent)
    if not chunk_size:
        run_logged(
            [
                PY,
                REPO / "run_batched.py",
                "apollo",
                opponent,
                str(matches),
                "--start-seed",
                str(start_seed),
                "--threads",
                str(effective_threads),
            ],
            log_path,
            cwd=REPO,
        )
        return

    chunk_paths = []
    for chunk_seed, chunk_matches in chunks(matches, start_seed, chunk_size):
        chunk_path = chunk_log_path(value, opponent, chunk_seed, chunk_matches)
        chunk_paths.append(chunk_path)
        if log_complete(chunk_path, chunk_matches) and not force:
            print(f"[skip] {chunk_path.relative_to(REPO)} exists")
            continue
        run_logged(
            [
                PY,
                REPO / "run_batched.py",
                "apollo",
                opponent,
                str(chunk_matches),
                "--start-seed",
                str(chunk_seed),
                "--threads",
                str(effective_threads),
            ],
            chunk_path,
            cwd=REPO,
        )

    if all(log_complete(path, chunk_matches) for path, (_chunk_seed, chunk_matches) in zip(chunk_paths, chunks(matches, start_seed, chunk_size))):
        combine_logs(chunk_paths, log_path)


def run_matrix(force: bool, threads: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        for value in VALUES:
            needed = [
                eval_log_path(value, opponent, matches, start_seed)
                for opponent, matches, start_seed in OPPONENTS
            ]
            if not force and all(
                log_complete(path, matches)
                for path, (_opponent, matches, _start_seed) in zip(needed, OPPONENTS)
            ):
                print(f"[skip] EARLY_GAME_END={value} all eval logs exist")
                continue
            set_early_game_end(value)
            build(value)
            for opponent, matches, start_seed in OPPONENTS:
                run_eval(value, opponent, matches, start_seed, threads, force)
    finally:
        set_early_game_end(0)
        build(0)


def parse_float(text: str) -> float | None:
    text = text.strip()
    if text == "None":
        return None
    return float(text)


def parse_log(path: Path) -> dict[str, float | int]:
    wins = losses = draws = rows = 0
    sum_ms0 = sum_ms1 = sum_steps = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SEED_RE.search(line)
        if not m:
            continue
        rows += 1
        r0 = parse_float(m.group("r0"))
        r1 = parse_float(m.group("r1"))
        if r0 is None or r1 is None or r0 == r1:
            draws += 1
        elif r0 > r1:
            wins += 1
        else:
            losses += 1
        sum_steps += int(m.group("steps"))
        sum_ms0 += float(m.group("ms0"))
        sum_ms1 += float(m.group("ms1"))
    if rows == 0:
        raise SystemExit(f"no match rows parsed from {path}")
    decided = wins + losses
    return {
        "matches": rows,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / decided if decided else 0.0,
        "avg_ms_apollo": sum_ms0 / rows,
        "avg_ms_opp": sum_ms1 / rows,
        "avg_steps": sum_steps / rows,
    }


def analyze() -> None:
    rows = []
    for value in VALUES:
        for opponent, matches, start_seed in OPPONENTS:
            path = eval_log_path(value, opponent, matches, start_seed)
            if not path.is_file():
                print(f"[missing] {path.relative_to(REPO)}", file=sys.stderr)
                continue
            row = {
                "early_game_end": value,
                "opponent": opponent,
                "start_seed": start_seed,
                **parse_log(path),
            }
            rows.append(row)
    out = OUT / "summary.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"\nwrote {out.relative_to(REPO)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("run", "analyze"), nargs="?", default="run")
    p.add_argument("--force", action="store_true")
    p.add_argument("--threads", type=int, default=14)
    args = p.parse_args()

    if args.stage == "run":
        run_matrix(args.force, args.threads)
    else:
        analyze()


if __name__ == "__main__":
    main()
