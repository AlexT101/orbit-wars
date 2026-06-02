#!/usr/bin/env python3
"""Run prometheus vs a configurable list of opponents via the Kaggle engine
and save each replay JSON to a date-stamped run directory under
``~/.cache/orbit-wars/prometheus/selfplay/raw/``.

Sides alternate by seed parity (even seed → prometheus is p0, odd → p1).
The saved JSON is ``env.toJSON()`` from kaggle-environments — bit-identical
to a Kaggle competition replay.

Layout written:
    ~/.cache/orbit-wars/prometheus/selfplay/raw/YYYY-MM-DD/run_001/
        prometheus_vs_alphaow_newest_seed_000001.json
        prometheus_vs_alphaow_newest_seed_000002.json
        ...

Override the default opponent list either by editing ``DEFAULT_OPPONENTS``
at the top of this file or via ``--opponents bot1 bot2 ...``.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
BOTS_DIR = ROOT / "bots"
CACHE_ROOT = (
    Path.home() / ".cache" / "orbit-wars" / "prometheus" / "selfplay" / "raw"
)

# === EDIT ME ====================================================
# Default opponent list — override per-invocation with --opponents.
DEFAULT_OPPONENTS = [
    "alphaow_newest",
    "apollo_fast",
]

# Default games per opponent.
DEFAULT_N = 20
# First seed in the sweep; subsequent games use start_seed + i.
DEFAULT_START_SEED = 1
# Where prometheus lives. The runner fails fast if this file is missing.
PROMETHEUS_PATH = (
    BOTS_DIR / "mine" / "alphaow_experimental" / "prometheus" / "main.py"
)
# === END EDIT ME ================================================


@contextlib.contextmanager
def _silence_imports():
    """Silence noisy kaggle-environments / litellm import-time logs."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    logging.disable(logging.CRITICAL)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)
        logging.disable(logging.NOTSET)


def bot_entry(bot_name: str) -> Path:
    """Resolve a bot's main.py the same way run_match.py does:
    bots/<name>/main.py, then bots/*/<name>/main.py."""
    direct = BOTS_DIR / bot_name / "main.py"
    if direct.is_file():
        return direct
    for subdir in BOTS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / bot_name / "main.py"
        if candidate.is_file():
            return candidate
    return direct


def next_run_dir(day_dir: Path) -> Path:
    """Pick the lowest unused run_NNN under day_dir, creating it.
    Skips numbers that already exist so reruns on the same day add to a
    new bucket instead of clobbering."""
    day_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = day_dir / f"run_{n:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        n += 1


def run_match(
    prometheus_path: Path,
    opponent_path: Path,
    prometheus_p0: bool,
    seed: int,
) -> tuple[dict, list[float]]:
    """Run one kaggle match and return (replay_json, [p0_reward, p1_reward])."""
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    if prometheus_p0:
        agents = [str(prometheus_path), str(opponent_path)]
    else:
        agents = [str(opponent_path), str(prometheus_path)]
    env.run(agents)
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    return env.toJSON(), rewards


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--opponents", nargs="+", default=DEFAULT_OPPONENTS,
        help=f"Opponent bot names (default: {DEFAULT_OPPONENTS})",
    )
    parser.add_argument(
        "-n", "--n-games", type=int, default=DEFAULT_N,
        help=f"Games per opponent (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--start-seed", type=int, default=DEFAULT_START_SEED,
        help=f"First seed (default: {DEFAULT_START_SEED})",
    )
    parser.add_argument(
        "--prometheus-path", type=Path, default=PROMETHEUS_PATH,
        help=f"Path to prometheus main.py (default: {PROMETHEUS_PATH})",
    )
    parser.add_argument(
        "--day", type=str, default=None,
        help="YYYY-MM-DD bucket override (default: today's date)",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Override output dir entirely; skips auto-numbered run_NNN",
    )
    args = parser.parse_args()

    if not args.prometheus_path.is_file():
        print(
            f"ERROR: prometheus bot not found at {args.prometheus_path}.\n"
            f"Edit PROMETHEUS_PATH at the top of this file or pass "
            f"--prometheus-path.",
            file=sys.stderr,
        )
        return 1

    opponent_paths: list[tuple[str, Path]] = []
    for name in args.opponents:
        p = bot_entry(name)
        if not p.is_file():
            print(
                f"ERROR: opponent '{name}' not found (looked at {p})",
                file=sys.stderr,
            )
            return 1
        opponent_paths.append((name, p))

    day = args.day or dt.date.today().isoformat()
    if args.run_dir is not None:
        out_dir = args.run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = next_run_dir(CACHE_ROOT / day)
    print(f"Output dir: {out_dir}")
    print(f"Opponents: {[name for name, _ in opponent_paths]}")
    print(f"Games per opponent: {args.n_games}  |  start_seed: {args.start_seed}")
    print()

    with _silence_imports():
        import kaggle_environments  # noqa: F401

    for opp_name, opp_path in opponent_paths:
        wins = losses = ties = 0
        t_opp = perf_counter()
        for i in range(args.n_games):
            seed = args.start_seed + i
            prometheus_p0 = (seed % 2 == 0)
            t0 = perf_counter()
            replay_json, rewards = run_match(
                args.prometheus_path, opp_path, prometheus_p0, seed
            )
            pr = rewards[0] if prometheus_p0 else rewards[1]
            if pr > 0:
                wins += 1
            elif pr < 0:
                losses += 1
            else:
                ties += 1
            fname = f"prometheus_vs_{opp_name}_seed_{seed:06d}.json"
            (out_dir / fname).write_text(
                json.dumps(replay_json, separators=(",", ":"))
            )
            elapsed = perf_counter() - t0
            print(
                f"  {opp_name}  seed={seed:06d}  "
                f"prometheus=p{0 if prometheus_p0 else 1}  "
                f"reward={pr:+g}  W/L/T={wins}/{losses}/{ties}  "
                f"({elapsed:.1f}s)"
            )
        wr = 100.0 * wins / args.n_games if args.n_games else 0.0
        elapsed = perf_counter() - t_opp
        print(
            f"=== {opp_name}: W={wins} L={losses} T={ties} "
            f"({wr:.1f}% wr over {args.n_games}, {elapsed:.0f}s total)\n"
        )

    print(f"All matches written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
