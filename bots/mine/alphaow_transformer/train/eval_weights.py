"""Evaluate one alphaow value-net checkpoint against older alphaow baselines.

This uses Kaggle's Python environment and the direct alphaow daemon wrapper from
collect.py. Each side gets its own subprocess and env, so candidate and baseline
weights can be compared in the same match without ALPHAOW_VALUE_NET_PATH leaks.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE / "weights"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import collect  # type: ignore

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"

DEFAULT_BASELINES = [
    "heuristic",
    "current",
    "round_2_h64_v6",
    "round_5_h32_v9",
    "v2_replays",
]


class KaggleDaemonAgent:
    def __init__(self, daemon: collect.AlphaowDaemon):
        self.daemon = daemon

    def __call__(self, obs, _config=None):
        return self.daemon(obs)


@contextlib.contextmanager
def silence_import_noise():
    """Mute noisy optional Kaggle imports while importing the environment."""

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


def color(enabled: bool, text: str, code: str) -> str:
    if not enabled:
        return text
    return f"{code}{text}{RESET}"


def pct(wins: int, losses: int, ties: int) -> float:
    n = wins + losses + ties
    return (wins + 0.5 * ties) / max(n, 1)


def path_label(path: Path | None) -> str:
    if path is None:
        return "heuristic"
    try:
        return path.relative_to(WEIGHTS_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_weight(name: str | None) -> Path | None:
    if name is None:
        return None
    key = name.strip()
    if key.lower() in {"heuristic", "none", "off", "no-value-net"}:
        return None

    direct = Path(key).expanduser()
    candidates = []
    if direct.is_absolute() or direct.parent != Path("."):
        candidates.append(direct)
    candidates.append(WEIGHTS_DIR / key)
    if direct.suffix != ".bin":
        candidates.append(WEIGHTS_DIR / f"{key}.bin")

    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise SystemExit(f"unknown weights baseline/candidate: {name}")


def newest_weight() -> Path:
    bins = sorted(WEIGHTS_DIR.glob("*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not bins:
        raise SystemExit(f"no .bin weights found in {WEIGHTS_DIR}")
    return bins[0].resolve()


def expand_baselines(names: list[str], candidate: Path | None) -> list[tuple[str, Path | None]]:
    expanded: list[str] = []
    for name in names:
        if name == "default":
            expanded.extend(DEFAULT_BASELINES)
        elif name == "all-old":
            expanded.append("heuristic")
            expanded.extend(p.stem for p in sorted(WEIGHTS_DIR.glob("*.bin"), key=lambda p: p.stat().st_mtime))
        else:
            expanded.append(name)

    baselines: list[tuple[str, Path | None]] = []
    seen: set[str] = set()
    candidate_key = str(candidate) if candidate is not None else "heuristic"
    for name in expanded:
        weight = resolve_weight(name)
        key = str(weight) if weight is not None else "heuristic"
        if key == candidate_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        baselines.append((name, weight))
    return baselines


def run_one(
    candidate: Path | None,
    baseline: Path | None,
    seed: int,
    candidate_as_p0: bool,
    budget_ms: int,
):
    with silence_import_noise():
        from kaggle_environments import make

    sides = [None, None]
    if candidate_as_p0:
        side_weights = [candidate, baseline]
    else:
        side_weights = [baseline, candidate]

    daemons: list[collect.AlphaowDaemon] = []
    try:
        for i, weight in enumerate(side_weights):
            daemon = collect.AlphaowDaemon(
                dump_path=None,
                budget_ms=budget_ms,
                weights_path=weight,
                value_net_off=(weight is None),
            )
            sides[i] = KaggleDaemonAgent(daemon)
            daemons.append(daemon)

        start = perf_counter()
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.run(sides)
        wall_s = perf_counter() - start

        final = env.steps[-1]
        rewards = [float(final[0].reward or 0.0), float(final[1].reward or 0.0)]
        statuses = [str(final[0].status), str(final[1].status)]
        candidate_reward = rewards[0] if candidate_as_p0 else rewards[1]
        baseline_reward = rewards[1] if candidate_as_p0 else rewards[0]
        outcome = "W" if candidate_reward > baseline_reward else ("L" if candidate_reward < baseline_reward else "T")

        return {
            "outcome": outcome,
            "rewards": rewards,
            "statuses": statuses,
            "wall_s": wall_s,
        }
    finally:
        for daemon in daemons:
            daemon.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a candidate alphaow value-net against old alphaow weights.")
    parser.add_argument(
        "--candidate",
        default="latest",
        help="Candidate weights path/name. Use latest, heuristic/none, or a file in train/weights.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["default"],
        help="Baseline names/paths. Special values: default, all-old, heuristic.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42, 100, 2025])
    parser.add_argument("--budget-ms", type=int, default=100)
    parser.add_argument("--no-swap", action="store_true", help="Only play candidate as player 0.")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--list", action="store_true", help="List known weight files and exit.")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    if args.list:
        print(color(use_color, "known weights", BOLD))
        print("  heuristic (value net off)")
        for p in sorted(WEIGHTS_DIR.glob("*.bin"), key=lambda x: x.stat().st_mtime, reverse=True):
            print(f"  {p.stem:<42} {p.stat().st_size:>9} bytes  {p.name}")
        return 0

    candidate = newest_weight() if args.candidate == "latest" else resolve_weight(args.candidate)
    baselines = expand_baselines(args.baselines, candidate)
    if not baselines:
        raise SystemExit("no baselines to evaluate after filtering out the candidate")

    print(
        color(use_color, "candidate ", BOLD)
        + color(use_color, path_label(candidate), CYAN)
        + f"  budget={args.budget_ms}ms  seeds={','.join(map(str, args.seeds))}"
    )

    all_results = []
    for baseline_name, baseline in baselines:
        label = path_label(baseline)
        print(color(use_color, f"\nvs {label}", BOLD))
        wins = losses = ties = 0
        wall = []
        sides = [True] if args.no_swap else [True, False]
        for seed in args.seeds:
            for candidate_as_p0 in sides:
                result = run_one(candidate, baseline, seed, candidate_as_p0, args.budget_ms)
                wall.append(result["wall_s"])
                outcome = result["outcome"]
                if outcome == "W":
                    wins += 1
                    outcome_text = color(use_color, "W", GREEN)
                elif outcome == "L":
                    losses += 1
                    outcome_text = color(use_color, "L", RED)
                else:
                    ties += 1
                    outcome_text = color(use_color, "T", YELLOW)
                if args.progress:
                    wr = pct(wins, losses, ties)
                    side = "p0" if candidate_as_p0 else "p1"
                    print(
                        f"  seed={seed:<6} cand={side} {outcome_text} "
                        f"score={result['rewards'][0]:.0f}:{result['rewards'][1]:.0f} "
                        f"wr={wr:5.1%} wall={result['wall_s']:.1f}s",
                        flush=True,
                    )

        n = wins + losses + ties
        wr = pct(wins, losses, ties)
        all_results.append((label, wins, losses, ties, wr, sum(wall) / max(len(wall), 1)))

    print(color(use_color, "\nsummary", BOLD))
    print("baseline                                      W   L   T   score   avg_wall")
    for label, wins, losses, ties, wr, avg_wall in sorted(all_results, key=lambda r: r[4], reverse=True):
        wr_text = f"{wr:6.1%}"
        if use_color:
            wr_text = color(use_color, wr_text, GREEN if wr >= 0.55 else RED if wr < 0.45 else YELLOW)
        print(f"{label:<43} {wins:>2} {losses:>3} {ties:>3} {wr_text}   {avg_wall:>6.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
