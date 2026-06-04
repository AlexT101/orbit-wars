"""Self-play improvement loop for aphrodite.

Round r starts with weights from round r-1 (or no weights at round 0).
For each round:
  1. Collect K games of (aphrodite_r-1 vs aphrodite_r-1) self-play, plus optional cross-opponent games.
  2. Train a fresh MLP on the accumulated data (all rounds so far).
  3. Evaluate aphrodite_r vs heuristic / apollo_fast.
  4. Save weights and stats to weights/round_<r>.bin + reports/round_<r>.json.

This script is a thin orchestrator that shells out to collect.py / train.py
/ eval.py so each step is independently rerunnable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA_DIR = HERE / "data"
WEIGHTS_DIR = HERE / "weights"
REPORTS_DIR = HERE / "reports"
LOG_DIR = HERE / "logs"


def sh(cmd: list[str], log_path: Path | None = None, env_extra: dict | None = None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"$ {' '.join(cmd)}")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as f:
            f.write(f"\n$ {' '.join(cmd)}\n".encode())
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)} (exit {r.returncode})")


def round_paths(r: int):
    return {
        "data_self": DATA_DIR / f"round_{r}_self.npz",
        "data_cross": DATA_DIR / f"round_{r}_cross.npz",
        "weights": WEIGHTS_DIR / f"round_{r}.bin",
        "report": REPORTS_DIR / f"round_{r}.json",
        "log": LOG_DIR / f"round_{r}.log",
    }


def collect_round(r: int, prev_weights: Path | None, games_self: int, games_cross: int, budget_ms: int, base_seed: int):
    rp = round_paths(r)
    args_self = [
        sys.executable,
        str(HERE / "collect.py"),
        "--out", str(rp["data_self"]),
        "--games", str(games_self),
        "--budget-ms", str(budget_ms),
        "--pairings", "aphrodite:aphrodite:1",
        "--seed", str(base_seed),
    ]
    if prev_weights is not None and prev_weights.exists():
        args_self += ["--weights", str(prev_weights)]
    sh(args_self, rp["log"])

    if games_cross > 0:
        args_cross = [
            sys.executable,
            str(HERE / "collect.py"),
            "--out", str(rp["data_cross"]),
            "--games", str(games_cross),
            "--budget-ms", str(budget_ms),
            "--pairings", "aphrodite:apollo_fast:1,aphrodite:heuristic:1",
            "--seed", str(base_seed + 500),
        ]
        if prev_weights is not None and prev_weights.exists():
            args_cross += ["--weights", str(prev_weights)]
        sh(args_cross, rp["log"])


def train_round(r: int, hidden: int, epochs: int):
    rp = round_paths(r)
    # Use ALL data up to this round, not just round r.
    data_paths = []
    for rr in range(r + 1):
        for kind in ("self", "cross"):
            p = round_paths(rr)[f"data_{kind}"]
            if p.exists():
                data_paths.append(str(p))
    if not data_paths:
        raise SystemExit(f"no data files for round {r}")
    args = [
        sys.executable,
        str(HERE / "train.py"),
        "--data", *data_paths,
        "--out", str(rp["weights"]),
        "--hidden", str(hidden),
        "--epochs", str(epochs),
        "--batch-size", "256",
        "--lr", "1e-3",
        "--wd", "1e-5",
        "--seed", str(r),
    ]
    sh(args, rp["log"])


def eval_round(r: int, opponents: list[str], seeds: list[int], budget_ms: int) -> dict:
    rp = round_paths(r)
    eval_log = LOG_DIR / f"round_{r}_eval.log"
    args = [
        sys.executable,
        str(HERE / "eval.py"),
        "--weights", str(rp["weights"]),
        "--opponents", *opponents,
        "--seeds", *[str(s) for s in seeds],
        "--budget-ms", str(budget_ms),
    ]
    sh(args, eval_log)
    # Parse the log into a tally.
    tally = {}
    lines = eval_log.read_text().splitlines()
    for ln in lines:
        if ln.startswith("vs "):
            try:
                head, rest = ln.split(":", 1)
                opp = head[3:].strip()
                stat = rest.split("|")[0].strip()
                tally[opp] = stat
            except Exception:
                pass
    rp["report"].parent.mkdir(parents=True, exist_ok=True)
    rp["report"].write_text(json.dumps({"round": r, "tally": tally}, indent=2))
    return tally


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--games-self", type=int, default=30)
    p.add_argument("--games-cross", type=int, default=10)
    p.add_argument("--budget-ms", type=int, default=100)
    p.add_argument("--eval-budget-ms", type=int, default=500)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--opponents", nargs="+", default=["heuristic", "apollo_fast"])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[1, 7, 42, 100, 2025])
    p.add_argument("--start-round", type=int, default=0)
    args = p.parse_args()

    for d in (DATA_DIR, WEIGHTS_DIR, REPORTS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    for r in range(args.start_round, args.rounds):
        t0 = time.time()
        prev_weights = round_paths(r - 1)["weights"] if r > 0 else None
        if not round_paths(r)["data_self"].exists():
            print(f"=== round {r}: collect ===")
            collect_round(r, prev_weights, args.games_self, args.games_cross, args.budget_ms, base_seed=10_000 + r * 1000)
        if not round_paths(r)["weights"].exists():
            print(f"=== round {r}: train ===")
            train_round(r, args.hidden, args.epochs)
        print(f"=== round {r}: eval ===")
        tally = eval_round(r, args.opponents, args.eval_seeds, args.eval_budget_ms)
        print(f"round {r} done in {(time.time() - t0) / 60:.1f}min — tally: {tally}")


if __name__ == "__main__":
    main()
