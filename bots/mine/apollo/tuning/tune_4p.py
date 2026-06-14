"""Apollo 4-player constant tuner.

Searches apollo's tunable agent constants (intervals.json) to maximise apollo's
1st-place rate in 4p games, writing the result to `config_4p.json`. The 2p config
(`config.json`) is held fixed at the adopted 2p winner — apollo uses it for the
endgame phase when a 4p game collapses to 1v1 (MODE switch in constants.rs).

Differences from the 2p tuner (tune.py):
  * Uses run_batched_4p (4 bots/game, seat rotation by seed built in).
  * Objective = binary 1st-place rate (match_winner == apollo). Draws (ties for
    1st) count as NON-wins, per how the engine/Elo scores. Baseline is ~25%.
  * Opponents become ROSTERS of 3 bots; each stage's games are split evenly
    across the roster set and the objective is the blended 1st-place rate.
    Default rosters: 3x apollo_baseline, 3x producer_v2, and a mix
    (apollo_baseline, producer_v2, producer) — equal weight.
  * Gates recalibrated to the 25% baseline; larger stages (20/60/120).
  * Writes config_4p.json (env APOLLO_CONFIG_4P); leaves config.json untouched.

Worker pools are short-lived per chunk (4 native modules per worker → more memory
than 2p; chunking reclaims it reliably). A fresh pool per trial is implied by the
per-chunk pools, so apollo always re-reads the current config_4p.json.

Usage (repo root, in the venv):
    python bots/mine/apollo/tuning/tune_4p.py --trials 100 --threads 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
APOLLO_DIR = HERE.parent
ROOT = APOLLO_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CONFIG_PATH = APOLLO_DIR / "config.json"          # 2p (fixed during 4p tuning)
CONFIG_4P_PATH = APOLLO_DIR / "config_4p.json"    # 4p (tuned here)
INTERVALS_PATH = APOLLO_DIR / "intervals.json"
OUT_DIR = HERE / "runs"

# apollo reads BOTH files (constants.rs); pin each so workers agree regardless of CWD.
os.environ["APOLLO_CONFIG"] = str(CONFIG_PATH)
os.environ["APOLLO_CONFIG_4P"] = str(CONFIG_4P_PATH)

# 4p match harness (distinct seat-rotation + 4-way win decode from the 2p one).
from run_batched_4p import (  # noqa: E402
    bot_entry,
    match_winner,
    reorder_by_input,
    run_match_job as rb4_run_match_job,
    slot_order_for_seed,
)
# Pure helpers shared with the 2p tuner.
from tune import append_jsonl, load_intervals, sample_config  # noqa: E402

# ---- tuning constants (edit freely) -----------------------------------------
# Each roster is 3 opponent bots; apollo is always the 4th (input slot 0).
DEFAULT_ROSTERS = [
    # apollo_tuned is a frozen clone at the adopted 2p (avg) config — it tests our
    # 4p weights against our own 2p strategy applied to 4p (a stronger, better-
    # matched benchmark than the old apollo_baseline default constants).
    ("3x_tuned", ["apollo_tuned", "apollo_tuned", "apollo_tuned"]),
    ("3x_producer_v2", ["producer_v2", "producer_v2", "producer_v2"]),
    ("mix_tuned_prodv2_prod", ["apollo_tuned", "producer_v2", "producer"]),
]
STAGE_GAMES = [20, 60, 120]        # 200 total; larger than 2p (compressed 4p effects)
STAGE_MIN_WINS = [4, 15, None]     # >=4/20 (20%), >=15/60 (25%) — ~25% baseline
DEFAULT_BASE_SEED_RANGE = (1, 2_000_000)
CHUNK_GAMES = 48                   # games per short-lived pool (memory bound)


def safe_run_match_job_4p(job):
    """Top-level (picklable). Never lets a native panic cross the pool boundary."""
    try:
        return rb4_run_match_job(job)
    except BaseException as exc:  # noqa: BLE001 - includes pyo3 PanicException
        return ("error", job[2], repr(exc))


def _blank():
    return {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "ms_sum": 0.0, "ms_n": 0}


def _avg_ms(d):
    return (d["ms_sum"] / d["ms_n"]) if d["ms_n"] else 0.0


def _nonerror(d):
    return d["wins"] + d["losses"] + d["draws"]


def play_batch_4p(apollo_path, roster_specs, n, start_seed, threads):
    """Play ``n`` 4p games (apollo = slot 0) over seeds [start_seed, +n),
    round-robin across rosters. Returns (agg, per_roster). A win = apollo is the
    sole 1st place (match_winner == 0); draws (ties for 1st) are non-wins."""
    m = len(roster_specs)
    jobs = []
    for k in range(n):
        _name, paths = roster_specs[k % m]
        seed = start_seed + k
        jobs.append(([apollo_path, *paths], seed, k, slot_order_for_seed(seed)))

    agg = _blank()
    per = {name: _blank() for name, _ in roster_specs}
    # Short-lived pools (4 heavy native modules per worker → bound memory).
    for i in range(0, len(jobs), CHUNK_GAMES):
        chunk = jobs[i:i + CHUNK_GAMES]
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futures = [ex.submit(safe_run_match_job_4p, j) for j in chunk]
            for fut in as_completed(futures):
                res = fut.result()
                if res[0] == "error":
                    name = roster_specs[res[1] % m][0]
                    agg["errors"] += 1
                    per[name]["errors"] += 1
                    continue
                idx, _seed, slot_order, rewards, _steps, avg_ms = res
                name = roster_specs[idx % m][0]
                rewards = reorder_by_input(rewards, slot_order)
                avg_ms = reorder_by_input(avg_ms, slot_order)
                w = match_winner(rewards)
                if w is None:
                    outcome = "errors"
                elif w == 0:
                    outcome = "wins"
                elif w == "draw":
                    outcome = "draws"
                else:
                    outcome = "losses"
                agg[outcome] += 1
                per[name][outcome] += 1
                if avg_ms[0] is not None:
                    agg["ms_sum"] += avg_ms[0]
                    agg["ms_n"] += 1
                    per[name]["ms_sum"] += avg_ms[0]
                    per[name]["ms_n"] += 1
    return agg, per


def write_config_4p(values: dict):
    """Merge sampled values over config_4p.json and write WITHOUT a BOM."""
    base = json.loads(CONFIG_4P_PATH.read_text())
    base.update(values)
    with open(CONFIG_4P_PATH, "w", encoding="utf-8") as fh:
        json.dump(base, fh, indent=2)
        fh.write("\n")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _log_trial(path, trial, cfg, base, stages, cum, cum_per, cum_games, names, status):
    append_jsonl(path, {
        "ts": _now(), "trial": trial.number, "status": status, "config": cfg,
        "base_seed": base, "rosters": names, "stages": stages, "cum": cum,
        "cum_by_roster": cum_per, "cum_nonerror": _nonerror(cum),
        "win_rate": (cum["wins"] / _nonerror(cum)) if _nonerror(cum) else None,
        "avg_ms": _avg_ms(cum),
    })


def make_objective(intervals, threads, trials_log, seed_range, apollo_path, roster_specs, max_ms):
    import optuna
    names = [n for n, _ in roster_specs]

    def objective(trial):
        cfg = sample_config(trial, intervals)
        write_config_4p(cfg)
        base = random.randint(*seed_range)
        trial.set_user_attr("base_seed", base)
        trial.set_user_attr("config", cfg)

        cum = _blank()
        cum_per = {n: _blank() for n in names}
        cum_games = 0
        stage_seed = base
        stages = []
        for si, n in enumerate(STAGE_GAMES):
            agg, per = play_batch_4p(apollo_path, roster_specs, n, stage_seed, threads)
            stage_seed += n
            for key in cum:
                cum[key] += agg[key]
            for rn in names:
                for key in cum_per[rn]:
                    cum_per[rn][key] += per[rn][key]
            cum_games += n
            stages.append({"games": n, "by_roster": per, **agg})
            trial.set_user_attr("stage_reached", si + 1)

            if agg["errors"] > 0:
                _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per,
                           cum_games, names, status="error")
                raise optuna.TrialPruned()
            if max_ms and _avg_ms(cum) > max_ms:
                trial.set_user_attr("avg_ms", _avg_ms(cum))
                _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per,
                           cum_games, names, status=f"pruned_slow_s{si + 1}")
                raise optuna.TrialPruned()

            rate = cum["wins"] / _nonerror(cum) if _nonerror(cum) else 0.0
            trial.report(rate, step=cum_games)

            floor = STAGE_MIN_WINS[si]
            if floor is not None and agg["wins"] < floor:
                _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per,
                           cum_games, names, status=f"pruned_floor_s{si + 1}")
                raise optuna.TrialPruned()
            if trial.should_prune():
                _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per,
                           cum_games, names, status=f"pruned_median_s{si + 1}")
                raise optuna.TrialPruned()

        _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per, cum_games,
                   names, status="complete")
        return cum["wins"] / _nonerror(cum) if _nonerror(cum) else 0.0

    return objective


def main():
    global STAGE_GAMES, STAGE_MIN_WINS
    import optuna

    parser = argparse.ArgumentParser(description="Tune apollo's agent constants for 4p.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--study-name", default="apollo_4p_v1")
    parser.add_argument("--max-ms", type=float, default=300.0)
    parser.add_argument("--seed-low", type=int, default=DEFAULT_BASE_SEED_RANGE[0])
    parser.add_argument("--seed-high", type=int, default=DEFAULT_BASE_SEED_RANGE[1])
    parser.add_argument("--stage-games", default=None, help="Override, e.g. '20,60,120'.")
    parser.add_argument("--stage-min-wins", default=None, help="Override, e.g. '4,15,'.")
    parser.add_argument("--prune-percentile", type=float, default=50.0)
    parser.add_argument("--prune-startup", type=int, default=10)
    args = parser.parse_args()

    if args.stage_games:
        STAGE_GAMES = [int(x) for x in args.stage_games.split(",")]
    if args.stage_min_wins is not None:
        STAGE_MIN_WINS = [int(x) if x.strip() else None
                          for x in args.stage_min_wins.split(",")]
    if len(STAGE_MIN_WINS) != len(STAGE_GAMES):
        STAGE_MIN_WINS = (STAGE_MIN_WINS + [None] * len(STAGE_GAMES))[:len(STAGE_GAMES)]

    apollo_path = bot_entry("apollo")
    roster_specs = []
    for name, bots in DEFAULT_ROSTERS:
        paths = [bot_entry(b) for b in bots]
        for b, p in zip(bots, paths):
            if not p.is_file():
                parser.error(f"bot '{b}' not found at {p}")
        roster_specs.append((name, paths))
    if not apollo_path.is_file():
        parser.error(f"apollo not found at {apollo_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials_log = OUT_DIR / f"{args.study_name}_trials.jsonl"
    best_path = OUT_DIR / f"{args.study_name}_best_config.json"
    seed_range = (args.seed_low, args.seed_high)
    intervals = load_intervals()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{(OUT_DIR / (args.study_name + '.db')).as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=20),
        pruner=optuna.pruners.PercentilePruner(
            args.prune_percentile, n_startup_trials=args.prune_startup,
            n_warmup_steps=0, n_min_trials=5),
    )

    print(f"Study '{args.study_name}' (4p): apollo vs rosters "
          f"{[n for n, _ in DEFAULT_ROSTERS]}, stages {STAGE_GAMES} "
          f"({sum(STAGE_GAMES)} games/trial), threads={args.threads}")
    print(f"Objective: blended 1st-place rate (baseline ~25%, draws=non-wins). "
          f"Floors {STAGE_MIN_WINS} | prune<{args.prune_percentile:.0f}pct | max_ms={args.max_ms}")
    print(f"Writes config_4p.json; config.json (2p) held fixed.")

    def callback(study_, trial_):
        completed = [t for t in study_.trials if t.state.name == "COMPLETE"]
        pruned = [t for t in study_.trials if t.state.name == "PRUNED"]
        p1 = sum(1 for t in study_.trials if (t.user_attrs.get("stage_reached") or 0) >= 2)
        p2 = sum(1 for t in study_.trials if (t.user_attrs.get("stage_reached") or 0) >= 3)
        bv = study_.best_value if completed else None
        v = trial_.value if trial_.value is not None else float("nan")
        print(f"  [#{trial_.number}] {trial_.state.name:8} val={v:.3f} | "
              f"total={len(study_.trials)} done={len(completed)} pruned={len(pruned)} "
              f"passed_s1={p1} passed_s2={p2} best={bv}")
        if trial_.state.name == "COMPLETE" and trial_.value is not None:
            try:
                if trial_.value >= study_.best_value - 1e-12:
                    with open(best_path, "w", encoding="utf-8") as fh:
                        json.dump({"trial": trial_.number, "win_rate": trial_.value,
                                   "config": trial_.user_attrs["config"]}, fh, indent=2)
            except ValueError:
                pass

    study.optimize(
        make_objective(intervals, args.threads, trials_log, seed_range,
                       apollo_path, roster_specs, args.max_ms),
        n_trials=args.trials, callbacks=[callback])

    print("\nDone.")
    try:
        best = study.best_trial
        print(f"  best 1st-place rate={best.value:.4f}  trial={best.number}")
        print(f"  config={json.dumps(best.user_attrs['config'])}")
    except ValueError:
        print("  no trial completed (all pruned).")


if __name__ == "__main__":
    main()
