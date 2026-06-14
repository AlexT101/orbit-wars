"""Apollo constant tuner.

Searches the tunable agent constants (see ``intervals.json``) to maximise
apollo's win rate against a fixed training opponent (default: producer_v2),
then validates the best configs against a fixed reference set to catch
regressions.

How it works
------------
* Each Optuna trial samples one config from ``intervals.json``, writes it to
  ``bots/mine/apollo/config.json`` (UTF-8, NO BOM), and plays a staged batch:
  15 -> 50 -> 100 games (165 total, all on distinct seeds). Apollo reads the
  config at process start, so NO recompile is needed between trials.
* Staged pruning: a hard floor culls obviously-bad configs early (default
  >=4/15 then >=25/50), and an Optuna MedianPruner additionally prunes configs
  below the running median at each checkpoint. TPE drives exploration vs.
  exploitation across trials.
* Per-trial the seed base is randomised (so survivors keep seeing fresh maps),
  but the 165 games within a trial are a contiguous, non-overlapping block.

Critical correctness note
-------------------------
apollo_native loads config.json ONCE per process (LazyLock) and caches it for
the process lifetime. We therefore spin up a FRESH worker pool per trial and
shut it down at trial end, so no worker ever reuses a previous trial's config.

Usage (from repo root, in the venv):
    python bots/mine/apollo/tuning/tune.py --trials 500 --threads 16

Resumable: re-running with the same --study-name continues the SQLite study.
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

# --- paths / sys.path (must run at import time so spawned workers inherit it) --
HERE = Path(__file__).resolve().parent          # .../apollo/tuning
APOLLO_DIR = HERE.parent                          # .../apollo
ROOT = APOLLO_DIR.parents[2]                       # repo root (.../orbit-wars)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = APOLLO_DIR / "config.json"
INTERVALS_PATH = APOLLO_DIR / "intervals.json"
OUT_DIR = HERE / "runs"

# Point apollo at our config explicitly (children inherit this env on spawn).
os.environ["APOLLO_CONFIG"] = str(CONFIG_PATH)

# Reuse the tested match harness.
from run_batched import (  # noqa: E402
    bot_entry,
    reorder_by_input,
    run_match_job,
    slot_order_for_seed,
)

# ---- tuning constants (edit freely) -----------------------------------------
# Training opponents: each stage's games are split evenly across these, and the
# objective is the BLENDED win rate (pushes the config to generalise, not
# overfit one opponent). Override on the CLI with --opponent a,b,c.
# producer_v2 (external), apollo_baseline (frozen pre-tuning clone) and
# apollo_tuned (frozen phase-1 winner) are all trained against: beating the two
# apollo clones guards against the scoring changes regressing vs our own prior
# strategy, while producer_v2 keeps it honest vs an external bot.
TRAIN_OPPONENTS = ["producer_v2", "apollo_baseline", "apollo_tuned"]
# Reference opponents the best configs are validated against (no-regression).
# Any training opponent is auto-excluded from this set at runtime.
VALIDATION_OPPONENTS = ["producer", "simpleagent", "owheuristic", "apollo_baseline"]

# Staged game budgets and hard floors (the user's 15/50/100 scheme).
STAGE_GAMES = [15, 50, 100]            # -> 165 total
STAGE_MIN_WINS = [4, 25, None]         # hard floor per stage; None = no floor
DEFAULT_BASE_SEED_RANGE = (1, 2_000_000)


def safe_run_match_job(job):
    """Top-level so it is picklable for the worker pool. Never lets an
    unpicklable native panic cross the process boundary."""
    try:
        return run_match_job(job)
    except BaseException as exc:  # noqa: BLE001 - includes pyo3 PanicException
        # Carry the match index so the caller can attribute the error to an
        # opponent (the unpicklable native panic itself never crosses back).
        return ("error", job[2], repr(exc))


def _blank():
    return {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "ms_sum": 0.0, "ms_n": 0}


def _avg_ms(d):
    return (d["ms_sum"] / d["ms_n"]) if d["ms_n"] else 0.0


def play_batch(executor, apollo_path, opp_specs, n, start_seed):
    """Play ``n`` games of apollo (P0) over seeds [start_seed, +n), round-robin
    across ``opp_specs`` (a list of (name, path)). Returns (agg, per_opp) where
    each is a {wins,losses,draws,errors} dict; per_opp is keyed by opponent name."""
    m = len(opp_specs)
    jobs = []
    for k in range(n):
        _name, opp_path = opp_specs[k % m]
        seed = start_seed + k
        jobs.append(([apollo_path, opp_path], seed, k, slot_order_for_seed(seed)))

    agg = _blank()
    per = {name: _blank() for name, _ in opp_specs}
    futures = [executor.submit(safe_run_match_job, job) for job in jobs]
    for fut in as_completed(futures):
        res = fut.result()
        if res[0] == "error":
            idx = res[1]
            agg["errors"] += 1
            per[opp_specs[idx % m][0]]["errors"] += 1
            continue
        idx, _seed, slot_order, rewards, _steps, avg_ms = res
        name = opp_specs[idx % m][0]
        rewards = reorder_by_input(rewards, slot_order)
        avg_ms = reorder_by_input(avg_ms, slot_order)
        r0, r1 = rewards[0], rewards[1]
        if r0 is None or r1 is None:
            outcome = "errors"
        elif r0 > r1:
            outcome = "wins"
        elif r1 > r0:
            outcome = "losses"
        else:
            outcome = "draws"
        agg[outcome] += 1
        per[name][outcome] += 1
        # apollo is P0 (index 0 after reorder); track its per-move time.
        if avg_ms[0] is not None:
            agg["ms_sum"] += avg_ms[0]
            agg["ms_n"] += 1
            per[name]["ms_sum"] += avg_ms[0]
            per[name]["ms_n"] += 1
    return agg, per


def load_intervals():
    data = json.loads(INTERVALS_PATH.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def write_config(values: dict):
    """Merge sampled values over the existing config and write WITHOUT a BOM."""
    base = json.loads(CONFIG_PATH.read_text())
    base.update(values)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(base, fh, indent=2)
        fh.write("\n")
    return base


def sample_config(trial, intervals):
    cfg = {}
    for name, spec in intervals.items():
        if spec["type"] == "int":
            cfg[name] = trial.suggest_int(name, int(spec["min"]), int(spec["max"]))
        else:
            cfg[name] = trial.suggest_float(name, float(spec["min"]), float(spec["max"]))
    return cfg


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _now():
    return datetime.now(timezone.utc).isoformat()


def make_objective(intervals, threads, trials_log, seed_range, apollo_path,
                   opp_specs, max_ms):
    import optuna
    opp_names = [name for name, _ in opp_specs]

    def objective(trial):
        cfg = sample_config(trial, intervals)
        write_config(cfg)
        base = random.randint(*seed_range)
        trial.set_user_attr("base_seed", base)
        trial.set_user_attr("config", cfg)

        cum = _blank()
        cum_per = {name: _blank() for name in opp_names}
        cum_games = 0
        stage_seed = base
        stages = []
        # One pool for the whole trial: config is constant within a trial, so
        # workers may be reused across its stages. A NEW pool per trial is what
        # guarantees fresh config reads (LazyLock is per-process).
        with ProcessPoolExecutor(max_workers=threads) as ex:
            for si, n in enumerate(STAGE_GAMES):
                agg, per = play_batch(ex, apollo_path, opp_specs, n, stage_seed)
                stage_seed += n
                for key in cum:
                    cum[key] += agg[key]
                for name in opp_names:
                    for key in cum_per[name]:
                        cum_per[name][key] += per[name][key]
                cum_games += n
                stages.append({"games": n, "by_opponent": per, **agg})
                trial.set_user_attr("stage_reached", si + 1)

                # A native panic on this config => unusable; abandon the trial.
                if agg["errors"] > 0:
                    _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per,
                               cum_games, opp_names, status="error")
                    raise optuna.TrialPruned()

                # Too slow for the Kaggle per-move budget => reject early.
                if max_ms and _avg_ms(cum) > max_ms:
                    trial.set_user_attr("avg_ms", _avg_ms(cum))
                    _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per,
                               cum_games, opp_names, status=f"pruned_slow_s{si + 1}")
                    raise optuna.TrialPruned()

                rate = cum["wins"] / cum_games
                trial.report(rate, step=cum_games)

                floor = STAGE_MIN_WINS[si]
                if floor is not None and agg["wins"] < floor:
                    _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per,
                               cum_games, opp_names, status=f"pruned_floor_s{si + 1}")
                    raise optuna.TrialPruned()
                if trial.should_prune():
                    _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per,
                               cum_games, opp_names, status=f"pruned_median_s{si + 1}")
                    raise optuna.TrialPruned()

        _log_trial(trials_log, trial, cfg, base, stages, cum, cum_per, cum_games,
                   opp_names, status="complete")
        return cum["wins"] / cum_games

    return objective


def _log_trial(path, trial, cfg, base, stages, cum, cum_per, cum_games, opp_names, status):
    append_jsonl(path, {
        "ts": _now(),
        "trial": trial.number,
        "status": status,
        "config": cfg,
        "base_seed": base,
        "opponents": opp_names,
        "stages": stages,
        "cum": cum,
        "cum_by_opponent": cum_per,
        "cum_games": cum_games,
        "win_rate": (cum["wins"] / cum_games) if cum_games else None,
        "avg_ms": _avg_ms(cum),
    })


def validate(cfg, threads, games, val_log, trial_number, train_rate,
             apollo_path, val_opponents, train_names):
    """Re-write cfg and play it against each reference opponent; log results."""
    write_config(cfg)
    results = {}
    with ProcessPoolExecutor(max_workers=threads) as ex:
        for opp in val_opponents:
            base = random.randint(*DEFAULT_BASE_SEED_RANGE)
            agg, _per = play_batch(ex, apollo_path, [(opp, bot_entry(opp))], games, base)
            decided = agg["wins"] + agg["losses"]
            results[opp] = {
                "games": games, **agg, "base_seed": base,
                "win_rate": (agg["wins"] / decided) if decided else None,
            }
    append_jsonl(val_log, {
        "ts": _now(), "trial": trial_number, "train_opponents": train_names,
        "train_win_rate": train_rate, "config": cfg, "validation": results,
    })
    return results


def main():
    global STAGE_GAMES, STAGE_MIN_WINS
    import optuna

    parser = argparse.ArgumentParser(description="Tune apollo's agent constants.")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--opponent", default=",".join(TRAIN_OPPONENTS),
                        help="Training opponent(s), comma-separated. Each stage's "
                             "games are split evenly across them; objective is the "
                             "blended win rate. Default: " + ",".join(TRAIN_OPPONENTS))
    parser.add_argument("--study-name", default="apollo_v1")
    parser.add_argument("--validate-games", type=int, default=40,
                        help="Games per reference opponent when a new best is found.")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--max-ms", type=float, default=300.0,
                        help="Prune configs whose mean per-move time exceeds this "
                             "(ms), checked after each stage. 0 disables.")
    parser.add_argument("--seed-low", type=int, default=DEFAULT_BASE_SEED_RANGE[0])
    parser.add_argument("--seed-high", type=int, default=DEFAULT_BASE_SEED_RANGE[1])
    parser.add_argument("--stage-games", default=None,
                        help="Override stage budgets, comma-separated (e.g. '4,4,4' for smoke tests).")
    parser.add_argument("--stage-min-wins", default=None,
                        help="Override hard win floors, comma-separated; use '' for no floor (e.g. '1,,').")
    parser.add_argument("--prune-percentile", type=float, default=50.0,
                        help="Prune trials below this percentile of completed trials "
                             "at each checkpoint. 50=median (default); higher=more "
                             "aggressive (e.g. 65).")
    parser.add_argument("--prune-startup", type=int, default=10,
                        help="Trials to observe before the percentile pruner activates.")
    parser.add_argument("--no-enqueue-current", action="store_true",
                        help="Skip warm-starting a fresh study with the current "
                             "config.json values (the identity/anchor trial).")
    args = parser.parse_args()

    if args.stage_games:
        STAGE_GAMES = [int(x) for x in args.stage_games.split(",")]
    if args.stage_min_wins is not None:
        STAGE_MIN_WINS = [int(x) if x.strip() else None
                          for x in args.stage_min_wins.split(",")]
    if len(STAGE_MIN_WINS) != len(STAGE_GAMES):
        STAGE_MIN_WINS = (STAGE_MIN_WINS + [None] * len(STAGE_GAMES))[:len(STAGE_GAMES)]

    train_names = [o.strip() for o in args.opponent.split(",") if o.strip()]
    apollo_path = bot_entry("apollo")
    opp_specs = [(name, bot_entry(name)) for name in train_names]
    for name, p in [("apollo", apollo_path)] + opp_specs:
        if not p.is_file():
            parser.error(f"bot '{name}' not found at {p}")
    # Don't validate against an opponent we trained on.
    val_opponents = [o for o in VALIDATION_OPPONENTS if o not in train_names]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials_log = OUT_DIR / f"{args.study_name}_trials.jsonl"
    val_log = OUT_DIR / f"{args.study_name}_validation.jsonl"
    best_path = OUT_DIR / f"{args.study_name}_best_config.json"
    seed_range = (args.seed_low, args.seed_high)

    intervals = load_intervals()
    enqueue_current = not args.no_enqueue_current
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

    total_games = sum(STAGE_GAMES)
    print(f"Study '{args.study_name}': apollo vs {train_names} (blended), "
          f"{len(STAGE_GAMES)} stages {STAGE_GAMES} ({total_games} games/trial), "
          f"threads={args.threads}")
    print(f"Validation opponents: {val_opponents}")
    print(f"Floors: {STAGE_MIN_WINS} | prune<{args.prune_percentile:.0f}pct "
          f"(startup {args.prune_startup}) | max_ms={args.max_ms}")
    print(f"Logs: {trials_log.name}, {val_log.name}; storage {args.study_name}.db")

    # Warm-start a FRESH study with the current config.json values for the
    # tunable keys (the identity/anchor trial) so the search measures current
    # behavior as a baseline. Skipped on resume (study already has trials).
    if enqueue_current and not study.trials:
        base_cfg = json.loads(CONFIG_PATH.read_text())
        anchor = {}
        for k, spec in intervals.items():
            if k not in base_cfg:
                continue
            anchor[k] = int(base_cfg[k]) if spec["type"] == "int" else float(base_cfg[k])
        if len(anchor) == len(intervals):
            study.enqueue_trial(anchor)
            print(f"Warm-start: enqueued current config as anchor trial: {anchor}")
        else:
            missing = [k for k in intervals if k not in anchor]
            print(f"Warm-start SKIPPED (config.json missing tunable keys: {missing})")

    # Track the best so we only validate on genuine improvements.
    try:
        best_so_far = {"value": study.best_value}
    except ValueError:
        best_so_far = {"value": -1.0}

    def callback(study_, trial_):
        completed = [t for t in study_.trials if t.state.name == "COMPLETE"]
        pruned = [t for t in study_.trials if t.state.name == "PRUNED"]
        passed_g1 = sum(1 for t in study_.trials
                        if (t.user_attrs.get("stage_reached") or 0) >= 2)
        passed_g2 = sum(1 for t in study_.trials
                        if (t.user_attrs.get("stage_reached") or 0) >= 3)
        bv = study_.best_value if completed else None
        print(f"  [#{trial_.number}] {trial_.state.name:8} "
              f"val={trial_.value if trial_.value is not None else float('nan'):.3f} | "
              f"total={len(study_.trials)} done={len(completed)} pruned={len(pruned)} "
              f"passed15={passed_g1} passed50={passed_g2} best={bv}")

        if (not args.no_validate and trial_.state.name == "COMPLETE"
                and trial_.value is not None and trial_.value > best_so_far["value"] + 1e-9):
            best_so_far["value"] = trial_.value
            cfg = trial_.user_attrs["config"]
            with open(best_path, "w", encoding="utf-8") as fh:
                json.dump({"trial": trial_.number, "train_win_rate": trial_.value,
                           "opponents": train_names, "config": cfg}, fh, indent=2)
            if val_opponents:
                print(f"  -> new best {trial_.value:.3f}; validating vs {val_opponents}")
                res = validate(cfg, args.threads, args.validate_games, val_log,
                               trial_.number, trial_.value, apollo_path,
                               val_opponents, train_names)
                for opp, r in res.items():
                    rr = r["win_rate"]
                    print(f"       {opp:16} {r['wins']}/{r['games']} "
                          f"({(rr * 100 if rr is not None else float('nan')):.1f}%)")
                # Restore the best config so the next trial starts from a known file.
                write_config(cfg)
            else:
                print(f"  -> new best {trial_.value:.3f} (no validation opponents)")

    study.optimize(
        make_objective(intervals, args.threads, trials_log, seed_range,
                       apollo_path, opp_specs, args.max_ms),
        n_trials=args.trials, callbacks=[callback])

    print("\nDone.")
    try:
        best = study.best_trial
        print(f"  best value={best.value:.4f}  trial={best.number}")
        print(f"  config={json.dumps(best.user_attrs['config'])}")
        print(f"  saved -> {best_path}")
    except ValueError:
        print("  no trial completed (all pruned).")


if __name__ == "__main__":
    main()
