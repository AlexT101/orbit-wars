# Apollo constant tuner

Searches apollo's tunable agent constants to maximise win rate against a fixed
training opponent set, then validates the best configs against a reference set.

## How it works

- apollo loads its agent constants from `bots/mine/apollo/config.json` at
  **runtime** (`constants.rs` → `LazyLock` + `serde_json`). The tuner rewrites
  that file each trial, so **no recompile** is needed between trials.
  - config.json must be UTF-8 **without a BOM** (serde_json rejects a BOM →
    apollo panics). Python `json.dump` is fine; PowerShell `Set-Content -Encoding
    utf8` is NOT.
- Each Optuna trial samples one config from `intervals.json`, plays a staged
  batch (default **15 → 50 → 100 = 165 games**, all distinct seeds), and returns
  the blended win rate. TPE drives exploration/exploitation.
- **Staged pruning:** a hard floor culls bad configs early (default ≥4/15, then
  ≥25/50) and an Optuna `MedianPruner` additionally prunes configs below the
  running median at each checkpoint.
- **Training opponents** (`--opponent`, comma-separated): each stage's games are
  split evenly across them; objective = blended win rate. Default:
  `producer_v2,apollo_baseline`.
- **Validation:** when a trial sets a new best, it is replayed against the
  reference opponents (`producer`, `simpleagent`, `owheuristic`, `apollo_baseline`
  minus any training opponent) and logged — a no-regression check.
- The LazyLock config cache is per-process, so the tuner uses a **fresh worker
  pool per trial** (workers never reuse a previous trial's constants).

## Run (repo root, in the venv)

```powershell
$env:VIRTUAL_ENV = "C:\Users\alext\Github\orbit-wars\venv"
python bots/mine/apollo/tuning/tune.py --trials 500 --threads 16
```

Resumable: re-run with the same `--study-name` to continue the SQLite study.

Useful flags: `--opponent a,b`, `--study-name NAME`, `--validate-games N`,
`--no-validate`, `--stage-games "15,50,100"`, `--stage-min-wins "4,25,"`,
`--seed-low/--seed-high`.

## Outputs (`runs/`, git-ignored)

- `<study>.db` — Optuna SQLite study (resumable, source of truth).
- `<study>_trials.jsonl` — every trial: config, base seed, per-stage and
  per-opponent W/L/D/E, status (complete / pruned_floor / pruned_median / error).
- `<study>_validation.jsonl` — validation results for each new best.
- `<study>_best_config.json` — best config so far.

## Applying a result

The tuner leaves `config.json` at the **last trial's** values (a scratch file
during search). To adopt a winner, copy the `config` block from
`runs/<study>_best_config.json` into `bots/mine/apollo/config.json`. To revert to
defaults, restore the committed `config.json`.

## Scope

Tunes 9 "easy" strategy constants (see `intervals.json`). Excluded for now and
held at compile-time defaults: all `EARLY_GAME_*` (early-game pre-pass is off,
`EARLY_GAME_END=0`), `REACTIVE_TURNS`, `AIM_HORIZON`, `MAX_SOURCES*`, and the
cone/nudge sim-cost knobs. Promote any of them by adding a key to `config.json`
+ `intervals.json` and wiring it in `constants.rs` the same way.
