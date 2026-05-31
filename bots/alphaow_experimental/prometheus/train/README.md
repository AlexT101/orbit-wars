# alphaow XGBoost training pipeline

## Run This

`pipeline.py` is the normal entry point. It reads `manifest.csv`,
downloads KaggleHub episode datasets, builds cached summary/extras
features if missing, combines them into a 58-d dataset, trains XGBoost,
and updates `dashboard.html`.

Install Python deps once:

```
python3 -m pip install -r train/requirements.txt
```

```
python3 train/pipeline.py --start-date 2026-05-24 --end-date 2026-05-28 --tune
```

For a tiny smoke run:

```
python3 train/pipeline.py --limit-days 1 --limit-entries 20 --max-train-rows 5000 --tune
```

Useful flags:

- `--build-only` — download/extract/combine but do not train.
- `--train-only` — reuse `train/data/pipeline/combined_46p12.npz`.
- `--force` — rebuild cached artifacts and retrain the model.
- `--skip-download` — require already downloaded KaggleHub datasets.
- `--filter-strong` — train only rows tagged by the strong-player gate.
- `--dashboard-html PATH` — write/update a shared dashboard HTML file.

Outputs by default:

- Dataset: `train/data/pipeline/combined_46p12.npz`
- Model: `train/weights/xgb_46p12_latest.json`
- Dashboard: `train/dashboard.html`

The dataset columns are exactly `summary_v2[46] + extras_v4[12]`.
Replay datasets are intentionally not bundled in this experimental copy;
the pipeline downloads or reuses them from the manifest.

## HTML dashboard

The trend graph distinguishes validation-training records from eval
records: training validation accuracy is the translucent blue line,
eval score is green, and validation logloss is amber. Feature rows are
labeled with their raw column name plus a hover description.
Feature names and descriptions are centralized near the top of
`model_dashboard.py`: edit `SUMMARY_V2_NAMES`, `EXTRA_12_NAMES`,
`EXTRA_16_NAMES`, `FEATURE_METADATA`, or `FEATURE_OVERRIDES` when the
feature layout changes.

Eval can append matchup summaries to the same dashboard:

```
python3 train/eval.py --weights train/weights/current.bin --dashboard-html train/dashboard.html
```

## Support Files

Most files here are helpers that `pipeline.py` calls or focused debug
tools:

- Dataset builders: `build_from_zip.py`, `extras_v4_build.py`,
  `from_replays_fast.py`, `select_strong_replays.py`.
- Feature validation: `validate_extract.py`, `check_parity.py`,
  `summary_features.py`.
- Experiments/debugging: `train_gbm.py`, `xgb_tune.py`,
  `bench_combined.py`, `topn_experiments.py`, `mcts_match.py`,
  `view_replay.py`.

## Tunables added to the Rust bot

- `ALPHAOW_VALUE_NET_PATH` — path to AOWV weights. Loader detects
  `input_dim==23` ⇒ summary path; `input_dim==2728` ⇒ raw path.
- `OW_VALUE_NET=0` — disable value net, force duck heuristic.
- `OW_VALUE_BLEND=<0..1>` — blend value-net output with heuristic:
  `value = blend * v_net + (1-blend) * v_heur`. Default 1.0.
- `OW_K_ROOT`, `OW_K_NON_ROOT` — override root / internal candidate
  counts (defaults 5 / 4).
- `OW_ROLLOUT`, `OW_ROLLOUT_DEPTH` — rollout policy + plies.
- `ALPHAOW_BUDGET_MS` — wall budget per turn (default 500).
- `ALPHAOW_DUMP_FEATURES_PATH` — emit per-turn raw 2728-d features to
  this file (training only).
