# aphrodite training

This directory keeps the pipeline needed to collect Orbit Wars data, extract
SummaryV2 features, and train the deployed fixed-extrapolation XGB model:

`weights/xgb_top10_d6_fixed.json`

## Kept Pipeline Pieces

- `collect.py`, `mcts_match.py`, `eval.py` - match/self-play/data collection
- `from_replays.py`, `from_replays_fast.py`, `build_from_zip.py` - replay to NPZ
- `validate_extract.py`, `check_parity.py`, `summary_features.py` - extraction checks
- `filter_top10_and_train_xgb.py` - train the deployed XGB model
- `rebuild_fixed_extrap.py`, `rebuild_and_retrain_local.py` - fixed extrapolation audits
- `kaggle_rebuild_v2.py`, `select_strong_replays.py`, `view_replay.py` - replay utilities

## Current Model

```bash
python filter_top10_and_train_xgb.py \
  --data data/fixed/combined_top10_fixed.npz \
  --model-out weights/xgb_top10_d6_fixed.json
```

Old AOWV/MLP/GNN/linear/first-owned experiments and their exported weights were
removed from this directory.
