# aphrodite training

This directory keeps the pipeline needed to collect Orbit Wars data, extract
SummaryV2 features, and train the deployed fixed-extrapolation XGB model:

`weights/xgb_top10_d6_fixed.json`

## Kept Pipeline Pieces

- `collect.py`, `mcts_match.py`, `eval.py` - match/self-play/data collection
- `from_replays.py`, `from_replays_fast.py`, `build_from_zip.py` - replay to NPZ
- `validate_extract.py`, `check_parity.py`, `summary_features.py` - extraction checks
- `combine_npz.py` - concatenate replay/self-play/candidate datasets with safe game-id offsets
- `filter_top10_and_train_xgb.py`, `train_xgb.py` - train XGB models
- `rebuild_fixed_extrap.py`, `rebuild_and_retrain_local.py` - fixed extrapolation audits
- `kaggle_rebuild_v2.py`, `select_strong_replays.py`, `view_replay.py` - replay utilities

## Current Model

```bash
python filter_top10_and_train_xgb.py \
  --input data/fixed/combined_top10_fixed.npz \
  --top10-out data/fixed/combined_top10_rebuilt.npz \
  --model-out weights/xgb_top10_d6_fixed.json
```

Useful split-weight commands:

```bash
python build_from_zip.py --players 2 --zip replays.zip --out data/2p/replays_2p.npz
python build_from_zip.py --players 4 --zip replays.zip --out data/4p/replays_4p.npz
python collect.py --players 4 --out data/4p/selfplay_4p.npz \
  --pairings aphrodite:aphrodite:apollo:apollo_fast:1.0
python combine_npz.py --out data/4p/train_4p_mixed.npz data/4p/replays_4p.npz data/4p/selfplay_4p.npz
python train_xgb.py --data data/4p/train_4p_mixed.npz --model-out weights/xgb_4p.json
```

Old AOWV/MLP/GNN/linear/first-owned experiments and their exported weights were
removed from this directory.
