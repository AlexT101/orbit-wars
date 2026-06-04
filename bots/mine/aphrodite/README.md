# aphrodite

Kaggle Orbit Wars bot: DUCT simultaneous-move search using Apollo's candidate
generator and final redirect pass, with a fixed-extrapolation XGBoost value net.

## Runtime

`main.py` is the Kaggle/dev wrapper. It launches the Rust `aphrodite` daemon,
pins leaf evaluation to the fixed XGB model, and sets:

- `APHRODITE_VALUE_NET_PATH` -> `train/weights/xgb_top10_d6_fixed.json`
- `APHRODITE_EXTRAP_FIX=1`

The Rust bot uses DUCT only. Apollo candidate generation and Apollo's final
`redirect_moves`-style pass are part of the active path.

## Build

```bash
cargo build --release --bin aphrodite
```

Kaggle bundle:

```bash
python build_submission.py
```

The bundle contains only:

- `main.py`
- `aphrodite`
- `xgb_top10_d6_fixed.json`

## Training

The remaining training tree is focused on replay/data collection, SummaryV2
feature extraction, and training/rebuilding `xgb_top10_d6_fixed.json`.

Typical fixed-XGB rebuild path:

```bash
python train/filter_top10_and_train_xgb.py \
  --data train/data/fixed/combined_top10_fixed.npz \
  --model-out train/weights/xgb_top10_d6_fixed.json
```

Useful support scripts include `collect.py`, `from_replays_fast.py`,
`build_from_zip.py`, `validate_extract.py`, `rebuild_fixed_extrap.py`, and
`rebuild_and_retrain_local.py`.
