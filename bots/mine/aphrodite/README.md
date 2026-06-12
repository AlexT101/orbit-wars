# aphrodite

Kaggle Orbit Wars bot: DUCT simultaneous-move search using Apollo's candidate
generator and final redirect pass, with a fixed-extrapolation XGBoost value net.

## Runtime

`main.py` is the Kaggle/dev wrapper. It launches the Rust `aphrodite` daemon
and pins leaf evaluation to the fixed XGB model:

- `APHRODITE_VALUE_NET_PATH` -> `train/weights/xgb_2p.json`,
  `train/weights/xgb_4p.json`, or fallback `train/weights/xgb_2p_old_top10.json`

The corrected fleet extrapolation is now the Rust default.

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
- `xgb_2p_old_top10.json`
- optional `xgb_2p.json` / `xgb_4p.json`

## Training

The remaining training tree is focused on replay/data collection, SummaryV2
feature extraction, and training/rebuilding `xgb_2p_old_top10.json`.

Typical fixed-XGB rebuild path:

```bash
python train/train_xgb.py \
  --input train/data/2p/old_top10.npz \
  --top10-out train/data/2p/old_top10_rebuilt.npz \
  --model-out train/weights/xgb_2p_old_top10.json
```

Train from an already-combined/gated dataset (full preprocessing):

```bash
python train/train_xgb.py \
  --data train/data/2p/_ladder_work/combined.npz --no-filter \
  --quality-weight --decisiveness-weight --drop-decided \
  --zero-cols 4,8,13,17,21,25,29,33,37,40,41,61,63,64 \
  --rounds 2000 --model-out train/weights/xgb_2p.json
```

For 4p replay extraction, pass `--players 4` to `build_from_zip.py`,
`from_replays_fast.py`, or `collect.py`, then train `train/weights/xgb_4p.json`.

Useful support scripts include `collect.py`, `from_replays_fast.py`,
`build_from_zip.py`, and `feature_importance.py`.
