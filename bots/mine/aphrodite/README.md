# aphrodite

Kaggle Orbit Wars bot: DUCT simultaneous-move search using Apollo's candidate
generator and final redirect pass, with XGBoost leaf evaluation.

## Runtime

`main.py` launches the Rust `aphrodite` daemon and points it at the checked-in
value-net weights and Apollo runtime configs.

The Rust bot uses DUCT, Apollo candidate generation, and Apollo's final
`redirect_moves`-style pass in the active path. Four-player games that collapse
to a live 1v1 expose the two-player value net through
`APHRODITE_VALUE_NET_PATH_2P`.

### Opening Shortcut

`APOLLO_ONLY_FIRST_TURNS` is a compile-time constant in `src/duct.rs`. While it
is active, `best_move` skips DUCT and leaf eval on opening turns and plays
Apollo's first candidate directly. Changing it requires a rebuild. With
`OW_DEBUG=1`, an Apollo-only turn prints a `[duck-apollo-only]` line.

## Build

```bash
cargo build --release --bin aphrodite
```

Kaggle bundle:

```bash
python build_submission.py
```

The bundle contains the Python wrapper, Rust binary, XGBoost weights, and Apollo
runtime configs.

## Training

The training tree contains replay/data collection, feature extraction, and
XGBoost training. See `train/README.md` for the ladder pipeline.

Train from an already-combined/gated dataset:

```bash
python train/train_xgb.py \
  --data train/data/2p/_ladder_work/combined.npz --no-filter \
  --quality-weight --decisiveness-weight --drop-decided \
  --zero-cols 4,8,13,17,21,25,29,33,37,40,41,61,63,64 \
  --rounds 2000 --model-out train/weights/xgb_2p.json
```

For four-player replay extraction, pass `--players 4` to `build_from_zip.py` or
`collect.py`, then train the four-player model.
