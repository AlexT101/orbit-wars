# aphrodite

Kaggle Orbit Wars bot: DUCT simultaneous-move search using Apollo's candidate
generator and final redirect pass, with a fixed-extrapolation XGBoost value net.

## Runtime

`main.py` is the Kaggle/dev wrapper. It launches the Rust `aphrodite` daemon
and pins leaf evaluation to the current fixed XGB models:

- 2p: `train/weights/xgb_2p_qsweep_r3_top20_floor050_dropdec.json`
- 4p: `train/weights/xgb_4p_6_08_6_14.json`
- 4p late 1v1 leaves also get `APHRODITE_VALUE_NET_PATH_2P` pointing at the
  2p model.

The corrected fleet extrapolation is now the Rust default.

The Rust bot uses DUCT only. Apollo candidate generation and Apollo's final
`redirect_moves`-style pass are part of the active path.

### Opening shortcut: `APOLLO_ONLY_FIRST_TURNS`

> **Heads up — if DUCT/eval looks "disabled" on the opening turns, check this
> first.** `APOLLO_ONLY_FIRST_TURNS` (a `const` in `src/duct.rs`, default `0`)
> makes `best_move` skip the whole DUCT search + leaf eval for the first N steps
> and play apollo's top-ranked candidate (`my_candidates[0]`) directly. At `0`
> the search runs from step 0 (normal behavior). It is a compile-time constant,
> **not** an env var, so changing it requires a rebuild. With `OW_DEBUG=1` an
> apollo-only turn prints a `[duck-apollo-only]` line instead of `[duck]`.

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
- `xgb_2p_qsweep_r3_top20_floor050_dropdec.json`
- `xgb_4p_6_08_6_14.json`
- `config.json`
- `config_4p.json`

## Training

The remaining training tree is focused on replay/data collection, SummaryV2
feature extraction for 2p, SummaryV3 feature extraction for 4p, and XGBoost
training. See `train/README.md` for the full ladder pipeline.

Train from an already-combined/gated dataset (full preprocessing):

```bash
python train/train_xgb.py \
  --data train/data/2p/_ladder_work/combined.npz --no-filter \
  --quality-weight --decisiveness-weight --drop-decided \
  --zero-cols 4,8,13,17,21,25,29,33,37,40,41,61,63,64 \
  --rounds 2000 --model-out train/weights/xgb_2p.json
```

For 4p replay extraction, pass `--players 4` to `build_from_zip.py` or
`collect.py`, then train the 4p model.

Useful support scripts include `collect.py`, `build_from_zip.py`,
`build_ladder.py`, `eval.py`, and `feature_importance.py`.
