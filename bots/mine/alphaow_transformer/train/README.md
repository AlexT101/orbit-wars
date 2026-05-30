# alphaow training pipeline

## Components

- `collect.py` — drives the rust engine, runs matches, dumps per-turn
  features from the alphaow bot (via `ALPHAOW_DUMP_FEATURES_PATH`) plus
  the binary win/loss label. Output: NPZ with arrays
  `features [N, 2728] float32`, `labels [N] float32`,
  `meta [N, 4] int32 = [game_idx, step, player, opp_id]`.
- `summary_features.py` — derives 23-d handcrafted summary features
  from the raw 2728-d block (Python-side). Mirrored by
  `value_net::summary_features::extract` in Rust; parity is verified
  by `check_parity.py`.
- `train_summary.py` — trains a 1-hidden-layer MLP on summary features
  and exports an AOWV-format weights file (`input_dim==23` ⇒ bot uses
  the native Rust summary extractor at inference).
- `train.py` — trains an MLP on the raw 2728-d features (kept for
  comparison; the summary MLP wins).
- `train_gnn.py` — trains a small planet-only GNN (also kept for
  comparison; underperforms summary MLP at the current data scale).
- `train_summary_deep.py` — 2-hidden-layer summary MLP experiment
  (marginal gain, not yet ported to Rust).
- `from_replays_tokens.py` — extracts entity-transformer token tensors
  from scraped Kaggle replay JSONs using the Rust `extract_tokens`
  binary. Output: `tokens [N, 77, 24]`, `mask [N, 77]`, `labels`, `meta`.
- `train_transformer.py` — trains a small entity transformer value net
  on those token tensors and exports AOWV version 3 weights for Rust
  inference.
- `train_transformer_from_manifest.py` — end-to-end manifest pipeline:
  reads `manifest.csv`, downloads only missing days with `kagglehub`,
  extracts per-day token NPZs, then trains the transformer.
- `eval.py` — runs alphaow vs a list of opponents at a given MCTS
  budget; reports per-pairing W/L/T tallies.
- `iterate.py` — collect → train → eval orchestrator (not yet used in
  the main loop, but ready).
- `weights/` — exported AOWV files. Latest used:
  `round_1_h64_v4.bin` (23-d, hidden=64, ~62k samples).

## Round / iteration loop

Round 0 (no value net) → train round-0 weights → round 1 (collect with
round-0 weights) → train round-1 weights → repeat.

## Tunables added to the Rust bot

- `ALPHAOW_VALUE_NET_PATH` — path to AOWV weights. Loader detects
  `input_dim==23` ⇒ summary path; `input_dim==46` ⇒ summary-v2 path;
  `input_dim==2728` ⇒ raw path; AOWV version 3 ⇒ entity transformer.
- `OW_VALUE_NET=0` — disable value net, force duck heuristic.
- `OW_VALUE_BLEND=<0..1>` — blend value-net output with heuristic:
  `value = blend * v_net + (1-blend) * v_heur`. Default 1.0.
- `OW_K_ROOT`, `OW_K_NON_ROOT` — override root / internal candidate
  counts (defaults 5 / 4).
- `OW_ROLLOUT`, `OW_ROLLOUT_DEPTH` — rollout policy + plies.
- `ALPHAOW_BUDGET_MS` — wall budget per turn (default 500).
- `ALPHAOW_DUMP_FEATURES_PATH` — emit per-turn raw 2728-d features to
  this file (training only).

## Transformer value net

Build the Rust extractor:

```bash
cd bots/mine/alphaow
cargo build --release
```

Process scraped replay JSONs:

```bash
python3 bots/mine/alphaow/train/from_replays_tokens.py \
  --replays replays \
  --out bots/mine/alphaow/train/data/replays_tokens.npz \
  --workers 6
```

Train/export a compact first transformer:

```bash
python3 bots/mine/alphaow/train/train_transformer.py \
  --data bots/mine/alphaow/train/data/replays_tokens.npz \
  --out bots/mine/alphaow/train/weights/transformer_d64_l2_h4.bin \
  --d-model 64 \
  --layers 2 \
  --heads 4 \
  --ff-dim 128 \
  --epochs 30
```

By default the transformer now trains with a bounded time-shaped target:
fast wins are slightly closer to `+1`, slow losses are slightly closer to
`0`, and the target never leaves `[-1, 1]`. Use `--target-mode outcome`
for pure win/loss labels, or tune `--time-coef` (default `0.10`) to change
how much finish time matters.

Compare value nets offline, without running games:

```bash
python3 bots/mine/alphaow/train/eval_value_nets.py \
  --weights latest comparable-old \
  --importance
```

The evaluator reports opener/midgame/endgame metrics using the first
1/5, middle 2/5, and final 2/5 of the 500-turn game.

Run it:

```bash
ALPHAOW_VALUE_NET_PATH="$PWD/bots/mine/alphaow/train/weights/transformer_d64_l2_h4.bin" \
ALPHAOW_BUDGET_MS=500 \
python3 run_match.py alphaow apollo --seed 42 --kaggle
```

For the daily Kaggle episode datasets, install `kagglehub` once:

```bash
python3 -m pip install kagglehub
```

Then run the manifest pipeline:

```bash
python3 bots/mine/alphaow/train/train_transformer_from_manifest.py \
  --start-date 2026-05-20 \
  --end-date 2026-05-28 \
  --workers 6 \
  --out bots/mine/alphaow/train/weights/transformer_manifest.bin
```

Useful dry-run/smaller-run flags:

```bash
--limit-days 1
--limit-replays-per-day 100
--max-samples 200000
--skip-train
--skip-download
```
