# aphrodite training pipeline

## Components

- `collect.py` — drives the rust engine, runs matches, dumps per-turn
  features from the aphrodite bot (via `APHRODITE_DUMP_FEATURES_PATH`) plus
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
- `eval.py` — runs aphrodite vs a list of opponents at a given MCTS
  budget; reports per-pairing W/L/T tallies.
- `iterate.py` — collect → train → eval orchestrator (not yet used in
  the main loop, but ready).
- `weights/` — exported AOWV files. Latest used:
  `round_1_h64_v4.bin` (23-d, hidden=64, ~62k samples).

## Round / iteration loop

Round 0 (no value net) → train round-0 weights → round 1 (collect with
round-0 weights) → train round-1 weights → repeat.

## Tunables added to the Rust bot

- `APHRODITE_VALUE_NET_PATH` — path to AOWV weights. Loader detects
  `input_dim==23` ⇒ summary path; `input_dim==2728` ⇒ raw path.
- `OW_VALUE_NET=0` — disable value net, force duck heuristic.
- `OW_VALUE_BLEND=<0..1>` — blend value-net output with heuristic:
  `value = blend * v_net + (1-blend) * v_heur`. Default 1.0.
- `OW_K_ROOT`, `OW_K_NON_ROOT` — override root / internal candidate
  counts (defaults 5 / 4).
- `OW_ROLLOUT`, `OW_ROLLOUT_DEPTH` — rollout policy + plies.
- `APHRODITE_BUDGET_MS` — wall budget per turn (default 500).
- `APHRODITE_DUMP_FEATURES_PATH` — emit per-turn raw 2728-d features to
  this file (training only).
