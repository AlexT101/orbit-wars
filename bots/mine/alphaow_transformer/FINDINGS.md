# alphaow findings

A long autonomous training/iteration run. Summary of what was tried,
what worked, and what didn't.

## Architecture (shipped)

- alphaow = duck (Decoupled-UCT) + learned value-net leaf evaluator.
- The value net is a **23-d handcrafted summary feature MLP**:
  1-hidden-layer (64 ReLU), trained on ~60k self-play / cross-opponent
  states (~420 games). Inference ~0.7 µs/leaf via NEON SIMD.
- Default weights: `train/weights/round_2_h64_v6.bin`
  (symlinked as `current.bin`).
- The shipped feature set lives in `value_net::summary_features`
  (Rust) with a Python mirror in `train/summary_features.py`. Parity
  verified by `train/check_parity.py` (max diff < 6e-5).

## What worked

- Handcrafted summary features (planet/ship totals, planet-count
  deltas, production, pairwise inverse-distance "pressure",
  frontline distance, max single-planet ships per side, log
  ship-ratio) reach **~89-91% val sign accuracy** on held-out games.
- vs heuristic at 500ms budget: ~80% win rate (baseline alphaow
  without value net is ~50%).
- The pathing bug in `ow2_plan.rs` (line 671) — stale `angle_at_min`
  reused for the boosted `send` ship count — fixed by dropping back
  to the verified `(min_s ships, angle_at_min)` pair instead of
  mixing the two.

## What didn't beat baseline

- vs apollo_fast at 500ms: roughly **comparable to baseline**
  (~30% win rate). The value net did NOT close the apollo_fast gap.
  Multiple training rounds (v4, v6, v7, v9) all clustered around the
  same effective playing strength against apollo_fast. The MCTS
  itself, not the leaf evaluator, is the binding constraint.

## What was tried and dropped

- Raw 2728-d two-stream + pairwise-distance MLP (the spec's original
  feature set): overfit to ~71-80% val sign acc. Code kept but not
  used.
- Planet-only GNN (per-node features + distance edges, GINE-style
  message passing): val_loss 0.145 vs 0.107 for the summary MLP at
  the same data scale. Handcrafted features dominate.
- Step / step-remaining / log-ratio extra features: noisy across
  random seeds — variance > signal at current data scale.
- 2-hidden-layer MLP: marginal improvement in val loss, not ported
  to Rust inference.
- Kaggle episode scraping: API requires auth/team IDs we don't have
  for Orbit Wars.

## Tunables added to the Rust bot

- `ALPHAOW_VALUE_NET_PATH` — load weights from this AOWV file.
  Loader detects `input_dim==23` ⇒ summary path,
  `input_dim==2728` ⇒ full path.
- `OW_VALUE_NET=0` — disable value net.
- `OW_VALUE_BLEND=<0..1>` — `value = blend * v_net + (1-blend) * heur`.
- `OW_VALUE_SCALE=<0..>` — pre-blend multiplier on the net's output
  (capped to [-1, 1]). Useful for dampening overconfidence without
  retraining.
- `OW_K_ROOT`, `OW_K_NON_ROOT` — override MCTS candidate counts
  (defaults 5 / 4).
- `ALPHAOW_BUDGET_MS` — override MCTS wall budget.
- `ALPHAOW_DUMP_FEATURES_PATH` — dump per-turn 2728-d features to a
  file (training-time data capture).

## Where to push next

1. **Score-share label** instead of binary win/loss. The reward
   signal at every state would be denser and smoother. Requires
   capturing final ship totals (engine snapshot.planets/fleets).
2. **Calibration sweep**: a proper grid over `OW_VALUE_BLEND`,
   `OW_VALUE_SCALE`, `OW_K_ROOT`, `OW_ROLLOUT_DEPTH` (with v6
   weights) might find a config that beats apollo_fast. The env-var
   knobs are in place; one missing test takes ~30 min each.
3. **apollo_fast-style training data**: most of our data is alphaow
   vs (heuristic | apollo_fast). Mirror games with apollo_fast vs
   itself (with alphaow as silent observer that just extracts
   features post-hoc) would give better positional signal.
4. **MCTS-level improvements**: faster sim::tick, cheaper rollout
   policy, or AlphaZero-style policy head guiding expansion. The
   bottleneck vs apollo_fast looks search-side, not value-side.

## Files of interest

```
bots/mine/alphaow/
├── README.md                  — usage + env vars
├── FINDINGS.md                — this doc
├── run.sh                     — convenience launcher
├── Cargo.toml                 — adds bench_valnet / init_weights /
│                                 summary_parity bins
├── src/
│   ├── value_net.rs           — full + summary inference (NEON-SIMD)
│   ├── duct.rs                — Decoupled-UCT search (added env knobs)
│   ├── ow2_plan.rs            — pathing-bug fix at line 671-680
│   ├── bin/bench_valnet.rs    — per-call cost benchmark
│   ├── bin/init_weights.rs    — generate stub AOWV files
│   └── bin/summary_parity.rs  — parity probe vs Python
└── train/
    ├── README.md
    ├── collect.py             — runs games, captures features
    ├── summary_features.py    — Python derivation (parity-checked)
    ├── train_summary.py       — production trainer (1-layer)
    ├── train_summary_deep.py  — 2-layer experiment
    ├── train_gnn.py           — GNN experiment
    ├── train.py               — raw-feature trainer
    ├── eval.py                — fixed-seed eval harness
    ├── iterate.py             — collect → train → eval orchestrator
    ├── check_parity.py        — Python ↔ Rust parity test
    ├── data/                  — collected NPZ files (~67k samples)
    └── weights/               — exported AOWV files (current.bin = v6)
```
