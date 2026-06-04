# alphaow

Decoupled-UCT MCTS over an ow2-derived candidate policy, with a learned
value network as the leaf evaluator (post short rollout).

## Quick start

```
cargo build --release
```

To use the trained value net (recommended):

```
ALPHAOW_VALUE_NET_PATH=bots/alphaow_experimental/prometheus/train/weights/current.bin \
  python3 run_match.py alphaow apollo_fast --seed 1
```

If `ALPHAOW_VALUE_NET_PATH` is unset, the bot falls back to duck's
heuristic.

## Architecture

- `src/main.rs` — daemon (stdin obs → stdout moves).
- `src/duct.rs` — Decoupled UCT search loop.
- `src/ow2_plan.rs` — strong ow2-style planner used as a baseline policy.
- `src/pathing.rs` — `dir_to_hit` (sun/orbiting/comet-aware) and obstacle geometry.
- `src/sim.rs` — engine-faithful tick simulation.
- `src/value_net.rs` — value-net inference. Two paths:
  - "Full" — 2728-d raw two-stream + pairwise distance MLP. Kept for completeness.
  - "Summary" — 23-d handcrafted feature MLP. **This is what actually works.**
- `train/` — focused XGBoost/data/eval tooling for rebuilding datasets,
  tuning boosted models, and tracking runs in `dashboard.html`.

## Tunables (env vars)

| Variable                      | Default       | Description |
|-------------------------------|---------------|-------------|
| `ALPHAOW_VALUE_NET_PATH`      | unset         | Path to AOWV weights. Loader detects format by `input_dim` (23 = summary, 2728 = full). |
| `OW_VALUE_NET`                | `1`           | Set to `0` to bypass the net entirely (force duck heuristic). |
| `OW_VALUE_BLEND`              | `1.0`         | Mix value-net with heuristic: `blend * v_net + (1 - blend) * heuristic`. |
| `OW_VALUE_SCALE`              | `1.0`         | Multiplicative dampener on value-net output (clamped to [-1, 1] after). |
| `OW_K_ROOT`                   | `5`           | Root-level candidate count. |
| `OW_K_NON_ROOT`               | `4`           | Internal-node candidate count. |
| `OW_ROLLOUT`                  | `ow2_full`    | Rollout policy (`none` / `fast` / `ow2_short` / `ow2_full` / `ow2_fast`). |
| `OW_ROLLOUT_DEPTH`            | mode-default  | Plies per rollout. |
| `OW_NO_COOP`                  | unset         | Disable cooperation in `ow2_plan`. |
| `ALPHAOW_BUDGET_MS`           | `500`         | MCTS wall budget per turn. |
| `ALPHAOW_DUMP_FEATURES_PATH`  | unset         | Emit 2728-d features each turn for training. |
| `OW_DEBUG`                    | unset         | Per-turn debug line to stderr. |

## Training pipeline

See `train/README.md`. Short version:

```
# Build missing KaggleHub datasets from train/manifest.csv, train XGBoost,
# save train/weights/xgb_46p12_latest.json, and update train/dashboard.html.
python3 train/pipeline.py --start-date 2026-05-24 --end-date 2026-05-28 --tune

# Evaluate the current bot and append matchup results to the same dashboard:
python3 train/eval.py --weights train/weights/current.bin --dashboard-html train/dashboard.html
```
