# rl_poc — Orbit Wars RL proof of concept

Minimum-viable PPO training loop. The point of this bot is not to win — it's
to validate that the RL pipeline (env → features → policy → PPO update →
checkpoint → wandb logging → playable agent) runs end-to-end. The full
design we'll grow into lives in [`docs/rl_design.md`](../../../docs/rl_design.md).

## Action space

Per turn, for each owned planet, the policy picks **one** of:

- one of the up-to-`MAX_PLANETS-1` other planets as a target, or
- the no-op "stay" option.

Ship count is fixed at `target.ships + 1` (capped at `source.ships - 1`).
Aim angle is straight at the target's current position — no
blocker/orbital-prediction. The point of this is intentionally crude.

## Model

Tiny actor-critic on a fixed-size padded planet table:

- Per-planet MLP encoder over `MAX_PLANETS=40` slots (mask invalid slots).
- For each source × target pair: 3-layer MLP scorer over
  `[src_embed, tgt_embed, global_features]` → logit.
- Value head: mean-pooled encoder output → MLP → scalar.

No attention, no apollo features, no shared encoder tricks. ~50k params.

## Files

| Path | Role |
|---|---|
| `main.py` | Kaggle agent entry — loads the latest checkpoint and runs the policy greedily |
| `src/rl_poc/features.py` | Observation → padded planet table + globals |
| `src/rl_poc/model.py` | Actor-critic |
| `src/rl_poc/policy.py` | Sample / argmax action given model + obs |
| `src/rl_poc/opponents.py` | Random and nearest-sniper baseline agents |
| `src/rl_poc/env.py` | Self-play wrapper around Kaggle's `orbit_wars` env |
| `src/rl_poc/train.py` | PPO training loop with wandb |
| `checkpoints/` | Where `train.py` saves `latest.pt`; `main.py` reads from here |

## Training

```
cd ~/orbitwars/pantheow/bots/mine/rl_poc
python -m rl_poc.train --opponent nearest-sniper --updates 50 --games-per-update 16
```

Use `--wandb-mode offline` to skip cloud logging while iterating.

## Playing a match

After at least one training update has written `checkpoints/latest.pt`:

```
cd ~/orbitwars/pantheow
python run_match.py rl_poc nearest-sniper --seed 0
```

If no checkpoint exists `main.py` falls back to a uniform random policy.
