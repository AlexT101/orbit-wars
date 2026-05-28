# experimental_arch

Fork of `rl_orbit_wars/` with the following deltas from the parent repo:

- **Planet ID feature removed** — the per-planet token no longer leaks
  the slot ordering via `pid/100.0`. That slot now carries
  `ships_resolved` (see below). `PLANET_FEATURES` grows from 19 → 29
  because of the new arrival-bin features (see further below).
- **Ground-truth ships-resolved.** For each planet we forward-sim the
  *real kaggle env* (deep-copied, stepped with empty actions) until no
  fleets remain, then read off final ownership and ship counts.
  `OrbitWarsDuelEnv` computes this once per step and stashes it on the
  obs dict as `obs["_resolved"]`. Both we and the opponent read the
  cached value, so the feature is symmetric and exact end-to-end. The
  signed log-normalized resolved garrison enters the per-planet token.

  The model NEVER sees error in this feature during training, eval, BC,
  or self-play. The validator confirms 0 / 584 planet mismatches.

  The exported Kaggle bot has no env handle, so it uses an obs-only
  fallback that calls kaggle's `interpreter()` directly. This is exact
  except for one quirk: kaggle scrubs the random seed from the obs before
  agents see it, so when the resolution rollout crosses a comet-spawn
  step boundary (50/150/250/350/450) the spawned-during-rollout comets
  use a default seed and don't match reality. Impact is small (~12% of
  planet-checks see a ship-count diff with mean ~0.9 ships off, max 18)
  and only when crossing those boundaries.
- **New "resolved+1" send action** — `SEND_FRACTIONS = (0.25, 0.5, 0.75, 1.0)`
  is unchanged, but there is now a 5th send bin that sends exactly
  `target.ships_resolved + 1` — i.e. the minimum needed to capture the
  target after the currently-flying fleets resolve against it. The action
  space grows from 16385 → 20481.
- **Per-planet arrival bins.** Each planet gets `2 × 5 = 10` extra
  feature dims encoding log-normalized ship counts of incoming **mine**
  vs **enemy** fleets bucketed by arrival-delta-turn:
  `[1-2, 3-5, 6-10, 11-20, 21-50]`. Arrivals beyond 50 turns are dropped.
  Right-inclusive bins, delta = arrival_turn − current_turn.
  Per-planet feature count grows from 19 → 29.
- **Reward shaping rewritten from scratch.** No more reward modes. The
  only reward terms now are:
  - `terminal` (±1) on game end
  - `terminal_time` — small ±0.10 bonus scaled by remaining turns
  - `production_delta` — `0.05 × (Δown_production − Δenemy_production)` per step
  - `launch_penalty` — `-0.001 × num_fleets_sent_this_step`
- **Wandb instead of HTML reports.** The HTML report generator (and
  `monitor.py`) is removed. Metrics go to wandb plus the existing JSONL
  files (the JSONL files are still needed for `--resume-checkpoint`).
  Use `--no-wandb` for offline smoke tests.

The architecture (MLP entity-pair or entity_transformer), training loop,
opponent rotation, BC pretraining, and evaluation harness are otherwise
identical to `rl_orbit_wars/`.

---

## Quick start

```bash
pip install -r requirements.txt
wandb login          # one-time, picks up your API key
```

### Smoke test (no wandb, ~30 seconds)

```bash
python train.py \
  --total-steps 64 --rollout-steps 16 \
  --opponent noop --no-wandb \
  --checkpoint-dir /tmp/exp_smoke
```

Should print 4 JSON metric lines and write `/tmp/exp_smoke/latest.pt`.
Note SPS is ~3-4 (vs ~9 in the original rl_orbit_wars) because each
step deep-copies the env and rolls it forward to compute the
ground-truth resolved-planet state cache.

### Validate the resolver

```bash
python validate_resolver.py --seeds 4 --states-per-seed 5
```

Reports three things:

- **Cached path** (what the model sees during training / eval / BC /
  self-play): must be `0 / N` mismatches. This is the source-of-truth
  data injected by `OrbitWarsDuelEnv` into every obs as `obs["_resolved"]`.
- **Interpreter fallback** (exported Kaggle bot only): typically ~12%
  ship mismatches when the rollout crosses a comet-spawn step boundary,
  because kaggle scrubs the seed from the obs. Mean off ~0.9 ships.
- **Spawn-boundary crossings**: how many of the tested states had
  rollouts that crossed step%100 ∈ {50, 150, 250, 350, 450}.

### Full training run

```bash
python train.py \
  --total-steps 100000 \
  --rollout-steps 256 \
  --opponent nearest,hellburner,heuristic,snapshot_sample \
  --snapshot-every-updates 25 \
  --snapshot-pool-size 4 \
  --eval-opponents noop,random,nearest,hellburner,heuristic,self \
  --eval-games 16 \
  --checkpoint-dir checkpoints/run1 \
  --wandb-project orbit-wars-rl-experimental \
  --wandb-name run1-mlp-mixed
```

Use `--model entity_transformer --hidden 128 --transformer-layers 3 --transformer-heads 4`
to switch to the transformer backbone. With the transformer, add
`--lr-schedule cosine --lr-warmup-steps 5000` and a nonzero
`--entropy-coef-final` for stability.

### BC warm-start (optional)

```bash
python pretrain_bc.py \
  --teacher hellburner \
  --samples 50000 --epochs 16 \
  --opponents noop,nearest \
  --max-noop-fraction 0.10 \
  --out checkpoints/bc_hellburner.pt
```

BC labels only emit fraction bins (never the resolved+1 bin), since
teacher bots don't know about that action.

Then pass `--init-checkpoint checkpoints/bc_hellburner.pt`
to `train.py`.

### Resume

Same args + `--resume-checkpoint checkpoints/run1/latest.pt`.
`--total-steps` becomes *additional* steps when resuming. The wandb run
will resume by name if you pass `--wandb-name` matching the original.

### Evaluate

```bash
python evaluate.py \
  checkpoints/run1/best.pt \
  --games 30 --opponent hellburner
```

### Export to a Kaggle bot

```bash
python export_submission.py \
  checkpoints/run1/best.pt \
  --out ../bots/mine/rl_ppo_experimental/main.py
```

⚠️ Export was inherited from the parent repo and has **not** been
re-verified against the new action space (20481 actions, resolved+1
bin, ships_resolved feature). The exported bot will likely need its
inline `encode_obs` updated to compute `ships_resolved` before this
works end-to-end.

---

## What to watch in wandb

| Field | What it means |
|---|---|
| `train/mean_return_25` | recent 25-episode return; should trend up |
| `train/clip_frac`, `train/approx_kl`, `train/entropy` | PPO health; clip_frac > 0.3 = trouble |
| `train/explained_var` | value head fit; should climb toward 1 |
| `train/noop_rate`, `train/launch_rate`, `train/avg_send_bin` | action mix; avg_send_bin near 4 = resolved+1 chosen often |
| `train/train_win_rate_<opp>` | live win-rate vs each rotating opponent |
| `train/reward_production_delta` etc. | per-component reward contributions; check that launch_penalty isn't dominating |
| `eval/win_rate_<opp>`, `eval/eval_score` | logged every `--eval-every-updates` updates |

---

## File map

```
experimental_arch/
├── README.md                       ← this file
├── requirements.txt                ← adds wandb
├── train.py                        ← --no-wandb / --wandb-mode flags
├── pretrain_bc.py                  ← BC pretraining
├── evaluate.py                     ← unchanged from parent
├── export_submission.py            ← inherited; needs patching for new schema
├── validate_resolver.py            ← runs cached vs interpreter resolver checks
└── orbit_wars_rl/
    ├── env.py                      ← rewritten compute_reward + RewardWeights;
    │                                 per-step env-rollout cache injected into obs
    ├── features.py                 ← +resolve_via_env_rollout (truth),
    │                                 +_resolve_via_interpreter (obs-only fallback),
    │                                 +arrival bins, +RESOLVED_BIN; pid removed
    ├── heuristics.py               ← unchanged
    ├── model.py                    ← NUM_SEND_OPTIONS instead of len(SEND_FRACTIONS)
    ├── opponents.py                ← unchanged
    ├── ppo.py                      ← wandb logging, HTML report removed
    └── visualization.py            ← trimmed to just append_jsonl
```

## Resolver dispatch

`resolve_all_planets(obs, env=None)` in `features.py` checks three sources
of resolved-planet data in priority order:

1. **`obs["_resolved"]`** — cached ground truth. `OrbitWarsDuelEnv`
   computes this once per env step via env-rollout and injects it into
   both players' obs dicts. Zero cost per `encode_obs` call, zero error,
   symmetric across the two players. This is the path used during
   training, eval, BC, and self-play.
2. **`env` handle** — falls back to live env-rollout. Used by any caller
   that has the env but didn't precompute the cache.
3. **Obs-only interpreter call** — constructs a minimal state from the
   obs and invokes kaggle's `interpreter()` directly until no fleets
   remain. Exact except for the comet-spawn RNG (kaggle scrubs the seed
   from obs). This is the path used by the exported Kaggle bot.

All three paths use kaggle's own physics — no reimplementation. The only
divergence between them is the comet-spawn seed availability.
