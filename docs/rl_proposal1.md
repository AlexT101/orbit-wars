# Orbit Wars RL design

This is the long-form plan we landed on after reviewing apollo. The
proof-of-concept under `bots/mine/rl_poc/` is a stripped-down precursor —
the full design below is what we'll grow into once the POC validates the
training pipeline.

## Reframing: apollo is an action-space generator, not a baseline

Apollo already solves the hard physics of Orbit Wars:

- Blocker-aware aiming + travel-time + L1/L2/L3 caches (`aim_with_prediction`).
- Per-target subset enumeration over source coalitions × {uncoordinated,
  coordinated@A_S+k} schedules × per-source delay sweep.
- Trial-timeline with halved enemy arrivals + final-owner check ⇒ only
  commit when the combined arrival actually flips the planet.
- Source-vulnerability bookkeeping (`not_doomed`, `neighbor_holds_under_worst_case`).
- Marginal-source halve-trim.
- Front-line BFS reinforcement.
- Early-game DFS for the first 3 turns.
- Rollout-based candidate selection across 4 selection strategies, plus
  2p opponent-variant minimax.

What apollo does crudely or not at all:

1. `score_capture = production × remaining × zero_sum_mult` — 1-line linear value.
2. `SelectionStrategy` — 4 hand-rolled keys, picked by rollout (coarse 4-arm bandit).
3. `score_probe` (rollout end-state eval) — another linear formula on a state
   that's already been distorted by 2 reactive + 20 ballistic turns.
4. Horizon ~30 turns — strategic positioning, endgame tempo, denial
   sacrifices, comet-window timing, 4p kingmaker dynamics are all invisible.
5. Opponent modeling is symmetric/greedy and exploitable.

So the RL job is: **replace the linear/handcrafted scoring functions
with learned ones while keeping apollo's heavy machinery intact.**

## Algorithm: actor–critic PPO

Policy-based with value head as critic.

- **Not Q-learning**: action space is *state-dependent* — apollo emits K
  candidate plans per turn, K varies, each plan is a concrete object, not
  an index. Q over a fixed enumeration doesn't apply.
- **Not REINFORCE**: 500-step episodes with ±1 terminal reward have huge
  variance without a baseline.
- **PPO**: standard for self-play zero-sum games (AlphaStar, OpenAI Five).
  Clip keeps updates stable under shifting opponent pool; tiny action
  space (K ≤ 32) means the clip rarely fights us.

## Network shape: shared encoder, two heads

```
   planet tokens (variable N, ~30–40)
         │
         ▼
   linear proj to d=64
         │
         ▼
   2× self-attention (4 heads, FFN) + learned [CLS]
         │
         ├──► [CLS] ─┬─► V head (MLP) → V(s) ∈ [-1, +1]
         │          │
    updated planet  │
    embeddings      ▼
         │      per-candidate scorer:
         │      for each plan a_k:
         │        pool source/target planet embeddings + plan features + globals
         │        MLP → logit ℓ_k
         │      softmax over ℓ_k → π(a|s)
         ▼
   (shared by V and π)
```

Shared encoder — V and π need overlapping board understanding, and the
encoder dominates compute. Both heads' gradients flow through it.

## Features

All directly available from apollo's `WorldState` and `timeline_cache.baseline(id)`.

### Per-planet token (~40 dims)

- **Identity (8)**: owner one-hot rotated so me=0; is_comet, is_static,
  is_my_home, is_frontline.
- **Static-ish (5)**: log(production), radius, (x−50)/50, (y−50)/50, dist_to_sun/50.
- **Comet (2)**: comet_life_remaining/30, has_expiry_within_horizon.
- **Current (2)**: log(1+ships), ships/100 clipped.
- **Topology (5)**: dist to nearest mine/enemy/neutral; count of
  mine/enemy planets within MAX_DISTANCE=38.
- **Timeline (5)** — from `baseline()`: keep_needed/ships, holds_full,
  fall_turn normalized, first_enemy normalized, min_owned/ships.
- **Trajectory samples (13)** — from `baseline().owner_at[t]` and
  `ships_at[t]` at t ∈ {5, 10, 20, 30}.

### No explicit fleet tokens

In-flight fleets are already absorbed into per-planet timeline features
via `timeline_cache.arrivals()`. Adding fleet tokens duplicates info.

### Global features (~16 dims)

step/500, remaining/500, my_strength_share, max_enemy_strength_share,
my_prod_share, enemy_prod_share, log(my_total), log(enemy_total),
log(my_prod), log(enemy_prod), planet-share my/enemy, num_players one-hot
(2 dims), turns_to_next_comet/100, angular_velocity normalized.

### Per-candidate plan features (~12 dims) — policy head only

For each candidate from apollo's extended `search_candidates`:

- num_orders, num_distinct_targets, num_distinct_sources.
- total_ships_committed / my_total.
- mean / max / std arrival_turn.
- predicted Δ-planets-captured (trial-timeline flips).
- predicted Δ-production over horizon.
- max source vulnerability spike.
- Jaccard target-set similarity vs other candidates (diversity).

Plus learned pooling from the encoder: mean of source-planet embeddings,
mean of target-planet embeddings.

## Heads in detail

**Value head V(s)**: `[CLS] (64) ⊕ globals (16)` → MLP 80 → 128 → 128 → 1
with tanh.

**Policy head π(a|s)**: per-candidate scoring.
- Input per candidate: `source_pool (64) ⊕ target_pool (64) ⊕ plan_features (12) ⊕ [CLS] (64) ⊕ globals (16)` = 220.
- MLP 220 → 128 → 128 → 1 → logit ℓ_k.
- Softmax over candidates.

## Action space

Each turn: pick one of K apollo candidates. **Always include the empty
plan** (no launches) so the policy can learn patience.

Extended candidate pool (cheap apollo variations):

- Base 4: `SelectionStrategy ∈ {PriorityFirst, ScoreFirst, ScorePerShip, ProductionFirst}`.
- Perturb `OFFSET_LOOKAHEAD ∈ {0, 5, 10}`.
- Perturb `A_S_LOOKAHEAD ∈ {0, 3, 5}`.

Dedup identical move sets. Practical K ≈ 8–24 per turn.

## Rewards

Terminal-only: +1 win, −1 loss, 0 draw. γ=0.998, GAE λ=0.95. No dense
shaping initially. Add `0.001 × Δ(my_production − enemy_production)`
later if learning is too slow.

## Training loop

1. **Bootstrap V**: ~10k apollo-vs-apollo games, supervised on terminal outcome.
2. **Warm-start π**: behavior-clone apollo's current rollout pick.
3. **PPO self-play**:
   - 50% vs current policy.
   - 30% vs sliding window of past checkpoints (last 10).
   - 20% vs frozen vanilla apollo.
4. **PPO**: GAE advantages, clip ε=0.2, value clip, entropy bonus 0.005.
5. **Curriculum**: 2p until >60% vs vanilla apollo, then add 4p.

## Compute

Inference: 1 encoder pass (~0.5 ms CPU) + K policy MLPs (each <50 μs) + 1
V MLP. Total <2 ms. Apollo's per-turn budget is ~100 ms in rollout mode —
plenty of headroom.

Training: 1M–10M episodes for convergence. Episode generation is the
bottleneck (~80 s/game), so CPU worker pool feeding a single GPU trainer.

## Deployment

Train in PyTorch, infer in Rust via burn or candle inside apollo's
existing PyO3 binary. No new Python deps in the Kaggle agent path.

## Decisions (called now)

- Algorithm: PPO actor–critic.
- Heads: V (scalar) + π (per-candidate softmax), shared encoder, joint training.
- Encoder: 2-layer self-attention, d=64, 4 heads.
- No explicit fleet tokens.
- Action = pick one of K apollo candidates including the empty plan.
- Reward: terminal ±1, γ=0.998, λ=0.95.
- Self-play with frozen + sliding opponent pool.
- Bootstrap V supervised → BC π → PPO.
- Deploy via Rust-native inference.

## Proof of concept (where we start)

The full design above is the target. The first thing we actually build is
much smaller, just to validate the training pipeline:

- Python-only, no apollo dependency.
- Action space: per owned planet, multinomial over targets ∪ {no-op};
  ship count fixed at `target.ships + 1` (capped at all-but-1).
- Tiny encoder: shared MLP per planet, no attention.
- Train PPO against the random and nearest-sniper baselines.
- Log to wandb.

This is just to prove the training loop works end-to-end. See
`bots/mine/rl_poc/` for the implementation.
