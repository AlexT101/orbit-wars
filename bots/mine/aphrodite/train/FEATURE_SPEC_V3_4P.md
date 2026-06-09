# summary_v3 — 4p value-net feature spec

A 4p-only redesign of the value-net input. **2p is untouched** (stays on the
65-dim `summary_v2`; its features are already correct because there is exactly
one opponent). All the problems this fixes are *pooling-over-multiple-opponents*
artifacts that only exist in FFA:

- `opp_cur`/`opp_ext` described only the **dominant** enemy → harming a weaker
  enemy was invisible, and `dominant_enemy` could **flip identity** between a
  parent and child state (noise in exactly the MCTS move-ranking signal).
- "enemy support" pooled all opponents → **rival opponents counted as each
  other's defenders**, so enemy planets read as over-defended and the eval
  **undervalued attacking opponents** (`avg_enemy_support`, `num_enemy_vulnerable`,
  `enemy_prod_at_opportunity`).
- pooled-enemy `centroid_dist` and `enemy_economic_dispersion` collapse to
  near-constants in 4p (3 opponents surround you → their pooled centroid ≈ map
  center).

## Core idea: canonical orbital ordering (flip-free)

Players sit at fixed 90° rotational positions (fourfold symmetry). Empirically
the seat order by increasing `atan2(y-50, x-50)` is **always the cycle
`C = [0, 1, 3, 2]`** (only the global phase rotates between games). Orbital
motion advances in the **+angle** direction (measured: angle increases over
time, `angular_velocity ≈ 0.044 rad/step`). So from player `me`:

```
o1 = next(me) in C   # orbit-DOWNSTREAM adjacent (you rotate toward them)
o2 = next(next(me))  # OPPOSITE
o3 = prev(me) in C   # orbit-UPSTREAM adjacent
```

`next`: 0→1→3→2→0. e.g. me=0 → (o1,o2,o3)=(1,3,2); me=1 → (3,2,0); me=3 → (2,0,1);
me=2 → (0,1,3). This is **pure seat-id arithmetic** — no per-state geometry, no
`initial_planets`, perfectly stable across parent/child (ids never change).

The two adjacents are **not interchangeable**: o1 is always the downstream
neighbour, o3 always upstream. Because orbit has a direction, the strategic
relationship differs, and fixing each to a distinct slot makes that asymmetry
learnable (don't ever swap o1/o3).

**Dead/eliminated opponent:** zero its *entire* block (economy, relational, and
the pairwise-matrix entries) **including ratios/shares — NOT a 1.0 "dominate"
default** — and set `is_alive_k = 0`. The encoding "all-zeros + flag off" is what
the model learns to treat as an empty slot; a 1.0 default would risk phantom
aggression toward a vacated area. Never compact remaining opponents up (that
reintroduces flips). The dead team's old planets are represented in *my* / neutral
/ other-opponent blocks, so the area is still accounted for.

**No discrete rank / no is_leader.** A discrete rank or leader-bit reintroduces
the same flip instability (two near-equal teams swap rank → feature jumps). The
leader priority is carried by **continuous** per-opponent strength scale
(`ship_share_me_vs_k`, `prod_share_me_vs_k`) and the continuous
`leader_strength_ratio`. `is_alive` is the only discrete per-opponent flag, and
alive/dead is a genuine binary (no "close call").

## Column layout (DIM = 145)

Most magnitude features are **share/fraction-normalized** for phase-stationarity
(see *Normalization* below); a few absolute scale anchors are retained.

| idx | block | feature |
|--|--|--|
| 0 | global | step |
| 1 | global | angular_velocity *(per-game constant: orbiter conveyor speed; gates orbiter features; doesn't discriminate sibling MCTS leaves but calibrates across positions)* |
| 2–10 | me_cur (9) | ships_on_planets, ships_flying, n_static, n_orbit, n_comet, prod_static, prod_orbit, n_neutrals_closer, n_enemies_closer *(share-normalized; see below)* |
| 11–18 | me_ext (8) | ships_on_planets, n_static, n_orbit, n_comet, prod_static, prod_orbit, n_neutrals_closer, n_enemies_closer *(extrapolated/post-fleet-resolution; share-normalized)* |
| 19–25 | neutral (7) | unchanged `neutral_block` |
| 26–40 | aggregate (15) | ship_share_me_vs_all, production_share_me_vs_all, num_my_vulnerable, my_prod_at_risk, max_enemy_pressure_on_me, pw_enemy_pressure_on_me, my_fleet_fraction, ally_economic_dispersion, avg_ally_ships, leader_strength_ratio *(my/max_opp, continuous)*, opponent_strength_spread *(continuous)*, n_alive, **total_board_production**, **total_board_ships**, **my_absolute_production** *(last 3 = absolute scale anchors)* |
| 41–64 | **opp o1 (24)** | per-opponent block (below) |
| 65–88 | **opp o2 (24)** | per-opponent block |
| 89–112 | **opp o3 (24)** | per-opponent block |
| 113–128 | pairwise **in-flight** matrix (16) | committed attacks; see below |
| 129–144 | pairwise **vulnerability** matrix (16) | latent opportunity; see below |

Threats-on-me (`num_my_vulnerable`, `my_prod_at_risk`, `max/pw_enemy_pressure_on_me`)
stay **aggregated over all enemies** — that is genuinely a pooled quantity and a
useful low-variance fallback. The per-opponent threat decomposition lives in the
opp blocks and the two pairwise matrices.

### Per-opponent block (24, ×3) — ordered o1, o2, o3

Vulnerability moved out to the pairwise matrix (below). What remains is k's
own state + directional pressure + scale.

cur economy (9): ships_on_planets, ships_flying, n_static, n_orbit, n_comet,
prod_static, prod_orbit, n_neutrals_closer_to_k, n_enemies_closer_to_k
*(share-normalized)*

ext economy (8): ships_on_planets, n_static, n_orbit, n_comet, prod_static,
prod_orbit, n_neutrals_closer, n_enemies_closer *(extrapolated → shows k's
post-arrival weakened state after my attack lands; share-normalized)*

relational + scale + alive (7):
1. `pw_my_pressure_on_k` — prod-weighted threat I project on k *(directional force;
   complements vuln[me→k], which is the realized opportunity given k's defense)*
2. `pw_k_pressure_on_me` — prod-weighted threat k projects on me *(directional)*
3. `centroid_dist_me_to_k` — my centroid ↔ k's centroid (true front-line
   proximity, per neighbour; fixes the pooled-centroid collapse)
4. `k_economic_dispersion` — dispersion of k's economy (fixes pooled-dispersion)
5. `ship_share_me_vs_k` — my_ships / (my_ships + k_ships) *(continuous scale)*
6. `prod_share_me_vs_k` — my_prod / (my_prod + k_prod) *(continuous scale)*
7. `is_alive_k`

Positional "closer" counts (cur idx 7–8, ext idx 6–7) are lower-value ablation
candidates; kept for parity with the self block.

### Two pairwise matrices (16 each) — committed vs latent

Both share the **same owner-bucketed per-planet pressure pass** (see
Implementation), reorder to canonical slots `[me, o1, o2, o3]`, and zero
dead-team rows *and* columns. Same 16-entry layout: 12 directed off-diagonal
`src→dst` among the four teams, then 4 `team→neutral`.

**In-flight matrix (committed attacks) — idx 113–128.** Raw ship mass in flight
from `src` toward a planet owned by `dst` (from `arrivals`: fleet → predicted
target planet → its owner). Normalized to **fraction of total in-flight ships**.

```
113 me→o1   114 me→o2   115 me→o3
116 o1→me   117 o1→o2   118 o1→o3
119 o2→me   120 o2→o1   121 o2→o3
122 o3→me   123 o3→o1   124 o3→o2
125 me→neut 126 o1→neut 127 o2→neut 128 o3→neut
```

**Vulnerability matrix (latent opportunity) — idx 129–144.**
`vuln[i→j] = (j's production on planets vulnerable to i) / (j's total production)`
— a phase-invariant fraction in [0,1] ("what % of j's economy is exposed to i").
A planet of j is vulnerable to i iff `pressure_from_owner[i] > j's own support +
garrison` — i.e. **defender support = j's OWN planets only** (this is the fix for
the mutual-defender bug; opponents no longer "defend" each other). For
`→neutral`, the defender support is 0 (neutrals exert no pressure), so
`vuln[i→neutral]` = i's expansion *reach* onto neutrals.

```
129 me→o1   130 me→o2   131 me→o3
132 o1→me   133 o1→o2   134 o1→o3
135 o2→me   136 o2→o1   137 o2→o3
138 o3→me   139 o3→o1   140 o3→o2
141 me→neut 142 o1→neut 143 o2→neut 144 o3→neut
```

The two matrices are complementary: **in-flight = committed** aggression,
**vulnerability = latent** structural opportunity (an opening not yet taken). The
`opp→opp` entries are the key new multi-agent signal — e.g. high `vuln[o1→o2]`
*and/or* `inflight[o1→o2]` ⇒ o2 is/will be under pressure from o1 ⇒ o2
distracted/weakening ⇒ opportunity for me. Possible extension: weight in-flight by
targeted-planet production.

## Normalization: normalize the distribution, anchor the scale

Ships and production **accrue over a game**, so a raw count means something
different at step 10 vs 400. A value net must generalize across phases, so the
*distribution* features are share/fraction-normalized (intrinsically stationary),
while a few *absolute* anchors are kept so true scale isn't lost.

**Normalized to shares/fractions:**
- per-opponent & self economy (`ships_on_planets`, `ships_flying`, `prod_*`) →
  **fraction of the board total** of that quantity (so each team's size is a
  phase-invariant share; the `me_vs_k` shares stay as pairwise scale).
- planet counts (`n_static/orbit/comet`) → fraction of total planet count.
- in-flight matrix → **fraction of total in-flight ships**.
- vulnerability matrix → **fraction of the defender's production** (already so).
- `prod_at_risk` / `at_opportunity` → fraction of the relevant team's production.

**Absolute scale anchors retained** (so the model can recover magnitude — a 60%
share of a tiny economy ≠ of a huge one): `total_board_production`,
`total_board_ships`, `my_absolute_production` (aggregate block, idx 38–40). Plus
`step` + `angular_velocity` give phase/pace context.

**Clamp every denominator** (board totals, per-team totals) to ≥ ε so early-game
(small totals) and dead/empty teams don't produce NaN/Inf or spurious 1.0s; this
dovetails with the dead-slot zeroing rule. Trees tolerate raw counts, but the
NN value heads require this normalization — so it's done at extraction, once.

## decisiveness_aux (separate array, training-only) — 9

Emitted alongside `summary_v3` in the training NPZ; the live eval ignores it.
Seat-invariant per-state: `[ship_strength[0..4], production[0..4], neutral_prod]`.
Lets `train_xgb` compute the player-count-correct decided/decisiveness signal
(top-two strength gap `(s1-s2)/(s1+s2)`, even=0 regardless of N; all-player
`claimed = Σplayer_prod/(Σplayer_prod+neutral_prod)`), fixing the 0.25-baseline
bug. (2p decisiveness already works and is unchanged.)

## Implementation notes

- **Single-pass owner-bucketed pressure (one source of truth, eval cost ≤ today).**
  Compute, for each planet `d`, `pressure_from_owner[d][0..3]` in **one** pass over
  source planets (+ inbound/arrivals bucketed by owner). Every downstream feature
  is then a cheap reduction over that array: aggregate threats-on-me, per-opponent
  `pw_*_pressure`, **both** pairwise matrices, and the all-enemy fallbacks. This
  issues the **same** `resolve_shot` calls as the aggregate code does today — in
  fact **fewer**, since the current code runs separate `pressure_on` passes over
  `mine` and `enemy`; here each `(src,d)` pressure is computed once. So v3's
  per-opponent + two-matrix richness adds **zero** aim computation.
- **L1 aim-cache threading (live-search win — do carefully).** `pressure_from`
  calls `resolve_shot(cache, src, dst, ships, off, **None**)` ([value_net.rs:419]).
  The eval already rides **L2** (`aim_cache`, turn-indexed) and **L3** (the
  cross-turn *invariant*, ship-count-independent geometry) fast paths, so the
  expensive aim geometry is cached across the ship-sweep and across sibling leaves
  at the same turn. The `None` only skips the **shared L1 hot cache** (`shot_l1`,
  per-`HellburnerModel` / per bot turn) that the policy/aimer populate. Thread that
  shared L1 into the eval's `resolve_shot` so eval and policy share one hot cache:
    - **Live search only.** Plumb the search's `shot_l1` through `extract_with_cache`
      → `relational_block`/`pressure_from` → `resolve_shot` (replace `None`).
    - **Bounded, honest benefit:** L1 is keyed by `(src,dst,ships,abs_launch)`, and
      the eval *sweeps* `ships = garrison + prod·off`, so it hits the shared L1 only
      where its swept ship counts coincide with the policy's. The dominant reuse
      remains L2/L3 (ship-independent); shared-L1 mainly trims redundant L2 traffic
      for pairs the policy already aimed this turn. Net: a real but modest live-eval
      speedup, larger now that eval does more pressure reductions.
    - **Training:** the offline `extract` builds a throwaway per-row cache with no
      policy/search L1 to share — leave it passing `None` (its own L1 + L2/L3 within
      the row suffice). The L1 threading is a live-bot change, not a training one.
- **Rust:** add `value_net::summary_features_v3` (4p) next to the existing v2;
  the live 4p bot's eval path returns v3, 2p stays on v2. `extract_v2` training
  binary gains a 4p/v3 mode (or a sibling `extract_v3`) emitting the 145-d row +
  9-d aux; `RECORD_BYTES` changes accordingly. Rebuild required.
- **Pipeline:** `build_from_zip.py` 4p path writes `summary_v3` + `decisiveness_aux`;
  `combine_npz.py` carries `decisiveness_aux`; `train_xgb.py` reads it for
  weighting/drop. 4p ablation `--zero-cols` list is rewritten for the new layout.
  2p pipeline/models untouched.
- **NN value heads (alphaduck/prometheus):** normalize the dense per-opponent and
  pairwise-matrix quantities (shares/ratios or divide by totals); XGBoost is
  threshold-based and tolerates raw counts.
- **Scope:** re-extract **4p only** (7 days). 2p models stay as-is. The cheap
  `build_aux.py` Python path is now moot (the corrected decisiveness inputs come
  for free in this 4p re-extraction).
- **Validate** with the game-level val split + `colsample`/regularization, and
  judge by **gauntlet**, not val acc. ~145 dims is fine on 2.6M 4p rows; the full
  set is intentionally ablatable via the existing `--zero-cols` workflow.
