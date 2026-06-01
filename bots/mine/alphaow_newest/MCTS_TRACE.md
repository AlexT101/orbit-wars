# `alphaow_newest` — one MCTS iteration, end to end

A single-iteration trace of `alphaow_newest`, with every algorithm/policy used,
where it lives in source, and how much time each phase typically takes.

## Configuration (set by `alphaow_newest/main.py` env)

| key | value | effect |
|---|---|---|
| `ALPHAOW_VALUE_NET_PATH` | `weights/xgb_top10_d6.json` | XGBoost (600 trees, depth 6) — auto-detected as JSON, parsed by `src/xgb.rs` |
| `OW_ROLLOUT` | `none` | DUCT's leaf-eval rollout loop runs **zero iterations** — leaves are evaluated by the value net directly |
| `OW_ROLLOUT_DEPTH` | `0` | (redundant with the above; both gate the same loop) |
| `OW_K_ROOT` (default) | `5` | up to 5 *my* candidate plans at root |
| `OW_K_NON_ROOT` (default) | `4` | up to 4 *my* candidate plans at every non-root node |
| `OW_PUCT_C` (default) | `0.3` | exploration constant in PUCT (`EXPLORATION` const in `duct.rs:23`) |
| budget | `ALPHAOW_BUDGET_MS` (default 500ms) | wall-clock cap on the iteration loop |

Median empirical iterations per turn at 200ms: **289**.
Per-iteration budget: **~692 µs**.

---

## The top-level loop

`bots/mine/alphaow/src/duct.rs:602-610`:

```rust
let deadline = Instant::now() + Duration::from_millis(budget_ms);
let mut iters = 0u32;
while Instant::now() < deadline {
    select_and_expand(&mut root, me, &mut rng, true);
    iters += 1;
    if iters > 100_000 { break; }
}
```

Each call to `select_and_expand` is **one MCTS iteration**: descend the tree,
expand one new leaf, evaluate it with XGB, propagate the value back to root.

The function recurses; here are the five phases it goes through.

---

## Phase 1 — Selection (PUCT walk down)

**Policy:** *Decoupled* PUCT. Each player independently picks their own action
by PUCT over their **marginal stats** (visits and value summed across what the
opponent might play). This is what makes alphaow's MCTS correct for
simultaneous-move games — *not* a sequential expectimax over alternating turns.

**Where:** `duct.rs::select_my` / `select_opp` (around line 200-260),
formula at `duct.rs:209`:

```text
PUCT(action_i) = mean_value_i + EXPLORATION * prior_i * sqrt(parent_visits) / (1 + visits_i)
```

where:
- `mean_value_i = sum_value_i / visits_i` (or 0 if unvisited)
- `prior_i` comes from `rank_prior(rank, total)` — geometrically-decaying prior
  by rank: action `i` gets `0.5^i / sum_k 0.5^k`. So rank-0 starts with ~50%
  of the prior mass, rank-1 ~25%, etc.
- `EXPLORATION = 0.3` (env-tunable via `OW_PUCT_C`)
- `prior * sqrt(N) / (1 + n_i)` is the exploration bonus
- mean_value is from MY perspective for `select_my`, and reflected for opp.

Both sides pick simultaneously; their joint choice is the key into
`Node::children: HashMap<(my_idx, opp_idx), Box<Node>>`.

**What happens:** recurse into that joint child if it exists; otherwise this
is the expansion site.

**Cost:** O(`my_K + opp_K`) PUCT scans per level × tree depth ~48 levels
(median, see profile). Each PUCT compute is ~4 floating-point ops; each
hash lookup is ~30 ns. Total selection cost per iteration: **~10-30 µs**.

---

## Phase 2 — Candidate generation (apollo) — *the expensive phase*

When `select_and_expand` reaches a node whose `candidates_initialized = false`,
it lazily populates the candidate lists.

**Policy:** [apollo's hellburner planner](../alphaow/src/apollo/strategy.rs)
— a greedy iterative planner over fleet orders.

**Where:** `apollo::hellburner::search_candidates(world)` — also called from
each player's perspective for `my_candidates` and `opp_candidates`.

`search_candidates` at `strategy.rs:1778`:

```rust
pub fn search_candidates(world: &WorldState) -> Vec<Vec<FleetOrder>> {
    if world.enemy_planets.is_empty() {
        return vec![Vec::new()];                  // no work to do
    }
    let model = HellburnerModel::build(world);    // ~50 µs
    if world.step < EARLY_ROUNDS {
        return vec![run_early_game(world, &model)];
    }
    let mut out = Vec::new();
    for &strat in &STRATEGIES {                   // 4 strategies
        let (moves, _) = run_strategy(world, &model, strat);
        if !out.iter().any(|prev| prev == &moves) {
            out.push(moves);                      // dedup
        }
    }
    out
}
```

`run_strategy` (the inner greedy planner, `strategy.rs:1594`):

```rust
fn run_strategy(world, model, strategy) -> (Vec<FleetOrder>, PlanState) {
    let candidate_ids = world.non-comet planets with inbound edges;
    let mut cache = HashMap::new();   // per-target (score, FrontlineWin)
    let mut dirty = candidate_ids;    // first iter recomputes everything
    for _ in 0..256 {                 // bounded by source-pool size
        // recompute dirty targets:
        for target in dirty {
            let result = evaluate_target(world, model, &plan, target, ...);
            cache.insert(target, result);
        }
        // pick the target maximising strategy.key(score, prod, ships):
        let best = cache.iter().max_by_key(|(_, (score,_,_))| strategy.key(...));
        // commit fleet orders; update dirty set for next iteration
        plan.commit(best); moves.push(best); dirty = outbound(best);
    }
}
```

The 4 `SelectionStrategy` variants differ only in the **sort key** used at
the "pick best target" step — `score`, `score`, `score / ships`, `production`
respectively. All 4 use the same `evaluate_target` (timeline simulation +
defender-fleet projection).

**Per-target evaluation (`evaluate_target`, `strategy.rs:1077`):**

1. Run `simulate_planet_timeline` (forward-sim of in-flight fleets and combat
   resolution at the target planet over the horizon, ~5-10 ticks ahead) —
   determines if/when this target falls and to whom.
2. If already won by baseline → skip.
3. Otherwise: sweep launch *offsets* (0, 1, …, 5) — for each offset, project
   what happens if we delay launch by that many ticks. Pick the offset that
   maximises `score_capture` = `production × remaining_horizon × zero_sum_mult`,
   where `zero_sum_mult = 2.0` if enemy-owned, `1.0` if neutral. The 2× weight
   for enemy targets is the **implicit enemy/neutral differentiation** in the
   scoring (`strategy.rs:452`).
4. Returns `(score, FrontlineWin{fleet_orders})`.

**Cost:** dominant phase of an MCTS iteration. For a typical mid-game world
(~30 planets, ~10-15 sources, ~15-20 candidate targets):
- `HellburnerModel::build`: **~50 µs**
- 4 × `run_strategy`:
  - First greedy iteration: ~15 targets × `evaluate_target` ~30 µs each = **~450 µs**
  - Subsequent iterations: only "dirty" targets recompute (typically 2-5) ≈ **~75 µs/iter**
  - ~10-20 greedy iterations until convergence → **~600-1500 µs total per strategy**
- 4 strategies → **~2-6 ms in worst case**, but most strategies share cached
  results (per-target `(score, FrontlineWin)`) via the `dirty` set so the
  amortised cost is closer to **~1-2 ms per `search_candidates` call**.

Apollo is also called from the **opponent's perspective** for `opp_candidates`
— same cost again. So expansion of a brand-new node is the dominant per-iter
cost (~2-4 ms on the *first* visit to each node).

After this lazy initialisation, the resulting `my_candidates` /
`opp_candidates` are **cached on the Node** and reused for all subsequent
visits — the apollo cost is amortised over the full tree life.

---

## Phase 3 — Engine step (apply the joint action)

**Policy:** Orbit Wars' engine, ticked one step under the chosen joint action.

**Where:** `crate::sim::{apply_launches, tick}` (called from
`select_and_expand` after candidate generation).

Steps:
1. `apply_launches(state, my_candidates[my_idx])` — push my new fleet orders
2. `apply_launches(state, opp_candidates[opp_idx])` — push opp's
3. `tick(state)` — move all in-flight fleets one step, resolve any combats
   that arrive this tick, advance planet production / orbiting positions

**Cost:** **~100-300 µs** depending on number of in-flight fleets and
combat resolutions this tick.

---

## Phase 4 — Leaf evaluation (XGB)

When `select_and_expand` reaches the bottom of the descent (either an
unexpanded child it just created, or a terminal state), it needs a value
estimate.

**Without rollout** (alphaow_newest's setting), this is just one call to
`value_net::predict(state, me)`.

**Where:** `bots/mine/alphaow/src/value_net.rs:465`, which dispatches on the
loaded model. With XGB JSON loaded:

```rust
Model::Xgb { model, kind: InputKind::SummaryV2 } => {
    let feats = summary_features_v2::extract(state, me);   // 46 floats
    model.predict_value(&feats)                            // tanh(z/2) in [-1, 1]
}
```

1. **Feature extraction** (`summary_features_v2::extract`,
   `value_net.rs:699`): build the 46-d feature vector — 10 me-current,
   10 opp-current, 9 me-extrapolated, 9 opp-extrapolated, 8 neutral block.
   The extrapolation step runs `extrapolate_fleets` to predict where every
   in-flight fleet will land. **~30-80 µs.**

2. **XGB inference** (`xgb.rs::predict_value`): walk all 600 trees of the
   gbtree. For each tree, follow internal nodes (compare
   `x[feat_idx] < threshold`, branch left/right) until a leaf. Sum the leaf
   logits, add `base_score_logit` (0 here), apply `tanh(z/2)` to map into
   `[-1, 1]`.

   - 600 trees × ~7 nodes deep × 1 compare/load each ≈ 4200 ops.
   - Branch-light: leaf nodes encode the logit in their `left_or_value` slot
     with `feat_idx = u32::MAX` as the leaf marker.
   - **~80 µs** end-to-end.

**Total leaf eval cost: ~110-160 µs.**

(For reference: with rollouts on, this phase is replaced by an 8-tick
*rollout* that calls apollo twice per tick — 16 apollo calls = ~30-50 ms
per leaf. That's the 80% budget eater we turned off.)

---

## Phase 5 — Backprop

Walk back up the recursion, updating two things at every node we visited:

1. **Joint child stats:** `child.visits += 1`, `child.sum_value += v`
   (where `v` is the leaf value, sign-flipped if we're on opp's side).
2. **Marginal stats:** `Node::my_stats[my_idx]` and `Node::opp_stats[opp_idx]`
   — these summary tables are what PUCT reads in Phase 1. Both get
   `visits += 1` and `sum_value += v` for the indices that were selected
   at that node.

This decoupled-marginal update is the difference from joint MCTS — the
selection uses marginal stats, not the joint cell. With my_K=5 and
opp_K=4 at a node, the joint cell only gets 1/20 of the per-node visit
mass on each visit; but each *marginal slot* gets 1/5 or 1/4. So marginal
stats grow much faster than joint stats — hence PUCT's per-side
exploration converges fast even though the joint subtree is sparse.

**Where:** `duct.rs::backprop` (called as a fold up the recursion).

**Cost:** O(depth). Per node: 2-3 floating-point updates + 1 hash lookup.
At depth ~48: **~5-10 µs** total.

---

## Per-iteration time budget — summary

| phase | "cold" (creates new node) | "warm" (revisits existing) |
|---|---|---|
| Selection (PUCT walk) | 10-30 µs | 10-30 µs |
| Candidate gen (apollo) | **1-4 ms** *(first time)* | 0 — cached on Node |
| Engine step | 100-300 µs | 100-300 µs |
| Leaf eval (XGB) | 110-160 µs | (not done — selection continues) |
| Backprop | 5-10 µs | 5-10 µs |
| **iter total** | **1.5 - 4.5 ms** | **0.13 - 0.34 ms** |

Empirically with subtree reuse (so most nodes are reused across turns),
the average iteration is closer to the "warm" column. Profile data: 289
iter / 200ms budget = ~692 µs/iter ⇒ **about 25% of iterations expand a
new node**, the rest reuse existing structure.

---

## Per-turn aggregate (median, 200ms budget)

From `OW_DEBUG=1` probe (alphaow_newest vs itself, seed 12345):

| metric | median |
|---|---|
| iterations / turn | **289** |
| root visits | 1,495 (accumulates from subtree reuse) |
| unique nodes | 829 |
| **max tree depth** | **48** plies |
| `my_K` at root (apollo's plan count after dedup) | 1 |
| fraction of turns where `my_K = 1` | **85%** |

**Important consequence:** on 85% of turns, root has only one *my* candidate
plan, so the bot's move is byte-identical to what `apollo::plan(world)` alone
would have produced — the XGB+MCTS contributes nothing to the move on those
turns (it still does work, on the opp axis, but that work doesn't change the
move). The remaining 15-25% of turns are where the value-net+MCTS actually
differentiate the bot from raw apollo — and that 15-25% appears to carry the
bulk of the in-play strength gap vs the production MLP build.

---

## What each phase's "policy" is, named

| phase | policy / algorithm |
|---|---|
| Selection | Decoupled PUCT with rank-prior `0.5^i / Σ 0.5^k` |
| Candidate generation | Apollo hellburner — 4 greedy-planner strategies (PriorityFirst, ScoreFirst, ScorePerShip, ProductionFirst), each running greedy iterations over targets sorted by their per-strategy key, with `evaluate_target` doing a per-target timeline simulation + offset sweep |
| Engine step | Orbit Wars rules (gravitational fleets, combat, comet handling) |
| Leaf evaluation | XGBoost binary-logistic gbtree (600 trees, depth-6) over the 46-d summary_v2 features; output `tanh(z/2) ∈ [-1, 1]` |
| Backprop | Decoupled-marginal: joint child + per-side marginal stats updated each iteration |

---

## Source map

| concept | file:line |
|---|---|
| iteration loop | `src/duct.rs:602` |
| PUCT formula constants | `src/duct.rs:23` (`EXPLORATION = 0.3`) |
| Node struct | `src/duct.rs:88` |
| `search_candidates` entrypoint | `src/apollo/strategy.rs:1778` |
| `run_strategy` (greedy planner) | `src/apollo/strategy.rs:1594` |
| `evaluate_target` (timeline sim per target) | `src/apollo/strategy.rs:1077` |
| `SelectionStrategy::key` | `src/apollo/strategy.rs:1058` |
| `zero_sum_mult` (enemy 2x weight) | `src/apollo/strategy.rs:452` |
| `predict()` (value-net dispatch) | `src/value_net.rs:465` |
| XGB inference | `src/xgb.rs::predict_value` |
| `summary_features_v2::extract` | `src/value_net.rs:699` |
| OW_PLANNER switch (duct vs mcts) | `src/main.rs:74` |
