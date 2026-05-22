# Top-8 Open-Source Bot Analysis

Analysis of the 8 highest-rated open-source Orbit Wars bots (deep read of every line of each `main.py`). Goal: understand why they win, group them into families, and catalog reusable helpers for a new bot.

## Elo recap

| Bot | 2P elo | 4P elo | Lines | Family |
|---|---|---|---|---|
| marco-dg | **859 (1st)** | 607 (7th) | 2502 | A |
| obnext | 841 (2nd) | 619 (6th) | 3198 | A |
| owproto | 839 (3rd) | 649 (3rd) | 807 | B |
| ppo | 834 (4th) | 651 (2nd) | 3230 | A |
| sim-search | 833 (5th) | 632 (4th) | 979 | B |
| heuristic | 808 (6th) | **683 (1st)** | 572 | C |
| structured-v4 | 790 (7th) | 605 (8th) | 3475 | A |
| tamrazov-starwars | 787 (8th) | 623 (5th) | 3306 | A |

---

## Family tree

### Family A — "WorldModel / pilkwang Structured Baseline" lineage (5 bots)
**Members:** marco-dg, obnext, ppo, structured-v4, tamrazov-starwars.

These are **near-identical at the foundation**. All five share the same ~55-function layout, the same `WorldModel` class, the same `ShotOption`/`Mission` dataclasses, the same `Planet`/`Fleet` namedtuples, and — critically — **identical constant names and values** (`ATTACK_COST_TURN_WEIGHT=0.55`, `STATIC_NEUTRAL_VALUE_MULT=1.4`, `SUN_SAFETY=1.5`, `ROTATION_LIMIT=50`, `MULTI_SOURCE_TOP_K=5`, etc.). The ppo `agent.yaml` states outright it is "built on **pilkwang Structured Baseline v11**"; structured-v4 calls itself "V4 Heuristic Planner." This is one codebase that several authors forked and tuned.

**Shared architecture (the baseline):**
- Parse obs → `WorldModel` precomputes an **arrival ledger** (`build_arrival_ledger` / `fleet_target_planet` ray-casts every in-flight fleet to its first-hit planet + ETA).
- **Per-planet forward simulation** (`simulate_planet_timeline`) over a ~110–180 turn horizon, producing `keep_needed`, `min_owned`, `first_enemy`, `fall_turn`, `holds_full`.
- Generate candidate **missions** (capture / snipe / swarm / reinforce / rescue / recapture / crash_exploit), score each as `value / (ships + turns·cost_weight + 1)`, sort, and **greedily commit** while updating `planned_commitments` so later missions see prior arrivals.
- Moving-target intercept via fixed-point iteration (`aim_with_prediction` + `search_safe_intercept`); sun-avoidance via `segment_hits_sun`.
- Exact combat via `resolve_arrival_event` (top-two incoming groups cancel, survivor vs garrison, ties → neutral) — a faithful copy of the engine's same-turn combat rule.

**What each fork adds over the baseline (the high-value diffs):**

| Bot | Distinctive addition |
|---|---|
| **obnext** | Closest to raw baseline. Longest horizon (`HORIZON=180`). Internal "v37" comments. The reference implementation of the family. |
| **marco-dg** | **Beam-search opening** (`_plan_beam_search`, depth 2–5, width 8) that plans a sequenced land-grab chain in the first 50 turns. **2P-only — disabled in 4P.** This is its 2P edge. |
| **ppo** | "Hyperion-inspired" mods on pilkwang v11: comet fallback detection, expanded elimination pressure (`ELIMINATION_BONUS=25`), quick reinforcements, domination consolidation ("total war"), late-aggression fleet ratio. *(Note: name is a label only — no neural net, no learned weights.)* |
| **tamrazov-starwars** | Adds `build_gang_up_missions` + `build_elimination_missions`, exposed-planet detection, much higher aggression (`HOSTILE_TARGET_VALUE_MULT=2.05`, `ELIMINATION_BONUS=55`), more defensive proactive ratios. Carries hand-tuning comments (`# was 1.95`). |
| **structured-v4** | Adds **FFA anti-kingmaking** (`ffa_owner_pressure_multiplier`: attack leader ×1.24, discount mid-rank wars ×0.82), total-war endgame, quartet (4-source) swarms, periodic harassment. Most code (3475 lines) yet **worst in both modes within the family** — evidence of over-tuning. |

### Family B — "Proto" lineage (2 bots)
**Members:** owproto, sim-search.

Shared signature: a closed-form `get_custom_score` target formula driven by `FORMULA_*` constants (`FORMULA_DIST=100`, `FORMULA_PROD_MULT=15`, `FORMULA_ENEMY_BONUS=10`, `FORMULA_*SHIPS*=0.7`), a `collides`/`collides_segment` segment-circle primitive, `get_planet_trajectories` (60-tick orbital propagation), moving-target intercept (`find_angle_to_moving_planet`/`find_intercept`), `MIN_SHIPS_*` + `COOP_PLANET_CAP=8` cooperative attacks, `calculate_req_ships[_moving]` fixed-point ship sizing, and persistent **module-global trajectory bookkeeping** (`fleet_trajectories`, `reinforcement_trajectories`, `moving_planets`). sim-search's comments literally say "from Proto, unchanged."

- **owproto** is the original "OW-Proto" (single author, peaked LB ~1080). Pure heuristic + precise reactive defense.
- **sim-search** = owproto + a **forward-sim layer** (`simulate_outcome`, 20-tick, top-K=6 candidates) + an **optional learned gradient-boosted value function** (`GBC_TREES`, hand-rolled tree-walk inference loaded from an external dump; falls back to a ship-lead heuristic if absent). Cites djenkivanov's ow-proto + aidensong123's "search + learned value function" notebooks.

### Family C — standalone (1 bot)
**Member:** heuristic ("Enders Fleet").

No shared lineage with A or B — completely different function set (`predict_orbit`, `solve_intercept`, `safe_angle`, `multi_leg_path`, `is_decoy_fleet`, `estimate_capture_bonus`) and a **phase state-machine** core (`smash/rush/expand/counter_attack/crush/aggressive/defend/dominate/grow`). Greedy weighted scoring (`prod*18 - tt*2.5`), no forward sim, lightest compute. Notably the **only top bot that is fully player-count-agnostic** — and it's **#1 in 4P**.

---

## Why they win — and the 2P/4P inversion

**The central finding:** the inversion is primarily *economic/strategic*, not a simulation artifact. Combat changes sign between modes.

- **2P is strictly zero-sum:** every ship you cost your single opponent is pure relative gain. Unchecked aggression is correct.
- **4P combat is negative-sum for the participants and positive-sum for bystanders:** ships are spent in clashes (send 5 at a 4-ship planet → net +1, not +5; encoded in `resolve_arrival_event`). So when you and another player fight, the *two uninvolved players* gain relative position for free. Aggression that's optimal in 2P actively hands the game to bystanders in 4P.

This explains the standings better than "simulation noise" does:
- **heuristic** — mild, expansion-first, no special aggression, fully player-count-agnostic — is **#1 in 4P**. It simply doesn't pay for negative-sum fights.
- **tamrazov** (`HOSTILE_TARGET_VALUE_MULT=2.05`, `ELIMINATION_BONUS=55`) and **structured-v4** (total-war endgame, most code, most 4P special-casing) **sink to the bottom of 4P** — they spend ships fighting while others coast.
- **marco-dg** is **#1 in 2P, #7 in 4P**: its beam-search opening is *explicitly disabled in 4P*, but more fundamentally its whole value system prices combat as if it were zero-sum.

*(Secondary effect: with three opponents the arrival-ledger/timeline forecasts are also noisier, modestly eroding the simulation edge — but the sign-flip of combat is the dominant cause.)*

**Implication for our bot:** the Family A simulation/intercept primitives are the strongest 2P engine to copy, but the *decision layer* must differ by mode. In 4P, avoid negative-sum fights unless they leave you the relative winner, and reason about **specific opponents** (who's leading, who's weak, who's about to fight whom) rather than collapsing everyone into "enemy." Notably, structured-v4's `ffa_owner_pressure_multiplier` (attack the leader, discount mid-rank wars) is the *only* per-opponent reasoning in the entire field, and it's crude — this is the clearest unfilled gap.

---

## Reusable helper catalog ("what to steal")

Ranked by generalizability. Several appear identically across Family A — those are the most battle-tested.

### Tier 1 — lift directly (HIGH, appears in many bots)
- **`fleet_speed(ships)`** — the log speed curve `1 + (MAX_SPEED-1)·(log(ships)/log(1000))^1.5`. Exact engine rule; every bot reimplements it. *(all 8)*
- **`resolve_arrival_event(owner, garrison, arrivals)`** — faithful same-turn multi-attacker combat resolution (top-two cancel, survivor vs garrison, tie→neutral). Encodes the core combat rule for any number of players. *(Family A)*
- **`simulate_planet_timeline(planet, arrivals, player, horizon)`** — single-planet forward ownership/garrison sim with binary-searched `keep_needed`/`fall_turn`. The simulation backbone. *(Family A)*
- **`fleet_target_planet(fleet, planets)`** — ray-vs-circle prediction of which planet an in-flight fleet hits and when. Threat attribution. *(Family A)*
- **`segment_hits_sun` / `point_to_segment_distance` / `collides`** — point-to-segment distance + sun keep-out test. Pure geometry, needed for every launch. *(all)*
- **`predict_planet_position` / `get_planet_trajectories` / `predict_orbit`** — orbital propagation of a rotating planet. *(all)*
- **`aim_with_prediction` + `search_safe_intercept`** (Family A) / **`find_intercept`** (Family B) / **`solve_intercept`** (heuristic) — moving-target intercept solver. Three independent implementations of the same idea; Family A's fixed-point + scan-fallback is the most complete.

### Tier 2 — strong, lightly game-specific (MED–HIGH)
- **`estimate_arrival` / `travel_time`** — boundary-aware ETA (angle + integer turns) used uniformly for routing, ranking, and reserves. *(Family A)*
- **`indirect_features(planet, planets, player)`** — production-weighted neighborhood wealth (`prod/(d+12)`), split friendly/neutral/enemy. Good positional value signal. *(Family A)*
- **`calculate_req_ships[_moving]`** — fixed-point solve of ships needed accounting for enemy production growth during transit. *(Family B)*
- **`get_under_attack` / `planet_under_threat`** — threat map by tracing enemy fleets forward against owned planets. *(Family B, heuristic)*
- **`detect_enemy_crashes` / `detect_enemy_planet_battles`** — find inter-enemy collisions to exploit (the key 4P "crash-exploit" tactic). *(Family A)*
- **`ships_needed_for_takeover`** — garrison + production-growth capture cost with margin. *(heuristic)*

### Tier 3 — patterns worth borrowing (not drop-in)
- **Greedy mission scoring** `value / (ships + turns·cost_weight + 1)` with `planned_commitments` bookkeeping to avoid double-spending. *(Family A — the heart of the planner)*
- **Phase / posture state machine** (`build_modes` behind/ahead/finishing in A; the 9-phase ladder in heuristic) for adapting aggression to game state.
- **Soft time-budget discipline** — `SOFT_ACT_DEADLINE≈0.82·actTimeout`, `expired()` checks, phase gates (`allow_heavy_phase`), and `HEAVY_ROUTE_PLANET_LIMIT=32` to bail gracefully. Essential given all heavy bots risk timeout. *(Family A)*
- **`_value_state_features` + `_value_score`** — hand-rolled GBDT inference over an exported tree dump; a template for adding a learned value head without ML deps at runtime. *(sim-search)*
- **`is_decoy_fleet`** — filter small enemy fleets out of threat assessment. *(heuristic)*
- **Cooperative multi-source attacks** synchronized by ETA (`COOP_PLANET_CAP` in B; 2/3/4-source swarms in A). *(both)*

---

## Suggested next steps
1. Decide our base: Family A's simulation core (best 2P primitives) vs. a leaner Family C-style decision layer (best 4P robustness). A hybrid — A's primitives + a simpler 4P policy — is the gap none of the top 8 fill well.
2. Extract Tier-1 helpers into a shared module and validate against `rust_engine`/`parity` for engine fidelity.
3. Investigate why 4P-specific tuning hurts (structured-v4, marco-dg) before adding our own 4P special-casing.
