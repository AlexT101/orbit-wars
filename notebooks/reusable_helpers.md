# Reusable, Strategy-Agnostic Helper Catalog

A toolkit of **mechanism, not policy** — physics, geometry, prediction, intercept solving, and forward simulation — that a new bot can reuse regardless of its strategy. Drawn from the three bot families (see [bot_analysis.md](bot_analysis.md)) and the [physics helper notebook](orbit-wars-physics-helper-module.ipynb).

**Scope rule:** everything here is independent of *what* to attack. Target scoring, value multipliers, posture/aggression, mission selection, and per-opponent reasoning are deliberately **excluded** — those are strategy and belong in the decision layer.

---

## Source map — where the canonical version of each layer lives

| Layer | Best source | Notes |
|---|---|---|
| Geometry / physics primitives | **physics notebook** | Cleanest, documented, spec-aligned. Identical math to Family A. |
| Orbit / comet prediction | **physics notebook** | Has the unified `predict_target_position` dispatcher. |
| Safe-path / arrival | **physics notebook** | v7 fix targets the surface entry point, not center. |
| **Verified intercept solver** | **physics notebook** | The crown jewel — adds forward-sim verification the bots lack. |
| **Combat resolution** (`resolve_arrival_event`) | **Family A** (e.g. `obnext/main.py`) | Faithful same-turn battle rule. Not in the notebook. |
| **Per-planet forward sim** (`simulate_planet_timeline`) | **Family A** | Ownership/garrison timeline. Not in the notebook. |
| **Arrival ledger** (`fleet_target_planet`, `build_arrival_ledger`) | **Family A** | Threat attribution. Not in the notebook. |
| Lightweight collision/threat | Family B (`owproto`) | Simpler alternative if you skip full timeline sim. |

**Key insight:** the physics notebook and Family A are complementary halves of the same engine. The notebook owns the **routing/aiming** half (and does it better, with verification); Family A owns the **combat/economy simulation** half. A complete strategy-agnostic core = notebook physics + Family A's sim layer. They share identical constants and the same underlying math.

> **API mismatch to reconcile (verified):** the two sources call the physics functions with *different signatures*. The notebook passes **flat coordinates** (e.g. `predict_planet_position(planet_id, cur_x, cur_y, radius, …)`, `aim_with_prediction(sx,sy,sr, target_id, tx,ty,tr, ships, …)`); Family A passes **objects** (`predict_planet_position(planet, …)`, `aim_with_prediction(src, target, ships, …)` where `src`/`target` are `Planet` namedtuples with `.x/.y/.radius/.id`). Same math, different calling convention. For a Rust port this is moot — define one struct-based API up front (the object form is cleaner) and there's no glue to write.

---

## 1. Constants (single source of truth)

From the physics notebook §2 — all match the spec and Family A:

```python
BOARD_SIZE   = 100.0
CENTER_X = CENTER_Y = 50.0      # sun / orbital center
SUN_RADIUS   = 10.0            # fleet destroyed inside this
SUN_SAFETY   = 1.5            # conservative buffer → effective keep-out 11.5
MAX_SHIP_SPEED = 6.0
ROTATION_LIMIT = 50.0         # orbital_radius + planet_radius >= 50 → static
LAUNCH_CLEARANCE = 0.1        # fleet spawns just outside planet surface
ROUTE_SEARCH_HORIZON = 150    # exhaustive intercept scan depth (notebook v7; bots use 60–65)
HORIZON = 110                 # forward-sim lookahead
EPISODE_STEPS = 500
COMET_MAX_CHASE_TURNS = 10
ANG_VEL_MIN, ANG_VEL_MAX = 0.025, 0.050
```

> Note the one divergence worth knowing: the notebook raised `ROUTE_SEARCH_HORIZON` to 150 (covers a speed-1 fleet over distance 150); Family A bots cap it at 60–65 for speed. Pick per your time budget.

---

## 2. Geometry & physics primitives — `HIGH`, lift verbatim

All pure, no global state beyond constants. The notebook versions are canonical.

| Function | Signature | Purpose | Deps |
|---|---|---|---|
| `dist` | `(ax,ay,bx,by) -> float` | Euclidean distance. | `math` |
| `orbital_radius` | `(px,py) -> float` | Distance of planet center from sun. | `dist` |
| `is_static_planet` | `(px,py,radius) -> bool` | `orbital_radius + radius >= 50` → doesn't orbit. | `orbital_radius` |
| `fleet_speed` | `(ships) -> float` | **Exact spec curve** `1+(MAX-1)·(log(ships)/log(1000))^1.5`, capped at 6. Every bot reimplements this. | `math` |
| `point_to_segment_distance` | `(px,py,x1,y1,x2,y2) -> float` | Min distance point→segment. Core of continuous collision. | `dist` |
| `segment_intersects_circle` | `(ax,ay,bx,by,cx,cy,r) -> bool` | Does motion A→B pass within `r` of C. Mirrors engine collision. | `point_to_segment_distance` |
| `segment_hits_sun` | `(x1,y1,x2,y2,safety=1.5) -> bool` | Path enters sun keep-out (11.5u). | `point_to_segment_distance` |
| `is_path_clear` | `(sx,sy,tx,ty) -> bool` | Positive-sense wrapper for decision points. | `segment_hits_sun` |
| `launch_point` | `(sx,sy,sr,angle) -> (x,y)` | Fleet spawns at `radius+0.1` in aim direction, not center. | `math` |

`fleet_speed` reference values: 1→1.0, 10→1.96, 50→3.13, 100→3.72, 500→5.27, 1000→6.0 u/turn. **Implication:** big fleets are faster — relevant for intercept and threat ETA, and a lever for any strategy.

---

## 3. Position prediction (orbit & comet) — `HIGH`

The whole reason naive "aim at current position" misses. Notebook §4.

| Function | Signature | Purpose |
|---|---|---|
| `predict_planet_position` | `(planet_id, cur_x, cur_y, radius, initial_by_id, ang_vel, turns_ahead) -> (x,y)` | Rotate an orbiting planet forward. **Anchors radius to the initial position** to avoid float drift; returns current pos for static planets. |
| `predict_comet_position` | `(planet_id, comets, turns) -> (x,y) \| None` | Index into a comet group's precomputed `paths`. |
| `comet_remaining_life` | `(planet_id, comets) -> int` | Turns before comet leaves board — gates whether chasing is worthwhile. |
| `predict_target_position` | `(planet_id, cur_x, cur_y, radius, initial_by_id, ang_vel, comets, comet_ids, turns) -> (x,y) \| None` | **Unified dispatcher** — routes to comet vs orbit predictor. Higher layers call only this. |
| `target_can_move` | `(planet_id, cur_x, cur_y, radius, initial_by_id, comet_ids) -> bool` | Cheap "does prediction even matter" check. |

> **Spec gotcha (documented in the notebook):** planet rotation happens *after* fleet movement each turn, so a fleet arriving on turn T has seen T full rotations — predict position at T, not T−1. Getting this wrong silently degrades every intercept.

`initial_by_id` is a `{planet_id: {'x':…, 'y':…}}` dict built once from the observation's `initial_planets`.

---

## 4. Arrival estimation & safe-path geometry — `HIGH`

Notebook §5. How long a shot takes and whether it clears the sun.

| Function | Signature | Purpose |
|---|---|---|
| `safe_angle_and_distance` | `(sx,sy,sr,tx,ty,tr) -> (angle,hit_dist) \| None` | Direct-shot angle + travel distance; **returns None if path hits sun**. v7: collision-checks to the target *surface entry point*, matching the engine. |
| `estimate_arrival` | `(sx,sy,sr,tx,ty,tr,ships) -> (angle,turns) \| None` | Integer-turn ETA (`ceil`), what the engine uses. |
| `estimate_arrival_frac` | same → `(angle, float_turns)` | Fractional ETA for convergence comparisons. |
| `travel_time` | `(sx,sy,sr,tx,ty,tr,ships) -> int` | Turn count only (1e9 if blocked). |
| `arc_safe_angle` | `(sx,sy,sr,tx,ty,tr,ships) -> (angle,turns) \| None` | **Sun-bypass:** samples 7 aim-points across the target disk and picks the shortest clear chord. Fleets fly straight, but an edge-of-disk chord can clear the sun when the center shot can't. |

This is strictly better than Family B's approach (`owproto`/`sim-search` use a `sun_collision` boolean + reject) and heuristic's `multi_leg_path` waypoint hack. `arc_safe_angle` recovers shots the others throw away.

---

## 5. Verified intercept solver — `HIGH`, the crown jewel

Notebook §6–10. This is the single most valuable thing to reuse, and it's **better than what any of the 8 bots ship**, because every result is forward-sim verified before launch.

| Function | Signature | Role |
|---|---|---|
| `_verify_shot_hits` | `(sx,sy,sr, angle, turns, ships, target_id, tx,ty,tr, initial_by_id, ang_vel, comets, comet_ids) -> bool` | **Ground-truth gate.** Steps the fleet turn-by-turn, checks sun collision each step, and confirms it actually intersects the (moving) target within a window. |
| `_dynamic_tolerance` | `(target_id, initial_by_id, ang_vel, comet_ids) -> int` | Allowed ETA error (1–2 turns) based on how fast the target orbits. |
| `search_safe_intercept` | `(sx,sy,sr, target_id, tx,ty,tr, ships, initial_by_id, ang_vel, comets, comet_ids, tolerance=None) -> (angle,turns,tx,ty) \| None` | **Exhaustive fallback** — scans every turn to `ROUTE_SEARCH_HORIZON`, each candidate verified. |
| `_aim_raw` | (same args as below, internal) | Fast iterative fixed-point convergence (≤16 iters). **Unverified** — caller must verify. |
| `aim_with_prediction` | `(sx,sy,sr, target_id, tx,ty,tr, ships, initial_by_id, ang_vel, comets, comet_ids) -> (angle,turns,tx,ty) \| None` | **Public entry point.** Pipeline: `_aim_raw` → `_verify_shot_hits` → on failure `search_safe_intercept` → else `None` (shot correctly suppressed). |

**Guarantee:** every non-`None` result from `aim_with_prediction` is verified — the bot never launches a fleet the solver predicts will miss (false positives ≈ 0). This directly raises capture efficiency, which matters even more in 4P where wasted ships are negative-sum.

`_fwd_window(turns) -> int` (`= max(8, turns//2)`) sets the verification scan headroom for slow fleets.

---

## 6. Forward simulation: combat & economy — `HIGH` (lift from Family A)

The notebook does **not** include these — take them from a Family A bot (`obnext/main.py` is the cleanest baseline; identical across the family). Signatures below are **verified against `obnext/main.py` (lines 491–742)**. These let you forecast ownership rather than just routing.

| Function | Signature (verified, obnext) | Purpose | Deps |
|---|---|---|---|
| `resolve_arrival_event` | `(owner, garrison, arrivals) -> (new_owner, new_garrison)` | **Faithful same-turn combat** (see rule below). Multi-owner — works for any player count. | none |
| `normalize_arrivals` | `(arrivals, horizon) -> [(eta,owner,ships)]` | Round ETAs up, drop ≤0-ship and beyond-horizon arrivals, sort by turn. | `math` |
| `simulate_planet_timeline` | `(planet, arrivals, player, horizon) -> dict` | Per-planet forward ownership/garrison sim. Returns `owner_at`, `ships_at`, `keep_needed` (binary-searched min garrison to hold), `min_owned`, `first_enemy`, `fall_turn`, `holds_full`, `horizon`. | `resolve_arrival_event`, `normalize_arrivals` |
| `state_at_timeline` | `(timeline, arrival_turn) -> (owner, ships)` | Query the timeline dict at a future turn (clamped to horizon). | none |
| `fleet_target_planet` | `(fleet, planets, initial_by_id={}, ang_vel=0, comets=(), comet_ids=()) -> (planet, eta) \| (None, None)` | **Threat attribution:** which planet an in-flight fleet hits and when. Analytic ray-circle for static planets; bounded sub-step sweep for moving ones. Returns the **planet object** (not id) and `ceil` eta. | `fleet_speed`, `point_to_segment_distance`, `predict_target_position`, `target_can_move` |
| `build_arrival_ledger` | `(fleets, planets, initial_by_id={}, ang_vel=0, comets=(), comet_ids=()) -> {planet_id: [(eta,owner,ships)]}` | Project *all* in-flight fleets (yours + enemies') onto their target planets. | `fleet_target_planet` |

These four (`resolve_arrival_event`, `simulate_planet_timeline`, `fleet_target_planet`, `build_arrival_ledger`) are the core of "will I still own this planet in N turns / how many ships to hold or take it" — strategy-agnostic forecasting that every decision layer can query.

**Combat rule to internalize** (exact, from `resolve_arrival_event` + the timeline loop):
1. **Production grows first:** at the start of each turn, an owned (non-neutral) planet's garrison gains `planet.production`; neutrals do *not* grow.
2. **Same-turn arrivals aggregate by owner**, then the **top two attackers cancel** (survivor = top − second). **A tie at the top destroys both** → survivor is neutral, 0 ships.
3. The surviving attacker group fights the garrison: same owner → ships add; different owner → garrison subtracts, and if it goes negative the attacker captures with `−garrison` ships.

This is the mechanical basis for the 4P negative-sum argument: ships are spent in clashes (send 5 at a 4-ship planet → net +1), so reuse the resolver and let strategy decide whether the trade is worth it.

> **Validation step:** before trusting this layer, diff `resolve_arrival_event` + `simulate_planet_timeline` against `rust_engine`/`parity` — especially the production-before-combat ordering and the tie-destroys-both rule, which are easy to get subtly wrong in a reimplementation.

---

## 7. Lighter alternatives (Family B / heuristic) — `MED`

Use these only if you want a smaller footprint than the full Family A sim layer.

| Function | Source | Purpose |
|---|---|---|
| `collides` / `collides_segment` | owproto / sim-search | Segment-circle test (same idea as `segment_intersects_circle`). |
| `get_planet_trajectories(p, vel, ticks=61)` | Family B | Precompute 60-tick future positions of an orbiting planet as a list (vs the notebook's on-demand `predict_planet_position`). Handy if you query many turns. |
| `get_under_attack` / `planet_under_threat` | Family B / heuristic | Trace enemy fleets forward against owned planets to build a threat map — a cheaper substitute for the full arrival ledger + timeline. |
| `calculate_req_ships[_moving]` | Family B | Fixed-point ships-needed accounting for production growth during transit (no full timeline). |
| `simulate_outcome` | sim-search | 20-tick whole-board forward sim (top-K candidates). Coupled to its value model; useful as a pattern, not drop-in. |
| `solve_intercept` | heuristic | Standalone fixed-point intercept (≤25 iters). Simpler than `aim_with_prediction`, but **unverified** — prefer the notebook solver. |

---

## 8. Utilities — `HIGH`

| Function | Source | Purpose |
|---|---|---|
| `probe_ship_candidates(need, avail, ships) -> list` | physics notebook §11 | Generate sensible ship-count options (25/50/75% of need, need±5/10, clamped to available) for "how many to send" searches. Strategy-agnostic search support. |
| `count_players(planets, fleets) -> int` | Family A | Derive player count from the board (= `max(2, distinct non-neutral owners)`). Needed to switch 2P/4P behavior. |
| `nearest_distance_to_set` / `nearest_sources_to_target` | Family A | k-nearest source planets by distance. Pure geometry. |
| `PhysicsStats` | physics notebook §12 | Diagnostics: aim rate, sun-blocked count, fleet→capture hit rate. Drop-in instrumentation for tuning. |

---

## 9. Recommended assembled core

A single strategy-agnostic `physics.py` / `simworld.py` for the new bot:

```
physics.py        (from the notebook, verbatim)
  ├─ constants
  ├─ dist, orbital_radius, is_static_planet, fleet_speed
  ├─ point_to_segment_distance, segment_intersects_circle,
  │  segment_hits_sun, is_path_clear, launch_point
  ├─ predict_planet_position, predict_comet_position,
  │  comet_remaining_life, predict_target_position, target_can_move
  ├─ safe_angle_and_distance, estimate_arrival[_frac], travel_time, arc_safe_angle
  ├─ _verify_shot_hits, _dynamic_tolerance, _aim_raw,
  │  search_safe_intercept, aim_with_prediction      ← verified solver
  ├─ probe_ship_candidates
  └─ PhysicsStats

simworld.py       (lifted from obnext/main.py, validated vs rust_engine/parity)
  ├─ resolve_arrival_event, normalize_arrivals
  ├─ simulate_planet_timeline, state_at_timeline
  └─ fleet_target_planet, build_arrival_ledger
```

Everything above is mechanism. The decision layer on top — target scoring, **per-opponent modeling, and the 2P-aggressive / 4P-conservative split** — is where our bot should differentiate, since all 8 reference bots reuse roughly this same mechanism but none reason about individual opponents well.

---

## Excluded on purpose (strategy, not mechanism)
`get_custom_score` / `target_value` / `apply_score_modifiers`, all `*_VALUE_MULT` / `*_COST_TURN_WEIGHT` / send-margin tables, `build_modes` (posture), `opening_filter`, mission builders (`build_*_missions`), `ffa_owner_pressure_multiplier`, reserve/`attack_budget` accounting, beam search. These encode *what to do*; reuse them only as design references, not as a shared library.
