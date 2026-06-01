# ow4 — strategy

Heuristic 2-player Orbit Wars bot. Re-evaluates from scratch every turn. No
state cached across turns, no value net, no MCTS, no rollout.

## Guiding principles

1. **Re-plan every turn from current state.** No commitment to multi-turn plans.
   If the opponent does something unexpected, the next turn's plan reacts.
2. **Every ship sent has an explicit justification.** No "send free ships somewhere"
   heuristics, no opportunity-cost harass. A ship leaves a planet only when the
   math says doing so is positive-ROI.
3. **No magic numbers without derivation.** Constants are either user-spec
   (`SCORE_HORIZON = 100`, `denial = 2× in 2p`) or computed from game physics
   (`fleet_speed(N)`, capture min via simulation).

## Per-turn pipeline

```
plan(state)
  ↓
1. build defense ledger     → per-planet surplus
2. snipe pass               → exploit enemy attacks on neutrals in my territory
3. opening search           → currently disabled, see below
4. attack pass              → ROI-scored captures + reinforcements
   ↓
output: list of (from_planet_id, angle_radians, num_ships)
```

## 1. Defense ledger (`ledger.rs`)

For each owned planet, compute the minimum garrison I must keep so the planet
never flips to enemy. Surplus = `current_ships − min_garrison_needed`.

The ledger considers two threats:

**a. Real in-flight enemy fleets.** Pulled directly from `state.fleets`. Each
predicts where it'll collide via `predict_fleet_collision`.

**b. One phantom threat per planet.** For each owned planet, find the single
strongest enemy planet (by *net delivered force*: their ships minus my
production buildup during their flight time at their real `fleet_speed`).
Add that enemy's full garrison as a phantom arrival at the predicted tick.

Simulating ownership over the next `DEFENSE_HORIZON = 40` ticks with these
arrivals tells us the worst point my ship count reaches. Reserve enough that
this worst point is ≥ 0; everything above is surplus.

Using *only the single strongest* enemy avoids over-reserving against multiple
enemies who can't realistically all attack the same planet simultaneously at
full force.

## 2. Snipe (`snipe.rs`)

The user's strategy notes call this "sniping": when the opponent commits a
fleet to a neutral planet, I send the minimum-sized fleet to arrive on the
same tick. Their fleet wears down the neutral garrison; mine finishes the
survivors and captures the planet with ~1 ship.

For each enemy arrival on a neutral:
1. Simulate combat at the enemy's arrival tick *without* my snipe. If the
   enemy doesn't actually capture (neutral survives), no snipe opportunity.
2. Compute snipe size = `enemy_survivors + 1` ships.
3. For each of my planets, find the smallest ship count whose `dir_to_hit`
   arrival tick is exactly the enemy's tick (or one tick later, after their
   capture). Multi-source: accumulate from multiple sources if no single one
   delivers enough alone.
4. If a viable plan exists, commit.

This is one of the few opportunities where I spend a *handful* of ships to
deny the opponent a planet they spent a *full attack force* on — strictly
favorable trade.

## 3. Opening search (`opening.rs`) — currently disabled

A beam search (width 6, depth 4) over the next four captures, picking the
first move that maximises `Σ production × (HORIZON − arrival)` for the
sequence. Constrained so the first action must launch *this turn*.

**Status:** empirically the search picks the same first move as the greedy
attack planner ~all the time, so it adds compile-and-run cost with no gain.
Disabled via `OPENING_STEP_LIMIT = -1`. Kept in the tree to revisit when
the criterion improves (e.g., adversarial response modeling).

## 4. Attack (`attack.rs`)

Generate one candidate per `(source, target)` pair where:
- target is *not* already projected to be mine at horizon (no double-commit)
- source is mine with surplus ≥ 1
- minimum ships from source that capture the target (binary search using
  the combat simulation against existing in-flight arrivals)
- the target is not race-sniped by any enemy planet

### The scoring formula

For each candidate:

```
score = (production × denial × remaining) / ships
```

Where:

- **`production`** is the target planet's per-turn ship output.
- **`denial`** is **2.0** if the *projected* end-owner without my action is
  the enemy, else **1.0**. Per the user's 2p spec: "stealing production is
  twice as valuable as merely having it." This means defending my own
  planets that are projected to flip is just as valuable as capturing
  enemy planets — both are denial.
- **`remaining`** is `min(SCORE_HORIZON, comet_remaining) − arrival_dt`,
  i.e. how many ticks I get to enjoy the production until either the
  scoring horizon or the comet dies, whichever is sooner. `SCORE_HORIZON =
  100` per user spec ("cap at 100 so infinities don't exist").
- **`ships`** is the minimum ship count from the binary search.

### The two gates

A candidate is dropped if either:

1. **Race gate**: any enemy planet's *real* `fleet_speed(enemy.ships)` race
   time to the target is strictly less than my arrival tick. They'd get
   there first and snipe.
2. **ROI gate**: `production × denial × remaining ≤ ships`. Sending more
   ships than I'll get back as production is a losing trade. Strict positive
   ROI required.

Candidates that pass both gates are sorted by score descending. Greedy
commit: take each in order, deduplicating by target, deducting from each
source's surplus as we go.

## Combat resolution (`combat.rs`)

Pure re-implementation of the game's rules:
- All arrivals on a planet at the same tick are sorted by ship count desc.
- Top two fight: `|top1 − top2|` survives, owned by top1 (ties → both die,
  planet unchanged).
- Survivor then fights the planet garrison: same-owner adds, otherwise
  subtracts and flips if negative.
- Production accrues at the start of each tick to owned planets only.
- Comets disappear when their path runs out.

This `simulate_planet(planet, arrivals, horizon)` is the load-bearing
primitive — both `captures` decisions and the snipe / attack scoring use it.

## Pathing (`pathing.rs`)

`dir_to_hit(source, target, num_ships, state, turns_in_future) →
PathResult { angle, time }` — adapted from apollo / duck. Handles:
- Source-circle / target-circle intersection arcs per candidate arrival tick
- Static obstacle (sun + non-orbiting planet) cone subtraction
- Moving obstacle subtraction (orbiting planets, comets) per tick
- Sub-tick collision verification via swept-circle math

`fleet_speed(ships, max_speed) = 1 + (max_speed − 1) × (log ships / log
1000)^1.5`, clamped to `[1, max_speed]`. So a 1-ship fleet moves at speed
1, a 1000-ship fleet at max speed.

Every action ow4 emits goes through `dir_to_hit`, so we never miss / shoot
into the sun / go out of bounds.

## What was deliberately rejected

- **"Send free ships to nearest enemy" harass.** Tested and load-bearing for
  win rate (3/20 vs 0/20 against apollo_fast), but the user rejected it as
  "randomly sending ships" — not principled. Removed in favor of "ships
  sit if no positive-ROI action exists."
- **Wave / split sends** (send 30 + 20 instead of 50). User pointed out
  `fleet_speed ∝ log(N)^1.5` makes a single bigger fleet strictly faster,
  and a competent opponent's counter math handles two arrivals fine.
- **Fork trajectories / bait-and-redirect.** Fleets can't be redirected
  after launch — physically impossible in this game.
- **Multi-source attack subset search.** Implemented but always regressed
  vs the simple single-source planner; the "first src dumps surplus while
  later srcs add minimum" approach wastes ships, and a true optimal subset
  search is combinatorial.
- **Long-term plans.** Per "best bots recompute every turn."

## Current performance

5-seed gauntlets (current bot — clean, no harass):

| opponent | wins/5 |
|---|---|
| starter | 5 |
| heuristic | varies, ~2-4 |
| hellburner | varies, ~0-1 |
| owheuristic | 0 |
| apollo_fast | varies, ~0-1 in 10 |

Performance vs apollo_fast is below the bar (target was >55%). The
principled foundation is sound; closing the gap to apollo would need
either a multi-strategy rollout (which apollo itself does) or a
fundamentally different planner architecture.
