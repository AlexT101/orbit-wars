//! Port of the open-source `hellburner` strategy
//! ([bots/external/hellburner/main.py](../../external/hellburner/main.py)).
//!
//! Reuses our Rust infra:
//!   * [`crate::helpers::aim_with_prediction`] — combines hellburner's
//!     `intercept_planet` + `first_planet_hit` (returns Some only when the
//!     shot reaches the target unblocked by sun/planet/comet).
//!   * [`crate::world::WorldState`] — per-turn snapshot incl. `TimelineCache`
//!     which already plays the role of hellburner's `destination_list`.
//!   * [`crate::helpers::simulate_planet_timeline`] /
//!     [`WorldState::projected_timeline`] — hellburner's `simulate_planet_timeline`.
//!
//! Hellburner-specific data we build here:
//!   * Proximity graph (`Config::max_distance`, `ROTATION_LOOK_AHEAD_TURNS=10`).
//!   * `reinforcement_target` per owned planet (pressure-weighted BFS).
//!   * Per-turn `PlanState` (spent ships + planned commitments).

#![allow(dead_code)]

use std::cell::RefCell;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::cache::{AimCacheVerdict, InvariantVerdict};
use crate::constants::{
    ally_pressure_ratio, enemy_offset_lookahead, frontier_pressure_ratio, offset_lookahead,
    reinforcement_pressure_decay, reinforcement_pressure_turns, rotation_look_ahead_turns,
};
use crate::early_game::OpeningEvent;
use crate::engine::{MoveAction, Planet};
use crate::helpers::{
    aim_ignoring_comets, aim_with_prediction, available_at_timeline, dist,
    simulate_checkpoint_into, simulate_planet_timeline, AimResult, ArrivalEvent, PlanetTimeline,
};
use crate::world::WorldState;

pub struct HellburnerModel<'a> {
    pub state: &'a WorldState<'a>,
    /// Planet ids excluding comets — comets never appear in the proximity
    /// graph, reinforcement BFS, or target loops.
    pub non_comet_ids: HashSet<i64>,
    pub inbound_edges: HashMap<i64, Vec<(i64, f64)>>,
    pub outbound_edges: HashMap<i64, Vec<(i64, f64)>>,
    pub reinforcement_target: HashMap<i64, i64>,
    /// Enemy-owned targets the planner skips this turn: our (ally) pressure on
    /// the target is below `ALLY_PRESSURE_RATIO` of the enemy pressure on it,
    /// so a capture attempt would be contesting a planet the enemy can
    /// out-reinforce. Computed from baseline timelines only (plan-independent),
    /// before any source selection.
    pub pressure_gated_targets: HashSet<i64>,
    /// L1 hot cache for `plan_shot`: per-`HellburnerModel` (i.e. one bot turn)
    /// memoization of `(src, target, ships, launch_turn_offset) → aim`.
    /// Avoids repeated traffic to the L2 `EntityCache::aim_cache` inside the
    /// inner loops of `evaluate_frontline_strategy`, `evaluate_target`
    /// (where the same shot can be re-queried several times across the main
    /// loop and the worst-case sub-rollout).
    shot_cache: RefCell<HashMap<(i64, i64, i64, i64), Option<AimResult>>>,
}

impl<'a> HellburnerModel<'a> {
    pub fn build(state: &'a WorldState<'a>) -> Self {
        let player = state.player;

        let non_comet_ids: HashSet<i64> = state
            .planets
            .iter()
            .filter(|p| !state.comet_ids.contains(&p.id))
            .map(|p| p.id)
            .collect();

        let non_comets: Vec<&Planet> = state
            .planets
            .iter()
            .filter(|p| non_comet_ids.contains(&p.id))
            .collect();

        let mut future_pos: HashMap<i64, [f64; 2]> = HashMap::default();
        for p in &non_comets {
            let pos = state
                .cache
                .position(p.id, 1 + rotation_look_ahead_turns())
                .unwrap_or([p.x, p.y]);
            future_pos.insert(p.id, pos);
        }

        let mut inbound_edges: HashMap<i64, Vec<(i64, f64)>> = HashMap::default();
        let mut outbound_edges: HashMap<i64, Vec<(i64, f64)>> = HashMap::default();
        for &pid in &non_comet_ids {
            inbound_edges.insert(pid, Vec::new());
            outbound_edges.insert(pid, Vec::new());
        }
        for src in &non_comets {
            for dst in &non_comets {
                if src.id == dst.id {
                    continue;
                }
                let [fx, fy] = future_pos[&dst.id];
                let travel = dist(src.x, src.y, fx, fy);
                if travel <= state.config.max_distance {
                    inbound_edges
                        .get_mut(&dst.id)
                        .unwrap()
                        .push((src.id, travel));
                    outbound_edges
                        .get_mut(&src.id)
                        .unwrap()
                        .push((dst.id, travel));
                }
            }
        }

        let mut model = Self {
            state,
            non_comet_ids,
            inbound_edges,
            outbound_edges,
            reinforcement_target: HashMap::default(),
            pressure_gated_targets: HashSet::default(),
            shot_cache: RefCell::new(HashMap::default()),
        };
        model.reinforcement_target = build_reinforcement_targets(state, &model, player);
        model.pressure_gated_targets = build_pressure_gate(state, &model);
        model
    }

    /// Cached aim with an optional future launch offset.
    ///
    /// Caching:
    ///   * L1 — per-`HellburnerModel` `shot_cache`, keyed by
    ///     `(src, target, ships, launch_turn_offset)`. Avoids repeated
    ///     traffic to L2 inside hot evaluation loops and the early-game DFS.
    ///   * L2 — `EntityCache::aim_cache`, indexed by absolute launch turn
    ///     so launch-now and delayed-launch entries share slots whenever
    ///     their `current_turn + offset` matches. Shared across every
    ///     `HellburnerModel` built during this bot turn, with rollout
    ///     forward-sim entries, and across real turns (with lazy comet-spawn
    ///     re-verification inside `aim_cache_lookup`).
    pub fn plan_shot(
        &self,
        src_id: i64,
        target_id: i64,
        ships: i64,
        launch_turn_offset: i64,
    ) -> Option<AimResult> {
        let ships = ships.max(1);
        let cache = self.state.cache;
        // L1 is keyed by the *absolute* launch turn so the step-scoped shared
        // cache stays correct as the rollout walks `current_turn` forward (a
        // relative-offset key would collide across turns). Falls back to the
        // model's own per-model cache when no shared L1 is threaded in.
        let abs_launch = cache.current_turn + launch_turn_offset;
        let key = (src_id, target_id, ships, abs_launch);
        let l1 = self.state.shot_l1.unwrap_or(&self.shot_cache);
        if let Some(&cached) = l1.borrow().get(&key) {
            return cached;
        }
        let lookup = cache.aim_cache_lookup(src_id, target_id, ships, launch_turn_offset);
        let result = match lookup {
            AimCacheVerdict::Hit(r) => r,
            AimCacheVerdict::Miss | AimCacheVerdict::Stale => {
                // L3 — cross-turn invariant fast path for disc-qualified
                // static→static / orbiting→orbiting shots. Skips lead_target_from and
                // the per-entity planet sweep, only re-checking comets per turn.
                match cache.invariant_aim_lookup(src_id, target_id, ships, launch_turn_offset) {
                    InvariantVerdict::Use(r) => Some(r),
                    InvariantVerdict::SingleSolve => {
                        let r = aim_with_prediction(
                            cache,
                            src_id,
                            target_id,
                            ships,
                            launch_turn_offset,
                        );
                        cache.aim_cache_store(src_id, target_id, ships, launch_turn_offset, r);
                        r
                    }
                    InvariantVerdict::DualSolve => {
                        // Populate the invariant base with one comet-free solve,
                        // then gate it against just the comets. Comet-clear ⇒ the
                        // base is exactly this turn's shot (no second solve);
                        // comet-blocked / disqualified ⇒ fall back to a normal
                        // full solve.
                        let base = aim_ignoring_comets(
                            cache,
                            src_id,
                            target_id,
                            ships,
                            launch_turn_offset,
                        );
                        cache.invariant_aim_store(
                            src_id,
                            target_id,
                            ships,
                            launch_turn_offset,
                            base,
                        );
                        match cache.invariant_aim_lookup(
                            src_id,
                            target_id,
                            ships,
                            launch_turn_offset,
                        ) {
                            InvariantVerdict::Use(r) => Some(r),
                            _ => {
                                let r = aim_with_prediction(
                                    cache,
                                    src_id,
                                    target_id,
                                    ships,
                                    launch_turn_offset,
                                );
                                cache.aim_cache_store(
                                    src_id,
                                    target_id,
                                    ships,
                                    launch_turn_offset,
                                    r,
                                );
                                r
                            }
                        }
                    }
                }
            }
        };
        l1.borrow_mut().insert(key, result);
        result
    }
}

fn build_reinforcement_targets(
    state: &WorldState,
    model: &HellburnerModel,
    player: i64,
) -> HashMap<i64, i64> {
    let pressure = reinforcement_pressure(state, model);
    let attraction = expansion_attraction(state, model, player);
    let mut best: HashMap<i64, ReinforcementRoute> = HashMap::default();
    let mut queue: Vec<i64> = Vec::new();

    // Every owned planet is a potential sink, keyed by (enemy pressure,
    // expansion attraction). Pressure dominates, so combat draining is
    // unchanged; attraction only breaks ties — chiefly the low-pressure early
    // game, where the enemy-driven pressure is uniformly ~0 and this instead
    // drifts ships toward the frontier rather than leaving the BFS dormant. The
    // BFS walks backward through owned edges, preserving the first hop each
    // source should use.
    for p in &state.my_planets {
        if !model.non_comet_ids.contains(&p.id) {
            continue;
        }
        best.insert(
            p.id,
            ReinforcementRoute {
                sink_pressure: pressure.get(&p.id).copied().unwrap_or(0.0),
                sink_attraction: attraction.get(&p.id).copied().unwrap_or(f64::NEG_INFINITY),
                hops: 0,
                next_hop: p.id,
                sink_id: p.id,
            },
        );
        queue.push(p.id);
    }

    let mut head = 0;
    while head < queue.len() {
        let node = queue[head];
        head += 1;
        let Some(route) = best.get(&node).copied() else {
            continue;
        };
        for (sid, _) in &model.inbound_edges[&node] {
            if !model.non_comet_ids.contains(sid) || state.planet(*sid).owner != player {
                continue;
            }
            let candidate = ReinforcementRoute {
                sink_pressure: route.sink_pressure,
                sink_attraction: route.sink_attraction,
                hops: route.hops + 1,
                next_hop: node,
                sink_id: route.sink_id,
            };
            if best
                .get(sid)
                .map(|current| reinforcement_route_is_better(candidate, *current))
                .unwrap_or(true)
            {
                best.insert(*sid, candidate);
                queue.push(*sid);
            }
        }
    }

    let mut out: HashMap<i64, i64> = HashMap::default();
    for p in &state.my_planets {
        if !model.non_comet_ids.contains(&p.id) {
            continue;
        }
        let own_pressure = pressure.get(&p.id).copied().unwrap_or(0.0);
        let own_attraction = attraction.get(&p.id).copied().unwrap_or(f64::NEG_INFINITY);
        let Some(route) = best.get(&p.id).copied() else {
            continue;
        };
        if route.next_hop == p.id {
            continue;
        }
        // Flow toward a strictly better sink by (pressure, attraction): a
        // higher-pressure combat sink as before, or — when pressure ties — a
        // more frontier-facing one (the early-game drift).
        let better = (route.sink_pressure, route.sink_attraction) > (own_pressure, own_attraction);
        if !better {
            continue;
        }
        // The frontier-ratio gate guards *combat* draining only — a frontier
        // planet must not bleed its own defense into a higher-pressure sink.
        // Pure drift (equal pressure, higher attraction) is exempt: pulling
        // idle ships toward the frontier is the entire point.
        let pressure_drain = route.sink_pressure > own_pressure;
        let frontier_source = is_reinforcement_frontier(state, model, p.id, player);
        let clears_frontier_ratio = !pressure_drain
            || !frontier_source
            || reinforcement_pressure_clears_frontier_ratio(route.sink_pressure, own_pressure);
        if clears_frontier_ratio {
            out.insert(p.id, route.next_hop);
        }
    }
    out
}

#[derive(Clone, Copy)]
struct ReinforcementRoute {
    sink_pressure: f64,
    /// Expansion attraction of the sink (see [`expansion_attraction`]). Pure
    /// tiebreaker below `sink_pressure`, so combat routing is unchanged; it only
    /// decides flow when pressure ties (chiefly the zero-pressure early game).
    sink_attraction: f64,
    hops: i64,
    next_hop: i64,
    sink_id: i64,
}

fn reinforcement_route_is_better(
    candidate: ReinforcementRoute,
    current: ReinforcementRoute,
) -> bool {
    let cand_key = (candidate.sink_pressure, candidate.sink_attraction);
    let cur_key = (current.sink_pressure, current.sink_attraction);
    cand_key > cur_key
        || (cand_key == cur_key
            && (candidate.hops < current.hops
                || (candidate.hops == current.hops
                    && (candidate.sink_id, candidate.next_hop)
                        < (current.sink_id, current.next_hop))))
}

/// Per owned planet, how frontier-facing it is: the negative distance to its
/// nearest reachable non-owned planet — enemy *or* neutral. Higher (closer)
/// means nearer to where ships are useful; planets that can't reach any
/// non-owned planet get `f64::NEG_INFINITY` so idle ships flow off them.
///
/// Used only as a tiebreaker below enemy reinforcement pressure. The pressure
/// BFS is enemy-driven and goes dormant when nothing is in range (the early
/// game), stranding ships on interior planets; this drifts them toward planets
/// that can actually launch at non-owned territory, pre-positioning for
/// expansion and eventual contact.
fn expansion_attraction(
    state: &WorldState,
    model: &HellburnerModel,
    player: i64,
) -> HashMap<i64, f64> {
    let mut out: HashMap<i64, f64> = HashMap::default();
    for p in &state.my_planets {
        if !model.non_comet_ids.contains(&p.id) {
            continue;
        }
        let nearest = model.outbound_edges[&p.id]
            .iter()
            .filter(|(did, _)| {
                model.non_comet_ids.contains(did) && state.planet(*did).owner != player
            })
            .map(|&(_, d)| d)
            .fold(f64::INFINITY, f64::min);
        let attraction = if nearest.is_finite() {
            -nearest
        } else {
            f64::NEG_INFINITY
        };
        out.insert(p.id, attraction);
    }
    out
}

fn is_reinforcement_frontier(
    state: &WorldState,
    model: &HellburnerModel,
    planet_id: i64,
    player: i64,
) -> bool {
    model.inbound_edges[&planet_id]
        .iter()
        .any(|(sid, _)| state.planet(*sid).owner != player)
        || model.outbound_edges[&planet_id]
            .iter()
            .any(|(did, _)| state.planet(*did).owner != player)
}

fn reinforcement_pressure_clears_frontier_ratio(
    target_pressure: f64,
    source_pressure: f64,
) -> bool {
    target_pressure >= source_pressure * frontier_pressure_ratio()
}

fn reinforcement_pressure(state: &WorldState, model: &HellburnerModel) -> HashMap<i64, f64> {
    let mut pressure: HashMap<i64, f64> = HashMap::default();
    for target in &state.my_planets {
        if !model.non_comet_ids.contains(&target.id) {
            continue;
        }
        let total = enemy_pressure_combined(
            state,
            model,
            target.id,
            &state.enemy_planets,
            enemy_offset_lookahead(),
        );
        pressure.insert(target.id, total);
    }
    pressure
}

/// Owner-bucketed pressure on `target_id`: for each source planet, the best
/// time-decayed deliverable force over launch offsets `0..=max_offset`
/// (`ships × reinforcement_pressure_weight(arrival)`), accumulated into the
/// source's owner. Planets owned by the same player cooperate (summed within an
/// owner); the caller decides how to combine across owners (`pressure_from`
/// sums, `enemy_pressure_combined` maxes the strongest owner and discounts the
/// rest). Availability is read from
/// baseline timelines only, so the result is plan-independent and safe to
/// compute once per model build.
fn pressure_by_owner(
    state: &WorldState,
    model: &HellburnerModel,
    target_id: i64,
    sources: &[Planet],
    max_offset: i64,
) -> HashMap<i64, f64> {
    let mut by_owner: HashMap<i64, f64> = HashMap::default();
    for src in sources {
        if src.id == target_id || !model.non_comet_ids.contains(&src.id) {
            continue;
        }
        let mut best_contribution = 0.0;
        for offset in 0..=max_offset.min(reinforcement_pressure_turns()) {
            let ships = owner_available_to_launch_at(state, src.id, src.owner, offset);
            if ships <= 0 {
                continue;
            }
            let Some((_, travel_turns, _, _, _)) =
                model.plan_shot(src.id, target_id, ships, offset)
            else {
                continue;
            };
            let arrival = (offset + travel_turns).max(1);
            if arrival <= reinforcement_pressure_turns() {
                let contribution = ships as f64 * reinforcement_pressure_weight(arrival);
                if contribution > best_contribution {
                    best_contribution = contribution;
                }
            }
        }
        if best_contribution > 0.0 {
            *by_owner.entry(src.owner).or_insert(0.0) += best_contribution;
        }
    }
    by_owner
}

/// Total pressure `sources` exert on `target_id`, summed across every source
/// regardless of owner. Correct only for cooperating sources (i.e. all of our
/// own planets) — use this for ally pressure.
fn pressure_from(
    state: &WorldState,
    model: &HellburnerModel,
    target_id: i64,
    sources: &[Planet],
    max_offset: i64,
) -> f64 {
    pressure_by_owner(state, model, target_id, sources, max_offset)
        .values()
        .sum()
}

/// Weight applied to every enemy owner's pressure except the single strongest
/// one when combining per-owner pressures into one threat number. `1.0` is a
/// full coalition (every enemy launches at once); `0.0` is only the strongest
/// single opponent. The middle ground counts the most dangerous opponent fully
/// while crediting the rest a discounted share — secondary opponents add real
/// risk without assuming full coordination.
const SECONDARY_ENEMY_PRESSURE_WEIGHT: f64 = 1.3;

/// Combined enemy pressure on `target_id`: the strongest single enemy owner's
/// pressure at full weight, plus `SECONDARY_ENEMY_PRESSURE_WEIGHT` × the summed
/// pressure of every other enemy owner. Models the most dangerous independent
/// opponent fully while still crediting secondary opponents a discounted share,
/// rather than an unrealistic full coalition (plain sum) or ignoring them
/// entirely (plain max). In 2p there is one enemy owner, so `total == strongest`
/// and this reduces exactly to `pressure_from`; 2p play is unchanged. It differs
/// only in 4p with multiple live opponents.
fn enemy_pressure_combined(
    state: &WorldState,
    model: &HellburnerModel,
    target_id: i64,
    sources: &[Planet],
    max_offset: i64,
) -> f64 {
    let by_owner = pressure_by_owner(state, model, target_id, sources, max_offset);
    let total: f64 = by_owner.values().sum();
    let strongest = by_owner.values().copied().fold(0.0, f64::max);
    // strongest at full weight + the remaining owners (total − strongest) discounted
    strongest + SECONDARY_ENEMY_PRESSURE_WEIGHT * (total - strongest)
}

/// Enemy-owned targets failing the ally-pressure gate. For each enemy planet,
/// ally pressure (our planets, our `OFFSET_LOOKAHEAD`) is compared against the
/// combined enemy pressure (all enemy planets except the target itself, bucketed
/// by owner: strongest owner full + the rest discounted, `ENEMY_OFFSET_LOOKAHEAD`)
/// — both computed exactly like reinforcement pressure. Targets where
/// ally < `ALLY_PRESSURE_RATIO` × enemy are gated.
fn build_pressure_gate(state: &WorldState, model: &HellburnerModel) -> HashSet<i64> {
    let mut gated: HashSet<i64> = HashSet::default();
    for target in &state.enemy_planets {
        if !model.non_comet_ids.contains(&target.id) {
            continue;
        }
        let enemy = enemy_pressure_combined(
            state,
            model,
            target.id,
            &state.enemy_planets,
            enemy_offset_lookahead(),
        );
        if enemy <= 0.0 {
            continue;
        }
        let ally = pressure_from(
            state,
            model,
            target.id,
            &state.my_planets,
            offset_lookahead(),
        );
        if ally < ally_pressure_ratio() * enemy {
            gated.insert(target.id);
        }
    }
    gated
}

fn reinforcement_pressure_weight(turns: i64) -> f64 {
    let turns = turns.max(1);
    if turns <= 1 {
        return 1.0;
    }
    let span = (reinforcement_pressure_turns() - 1).max(1) as f64;
    let exponent = (turns - 1) as f64 / span;
    (reinforcement_pressure_decay()).powf(exponent)
}

fn owner_available_to_launch_at(
    state: &WorldState,
    planet_id: i64,
    owner: i64,
    offset: i64,
) -> i64 {
    let planet = state.planet(planet_id);
    if planet.owner != owner || owner == -1 {
        return 0;
    }
    match state.timeline_cache.baseline(planet_id) {
        Some(timeline) => available_at_timeline_for_owner(timeline, owner, state.player, offset),
        None => (planet.ships + planet.production * offset.max(0)).max(0),
    }
}

// ── PlanState: turn-local commitments ────────────────────────────────────

#[derive(Default)]
struct PlanState {
    spent: HashMap<i64, i64>,
    planned: HashMap<i64, Vec<ArrivalEvent>>,
}

impl PlanState {
    fn ships_available(&self, world: &WorldState, src: &Planet) -> i64 {
        self.ships_available_at(world, src, world.player, 0)
    }
    /// Rollout-aware available ships at a future launch offset.
    ///
    /// Returns the most this source can ship out at `offset` without driving
    /// any later turn it still owns negative. Withdrawing ships at `offset`
    /// removes them from every turn `≥ offset`, so the bound is the forward-min
    /// of the garrison over the owned run starting at `offset`
    /// ([`available_at_timeline`]) — not the single-turn garrison, which would
    /// over-commit whenever a future enemy arrival shrinks the planet.
    ///
    /// O(1) in the common friendly-source case — it reads the prebuilt
    /// player-specific forward-min. Enemy-owner queries scan the same
    /// player-agnostic owner/ship timeline over the continuous run controlled by
    /// that owner. Only when this source has planned arrivals queued *this* turn
    /// does it pay a single per-call planet sim to fold those in. Conservative
    /// against `spent` for our own sources: every prior commitment from this
    /// source is subtracted regardless of when those ships are scheduled to
    /// leave (so a commit at any offset correctly debits all offsets, keeping
    /// the greedy planner from going negative).
    fn ships_available_at(&self, world: &WorldState, src: &Planet, owner: i64, offset: i64) -> i64 {
        let offset = offset.max(0);
        let spent = if owner == world.player {
            self.spent.get(&src.id).copied().unwrap_or(0)
        } else {
            0
        };
        let planned: &[ArrivalEvent] = self
            .planned
            .get(&src.id)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let available = if planned.is_empty() {
            baseline_available_at_for_owner(world, src.id, owner, offset)
        } else {
            // This source also has reinforcements we've planned this turn.
            // Sim the full horizon (not just up to `offset`) so the forward-min
            // sees every later turn.
            let tl = world.projected_timeline(src.id, world.timeline_cache.horizon, planned, &[]);
            available_at_timeline_for_owner(&tl, owner, world.player, offset)
        };
        (available - spent).max(0)
    }
    fn commit(&mut self, src_id: i64, target_id: i64, ships: i64, arrival_turn: i64, owner: i64) {
        *self.spent.entry(src_id).or_insert(0) += ships;
        self.planned
            .entry(target_id)
            .or_default()
            .push(ArrivalEvent {
                turns: arrival_turn.max(1),
                owner,
                ships,
            });
    }

    /// Move a previously-committed friendly arrival from one target to another.
    /// Used by `redirect_moves` when a launch-this-turn fleet is retargeted to
    /// an intermediate `C`: the ships no longer reach `B` this turn, they land
    /// at `C` instead. `spent` is left untouched — the same source still
    /// launches the same ships — so this only rewrites the `planned` ledger that
    /// subsequent reroute decisions consult for ownership projection.
    fn reroute(
        &mut self,
        old_target: i64,
        old_arrival: i64,
        new_target: i64,
        new_arrival: i64,
        ships: i64,
        owner: i64,
    ) {
        if let Some(events) = self.planned.get_mut(&old_target) {
            let want = old_arrival.max(1);
            if let Some(pos) = events
                .iter()
                .position(|e| e.turns == want && e.ships == ships && e.owner == owner)
            {
                events.swap_remove(pos);
            }
        }
        self.planned
            .entry(new_target)
            .or_default()
            .push(ArrivalEvent {
                turns: new_arrival.max(1),
                owner,
                ships,
            });
    }
}

// ── Timeline helpers ─────────────────────────────────────────────────────

fn target_timeline(
    world: &WorldState,
    target_id: i64,
    extras: &[ArrivalEvent],
    plan: &PlanState,
) -> PlanetTimeline {
    let planned: &[ArrivalEvent] = plan
        .planned
        .get(&target_id)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    world.projected_timeline(target_id, world.timeline_cache.horizon, planned, extras)
}

fn final_owner(timeline: &PlanetTimeline) -> i64 {
    timeline.owner_at[timeline.horizon as usize]
}

pub(crate) fn available_at_timeline_for_owner(
    timeline: &PlanetTimeline,
    owner: i64,
    timeline_player: i64,
    offset: i64,
) -> i64 {
    if owner == timeline_player {
        return available_at_timeline(timeline, offset);
    }
    let start = offset.max(0).min(timeline.horizon) as usize;
    if timeline.owner_at[start] != owner {
        return 0;
    }
    let mut available = i64::MAX;
    for t in start..=timeline.horizon as usize {
        if timeline.owner_at[t] != owner {
            break;
        }
        available = available.min(timeline.ships_at[t].max(0));
    }
    if available == i64::MAX {
        0
    } else {
        available.max(0)
    }
}

/// Baseline (plan-free) launchable ships for `owner` at launch `offset` from
/// planet `id`: the forward-min over `owner`'s owned run from `offset`, read
/// from the cached baseline timeline, or linear growth when no trajectory is
/// cached (a purely growing series has its forward-min at the read offset).
/// Callers that track turn-local commitments subtract their own `spent`/
/// `planned` on top.
pub(crate) fn baseline_available_at_for_owner(
    world: &WorldState,
    id: i64,
    owner: i64,
    offset: i64,
) -> i64 {
    match world.timeline_cache.baseline(id) {
        Some(b) => available_at_timeline_for_owner(b, owner, world.player, offset),
        None => {
            let p = world.planet(id);
            if p.owner == owner && owner != -1 {
                p.ships + p.production * offset.max(0)
            } else {
                0
            }
        }
    }
}

fn baseline_owns(world: &WorldState, planet_id: i64) -> bool {
    let h = world.timeline_cache.horizon as usize;
    match world.timeline_cache.baseline(planet_id) {
        Some(b) => b.owner_at[h] == world.player,
        None => world.planet(planet_id).owner == world.player,
    }
}

// ── Unified zero-sum scoring ─────────────────────────────────────────────

fn owner_value(owner: i64, player: i64) -> f64 {
    if owner == player {
        1.0
    } else if owner == -1 {
        0.0
    } else {
        // Enemy magnitude is tunable. At the default 1.0 this is the original
        // symmetric ±1 (so capturing from an enemy is worth a 2.0 owner swing
        // vs 1.0 for a neutral — the implicit 2:1). Raising it makes taking
        // from / losing to an enemy weigh more without touching neutral value.
        -crate::constants::score_enemy_capture_bonus()
    }
}

fn signed_ships(owner: i64, ships: i64, player: i64) -> f64 {
    owner_value(owner, player) * ships.max(0) as f64
}

/// Zero-sum value of a trial target timeline relative to its baseline.
///
/// The old local score valued only "we own it at horizon" and latest arrival
/// time. This consumes the full simulated timeline already produced for each
/// trial, so temporary steals, third-party pileups, weak holds, and costly
/// captures are priced by actual ownership duration and final ship delta.
fn timeline_delta_score(
    world: &WorldState,
    target: &Planet,
    baseline: &PlanetTimeline,
    owner_at: &[i64],
    ships_at: &[i64],
    ships_committed: i64,
    start_turn: i64,
) -> f64 {
    let player = world.player;
    let h = baseline.horizon as usize;
    let production = target.production as f64;
    let w_production = crate::constants::score_w_production();
    let w_final_ships = crate::constants::score_w_final_ships();
    let w_ship_cost = crate::constants::score_w_ship_cost();
    let mut score = 0.0;

    // Turns before `start_turn` are copied verbatim from `baseline` by
    // `simulate_checkpoint_into`, so their owner delta is exactly zero — start
    // the integral at the first rewritten turn. `start_turn` is clamped the same
    // way the checkpoint clamps it, so turn `h` is never skipped.
    let start = (start_turn.clamp(1, h.max(1) as i64)) as usize;
    for t in start..=h {
        score += w_production
            * production
            * (owner_value(owner_at[t], player) - owner_value(baseline.owner_at[t], player));
    }

    score += w_final_ships
        * (signed_ships(owner_at[h], ships_at[h], player)
            - signed_ships(baseline.owner_at[h], baseline.ships_at[h], player));
    score -= w_ship_cost * ships_committed as f64;

    // ── Neutral-capture discipline (phase 3, see tuning/PHASE3_DESIGN.md) ────
    // Applies only when the target is NEUTRAL at the turn our fleet arrives
    // (`start` = earliest rewritten/arrival turn). Both terms default to no-ops.
    if baseline.owner_at[start] == -1 {
        // (a) Flat marginal-neutral penalty: a fixed shift bites low-score
        // (marginal) neutral grabs hardest; barely dents a high-value capture.
        score -= crate::constants::neutral_capture_penalty();

        // (b) Payback surcharge for slow-to-recoup garrisons, waived when we'd
        // keep a comfortable ship lead after paying for it. The surcharge scales
        // by ships_committed (large for high-garrison planets), so it accelerates
        // with garrison size without needing an explicit exponent.
        let penalty = crate::constants::neutral_payback_penalty();
        if penalty > 0.0 && production > 0.0 {
            let garrison = target.ships.max(0) as f64;
            let payback = garrison / production; // turns of own output to recoup
            let excess = (payback - crate::constants::neutral_payback_turns()).max(0.0);
            let lead_after = world.ship_lead - garrison;
            if excess > 0.0 && lead_after < crate::constants::lead_gate() {
                score -= w_ship_cost * ships_committed as f64 * penalty * excess;
            }
        }
    }
    score
}

// ── evaluate_frontline_strategy ──────────────────────────────────────────

/// A single source's contribution to a winning attack. `effective_offset`
/// is the source's chosen launch delay from the current step — orders with
/// `effective_offset == 0` are emitted as fleet moves this turn, the rest
/// are *reservations* (recorded in `PlanState` so other targets can't grab
/// the ships, but no fleet order is emitted; next bot turn re-plans).
#[derive(Clone, Debug, PartialEq)]
struct PlannedOrder {
    src_id: i64,
    angle: f64,
    ships: i64,
    arrival: i64,          // turns from current step until arrival
    effective_offset: i64, // launch_offset; 0 ⇒ emit this turn
}

/// A winning commitment for a target. Built by `evaluate_frontline_strategy`
/// from one (subset, arrival-schedule) combination.
struct FrontlineWin {
    orders: Vec<PlannedOrder>,
    /// Timeline-delta score of this target commitment relative to baseline.
    score: f64,
}

/// Reusable scratch buffers for [`evaluate_frontline_strategy`], pooled across
/// a whole [`run_strategy`] run (every greedy iteration, target, and offset) so
/// the working set is allocated once, not per call. None of these carry state
/// between calls — each field is cleared (or fully overwritten by
/// `simulate_checkpoint_into`) before use.
#[derive(Default)]
struct FrontlineScratch {
    candidates: Vec<i64>,
    option_table: Vec<Option<SourceOption>>,
    bucket_options: Vec<SourceOption>,
    plan_orders: Vec<PlannedOrder>,
    trial: Vec<ArrivalEvent>,
    fixed_arrivals: Vec<ArrivalEvent>,
    merged_scratch: Vec<ArrivalEvent>,
    owner_buf: Vec<i64>,
    ships_buf: Vec<i64>,
    by_turn_buf: Vec<Vec<ArrivalEvent>>,
    counter: CounterScratch,
}

#[derive(Default)]
struct CounterScratch {
    bucket_options: Vec<SourceOption>,
    trial: Vec<ArrivalEvent>,
    merged_scratch: Vec<ArrivalEvent>,
    owner_buf: Vec<i64>,
    ships_buf: Vec<i64>,
    by_turn_buf: Vec<Vec<ArrivalEvent>>,
}

/// Per-target inputs to the frontline subset search. Source ownership and graph
/// edges are turn-constant within a planning turn, so they are computed once per
/// target and shared across the exact-arrival bucket search.
struct TargetContext {
    /// Inbound sources of the target, distance-sorted (nearest first).
    origins: Vec<(i64, f64)>,
}

fn target_context(model: &HellburnerModel, target: &Planet) -> TargetContext {
    let empty: Vec<(i64, f64)> = Vec::new();
    let mut origins: Vec<(i64, f64)> = model
        .inbound_edges
        .get(&target.id)
        .unwrap_or(&empty)
        .iter()
        .copied()
        .collect();
    origins.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    TargetContext { origins }
}

/// Frontline assembly via exact-arrival buckets. For each candidate source we
/// precompute absolute launch-offset options (`0..=OFFSET_LOOKAHEAD`), then only
/// evaluate attacks whose selected sources arrive on the same turn. Orders with
/// `effective_offset == 0` are emitted this turn; delayed orders are
/// reservations that later greedy choices cannot spend.
fn evaluate_frontline_strategy(
    world: &WorldState,
    model: &HellburnerModel,
    target: &Planet,
    plan: &PlanState,
    ctx: &TargetContext,
    scratch: &mut FrontlineScratch,
) -> Option<FrontlineWin> {
    // ── 1. Per-source absolute launch options. ──
    collect_source_candidates(
        world,
        model,
        target,
        plan,
        ctx,
        world.player,
        &mut scratch.candidates,
        &mut scratch.option_table,
    );
    if scratch.candidates.is_empty() {
        return None;
    }
    let n = scratch.candidates.len();

    // ── 2. Enumerate exact-arrival buckets. ──
    //       Each bucket contains at most one option per source for that arrival
    //       turn, so every simulated attack lands as a coordinated same-turn
    //       force.
    let mut best_score = f64::NEG_INFINITY;
    let mut best_ships = i64::MAX;
    let mut best_orders: Vec<PlannedOrder> = Vec::new();
    let mut best_max_arrival: i64 = 0;

    // ── Shared per-target arrival context (fixed across all masks). ──
    // Every (subset, schedule) trial layers its candidate arrivals on top of
    // the same base (in-flight) + planned (this turn's prior commitments)
    // arrivals. Build that fixed prefix once and `simulate_checkpoint_into` the
    // trial deltas, so the subset enumeration allocates no per-mask timeline.
    let horizon = world.timeline_cache.horizon;
    let base_arrivals = world.timeline_cache.arrivals(target.id);
    let planned: &[ArrivalEvent] = plan
        .planned
        .get(&target.id)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let expiry = world.timeline_cache.expiry(target.id);

    scratch.fixed_arrivals.clear();
    scratch.fixed_arrivals.extend_from_slice(base_arrivals);
    scratch.fixed_arrivals.extend_from_slice(planned);

    // Prefix baseline incorporating base + planned (but not the trial extras).
    // With no planned commitments the cache's pre-built baseline already
    // reflects the base arrivals, so reuse it allocation-free.
    let owned_baseline: Option<PlanetTimeline> = if planned.is_empty() {
        None
    } else {
        Some(simulate_planet_timeline(
            target,
            &scratch.fixed_arrivals,
            world.player,
            horizon,
            expiry,
        ))
    };
    let prefix_baseline: &PlanetTimeline = match &owned_baseline {
        Some(b) => b,
        None => world
            .timeline_cache
            .baseline(target.id)
            .expect("target planet must be in the timeline cache"),
    };
    let enemy_counter_options = if target.owner == -1 {
        collect_enemy_counter_options(world, model, target, plan, ctx)
    } else {
        Vec::new()
    };

    // Layer `trial` on the fixed prefix baseline via `simulate_checkpoint_into`,
    // writing the per-turn owner/ships arrays into the reusable buffers and
    // returning the horizon owner. Equivalent to
    // `final_owner(&target_timeline(world, target.id, trial, plan))`. Buffers are
    // passed in rather than captured so callers can hand over disjoint
    // `FrontlineScratch` fields without a closure-capture borrow conflict.
    let run_trial = |trial: &[ArrivalEvent],
                     fixed_arrivals: &[ArrivalEvent],
                     merged_scratch: &mut Vec<ArrivalEvent>,
                     owner_buf: &mut Vec<i64>,
                     ships_buf: &mut Vec<i64>,
                     by_turn_buf: &mut Vec<Vec<ArrivalEvent>>|
     -> (i64, i64) {
        let start_turn = trial.iter().map(|e| e.turns.max(1)).min().unwrap_or(1);
        merged_scratch.clear();
        merged_scratch.extend_from_slice(fixed_arrivals);
        merged_scratch.extend_from_slice(trial);
        simulate_checkpoint_into(
            target,
            prefix_baseline,
            start_turn,
            merged_scratch.as_slice(),
            expiry,
            owner_buf,
            ships_buf,
            by_turn_buf,
        );
        (owner_buf[horizon as usize], start_turn)
    };

    let consider = |orders: &Vec<PlannedOrder>,
                    max_arrival: i64,
                    ships_total: i64,
                    score: f64,
                    best_score: &mut f64,
                    best_ships: &mut i64,
                    best_orders: &mut Vec<PlannedOrder>,
                    best_max_arrival: &mut i64| {
        if score <= crate::constants::capture_min_score() {
            return;
        }
        let better = score > *best_score
            || (score == *best_score
                && (max_arrival < *best_max_arrival
                    || (max_arrival == *best_max_arrival && ships_total < *best_ships)));

        if better {
            *best_score = score;
            *best_ships = ships_total;
            *best_orders = orders.clone();
            *best_max_arrival = max_arrival;
        }
    };

    for arrival in 1..=horizon {
        collect_bucket_options_for_arrival(
            n,
            &scratch.option_table,
            arrival,
            &mut scratch.bucket_options,
        );

        let bucket_len = scratch.bucket_options.len();
        if bucket_len == 0 {
            continue;
        }

        for mask in 1u32..(1u32 << bucket_len) {
            if mask.count_ones() as usize > world.config.max_sources {
                continue;
            }

            scratch.plan_orders.clear();
            scratch.trial.clear();
            let mut ships_total: i64 = 0;
            for i in 0..bucket_len {
                if mask & (1u32 << i) == 0 {
                    continue;
                }
                let option = scratch.bucket_options[i];
                scratch.plan_orders.push(PlannedOrder {
                    src_id: option.src_id,
                    angle: option.angle,
                    ships: option.ships,
                    arrival: option.arrival,
                    effective_offset: option.launch_offset,
                });
                scratch.trial.push(ArrivalEvent {
                    turns: option.arrival,
                    owner: world.player,
                    ships: option.ships,
                });
                ships_total += option.ships;
            }

            let (final_owner_b, start_turn_b) = run_trial(
                &scratch.trial,
                &scratch.fixed_arrivals,
                &mut scratch.merged_scratch,
                &mut scratch.owner_buf,
                &mut scratch.ships_buf,
                &mut scratch.by_turn_buf,
            );
            if final_owner_b == world.player {
                if target.owner == -1
                    && enemy_can_recapture_after(
                        world,
                        target,
                        &scratch.trial,
                        &scratch.fixed_arrivals,
                        prefix_baseline,
                        expiry,
                        arrival,
                        &enemy_counter_options,
                        &mut scratch.counter,
                    )
                {
                    continue;
                }
                let score_b = timeline_delta_score(
                    world,
                    target,
                    prefix_baseline,
                    &scratch.owner_buf,
                    &scratch.ships_buf,
                    ships_total,
                    start_turn_b,
                );
                consider(
                    &scratch.plan_orders,
                    arrival,
                    ships_total,
                    score_b,
                    &mut best_score,
                    &mut best_ships,
                    &mut best_orders,
                    &mut best_max_arrival,
                );
            }
        }
    }

    if best_orders.is_empty() {
        return None;
    }

    Some(FrontlineWin {
        orders: best_orders,
        score: best_score,
    })
}

#[derive(Clone, Copy)]
struct SourceOption {
    src_id: i64,
    launch_offset: i64,
    angle: f64,
    arrival: i64,
    ships: i64,
}

struct OwnerSourceOptions {
    owner: i64,
    source_count: usize,
    option_table: Vec<Option<SourceOption>>,
}

fn collect_bucket_options_for_arrival(
    source_count: usize,
    option_table: &[Option<SourceOption>],
    arrival: i64,
    out: &mut Vec<SourceOption>,
) {
    out.clear();
    let option_stride = (offset_lookahead() + 1) as usize;
    for i in 0..source_count {
        let row = i * option_stride;
        let mut best_option: Option<SourceOption> = None;
        for l in 0..option_stride {
            let Some(option) = option_table[row + l] else {
                continue;
            };
            if option.arrival != arrival {
                continue;
            }
            let take = match best_option {
                None => true,
                Some(prev) => {
                    option.ships > prev.ships
                        || (option.ships == prev.ships && option.launch_offset < prev.launch_offset)
                }
            };
            if take {
                best_option = Some(option);
            }
        }
        if let Some(option) = best_option {
            out.push(option);
        }
    }
}

fn collect_source_candidates(
    world: &WorldState,
    model: &HellburnerModel,
    target: &Planet,
    plan: &PlanState,
    ctx: &TargetContext,
    owner: i64,
    out: &mut Vec<i64>,
    option_table: &mut Vec<Option<SourceOption>>,
) {
    out.clear();
    option_table.clear();
    let option_stride = (offset_lookahead() + 1) as usize;
    for &(src_id, _travel) in &ctx.origins {
        // Cap the candidate-search width: `origins` is distance-sorted (nearest
        // first), so once we've collected enough viable sources we stop —
        // keeping the soonest-arriving candidates while bounding the `2^n`
        // enumeration (and avoiding the `1u32 << n` overflow for large n).
        if out.len() >= world.config.max_sources_to_consider {
            break;
        }
        let src = *world.planet(src_id);
        if src.owner != owner {
            continue;
        }
        let mut row = Vec::with_capacity(option_stride);
        let mut has_option = false;
        for launch_offset in 0..=offset_lookahead() {
            let ships = plan.ships_available_at(world, &src, owner, launch_offset);
            let option = if ships > 0 {
                model
                    .plan_shot(src_id, target.id, ships, launch_offset)
                    .map(|(angle, turns, _, _, _)| SourceOption {
                        src_id,
                        launch_offset,
                        angle,
                        arrival: (launch_offset + turns).max(1),
                        ships,
                    })
            } else {
                None
            };
            has_option |= option.is_some();
            row.push(option);
        }
        if !has_option {
            continue;
        }
        out.push(src_id);
        option_table.extend(row);
    }
}

fn collect_enemy_counter_options(
    world: &WorldState,
    model: &HellburnerModel,
    target: &Planet,
    plan: &PlanState,
    ctx: &TargetContext,
) -> Vec<OwnerSourceOptions> {
    let mut owners: Vec<i64> = ctx
        .origins
        .iter()
        .map(|(sid, _)| world.planet(*sid).owner)
        .filter(|&owner| owner != -1 && owner != world.player)
        .collect();
    owners.sort_unstable();
    owners.dedup();

    let mut out = Vec::with_capacity(owners.len());
    for owner in owners {
        let mut candidates = Vec::new();
        let mut option_table = Vec::new();
        collect_source_candidates(
            world,
            model,
            target,
            plan,
            ctx,
            owner,
            &mut candidates,
            &mut option_table,
        );
        if !candidates.is_empty() {
            out.push(OwnerSourceOptions {
                owner,
                source_count: candidates.len(),
                option_table,
            });
        }
    }
    out
}

fn enemy_can_recapture_after(
    world: &WorldState,
    target: &Planet,
    friendly_trial: &[ArrivalEvent],
    fixed_arrivals: &[ArrivalEvent],
    prefix_baseline: &PlanetTimeline,
    expiry: Option<i64>,
    our_arrival: i64,
    enemy_options: &[OwnerSourceOptions],
    scratch: &mut CounterScratch,
) -> bool {
    if enemy_options.is_empty() || our_arrival >= prefix_baseline.horizon {
        return false;
    }

    for options in enemy_options {
        for arrival in (our_arrival + 1)..=prefix_baseline.horizon {
            collect_bucket_options_for_arrival(
                options.source_count,
                &options.option_table,
                arrival,
                &mut scratch.bucket_options,
            );
            let bucket_len = scratch.bucket_options.len();
            if bucket_len == 0 {
                continue;
            }

            for mask in 1u32..(1u32 << bucket_len) {
                if mask.count_ones() as usize > world.config.max_sources {
                    continue;
                }

                scratch.trial.clear();
                for i in 0..bucket_len {
                    if mask & (1u32 << i) == 0 {
                        continue;
                    }
                    let option = scratch.bucket_options[i];
                    scratch.trial.push(ArrivalEvent {
                        turns: option.arrival,
                        owner: options.owner,
                        ships: option.ships,
                    });
                }

                let start_turn = friendly_trial
                    .iter()
                    .chain(scratch.trial.iter())
                    .map(|e| e.turns.max(1))
                    .min()
                    .unwrap_or(1);
                scratch.merged_scratch.clear();
                scratch.merged_scratch.extend_from_slice(fixed_arrivals);
                scratch.merged_scratch.extend_from_slice(friendly_trial);
                scratch.merged_scratch.extend_from_slice(&scratch.trial);
                simulate_checkpoint_into(
                    target,
                    prefix_baseline,
                    start_turn,
                    scratch.merged_scratch.as_slice(),
                    expiry,
                    &mut scratch.owner_buf,
                    &mut scratch.ships_buf,
                    &mut scratch.by_turn_buf,
                );

                if scratch.owner_buf[arrival as usize] == options.owner {
                    return true;
                }
            }
        }
    }

    false
}

// ── target evaluation ────────────────────────────────────────────────────

/// Which target each greedy iteration of `run_strategy` should commit. The
/// rollouts in `plan()` try every variant and pick whichever resulting
/// `PlanState` integrates the most own-production over the horizon — so
/// "which sort key is right" is decided empirically per turn, not baked in.
#[derive(Clone, Copy)]
enum SelectionStrategy {
    /// Pure timeline-delta score: production control, final ship delta, and
    /// committed-ship cost relative to the baseline target timeline.
    ScoreFirst,
    /// `score_now / ships_total` — favours efficient captures, freeing
    /// fleet for subsequent iterations.
    ScorePerShip,
    /// Raw `target.production`. Naive but sometimes wins when the
    /// score/urgency machinery picks a small-but-urgent target over a
    /// large-but-relaxed one.
    ProductionFirst,
}

impl SelectionStrategy {
    fn key(self, score: f64, production: i64, ships_total: i64) -> f64 {
        match self {
            SelectionStrategy::ScoreFirst => score,
            SelectionStrategy::ScorePerShip => {
                score / (crate::constants::score_per_ship_smoothing() + ships_total as f64)
            }
            SelectionStrategy::ProductionFirst => production as f64,
        }
    }
}

/// Best `(score, winning commitment)` for a single target, or `None` when the
/// target is already won by baseline+planned commitments or no exact-arrival
/// bucket yields a capture. Strategy-independent: the selection key that ranks
/// targets against each other is applied by the caller ([`run_strategy`]), so
/// this result can be cached and reused across greedy iterations for any target
/// whose plan inputs haven't changed.
fn evaluate_target(
    world: &WorldState,
    model: &HellburnerModel,
    plan: &PlanState,
    target: &Planet,
    scratch: &mut FrontlineScratch,
) -> Option<(f64, FrontlineWin)> {
    // Skip targets already won by baseline + planned commitments. With no
    // planned commitments for this target the prebuilt cache baseline's final
    // owner is identical (same arrivals → same trajectory), so read it
    // allocation-free; only fall back to a full projection when planned
    // commitments exist.
    let planned_here = plan
        .planned
        .get(&target.id)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let already_won = if planned_here.is_empty() {
        baseline_owns(world, target.id)
    } else {
        final_owner(&target_timeline(world, target.id, &[], plan)) == world.player
    };
    if already_won {
        return None;
    }

    // Source/edge inputs shared across the exact-arrival bucket search.
    let ctx = target_context(model, target);

    let win = evaluate_frontline_strategy(world, model, target, plan, &ctx, scratch)?;
    Some((win.score, win))
}

// ── send_reinforcements ──────────────────────────────────────────────────

fn send_reinforcements(
    world: &WorldState,
    model: &HellburnerModel,
    plan: &PlanState,
) -> Vec<MoveAction> {
    let mut out = Vec::new();
    for p in &world.my_planets {
        if !model.non_comet_ids.contains(&p.id) {
            continue;
        }
        // The pressure-BFS target map points to the next owned hop on a route
        // toward a strictly higher-pressure planet. Frontier planets can appear
        // here when they are not themselves the highest-pressure local need.
        let Some(target_id) = model.reinforcement_target.get(&p.id).copied() else {
            continue;
        };
        let available = plan.ships_available(world, p);
        if available <= 0 {
            continue;
        }
        let ships = available;
        let Some((angle, turns_now, _, _, _)) = model.plan_shot(p.id, target_id, ships, 0) else {
            // Blocked now — we can only emit launch-this-turn orders, so nothing
            // to send regardless of how waiting would compare.
            continue;
        };
        let arrival_now = turns_now.max(1);

        // Gate on the planning horizon, matching how combat fleets are bounded.
        // The frontline planner only scores a capture at `owner_buf[horizon]`, so
        // it never proposes a fleet whose ownership flip lands past the horizon (it
        // scores 0). Reinforcement has no such scoring, so without this check a slow
        // shuttle that only arrives after the window the bot can value would still
        // launch — wasting ships that the rollout never sees deliver. `<=` mirrors
        // the inclusive combat boundary (an arrival at exactly `horizon` still
        // flips `owner_buf[horizon]`).
        if arrival_now > world.config.horizon {
            continue;
        }

        // Hold if waiting delivers the fleet no later than launching now.
        // Fleet speed is log-shaped in ship count, so `production·d` extra ships
        // accumulated over `d` turns (and any shifted geometry / cleared blockers
        // at the future launch turn) can speed the fleet enough to offset the
        // launch delay. When that happens, sending now is strictly dominated:
        // same-or-earlier arrival while delivering fewer ships. We re-plan every
        // turn, so this is a per-turn send-vs-hold decision, not a commitment to
        // a specific delay. Replaces the old fixed `REINFORCEMENT_SIZE` floor.
        let wait_is_better = (1..=offset_lookahead()).any(|d| {
            // Use the same forward-min availability model the planner relies on,
            // not raw linear growth (`ships + production·d`): a future enemy
            // arrival can shrink the garrison between now and `d`, so the
            // point-read would over-count ships that won't actually be available
            // and could wrongly decide to hold.
            // let ships_d = plan.ships_available_at(world, p, d);

            // Above change slightly dropped winrate (428 -> 423 wins out of 500 in 4p test)
            // TODO: Theoretically should be better, so investigate further
            let ships_d = ships + p.production * d;
            if ships_d <= 0 {
                return false;
            }
            match model.plan_shot(p.id, target_id, ships_d, d) {
                Some((_, turns_d, _, _, _)) => (d + turns_d).max(1) <= arrival_now,
                None => false,
            }
        });
        if wait_is_better {
            continue;
        }
        out.push(MoveAction {
            from_id: p.id,
            angle,
            ships,
            target: target_id,
        });
    }
    out
}

// ── Public entry ─────────────────────────────────────────────────────────

/// Debug-only reference for [`run_strategy`]'s per-iteration target selection:
/// recomputes *every* candidate from scratch (no persistent cache) and applies
/// the same selection key/tiebreak. Used by a `debug_assert_eq!` in
/// `run_strategy` to prove the incremental dirty-set cache picks exactly what
/// the full recompute would, every iteration. Compiled out of release builds.
#[cfg(debug_assertions)]
fn select_best_uncached(
    world: &WorldState,
    model: &HellburnerModel,
    plan: &PlanState,
    strategy: SelectionStrategy,
    candidate_ids: &[i64],
) -> Option<(i64, Vec<PlannedOrder>)> {
    let mut scratch = FrontlineScratch::default();
    let mut best: Option<(f64, f64, usize, i64, Vec<PlannedOrder>)> = None;
    for &tid in candidate_ids {
        let Some((score, win)) =
            evaluate_target(world, model, plan, world.planet(tid), &mut scratch)
        else {
            continue;
        };
        let production = world.planet(tid).production;
        let ships_total: i64 = win.orders.iter().map(|o| o.ships).sum();
        let primary = strategy.key(score, production, ships_total);
        let better = match &best {
            None => true,
            Some((bp, bs, blen, _, _)) => {
                primary > *bp
                    || (primary == *bp && score > *bs)
                    || (primary == *bp && score == *bs && win.orders.len() < *blen)
            }
        };
        if better {
            best = Some((primary, score, win.orders.len(), tid, win.orders));
        }
    }
    best.map(|(_, _, _, tid, orders)| (tid, orders))
}

/// One full pipeline run under a fixed target-selection strategy, seeded
/// with the opening pre-pass events (empty outside the opening phase).
/// Returns the emitted moves and the resulting PlanState (used by
/// `rollout_score`).
fn run_strategy(
    world: &WorldState,
    model: &HellburnerModel,
    strategy: SelectionStrategy,
    opening: &[OpeningEvent],
) -> (Vec<MoveAction>, PlanState) {
    let mut state = PlanState::default();
    let mut moves: Vec<MoveAction> = Vec::new();

    // Opening pre-pass: commit every DFS capture event before the greedy
    // iterations, exactly like greedy's own orders — offset-0 events are
    // emitted as fleet moves, future offsets stay reservations (spent ships
    // later iterations cannot poach, targets they see as already won),
    // re-planned next turn. Greedy combat, defense, and reinforcement then
    // run on whatever the opening leaves over.
    for ev in opening {
        if ev.offset == 0 {
            // Re-derive the angle (L1-cached: this exact shot was solved during
            // the opening search) *before* committing, so a re-derivation that
            // unexpectedly fails leaves no orphaned reservation (ships reserved
            // but never launched).
            let Some((angle, _, _, _, _)) = model.plan_shot(ev.src, ev.target, ev.ships, 0) else {
                debug_assert!(false, "opening shot not re-derivable at emission");
                continue;
            };
            moves.push(MoveAction {
                from_id: ev.src,
                angle,
                ships: ev.ships,
                target: ev.target,
            });
        }
        state.commit(ev.src, ev.target, ev.ships, ev.arrival, world.player);
    }

    // Fixed-order candidate targets (non-comet, not pressure-gated, with
    // inbound edges). The scan order matches the original per-iteration sweep
    // so selection tie-breaking stays deterministic and identical to the
    // uncached path.
    let candidate_ids: Vec<i64> = world
        .planets
        .iter()
        .filter(|p| model.non_comet_ids.contains(&p.id))
        .filter(|p| !model.pressure_gated_targets.contains(&p.id))
        .filter(|p| {
            model
                .inbound_edges
                .get(&p.id)
                .map(|v| !v.is_empty())
                .unwrap_or(false)
        })
        .map(|p| p.id)
        .collect();

    // Per-target evaluation cache, persisted across greedy iterations. Each
    // iteration recomputes only the targets whose inputs the previous commit
    // could have changed (`dirty`); the rest reuse their cached eval.
    //
    // `evaluate_target(T)` reads `plan` only through (a) `planned[T]` (its own
    // prefix baseline / already-won check) and (b) the `spent`/`planned` state
    // of T's inbound sources for availability. Committing target `C` from
    // sources `S` mutates `spent[s]` for `s ∈ S` and `planned[C]`, so the
    // exactly-affected set is
    // `{C} ∪ outbound(C) ∪ ⋃_{s∈S} outbound(s)` — every target fed by a touched
    // source, plus targets for which `C` is itself a source. Over-invalidation
    // would only cost time; this set is exact.
    let mut cache: HashMap<i64, Option<(f64, FrontlineWin)>> =
        HashMap::with_capacity_and_hasher(candidate_ids.len(), Default::default());
    let mut dirty: HashSet<i64> = candidate_ids.iter().copied().collect();
    let mut scratch = FrontlineScratch::default();

    // Each iteration commits ≥1 ship from at least one source, so the loop is
    // bounded by the total source pool. A fixed safety cap protects against any
    // pathological selector that re-picks the same target with a vanishing
    // commitment.
    for _ in 0..256 {
        // Recompute dirty targets against the current plan.
        if !dirty.is_empty() {
            for &tid in &dirty {
                let eval = evaluate_target(world, model, &state, world.planet(tid), &mut scratch);
                cache.insert(tid, eval);
            }
            dirty.clear();
        }

        // Select the best target under `strategy`, scanning candidates in fixed
        // order so ties resolve exactly as the uncached path did.
        let mut best: Option<(f64, f64, usize, i64)> = None; // (primary, score, len, tid)
        for &tid in &candidate_ids {
            let Some(Some((score, win))) = cache.get(&tid) else {
                continue;
            };
            let production = world.planet(tid).production;
            let ships_total: i64 = win.orders.iter().map(|o| o.ships).sum();
            let primary = strategy.key(*score, production, ships_total);
            let better = match &best {
                None => true,
                Some((bp, bs, blen, _)) => {
                    primary > *bp
                        || (primary == *bp && *score > *bs)
                        || (primary == *bp && *score == *bs && win.orders.len() < *blen)
                }
            };
            if better {
                best = Some((primary, *score, win.orders.len(), tid));
            }
        }
        // Debug-only: prove the incremental cache selected exactly what a full
        // from-scratch recompute would, against the current plan.
        #[cfg(debug_assertions)]
        let reference_pick = select_best_uncached(world, model, &state, strategy, &candidate_ids);

        let Some((_, _, _, target_id)) = best else {
            #[cfg(debug_assertions)]
            debug_assert!(
                reference_pick.is_none(),
                "run_strategy cache returned no target but uncached recompute found one"
            );
            break;
        };

        // Clone the winning orders out of the cache for committing.
        let orders: Vec<PlannedOrder> = match cache.get(&target_id) {
            Some(Some((_, win))) => win.orders.clone(),
            _ => break,
        };
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            reference_pick,
            Some((target_id, orders.clone())),
            "run_strategy cache diverged from uncached recompute"
        );
        if orders.is_empty() {
            break;
        }

        // Commit every order to PlanState (reservation), but only emit a fleet
        // move when the order launches this turn. Reservations make subsequent
        // iterations see the reserved ships as spent, so they can't be poached
        // for lower-value captures — the no-op deferral emerges from the score
        // sweep. Mark every target fed by a touched source dirty for recompute.
        for o in &orders {
            state.commit(o.src_id, target_id, o.ships, o.arrival, world.player);
            if o.effective_offset == 0 {
                moves.push(MoveAction {
                    from_id: o.src_id,
                    angle: o.angle,
                    ships: o.ships,
                    target: target_id,
                });
            }
            if let Some(outs) = model.outbound_edges.get(&o.src_id) {
                for (did, _) in outs {
                    dirty.insert(*did);
                }
            }
        }
        dirty.insert(target_id);
        if let Some(outs) = model.outbound_edges.get(&target_id) {
            for (did, _) in outs {
                dirty.insert(*did);
            }
        }
    }

    moves.extend(send_reinforcements(world, model, &state));
    (moves, state)
}

/// Strategies tried by `search_candidates`. The first entry is the one
/// `plan()` runs directly (used as the cheap reply-policy hook inside the
/// rollout layer), so its position is load-bearing — see the
/// `search_candidates_includes_greedy_plan` test.
///
/// The reply policy (index 0) is the tunable `default_strategy`: 0 ⇒
/// ScorePerShip (the original default), 1 ⇒ ScoreFirst. ProductionFirst stays
/// in the search set as a rollout candidate but is never the default. The full
/// set of plans is unchanged — only the order is — so `search_candidates`
/// behavior is unaffected; only `plan()`'s direct policy moves.
fn strategies() -> [SelectionStrategy; 3] {
    match crate::constants::default_strategy() {
        1 => [
            SelectionStrategy::ScoreFirst,
            SelectionStrategy::ScorePerShip,
            SelectionStrategy::ProductionFirst,
        ],
        _ => [
            SelectionStrategy::ScorePerShip,
            SelectionStrategy::ProductionFirst,
            SelectionStrategy::ScoreFirst,
        ],
    }
}

pub fn plan(world: &WorldState) -> Vec<MoveAction> {
    if world.enemy_planets.is_empty() {
        return Vec::new();
    }
    let model = HellburnerModel::build(world);

    // Opening expansion pre-pass (empty past the phase, inside rollouts, or
    // when no positive-value capture plan exists), committed ahead of the
    // greedy iterations inside `run_strategy`.
    let opening = crate::early_game::plan_opening(&model);

    // Single greedy run under the default strategy. This is the policy hook
    // the rollout layer (see `crate::rollout`) invokes for opponent replies
    // *and* our own replanning during the reactive phase, so it must stay
    // cheap and deterministic. Multi-strategy search happens one level up,
    // in `search_candidates`.
    let (moves, _) = run_strategy(world, &model, strategies()[0], &opening);
    moves
}

/// Final post-processing applied to the chosen move set *after* any rollout
/// selection — never inside the rollout itself, so the policy the rollout scores
/// is untouched. This only rewrites the moves we are about to return, and it is a
/// strict no-loss reroute: for each launch-this-turn fleet `A → B`, if routing
/// through an intermediate planet `C` reaches `B` in the same number of turns or
/// fewer (`A → C → B`, measured via `plan_shot`), we retarget the fleet to `C`.
/// We re-plan every turn, so the strategy naturally decides what to do with the
/// ships once they arrive at `C` — no state is carried across turns.
///
/// `C` must be projected to be owned by us on the turn the fleet *arrives* there
/// (so the ships reinforce rather than fight), accounting for the arrivals of
/// every other move in this same final plan.
pub fn redirect_moves(world: &WorldState, mut moves: Vec<MoveAction>) -> Vec<MoveAction> {
    if moves.is_empty() {
        return moves;
    }
    let model = HellburnerModel::build(world);
    let player = world.player;
    let horizon = world.timeline_cache.horizon;

    // Direct A→B travel time per move, plus a PlanState capturing every move's
    // friendly arrival so each candidate C's ownership check sees the full plan.
    let mut plan = PlanState::default();
    let mut direct: Vec<Option<i64>> = Vec::with_capacity(moves.len());
    for mv in &moves {
        if mv.target < 0 || !model.non_comet_ids.contains(&mv.target) {
            direct.push(None);
            continue;
        }
        match model.plan_shot(mv.from_id, mv.target, mv.ships, 0) {
            Some((_, turns_ab, _, _, _)) => {
                plan.commit(mv.from_id, mv.target, mv.ships, turns_ab, player);
                direct.push(Some(turns_ab));
            }
            None => direct.push(None),
        }
    }

    for (i, mv) in moves.iter_mut().enumerate() {
        let Some(turns_ab) = direct[i] else {
            continue;
        };
        let a = mv.from_id;
        let b = mv.target;
        let ships = mv.ships;
        // (total_turns, turns_ac, ships_c, c_id, angle_ac). Selection order:
        // minimize total turns; then the one holding the most ships when the fleet
        // arrives; then prefer the intermediate closest to A (smallest turns_ac);
        // finally smallest id so the pick stays deterministic regardless
        // of HashSet iteration order.
        let mut best: Option<(i64, i64, i64, i64, f64)> = None;
        for &c in &model.non_comet_ids {
            if c == a || c == b {
                continue;
            }
            let Some((angle_ac, turns_ac, _, _, _)) = model.plan_shot(a, c, ships, 0) else {
                continue;
            };
            // A→C alone must be shorter than A→B (C→B costs ≥ 1 turn), and the
            // arrival turn must fall inside the timeline horizon to check it.
            if turns_ac >= turns_ab || turns_ac < 0 || turns_ac > horizon {
                continue;
            }
            // C must be friendly when the fleet arrives, given the full plan.
            let tl = target_timeline(world, c, &[], &plan);
            if tl.owner_at[turns_ac as usize] != player {
                continue;
            }
            // C→B is launched at the arrival turn (offset = turns_ac), so its
            // geometry/obstacles are evaluated at that future turn.
            let Some((_, turns_cb, _, _, _)) = model.plan_shot(c, b, ships, turns_ac) else {
                continue;
            };
            let total = turns_ac + turns_cb;
            if total <= turns_ab {
                let ships_c = tl.ships_at[turns_ac as usize];
                // Lexicographic minimize on (total, -ships_c, turns_ac, c): the
                // negated ship count makes "most ships on arrival" rank first.
                let take = match best {
                    None => true,
                    Some((bt, b_tac, b_ships, bc, _)) => {
                        (total, -ships_c, turns_ac, c) < (bt, -b_ships, b_tac, bc)
                    }
                };
                if take {
                    best = Some((total, turns_ac, ships_c, c, angle_ac));
                }
            }
        }
        if let Some((_, turns_ac, _, c, angle_ac)) = best {
            // Keep the ledger consistent for later moves' ownership checks: the
            // fleet now lands at C (turns_ac), not B (turns_ab).
            plan.reroute(b, turns_ab, c, turns_ac, ships, player);
            mv.angle = angle_ac;
            mv.target = c;
        }
    }
    moves
}

/// Produces one candidate plan per strategy in `STRATEGIES`. The rollout
/// layer (`pick_plan_by_rollout`) scores each via a real engine simulation
/// — including opponent replanning and a zero-sum end-state delta — and
/// picks the best. That's strictly more informative than scoring strategies
/// against an own-side-only projection here, so we don't pre-rank.
///
/// Duplicate plans are deduplicated so the rollout doesn't pay for the same
/// move set twice (different strategies often converge on the same answer
/// once trial-timeline ownership is the binding constraint).
pub fn search_candidates(world: &WorldState) -> Vec<Vec<MoveAction>> {
    if world.enemy_planets.is_empty() {
        return vec![Vec::new()];
    }
    let model = HellburnerModel::build(world);
    // Opening pre-pass, computed once and shared by every strategy run so
    // the greedy plan (`STRATEGIES[0]` + opening) is always candidate 0.
    let opening = crate::early_game::plan_opening(&model);

    // Stress test: probe `plan_shot` for every ordered pair of planets
    // (both directions) with fleet sizes up to 50. Results are discarded —
    // this just exercises the function. The `std::hint::black_box` keeps the
    // optimizer from eliding the calls.

    // for i in 0..50 {
    //     for src in &world.planets {
    //         for dst in &world.planets {
    //             if src.id == dst.id {
    //                 continue;
    //             }
    //             std::hint::black_box(model.plan_shot(src.id, dst.id, i, 0));
    //         }
    //     }
    // }

    let strats = strategies();
    let mut out: Vec<Vec<MoveAction>> = Vec::with_capacity(strats.len() + 1);

    for &strat in &strats {
        let (moves, _) = run_strategy(world, &model, strat, &opening);
        if !out.iter().any(|prev| prev == &moves) {
            out.push(moves);
        }
    }
    // The opening was planned with no enemy model; offer the rollout minimax
    // a no-opening alternative so a bad opening can be rejected wholesale.
    if !opening.is_empty() {
        let (moves, _) = run_strategy(world, &model, strats[0], &[]);
        if !out.iter().any(|prev| prev == &moves) {
            out.push(moves);
        }
    }
    out
}
