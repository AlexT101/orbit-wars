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
//!   * `reinforcement_target` per owned planet (frontline BFS).
//!   * Per-turn `PlanState` (spent ships + planned commitments).

#![allow(dead_code)]

use std::cell::RefCell;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::cache::{AimCacheVerdict, InvariantVerdict};
use crate::constants::{OFFSET_LOOKAHEAD, ROTATION_LOOK_AHEAD_TURNS};
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
                .position(p.id, 1 + ROTATION_LOOK_AHEAD_TURNS)
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

        let reinforcement_target = build_reinforcement_targets(
            state,
            &non_comet_ids,
            &inbound_edges,
            &outbound_edges,
            player,
        );

        Self {
            state,
            non_comet_ids,
            inbound_edges,
            outbound_edges,
            reinforcement_target,
            shot_cache: RefCell::new(HashMap::default()),
        }
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
    non_comet_ids: &HashSet<i64>,
    inbound: &HashMap<i64, Vec<(i64, f64)>>,
    outbound: &HashMap<i64, Vec<(i64, f64)>>,
    player: i64,
) -> HashMap<i64, i64> {
    let mut front_line: HashSet<i64> = HashSet::default();
    for p in &state.my_planets {
        if !non_comet_ids.contains(&p.id) {
            continue;
        }
        let pid = p.id;
        let has_outsider = inbound[&pid]
            .iter()
            .any(|(sid, _)| state.planet(*sid).owner != player)
            || outbound[&pid]
                .iter()
                .any(|(did, _)| state.planet(*did).owner != player);
        if has_outsider {
            front_line.insert(pid);
        }
    }

    // BFS hop-distance back through owned-planet edges; frontline are sinks.
    let mut hops: HashMap<i64, i64> = HashMap::default();
    let mut queue: Vec<i64> = Vec::new();
    for &fid in &front_line {
        hops.insert(fid, 0);
        queue.push(fid);
    }
    let mut head = 0;
    while head < queue.len() {
        let node = queue[head];
        head += 1;
        let dh = hops[&node];
        for (sid, _) in &inbound[&node] {
            if state.planet(*sid).owner != player || hops.contains_key(sid) {
                continue;
            }
            hops.insert(*sid, dh + 1);
            queue.push(*sid);
        }
    }

    let mut out: HashMap<i64, i64> = HashMap::default();
    for p in &state.my_planets {
        if !non_comet_ids.contains(&p.id) || front_line.contains(&p.id) {
            continue;
        }
        let mut direct: Vec<i64> = outbound[&p.id]
            .iter()
            .filter_map(|(did, _)| {
                if front_line.contains(did) {
                    Some(*did)
                } else {
                    None
                }
            })
            .collect();
        if !direct.is_empty() {
            direct.sort_by_key(|d| state.planet(*d).ships);
            out.insert(p.id, direct[0]);
            continue;
        }
        let mut reachable: Vec<i64> = outbound[&p.id]
            .iter()
            .filter_map(|(did, _)| {
                if state.planet(*did).owner == player
                    && !front_line.contains(did)
                    && hops.contains_key(did)
                {
                    Some(*did)
                } else {
                    None
                }
            })
            .collect();
        if reachable.is_empty() {
            continue;
        }
        reachable.sort_by_key(|d| (hops[d], state.planet(*d).ships));
        out.insert(p.id, reachable[0]);
    }
    out
}

// ── PlanState: turn-local commitments ────────────────────────────────────

#[derive(Default)]
struct PlanState {
    spent: HashMap<i64, i64>,
    planned: HashMap<i64, Vec<ArrivalEvent>>,
}

impl PlanState {
    fn ships_available(&self, world: &WorldState, src: &Planet) -> i64 {
        self.ships_available_at(world, src, 0)
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
    /// O(1) in the common case — it reads the prebuilt baseline trajectory's
    /// precomputed forward-min. Only when this source has its own planned
    /// reinforcements queued *this* turn does it pay a single per-call planet
    /// sim to fold those in. Conservative against `spent`: every prior
    /// commitment from this source is subtracted regardless of when those ships
    /// are scheduled to leave (so a commit at any offset correctly debits all
    /// offsets, keeping the greedy planner from going negative).
    fn ships_available_at(&self, world: &WorldState, src: &Planet, offset: i64) -> i64 {
        let offset = offset.max(0);
        let spent = self.spent.get(&src.id).copied().unwrap_or(0);
        let planned: &[ArrivalEvent] = self
            .planned
            .get(&src.id)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let available = if planned.is_empty() {
            match world.timeline_cache.baseline(src.id) {
                Some(b) => available_at_timeline(b, offset),
                // No cached trajectory: fall back to linear growth. A purely
                // growing series has its minimum at `offset`, so the point
                // value is already the forward-min.
                None if src.owner == world.player => src.ships + src.production * offset,
                None => 0,
            }
        } else {
            // This source also has reinforcements we've planned this turn.
            // Sim the full horizon (not just up to `offset`) so the forward-min
            // sees every later turn.
            let tl = world.projected_timeline(src.id, world.timeline_cache.horizon, planned, &[]);
            available_at_timeline(&tl, offset)
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
        -1.0
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
    let mut score = 0.0;

    // Turns before `start_turn` are copied verbatim from `baseline` by
    // `simulate_checkpoint_into`, so their owner delta is exactly zero — start
    // the integral at the first rewritten turn. `start_turn` is clamped the same
    // way the checkpoint clamps it, so turn `h` is never skipped.
    let start = (start_turn.clamp(1, h.max(1) as i64)) as usize;
    for t in start..=h {
        score += production
            * (owner_value(owner_at[t], player) - owner_value(baseline.owner_at[t], player));
    }

    score += signed_ships(owner_at[h], ships_at[h], player)
        - signed_ships(baseline.owner_at[h], baseline.ships_at[h], player);
    score - ships_committed as f64
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
}

/// Per-target inputs to the frontline subset search. Source ownership and graph
/// edges are turn-constant within a planning turn, so they are computed once per
/// target and shared across the exact-arrival bucket search.
struct TargetContext {
    /// Owned inbound sources of the target, distance-sorted (nearest first).
    origins: Vec<(i64, f64)>,
}

fn target_context(world: &WorldState, model: &HellburnerModel, target: &Planet) -> TargetContext {
    let empty: Vec<(i64, f64)> = Vec::new();
    let mut origins: Vec<(i64, f64)> = model
        .inbound_edges
        .get(&target.id)
        .unwrap_or(&empty)
        .iter()
        .filter(|(sid, _)| world.planet(*sid).owner == world.player)
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
        &mut scratch.candidates,
        &mut scratch.option_table,
    );
    if scratch.candidates.is_empty() {
        return None;
    }
    let n = scratch.candidates.len();
    let option_stride = (OFFSET_LOOKAHEAD + 1) as usize;

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
        if score <= 0.0 {
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
        scratch.bucket_options.clear();
        for i in 0..n {
            let row = i * option_stride;
            let mut best_option: Option<SourceOption> = None;
            for l in 0..option_stride {
                let Some(option) = scratch.option_table[row + l] else {
                    continue;
                };
                if option.arrival != arrival {
                    continue;
                }
                let take = match best_option {
                    None => true,
                    Some(prev) => {
                        option.ships > prev.ships
                            || (option.ships == prev.ships
                                && option.launch_offset < prev.launch_offset)
                    }
                };
                if take {
                    best_option = Some(option);
                }
            }
            if let Some(option) = best_option {
                scratch.bucket_options.push(option);
            }
        }

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

fn collect_source_candidates(
    world: &WorldState,
    model: &HellburnerModel,
    target: &Planet,
    plan: &PlanState,
    ctx: &TargetContext,
    out: &mut Vec<i64>,
    option_table: &mut Vec<Option<SourceOption>>,
) {
    out.clear();
    option_table.clear();
    let option_stride = (OFFSET_LOOKAHEAD + 1) as usize;
    for &(src_id, _travel) in &ctx.origins {
        // Cap the candidate-search width: `origins` is distance-sorted (nearest
        // first), so once we've collected enough viable sources we stop —
        // keeping the soonest-arriving candidates while bounding the `2^n`
        // enumeration (and avoiding the `1u32 << n` overflow for large n).
        if out.len() >= world.config.max_sources_to_consider {
            break;
        }
        let src = *world.planet(src_id);
        let mut row = Vec::with_capacity(option_stride);
        let mut has_option = false;
        for launch_offset in 0..=OFFSET_LOOKAHEAD {
            let ships = plan.ships_available_at(world, &src, launch_offset);
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
            SelectionStrategy::ScorePerShip => score / (1.0 + ships_total as f64),
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
    let ctx = target_context(world, model, target);

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
        // Only non-frontline planets get a reinforcement target (frontline
        // planets — those with a non-player neighbor — are excluded when the
        // target map is built), so a source here never has enemy graph edges.
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
        let wait_is_better = (1..=OFFSET_LOOKAHEAD).any(|d| {
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

/// One full pipeline run under a fixed target-selection strategy. Returns
/// the emitted moves and the resulting PlanState (used by `rollout_score`).
fn run_strategy(
    world: &WorldState,
    model: &HellburnerModel,
    strategy: SelectionStrategy,
) -> (Vec<MoveAction>, PlanState) {
    let mut state = PlanState::default();
    let mut moves: Vec<MoveAction> = Vec::new();

    // Fixed-order candidate targets (non-comet, with inbound edges). The scan
    // order matches the original per-iteration sweep so selection tie-breaking
    // stays deterministic and identical to the uncached path.
    let candidate_ids: Vec<i64> = world
        .planets
        .iter()
        .filter(|p| model.non_comet_ids.contains(&p.id))
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
const STRATEGIES: [SelectionStrategy; 3] = [
    SelectionStrategy::ScorePerShip,
    SelectionStrategy::ProductionFirst,
    SelectionStrategy::ScoreFirst,
];

pub fn plan(world: &WorldState) -> Vec<MoveAction> {
    if world.enemy_planets.is_empty() {
        return Vec::new();
    }
    let model = HellburnerModel::build(world);

    // Single greedy run under the default strategy. This is the policy hook
    // the rollout layer (see `crate::rollout`) invokes for opponent replies
    // *and* our own replanning during the reactive phase, so it must stay
    // cheap and deterministic. Multi-strategy search happens one level up,
    // in `search_candidates`.
    let (moves, _) = run_strategy(world, &model, STRATEGIES[0]);
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

    let mut out: Vec<Vec<MoveAction>> = Vec::with_capacity(STRATEGIES.len());

    for &strat in &STRATEGIES {
        let (moves, _) = run_strategy(world, &model, strat);
        if !out.iter().any(|prev| prev == &moves) {
            out.push(moves);
        }
    }
    out
}
