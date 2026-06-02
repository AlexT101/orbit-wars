//! Port of the open-source `hellburner` strategy
//! ([bots/external/hellburner/main.py](../../external/hellburner/main.py)).
//!
//! Reuses our Rust infra:
//!   * [`crate::apollo::helpers::aim_with_prediction`] — combines hellburner's
//!     `intercept_planet` + `first_planet_hit` (returns Some only when the
//!     shot reaches the target unblocked by sun/planet/comet).
//!   * [`crate::apollo::world::WorldState`] — per-turn snapshot incl. `TimelineCache`
//!     which already plays the role of hellburner's `destination_list`.
//!   * [`crate::apollo::helpers::simulate_planet_timeline`] /
//!     [`WorldState::projected_timeline`] — hellburner's `simulate_planet_timeline`.
//!
//! Hellburner-specific data we build here:
//!   * Proximity graph (`MAX_DISTANCE=38`, `ROTATION_LOOK_AHEAD=10`).
//!   * `reinforcement_target` per owned planet (frontline BFS).
//!   * Per-turn `PlanState` (spent ships + planned commitments).

#![allow(dead_code)]

use std::cell::RefCell;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::apollo::constants::HORIZON;
use crate::apollo::engine::Planet;
use crate::apollo::entity_cache::{AimCacheVerdict};
use crate::apollo::helpers::{
    aim_with_prediction, dist, simulate_planet_timeline, AimResult, ArrivalEvent,
    PlanetTimeline,
};
use crate::apollo::world::{merge_arrivals, WorldState};

// ── Constants ────────────────────────────────────────────────────────────
const EARLY_ROUNDS: i64 = 3;
const MAX_DISTANCE: f64 = 38.0;
const ROTATION_LOOK_AHEAD: i64 = 10;
const REINFORCEMENT_SIZE: i64 = 17;
const GARRISON_SIZE: i64 = 11;
const SECOND_ENEMY_ARRIVAL_TOL: i64 = 1;
const TRIM_MIN_SHIPS: i64 = 10;
/// How many turns of delayed-launch we sweep per target when computing the
/// urgency of acting **this** turn. The δ=0 entry decides what we actually
/// commit; δ>0 entries only feed the priority calculation.
const OFFSET_LOOKAHEAD: i64 = 5;
/// Cap on inbound owned sources we enumerate for full 2^N subset search.
/// Beyond this the nearest `MAX_SUBSET_SOURCES` are kept (sources are
/// already distance-ordered by `inbound_edges`) — in practice maps in this
/// game rarely have more than a handful of inbound owned neighbors.
const MAX_SUBSET_SOURCES: usize = 10;
/// Max extra launch delay (beyond the subset's base offset) a single source
/// will accept when coordinating arrivals to land on the same turn as the
/// subset's latest-arriving source. Per-source scan, cache-friendly.
const MAX_COORD_DELAY: i64 = 5;
/// How many turns past the natural max-arrival the coordinated schedule will
/// push the cluster. Lets slow-growing sources accumulate `production·d`
/// extra ships at the cost of arriving later — a richer brute-force sweep
/// that complements the `MAX_COORD_DELAY` per-source delay budget.
const A_S_LOOKAHEAD: i64 = 3;

type FleetOrder = (i64, f64, i64); // (src_id, angle, ships)

pub struct HellburnerModel<'a> {
    pub state: &'a WorldState<'a>,
    /// Planet ids excluding comets — comets never appear in the proximity
    /// graph, reinforcement BFS, or target loops.
    pub non_comet_ids: HashSet<i64>,
    pub future_pos: HashMap<i64, [f64; 2]>,
    pub inbound_edges: HashMap<i64, Vec<(i64, f64)>>,
    pub outbound_edges: HashMap<i64, Vec<(i64, f64)>>,
    pub reinforcement_target: HashMap<i64, i64>,
    /// L1 hot cache for `plan_shot`: per-`HellburnerModel` (i.e. one bot turn)
    /// memoization of `(src, target, ships, launch_turn_offset) → aim`.
    /// Avoids repeated traffic to the L2 `EntityCache::aim_cache` inside the
    /// inner loops of `evaluate_frontline_strategy`, `evaluate_move_orders`
    /// (where the same shot can be re-queried several times across the main
    /// loop and the worst-case sub-rollout), and `run_early_game`'s DFS
    /// (which probes both `offset == 0` shots and delayed-launch shots).
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

        let mut future_pos: HashMap<i64, [f64; 2]> = HashMap::default();
        for p in &state.planets {
            if !non_comet_ids.contains(&p.id) {
                continue;
            }
            let pos = state
                .entity_cache
                .position(p.id, 1 + ROTATION_LOOK_AHEAD)
                .unwrap_or([p.x, p.y]);
            future_pos.insert(p.id, pos);
        }

        let mut inbound_edges: HashMap<i64, Vec<(i64, f64)>> = HashMap::default();
        let mut outbound_edges: HashMap<i64, Vec<(i64, f64)>> = HashMap::default();
        for &pid in &non_comet_ids {
            inbound_edges.insert(pid, Vec::new());
            outbound_edges.insert(pid, Vec::new());
        }
        for src in &state.planets {
            if !non_comet_ids.contains(&src.id) {
                continue;
            }
            for dst in &state.planets {
                if dst.id == src.id || !non_comet_ids.contains(&dst.id) {
                    continue;
                }
                let [fx, fy] = future_pos[&dst.id];
                let travel = dist(src.x, src.y, fx, fy);
                if travel <= MAX_DISTANCE {
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

        let reinforcement_target =
            build_reinforcement_targets(state, &non_comet_ids, &inbound_edges, &outbound_edges, player);

        Self {
            state,
            non_comet_ids,
            future_pos,
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
        let key = (src_id, target_id, ships, launch_turn_offset);
        if let Some(&cached) = self.shot_cache.borrow().get(&key) {
            return cached;
        }
        let cache = self.state.entity_cache;
        let result = match cache.aim_cache_lookup(src_id, target_id, ships, launch_turn_offset) {
            AimCacheVerdict::Hit(r) => r,
            AimCacheVerdict::Miss | AimCacheVerdict::Stale => {
                let r = aim_with_prediction(cache, src_id, target_id, ships, launch_turn_offset);
                cache.aim_cache_store(src_id, target_id, ships, launch_turn_offset, r);
                r
            }
        };
        self.shot_cache.borrow_mut().insert(key, result);
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
    let owned_ids: HashSet<i64> = state
        .my_planets
        .iter()
        .filter(|p| non_comet_ids.contains(&p.id))
        .map(|p| p.id)
        .collect();

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
            if !owned_ids.contains(sid) || hops.contains_key(sid) {
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
                if owned_ids.contains(did)
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
    fn ships_available(&self, src: &Planet) -> i64 {
        (src.ships - self.spent.get(&src.id).copied().unwrap_or(0)).max(0)
    }
    /// Growth-aware available ships at a future launch offset. Counts the
    /// production a source will accumulate over `offset` turns on top of its
    /// current pool. Conservative against `spent`: every prior commitment is
    /// subtracted regardless of when those ships are scheduled to leave.
    fn ships_available_at(&self, src: &Planet, offset: i64) -> i64 {
        let spent = self.spent.get(&src.id).copied().unwrap_or(0);
        (src.ships - spent + src.production * offset.max(0)).max(0)
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

// ── Worst-case exposure check ────────────────────────────────────────────

/// (holds_under_worst_case, total_half_pressure)
fn neighbor_holds_under_worst_case(
    world: &WorldState,
    model: &HellburnerModel,
    neighbor: &Planet,
    plan: &PlanState,
) -> (bool, i64) {
    let mut extras: Vec<ArrivalEvent> = Vec::new();
    let mut half_pressure: i64 = 0;
    let empty: Vec<(i64, f64)> = Vec::new();
    let inbound = model.inbound_edges.get(&neighbor.id).unwrap_or(&empty);
    for (attacker_id, _) in inbound {
        let attacker = world.planet(*attacker_id);
        if attacker.owner == world.player || attacker.owner == -1 || attacker.ships == 0 {
            continue;
        }
        let half_ships = (attacker.ships / 2).max(1);
        let Some((_, turns, _, _, _)) =
            model.plan_shot(*attacker_id, neighbor.id, attacker.ships, 0)
        else {
            continue;
        };
        extras.push(ArrivalEvent {
            turns: turns.max(1),
            owner: attacker.owner,
            ships: half_ships,
        });
        half_pressure += half_ships;
    }
    if extras.is_empty() {
        return (true, 0);
    }
    let mut neighbor_zero = neighbor.clone();
    neighbor_zero.ships = 0;
    let planned: &[ArrivalEvent] = plan
        .planned
        .get(&neighbor.id)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let merged = merge_arrivals(
        world.timeline_cache.arrivals(neighbor.id),
        planned,
        &extras,
        world.timeline_cache.horizon,
    );
    let expiry = world.timeline_cache.expiry(neighbor.id);
    let tl = simulate_planet_timeline(
        &neighbor_zero,
        &merged,
        world.player,
        world.timeline_cache.horizon,
        expiry,
    );
    (final_owner(&tl) == world.player, half_pressure)
}

// ── Unified zero-sum scoring ─────────────────────────────────────────────

/// Captured production weighting: enemy targets count double (we gain the
/// production *and* they lose it — true zero-sum delta), neutral and
/// own-planet defense are 1×.
fn zero_sum_mult(world: &WorldState, target: &Planet) -> f64 {
    if target.owner != world.player && target.owner != -1 {
        2.0
    } else {
        1.0
    }
}

/// Score of capturing/holding `target` with last arrival at `arrival_turn`
/// (relative to current step). The integral `prod × remaining` automatically
/// trades arrival-time against production: waiting 2 turns for a 5-prod
/// target beats grabbing a 2-prod target now once `5(H−A−2) > 2(H−A)`.
fn score_capture(world: &WorldState, target: &Planet, arrival_turn: i64) -> f64 {
    let h = world.timeline_cache.horizon;
    let remaining = (h - arrival_turn).max(0) as f64;
    target.production as f64 * remaining * zero_sum_mult(world, target)
}

// ── evaluate_frontline_strategy ──────────────────────────────────────────

/// A single source's contribution to a winning attack. `effective_offset`
/// is the source's chosen launch delay from the current step — orders with
/// `effective_offset == 0` are emitted as fleet moves this turn, the rest
/// are *reservations* (recorded in `PlanState` so other targets can't grab
/// the ships, but no fleet order is emitted; next bot turn re-plans).
#[derive(Clone)]
struct PlannedOrder {
    src_id: i64,
    angle: f64,
    ships: i64,
    arrival: i64,           // turns from current step until arrival
    effective_offset: i64,  // launch_offset; 0 ⇒ emit this turn
}

/// A winning commitment for a target. Built by `evaluate_frontline_strategy`
/// from one (subset, arrival-schedule) combination.
struct FrontlineWin {
    orders: Vec<PlannedOrder>,
    /// Latest arrival turn among `orders`, relative to current step.
    max_arrival: i64,
}

/// Offset-aware frontline assembly. `offset == 0` is "launch this turn" —
/// those orders are what `plan` actually emits. For `offset > 0`,
/// source/target/obstacle positions are evaluated at the future launch turn
/// via `plan_shot(..., offset)` and arrival times in the trial timeline are
/// shifted by `offset` so the ownership check stays correct.
///
/// Source ship-availability and the worst-case defense reservation are
/// computed at *current* state regardless of `offset` — a conservative
/// approximation that keeps urgency comparisons apples-to-apples (the
/// δ-sweep is about geometry shifts, not production growth).
fn evaluate_frontline_strategy(
    world: &WorldState,
    model: &HellburnerModel,
    target: &Planet,
    plan: &PlanState,
    offset: i64,
) -> Option<FrontlineWin> {
    // ── 1. Per-source candidate baseline (all 2^N subsets share these). ──
    let candidates = collect_source_candidates(world, model, target, plan, offset);
    if candidates.is_empty() {
        return None;
    }
    let n = candidates.len().min(MAX_SUBSET_SOURCES);

    // ── 2. Enumerate non-empty subsets × {uncoordinated, coordinated}. ──
    //       Schedule A (uncoordinated): each source at `offset`. Earliest
    //       arrivals, fights resolved serially by `simulate_planet_timeline`.
    //       Schedule B (coordinated at A_S = max-natural-arrival): every
    //       earlier source delays to land on the same turn as the latest
    //       source, presenting a combined force the defender can't split.
    //       Per-source delay search is cache-friendly (small δ scan).
    let mut best_score = f64::NEG_INFINITY;
    let mut best_ships = i64::MAX;
    let mut best_orders: Vec<PlannedOrder> = Vec::new();
    let mut best_max_arrival: i64 = 0;
    let mut best_marginal_in_orders: usize = 0;
    let mut best_marginal_not_doomed = false;

    let mut plan_orders: Vec<PlannedOrder> = Vec::with_capacity(n);
    let mut trial: Vec<ArrivalEvent> = Vec::with_capacity(n);

    let consider = |
        orders: &Vec<PlannedOrder>,
        max_arrival: i64,
        ships_total: i64,
        marginal_idx: usize,
        marginal_not_doomed: bool,
        best_score: &mut f64,
        best_ships: &mut i64,
        best_orders: &mut Vec<PlannedOrder>,
        best_max_arrival: &mut i64,
        best_marginal_in_orders: &mut usize,
        best_marginal_not_doomed: &mut bool,
    | {
        let score = score_capture(world, target, max_arrival);
        let better = score > *best_score || (score == *best_score && ships_total < *best_ships);
        if better {
            *best_score = score;
            *best_ships = ships_total;
            *best_orders = orders.clone();
            *best_max_arrival = max_arrival;
            *best_marginal_in_orders = marginal_idx;
            *best_marginal_not_doomed = marginal_not_doomed;
        }
    };

    for mask in 1u32..(1u32 << n) {
        // ── Schedule A: uncoordinated. ──
        plan_orders.clear();
        trial.clear();
        let mut ships_total: i64 = 0;
        let mut max_arrival_a: i64 = 0;
        let mut marginal_idx_a: usize = 0;
        let mut marginal_not_doomed_a = false;
        for i in 0..n {
            if mask & (1u32 << i) == 0 {
                continue;
            }
            let c = &candidates[i];
            if c.arrival > max_arrival_a {
                max_arrival_a = c.arrival;
                marginal_idx_a = plan_orders.len();
                marginal_not_doomed_a = c.not_doomed;
            }
            plan_orders.push(PlannedOrder {
                src_id: c.id,
                angle: c.angle,
                ships: c.ships_max,
                arrival: c.arrival,
                effective_offset: offset,
            });
            trial.push(ArrivalEvent {
                turns: c.arrival,
                owner: world.player,
                ships: c.ships_max,
            });
            ships_total += c.ships_max;
        }
        let tl = target_timeline(world, target.id, &trial, plan);
        if final_owner(&tl) == world.player {
            consider(
                &plan_orders, max_arrival_a, ships_total,
                marginal_idx_a, marginal_not_doomed_a,
                &mut best_score, &mut best_ships, &mut best_orders,
                &mut best_max_arrival, &mut best_marginal_in_orders,
                &mut best_marginal_not_doomed,
            );
        }

        // ── Schedule B: coordinated cluster at A_S + k, k in 0..=A_S_LOOKAHEAD.
        //   k = 0 mirrors the original "land together at the natural max
        //     arrival" coordination.
        //   k > 0 pushes the cluster further out so slow-growing sources can
        //     accumulate `production·d` extra ships before launch (growth-
        //     aware). Score-wise this is only attractive when the heavier
        //     fleet is what flips the trial timeline — otherwise Schedule A
        //     or k = 0 will dominate via `consider`'s arrival-aware score.
        let a_s = max_arrival_a;
        let mut has_earlier = false;
        for i in 0..n {
            if mask & (1u32 << i) == 0 { continue; }
            if candidates[i].arrival < a_s {
                has_earlier = true;
                break;
            }
        }
        // When no source is earlier than A_S, k = 0 reduces to Schedule A
        // exactly — skip it to avoid duplicate work. k > 0 still adds value
        // (growth on every source).
        let start_k: i64 = if has_earlier { 0 } else { 1 };
        for k in start_k..=A_S_LOOKAHEAD {
            let target_a_s = a_s + k;
            let max_delay = MAX_COORD_DELAY + k;
            plan_orders.clear();
            trial.clear();
            let mut ships_total: i64 = 0;
            let mut max_arrival_b: i64 = 0;
            let mut marginal_idx_b: usize = 0;
            let mut marginal_not_doomed_b = false;
            let mut feasible = true;
            for i in 0..n {
                if mask & (1u32 << i) == 0 { continue; }
                let c = &candidates[i];
                // Pick the latest arrival ≤ target_a_s achievable within the
                // delay budget. Per source, growth scales ships with delay.
                let mut best_d: i64 = -1;
                let mut best_arr: i64 = -1;
                let mut best_ang: f64 = c.angle;
                let mut best_ships: i64 = c.ships_max;
                for d in 0..=max_delay {
                    let ships_try = c.ships_max + c.production * d;
                    let Some((a, t, _, _, _)) =
                        model.plan_shot(c.id, target.id, ships_try, offset + d)
                    else { continue };
                    let arr = (offset + d + t).max(1);
                    if arr <= target_a_s && arr > best_arr {
                        best_d = d;
                        best_arr = arr;
                        best_ang = a;
                        best_ships = ships_try;
                    }
                }
                if best_d < 0 {
                    feasible = false;
                    break;
                }
                if best_arr > max_arrival_b {
                    max_arrival_b = best_arr;
                    marginal_idx_b = plan_orders.len();
                    marginal_not_doomed_b = c.not_doomed;
                }
                plan_orders.push(PlannedOrder {
                    src_id: c.id,
                    angle: best_ang,
                    ships: best_ships,
                    arrival: best_arr,
                    effective_offset: offset + best_d,
                });
                trial.push(ArrivalEvent {
                    turns: best_arr,
                    owner: world.player,
                    ships: best_ships,
                });
                ships_total += best_ships;
            }
            if !feasible {
                continue;
            }
            let tl = target_timeline(world, target.id, &trial, plan);
            if final_owner(&tl) == world.player {
                consider(
                    &plan_orders, max_arrival_b, ships_total,
                    marginal_idx_b, marginal_not_doomed_b,
                    &mut best_score, &mut best_ships, &mut best_orders,
                    &mut best_max_arrival, &mut best_marginal_in_orders,
                    &mut best_marginal_not_doomed,
                );
            }
        }
    }

    if best_orders.is_empty() {
        return None;
    }

    // ── 3. Halve-trim on the marginal (latest-arriving) source. ──
    // (Binary-search-to-minimum was tested and dropped win rate: smaller
    // marginal fleets are also slower under log-shaped fleet_speed, and
    // arriving earlier with overcommitted ships forces the opponent's hand.)
    if best_marginal_not_doomed {
        trial.clear();
        for o in &best_orders {
            trial.push(ArrivalEvent {
                turns: o.arrival,
                owner: world.player,
                ships: o.ships,
            });
        }
        let tl = target_timeline(world, target.id, &trial, plan);
        let horizon = tl.horizon as usize;
        let arrival_idx = (best_max_arrival as usize).min(horizon);
        let mut excess: i64 = i64::MAX;
        for t in arrival_idx..=horizon {
            let margin = if tl.owner_at[t] == world.player {
                tl.ships_at[t]
            } else {
                0
            };
            if margin < excess {
                excess = margin;
            }
        }
        if excess == i64::MAX {
            excess = 0;
        }
        let marginal = &best_orders[best_marginal_in_orders];
        let src_id = marginal.src_id;
        let max_ships = marginal.ships;
        let marginal_eff_offset = marginal.effective_offset;
        let excess = excess.min(max_ships);
        let keep = excess / 2;
        let trimmed = (max_ships - keep).max(TRIM_MIN_SHIPS);
        if trimmed < max_ships {
            if let Some((t_angle, t_turns, _, _, _)) =
                model.plan_shot(src_id, target.id, trimmed, marginal_eff_offset)
            {
                let t_arrival = (marginal_eff_offset + t_turns).max(1);
                let saved = trial[best_marginal_in_orders];
                trial[best_marginal_in_orders] = ArrivalEvent {
                    turns: t_arrival,
                    owner: world.player,
                    ships: trimmed,
                };
                let tl2 = target_timeline(world, target.id, &trial, plan);
                if final_owner(&tl2) == world.player {
                    best_orders[best_marginal_in_orders] = PlannedOrder {
                        src_id,
                        angle: t_angle,
                        ships: trimmed,
                        arrival: t_arrival,
                        effective_offset: marginal_eff_offset,
                    };
                    best_max_arrival = best_orders.iter().map(|o| o.arrival).max().unwrap_or(0);
                } else {
                    trial[best_marginal_in_orders] = saved;
                }
            }
        }
    }

    Some(FrontlineWin {
        orders: best_orders,
        max_arrival: best_max_arrival,
    })
}

/// Per-source baseline for the subset enumeration: the maximum ships this
/// source is willing to commit to `target` at launch `offset`, plus the
/// shot's angle and arrival turn. Sources unable to contribute (insufficient
/// ships, blocked shot, would-drop-in-mid-battle, or hopeless defense) are
/// filtered out entirely so the 2^N loop only enumerates real options.
struct SourceCandidate {
    id: i64,
    angle: f64,
    arrival: i64,   // turns from current step until arrival
    ships_max: i64, // pre-trim ships willing to send at base `offset`
    not_doomed: bool,
    /// Production rate; used by the coordinated schedule to grow `ships_max`
    /// when this source delays beyond its natural arrival.
    production: i64,
}

fn collect_source_candidates(
    world: &WorldState,
    model: &HellburnerModel,
    target: &Planet,
    plan: &PlanState,
    offset: i64,
) -> Vec<SourceCandidate> {
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

    // Don't drop in mid-battle between two other players: if target isn't
    // ours, skip our arrivals that land at or before another enemy's earliest
    // arrival window.
    let second_enemy_arrival: Option<i64> = if target.owner != world.player {
        world
            .timeline_cache
            .arrivals(target.id)
            .iter()
            .filter(|ev| ev.owner != world.player && ev.owner != target.owner)
            .map(|ev| ev.turns)
            .min()
    } else {
        None
    };

    let mut out = Vec::with_capacity(origins.len());
    for (src_id, _travel) in origins {
        let src = world.planet(src_id).clone();
        // Growth-aware: at launch offset the source will have accumulated
        // `production·offset` extra ships on top of the current pool.
        let available = plan.ships_available_at(&src, offset);
        if available == 0 {
            continue;
        }
        let mut ships_to_send = available;
        let not_doomed = baseline_owns(world, src_id);
        if not_doomed {
            let (holds, half_pressure) =
                neighbor_holds_under_worst_case(world, model, &src, plan);
            if !holds {
                if target.production <= src.production {
                    continue;
                }
                // Knowingly sacrificing source: send all.
            } else {
                ships_to_send = (available - half_pressure).max(0);
                if ships_to_send == 0 {
                    continue;
                }
            }
        }
        let Some((angle, turns, _, _, _)) =
            model.plan_shot(src_id, target.id, ships_to_send, offset)
        else {
            continue;
        };
        let arrival = (offset + turns).max(1);
        if let Some(sea) = second_enemy_arrival {
            if arrival <= sea + SECOND_ENEMY_ARRIVAL_TOL {
                continue;
            }
        }
        out.push(SourceCandidate {
            id: src_id,
            angle,
            arrival,
            ships_max: ships_to_send,
            not_doomed,
            production: src.production,
        });
    }
    out
}

// ── evaluate_move_orders ─────────────────────────────────────────────────

/// Which target each greedy iteration of `run_strategy` should commit. The
/// rollouts in `plan()` try every variant and pick whichever resulting
/// `PlanState` integrates the most own-production over the horizon — so
/// "which sort key is right" is decided empirically per turn, not baked in.
#[derive(Clone, Copy)]
enum SelectionStrategy {
    /// `score_now − max(0, best_score_later)` — urgency-aware. The default
    /// from when we added the offset sweep.
    PriorityFirst,
    /// Pure `score_capture(now)`. Ignores how the target's value would
    /// decay if deferred — better when timing isn't fragile.
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
            // PriorityFirst and ScoreFirst both key on the offset-aware score;
            // the old urgency-delta term is subsumed because we now pick the
            // best offset per target, so the score already reflects whether
            // waiting helps.
            SelectionStrategy::PriorityFirst | SelectionStrategy::ScoreFirst => score,
            SelectionStrategy::ScorePerShip => score / (1.0 + ships_total as f64),
            SelectionStrategy::ProductionFirst => production as f64,
        }
    }
}

/// Picks the single best (target, launch-offset) commitment for this greedy
/// iteration. Sweeps `offset ∈ 0..=OFFSET_LOOKAHEAD` per target and keeps the
/// highest-scoring [`score_capture`] commitment. Orders with
/// `effective_offset > 0` are *reservations* — [`run_strategy`] reserves the
/// ships in [`PlanState`] but doesn't emit a fleet move this turn, so
/// "wait and grow" emerges naturally when a delayed plan outscores any
/// offset-0 alternative.
fn evaluate_move_orders(
    world: &WorldState,
    model: &HellburnerModel,
    plan: &PlanState,
    strategy: SelectionStrategy,
) -> Option<(i64, Vec<PlannedOrder>, i64)> {
    let candidates: Vec<&Planet> = world
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
        .collect();

    // Track best by (priority, score_now), tiebreak shorter order lists.
    let mut best: Option<(f64, f64, usize, i64, Vec<PlannedOrder>)> = None;

    for target in candidates {
        // Skip targets already won by baseline + planned commitments.
        let tl = target_timeline(world, target.id, &[], plan);
        if final_owner(&tl) == world.player {
            continue;
        }

        // Sweep offsets and keep the highest-scoring commitment. Acting now
        // (offset 0) competes head-to-head against waiting (offset > 0):
        // whichever offset yields the better `score_capture` wins. Delayed
        // wins return `effective_offset > 0` orders, which `run_strategy`
        // commits as reservations (no emission this turn).
        let mut best_for_target: Option<(f64, FrontlineWin)> = None;
        for delta in 0..=OFFSET_LOOKAHEAD {
            let Some(win) = evaluate_frontline_strategy(world, model, target, plan, delta)
            else { continue };
            let s = score_capture(world, target, win.max_arrival);
            match &best_for_target {
                None => best_for_target = Some((s, win)),
                Some((bs, _)) if s > *bs => best_for_target = Some((s, win)),
                _ => {}
            }
        }
        let Some((score, win)) = best_for_target else { continue };

        let ships_total: i64 = win.orders.iter().map(|o| o.ships).sum();
        let primary = strategy.key(score, target.production, ships_total);
        // Secondary tiebreak: raw score. Tertiary: shorter order list.
        let better = match &best {
            None => true,
            Some((bp, bs, blen, _, _)) => {
                primary > *bp
                    || (primary == *bp && score > *bs)
                    || (primary == *bp && score == *bs && win.orders.len() < *blen)
            }
        };
        if better {
            best = Some((primary, score, win.orders.len(), target.id, win.orders));
        }
    }

    best.map(|(_, score, _, t, o)| (t, o, score as i64))
}

// ── send_reinforcements ──────────────────────────────────────────────────

fn send_reinforcements(
    world: &WorldState,
    model: &HellburnerModel,
    plan: &PlanState,
) -> Vec<FleetOrder> {
    let mut out = Vec::new();
    let player = world.player;
    let empty: Vec<(i64, f64)> = Vec::new();
    for p in &world.my_planets {
        if !model.non_comet_ids.contains(&p.id) {
            continue;
        }
        let Some(target_id) = model.reinforcement_target.get(&p.id).copied() else {
            continue;
        };
        let available = plan.ships_available(p);
        if available < REINFORCEMENT_SIZE + GARRISON_SIZE {
            continue;
        }
        let has_enemy_incoming = model
            .inbound_edges
            .get(&p.id)
            .unwrap_or(&empty)
            .iter()
            .any(|(sid, _)| world.planet(*sid).owner != player);
        if has_enemy_incoming {
            continue;
        }
        let ships = available - GARRISON_SIZE;
        let Some((angle, _turns, _, _, _)) =
            model.plan_shot(p.id, target_id, ships, 0)
        else {
            continue;
        };
        out.push((p.id, angle, ships));
    }
    out
}

// ── Early game DFS ───────────────────────────────────────────────────────

#[derive(Clone)]
struct EarlyFleet {
    destination_id: i64,
    fleet_size: i64,
    garrison_on_arrival: i64,
    arrival_turn: i64,
    is_capture: bool,
}

#[derive(Clone)]
struct EarlyState {
    turn: i64,
    garrison: HashMap<i64, f64>,
    production: HashMap<i64, i64>,
    owned: HashSet<i64>,
    fleets: Vec<EarlyFleet>,
}

fn early_production_of(world: &WorldState, planet_id: i64) -> i64 {
    world
        .planet_by_id
        .get(&planet_id)
        .map(|p| p.production)
        .unwrap_or(0)
}

/// Travel time for the early-game DFS. Routes through
/// `model.plan_shot` so the search is obstacle-aware *and*
/// future-launch-aware: source, target, and obstacles are all evaluated at
/// the real launch turn (`launch_turn_offset = launch_turn - world.step`),
/// not at the current turn. Returns `+inf` when the shot is blocked or no
/// viable intercept exists.
fn early_travel_turns(
    model: &HellburnerModel,
    source_id: i64,
    target_id: i64,
    fleet_size: i64,
    launch_turn_offset: i64,
) -> f64 {
    match model.plan_shot(source_id, target_id, fleet_size, launch_turn_offset) {
        Some((_angle, turns, _, _, _)) => turns as f64,
        None => f64::INFINITY,
    }
}

fn early_find_capture_turn(
    model: &HellburnerModel,
    state: &EarlyState,
    target: &Planet,
) -> Option<i64> {
    let garrison_size = target.ships;
    let horizon = state.turn + HORIZON;
    let mut best: Option<i64> = None;
    for &source in &state.owned {
        let current_ships = *state.garrison.get(&source).unwrap_or(&0.0);
        let production_rate = *state.production.get(&source).unwrap_or(&0) as f64;
        for wait_turns in 0..HORIZON {
            let fleet_size = (current_ships + production_rate * (wait_turns as f64)) as i64;
            if fleet_size <= garrison_size {
                continue;
            }
            let launch_turn = state.turn + wait_turns;
            if launch_turn >= horizon {
                break;
            }
            let launch_offset = launch_turn - model.state.step;
            let travel = early_travel_turns(model, source, target.id, fleet_size, launch_offset);
            if !travel.is_finite() {
                // Shot is blocked at this fleet size. Different ship counts
                // produce different fleet_speed and thus different intercept
                // geometry, so a heavier/lighter fleet could still be viable —
                // keep searching this source rather than abandoning it.
                continue;
            }
            let arrival = launch_turn + travel.ceil() as i64;
            if arrival <= horizon {
                best = Some(match best {
                    Some(b) => b.min(arrival),
                    None => arrival,
                });
                break; // larger fleets from this source arrive no earlier
            }
        }
    }
    best
}

/// (source_id, fleet_size, launch_turn). Arrival equals `capture_turn` by
/// construction: this mirrors `early_find_capture_turn`'s per-source search
/// (same break-on-first-viable-fleet rule), so the source that produced the
/// minimum arrival there is rediscovered here.
fn early_assign_fleet(
    model: &HellburnerModel,
    state: &EarlyState,
    target: &Planet,
    capture_turn: i64,
) -> Option<(i64, i64, i64)> {
    let garrison_size = target.ships;
    let mut best: Option<(i64, i64, i64)> = None;
    let mut best_arrival = i64::MAX;
    for &source in &state.owned {
        let current_ships = *state.garrison.get(&source).unwrap_or(&0.0);
        let production_rate = *state.production.get(&source).unwrap_or(&0) as f64;
        for wait_turns in 0..(capture_turn - state.turn) {
            let fleet_size = (current_ships + production_rate * (wait_turns as f64)) as i64;
            if fleet_size <= garrison_size {
                continue;
            }
            let launch_turn = state.turn + wait_turns;
            let launch_offset = launch_turn - model.state.step;
            let travel = early_travel_turns(model, source, target.id, fleet_size, launch_offset);
            if !travel.is_finite() {
                continue;
            }
            let arrival = launch_turn + travel.ceil() as i64;
            if arrival <= capture_turn && arrival < best_arrival {
                best_arrival = arrival;
                best = Some((source, fleet_size, launch_turn));
            }
            break;
        }
    }
    best
}

fn early_advance(state: &mut EarlyState, world: &WorldState, from_turn: i64, to_turn: i64) {
    for current_turn in (from_turn + 1)..=to_turn {
        let mut i = 0;
        while i < state.fleets.len() {
            if state.fleets[i].arrival_turn == current_turn {
                let f = state.fleets.remove(i);
                if f.is_capture {
                    state
                        .garrison
                        .insert(f.destination_id, f.garrison_on_arrival as f64);
                    state.owned.insert(f.destination_id);
                    state
                        .production
                        .entry(f.destination_id)
                        .or_insert_with(|| early_production_of(world, f.destination_id));
                } else {
                    *state.garrison.entry(f.destination_id).or_insert(0.0) +=
                        f.garrison_on_arrival as f64;
                }
            } else {
                i += 1;
            }
        }
        for &pid in &state.owned {
            let prod = *state.production.get(&pid).unwrap_or(&0) as f64;
            *state.garrison.entry(pid).or_insert(0.0) += prod;
        }
        state.turn = current_turn;
    }
}

fn early_execute(
    state: &mut EarlyState,
    world: &WorldState,
    target: &Planet,
    assign: (i64, i64, i64),
    capture_turn: i64,
) {
    let (source, fleet_size, launch_turn) = assign;
    early_advance(state, world, state.turn, launch_turn);
    *state.garrison.entry(source).or_insert(0.0) -= fleet_size as f64;
    let garrison_size = target.ships;
    state.fleets.push(EarlyFleet {
        destination_id: target.id,
        fleet_size,
        garrison_on_arrival: fleet_size - garrison_size,
        arrival_turn: capture_turn,
        is_capture: true,
    });
    early_advance(state, world, state.turn, capture_turn);
}

fn early_score(state: &EarlyState, world: &WorldState) -> i64 {
    let horizon = state.turn + HORIZON;
    let mut total: i64 = 0;
    for &pid in &state.owned {
        let g = *state.garrison.get(&pid).unwrap_or(&0.0);
        let p = *state.production.get(&pid).unwrap_or(&0);
        total += g as i64 + p * (horizon - state.turn);
    }
    for f in &state.fleets {
        total += f.garrison_on_arrival;
        if f.is_capture {
            let prod = early_production_of(world, f.destination_id);
            total += prod * (horizon - f.arrival_turn).max(0);
        }
    }
    total
}

fn run_early_game(world: &WorldState, model: &HellburnerModel) -> Vec<FleetOrder> {
    let player = world.player;
    let owned_ids: HashSet<i64> = world
        .my_planets
        .iter()
        .filter(|p| model.non_comet_ids.contains(&p.id))
        .map(|p| p.id)
        .collect();

    let neutral_candidates: Vec<Planet> = world
        .planets
        .iter()
        .filter(|p| p.owner == -1 && model.non_comet_ids.contains(&p.id))
        .filter(|p| {
            model
                .inbound_edges
                .get(&p.id)
                .map(|edges| edges.iter().any(|(sid, _)| owned_ids.contains(sid)))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    // In-flight friendly fleets from TimelineCache.
    let mut in_flight: Vec<EarlyFleet> = Vec::new();
    for planet in &world.planets {
        if !model.non_comet_ids.contains(&planet.id) {
            continue;
        }
        for ev in world.timeline_cache.arrivals(planet.id) {
            if ev.owner != player {
                continue;
            }
            let is_cap = planet.owner != player;
            let surplus = ev.ships - planet.ships;
            in_flight.push(EarlyFleet {
                destination_id: planet.id,
                fleet_size: ev.ships,
                garrison_on_arrival: if is_cap { surplus } else { ev.ships },
                arrival_turn: world.step + ev.turns,
                is_capture: is_cap,
            });
        }
    }

    let initial_state = EarlyState {
        turn: world.step,
        garrison: world
            .my_planets
            .iter()
            .filter(|p| model.non_comet_ids.contains(&p.id))
            .map(|p| (p.id, p.ships as f64))
            .collect(),
        production: world
            .my_planets
            .iter()
            .filter(|p| model.non_comet_ids.contains(&p.id))
            .map(|p| (p.id, p.production))
            .collect(),
        owned: owned_ids,
        fleets: in_flight,
    };

    let initial_gain = |planet: &Planet, state: &EarlyState| -> f64 {
        let Some(ct) = early_find_capture_turn(model, state, planet) else {
            return f64::NEG_INFINITY;
        };
        let horizon = state.turn + HORIZON;
        (planet.production * (horizon - ct) - planet.ships) as f64
    };

    let mut candidates: Vec<Planet> = neutral_candidates;
    candidates.sort_by(|a, b| {
        initial_gain(b, &initial_state)
            .partial_cmp(&initial_gain(a, &initial_state))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.retain(|p| initial_gain(p, &initial_state) > 0.0);

    if candidates.is_empty() {
        return vec![];
    }

    let mut best_score = early_score(&initial_state, world);
    let mut best_sequence: Vec<(Planet, (i64, i64, i64), i64)> = Vec::new();
    let mut sequence: Vec<(Planet, (i64, i64, i64), i64)> = Vec::new();

    fn dfs(
        world: &WorldState,
        model: &HellburnerModel,
        state: &EarlyState,
        remaining: &[Planet],
        sequence: &mut Vec<(Planet, (i64, i64, i64), i64)>,
        best_score: &mut i64,
        best_sequence: &mut Vec<(Planet, (i64, i64, i64), i64)>,
    ) {
        let cur_score = early_score(state, world);
        if cur_score > *best_score {
            *best_score = cur_score;
            *best_sequence = sequence.clone();
        }

        let horizon = state.turn + HORIZON;
        let mut bound = cur_score;
        for planet in remaining {
            if let Some(ct) = early_find_capture_turn(model, state, planet) {
                let gain = planet.production * (horizon - ct) - planet.ships;
                if gain > 0 {
                    bound += gain;
                }
            }
        }
        if bound <= *best_score {
            return;
        }

        let already_targeted: HashSet<i64> = state
            .fleets
            .iter()
            .filter(|f| f.is_capture)
            .map(|f| f.destination_id)
            .collect();
        for (idx, planet) in remaining.iter().enumerate() {
            if already_targeted.contains(&planet.id) {
                continue;
            }
            let Some(ct) = early_find_capture_turn(model, state, planet) else {
                continue;
            };
            if planet.production * (horizon - ct) - planet.ships <= 0 {
                continue;
            }
            let Some(assign) = early_assign_fleet(model, state, planet, ct) else {
                continue;
            };
            let mut next_state = state.clone();
            early_execute(&mut next_state, world, planet, assign, ct);
            sequence.push((planet.clone(), assign, ct));
            let mut next_remaining: Vec<Planet> = remaining[..idx].to_vec();
            next_remaining.extend(remaining[idx + 1..].iter().cloned());
            dfs(
                world,
                model,
                &next_state,
                &next_remaining,
                sequence,
                best_score,
                best_sequence,
            );
            sequence.pop();
        }
    }

    dfs(
        world,
        model,
        &initial_state,
        &candidates,
        &mut sequence,
        &mut best_score,
        &mut best_sequence,
    );

    // Emit only moves whose launch_turn == current step.
    let mut moves: Vec<FleetOrder> = Vec::new();
    for (target_planet, (source_id, fleet_size, launch_turn), _) in &best_sequence {
        if *launch_turn != world.step {
            continue;
        }
        let Some((angle, _turns, _, _, _)) =
            model.plan_shot(*source_id, target_planet.id, *fleet_size, 0)
        else {
            continue;
        };
        moves.push((*source_id, angle, *fleet_size));
    }
    moves
}

// ── Public entry ─────────────────────────────────────────────────────────

/// One full pipeline run under a fixed target-selection strategy. Returns
/// the emitted moves and the resulting PlanState (used by `rollout_score`).
fn run_strategy(
    world: &WorldState,
    model: &HellburnerModel,
    strategy: SelectionStrategy,
) -> (Vec<FleetOrder>, PlanState) {
    let mut state = PlanState::default();
    let mut moves: Vec<FleetOrder> = Vec::new();
    // Each iteration commits ≥1 ship from at least one source, so the loop
    // is bounded by the total source pool. A fixed safety cap protects
    // against any pathological selector that re-picks the same target with
    // a vanishing commitment.
    for _ in 0..256 {
        let Some((target_id, fleet_orders, _value)) =
            evaluate_move_orders(world, model, &state, strategy)
        else {
            break;
        };
        if fleet_orders.is_empty() {
            break;
        }
        // Commit every order to PlanState (reservation), but only emit a
        // fleet move when the order launches this turn. Reservations make
        // subsequent iterations see the reserved ships as spent, so they
        // can't be poached for lower-value captures — the no-op deferral
        // emerges from the score sweep in `evaluate_move_orders`.
        for o in fleet_orders {
            state.commit(o.src_id, target_id, o.ships, o.arrival, world.player);
            if o.effective_offset == 0 {
                moves.push((o.src_id, o.angle, o.ships));
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
const STRATEGIES: [SelectionStrategy; 4] = [
    SelectionStrategy::PriorityFirst,
    SelectionStrategy::ScoreFirst,
    SelectionStrategy::ScorePerShip,
    SelectionStrategy::ProductionFirst,
];

pub fn plan(world: &WorldState) -> Vec<FleetOrder> {
    if world.enemy_planets.is_empty() {
        return Vec::new();
    }
    let model = HellburnerModel::build(world);

    if world.step < EARLY_ROUNDS {
        return run_early_game(world, &model);
    }

    // Single greedy run under the default strategy. This is the policy hook
    // the rollout layer (see `crate::apollo::rollout`) invokes for opponent replies
    // *and* our own replanning during the reactive phase, so it must stay
    // cheap and deterministic. Multi-strategy search happens one level up,
    // in `search_candidates`.
    let (moves, _) = run_strategy(world, &model, STRATEGIES[0]);
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
pub fn search_candidates(world: &WorldState) -> Vec<Vec<FleetOrder>> {
    if world.enemy_planets.is_empty() {
        return vec![Vec::new()];
    }
    let model = HellburnerModel::build(world);

    if world.step < EARLY_ROUNDS {
        return vec![run_early_game(world, &model)];
    }

    let mut out: Vec<Vec<FleetOrder>> = Vec::with_capacity(STRATEGIES.len());
    for &strat in &STRATEGIES {
        let (moves, _) = run_strategy(world, &model, strat);
        if !out.iter().any(|prev| prev == &moves) {
            out.push(moves);
        }
    }
    out
}
