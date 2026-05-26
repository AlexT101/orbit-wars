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
//!   * Proximity graph (`MAX_DISTANCE=38`, `ROTATION_LOOK_AHEAD=10`).
//!   * `reinforcement_target` per owned planet (frontline BFS).
//!   * Per-turn `PlanState` (spent ships + planned commitments).

#![allow(dead_code)]

use std::cell::RefCell;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::constants::CENTER;
use crate::engine::Planet;
use crate::entity_cache::{AimCacheVerdict, Entity};
use crate::helpers::{
    aim_with_prediction, dist, fleet_speed, simulate_planet_timeline, AimResult, ArrivalEvent,
    PlanetTimeline,
};
use crate::world::{merge_arrivals, WorldState};

// ── Constants (mirror hellburner) ────────────────────────────────────────
const EARLY_ROUNDS: i64 = 3;
const EARLY_LOOK_AHEAD: i64 = 33;
const MAX_DISTANCE: f64 = 38.0;
const ROTATION_LOOK_AHEAD: i64 = 10;
const REINFORCEMENT_SIZE: i64 = 17;
const GARRISON_SIZE: i64 = 11;
const SECOND_ENEMY_ARRIVAL_TOL: i64 = 1;
const TRIM_MIN_SHIPS: i64 = 10;

type FleetOrder = (i64, f64, i64); // (src_id, angle, ships)

// ── Iterative intercept (mirrors hellburner's `intercept_planet`) ────────

/// Continuous-time orbital position. `t_abs` is the absolute game step
/// (allowed to be non-integer). For static planets / non-orbiters returns
/// the fixed position.
fn planet_pos_at(entity: &Entity, omega: f64, t_abs: f64) -> [f64; 2] {
    if entity.is_static() {
        return entity.positions[0].unwrap_or([CENTER, CENTER]);
    }
    let r = entity.orbital_radius;
    let initial_pos = entity.positions[0].unwrap_or([CENTER + r, CENTER]);
    let ia = (initial_pos[1] - CENTER).atan2(initial_pos[0] - CENTER);
    let a = ia + omega * t_abs;
    [CENTER + r * a.cos(), CENTER + r * a.sin()]
}

/// Port of hellburner's `intercept_planet` — iterative damped fixed-point on
/// travel time for orbital targets. Returns `(angle, tx, ty, travel)` where
/// `travel` is in continuous turns; returns `+inf` travel if the iteration
/// diverges (fleet too slow to catch orbital target). `scene_step` is the
/// absolute game step at which the fleet *launches*.
fn intercept_from(
    world: &WorldState,
    sx: f64,
    sy: f64,
    target_id: i64,
    ships: i64,
    scene_step: f64,
) -> (f64, f64, f64, f64) {
    let target = world.planet(target_id);
    let speed = fleet_speed(ships);
    let entity = match world.entity_cache.get(target_id) {
        Some(e) => e,
        None => {
            let travel = dist(sx, sy, target.x, target.y) / speed;
            let angle = (target.y - sy).atan2(target.x - sx);
            return (angle, target.x, target.y, travel);
        }
    };
    let omega = world.entity_cache.angular_velocity;

    if entity.is_static() || entity.is_comet() {
        // Hellburner excludes comets from `self.planets` so this branch is
        // only hit for static planets. Use target's current position.
        let tx = target.x;
        let ty = target.y;
        let travel = dist(sx, sy, tx, ty) / speed;
        let angle = (ty - sy).atan2(tx - sx);
        return (angle, tx, ty, travel);
    }

    let r = entity.orbital_radius;
    let initial_pos = entity.positions[0].unwrap_or([CENTER + r, CENTER]);
    let ia = (initial_pos[1] - CENTER).atan2(initial_pos[0] - CENTER);

    let mut travel = dist(sx, sy, target.x, target.y) / speed;
    let mut converged = false;
    for _ in 0..30 {
        let a = ia + omega * (scene_step + travel - 0.5);
        let nx = CENTER + r * a.cos();
        let ny = CENTER + r * a.sin();
        let raw_new = dist(sx, sy, nx, ny) / speed;
        let new_travel = 0.5 * (travel + raw_new - 0.5);
        if (new_travel - travel).abs() < 1e-6 {
            travel = new_travel;
            converged = true;
            break;
        }
        travel = new_travel;
    }
    if !converged {
        return (0.0, target.x, target.y, f64::INFINITY);
    }
    let a = ia + omega * (scene_step + travel - 0.5);
    let tx = CENTER + r * a.cos();
    let ty = CENTER + r * a.sin();
    let angle = (ty - sy).atan2(tx - sx);
    (angle, tx, ty, travel)
}

/// Source position at fractional absolute step `t_abs`. Matches hellburner's
/// "launch_turn - 0.5" formula for orbiting sources.
fn source_pos_at(world: &WorldState, source_id: i64, t_abs: f64) -> (f64, f64) {
    let entity = match world.entity_cache.get(source_id) {
        Some(e) => e,
        None => {
            let s = world.planet(source_id);
            return (s.x, s.y);
        }
    };
    let omega = world.entity_cache.angular_velocity;
    let p = planet_pos_at(entity, omega, t_abs);
    (p[0], p[1])
}

// ── HellburnerModel ──────────────────────────────────────────────────────

pub struct HellburnerModel<'a> {
    pub state: &'a WorldState<'a>,
    /// Planet ids excluding comets — hellburner removes comets from
    /// `self.planets` entirely so they never appear in the proximity graph,
    /// reinforcement BFS, or target loops.
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

/// Trial timeline for attacking `target_id` — mirrors hellburner's
/// `trial_destination_list` which halves every non-self enemy arrival ship
/// count (Line 345-348 of main.py). Used only inside
/// `evaluate_frontline_strategy` to decide whether our planned send wins;
/// other timelines (e.g. baseline ownership for `evaluate_move_orders` or
/// neighbor's exposure rollout) use the full-strength arrivals.
fn target_timeline_halved(
    world: &WorldState,
    target_id: i64,
    extras: &[ArrivalEvent],
    plan: &PlanState,
) -> PlanetTimeline {
    let horizon = world.timeline_cache.horizon;
    let base = world.timeline_cache.arrivals(target_id);
    let planned: &[ArrivalEvent] = plan
        .planned
        .get(&target_id)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let mut arrivals: Vec<ArrivalEvent> =
        Vec::with_capacity(base.len() + planned.len() + extras.len());
    for ev in base {
        let ships = if ev.owner != world.player {
            ev.ships / 2 // mirror Python's `int(s * 0.5)`
        } else {
            ev.ships
        };
        if ships > 0 {
            arrivals.push(ArrivalEvent { turns: ev.turns, owner: ev.owner, ships });
        }
    }
    arrivals.extend_from_slice(planned);
    arrivals.extend_from_slice(extras);

    let target = world.planet(target_id);
    let expiry = world.timeline_cache.expiry(target_id);
    simulate_planet_timeline(target, &arrivals, world.player, horizon, expiry)
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

// ── evaluate_frontline_strategy ──────────────────────────────────────────

/// (fleet_orders, battle_won) — each order is (src_id, angle, ships, arrival_turn).
fn evaluate_frontline_strategy(
    world: &WorldState,
    model: &HellburnerModel,
    target: &Planet,
    plan: &PlanState,
) -> (Vec<(i64, f64, i64, i64)>, bool) {
    let empty: Vec<(i64, f64)> = Vec::new();
    let mut possible_origins: Vec<(i64, f64)> = model
        .inbound_edges
        .get(&target.id)
        .unwrap_or(&empty)
        .iter()
        .filter(|(sid, _)| world.planet(*sid).owner == world.player)
        .copied()
        .collect();
    possible_origins.sort_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // If target isn't ours, find earliest arrival from a non-us owner that isn't
    // the current owner — don't drop in mid-battle between two other players.
    // (For neutral targets, target.owner = -1, so any enemy arrival qualifies.)
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

    let mut trial: Vec<ArrivalEvent> = Vec::new();
    let mut fleet_orders: Vec<(i64, f64, i64, i64)> = Vec::new();

    for (src_id, _travel) in possible_origins {
        let src = world.planet(src_id).clone();
        let available = plan.ships_available(&src);
        if available == 0 {
            continue;
        }
        let mut ships_to_send = available;

        let not_doomed = baseline_owns(world, src_id);
        if not_doomed {
            let (holds, half_pressure) =
                neighbor_holds_under_worst_case(world, model, &src, plan);
            if !holds {
                // Source would fall under worst-case; skip unless target prod offsets.
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
            model.plan_shot(src_id, target.id, ships_to_send, 0)
        else {
            continue;
        };

        if let Some(sea) = second_enemy_arrival {
            if turns <= sea + SECOND_ENEMY_ARRIVAL_TOL {
                continue;
            }
        }

        let arrival_turn = turns.max(1);
        trial.push(ArrivalEvent {
            turns: arrival_turn,
            owner: world.player,
            ships: ships_to_send,
        });
        fleet_orders.push((src_id, angle, ships_to_send, arrival_turn));

        let tl = target_timeline_halved(world, target.id, &trial, plan);
        if final_owner(&tl) == world.player {
            // Trim if there's excess (only when not_doomed — see hellburner).
            // Python's `excess_ships` is the running minimum of post-combat
            // ships across every bucket turn ≥ last_turn (= our arrival turn),
            // with margin=0 whenever we don't own at that bucket, capped by
            // the size of our fleet (`last_ships`).
            //
            // We sweep every turn in `[arrival, horizon]` rather than only
            // bucket turns — equivalent because owner_at is constant and
            // ships_at is non-decreasing (production only) between buckets,
            // so the min is achieved on a bucket turn either way.
            if not_doomed {
                let horizon = tl.horizon as usize;
                let arrival_idx = (arrival_turn as usize).min(horizon);
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
                let excess = excess.min(ships_to_send);
                let keep = excess / 2;
                let trimmed = (ships_to_send - keep).max(TRIM_MIN_SHIPS);
                if trimmed < ships_to_send {
                    if let Some((t_angle, t_turns, _, _, _)) =
                        model.plan_shot(src_id, target.id, trimmed, 0)
                    {
                        let last_t = trial.len() - 1;
                        let saved = trial[last_t];
                        trial[last_t] = ArrivalEvent {
                            turns: t_turns.max(1),
                            owner: world.player,
                            ships: trimmed,
                        };
                        let tl2 = target_timeline_halved(world, target.id, &trial, plan);
                        if final_owner(&tl2) == world.player {
                            let last_o = fleet_orders.len() - 1;
                            fleet_orders[last_o] = (src_id, t_angle, trimmed, t_turns.max(1));
                        } else {
                            trial[last_t] = saved;
                        }
                    }
                }
            }
            return (fleet_orders, true);
        }
    }

    (fleet_orders, false)
}

// ── evaluate_move_orders ─────────────────────────────────────────────────

/// Returns `(target_id, fleet_orders, value)` for the best target found.
fn evaluate_move_orders(
    world: &WorldState,
    model: &HellburnerModel,
    plan: &PlanState,
) -> Option<(i64, Vec<(i64, f64, i64, i64)>, i64)> {
    let mut planets_sorted: Vec<&Planet> = world
        .planets
        .iter()
        .filter(|p| model.non_comet_ids.contains(&p.id))
        .collect();
    planets_sorted.sort_by(|a, b| b.ships.cmp(&a.ships));

    let mut best: Option<(i64, i64, Vec<(i64, f64, i64, i64)>)> = None;

    for target in planets_sorted {
        let has_inbound = model
            .inbound_edges
            .get(&target.id)
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        if !has_inbound {
            continue;
        }

        let tl = target_timeline(world, target.id, &[], plan);
        let owned_by_us_now = target.owner == world.player;

        if owned_by_us_now {
            // Defend only if currently being lost.
            if final_owner(&tl) == world.player {
                continue;
            }
        } else {
            // Attack only if not already won by in-flight + planned.
            if final_owner(&tl) == world.player {
                continue;
            }
        }

        let (orders, won) = evaluate_frontline_strategy(world, model, target, plan);
        if !won {
            continue;
        }

        let mut value = target.production;
        if !owned_by_us_now && target.owner == -1 {
            value -= 1;
        }

        let better = match &best {
            None => true,
            Some((_, bv, bo)) => value > *bv || (value == *bv && orders.len() < bo.len()),
        };
        if better {
            best = Some((target.id, value, orders));
        }
    }

    best.map(|(t, v, o)| (t, o, v))
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
    let horizon = state.turn + EARLY_LOOK_AHEAD;
    let mut best: Option<i64> = None;
    for &source in &state.owned {
        let current_ships = *state.garrison.get(&source).unwrap_or(&0.0);
        let production_rate = *state.production.get(&source).unwrap_or(&0) as f64;
        for wait_turns in 0..EARLY_LOOK_AHEAD {
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

/// (source_id, fleet_size, launch_turn, arrival_turn)
fn early_assign_fleet(
    model: &HellburnerModel,
    state: &EarlyState,
    target: &Planet,
    capture_turn: i64,
) -> Option<(i64, i64, i64, i64)> {
    let garrison_size = target.ships;
    let mut best: Option<(i64, i64, i64, i64)> = None;
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
                best = Some((source, fleet_size, launch_turn, arrival));
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
    assign: (i64, i64, i64, i64),
    capture_turn: i64,
) {
    let (source, fleet_size, launch_turn, _arrival) = assign;
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
    let horizon = state.turn + EARLY_LOOK_AHEAD;
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

    // In-flight friendly fleets from TimelineCache (hellburner's destination_list).
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
        let horizon = state.turn + EARLY_LOOK_AHEAD;
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
    let mut best_sequence: Vec<(Planet, (i64, i64, i64, i64), i64)> = Vec::new();
    let mut sequence: Vec<(Planet, (i64, i64, i64, i64), i64)> = Vec::new();

    fn dfs(
        world: &WorldState,
        model: &HellburnerModel,
        state: &EarlyState,
        remaining: &[Planet],
        sequence: &mut Vec<(Planet, (i64, i64, i64, i64), i64)>,
        best_score: &mut i64,
        best_sequence: &mut Vec<(Planet, (i64, i64, i64, i64), i64)>,
    ) {
        let cur_score = early_score(state, world);
        if cur_score > *best_score {
            *best_score = cur_score;
            *best_sequence = sequence.clone();
        }

        let horizon = state.turn + EARLY_LOOK_AHEAD;
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
    for (target_planet, (source_id, fleet_size, launch_turn, _), _) in &best_sequence {
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

pub fn plan(world: &WorldState) -> Vec<FleetOrder> {
    if world.enemy_planets.is_empty() {
        return Vec::new();
    }
    let model = HellburnerModel::build(world);

    if world.step < EARLY_ROUNDS {
        return run_early_game(world, &model);
    }

    let mut state = PlanState::default();
    let mut moves: Vec<FleetOrder> = Vec::new();
    loop {
        let Some((target_id, fleet_orders, _value)) =
            evaluate_move_orders(world, &model, &state)
        else {
            break;
        };
        if fleet_orders.is_empty() {
            // Target already won by baseline + planned — no new commitments to make,
            // but the loop would re-pick the same target indefinitely. Stop.
            break;
        }
        for (src_id, angle, ships, arrival) in fleet_orders {
            state.commit(src_id, target_id, ships, arrival, world.player);
            moves.push((src_id, angle, ships));
        }
    }
    moves.extend(send_reinforcements(world, &model, &state));
    moves
}

pub fn search_candidates(world: &WorldState) -> Vec<Vec<FleetOrder>> {
    vec![plan(world)]
}
