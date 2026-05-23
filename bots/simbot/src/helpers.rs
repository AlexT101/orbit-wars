#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::constants::{
    CENTER, EDGE_AIM_FRACS, HORIZON, FWD_ITER_MAX, LAUNCH_CLEARANCE, MAX_SHIP_SPEED, SUN_RADIUS,
};

use crate::engine::{Planet, Fleet, EngineState};
use crate::entity_cache::EntityCache;
use crate::sim_probe::SimProbe;
pub use crate::sim_probe::ArrivalEvent;


// ── Basic Helpers ────────────────────────────────────────────────────

/// Euclidean distance between two points
#[inline]
pub fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    crate::engine::distance((ax, ay), (bx, by))
}

///  Logarithmic speed curve between 1 and 6
#[inline]
pub fn fleet_speed(ships: i64) -> f64 {
    crate::engine::fleet_speed(ships.max(1), MAX_SHIP_SPEED)
}


/// Squared distance from point (px, py) to line segment (x1, y1)-(x2, y2)
#[inline]
pub fn point_to_segment_distance_sq(
    px: f64, py: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let l2 = dx * dx + dy * dy;
    let (qx, qy) = if l2 == 0.0 {
        (x1, y1)
    } else {
        let t = (((px - x1) * dx + (py - y1) * dy) / l2).clamp(0.0, 1.0);
        (x1 + t * dx, y1 + t * dy)
    };
    let ex = px - qx;
    let ey = py - qy;
    ex * ex + ey * ey
}

/// Returns true if movement segment (ax, ay)-(bx, by) comes within `r` of (cx, cy).
#[inline]
pub fn segment_intersects_circle(
    ax: f64, ay: f64,
    bx: f64, by: f64,
    cx: f64, cy: f64,
    r: f64,
) -> bool {
    point_to_segment_distance_sq(cx, cy, ax, ay, bx, by) <= r * r
}

/// Returns true if movement segment intersects sun
#[inline]
pub fn segment_hits_sun(
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> bool {
    point_to_segment_distance_sq(CENTER, CENTER, x1, y1, x2, y2) < SUN_RADIUS * SUN_RADIUS
}

/// Fleet spawns at (radius + 0.1) from planet center, at launch angle
#[inline]
pub fn launch_point(sx: f64, sy: f64, sr: f64, angle: f64) -> (f64, f64) {
    let c = sr + LAUNCH_CLEARANCE;
    (sx + angle.cos() * c, sy + angle.sin() * c)
}

/// Counts distinct active players: every non-neutral planet owner plus every
/// fleet owner. Floored at 2 since a match always has at least two players,
/// even if one is currently wiped off the map but still has a fleet in flight.
pub fn count_players(planets: &[Planet], fleets: &[Fleet]) -> usize {
    let mut owners: HashSet<i64> = HashSet::new();
    for p in planets {
        if p.owner != -1 {
            owners.insert(p.owner);
        }
    }
    for f in fleets {
        owners.insert(f.owner);
    }
    owners.len().max(2)
}

/// Shortest distance from `(px, py)` to the center of any planet in `set`.
/// Returns `f64::INFINITY` for an empty set so callers can compare freely.
pub fn nearest_distance_to_set(px: f64, py: f64, set: &[Planet]) -> f64 {
    set.iter()
        .map(|p| dist(px, py, p.x, p.y))
        .fold(f64::INFINITY, f64::min)
}

/// `(planet, distance)` pairs sorted ascending by distance from `(tx, ty)`.
pub fn sorted_by_distance_to(
    planets: &[Planet],
    tx: f64, ty: f64,
) -> Vec<(Planet, f64)> {
    let mut out: Vec<(Planet, f64)> = planets
        .iter()
        .map(|p| (p.clone(), dist(p.x, p.y, tx, ty)))
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}


// ── Aiming Helpers ────────────────────────────────────────────────────────────
// Solve for a shot that hits a (possibly moving) target.
// Layered approach: distance estimators → sun-bypass sampling → verification → solvers.

/// Calculates angle and travel distance between points
/// Returns `None` if direct path blocked by sun
pub fn safe_angle_and_distance(
    sx: f64, sy: f64, sr: f64,
    tx: f64, ty: f64, tr: f64,
) -> Option<(f64, f64)> {
    let angle = (ty - sy).atan2(tx - sx);
    let (lx, ly) = launch_point(sx, sy, sr, angle);
    let hit_dist = (dist(sx, sy, tx, ty) - sr - LAUNCH_CLEARANCE - tr).max(0.0);
    let ex = lx + angle.cos() * hit_dist;
    let ey = ly + angle.sin() * hit_dist;
    if segment_hits_sun(lx, ly, ex, ey) {
        return None;
    }
    Some((angle, hit_dist))
}

/// Calculate turns to travel distance at given fleet size, without rounding
#[inline]
fn fractional_turns(total_d: f64, ships: i64) -> f64 {
    total_d / fleet_speed(ships)
}

/// Fractional turn count used for convergence comparisons
#[inline]
pub fn estimate_arrival_frac(
    sx: f64, sy: f64, sr: f64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
) -> Option<(f64, f64)> {
    let (angle, total_d) = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)?;
    let turns = fractional_turns(total_d, ships).max(1.0);
    Some((angle, turns))
}

/// Integer-turn turn count
#[inline]
pub fn estimate_arrival(
    sx: f64, sy: f64, sr: f64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
) -> Option<(f64, i64)> {
    let (angle, frac_turns) =
        estimate_arrival_frac(sx, sy, sr, tx, ty, tr, ships)?;
    let turns = (frac_turns.ceil() as i64).max(1);
    Some((angle, turns))
}

/// Sun-bypass chord sampling. Samples aim-points across the target's disk
/// (center plus `EDGE_AIM_FRACS` mirrored offsets) and returns the one with
/// the shortest clear path. A chord aimed at the edge of a disk can clear
/// the sun when a center-aimed shot can't.
pub fn arc_safe_angle(
    sx: f64, sy: f64, sr: f64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
) -> Option<(f64, i64)> {
    let dx = tx - sx;
    let dy = ty - sy;
    let d = dx.hypot(dy);
    if d < 1e-9 {
        return None;
    }
    let ux = dx / d;
    let uy = dy / d;
    let nx = -uy;
    let ny = ux;

    // 1 center + 2 mirrored offsets per fraction → 1 + 2 * N candidate aims.
    let mut aim_points: Vec<(f64, f64)> = Vec::with_capacity(1 + 2 * EDGE_AIM_FRACS.len());
    aim_points.push((tx, ty));
    for &frac in EDGE_AIM_FRACS.iter() {
        let off = tr * frac;
        aim_points.push((tx + nx * off, ty + ny * off));
        aim_points.push((tx - nx * off, ty - ny * off));
    }

    // Score is (turns, entry_dist) — fewer turns first, then shortest dist.
    let mut best: Option<((i64, f64), f64, i64)> = None;
    for (ax, ay) in aim_points {
        let angle = (ay - sy).atan2(ax - sx);
        let (lx, ly) = launch_point(sx, sy, sr, angle);
        let rvx = angle.cos();
        let rvy = angle.sin();
        let cx = tx - lx;
        let cy = ty - ly;
        let proj = cx * rvx + cy * rvy;
        let closest_sq = cx * cx + cy * cy - proj * proj;
        if proj <= 0.0 || closest_sq > tr * tr {
            continue;
        }
        let entry_dist = (proj - (tr * tr - closest_sq).max(0.0).sqrt()).max(0.0);
        let ex = lx + rvx * entry_dist;
        let ey = ly + rvy * entry_dist;
        if segment_hits_sun(lx, ly, ex, ey) {
            continue;
        }
        let turns = (fractional_turns(entry_dist, ships).ceil() as i64).max(1);
        let score = (turns, entry_dist);
        let better = match &best {
            None => true,
            Some((cur, _, _)) => score.0 < cur.0 || (score.0 == cur.0 && score.1 < cur.1),
        };
        if better {
            best = Some((score, angle, turns));
        }
    }
    best.map(|(_, angle, turns)| (angle, turns))
}

/// Forward-sim scan window — `max(8, turns / 2)`, wide enough for slow
/// fleets on long intercepts to still confirm the hit.
#[inline]
pub fn fwd_window(turns: i64) -> i64 {
    (turns / 2).max(8)
}

/// Ground-truth forward-sim. Returns `true` only if the fleet physically hits
/// the target within the scan window. Mandatory gate before any intercept is
/// accepted.
pub fn verify_shot_hits(
    sx: f64, sy: f64, sr: f64,
    angle: f64, turns: i64, ships: i64,
    target_id: i64,
    cache: &EntityCache,
) -> bool {
    let Some(target) = cache.get(target_id) else {
        return false;
    };
    let tr = target.radius;
    let speed = fleet_speed(ships);
    let (mut fx, mut fy) = launch_point(sx, sy, sr, angle);
    let vx = angle.cos() * speed;
    let vy = angle.sin() * speed;
    let window = fwd_window(turns);

    for t in 1..=(turns + window) {
        let pfx = fx;
        let pfy = fy;
        fx += vx;
        fy += vy;
        if segment_hits_sun(pfx, pfy, fx, fy) {
            return false;
        }
        let Some([px, py]) = cache.position(target_id, t) else {
            continue;
        };
        if segment_intersects_circle(pfx, pfy, fx, fy, px, py, tr) {
            return true;
        }
    }
    false
}

/// Iterative convergence solver. Calculates direct or sun-blocked trajectories
/// by repeatedly refining the predicted intercept position. All results are
/// UNVERIFIED — caller must verify via `verify_shot_hits`.
pub fn aim_raw(
    sx: f64, sy: f64, sr: f64,
    target_id: i64,
    ships: i64,
    cache: &EntityCache,
) -> Option<(f64, i64, f64, f64)> {
    let target = cache.get(target_id)?;
    let tr = target.radius;
    let tol = target.tolerance;
    let [tx, ty] = cache.position(target_id, 0)?;

    let mut est = match estimate_arrival_frac(sx, sy, sr, tx, ty, tr, ships) {
        Some(v) => v,
        None => {
            // Direct shot blocked by the sun — try a chord around it.
            return arc_safe_angle(sx, sy, sr, tx, ty, tr, ships)
                .map(|(a, t)| (a, t, tx, ty));
        }
    };

    for _ in 0..FWD_ITER_MAX {
        let (_, turns_f) = est;
        let turns_i = turns_f.ceil() as i64;
        let Some([ntx, nty]) = cache.position(target_id, turns_i) else {
            return None;
        };
        let Some(next_est) = estimate_arrival_frac(sx, sy, sr, ntx, nty, tr, ships) else {
            return arc_safe_angle(sx, sy, sr, ntx, nty, tr, ships)
                .map(|(a, t)| (a, t, ntx, nty));
        };
        let (_, next_turns_f) = next_est;
        if (next_turns_f - turns_f).abs() <= tol as f64 {
            // Converged — return integer-turn result.
            return match estimate_arrival(sx, sy, sr, ntx, nty, tr, ships) {
                Some((a, t)) => Some((a, t, ntx, nty)),
                None => arc_safe_angle(sx, sy, sr, ntx, nty, tr, ships)
                    .map(|(a, t)| (a, t, ntx, nty)),
            };
        }
        est = next_est;
    }

    // Fallthrough: best effort with the last predicted position.
    let final_turns = est.1.ceil() as i64;
    let [fpx, fpy] = cache.position(target_id, final_turns)?;
    match estimate_arrival(sx, sy, sr, fpx, fpy, tr, ships) {
        Some((a, t)) => Some((a, t, fpx, fpy)),
        None => arc_safe_angle(sx, sy, sr, fpx, fpy, tr, ships)
            .map(|(a, t)| (a, t, fpx, fpy)),
    }
}

/// Exhaustive scan: find the earliest valid intercept window. Every candidate
/// is forward-sim verified before being accepted. Returns
/// `(angle, turns, target_x, target_y)` on success.
pub fn search_safe_intercept(
    sx: f64, sy: f64, sr: f64,
    target_id: i64,
    ships: i64,
    cache: &EntityCache,
    tolerance: Option<i64>,
) -> Option<(f64, i64, f64, f64)> {
    let target = cache.get(target_id)?;
    let tr = target.radius;
    let tolerance = tolerance.unwrap_or(target.tolerance);

    let mut max_turns = HORIZON;
    if target.is_comet() {
        max_turns = max_turns.min((cache.remaining_life(target_id) - 1).max(0));
    }

    for candidate_turns in 1..=max_turns {
        let Some([px, py]) = cache.position(target_id, candidate_turns) else {
            continue;
        };

        let est = estimate_arrival(sx, sy, sr, px, py, tr, ships)
            .or_else(|| arc_safe_angle(sx, sy, sr, px, py, tr, ships));
        let Some((_, turns)) = est else {
            continue;
        };
        if (turns - candidate_turns).abs() > tolerance {
            continue;
        }

        let actual_turns = turns.max(candidate_turns);
        let Some([apx, apy]) = cache.position(target_id, actual_turns) else {
            continue;
        };

        let confirm = estimate_arrival(sx, sy, sr, apx, apy, tr, ships)
            .or_else(|| arc_safe_angle(sx, sy, sr, apx, apy, tr, ships));
        let Some((angle_out, turns_out)) = confirm else {
            continue;
        };

        if (turns_out - actual_turns).abs() > tolerance {
            continue;
        }

        // Verify before accepting — stops false positives in exhaustive search.
        if verify_shot_hits(
            sx, sy, sr, angle_out, turns_out, ships,
            target_id, cache,
        ) {
            return Some((angle_out, turns_out, apx, apy));
        }
    }

    None
}

/// Public solver. Returns `(angle, turns, target_x, target_y)` or `None`.
/// Every `Some` result is verified by `verify_shot_hits` — callers will
/// never send a fleet this predicts will miss.
///
/// Pipeline: `aim_raw` (fast iterative) → `verify_shot_hits` gate → fall
/// back to `search_safe_intercept` (exhaustive, verifies internally) → `None`.
pub fn aim_with_prediction(
    sx: f64, sy: f64, sr: f64,
    target_id: i64,
    ships: i64,
    cache: &EntityCache,
) -> Option<(f64, i64, f64, f64)> {
    if let Some(res) = aim_raw(sx, sy, sr, target_id, ships, cache) {
        let (angle, turns, _, _) = res;
        if verify_shot_hits(sx, sy, sr, angle, turns, ships, target_id, cache) {
            return Some(res);
        }
    }

    // Raw missed or failed to verify — exhaustive search (verifies internally).
    search_safe_intercept(sx, sy, sr, target_id, ships, cache, None)
}


// ── Timeline Helpers ──────────────────────────────────────────────────────────
// Forward simulation with initial timeline and hypothetical queries

/// Per-planet arrival ledger: `{planet_id → [ArrivalEvent, ...]}`.
/// Built once per turn via [`TimelineCache::build`].
pub type ArrivalsByPlanet = HashMap<i64, Vec<ArrivalEvent>>;

/// Same-turn combat resolution. Arrivals aggregated by owner; the top two
/// attackers cancel out; the survivor fights the garrison. Top-2 ties
/// neutralise to ownerless (`-1`, 0 ships).
pub fn resolve_arrival_event(
    owner: i64,
    garrison: i64,
    arrivals: &[ArrivalEvent],
) -> (i64, i64) {
    if arrivals.is_empty() {
        return (owner, garrison.max(0));
    }
    let mut by_owner: HashMap<i64, i64> = HashMap::new();
    for ev in arrivals {
        *by_owner.entry(ev.owner).or_insert(0) += ev.ships;
    }
    if by_owner.is_empty() {
        return (owner, garrison.max(0));
    }
    let mut sorted: Vec<(i64, i64)> = by_owner.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let (top_owner, top_ships) = sorted[0];
    let (survivor_owner, survivor_ships) = if sorted.len() > 1 {
        let second_ships = sorted[1].1;
        if top_ships == second_ships {
            (-1i64, 0i64)
        } else {
            (top_owner, top_ships - second_ships)
        }
    } else {
        (top_owner, top_ships)
    };

    if survivor_ships <= 0 {
        return (owner, garrison.max(0));
    }
    if owner == survivor_owner {
        return (owner, garrison + survivor_ships);
    }
    let new_garrison = garrison - survivor_ships;
    if new_garrison < 0 {
        (survivor_owner, -new_garrison)
    } else {
        (owner, new_garrison)
    }
}

/// Filter, clamp, and sort raw arrivals into a clean per-turn event list:
/// drops non-positive ship counts, pulls every `turns` up to at least 1,
/// drops anything past `horizon`, then sorts by ETA ascending.
pub fn normalize_arrivals(arrivals: &[ArrivalEvent], horizon: i64) -> Vec<ArrivalEvent> {
    let mut out: Vec<ArrivalEvent> = arrivals
        .iter()
        .filter(|ev| ev.ships > 0)
        .map(|ev| ArrivalEvent {
            turns: ev.turns.max(1),
            owner: ev.owner,
            ships: ev.ships,
        })
        .filter(|ev| ev.turns <= horizon)
        .collect();
    out.sort_by_key(|ev| ev.turns);
    out
}

/// Forward-simulated state for one planet across turns `0..=horizon`.
/// Indexable directly by turn — `owner_at[t]` and `ships_at[t]` are the
/// post-combat snapshot at end of turn `t`.
#[derive(Debug, Clone)]
pub struct PlanetTimeline {
    pub owner_at: Vec<i64>,
    pub ships_at: Vec<i64>,
    /// Minimum garrison that, if kept on the planet, still survives every
    /// arrival through `horizon` (binary-searched). Only meaningful when the
    /// planet currently belongs to `player`.
    pub keep_needed: i64,
    /// Smallest garrison observed while `player` continuously owns the
    /// planet. 0 when `player` does not currently own it.
    pub min_owned: i64,
    /// First turn within the horizon where an enemy arrival lands while we
    /// own the planet.
    pub first_enemy: Option<i64>,
    /// Turn we lose the planet to a non-player owner, if it falls within the
    /// horizon.
    pub fall_turn: Option<i64>,
    /// `false` iff even keeping every current ship can't hold the planet.
    pub holds_full: bool,
    pub horizon: i64,
}

/// Turn-by-turn rollout of one planet under a given arrival schedule.
/// Applies production each turn (only while owned), then resolves arrivals
/// via `resolve_arrival_event`. Records the trajectory plus queryable summaries.
///
/// `expiry_turn`, when set, is the turn the planet leaves the board. Turns
/// at or past expiry are recorded as ownerless with zero ships.
pub fn simulate_planet_timeline(
    planet: &Planet,
    arrivals: &[ArrivalEvent],
    player: i64,
    horizon: i64,
    expiry_turn: Option<i64>,
) -> PlanetTimeline {
    let horizon = horizon.max(0);
    let effective_horizon = match expiry_turn {
        Some(exp) => horizon.min((exp - 1).max(0)),
        None => horizon,
    };
    let events = normalize_arrivals(arrivals, effective_horizon);

    let len = (horizon + 1) as usize;
    let mut by_turn: Vec<Vec<ArrivalEvent>> = vec![Vec::new(); len];
    for ev in &events {
        by_turn[ev.turns as usize].push(*ev);
    }

    let mut owner = planet.owner;
    let mut garrison = planet.ships;
    let mut owner_at: Vec<i64> = vec![owner; len];
    let mut ships_at: Vec<i64> = vec![garrison.max(0); len];
    let mut min_owned: i64 = if owner == player { garrison } else { 0 };
    let mut first_enemy: Option<i64> = None;
    let mut fall_turn: Option<i64> = None;

    for turn in 1..=effective_horizon {
        if owner != -1 {
            garrison += planet.production;
        }
        let group = &by_turn[turn as usize];
        let prev_owner = owner;
        if !group.is_empty() {
            if prev_owner == player
                && first_enemy.is_none()
                && group.iter().any(|ev| ev.owner != -1 && ev.owner != player)
            {
                first_enemy = Some(turn);
            }
            let (no, ng) = resolve_arrival_event(owner, garrison, group);
            owner = no;
            garrison = ng;
            if prev_owner == player && owner != player && fall_turn.is_none() {
                fall_turn = Some(turn);
            }
        }
        owner_at[turn as usize] = owner;
        ships_at[turn as usize] = garrison.max(0);
        if owner == player {
            min_owned = min_owned.min(garrison);
        }
    }

    // Past expiry the planet doesn't exist: no owner, no ships.
    for turn in (effective_horizon + 1)..=horizon {
        owner_at[turn as usize] = -1;
        ships_at[turn as usize] = 0;
    }

    let mut keep_needed: i64 = 0;
    let mut holds_full = true;
    if planet.owner == player {
        let survives = |keep: i64| -> bool {
            let mut sim_owner = planet.owner;
            let mut sim_garrison = keep;
            for turn in 1..=effective_horizon {
                if sim_owner != -1 {
                    sim_garrison += planet.production;
                }
                let group = &by_turn[turn as usize];
                if !group.is_empty() {
                    let (no, ng) =
                        resolve_arrival_event(sim_owner, sim_garrison, group);
                    sim_owner = no;
                    sim_garrison = ng;
                    if sim_owner != player {
                        return false;
                    }
                }
            }
            sim_owner == player
        };

        if survives(planet.ships) {
            let (mut lo, mut hi) = (0i64, planet.ships);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if survives(mid) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            keep_needed = lo;
        } else {
            holds_full = false;
            keep_needed = planet.ships;
        }
    }

    PlanetTimeline {
        owner_at,
        ships_at,
        keep_needed,
        min_owned: if planet.owner == player {
            min_owned.max(0)
        } else {
            0
        },
        first_enemy,
        fall_turn,
        holds_full,
        horizon,
    }
}

/// Reads `(owner, ships)` out of a timeline at `arrival_turn`. Clamps the
/// query into `[0, horizon]` so callers don't have to bounds-check.
pub fn state_at_timeline(timeline: &PlanetTimeline, arrival_turn: i64) -> (i64, i64) {
    let turn = arrival_turn.max(0).min(timeline.horizon) as usize;
    (timeline.owner_at[turn], timeline.ships_at[turn].max(0))
}

/// Checkpointed re-simulation: reuses `baseline` through `start_turn - 1`
/// and re-runs only `start_turn..=horizon` with the full arrival list.
///
/// Precondition: arrivals differing from baseline must land at turn
/// `>= start_turn`; earlier arrivals are assumed already in `baseline`.
///
/// Only `owner_at`/`ships_at` are valid; per-player metrics are left at
/// defaults. Use `simulate_planet_timeline` if those are needed.
pub fn simulate_planet_timeline_from(
    planet: &Planet,
    baseline: &PlanetTimeline,
    start_turn: i64,
    arrivals: &[ArrivalEvent],
    expiry_turn: Option<i64>,
) -> PlanetTimeline {
    let horizon = baseline.horizon;
    let start_turn = start_turn.clamp(1, horizon.max(1));
    let effective_horizon = match expiry_turn {
        Some(exp) => horizon.min((exp - 1).max(0)),
        None => horizon,
    };
    let len = (horizon + 1) as usize;

    let events = normalize_arrivals(arrivals, effective_horizon);
    let mut by_turn: Vec<Vec<ArrivalEvent>> = vec![Vec::new(); len];
    for ev in &events {
        if ev.turns >= start_turn {
            by_turn[ev.turns as usize].push(*ev);
        }
    }

    // Reuse baseline state for the unchanged prefix; only turns >= start_turn
    // get rewritten below.
    let mut owner_at = baseline.owner_at.clone();
    let mut ships_at = baseline.ships_at.clone();
    let checkpoint_idx = (start_turn - 1) as usize;
    let mut owner = owner_at[checkpoint_idx];
    let mut garrison = ships_at[checkpoint_idx];

    for turn in start_turn..=effective_horizon {
        if owner != -1 {
            garrison += planet.production;
        }
        let group = &by_turn[turn as usize];
        if !group.is_empty() {
            let (no, ng) = resolve_arrival_event(owner, garrison, group);
            owner = no;
            garrison = ng;
        }
        owner_at[turn as usize] = owner;
        ships_at[turn as usize] = garrison.max(0);
    }

    // Past expiry the planet doesn't exist.
    for turn in (effective_horizon + 1).max(start_turn)..=horizon {
        owner_at[turn as usize] = -1;
        ships_at[turn as usize] = 0;
    }

    PlanetTimeline {
        owner_at,
        ships_at,
        keep_needed: 0,
        min_owned: 0,
        first_enemy: None,
        fall_turn: None,
        holds_full: true,
        horizon,
    }
}

/// One-call cache that holds both the arrival ledger and per-planet baseline
/// timelines, built from a single `SimProbe` rollout.
///
/// Typical use: call `TimelineCache::build` once per bot turn, then pass the
/// cache to capture/hold queries (`min_ships_to_own_by`,
/// `reinforcement_needed_to_hold_until`) and any per-planet timeline reads.
/// Subsequent hypothetical-arrival queries pay only for the planets they
/// touch, starting from the baseline checkpoint at the arrival turn.
#[derive(Debug, Clone)]
pub struct TimelineCache {
    pub player: i64,
    pub horizon: i64,
    pub ledger: ArrivalsByPlanet,
    pub baselines: HashMap<i64, PlanetTimeline>,
    /// Turn at which a planet leaves the board, populated only for planets
    /// that expire within `horizon` (i.e. comets near end of life). Missing
    /// entry means the planet survives the entire window.
    pub expiry_at: HashMap<i64, i64>,
}

impl TimelineCache {
    /// Build the cache from a single SimProbe rollout. `O(horizon * |planets|)`.
    pub fn build(
        state: &EngineState,
        player: i64,
        horizon: i64,
        entity_cache: &EntityCache,
    ) -> Self {
        let mut probe = SimProbe::from_engine(state);
        probe.step_n(horizon);
        let mut ledger = probe.collect_arrivals();
        for planet in &state.planets {
            ledger.entry(planet.id).or_default();
        }

        let mut baselines = HashMap::with_capacity(state.planets.len());
        let mut expiry_at: HashMap<i64, i64> = HashMap::new();
        for planet in &state.planets {
            let arrivals = ledger
                .get(&planet.id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let expiry = expiry_within_horizon(entity_cache, planet.id, horizon);
            if let Some(exp) = expiry {
                expiry_at.insert(planet.id, exp);
            }
            baselines.insert(
                planet.id,
                simulate_planet_timeline(planet, arrivals, player, horizon, expiry),
            );
        }

        Self {
            player,
            horizon,
            ledger,
            baselines,
            expiry_at,
        }
    }

    /// Arrival list for a planet (empty if nothing is incoming or the planet
    /// isn't in the cache).
    pub fn arrivals(&self, planet_id: i64) -> &[ArrivalEvent] {
        self.ledger
            .get(&planet_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Baseline timeline for a planet, or `None` if the planet wasn't present
    /// when the cache was built.
    pub fn baseline(&self, planet_id: i64) -> Option<&PlanetTimeline> {
        self.baselines.get(&planet_id)
    }

    /// Turn at which a planet leaves the board, if within the cache's horizon.
    #[inline]
    pub fn expiry(&self, planet_id: i64) -> Option<i64> {
        self.expiry_at.get(&planet_id).copied()
    }
}

/// Returns the planet's expiry turn iff it falls within `horizon`. Static and
/// orbiting planets last the whole game, so they always return `None`.
fn expiry_within_horizon(
    entity_cache: &EntityCache,
    planet_id: i64,
    horizon: i64,
) -> Option<i64> {
    let life = entity_cache.remaining_life(planet_id);
    if life <= horizon { Some(life) } else { None }
}

/// Smallest ship count that, if it lands on `planet` at `arrival_turn` for
/// `attacker_owner`, makes them own the planet by `eval_turn`. Returns 0 if
/// the planet is already theirs at `eval_turn` without extras. Returns
/// `upper_bound + 1` when not achievable within budget.
///
/// `eval_turn` is clamped to `cache.horizon`; if `arrival_turn > eval_turn`
/// after clamping, returns `upper_bound + 1`.
pub fn min_ships_to_own_by(
    cache: &TimelineCache,
    planet: &Planet,
    attacker_owner: i64,
    arrival_turn: i64,
    eval_turn: i64,
    upper_bound: i64,
) -> i64 {
    let arrival_turn = arrival_turn.max(1);
    let eval_turn = eval_turn.max(1).min(cache.horizon);
    if arrival_turn > eval_turn {
        return upper_bound + 1;
    }

    let baseline = cache.baseline(planet.id);
    let base_arrivals = cache.arrivals(planet.id);
    let expiry = cache.expiry(planet.id);

    // owner_at is viewpoint-independent — the cache's baseline gives the
    // right "no-extras" prediction for any attacker.
    if let Some(baseline) = baseline {
        if state_at_timeline(baseline, eval_turn).0 == attacker_owner {
            return 0;
        }
    } else {
        // Planet outside the cache (e.g. spawned later) — full sim fallback.
        let base_tl =
            simulate_planet_timeline(planet, base_arrivals, attacker_owner, eval_turn, expiry);
        if state_at_timeline(&base_tl, eval_turn).0 == attacker_owner {
            return 0;
        }
    }

    let mut scratch: Vec<ArrivalEvent> = Vec::with_capacity(base_arrivals.len() + 1);
    scratch.extend_from_slice(base_arrivals);
    scratch.push(ArrivalEvent {
        turns: arrival_turn,
        owner: attacker_owner,
        ships: 0,
    });
    let last = scratch.len() - 1;

    let owns_at = |ships: i64, buf: &mut [ArrivalEvent]| -> bool {
        buf[last].ships = ships;
        let tl = if let Some(baseline) = baseline {
            simulate_planet_timeline_from(planet, baseline, arrival_turn, buf, expiry)
        } else {
            simulate_planet_timeline(planet, buf, attacker_owner, eval_turn, expiry)
        };
        state_at_timeline(&tl, eval_turn).0 == attacker_owner
    };

    let hi_init = upper_bound.max(1);
    if !owns_at(hi_init, &mut scratch) {
        return hi_init + 1;
    }
    let (mut lo, mut hi) = (1i64, hi_init);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if owns_at(mid, &mut scratch) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Smallest reinforcement that arrives at `arrival_turn` and keeps
/// `cache.player` in continuous ownership through `hold_until`. If the planet
/// isn't currently `cache.player`'s, collapses to `min_ships_to_own_by` at
/// `hold_until`. Returns `upper_bound + 1` if no value in `1..=upper_bound` works.
pub fn reinforcement_needed_to_hold_until(
    cache: &TimelineCache,
    planet: &Planet,
    arrival_turn: i64,
    hold_until: i64,
    upper_bound: i64,
) -> i64 {
    let player = cache.player;
    let arrival_turn = arrival_turn.max(1);
    let hold_until = hold_until.max(arrival_turn).min(cache.horizon);

    if planet.owner != player {
        return min_ships_to_own_by(cache, planet, player, arrival_turn, hold_until, upper_bound);
    }

    let baseline = cache.baseline(planet.id);
    let base_arrivals = cache.arrivals(planet.id);
    let expiry = cache.expiry(planet.id);

    let mut scratch: Vec<ArrivalEvent> = Vec::with_capacity(base_arrivals.len() + 1);
    scratch.extend_from_slice(base_arrivals);
    scratch.push(ArrivalEvent {
        turns: arrival_turn,
        owner: player,
        ships: 0,
    });
    let last = scratch.len() - 1;

    let holds = |ships: i64, buf: &mut [ArrivalEvent]| -> bool {
        buf[last].ships = ships;
        let tl = if let Some(baseline) = baseline {
            simulate_planet_timeline_from(planet, baseline, arrival_turn, buf, expiry)
        } else {
            simulate_planet_timeline(planet, buf, player, hold_until, expiry)
        };
        (arrival_turn..=hold_until).all(|t| tl.owner_at[t as usize] == player)
    };

    let hi_init = upper_bound.max(1);
    if !holds(hi_init, &mut scratch) {
        return hi_init + 1;
    }
    let (mut lo, mut hi) = (1i64, hi_init);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if holds(mid, &mut scratch) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}
