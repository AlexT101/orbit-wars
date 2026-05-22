#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::constants::{
    CENTER, EDGE_AIM_FRACS, HORIZON, FWD_ITER_MAX, LAUNCH_CLEARANCE, MAX_SHIP_SPEED, ROTATION_LIMIT, SUN_RADIUS,
};

use crate::engine::{Planet, Fleet, CometGroup, EngineState};
use crate::sim_probe::SimProbe;
pub use crate::sim_probe::ArrivalEvent;

#[derive(Debug, Clone, Copy)]
pub struct InitialPlanetPos {
    pub x: f64,
    pub y: f64,
}

/// Euclidean distance between two points
#[inline]
pub fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    crate::engine::distance((ax, ay), (bx, by))
}

/// Distance of a planet's centre from the sun
#[inline]
pub fn orbital_radius(px: f64, py: f64) -> f64 {
    dist(px, py, CENTER, CENTER)
}

/// A planet is static (does not orbit each turn) when its orbital radius plus its own physical radius equals or exceeds 50
#[inline]
pub fn is_static_planet(px: f64, py: f64, radius: f64) -> bool {
    orbital_radius(px, py) + radius >= ROTATION_LIMIT
}

/// Official speed formula from the game spec:
/// `speed = 1 + (maxSpeed-1) * (log(ships)/log(1000))^1.5`.
/// Speed follows a logarithmic curve: 1 ship moves at 1 unit/turn; ~1000 ships reaches the cap of 6 units/turn.
/// Delegates to the engine; `ships.max(1)` guards against `ln(0) = -inf`.
#[inline]
pub fn fleet_speed(ships: i64) -> f64 {
    crate::engine::fleet_speed(ships.max(1), MAX_SHIP_SPEED)
}

/// Minimum distance from a point to a line segment. Delegates to the engine
/// (which uses the same projection-clamp-distance math) so parity is exact.
#[inline]
pub fn point_to_segment_distance(
    px: f64, py: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> f64 {
    crate::engine::point_to_segment_distance((px, py), (x1, y1), (x2, y2))
}

/// Continuous collision test: returns `true` if the movement segment A→B
/// passes within distance `r` of centre C. This mirrors the game engine's
/// own collision check.
#[inline]
pub fn segment_intersects_circle(
    ax: f64, ay: f64,
    bx: f64, by: f64,
    cx: f64, cy: f64,
    r: f64,
) -> bool {
    point_to_segment_distance(cx, cy, ax, ay, bx, by) <= r
}

/// True if the fleet's path segment comes within `SUN_RADIUS` of the sun
/// centre — engine-exact, no buffer.
#[inline]
pub fn segment_hits_sun(
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> bool {
    point_to_segment_distance(CENTER, CENTER, x1, y1, x2, y2) < SUN_RADIUS
}

/// Fleet spawn position. The fleet does not spawn at the planet's centre; it
/// spawns just outside the planet's surface (radius + 0.1) in the aimed
/// direction, matching the game engine's launch logic.
#[inline]
pub fn launch_point(sx: f64, sy: f64, sr: f64, angle: f64) -> (f64, f64) {
    let c = sr + LAUNCH_CLEARANCE;
    (sx + angle.cos() * c, sy + angle.sin() * c)
}

// ─────────────────────────────────────────────────────────────────────────
// 4. Orbit and comet position prediction
// ─────────────────────────────────────────────────────────────────────────

/// Forward-projects an orbiting planet's position. Uses the **initial** planet
/// position (not the current one) to anchor the orbital radius, preventing
/// floating-point drift over many turns. Applies
/// `angular_velocity * turns_ahead` to the current angle. Returns the current
/// position unchanged for static planets.
///
/// Spec note: planet rotation happens *after* fleet movement each turn, so a
/// fleet arriving on turn T has experienced T full rotations.
pub fn predict_planet_position(
    planet_id: i64,
    cur_x: f64, cur_y: f64, radius: f64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    angular_velocity: f64,
    turns_ahead: i64,
) -> (f64, f64) {
    let Some(init) = initial_by_id.get(&planet_id) else {
        return (cur_x, cur_y);
    };
    if is_static_planet(init.x, init.y, radius) {
        return (cur_x, cur_y);
    }
    let r = orbital_radius(init.x, init.y);
    let cur_ang = (cur_y - CENTER).atan2(cur_x - CENTER);
    let new_ang = cur_ang + angular_velocity * turns_ahead as f64;
    (
        CENTER + r * new_ang.cos(),
        CENTER + r * new_ang.sin(),
    )
}

/// Looks up a comet's pre-computed path. Comets follow pre-computed elliptical
/// trajectories stored in each group's `paths`. This function indexes into
/// `paths[idx][path_index + turns]` to find the comet's world position at the
/// requested turn offset.
pub fn predict_comet_position(
    planet_id: i64,
    comets: &[CometGroup],
    turns: i64,
) -> Option<(f64, f64)> {
    for group in comets {
        let Some(idx) = group.planet_ids.iter().position(|&p| p == planet_id) else {
            continue;
        };
        if idx >= group.paths.len() {
            return None;
        }
        let future = group.path_index + turns;
        let path = &group.paths[idx];
        if future >= 0 && (future as usize) < path.len() {
            let p = path[future as usize];
            return Some((p[0], p[1]));
        }
        return None;
    }
    None
}

/// How many turns until the comet leaves the board. Returns the number of path
/// entries remaining from the current `path_index`. Chasing a comet that
/// expires in 2 turns with a 10-turn flight is pointless, and this value
/// gates such decisions.
pub fn comet_remaining_life(planet_id: i64, comets: &[CometGroup]) -> i64 {
    for group in comets {
        let Some(idx) = group.planet_ids.iter().position(|&p| p == planet_id) else {
            continue;
        };
        if idx < group.paths.len() {
            return (group.paths[idx].len() as i64 - group.path_index).max(0);
        }
    }
    0
}

/// Unified dispatcher for orbiting planets and comets. Routes to the
/// appropriate predictor based on whether the target ID appears in
/// `comet_ids`. All higher-level functions call this rather than the
/// individual predictors.
pub fn predict_target_position(
    planet_id: i64,
    cur_x: f64, cur_y: f64, radius: f64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    angular_velocity: f64,
    comets: &[CometGroup],
    comet_ids: &HashSet<i64>,
    turns: i64,
) -> Option<(f64, f64)> {
    if comet_ids.contains(&planet_id) {
        return predict_comet_position(planet_id, comets, turns);
    }
    Some(predict_planet_position(
        planet_id, cur_x, cur_y, radius,
        initial_by_id, angular_velocity, turns,
    ))
}

/// Quick check for whether a target has orbital motion. Returns `true` for
/// comets (always moving) and for planets whose orbital radius keeps them
/// inside the rotation threshold. Used to decide whether prediction is
/// necessary at all.
pub fn target_can_move(
    planet_id: i64,
    _cur_x: f64, _cur_y: f64, radius: f64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    comet_ids: &HashSet<i64>,
) -> bool {
    if comet_ids.contains(&planet_id) {
        return true;
    }
    let Some(init) = initial_by_id.get(&planet_id) else {
        return false;
    };
    !is_static_planet(init.x, init.y, radius)
}

// ─────────────────────────────────────────────────────────────────────────
// 5. Arrival estimation and safe-path geometry
// ─────────────────────────────────────────────────────────────────────────

/// Computes angle and travel distance while checking for sun collision.
///
/// The sun-collision check uses the segment from launch point to the *target
/// surface entry point* (not the target centre). This matches the game engine
/// exactly — a fleet is destroyed when it enters the sun's exclusion zone,
/// not merely when the line passes through the centre.
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

#[inline]
fn fractional_turns(total_d: f64, ships: i64) -> f64 {
    total_d / fleet_speed(ships.max(1))
}

/// Integer-turn turn count for a direct shot (the value the game engine
/// actually uses).
pub fn estimate_arrival(
    sx: f64, sy: f64, sr: f64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
) -> Option<(f64, i64)> {
    let (angle, total_d) = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)?;
    let turns = (fractional_turns(total_d, ships).ceil() as i64).max(1);
    Some((angle, turns))
}

/// Fractional turn count used for convergence comparisons.
pub fn estimate_arrival_frac(
    sx: f64, sy: f64, sr: f64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
) -> Option<(f64, f64)> {
    let (angle, total_d) = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)?;
    Some((angle, fractional_turns(total_d, ships).max(1.0)))
}

/// Convenience wrapper returning only the turn count. Returns `10^9` when no
/// valid path exists, matching the notebook's sentinel.
pub fn travel_time(
    sx: f64, sy: f64, sr: f64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
) -> i64 {
    estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
        .map(|(_, turns)| turns)
        .unwrap_or(1_000_000_000)
}

/// Forward-simulation scan window. `window = max(8, turns / 2)` provides
/// enough headroom for slow (speed=1) fleets aimed at long intercepts to
/// still have runway to confirm the hit.
#[inline]
pub fn fwd_window(turns: i64) -> i64 {
    (turns / 2).max(8)
}

/// Sun-bypass chord sampling. When the direct path is blocked by the sun,
/// this samples seven aim-points distributed across the target's disk
/// (centre + fractions of the radius in both perpendicular directions) and
/// returns the one with the shortest clear path. Because fleets travel in
/// straight lines there are no curved routes — but a chord aimed at the edge
/// of a planet's disk can clear the sun even when the centre-aimed shot
/// cannot.
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

    // 1 centre + 2 mirrored offsets per fraction → 1 + 2 * N candidate aims.
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
        let turns = (entry_dist / fleet_speed(ships.max(1))).ceil().max(1.0) as i64;
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

// ─────────────────────────────────────────────────────────────────────────
// 6. Forward-simulation verification
// ─────────────────────────────────────────────────────────────────────────

/// Ground-truth forward-sim. Returns `true` only if the fleet physically hits
/// the target within the scan window. Used as a mandatory gate before any
/// target intercept is accepted, helping eliminate false positives from
/// predictions.
#[allow(clippy::too_many_arguments)]
pub fn verify_shot_hits(
    sx: f64, sy: f64, sr: f64,
    angle: f64, turns: i64, ships: i64,
    target_id: i64,
    tx: f64, ty: f64, tr: f64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    angular_velocity: f64,
    comets: &[CometGroup],
    comet_ids: &HashSet<i64>,
) -> bool {
    let speed = fleet_speed(ships.max(1));
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
        let Some((px, py)) = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, t,
        ) else {
            continue;
        };
        if segment_intersects_circle(pfx, pfy, fx, fy, px, py, tr) {
            return true;
        }
    }
    false
}

// ─────────────────────────────────────────────────────────────────────────
// 7. Dynamic tolerance
// ─────────────────────────────────────────────────────────────────────────

/// Maximum error in turns allowed for candidate intercept checks. Capped at 2
/// to avoid picking incorrect orbital positions. Comets get 2 (faster, less
/// predictable). Orbiting planets at >= 1 unit/turn get 2; else 1.
pub fn dynamic_tolerance(
    target_id: i64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    angular_velocity: f64,
    comet_ids: &HashSet<i64>,
) -> i64 {
    if comet_ids.contains(&target_id) {
        return 2;
    }
    let Some(init) = initial_by_id.get(&target_id) else {
        return 1;
    };
    let orb_r = orbital_radius(init.x, init.y);
    let orb_speed = orb_r * angular_velocity.abs();
    if orb_speed >= 1.0 { 2 } else { 1 }
}

// ─────────────────────────────────────────────────────────────────────────
// 8. Exhaustive intercept search
// ─────────────────────────────────────────────────────────────────────────

/// Exhaustive scan: find the earliest valid intercept window. Every candidate
/// is forward-sim verified before being accepted. Returns
/// `(angle, turns, target_x, target_y)` on success.
#[allow(clippy::too_many_arguments)]
pub fn search_safe_intercept(
    sx: f64, sy: f64, sr: f64,
    target_id: i64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    angular_velocity: f64,
    comets: &[CometGroup],
    comet_ids: &HashSet<i64>,
    tolerance: Option<i64>,
) -> Option<(f64, i64, f64, f64)> {
    let tolerance = tolerance.unwrap_or_else(|| {
        dynamic_tolerance(target_id, initial_by_id, angular_velocity, comet_ids)
    });
    let mut max_turns = HORIZON;
    if comet_ids.contains(&target_id) {
        max_turns = max_turns.min((comet_remaining_life(target_id, comets) - 1).max(0));
    }

    for candidate_turns in 1..=max_turns {
        let Some((px, py)) = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, candidate_turns,
        ) else {
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
        let Some((apx, apy)) = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, actual_turns,
        ) else {
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
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity, comets, comet_ids,
        ) {
            return Some((angle_out, turns_out, apx, apy));
        }
    }

    None
}

// ─────────────────────────────────────────────────────────────────────────
// 9. Iterative aiming solver
// ─────────────────────────────────────────────────────────────────────────

/// Iterative convergence solver. Calculates direct or sun-blocked trajectories
/// by repeatedly refining the predicted intercept position. All results are
/// UNVERIFIED — caller must verify via `verify_shot_hits`.
#[allow(clippy::too_many_arguments)]
pub fn aim_raw(
    sx: f64, sy: f64, sr: f64,
    target_id: i64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    angular_velocity: f64,
    comets: &[CometGroup],
    comet_ids: &HashSet<i64>,
) -> Option<(f64, i64, f64, f64)> {
    let tol = dynamic_tolerance(target_id, initial_by_id, angular_velocity, comet_ids);

    let mut est = match estimate_arrival_frac(sx, sy, sr, tx, ty, tr, ships) {
        Some(v) => v,
        None => {
            // Direct shot blocked; try edge aim, then fall back to a raw shot
            // if the target is static (won't move while we wait).
            if let Some((arc_angle, arc_turns)) = arc_safe_angle(sx, sy, sr, tx, ty, tr, ships) {
                return Some((arc_angle, arc_turns, tx, ty));
            }
            if !target_can_move(target_id, tx, ty, tr, initial_by_id, comet_ids) {
                let angle = (ty - sy).atan2(tx - sx);
                let total_d = (dist(sx, sy, tx, ty) - sr - LAUNCH_CLEARANCE - tr).max(0.0);
                let turns = ((total_d / fleet_speed(ships.max(1))).ceil() as i64).max(1);
                return Some((angle, turns, tx, ty));
            }
            return None;
        }
    };

    for _ in 0..FWD_ITER_MAX {
        let (_, turns_f) = est;
        let turns_i = turns_f.ceil() as i64;
        let Some((ntx, nty)) = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, turns_i,
        ) else {
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
    let Some((fpx, fpy)) = predict_target_position(
        target_id, tx, ty, tr,
        initial_by_id, angular_velocity,
        comets, comet_ids, final_turns,
    ) else {
        return None;
    };
    match estimate_arrival(sx, sy, sr, fpx, fpy, tr, ships) {
        Some((a, t)) => Some((a, t, fpx, fpy)),
        None => arc_safe_angle(sx, sy, sr, fpx, fpy, tr, ships)
            .map(|(a, t)| (a, t, fpx, fpy)),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 10. Main aiming solver
// ─────────────────────────────────────────────────────────────────────────

/// Public solver. Returns `(angle, turns, target_x, target_y)` or `None`.
///
/// Guarantee: every `Some` result is VERIFIED by `verify_shot_hits` before
/// being returned, so the caller will never send a fleet this solver predicts
/// will miss. False positives should be ~0.
///
/// Pipeline:
///   1. `aim_raw()`          — fast iterative convergence (unverified)
///   2. `verify_shot_hits()` — mandatory forward-sim gate
///   3. If raw fails verify → `search_safe_intercept()` (exhaustive, pre-verified)
///   4. If both fail        → `None` (shot correctly suppressed)
#[allow(clippy::too_many_arguments)]
pub fn aim_with_prediction(
    sx: f64, sy: f64, sr: f64,
    target_id: i64,
    tx: f64, ty: f64, tr: f64,
    ships: i64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    angular_velocity: f64,
    comets: &[CometGroup],
    comet_ids: &HashSet<i64>,
) -> Option<(f64, i64, f64, f64)> {
    if let Some(res) = aim_raw(
        sx, sy, sr, target_id, tx, ty, tr, ships,
        initial_by_id, angular_velocity, comets, comet_ids,
    ) {
        let (angle, turns, _, _) = res;
        if verify_shot_hits(
            sx, sy, sr, angle, turns, ships,
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity, comets, comet_ids,
        ) {
            return Some(res);
        }
    }

    // Raw missed or failed to verify — exhaustive search (verifies internally).
    search_safe_intercept(
        sx, sy, sr, target_id, tx, ty, tr, ships,
        initial_by_id, angular_velocity, comets, comet_ids,
        None,
    )
}

// ─────────────────────────────────────────────────────────────────────────
// 11. Ship probe helpers
// ─────────────────────────────────────────────────────────────────────────

/// Ship count candidate options derived from standard fractions of the
/// target/available ship counts.
pub fn probe_ship_candidates(need: i64, avail: i64) -> Vec<i64> {
    if need < 0 {
        return Vec::new();
    }
    let need_f = need as f64;
    let mut candidates = [
        (0.25 * need_f) as i64,
        (0.50 * need_f) as i64,
        (0.75 * need_f) as i64,
        need - 5,
        need,
        avail.min(need + 5),
        avail.min(need + 10),
    ];
    for c in candidates.iter_mut() {
        *c = (*c).max(1);
    }
    let mut out: Vec<i64> = candidates.to_vec();
    out.sort();
    out.dedup();
    out.retain(|c| *c >= 1 && *c <= avail);
    out
}

// ─────────────────────────────────────────────────────────────────────────
// 13. Population statistics & topology features
// ─────────────────────────────────────────────────────────────────────────

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

/// Shortest distance from `(px, py)` to the centre of any planet in `set`.
/// Returns `f64::INFINITY` for an empty set so callers can compare freely.
pub fn nearest_distance_to_set(px: f64, py: f64, set: &[Planet]) -> f64 {
    set.iter()
        .map(|p| dist(px, py, p.x, p.y))
        .fold(f64::INFINITY, f64::min)
}

/// Indirect threat/wealth features: sum of `production / (distance + 12)`
/// over every other planet, split by owner class (friendly, neutral, enemy)
/// from `player`'s point of view. Cheap topological pressure metric.
pub fn indirect_features(
    planet: &Planet,
    planets: &[Planet],
    player: i64,
) -> (f64, f64, f64) {
    let mut friendly = 0.0;
    let mut neutral = 0.0;
    let mut enemy = 0.0;
    for other in planets {
        if other.id == planet.id {
            continue;
        }
        let d = dist(planet.x, planet.y, other.x, other.y);
        if d < 1.0 {
            continue;
        }
        let factor = other.production as f64 / (d + 12.0);
        if other.owner == player {
            friendly += factor;
        } else if other.owner == -1 {
            neutral += factor;
        } else {
            enemy += factor;
        }
    }
    (friendly, neutral, enemy)
}

/// Returns `(planet, distance)` pairs sorted ascending by distance from
/// `(tx, ty)`. NaN distances (shouldn't happen but be defensive) sort as
/// equal to keep the sort total.
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

/// Top-K nearest sources to a target. Lexicographic tiebreak `(distance,
/// -ships, id)` matches obnext's deterministic preference for nearer, then
/// larger, then lower-id sources. Returns all sources when `top_k` is 0 or
/// `sources.len() <= top_k`.
pub fn nearest_sources_to_target(
    target: &Planet,
    sources: &[Planet],
    top_k: usize,
) -> Vec<Planet> {
    if top_k == 0 || sources.len() <= top_k {
        return sources.to_vec();
    }
    let mut indexed: Vec<(f64, Planet)> = sources
        .iter()
        .map(|s| (dist(s.x, s.y, target.x, target.y), s.clone()))
        .collect();
    indexed.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.ships.cmp(&a.1.ships))
            .then_with(|| a.1.id.cmp(&b.1.id))
    });
    indexed.into_iter().take(top_k).map(|(_, p)| p).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// 14. Bulk orbital trajectory precompute
// ─────────────────────────────────────────────────────────────────────────

/// Pre-computes `(x, y)` positions for an orbiting planet at turns
/// `1..=turns`. Amortises the trig when the same planet is queried many
/// times (e.g. swept collision checks). Returns a vector of `turns` entries;
/// for static or unknown planets every entry is `(cur_x, cur_y)`.
pub fn planet_trajectory(
    planet_id: i64,
    cur_x: f64, cur_y: f64, radius: f64,
    initial_by_id: &HashMap<i64, InitialPlanetPos>,
    angular_velocity: f64,
    turns: i64,
) -> Vec<(f64, f64)> {
    (1..=turns)
        .map(|t| {
            predict_planet_position(
                planet_id, cur_x, cur_y, radius,
                initial_by_id, angular_velocity, t,
            )
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────
// 15. Arrival ledger — engine-driven via SimProbe
// ─────────────────────────────────────────────────────────────────────────

/// Per-planet arrival ledger: `{planet_id → [ArrivalEvent, ...]}`.
pub type ArrivalsByPlanet = HashMap<i64, Vec<ArrivalEvent>>;

/// Build a per-planet arrival ledger by running the engine forward via
/// `SimProbe` for `horizon` turns and bucketing the `FleetLanded` events by
/// destination. Engine-exact: the swept-circle collision detection that
/// determines who lands where is the same code path the real engine step uses.
///
/// Every planet currently in `state.planets` is guaranteed to appear in the
/// result (with an empty vec when nothing is incoming), so callers can index
/// without `Option` checks.
pub fn build_arrival_ledger(state: &EngineState, horizon: i64) -> ArrivalsByPlanet {
    let mut probe = SimProbe::from_engine(state);
    probe.step_n(horizon);
    let mut ledger = probe.collect_arrivals();
    for planet in &state.planets {
        ledger.entry(planet.id).or_default();
    }
    ledger
}

// ─────────────────────────────────────────────────────────────────────────
// 16. Per-planet timeline simulation
// ─────────────────────────────────────────────────────────────────────────

/// Engine-faithful same-turn combat resolution. Arrivals are aggregated by
/// owner; the top two attackers cancel out; the survivor then fights the
/// garrison. Ties at the top neutralise to ownerless (`-1`, 0 ships).
/// Mirrors the engine's exact combat order — the foundation of every
/// timeline query.
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
/// Applies production each turn (only while owned), then resolves the
/// arrivals landing that turn via `resolve_arrival_event`. Records the
/// owner/ship trajectory plus several queryable summaries.
pub fn simulate_planet_timeline(
    planet: &Planet,
    arrivals: &[ArrivalEvent],
    player: i64,
    horizon: i64,
) -> PlanetTimeline {
    let horizon = horizon.max(0);
    let events = normalize_arrivals(arrivals, horizon);

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

    for turn in 1..=horizon {
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

    // keep_needed: smallest starting garrison that survives every arrival.
    let mut keep_needed: i64 = 0;
    let mut holds_full = true;
    if planet.owner == player {
        let survives = |keep: i64| -> bool {
            let mut sim_owner = planet.owner;
            let mut sim_garrison = keep;
            for turn in 1..=horizon {
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

/// Checkpointed re-simulation: starts from the baseline's state at turn
/// `start_turn - 1` and re-runs only `start_turn..=horizon` with the full
/// (baseline + hypothetical) arrival list. Used by capture/hold queries when
/// only a hypothetical fleet at a known turn perturbs the schedule — the
/// unchanged prefix `0..start_turn` is read directly from `baseline`.
///
/// Precondition: every arrival in `arrivals` whose turn differs from the
/// baseline's must land at turn `>= start_turn`. Earlier-than-`start_turn`
/// arrivals are taken as already-accounted-for in `baseline` and skipped.
///
/// Returns a `PlanetTimeline` whose `owner_at` and `ships_at` fields are
/// fully valid; the per-player metrics (`keep_needed`, `min_owned`,
/// `first_enemy`, `fall_turn`, `holds_full`) are NOT recomputed and are left
/// at safe defaults. Callers that need them must use the full
/// `simulate_planet_timeline` instead.
pub fn simulate_planet_timeline_from(
    planet: &Planet,
    baseline: &PlanetTimeline,
    start_turn: i64,
    arrivals: &[ArrivalEvent],
) -> PlanetTimeline {
    let horizon = baseline.horizon;
    let start_turn = start_turn.clamp(1, horizon.max(1));
    let len = (horizon + 1) as usize;

    let events = normalize_arrivals(arrivals, horizon);
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

    for turn in start_turn..=horizon {
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
}

impl TimelineCache {
    /// Build the cache from a single SimProbe rollout. `O(horizon * |planets|)`
    /// total: one rollout for the ledger, plus one per-planet timeline sim
    /// each (each `O(horizon)`).
    pub fn build(state: &EngineState, player: i64, horizon: i64) -> Self {
        let mut probe = SimProbe::from_engine(state);
        probe.step_n(horizon);
        let mut ledger = probe.collect_arrivals();
        for planet in &state.planets {
            ledger.entry(planet.id).or_default();
        }

        let mut baselines = HashMap::with_capacity(state.planets.len());
        for planet in &state.planets {
            let arrivals = ledger
                .get(&planet.id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            baselines.insert(
                planet.id,
                simulate_planet_timeline(planet, arrivals, player, horizon),
            );
        }

        Self {
            player,
            horizon,
            ledger,
            baselines,
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
}

// ─────────────────────────────────────────────────────────────────────────
// 17. Capture / hold queries (binary-search on the timeline)
// ─────────────────────────────────────────────────────────────────────────

/// Smallest ship count that, if it lands on `planet` at `arrival_turn` for
/// `attacker_owner`, makes them own the planet by `eval_turn`. Returns 0 when
/// the planet is already going to belong to `attacker_owner` at `eval_turn`
/// without any extra ships. Returns `upper_bound + 1` to signal "not
/// achievable within the budget".
///
/// Reuses `cache`'s baseline timeline as a checkpoint at turn `arrival_turn`,
/// so each binary-search iteration only re-simulates `arrival_turn..=eval_turn`
/// instead of the full horizon.
///
/// `eval_turn` is clamped to `cache.horizon`. If `arrival_turn > eval_turn`
/// after clamping, returns `upper_bound + 1` (no valid action).
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

    // owner_at is viewpoint-independent, so the cache's baseline (built for
    // cache.player) gives the right "no-extras" prediction for any attacker.
    if let Some(baseline) = baseline {
        if state_at_timeline(baseline, eval_turn).0 == attacker_owner {
            return 0;
        }
    } else {
        // Planet outside the cache (e.g. spawned later). Fall back to a full
        // sim — same behaviour as before, just less efficient.
        let base_tl = simulate_planet_timeline(planet, base_arrivals, attacker_owner, eval_turn);
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
            simulate_planet_timeline_from(planet, baseline, arrival_turn, buf)
        } else {
            simulate_planet_timeline(planet, buf, attacker_owner, eval_turn)
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
/// is not currently `cache.player`'s, this collapses to `min_ships_to_own_by`
/// evaluated at `hold_until`. Returns `upper_bound + 1` if no value in
/// `1..=upper_bound` works.
///
/// Like `min_ships_to_own_by`, uses the cached baseline as a checkpoint at
/// `arrival_turn` to avoid re-simulating the unchanged prefix.
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
            simulate_planet_timeline_from(planet, baseline, arrival_turn, buf)
        } else {
            simulate_planet_timeline(planet, buf, player, hold_until)
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

// ─────────────────────────────────────────────────────────────────────────
// 18. Strategy primitives derived from the arrival ledger
// ─────────────────────────────────────────────────────────────────────────

/// Two enemy fleets converging on the same planet within a narrow ETA window
/// — a free-attrition opportunity. Pairs are unordered; the earlier-arriving
/// fleet is reported first.
#[derive(Debug, Clone, Copy)]
pub struct EnemyCrash {
    pub target_id: i64,
    pub crash_turn: i64,
    pub owners: (i64, i64),
    pub ships: (i64, i64),
}

/// Scans the arrival ledger for pairs of arrivals from different non-player,
/// non-neutral owners landing on the same planet within `eta_window` turns of
/// each other, with combined strength of at least `min_total_ships`.
///
/// Within a single planet's arrival list, sorted by ETA ascending, the inner
/// loop breaks as soon as the ETA gap exceeds the window — correctly because
/// subsequent entries are even further out.
pub fn detect_enemy_crashes(
    arrivals_by_planet: &ArrivalsByPlanet,
    player: i64,
    eta_window: i64,
    min_total_ships: i64,
) -> Vec<EnemyCrash> {
    let mut crashes = Vec::new();
    for (&target_id, arrivals) in arrivals_by_planet {
        let mut enemy_events: Vec<ArrivalEvent> = arrivals
            .iter()
            .filter(|ev| ev.owner != -1 && ev.owner != player && ev.ships > 0)
            .map(|ev| ArrivalEvent {
                turns: ev.turns.max(1),
                owner: ev.owner,
                ships: ev.ships,
            })
            .collect();
        enemy_events.sort_by_key(|ev| ev.turns);

        for i in 0..enemy_events.len() {
            let a = enemy_events[i];
            for j in (i + 1)..enemy_events.len() {
                let b = enemy_events[j];
                if a.owner == b.owner {
                    continue;
                }
                if (a.turns - b.turns).abs() > eta_window {
                    break;
                }
                if a.ships + b.ships < min_total_ships {
                    continue;
                }
                crashes.push(EnemyCrash {
                    target_id,
                    crash_turn: a.turns.max(b.turns),
                    owners: (a.owner, b.owner),
                    ships: (a.ships, b.ships),
                });
            }
        }
    }
    crashes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Configuration;

    fn planet(id: i64, owner: i64, ships: i64, production: i64) -> Planet {
        Planet {
            id,
            owner,
            x: 60.0,
            y: 60.0,
            radius: 2.0,
            ships,
            production,
        }
    }

    /// Checkpointed re-sim must produce the same `owner_at`/`ships_at` as a
    /// full sim when the hypothetical fleet lands at `start_turn`.
    #[test]
    fn checkpointed_timeline_matches_full() {
        let p = planet(0, 0, 30, 2);
        let player = 0;
        let horizon = 20;

        let base_arrivals = vec![
            ArrivalEvent { turns: 3, owner: 1, ships: 10 },
            ArrivalEvent { turns: 7, owner: 1, ships: 25 },
            ArrivalEvent { turns: 12, owner: 0, ships: 5 },
        ];

        let baseline = simulate_planet_timeline(&p, &base_arrivals, player, horizon);

        let mut full_arrivals = base_arrivals.clone();
        full_arrivals.push(ArrivalEvent {
            turns: 8,
            owner: 0,
            ships: 40,
        });

        let full = simulate_planet_timeline(&p, &full_arrivals, player, horizon);
        let checkpoint = simulate_planet_timeline_from(&p, &baseline, 8, &full_arrivals);

        for t in 0..=horizon as usize {
            assert_eq!(
                full.owner_at[t], checkpoint.owner_at[t],
                "owner mismatch at turn {t}"
            );
            assert_eq!(
                full.ships_at[t], checkpoint.ships_at[t],
                "ships mismatch at turn {t}"
            );
        }
    }

    /// Cache builds a baseline for every planet in the engine state and the
    /// ledger contains an entry (possibly empty) for each.
    #[test]
    fn timeline_cache_covers_all_planets() {
        let state = EngineState::new(42, 2, Configuration::default());
        let cache = TimelineCache::build(&state, 0, 20);

        assert_eq!(cache.player, 0);
        assert_eq!(cache.horizon, 20);
        for planet in &state.planets {
            assert!(
                cache.baselines.contains_key(&planet.id),
                "planet {} missing baseline",
                planet.id
            );
            assert!(
                cache.ledger.contains_key(&planet.id),
                "planet {} missing ledger entry",
                planet.id
            );
        }
    }

    /// `min_ships_to_own_by` with the cache should return identical answers
    /// to the equivalent full-sim binary search.
    #[test]
    fn min_ships_via_cache_matches_full_sim() {
        let p = planet(0, 1, 20, 1); // owned by attacker target (enemy of attacker_owner=0)
        let player = 0;
        let horizon = 15;
        let arrival_turn = 5;
        let eval_turn = 12;
        let upper_bound = 200;

        let base_arrivals: Vec<ArrivalEvent> = vec![
            ArrivalEvent { turns: 4, owner: 1, ships: 8 }, // defender reinforces
        ];

        // Reference (full sim) implementation, inline to avoid relying on the
        // old function signature.
        let reference = {
            let base_tl = simulate_planet_timeline(&p, &base_arrivals, 0, eval_turn);
            if state_at_timeline(&base_tl, eval_turn).0 == 0 {
                0
            } else {
                let mut scratch = base_arrivals.clone();
                scratch.push(ArrivalEvent { turns: arrival_turn, owner: 0, ships: 0 });
                let last = scratch.len() - 1;
                let owns = |ships: i64, buf: &mut [ArrivalEvent]| {
                    buf[last].ships = ships;
                    let tl = simulate_planet_timeline(&p, buf, 0, eval_turn);
                    state_at_timeline(&tl, eval_turn).0 == 0
                };
                if !owns(upper_bound, &mut scratch) {
                    upper_bound + 1
                } else {
                    let (mut lo, mut hi) = (1i64, upper_bound);
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if owns(mid, &mut scratch) { hi = mid; } else { lo = mid + 1; }
                    }
                    lo
                }
            }
        };

        // Build a cache containing this planet and pre-seeded ledger.
        let mut cache = TimelineCache {
            player,
            horizon,
            ledger: HashMap::new(),
            baselines: HashMap::new(),
        };
        cache.ledger.insert(p.id, base_arrivals.clone());
        cache.baselines.insert(
            p.id,
            simulate_planet_timeline(&p, &base_arrivals, player, horizon),
        );

        let via_cache =
            min_ships_to_own_by(&cache, &p, 0, arrival_turn, eval_turn, upper_bound);

        assert_eq!(via_cache, reference);
    }
}
