//! Line-of-sight obstacle tester for fleet shots.
//!
//! [`aim_with_prediction`] leads the target ([`lead_target`]) at the true fleet
//! speed, then checks the path with [`shot_blocked_exact`], which runs the
//! engine's own swept-pair collision ([`crate::apollo::engine::swept_pair_hit`]) per
//! turn against every obstacle (sun, planet, comet) over the already-cached
//! entity positions. Being the simulator's own primitive, the verdict is exact
//! — no precomputed, speed-quantized arc table. If the direct line is blocked,
//! the target's radius admits a small cone of aim angles that still land on it;
//! [`aim_with_prediction`] scans that cone for one whose path is clear.

#![allow(dead_code)]

use std::f64::consts::FRAC_PI_2;

use crate::apollo::constants::{
    CENTER, EPISODE_STEPS, HORIZON, LAUNCH_CLEARANCE, MAX_SHIP_SPEED, SUN_RADIUS, NUDGE_SCAN
};
use crate::apollo::engine::fleet_speed;
use crate::apollo::entity_cache::EntityCache;

/// Aim solver result: `(angle_radians, integer_turns, target_x, target_y,
/// fractional_flight_time)`. The fifth component is the real-valued flight
/// time at which the swept-pair test fires — strictly `≤ turns` and used by
/// the aim cache when re-verifying a stored shot after a comet spawn. Passing
/// `turns as f64` there is incorrect (over-conservative): a blocker whose
/// `flight_t` lies in `(flight_time, turns]` would falsely evict a still-valid
/// shot.
pub type AimResult = (f64, i64, f64, f64, f64);

/// Fraction `s* ∈ [0, 1]` of closest fleet-to-target approach during one turn,
/// and that minimum gap `|K(s*) − D(s*)|`. The fleet ring distance is
/// `D(s) = d0 + s·v`; the target's distance from launch is
/// `K(s) = |Q0 + s·dq − L|`. Aiming at `bearing(L → Q(s*))` makes the
/// fleet-to-target distance exactly `|K(s*) − D(s*)|`, so the engine's
/// swept-pair test fires this turn iff the returned gap ≤ `target_radius`.
///
/// `h(s) = K(s) − D(s)` is convex (norm of an affine function, minus an affine
/// function), so the closest approach is solved exactly rather than by uniform
/// sub-sampling (which could tunnel past the true minimum between samples):
///   * Exact contact (`h = 0`) is the quadratic `K²(s) = D²(s)` — the dominant
///     intercept case. `K, D > 0`, so a real root in `[0, 1]` is a genuine
///     `K = D` crossing (no spurious `K = −D`); the earliest gives gap 0.
///   * With no crossing `h` keeps one sign on `[0, 1]`. If `h > 0` the closest
///     approach is the convex interior minimum, where `h'(s) = 0` ⇒
///     `(b + c·s) = v·√g(s)`; squaring gives a quadratic whose in-range root
///     (plus the two endpoints) are the only argmin candidates. If `h < 0`
///     the minimum of `|h|` is at the endpoint where `h` is largest — a convex
///     function attains its maximum over an interval at an endpoint.
#[allow(clippy::too_many_arguments)]
fn closest_approach(
    lx: f64,
    ly: f64,
    q0x: f64,
    q0y: f64,
    dqx: f64,
    dqy: f64,
    d0: f64,
    v: f64,
) -> (f64, f64) {
    let ux = q0x - lx;
    let uy = q0y - ly;
    let a = ux * ux + uy * uy; // K²(0)
    let b = ux * dqx + uy * dqy; // ½ dK²/ds at s=0
    let c = dqx * dqx + dqy * dqy; // ½ d²K²/ds² (constant)
    let h = |s: f64| (a + 2.0 * b * s + c * s * s).max(0.0).sqrt() - (d0 + v * s);

    // Exact contact: roots of K²(s) = D²(s) → (c−v²)s² + 2(b−d0·v)s + (a−d0²)=0,
    // written `big_a·s² + 2·big_b·s + big_c` so roots are (−big_b ± √disc)/big_a.
    let big_a = c - v * v;
    let big_b = b - d0 * v;
    let big_c = a - d0 * d0;
    let mut earliest_root = f64::INFINITY;
    if big_a.abs() < 1e-12 {
        // Degenerate (target chord speed equals fleet speed): linear in s.
        if big_b.abs() >= 1e-12 {
            let s = -big_c / (2.0 * big_b);
            if (0.0..=1.0).contains(&s) {
                earliest_root = s;
            }
        }
    } else {
        let disc = big_b * big_b - big_a * big_c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            for &root in &[(-big_b - sq) / big_a, (-big_b + sq) / big_a] {
                if (0.0..=1.0).contains(&root) && root < earliest_root {
                    earliest_root = root;
                }
            }
        }
    }
    if earliest_root.is_finite() {
        return (earliest_root, 0.0);
    }

    // No crossing → h is single-signed on [0, 1] (h = 0 at s = 0 would have
    // been caught above as a root, so h(0) ≠ 0 here).
    let h0 = h(0.0);
    if h0 <= 0.0 {
        // h < 0 throughout: |h| is smallest where h is largest; a convex
        // function attains its max over an interval at an endpoint.
        let h1 = h(1.0);
        return if h0 >= h1 { (0.0, -h0) } else { (1.0, -h1) };
    }

    // h > 0 throughout and convex → the closest approach is the interior
    // minimum where h'(s) = 0. Squaring `(b + c·s) = v·√g(s)` gives
    // `c(c−v²)s² + 2b(c−v²)s + (b²−v²a) = 0`. The convex stationary point is a
    // root of this; the two endpoints are the only other argmin candidates.
    // Evaluating h at each and taking the minimum is exact and degrades
    // gracefully when the quadratic is degenerate (c → 0 static target, or
    // c → v²): no in-range root ⇒ the minimum is at an endpoint.
    let mut best_s = 0.0;
    let mut best_gap = h0;
    let h1 = h(1.0);
    if h1 < best_gap {
        best_s = 1.0;
        best_gap = h1;
    }
    let qa = c * (c - v * v);
    let qb = 2.0 * b * (c - v * v);
    let qc = b * b - v * v * a;
    let consider = |s: f64, best_s: &mut f64, best_gap: &mut f64| {
        if (0.0..=1.0).contains(&s) {
            let hs = h(s);
            if hs < *best_gap {
                *best_gap = hs;
                *best_s = s;
            }
        }
    };
    if qa.abs() < 1e-12 {
        if qb.abs() >= 1e-12 {
            consider(-qc / qb, &mut best_s, &mut best_gap);
        }
    } else {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            consider((-qb - sq) / (2.0 * qa), &mut best_s, &mut best_gap);
            consider((-qb + sq) / (2.0 * qa), &mut best_s, &mut best_gap);
        }
    }
    (best_s, best_gap.max(0.0))
}

/// Lead a (possibly moving) target. Returns `(angle, integer_turns, target_x,
/// target_y, fractional_flight_time)` where `(target_x, target_y) = Q(s*)` is
/// the point on the target's chord during turn `integer_turns` at which the
/// engine's swept-pair test fires, and `fractional_flight_time = integer_turns
/// − 1 + s*`. `v` is the quantized fleet speed.
///
/// Approach: walk integer turn `t` forward from the earliest geometrically
/// feasible turn. For each `t`, [`closest_approach`] solves for the
/// `s* ∈ [0, 1]` minimizing the radial gap `|K(s) − D(s)|` between the target's
/// chord position `Q(s)` and the fleet's chord distance
/// `D(s) = launch_offset + (t − 1 + s)·v` (same chord linearization the engine
/// uses) — exactly, since `K − D` is convex (closed-form `K = D` crossing, else
/// golden-section on the convex minimum). Aiming `θ = bearing(L → Q(s*))` puts
/// the fleet at distance `D(s*)` along the line through `Q(s*)`, so the actual
/// fleet-to-target distance at `s*` is exactly `|K(s*) − D(s*)|`. If that is
/// ≤ `target_radius`, the engine's `swept_pair_hit` fires for turn `t` —
/// return immediately with the earliest such turn.
///
/// This replaces a prior end-of-turn fixed-point iteration that could settle
/// on a self-consistent `(angle, turns)` whose chord never actually intersects
/// the target's chord during that turn — the cause of fleets launched at
/// orbiters flying clean past and off the map.
pub fn lead_target(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    launch_turn_offset: i64,
    v: f64,
) -> Option<(f64, i64, f64, f64, f64)> {
    let [lx, ly] = cache.position(shooter_id, launch_turn_offset)?;
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
    let target = cache.get(target_id)?;
    let tr = target.radius;

    let abs_launch = cache.current_turn + launch_turn_offset;
    let max_lookahead = HORIZON.min((EPISODE_STEPS - 1 - abs_launch).max(0));
    if max_lookahead < 1 {
        return None;
    }

    // Seed: lower bound on the earliest feasible arrival turn. The stationary
    // assumption (`seed_d / v`) is only valid if the target isn't *closing* on
    // the shooter — but an orbiting target at ROTATION_LIMIT (~50) with max
    // ω (0.05) can approach at up to ω·r ≈ 2.5 units/turn, on top of the
    // fleet's own outward speed. Using just `v` as the closing rate skips
    // past the actual intercept turn for slow fleets vs approaching orbiters
    // — a real bug that caused valid shots to be silently rejected (None
    // returned, no fleet fired). max_lookahead is bounded by HORIZON=30, so
    // it's cheap to just start at turn 1.
    let start: i64 = 1;
    let [_seed_x, _seed_y] = cache.position(target_id, launch_turn_offset)?;

    for t in start..=max_lookahead {
        let [q0x, q0y] = cache.position(target_id, launch_turn_offset + t - 1)?;
        let [q1x, q1y] = cache.position(target_id, launch_turn_offset + t)?;
        let dqx = q1x - q0x;
        let dqy = q1y - q0y;
        let d0 = launch_offset + (t as f64 - 1.0) * v;

        // Exact closest fleet-to-target approach this turn (convex `K − D`):
        // closed-form `K = D` crossing, else golden-section minimum.
        let (best_s, best_gap) = closest_approach(lx, ly, q0x, q0y, dqx, dqy, d0, v);

        if best_gap > tr {
            continue;
        }

        let qx = q0x + best_s * dqx;
        let qy = q0y + best_s * dqy;
        // Skip pathological cases where the target's chord passes through the
        // launcher position — bearing is undefined.
        let kx = qx - lx;
        let ky = qy - ly;
        if kx * kx + ky * ky < 1e-18 {
            continue;
        }
        let angle = ky.atan2(kx);
        let flight_time = t as f64 - 1.0 + best_s;
        return Some((angle, t, qx, qy, flight_time));
    }

    None
}

/// Earliest swept-pair contact fraction `s ∈ [0, 1]` within one turn between
/// the fleet segment `a → b` and a blocker disk of radius `r` sweeping
/// `p0 → p1`, or `None` if they don't collide this turn. Uses the **exact**
/// coefficients of the engine's [`crate::apollo::engine::swept_pair_hit`] (and its
/// `t2 ≥ 0 && t1 ≤ 1` interval test), so the verdict matches the simulator's
/// collision rule bit-for-bit rather than approximating it. A static disk is
/// the degenerate `p0 == p1` case and needs no special handling.
#[inline]
#[allow(clippy::too_many_arguments)]
fn segment_contact_s(
    ax: f64, ay: f64, bx: f64, by: f64,
    p0x: f64, p0y: f64, p1x: f64, p1y: f64,
    r: f64,
) -> Option<f64> {
    let d0x = ax - p0x;
    let d0y = ay - p0y;
    let dvx = (bx - ax) - (p1x - p0x);
    let dvy = (by - ay) - (p1y - p0y);
    let aco = dvx * dvx + dvy * dvy;
    let bco = 2.0 * (d0x * dvx + d0y * dvy);
    let cco = d0x * d0x + d0y * d0y - r * r;
    if aco < 1e-12 {
        return if cco <= 0.0 { Some(0.0) } else { None };
    }
    let disc = bco * bco - 4.0 * aco * cco;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t1 = (-bco - sq) / (2.0 * aco);
    let t2 = (-bco + sq) / (2.0 * aco);
    if t2 >= 0.0 && t1 <= 1.0 {
        Some(t1.max(0.0))
    } else {
        None
    }
}

/// Exact obstacle verdict for a fleet leaving `shooter_id` at bearing `angle`
/// with speed `v` (launching `launch_turn_offset` turns ahead) that reaches its
/// target at fractional `flight_time`. Returns `true` iff some obstacle
/// (sun, planet, comet) is struck **at or before** `flight_time`.
///
/// This is computed directly at the true engine speed `v` over the already-cached
/// per-turn entity positions — there is no precomputed, speed-quantized arc
/// table, so there is no `V_QUANT` mismatch: every per-turn test is the engine's
/// own swept-pair ([`segment_contact_s`]). The source planet is **not** skipped
/// (the engine doesn't either — an orbiting source can overtake a slow prograde
/// launch within turn 1); a static source is filtered out naturally because the
/// outbound fleet never re-enters its disk. The target is skipped (reaching it
/// is success, not a block) and obstacle contacts strictly after `flight_time`
/// don't count — matching the launch-now arrival ordering.
pub fn shot_blocked_exact(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    angle: f64,
    flight_time: f64,
    v: f64,
    launch_turn_offset: i64,
) -> bool {
    let Some([lx, ly]) = cache.position(shooter_id, launch_turn_offset) else {
        return false;
    };
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
    let ux = angle.cos();
    let uy = angle.sin();

    // Only turns whose `[t-1, t]` flight-time window can contain a contact at or
    // before arrival matter; later turns are irrelevant.
    let max_turn = (flight_time.ceil() as i64).max(1);

    // The fleet flies one straight ray at constant speed from the launch ring
    // out to the arrival distance `ring_d`. Precompute that whole segment once.
    let ring_d = launch_offset + flight_time * v;
    let sx = lx + launch_offset * ux;
    let sy = ly + launch_offset * uy;
    let ex = lx + ring_d * ux;
    let ey = ly + ring_d * uy;

    // Static disk: a single swept-pair test over the entire flight segment. The
    // per-turn segments merely tile this same ray, so one test is exact; and the
    // segment already stops at `ring_d`, so no `<= flight_time` gate is needed.
    let static_hit = |cx: f64, cy: f64, r: f64| -> bool {
        segment_contact_s(sx, sy, ex, ey, cx, cy, cx, cy, r).is_some()
    };

    // Fleet segment for turn `t`: ring distance `D(s) = launch_offset + (t-1+s)·v`
    // along the fixed bearing. Returns true iff the disk is struck by arrival.
    // Only needed for *moving* blockers (each turn is a different chord).
    let contact_before = |p0x: f64, p0y: f64, p1x: f64, p1y: f64, r: f64, t: i64| -> bool {
        let d_start = launch_offset + (t as f64 - 1.0) * v;
        let d_end = launch_offset + t as f64 * v;
        let ax = lx + d_start * ux;
        let ay = ly + d_start * uy;
        let bx = lx + d_end * ux;
        let by = ly + d_end * uy;
        match segment_contact_s(ax, ay, bx, by, p0x, p0y, p1x, p1y, r) {
            Some(s) => (t as f64 - 1.0 + s) <= flight_time + 1e-9,
            None => false,
        }
    };

    // Sun: static disk at board center.
    if static_hit(CENTER, CENTER, SUN_RADIUS) {
        return true;
    }

    // Planets and comets (including the source — see above).
    let abs_base = cache.current_turn + launch_turn_offset;
    for (&bid, ent) in cache.entities.iter() {
        if bid == target_id {
            continue;
        }
        let r = ent.radius;

        // Static planets don't move: their disk is fixed across the whole flight
        // window, so one segment test suffices instead of the per-turn loop.
        if ent.is_static() {
            let idx = abs_base.clamp(0, EPISODE_STEPS - 1) as usize;
            if let Some(Some(p)) = ent.positions.get(idx) {
                if static_hit(p[0], p[1], r) {
                    return true;
                }
            }
            continue;
        }

        let positions = &ent.positions;
        for t in 1..=max_turn {
            let abs0 = abs_base + t - 1;
            let abs1 = abs_base + t;
            if abs0 < 0 || abs1 < 0 || (abs1 as usize) >= positions.len() {
                continue;
            }
            let (Some(p0), Some(p1)) = (positions[abs0 as usize], positions[abs1 as usize]) else {
                continue;
            };
            if contact_before(p0[0], p0[1], p1[0], p1[1], r, t) {
                return true;
            }
        }
    }
    false
}

/// Wrap `a` into `(-π, π]`.
#[inline]
fn wrap_pi(a: f64) -> f64 {
    use std::f64::consts::PI;
    a - 2.0 * PI * ((a + PI) * (1.0 / (2.0 * PI))).floor()
}

/// Closed-form early-out for the blocked path: `true` iff the whole aim cone
/// `[beta ± phi_max]` is already covered by obstacles that **definitely** block
/// (so the cone scan would find nothing and can be skipped).
///
/// Only the sun and *static* planets are considered — their blocked arc is the
/// exact closed-form angular shadow `bearing ± asin(r/d)`, gated so it's a
/// **subset** of the truly-blocked angles: the disk must be struck after the
/// fleet clears the launch ring (`d ≥ launch_offset + r`) and before reaching
/// the target (its farthest in-arc entry `√(d²−r²) ≤ ring_d`). Because every
/// reported arc is genuinely blocked, full coverage of the cone proves no clear
/// angle exists, so returning `None` cannot drop a recoverable shot. Movers are
/// left to the scan fallback (their shadow isn't a fixed arc).
fn cone_clear_impossible(
    cache: &EntityCache,
    target_id: i64,
    lx: f64,
    ly: f64,
    launch_offset: f64,
    beta: f64,
    phi_max: f64,
    ring_d: f64,
    launch_turn_offset: i64,
) -> bool {
    let mut intervals: Vec<(f64, f64)> = Vec::new();
    let mut add = |cx: f64, cy: f64, r: f64| {
        let dx = cx - lx;
        let dy = cy - ly;
        let d = (dx * dx + dy * dy).sqrt();
        if d < launch_offset + r {
            return; // overlaps the launch ring — not a clean forward block
        }
        if (d * d - r * r).sqrt() > ring_d {
            return; // farthest in-arc entry is beyond the target — not struck first
        }
        let half = (r / d).asin();
        let delta = wrap_pi(dy.atan2(dx) - beta);
        let lo = (delta - half).max(-phi_max);
        let hi = (delta + half).min(phi_max);
        if lo <= hi {
            intervals.push((lo, hi));
        }
    };

    add(CENTER, CENTER, SUN_RADIUS);
    let abs = cache.current_turn + launch_turn_offset;
    if abs >= 0 {
        for (&bid, ent) in cache.entities.iter() {
            if bid == target_id || !ent.is_static() {
                continue;
            }
            if let Some(Some(p)) = ent.positions.get(abs as usize) {
                add(p[0], p[1], ent.radius);
            }
        }
    }

    if intervals.is_empty() {
        return false;
    }
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut covered = -phi_max;
    for (lo, hi) in intervals {
        if lo > covered + 1e-9 {
            return false; // gap in coverage
        }
        if hi > covered {
            covered = hi;
        }
    }
    covered >= phi_max - 1e-9
}

/// Aim from `shooter_id` to `target_id` with `ships`, for a fleet launching
/// at `launch_turn_offset` turns after the cache's current turn (pass `0`
/// for "launch now"). Source, target, and obstacle positions are all
/// evaluated at the launch turn so obstacle tests reflect the real flight
/// window — required by the early-game DFS, which scores delayed launches and
/// would otherwise falsely assume current geometry.
///
/// Every verdict is the exact swept-pair ([`shot_blocked_exact`]) at the true
/// engine speed. If the direct line is blocked, the target's radius still admits
/// a small cone of aim angles that land on it at `flight_time`; we scan that
/// cone (smallest deviation first) for one whose path is clear.
pub fn aim_with_prediction(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    launch_turn_offset: i64,
) -> Option<AimResult> {
    // Lead the target at the exact engine speed so (angle, turns) land on the
    // actual orbital intercept point.
    let v_true = fleet_speed(ships.max(1), MAX_SHIP_SPEED);
    let (angle, turns, tx, ty, flight_time) =
        lead_target(cache, shooter_id, target_id, launch_turn_offset, v_true)?;

    if !shot_blocked_exact(cache, shooter_id, target_id, angle, flight_time, v_true, launch_turn_offset) {
        return Some((angle, turns, tx, ty, flight_time));
    }

    // Direct line blocked: scan the cone of aim angles that still land on the
    // target. The cone is centred on the bearing to the target point `(tx, ty)`
    // and spans the target disk's angular radius `asin(target_radius / K)` from
    // the launcher (padded; the exact membership test below is the real gate).
    let [lx, ly] = cache.position(shooter_id, launch_turn_offset)?;
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let target_radius = cache.get(target_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
    let ring_d = launch_offset + flight_time * v_true;
    if ring_d <= 0.0 {
        return None;
    }
    let r_sq = target_radius * target_radius;
    let tdx = tx - lx;
    let tdy = ty - ly;
    let k_dist = (tdx * tdx + tdy * tdy).sqrt();
    let beta = tdy.atan2(tdx);
    let phi_max = if k_dist <= target_radius {
        FRAC_PI_2
    } else {
        ((target_radius / k_dist).asin() * 1.1).min(FRAC_PI_2)
    };

    // Closed-form early-out: if the sun + static planets already cover the whole
    // cone, no scan can find an opening — skip the 2·NUDGE_SCAN probes.
    if cone_clear_impossible(
        cache, target_id, lx, ly, launch_offset, beta, phi_max, ring_d, launch_turn_offset,
    ) {
        return None;
    }

    let step = phi_max / NUDGE_SCAN as f64;
    for k in 1..=NUDGE_SCAN {
        let d = k as f64 * step;
        for &theta_try in &[beta + d, beta - d] {
            let fx = lx + ring_d * theta_try.cos();
            let fy = ly + ring_d * theta_try.sin();
            let dx = fx - tx;
            let dy = fy - ty;
            if dx * dx + dy * dy > r_sq {
                continue; // outside the target's valid arc at this turn
            }
            if !shot_blocked_exact(
                cache, shooter_id, target_id, theta_try, flight_time, v_true, launch_turn_offset,
            ) {
                return Some((theta_try, turns, tx, ty, flight_time));
            }
        }
    }
    None
}

/// Cheap revalidation of a previously-computed shot against the current
/// obstacle set. Used by the aim cache when a comet may have spawned since
/// the result was cached. `flight_time` must be the fractional flight time
/// from the original solve — not `turns as f64` (see [`AimResult`]).
/// `launch_turn_offset` is the offset of the launch turn *relative to the
/// cache's current turn* at re-verification time — for a delayed-launch
/// entry whose abs-launch slot has been partially overtaken, this is
/// `slot - current_turn`, so the exact swept-pair sees obstacle positions at
/// the actual launch time.
pub fn shot_still_clear(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    angle: f64,
    flight_time: f64,
    launch_turn_offset: i64,
) -> bool {
    let v = fleet_speed(ships.max(1), MAX_SHIP_SPEED);
    !shot_blocked_exact(cache, shooter_id, target_id, angle, flight_time, v, launch_turn_offset)
}
