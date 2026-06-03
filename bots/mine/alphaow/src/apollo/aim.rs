//! Line-of-sight obstacle tester for fleet shots.
//!
//! [`aim_with_prediction`] leads the target ([`lead_target_from`]) at the true fleet
//! speed, then checks the path with [`shot_blocked_exact`], which runs the
//! engine's own swept-pair collision ([`crate::apollo::engine::swept_pair_hit`]) per
//! turn against every obstacle (sun, planet, comet) over the already-cached
//! entity positions. Being the simulator's own primitive, the verdict is exact
//! — no precomputed, speed-quantized arc table. If the direct line is blocked,
//! the target's radius admits a small cone of aim angles that still land on it;
//! [`aim_with_prediction`] scans that cone for one whose path is clear.

#![allow(dead_code)]

use std::f64::consts::FRAC_PI_2;

use crate::apollo::cache::{Entity, EntityCache};
use crate::apollo::constants::{
    CENTER, EPISODE_STEPS, HORIZON, LAUNCH_CLEARANCE, MAX_CONE_PROBES, MAX_CONE_STEP,
    MAX_SHIP_SPEED, NUDGE_SCAN, SUN_RADIUS,
};
use crate::apollo::engine::fleet_speed;

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
/// Resumable: only considers intercept turns `≥ from_turn`. Returns the earliest
/// feasible intercept at or after `from_turn`, so a caller whose earliest
/// intercept was blocked can re-solve from `that_turn + 1` to find the next
/// geometric intercept (the target has moved, opening a different clear angle).
/// Pass `from_turn = 1` for the earliest feasible intercept overall.
pub fn lead_target_from(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    launch_turn_offset: i64,
    v: f64,
    from_turn: i64,
) -> Option<(f64, i64, f64, f64, f64)> {
    let [lx, ly] = cache.position(shooter_id, launch_turn_offset)?;
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
    let target = cache.get(target_id)?;
    let tr = target.radius;

    let abs_launch = cache.current_turn + launch_turn_offset;
    // A fleet launched at engine step `s` moves and can collide during that same
    // step (reported as intercept turn 1), and the engine resolves steps through
    // the last one, `EPISODE_STEPS - 1`. The intercept at turn `t` resolves at
    // step `abs_launch + t - 1`, so the latest useful `t` is `EPISODE_STEPS -
    // abs_launch` (lands on the final tick). The position table now carries the
    // extra index `EPISODE_STEPS` that this final shot reads (see `build_*_entity`).
    let max_lookahead = HORIZON.min((EPISODE_STEPS - abs_launch).max(0));
    if max_lookahead < 1 {
        return None;
    }

    // Sound lower bound on the earliest feasible arrival turn. The fleet's far
    // radius at turn `t` is `launch_offset + t·v`, and the target's distance from
    // launch is always ≥ `k_min` — the closest any point of its circular orbit
    // gets to `L` (`|dist(L, center) − orbital_radius|`). A comet's path is not a
    // fixed-radius circle, so it gets `k_min = 0` (no skip). Intercept needs
    // `launch_offset + t·v + tr ≥ k_min`, so earlier turns are provably out of
    // reach — start the (identical) scan there. This *under*-skips, the safe
    // direction: unlike the old `seed_d / v` guess it can never step past a
    // closing orbiter's true intercept turn (the bug that silently dropped shots).
    let dlc = ((lx - CENTER).powi(2) + (ly - CENTER).powi(2)).sqrt();
    let k_min = if target.is_comet() {
        0.0
    } else {
        (dlc - target.orbital_radius).abs()
    };
    let start = (((k_min - tr - launch_offset) / v).ceil() as i64)
        .max(1)
        .max(from_turn);

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
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    p0x: f64,
    p0y: f64,
    p1x: f64,
    p1y: f64,
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

/// Squared shortest distance between 2D segments `p1->q1` and `p2->q2` (the
/// canonical clamped closest-points formulation). Returns `0.0` when they cross.
/// Used as a `sqrt`-free capsule reject in [`comet_blocks_path`]; over-reporting
/// distance would risk a false reject (a missed block), so the clamping keeps the
/// result an exact lower bound on the true segment separation.
#[allow(clippy::too_many_arguments)]
fn segment_segment_dist_sq(
    p1x: f64,
    p1y: f64,
    q1x: f64,
    q1y: f64,
    p2x: f64,
    p2y: f64,
    q2x: f64,
    q2y: f64,
) -> f64 {
    let d1x = q1x - p1x;
    let d1y = q1y - p1y;
    let d2x = q2x - p2x;
    let d2y = q2y - p2y;
    let rx = p1x - p2x;
    let ry = p1y - p2y;
    let a = d1x * d1x + d1y * d1y; // |d1|^2
    let e = d2x * d2x + d2y * d2y; // |d2|^2
    let f = d2x * rx + d2y * ry;
    const EPS: f64 = 1e-12;

    let (s, t) = if a <= EPS && e <= EPS {
        (0.0, 0.0) // both segments are points
    } else if a <= EPS {
        (0.0, (f / e).clamp(0.0, 1.0)) // first is a point
    } else {
        let c = d1x * rx + d1y * ry;
        if e <= EPS {
            ((-c / a).clamp(0.0, 1.0), 0.0) // second is a point
        } else {
            let b = d1x * d2x + d1y * d2y;
            let denom = a * e - b * b;
            let s = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0 // parallel: pick an arbitrary point on the first segment
            };
            let t = (b * s + f) / e;
            if t < 0.0 {
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else if t > 1.0 {
                (((b - c) / a).clamp(0.0, 1.0), 1.0)
            } else {
                (s, t)
            }
        }
    };

    let c1x = p1x + d1x * s;
    let c1y = p1y + d1y * s;
    let c2x = p2x + d2x * t;
    let c2y = p2y + d2y * t;
    let dx = c1x - c2x;
    let dy = c1y - c2y;
    dx * dx + dy * dy
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
    blocked_on_path(
        cache,
        shooter_id,
        target_id,
        angle,
        flight_time,
        v,
        launch_turn_offset,
        true,
        |_| true,
    )
}

/// Comet-only gate: tests the path against **just the cached comets**
/// ([`EntityCache::comet_ids`], ≤4 per group) — no sun, no planets, no scan of
/// the full entity map. The invariant-aim cache uses this per turn: for a
/// disc-qualified static→static / orbiting→orbiting shot the sun+planet verdict
/// is fixed (or rotation-equivariant), so only a comet can change it. Returns
/// `true` iff some comet is struck at or before `flight_time`. Comets always
/// move, so this only runs the per-turn swept test (no static-disk branch), and
/// short-circuits instantly when no comets exist.
pub fn comet_blocks_path(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    angle: f64,
    flight_time: f64,
    v: f64,
    launch_turn_offset: i64,
) -> bool {
    if cache.comet_ids.is_empty() {
        return false;
    }
    let Some([lx, ly]) = cache.position(shooter_id, launch_turn_offset) else {
        return false;
    };
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
    let ux = angle.cos();
    let uy = angle.sin();
    let max_turn = (flight_time.ceil() as i64).max(1);
    let abs_base = cache.current_turn + launch_turn_offset;

    // The fleet flies one straight segment from the launch ring out to arrival.
    let ring_d = launch_offset + flight_time * v;
    let sx = lx + launch_offset * ux;
    let sy = ly + launch_offset * uy;
    let ex = lx + ring_d * ux;
    let ey = ly + ring_d * uy;

    for &cid in &cache.comet_ids {
        if cid == target_id {
            continue;
        }
        let Some(ent) = cache.get(cid) else {
            continue;
        };
        let r = ent.radius;

        // Capsule reject: the comet's arc lies within `bulge` of its chord, so a
        // contact needs the fleet segment within `r + bulge` of that chord. This
        // hugs even a long *diagonal* arc tightly (an axis-aligned box would be
        // mostly empty and reject almost nothing). `sqrt`-free, in squared form.
        let [cax, cay, cbx, cby] = ent.chord;
        let reach = r + ent.bulge;
        if segment_segment_dist_sq(sx, sy, ex, ey, cax, cay, cbx, cby) > reach * reach {
            continue;
        }

        // Sweep from the launch turn (`t = 1`) up to the comet's last on-board
        // turn. The lower end needs no `on_board` clamp: a comet is only ever in
        // the cache once it is on the board, and launches are never in the past,
        // so the comet is always present from the launch turn onward — only its
        // `off_board_turn` edge can fall inside the flight. Empty range (already
        // expired) ⇒ comet skipped.
        let positions = &ent.positions;
        // The engine parks an expiring comet at its last on-board position for one
        // extra tick and still collides fleets with it before removing it (it
        // can't be captured that tick — removed pre-combat — so only the swept
        // collision matters, not `off_board_turn`/economy). Sweep that exit turn
        // too: extend the range by one and clamp slot reads to the last on-board
        // slot so the exit turn is a static disk at the comet's final position.
        let last_slot = (ent.off_board_turn - 1).clamp(0, EPISODE_STEPS - 1) as usize;
        let exit_extend = ent.off_board_turn < EPISODE_STEPS;
        let hi = (ent.off_board_turn - if exit_extend { 0 } else { 1 } - abs_base).min(max_turn);
        // Engine id-order tiebreak: a comet with id >= target_id loses to the
        // target on the arrival turn (`max_turn`), so it can't block there.
        let hi = if cid >= target_id {
            hi.min(max_turn - 1)
        } else {
            hi
        };
        for t in 1..=hi {
            // `t ≥ 1` ⇒ start slot ≥ `abs_base` ≥ on-board. The end slot is clamped
            // to `last_slot` so the comet's exit turn reads its last position twice
            // (static); both reads are thus `Some` — unwrap encodes that.
            let p0 =
                positions[((abs_base + t - 1) as usize).min(last_slot)].expect("on-board by clamp");
            let p1 =
                positions[((abs_base + t) as usize).min(last_slot)].expect("on-board by clamp");
            let d_start = launch_offset + (t as f64 - 1.0) * v;
            let d_end = launch_offset + t as f64 * v;
            let ax = lx + d_start * ux;
            let ay = ly + d_start * uy;
            let bx = lx + d_end * ux;
            let by = ly + d_end * uy;
            // Any swept contact this tick blocks (the engine resolves the fleet
            // against the comet on contact regardless of fraction); the arrival-
            // turn id tiebreak is handled by the `hi` cap above.
            if segment_contact_s(ax, ay, bx, by, p0[0], p0[1], p1[0], p1[1], r).is_some() {
                return true;
            }
        }
    }
    false
}

/// Shared swept-path obstacle test backing [`shot_blocked_exact`] and the
/// [`aim_with_blocker`] scans. `include_sun` toggles the board-center sun test;
/// `consider` selects which entities participate (the target is always skipped).
/// Returns `true` iff a selected obstacle is struck at or before `flight_time`.
#[allow(clippy::too_many_arguments)]
fn blocked_on_path(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    angle: f64,
    flight_time: f64,
    v: f64,
    launch_turn_offset: i64,
    include_sun: bool,
    mut consider: impl FnMut(&Entity) -> bool,
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
    // outward. `ring_d = launch_offset + flight_time * v` is the fractional
    // target-contact distance; the precomputed endpoints below tile that ray.
    let sx = lx + launch_offset * ux;
    let sy = ly + launch_offset * uy;
    // Endpoint at the START of the arrival turn (one tick before target arrival).
    // On the arrival turn the target is hit, and the engine resolves the fleet
    // against the lowest-id planet first (then the sun, only if no planet hit).
    // So the sun, and any planet with id >= target_id, lose to the target on that
    // turn and can only block *before* it — testing their static disks just up to
    // here applies the same id-order tiebreak the per-turn moving loop uses.
    let pre_arrival_d = launch_offset + (max_turn - 1).max(0) as f64 * v;
    let e2x = lx + pre_arrival_d * ux;
    let e2y = ly + pre_arrival_d * uy;
    // Endpoint at the END of the arrival turn. The fleet moves a full tick each
    // turn, so on the arrival turn it sweeps out to here — past the target at
    // `ring_d`. A *lower*-id obstacle struck anywhere up to this point wins the
    // arrival-turn tiebreak and kills the fleet (the engine doesn't stop at the
    // target's fraction), so lower-id obstacles are tested over the full ray to
    // here rather than only to `ring_d`.
    let arrival_end_d = launch_offset + max_turn as f64 * v;
    let efx = lx + arrival_end_d * ux;
    let efy = ly + arrival_end_d * uy;

    // Distance-from-board-center span of the fleet ray, computed once. An
    // orbiting planet lives permanently on the circle of radius `orbital_radius`,
    // so its body only ever occupies the annulus `[orbital_radius ± radius]`; if
    // that band is disjoint from this ray span the planet can't be struck at any
    // angle or turn (see the reject in the loop). Max is at an endpoint; min is
    // at the perpendicular foot of center onto the segment (clamped).
    //
    // The span runs to `efx,efy` (the END of the arrival turn, `arrival_end_d`),
    // not just to the target-contact point `ex,ey` (`ring_d`): a lower-id orbiter
    // wins the arrival-turn tiebreak and is swept over the full arrival tick
    // (out to `arrival_end_d` in the per-turn loop), so its reachable region can
    // extend past `ring_d`. Bounding only to `ring_d` could reject such a planet
    // before the exact swept test and let through a shot the engine kills.
    // Obstacles whose sweep stops at/before `ring_d` (sun, id >= target_id) are
    // unaffected — the wider band is merely conservative for them.
    let (ray_d_min, ray_d_max) = {
        let dsx = sx - CENTER;
        let dsy = sy - CENTER;
        let dex = efx - CENTER;
        let dey = efy - CENTER;
        let d_end_s = (dsx * dsx + dsy * dsy).sqrt();
        let d_end_e = (dex * dex + dey * dey).sqrt();
        let segx = efx - sx;
        let segy = efy - sy;
        let l2 = segx * segx + segy * segy;
        let d_min = if l2 > 1e-12 {
            let u = (((CENTER - sx) * segx + (CENTER - sy) * segy) / l2).clamp(0.0, 1.0);
            let fx = sx + u * segx - CENTER;
            let fy = sy + u * segy - CENTER;
            (fx * fx + fy * fy).sqrt()
        } else {
            d_end_s
        };
        (d_min, d_end_s.max(d_end_e))
    };

    // Static disk: a single swept-pair test over a flight segment. The per-turn
    // segments tile the ray, so one test is exact. `static_hit_full` covers the
    // whole flight including the full arrival tick (to `efx,efy`) — used for
    // lower-id planets, which win the arrival-turn tiebreak anywhere they're hit.
    let static_hit_full = |cx: f64, cy: f64, r: f64| -> bool {
        segment_contact_s(sx, sy, efx, efy, cx, cy, cx, cy, r).is_some()
    };
    // Same test but only up to the start of the arrival turn — for obstacles that
    // lose the arrival-turn tiebreak to the target (the sun; planets with
    // id >= target_id).
    let static_hit_pre = |cx: f64, cy: f64, r: f64| -> bool {
        segment_contact_s(sx, sy, e2x, e2y, cx, cy, cx, cy, r).is_some()
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
        // Any swept contact during this tick is a real collision: the engine moves
        // the fleet a full tick and resolves it against the lowest-id planet hit
        // anywhere in `[0, 1]` (the fraction is irrelevant). The arrival-turn
        // id-order tiebreak (a planet with id >= target_id can't block on the
        // arrival turn) is handled by the caller's `hi_t` cap, so this just
        // reports whether the disk is struck this tick.
        segment_contact_s(ax, ay, bx, by, p0x, p0y, p1x, p1y, r).is_some()
    };

    // Per-obstacle swept test. Returns `true` iff `ent` is struck by arrival.
    let abs_base = cache.current_turn + launch_turn_offset;
    let entity_blocks = |ent: &Entity| -> bool {
        let r = ent.radius;

        // Static planets don't move: their disk is fixed across the whole flight
        // window, so one segment test suffices instead of the per-turn loop. Like
        // moving obstacles, a static planet with id >= target_id loses the
        // arrival-turn tiebreak to the target, so test it only up to the start of
        // the arrival turn (`static_hit_pre`); lower-id planets use the full ray.
        if ent.is_static() {
            let idx = abs_base.clamp(0, EPISODE_STEPS - 1) as usize;
            if let Some(Some(p)) = ent.positions.get(idx) {
                return if ent.id >= target_id {
                    static_hit_pre(p[0], p[1], r)
                } else {
                    static_hit_full(p[0], p[1], r)
                };
            }
            return false;
        }

        // Orbiting planet: annulus reject. Its center is always `orbital_radius`
        // from board center, so a hit needs the fleet ray to enter the band
        // `[orbital_radius ± r]`. Disjoint from the ray's span ⇒ never struck,
        // skip the per-turn sweep. Comets don't orbit (`orbital_radius == 0`), so
        // this never applies to them — they fall straight through to the loop.
        if !ent.is_comet() {
            let orb = ent.orbital_radius;
            if orb + r < ray_d_min || orb - r > ray_d_max {
                return false;
            }
        }

        // Moving obstacle (orbiter or comet). Sweep from the launch turn up to
        // `off_board_turn`, so both endpoints are `Some` (orbiters are on the
        // whole game; a comet is only ever cached once on board, and launches
        // are never in the past, so it's present from `t = 1` onward — only its
        // `off_board_turn` edge can fall inside the flight). This also subsumes
        // the old `abs0 ≥ 0` / in-bounds guards.
        let positions = &ent.positions;
        let lo_t = 1;
        // A comet leaving the board is parked at its last on-board position for
        // one extra tick by the engine and still collides fleets with it (it's
        // removed pre-combat, so capture/economy is unaffected — only this swept
        // collision is). Sweep that exit turn as a static disk: extend the range
        // by one and clamp end-slot reads to the last on-board slot. Orbiters
        // never leave (`off_board_turn == EPISODE_STEPS`), so they're unchanged.
        let last_slot = (ent.off_board_turn - 1).clamp(0, EPISODE_STEPS - 1) as usize;
        let exit_extend = ent.is_comet() && ent.off_board_turn < EPISODE_STEPS;
        let hi_t = (ent.off_board_turn - if exit_extend { 0 } else { 1 } - abs_base).min(max_turn);
        // Engine same-tick collision rule: the engine sweeps planets in id order
        // and the fleet is consumed by the first (lowest-id) one it touches. On
        // the arrival turn (`max_turn`) the target is hit, so a moving obstacle
        // only kills the fleet there if its id is below the target's; an obstacle
        // with `id >= target_id` loses to the target and cannot block on the
        // arrival turn. Earlier turns are unaffected (the target isn't reached
        // yet, so any obstacle consumes the fleet regardless of id). The sun and
        // static planets are left conservative (not exploited) — see notes above.
        let hi_t = if ent.id >= target_id {
            hi_t.min(max_turn - 1)
        } else {
            hi_t
        };
        if lo_t > hi_t {
            return false;
        }

        // Distance from launch to the obstacle at the *start* of turn `t` (its
        // `abs_base + t - 1` slot, clamped to the last on-board slot for the
        // comet exit turn).
        let dist_at = |t: i64| -> f64 {
            let p =
                positions[((abs_base + t - 1) as usize).min(last_slot)].expect("on-board by clamp");
            let dx = p[0] - lx;
            let dy = p[1] - ly;
            (dx * dx + dy * dy).sqrt()
        };

        let os = ent.max_step;
        if os <= v {
            // The fleet's distance from launch grows by `v` each turn while the
            // obstacle's changes by at most `os ≤ v`, so `h(t) = dist_at(t) − a_t`
            // is non-increasing and the radially-reachable turns form one
            // contiguous band. Contact needs `h(t) ∈ [-(r+os), v+r+os]` — a sound
            // necessary condition: outside it the fleet and obstacle radii stay
            // more than `r` apart for the whole turn, so the swept test is `false`.
            // Binary-search the band's near edge, then run the *exact* swept test
            // across it — bit-identical to scanning every turn, in O(log H + band).
            let upper = v + r + os;
            let lower = -(r + os);
            let a_of = |t: i64| launch_offset + (t as f64 - 1.0) * v;
            // Smallest `t` in `[lo_t, hi_t]` with `h(t) ≤ upper` (h non-increasing).
            let mut t_enter = hi_t + 1;
            let (mut blo, mut bhi) = (lo_t, hi_t);
            while blo <= bhi {
                let mid = blo + (bhi - blo) / 2;
                if dist_at(mid) - a_of(mid) <= upper {
                    t_enter = mid;
                    bhi = mid - 1;
                } else {
                    blo = mid + 1;
                }
            }
            for t in t_enter..=hi_t {
                if dist_at(t) - a_of(t) < lower {
                    break; // past the far edge; no later turn can contact either
                }
                let p0 = positions[((abs_base + t - 1) as usize).min(last_slot)]
                    .expect("on-board by clamp");
                let p1 =
                    positions[((abs_base + t) as usize).min(last_slot)].expect("on-board by clamp");
                if contact_before(p0[0], p0[1], p1[0], p1[1], r, t) {
                    return true;
                }
            }
        } else {
            // Obstacle can out-pace the fleet radially (slow fleet vs a fast
            // comet): `h` may be non-monotonic and the band non-contiguous, so
            // fall back to the exact full scan over the clamped window.
            for t in lo_t..=hi_t {
                let p0 = positions[((abs_base + t - 1) as usize).min(last_slot)]
                    .expect("on-board by clamp");
                let p1 =
                    positions[((abs_base + t) as usize).min(last_slot)].expect("on-board by clamp");
                if contact_before(p0[0], p0[1], p1[0], p1[1], r, t) {
                    return true;
                }
            }
        }
        false
    };

    // Sun: static disk at board center. The engine checks the sun only after the
    // (id-ordered) planet loop and only when no planet was hit that tick, so on
    // the arrival turn the target wins and the sun can't block — test it only up
    // to the start of the arrival turn.
    if include_sun && static_hit_pre(CENTER, CENTER, SUN_RADIUS) {
        return true;
    }

    // Planets and comets (including the source — see above).
    cache
        .entities
        .iter()
        .any(|ent| ent.id != target_id && consider(ent) && entity_blocks(ent))
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
        for ent in cache.entities.iter() {
            if ent.id == target_id || !ent.is_static() {
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
    aim_with_blocker(
        cache,
        shooter_id,
        target_id,
        ships,
        launch_turn_offset,
        true,
    )
}

/// Comet-free aim: identical to [`aim_with_prediction`] but treats the board as
/// having no comets (sun + planets only). The result is the turn-invariant base
/// the invariant-aim cache stores — it reproduces (rotated, for orbiters) at
/// every turn, so a comet-dodging nudge (which would not reproduce) is never
/// baked in. Whether the per-turn comet field actually permits the base is left
/// to the [`comet_blocks_path`] gate; that gate is exact because the cone scan
/// is smallest-deviation first and comets only ever *add* blocked angles, so a
/// comet-free angle stays chosen iff it is comet-clear.
pub fn aim_ignoring_comets(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    launch_turn_offset: i64,
) -> Option<AimResult> {
    aim_with_blocker(
        cache,
        shooter_id,
        target_id,
        ships,
        launch_turn_offset,
        false,
    )
}

/// Max number of successive geometric intercept turns [`aim_with_blocker`] will
/// try before declining. The earliest intercept is the natural shot; if it is
/// fully blocked (direct path + entire cone), the next feasible turn is tried,
/// since a target orbiting a few degrees on often opens a clear approach one
/// turn later (measured: ~all recoverable reachable misses clear at `lead + 1`).
/// Capped at 2 so the retry — which only runs for an otherwise-declined shot —
/// adds at most one extra cone scan and can't blow up runtime.
const MAX_INTERCEPT_TRIES: i64 = 2;

/// Try a single intercept turn. If the direct lead path is clear, accept it;
/// otherwise scan the target's aim cone at this turn's flight time for a clear
/// angle. Returns the accepted [`AimResult`], or `None` if this turn is fully
/// blocked (the caller may then try a later intercept turn). Factored out of
/// [`aim_with_blocker`] so the multi-turn loop runs the identical per-turn logic.
#[allow(clippy::too_many_arguments)]
fn try_intercept_turn(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    angle: f64,
    turns: i64,
    tx: f64,
    ty: f64,
    flight_time: f64,
    v_true: f64,
    launch_turn_offset: i64,
    include_comets: bool,
) -> Option<AimResult> {
    if !blocked_on_path(
        cache,
        shooter_id,
        target_id,
        angle,
        flight_time,
        v_true,
        launch_turn_offset,
        true,
        |e| include_comets || !e.is_comet(),
    ) {
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
    // Target's swept chord during the lead's turn, for the exact per-angle
    // first-contact test in the scan (replaces a static disk at `(tx, ty)`).
    let [q0x, q0y] = cache.position(target_id, launch_turn_offset + turns - 1)?;
    let [q1x, q1y] = cache.position(target_id, launch_turn_offset + turns)?;
    let dqx = q1x - q0x;
    let dqy = q1y - q0y;
    let tdx = tx - lx;
    let tdy = ty - ly;
    let k_dist = (tdx * tdx + tdy * tdy).sqrt();
    let beta = tdy.atan2(tdx);
    // Cone half-width. Not just the target's instantaneous angular radius: the
    // target sweeps `q0 → q1` across the turn, and a fast/long-turn target moves
    // more than its own radius, so valid hitting angles span the bearings to both
    // chord endpoints plus the disk's angular radius (with a small margin). The
    // exact `segment_contact_s` membership in the scan still gates which angles
    // actually land, so widening only adds candidates — never false hits.
    let phi_max = if k_dist <= target_radius {
        FRAC_PI_2
    } else {
        let disk_half = (target_radius / k_dist).asin();
        let dev0 = wrap_pi((q0y - ly).atan2(q0x - lx) - beta).abs();
        let dev1 = wrap_pi((q1y - ly).atan2(q1x - lx) - beta).abs();
        ((dev0.max(dev1) + disk_half) * 1.1).min(FRAC_PI_2)
    };

    // Closed-form early-out: if the sun + static planets already cover the whole
    // cone, no scan can find an opening — skip the 2·NUDGE_SCAN probes.
    if cone_clear_impossible(
        cache,
        target_id,
        lx,
        ly,
        launch_offset,
        beta,
        phi_max,
        ring_d,
        launch_turn_offset,
    ) {
        return None;
    }

    // Scale the probe count with the cone width so the angular step stays fine
    // (≤ MAX_CONE_STEP) even for a wide swept-chord cone — a wider cone must not
    // mean coarser sampling, or a thin hitting window in the widened region would
    // be stepped over. Small cones keep the baseline NUDGE_SCAN; very wide cones
    // are capped at MAX_CONE_PROBES to bound worst-case cost.
    let n_probes = NUDGE_SCAN
        .max((phi_max / MAX_CONE_STEP).ceil() as i64)
        .min(MAX_CONE_PROBES);
    let step = phi_max / n_probes as f64;
    let d_start = launch_offset + (turns as f64 - 1.0) * v_true;
    let d_end = launch_offset + turns as f64 * v_true;
    for k in 1..=n_probes {
        let d = k as f64 * step;
        for &theta_try in &[beta + d, beta - d] {
            let (ux, uy) = (theta_try.cos(), theta_try.sin());
            // Fleet's swept segment during turn `turns` along this angle.
            let ax = lx + d_start * ux;
            let ay = ly + d_start * uy;
            let bx = lx + d_end * ux;
            let by = ly + d_end * uy;
            // Exact first contact with the moving target this turn (the engine's
            // own swept-pair). `None` ⇒ this angle doesn't reach the target during
            // turn `turns` — skip. The fractional contact gives this angle's *own*
            // arrival time.
            let Some(s_hit) = segment_contact_s(ax, ay, bx, by, q0x, q0y, q1x, q1y, target_radius)
            else {
                continue;
            };
            let arrival_ft = turns as f64 - 1.0 + s_hit;
            // Check obstacles only up to this angle's own arrival — not the lead's
            // (later) flight time. An obstacle past where this angle reaches the
            // target is irrelevant (the fleet is consumed on contact); reusing the
            // lead's flight time falsely rejects a clear nudge when an obstacle —
            // e.g. a comet — sits just beyond the target.
            if !blocked_on_path(
                cache,
                shooter_id,
                target_id,
                theta_try,
                arrival_ft,
                v_true,
                launch_turn_offset,
                true,
                |e| include_comets || !e.is_comet(),
            ) {
                let hx = q0x + s_hit * dqx;
                let hy = q0y + s_hit * dqy;
                return Some((theta_try, turns, hx, hy, arrival_ft));
            }
        }
    }
    None
}

/// Shared aim core. `include_comets` selects the obstacle set (sun + planets +
/// comets for [`aim_with_prediction`], sun + planets only for
/// [`aim_ignoring_comets`]).
fn aim_with_blocker(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    launch_turn_offset: i64,
    include_comets: bool,
) -> Option<AimResult> {
    // Lead the target at the exact engine speed so (angle, turns) land on the
    // actual orbital intercept point. Walk successive intercept turns: the
    // earliest is the natural shot, but if it is fully blocked the target may
    // have orbited into a clear approach a turn later, so try the next feasible
    // intercept turn (bounded by `MAX_INTERCEPT_TRIES`). The loop only iterates
    // for shots whose direct path is blocked — the common clear shot returns on
    // the first turn at no extra cost.
    let v_true = fleet_speed(ships.max(1), MAX_SHIP_SPEED);
    let mut from = 1i64;
    for _ in 0..MAX_INTERCEPT_TRIES {
        let (angle, turns, tx, ty, flight_time) = lead_target_from(
            cache,
            shooter_id,
            target_id,
            launch_turn_offset,
            v_true,
            from,
        )?;
        if let Some(res) = try_intercept_turn(
            cache,
            shooter_id,
            target_id,
            angle,
            turns,
            tx,
            ty,
            flight_time,
            v_true,
            launch_turn_offset,
            include_comets,
        ) {
            return Some(res);
        }
        from = turns + 1;
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
    !shot_blocked_exact(
        cache,
        shooter_id,
        target_id,
        angle,
        flight_time,
        v,
        launch_turn_offset,
    )
}
