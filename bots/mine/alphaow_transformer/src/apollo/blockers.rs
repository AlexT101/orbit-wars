//! Parametric line-of-sight obstacle tester.
//!
//! For each potential shooter L and launch turn t0, a [`BlockerTable`] is
//! built lazily and cached. The table is a flat list of [`BlockerEntry`]
//! records — one per (obstacle, turn) pair on which a swept-pair collision
//! is geometrically feasible. Each entry stores the exact arc of launch
//! bearings that produces a collision at that turn, plus the fractional
//! flight-time of the closest-approach moment within the turn.
//!
//! Shot queries are O(N_entries) with a tight bounding-box prefilter: each
//! entry's `[aim_min, aim_max]` interval is narrow (one turn's worth of arc
//! widening), so the vast majority of entries reject in two compares.
//!
//! Geometry primer
//! ───────────────
//! Fleet at launch L, angle θ, speed v sweeps from `f(s) = L + (launch_offset
//! + (t−1+s)·v)·û(θ)` to `f(s=1)` during turn `t`. The blocker (orbiter, comet)
//! sweeps from `Q(t−1)` to `Q(t)` over the same `s ∈ [0, 1]` — same chord
//! linearization the engine uses in `swept_pair_hit` for orbital motion.
//!
//! Per-turn arc construction. For each turn `t` we compute the **envelope**
//! of the per-`s` blocking arcs over `s ∈ [0, 1]`:
//!   1. At each `s`, the law of cosines on the (L, Q(s), fleet) triangle
//!      gives a per-`s` arc `θ ∈ [bearing(s) − α(s), bearing(s) + α(s)]`
//!      with `α(s) = acos((K² + D² − r²) / (2 D K))`.
//!   2. The union of those arcs over `s ∈ [0, 1]` is the set of aim angles
//!      blocked during turn `t`. Since `bearing(s)` and `α(s)` are smooth,
//!      the envelope edges are `max_s (bearing + α)` and `min_s (bearing − α)`
//!      — located via a coarse scan + parabolic refinement (no closed form
//!      since the critical-point equation is transcendental).
//!
//! Each [`BlockerEntry`] is one such envelope: a single bearing, a single
//! earliest-collision flight time, and a single tight half-arc per
//! (obstacle, turn). No max-pooling across turns.

#![allow(dead_code)]

use std::f64::consts::{FRAC_PI_2, PI};

use crate::apollo::constants::{
    CENTER, EPISODE_STEPS, HORIZON, LAUNCH_CLEARANCE, MAX_SHIP_SPEED, SUN_RADIUS,
};
use crate::apollo::engine::fleet_speed;
use crate::apollo::entity_cache::{EntityCache, EntityKind};

/// Per-turn sub-sampling resolution for [`lead_target`]. For each candidate
/// arrival turn we sample the target's chord at this many evenly-spaced `s`
/// values and pick the one that best aligns the fleet's radial chord with the
/// target's tangential chord (smallest radial gap). 11 samples gives ~0.1
/// turn-fraction resolution — enough to land within `target_radius` whenever
/// a swept-pair hit is geometrically possible.
const LEAD_SAMPLES: usize = 11;

/// Synthetic id used for the sun in [`BlockerEntry::blocker_id`]. Negative so
/// it cannot collide with any real planet/comet id.
const SUN_BLOCKER_ID: i64 = -1;

/// Fleet-speed quantization granularity for the blocker-table cache key.
/// Tables built with speeds within `1 / V_QUANT` of each other are pooled,
/// so different `ships` counts that round to the same speed bucket share
/// one cached table. 20 → 0.05 speed steps → ~2.5% max speed error.
const V_QUANT: f64 = 20.0;

/// Aim solver result: `(angle_radians, integer_turns, target_x, target_y,
/// fractional_flight_time)`. The fifth component is the real-valued flight
/// time at which the swept-pair test fires — strictly `≤ turns` and used by
/// the aim cache when re-verifying a stored shot after a comet spawn. Passing
/// `turns as f64` there is incorrect (over-conservative): a blocker whose
/// `flight_t` lies in `(flight_time, turns]` would falsely evict a still-valid
/// shot.
pub type AimResult = (f64, i64, f64, f64, f64);

/// One turn's worth of blocking arc for one obstacle.
///
/// A query `(aim, T)` is blocked iff `flight_t ≤ T` AND
/// `aim ≡ bearing (mod 2π)` lies in `[aim_min, aim_max]`. With per-turn
/// entries the bounding-box compare is also the exact arc check, so there
/// is no parametric solve at query time.
#[derive(Debug, Clone, Copy)]
pub struct BlockerEntry {
    pub bearing: f64,
    pub flight_t: f64,
    pub half_arc: f64,
    pub aim_min: f64,
    pub aim_max: f64,
    pub blocker_id: i64,
}

#[derive(Debug, Default, Clone)]
pub struct BlockerTable {
    pub entries: Vec<BlockerEntry>,
}

/// Branchless wrap of `a` into `(-π, π]`.
#[inline]
fn wrap_pi(a: f64) -> f64 {
    a - 2.0 * PI * ((a + PI) * (1.0 / (2.0 * PI))).floor()
}

/// Map a raw fleet speed to a quantized bucket key.
#[inline]
pub fn speed_bucket(ships: i64) -> i64 {
    let v_raw = fleet_speed(ships.max(1), MAX_SHIP_SPEED);
    (v_raw * V_QUANT).round() as i64
}

/// Inverse of [`speed_bucket`] — the canonical speed for a bucket.
#[inline]
pub fn bucket_to_speed(bucket: i64) -> f64 {
    bucket as f64 / V_QUANT
}

/// True iff `(aim, flight_time)` lies inside the entry's blocking arc.
/// `target_id` is skipped (the target isn't a blocker of itself).
#[inline]
fn entry_blocks(e: &BlockerEntry, target_id: i64, aim: f64, flight_time: f64) -> bool {
    if e.blocker_id == target_id {
        return false;
    }
    if flight_time < e.flight_t {
        return false;
    }
    let aim_w = e.bearing + wrap_pi(aim - e.bearing);
    aim_w >= e.aim_min && aim_w <= e.aim_max
}

/// `true` iff any non-target blocker in `table` blocks `(aim, flight_time)`.
pub fn is_blocked(table: &BlockerTable, target_id: i64, aim: f64, flight_time: f64) -> bool {
    table
        .entries
        .iter()
        .any(|e| entry_blocks(e, target_id, aim, flight_time))
}


/// Build the full blocker table for `(shooter_id, launch_turn_offset, v)`.
/// `v` is the quantized fleet speed — see [`speed_bucket`] / [`bucket_to_speed`].
pub fn build_blocker_table(
    cache: &EntityCache,
    shooter_id: i64,
    launch_turn_offset: i64,
    v: f64,
) -> BlockerTable {
    let Some([lx, ly]) = cache.position(shooter_id, launch_turn_offset) else {
        return BlockerTable::default();
    };
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;

    let abs_launch = cache.current_turn + launch_turn_offset;
    let max_lookahead = HORIZON.min((EPISODE_STEPS - 1 - abs_launch).max(0));

    // Heuristic: ~30 turns × ~6 dynamic blockers + a handful of static bands.
    let mut entries: Vec<BlockerEntry> =
        Vec::with_capacity((max_lookahead as usize + 1) * cache.entities.len());

    // Sun — stationary disk at the board center.
    add_static_band(
        &mut entries,
        SUN_BLOCKER_ID,
        lx,
        ly,
        CENTER,
        CENTER,
        SUN_RADIUS,
        launch_offset,
        v,
        max_lookahead as f64,
    );

    for (&bid, ent) in cache.entities.iter() {
        if bid == shooter_id {
            // Source planet is normally skipped — the fleet launches at
            // `radius + LAUNCH_CLEARANCE` from its center, outside the disk.
            // BUT: an orbiting source with tangential speed `ω·r_orbital`
            // (up to ~2.5/turn at ROTATION_LIMIT) can overtake a slow
            // prograde-launched fleet within turn 1. The engine's collision
            // loop doesn't skip the source, so without this we'd authorize
            // shots the engine will instantly delete. Static sources can't
            // move, so they remain skipped. Only emit the t=1 band — by
            // turn 2 the fleet has moved ≥ v ≥ 1 outward while the planet
            // moves ≤ 2.5 tangentially, so the chord geometry can no longer
            // re-contact.
            if matches!(ent.kind, EntityKind::OrbitingPlanet) {
                add_dynamic_bands(
                    &mut entries,
                    cache,
                    bid,
                    ent.radius,
                    lx,
                    ly,
                    launch_turn_offset,
                    1, // t=1 band only
                    launch_offset,
                    v,
                );
            }
            continue;
        }
        match ent.kind {
            EntityKind::StaticPlanet => {
                let Some([bx, by]) = cache.position(bid, launch_turn_offset) else {
                    continue;
                };
                add_static_band(
                    &mut entries,
                    bid,
                    lx,
                    ly,
                    bx,
                    by,
                    ent.radius,
                    launch_offset,
                    v,
                    max_lookahead as f64,
                );
            }
            EntityKind::OrbitingPlanet | EntityKind::Comet => {
                add_dynamic_bands(
                    &mut entries,
                    cache,
                    bid,
                    ent.radius,
                    lx,
                    ly,
                    launch_turn_offset,
                    max_lookahead,
                    launch_offset,
                    v,
                );
            }
        }
    }

    BlockerTable { entries }
}

/// One entry for a stationary disk: aim cone of half-width `asin(r/d)`,
/// `flight_t` = time fleet ring first reaches the disk's near edge.
fn add_static_band(
    out: &mut Vec<BlockerEntry>,
    blocker_id: i64,
    lx: f64,
    ly: f64,
    cx: f64,
    cy: f64,
    r: f64,
    launch_offset: f64,
    v: f64,
    max_flight_time: f64,
) {
    let dx = cx - lx;
    let dy = cy - ly;
    let d = (dx * dx + dy * dy).sqrt();
    if d < 1e-9 {
        return;
    }
    let near = (d - r - launch_offset) / v;
    if near > max_flight_time {
        return;
    }
    let t_lo = near.max(0.0);
    let bearing = dy.atan2(dx);
    let half = if d > r { (r / d).asin() } else { FRAC_PI_2 };
    out.push(BlockerEntry {
        bearing,
        flight_t: t_lo,
        half_arc: half,
        aim_min: bearing - half,
        aim_max: bearing + half,
        blocker_id,
    });
}

/// Number of `s ∈ [0, 1]` samples used to *bracket* the per-turn envelope
/// extrema in [`add_dynamic_bands`]. The envelope `u(s) = bearing(s) + α(s)`
/// is smooth in `s`, so a moderate scan locates each extremum's grid cell;
/// a golden-section search over the surrounding cell then refines to the
/// continuous extremum. 21 samples → step 0.05 in `s`, plenty fine to
/// resolve which cell contains the peak.
const ENVELOPE_SCAN: usize = 21;

/// Golden-section search precision used by [`add_dynamic_bands`] to refine
/// each extremum after the coarse scan. Iteration stops when the bracket
/// shrinks below this width in `s`; with envelope sensitivity bounded by the
/// fleet's chord speed (`v ≲ 6`/turn), 1e-5 in `s` corresponds to <1e-4 rad
/// in aim — well below any practical aim quantization.
const ENVELOPE_GSS_TOL: f64 = 1e-5;

/// Hard cap on golden-section iterations in case the bracket fails to shrink
/// (degenerate or numerically pathological envelope). With `φ ≈ 0.618`,
/// 50 iters gives `φ⁵⁰ ≈ 7e-11` — far below any geometry we'll see.
const ENVELOPE_GSS_MAX_ITERS: usize = 50;

/// Per-`s` blocking arc: aim bearings `θ` with `|fleet(s, θ) − Q(s)| ≤ r`.
/// Returns `(bearing(s), α(s))` or `None` if the fleet ring cannot reach the
/// blocker disk at this `s` (then the triangle inequality fails).
#[inline]
#[allow(clippy::too_many_arguments)]
fn arc_at(
    lx: f64,
    ly: f64,
    q0x: f64,
    q0y: f64,
    dqx: f64,
    dqy: f64,
    d0: f64,
    v: f64,
    radius: f64,
    s: f64,
) -> Option<(f64, f64)> {
    let kx = q0x + s * dqx - lx;
    let ky = q0y + s * dqy - ly;
    let k_sq = kx * kx + ky * ky;
    if k_sq < 1e-18 {
        return None;
    }
    let k_mag = k_sq.sqrt();
    let d_s = d0 + s * v;
    if (k_mag - d_s).abs() > radius {
        return None;
    }
    // Haversine form of the half-arc: from `2·sin²(α/2) = 1 − cos α =
    // (r² − (K−D)²)/(2DK)`, we get `α = 2·asin(√((r + D − K)(r + K − D) /
    // (4DK)))`. Equivalent to `acos((K² + D² − r²)/(2DK))` but numerically
    // robust near the common case `α → 0` (where `cos α → 1` and `acos`
    // loses precision to cancellation in `1 − cos α`).
    let factor_a = (radius + d_s - k_mag).max(0.0);
    let factor_b = (radius + k_mag - d_s).max(0.0);
    let denom = 4.0 * d_s * k_mag;
    let half = if denom < 1e-12 {
        FRAC_PI_2
    } else {
        2.0 * (factor_a * factor_b / denom).sqrt().min(1.0).asin()
    };
    Some((ky.atan2(kx), half))
}

/// Analytic feasibility window `[s_lo, s_hi] ⊆ [0, 1]` where the fleet ring
/// can collide with the blocker disk: the set of `s` satisfying
/// `|K(s) − D(s)| ≤ r`. Equivalent to `(D − r)² ≤ K² ≤ (D + r)²`, two
/// quadratics in `s` (since `K²` is quadratic and `(D ± r)²` is quadratic).
///
/// Returns `None` if no `s ∈ [0, 1]` is feasible — the caller can skip the
/// turn entirely. The exact `s_lo` becomes the entry's `flight_t` so engine
/// collisions arriving slightly before the first discrete scan sample are
/// not falsely pruned.
#[inline]
#[allow(clippy::too_many_arguments)]
fn feasibility_range(
    lx: f64,
    ly: f64,
    q0x: f64,
    q0y: f64,
    dqx: f64,
    dqy: f64,
    d0: f64,
    v: f64,
    radius: f64,
) -> Option<(f64, f64)> {
    let ux = q0x - lx;
    let uy = q0y - ly;
    let a = ux * ux + uy * uy; // K²(0)
    let b = ux * dqx + uy * dqy; // ½ d/ds K² at s=0
    let c = dqx * dqx + dqy * dqy; // ½ d²/ds² K² (constant)
    let big_a = c - v * v;

    // Boundary candidates: roots of K²(s) = (D(s) ± r)² clipped to [0, 1],
    // plus the [0, 1] endpoints. Max 6 candidates (2 endpoints + 2 roots × 2
    // sign choices).
    let mut bounds = [0.0_f64; 6];
    bounds[0] = 0.0;
    bounds[1] = 1.0;
    let mut n = 2usize;

    for &r_signed in &[radius, -radius] {
        let dr = d0 + r_signed;
        let big_b = b - dr * v;
        let big_c = a - dr * dr;
        if big_a.abs() < 1e-12 {
            // Degenerate (blocker chord speed equals fleet speed): linear.
            if big_b.abs() >= 1e-12 {
                let s = -big_c / (2.0 * big_b);
                if (0.0..=1.0).contains(&s) {
                    bounds[n] = s;
                    n += 1;
                }
            }
        } else {
            let disc = big_b * big_b - big_a * big_c;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                for &root in &[(-big_b - sq) / big_a, (-big_b + sq) / big_a] {
                    if (0.0..=1.0).contains(&root) {
                        bounds[n] = root;
                        n += 1;
                    }
                }
            }
        }
    }
    bounds[..n].sort_by(|x, y| x.partial_cmp(y).unwrap());

    // Test feasibility at each interval midpoint using sqrt-free comparisons.
    let feasible_at = |s: f64| -> bool {
        let k_sq = a + 2.0 * b * s + c * s * s;
        if k_sq < 0.0 {
            return false;
        }
        let d_s = d0 + s * v;
        let d_plus = d_s + radius;
        if k_sq > d_plus * d_plus {
            return false;
        }
        let d_minus = d_s - radius;
        // If D ≤ r, lower bound K ≥ D − r is trivially satisfied (K ≥ 0).
        d_minus <= 0.0 || k_sq >= d_minus * d_minus
    };

    let mut s_lo = f64::INFINITY;
    let mut s_hi = f64::NEG_INFINITY;
    for i in 0..n.saturating_sub(1) {
        let lo = bounds[i];
        let hi = bounds[i + 1];
        if (hi - lo) < 1e-15 {
            continue;
        }
        if feasible_at(0.5 * (lo + hi)) {
            if lo < s_lo {
                s_lo = lo;
            }
            if hi > s_hi {
                s_hi = hi;
            }
        }
    }
    if s_lo <= s_hi {
        Some((s_lo, s_hi))
    } else {
        None
    }
}

/// For each integer turn `t`, emit a single [`BlockerEntry`] whose aim arc is
/// the **per-turn envelope** of the per-`s` arcs over `s ∈ [0, 1]`:
///
///   `aim_max(t) = max_{s ∈ [0,1]} (bearing(s) + α(s))`
///   `aim_min(t) = min_{s ∈ [0,1]} (bearing(s) − α(s))`
///
/// The earlier per-`s` sample union (DYN_SAMPLES entries per turn, plus
/// analytical `K(s) = D(s)` ring-touch samples) had two bugs:
///   1. *Tunneling*: the envelope peak `s*` falls between grid points; each
///      grid sample's arc just barely misses the engine's swept-pair aim,
///      while the continuous envelope covers it. Adding `K = D` samples
///      didn't fix this — `K = D` maximizes `α(s)`, but the envelope edge is
///      `bearing(s) + α(s)`, a different critical point.
///   2. Bloated entry count: ~23 entries / turn / blocker, all hit on every
///      `is_blocked` query.
///
/// The envelope is smooth (`bearing(s)` and `α(s)` are both C¹ on the
/// feasibility range), so a coarse scan reliably brackets each extremum's
/// grid cell. A golden-section search over the bracketing cell then
/// refines to the continuous extremum, closing the tunneling gap that
/// uniform sampling alone could not.
fn add_dynamic_bands(
    out: &mut Vec<BlockerEntry>,
    cache: &EntityCache,
    blocker_id: i64,
    radius: f64,
    lx: f64,
    ly: f64,
    launch_turn_offset: i64,
    max_lookahead: i64,
    launch_offset: f64,
    v: f64,
) {
    // Hoist the entity lookup once (saves one HashMap.get per turn) and
    // bound the *latest* feasible turn from `K(s)`'s integer-turn maximum.
    // We do **not** prune the earliest turn: `K²(s)` is convex on each
    // chord (an upward parabola in `s`), so its *maximum* over `[0,1]` is
    // always at an endpoint, making `k_max` from integer turns a true
    // upper bound. Its *minimum*, though, can be interior — a fast comet
    // passing near the launcher can have a chord whose closest point to
    // `L` is well below both endpoint distances, so an integer-turn
    // `k_min` would over-estimate the true minimum and wrongly skip
    // feasible early turns. `feasibility_range` handles the early-turn
    // case correctly, so the inner loop starts at `t = 1` as before.
    let Some(entity) = cache.get(blocker_id) else {
        return;
    };
    let positions = &entity.positions;
    let abs_base = cache.current_turn + launch_turn_offset;

    let mut k_max_sq = 0.0_f64;
    let mut any_position = false;
    for t in 0..=max_lookahead {
        let abs = abs_base + t;
        if abs < 0 || (abs as usize) >= positions.len() {
            continue;
        }
        let Some([qx, qy]) = positions[abs as usize] else {
            continue;
        };
        any_position = true;
        let dx = qx - lx;
        let dy = qy - ly;
        let d_sq = dx * dx + dy * dy;
        if d_sq > k_max_sq {
            k_max_sq = d_sq;
        }
    }
    if !any_position {
        return;
    }
    let k_max = k_max_sq.sqrt();

    // `D(start of turn t) = launch_offset + (t-1)·v`. If at the *start* of
    // turn `t` the fleet ring is already past the farthest blocker position
    // (`D ≥ k_max + r`), then for every `s ∈ [0,1]` of this and later
    // turns, `D > K + r`, no collision possible. So we cap at
    // `t_max = ⌈(k_max + r − launch_offset)/v⌉ + 1`.
    let t_max_raw = ((k_max + radius - launch_offset) / v).ceil() + 1.0;
    let t_max = (t_max_raw as i64).min(max_lookahead).max(1);

    for t in 1..=t_max {
        let abs_t0 = (abs_base + t - 1) as usize;
        let abs_t1 = (abs_base + t) as usize;
        let Some([q0x, q0y]) = positions[abs_t0] else {
            continue;
        };
        let Some([q1x, q1y]) = positions[abs_t1] else {
            continue;
        };
        let dqx = q1x - q0x;
        let dqy = q1y - q0y;
        let d0 = launch_offset + (t as f64 - 1.0) * v;
        let t_base = t as f64 - 1.0;

        // Analytic feasibility window for this turn — exact roots of
        // `K²(s) = (D(s) ± r)²`. Empty means the fleet ring never reaches
        // the blocker disk during this turn; skip without scanning.
        let Some((s_lo, s_hi)) =
            feasibility_range(lx, ly, q0x, q0y, dqx, dqy, d0, v, radius)
        else {
            continue;
        };

        // Sample envelope values at uniform `s` within the feasibility window.
        // Restricting to the window tightens effective resolution and ensures
        // every sample yields a valid arc.
        let span = s_hi - s_lo;
        let scan_step = span / (ENVELOPE_SCAN - 1) as f64;
        let mut scan_u = [f64::NAN; ENVELOPE_SCAN];
        let mut scan_l = [f64::NAN; ENVELOPE_SCAN];
        let mut bearing_ref = f64::NAN;
        let mut i_u: Option<usize> = None;
        let mut i_l: Option<usize> = None;

        for i in 0..ENVELOPE_SCAN {
            let s = s_lo + i as f64 * scan_step;
            let Some((b, h)) = arc_at(lx, ly, q0x, q0y, dqx, dqy, d0, v, radius, s) else {
                continue;
            };
            if !bearing_ref.is_finite() {
                bearing_ref = b;
            }
            // Unwrap relative to bearing_ref so consecutive samples that
            // straddle the ±π branch cut compare correctly.
            let b_u = bearing_ref + wrap_pi(b - bearing_ref);
            scan_u[i] = b_u + h;
            scan_l[i] = b_u - h;
            if i_u.map_or(true, |j| scan_u[i] > scan_u[j]) {
                i_u = Some(i);
            }
            if i_l.map_or(true, |j| scan_l[i] < scan_l[j]) {
                i_l = Some(i);
            }
        }

        let Some(i_u) = i_u else { continue };
        let i_l = i_l.expect("argmin_l exists when argmax_u exists");

        // Golden-section refinement on the cell bracketing each discrete
        // extremum. Clipped to the feasibility window so the search never
        // wanders into infeasible `s`.
        let eval = |s: f64, maximize: bool| -> f64 {
            let Some((br, h)) = arc_at(lx, ly, q0x, q0y, dqx, dqy, d0, v, radius, s) else {
                return if maximize { f64::NEG_INFINITY } else { f64::INFINITY };
            };
            let b_u = bearing_ref + wrap_pi(br - bearing_ref);
            if maximize { b_u + h } else { b_u - h }
        };
        let golden = |a0: f64, b0: f64, maximize: bool| -> f64 {
            let phi = (5f64.sqrt() - 1.0) * 0.5; // ≈ 0.618
            let mut a = a0;
            let mut b = b0;
            let mut c = b - phi * (b - a);
            let mut d = a + phi * (b - a);
            let mut fc = eval(c, maximize);
            let mut fd = eval(d, maximize);
            for _ in 0..ENVELOPE_GSS_MAX_ITERS {
                if (b - a) < ENVELOPE_GSS_TOL {
                    break;
                }
                let pick_left = if maximize { fc > fd } else { fc < fd };
                if pick_left {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - phi * (b - a);
                    fc = eval(c, maximize);
                } else {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + phi * (b - a);
                    fd = eval(d, maximize);
                }
            }
            if maximize { fc.max(fd) } else { fc.min(fd) }
        };
        let refine = |i: usize, f_b: f64, maximize: bool| -> f64 {
            let bracket_lo = if i == 0 { s_lo } else { s_lo + (i - 1) as f64 * scan_step };
            let bracket_hi = if i + 1 >= ENVELOPE_SCAN {
                s_hi
            } else {
                s_lo + (i + 1) as f64 * scan_step
            };
            let refined = golden(bracket_lo, bracket_hi, maximize);
            if maximize { refined.max(f_b) } else { refined.min(f_b) }
        };

        let aim_max = refine(i_u, scan_u[i_u], true);
        let aim_min = refine(i_l, scan_l[i_l], false);
        let bearing = 0.5 * (aim_min + aim_max);
        let half_arc = 0.5 * (aim_max - aim_min);
        let s_feasible_min = s_lo;

        out.push(BlockerEntry {
            bearing,
            flight_t: t_base + s_feasible_min,
            half_arc,
            aim_min,
            aim_max,
            blocker_id,
        });
    }
}

/// Lead a (possibly moving) target. Returns `(angle, integer_turns, target_x,
/// target_y, fractional_flight_time)` where `(target_x, target_y) = Q(s*)` is
/// the point on the target's chord during turn `integer_turns` at which the
/// engine's swept-pair test fires, and `fractional_flight_time = integer_turns
/// − 1 + s*`. `v` is the quantized fleet speed.
///
/// Approach: walk integer turn `t` forward from the earliest geometrically
/// feasible turn. For each `t`, find `s* ∈ [0, 1]` that minimizes the radial
/// gap `|K(s) − D(s)|` between the target's chord position `Q(s)` and the
/// fleet's chord distance `D(s) = launch_offset + (t − 1 + s)·v` (same chord
/// linearization the engine uses). Aiming `θ = bearing(L → Q(s*))` puts the
/// fleet at distance `D(s*)` along the line through `Q(s*)`, so the actual
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

    let inv_samples = 1.0 / (LEAD_SAMPLES - 1) as f64;

    for t in start..=max_lookahead {
        let [q0x, q0y] = cache.position(target_id, launch_turn_offset + t - 1)?;
        let [q1x, q1y] = cache.position(target_id, launch_turn_offset + t)?;
        let dqx = q1x - q0x;
        let dqy = q1y - q0y;
        let d0 = launch_offset + (t as f64 - 1.0) * v;

        // `sqrt(a² + b²)` over `hypot`: coordinates are bounded by BOARD_SIZE
        // ≈ 100 so overflow guards are dead weight; ~3-4× faster in this hot
        // argmin loop.
        let mut best_gap = f64::INFINITY;
        let mut best_s = 0.0_f64;
        for i in 0..LEAD_SAMPLES {
            let s = i as f64 * inv_samples;
            let qx = q0x + s * dqx - lx;
            let qy = q0y + s * dqy - ly;
            let k = (qx * qx + qy * qy).sqrt();
            let d = d0 + s * v;
            let gap = (k - d).abs();
            if gap < best_gap {
                best_gap = gap;
                best_s = s;
            }
        }

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

/// Aim from `shooter_id` to `target_id` with `ships`, for a fleet launching
/// at `launch_turn_offset` turns after the cache's current turn (pass `0`
/// for "launch now"). Source, target, and obstacle positions are all
/// evaluated at the launch turn so the blocker table reflects the real
/// flight window — required by the early-game DFS, which scores delayed
/// launches and would otherwise falsely assume current geometry.
///
/// Quantizes the speed once, then routes the quantized value through both
/// the leader and the cached blocker table so the (angle, turns) result is
/// consistent with what the table tested.
pub fn aim_with_prediction(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    launch_turn_offset: i64,
) -> Option<AimResult> {
    // Lead the target at the **exact** engine speed so the (angle, turns) we
    // emit lands on the actual orbital intercept point — quantizing here was
    // a real bug: a ~2.5% speed mismatch is enough for the planet to rotate
    // out from under our predicted hit point on long shots.
    let v_true = fleet_speed(ships.max(1), MAX_SHIP_SPEED);
    let (angle, turns, tx, ty, flight_time) =
        lead_target(cache, shooter_id, target_id, launch_turn_offset, v_true)?;
    // The blocker table is still keyed by the quantized bucket so different
    // ship counts that round to the same speed share a cached table; the
    // angle/flight_time we query it with come from the precise lead solve,
    // which is the conservative direction (slight over-rejection at the
    // boundary, never under-rejection).
    let table = cache.blocker_table(shooter_id, launch_turn_offset, ships);

    // Extreme edges of the union of blocking arcs, computed as signed deltas
    // relative to `angle` via wrap_pi so entries with bearings on opposite
    // sides of the ±π branch cut are compared in the same coordinate frame.
    // entry_blocks guarantees lo ≤ 0 ≤ hi for any blocking entry.
    let (mut delta_lo, mut delta_hi) = (0.0_f64, 0.0_f64);
    let mut any_blocked = false;
    for e in table.entries.iter() {
        if entry_blocks(e, target_id, angle, flight_time) {
            let lo = wrap_pi(e.aim_min - angle);
            let hi = wrap_pi(e.aim_max - angle);
            if lo < delta_lo {
                delta_lo = lo;
            }
            if hi > delta_hi {
                delta_hi = hi;
            }
            any_blocked = true;
        }
    }
    if !any_blocked {
        return Some((angle, turns, tx, ty, flight_time));
    }

    // Direct path is blocked. Try angles just outside the full blocked arc's
    // edges — if either still hits the target at the same arrival turn and
    // clears all obstacles, prefer it over returning None. Target validity is
    // checked by placing the fleet at `flight_time` along the candidate angle
    // and verifying its distance to (tx, ty) is within target_radius.
    let [lx, ly] = cache.position(shooter_id, launch_turn_offset)?;
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let target_radius = cache.get(target_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
    let ring_d = launch_offset + flight_time * v_true;
    let r_sq = target_radius * target_radius;

    // 1e-4 rad is enough to numerically clear the arc edge while staying well
    // within any real target's valid hit arc (target radii are several units,
    // angular shift of 1e-4 rad at typical ranges moves the fleet < 0.01 units).
    const STEP: f64 = 1e-4;
    for &theta_try in &[angle + delta_lo - STEP, angle + delta_hi + STEP] {
        let fx = lx + ring_d * theta_try.cos();
        let fy = ly + ring_d * theta_try.sin();
        let dx = fx - tx;
        let dy = fy - ty;
        if dx * dx + dy * dy > r_sq {
            continue; // outside target's valid arc at this turn
        }
        if !is_blocked(&table, target_id, theta_try, flight_time) {
            return Some((theta_try, turns, tx, ty, flight_time));
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
/// `slot - current_turn`. The blocker table is built for that offset so
/// re-verification sees obstacle positions at the actual launch time.
pub fn shot_still_clear(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    angle: f64,
    flight_time: f64,
    launch_turn_offset: i64,
) -> bool {
    let table = cache.blocker_table(shooter_id, launch_turn_offset, ships);
    !is_blocked(&table, target_id, angle, flight_time)
}
