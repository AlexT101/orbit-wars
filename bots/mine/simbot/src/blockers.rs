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
//! Per-turn collision is the relative-motion quadratic. For each turn we:
//!   1. Compute the fleet-mass / blocker relative trajectory `R(s)` at probe
//!      aim θ₀ = bearing(L → mid-turn blocker pos).
//!   2. Solve the closed-form vertex `s*` minimizing `|R(s)|²` on `[0, 1]`.
//!   3. Lock `s*` and find the exact arc of θ that keeps `|R(s*; θ)| ≤ r`
//!      via the law of cosines on the (L, Q(s*), fleet) triangle. The arc
//!      center is `bearing(L → Q(s*))` (refined from the mid-turn probe).
//!
//! Each [`BlockerEntry`] is one such arc: a single bearing, a single
//! closest-approach flight time, and a single tight half-arc. No max-pooling
//! across turns — adjacent turns produce adjacent entries, each as tight as
//! the geometry allows.

#![allow(dead_code)]

use std::f64::consts::{FRAC_PI_2, PI};

use crate::constants::{
    CENTER, EPISODE_STEPS, HORIZON, LAUNCH_CLEARANCE, MAX_SHIP_SPEED, SUN_RADIUS,
};
use crate::engine::fleet_speed;
use crate::entity_cache::{EntityCache, EntityKind};

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

/// Aim solver result: `(angle_radians, integer_turns, target_x, target_y)`.
pub type AimResult = (f64, i64, f64, f64);

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

/// Per-turn sub-sampling resolution for [`add_dynamic_bands`]. Each turn is
/// split into `DYN_SAMPLES` evenly-spaced `s` values in `[0, 1]`, and one
/// [`BlockerEntry`] is emitted per sample whose geometry can collide.
///
/// One entry per turn would be wrong: the law-of-cosines arc derived at a
/// single `s` only describes which aim angles `l` satisfy
/// `|fleet@s, l − Q(s)| ≤ r` for *that* `s`. For an off-`s` aim angle the
/// fleet's true closest-approach moment is a different `s`, and the arc
/// shifts. Sampling `s` and emitting an entry per sample lets the union of
/// the per-`s` arcs cover the full set of blocking aim angles, with adjacent
/// arcs overlapping for any reasonable obstacle motion (`Δθ` per sample
/// `≪ α`).
const DYN_SAMPLES: usize = 21;

/// For each integer turn `t`, sample `s ∈ [0, 1]` at [`DYN_SAMPLES`] points
/// and emit one [`BlockerEntry`] per sample whose triangle-inequality check
/// passes. Each entry is the exact blocking arc at its sample `s`; the union
/// of all entries' arcs covers the full set of aim angles whose chord during
/// turn `t` intersects the blocker disk.
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
    let r_sq = radius * radius;
    let inv_samples = 1.0 / (DYN_SAMPLES - 1) as f64;
    for t in 1..=max_lookahead {
        let Some([q0x, q0y]) = cache.position(blocker_id, launch_turn_offset + t - 1) else {
            continue;
        };
        let Some([q1x, q1y]) = cache.position(blocker_id, launch_turn_offset + t) else {
            continue;
        };

        let d0 = launch_offset + (t as f64 - 1.0) * v;
        let d_step = v; // d1 − d0
        let dqx = q1x - q0x;
        let dqy = q1y - q0y;
        let t_base = t as f64 - 1.0;

        for i in 0..DYN_SAMPLES {
            let s = i as f64 * inv_samples;
            let qsx = q0x + s * dqx;
            let qsy = q0y + s * dqy;
            let kx = qsx - lx;
            let ky = qsy - ly;
            let k_sq = kx * kx + ky * ky;
            if k_sq < 1e-18 {
                continue;
            }
            let k_mag = k_sq.sqrt();
            let d_s = d0 + s * d_step;

            // Triangle inequality: at this s, the fleet's circle of radius
            // D(s) around L can't reach Q(s)'s disk of radius r. Skip — no
            // aim angle collides at this instant.
            if (k_mag - d_s).abs() > radius {
                continue;
            }

            // Law of cosines on (L, fleet@s, Q(s)): the half-arc of θ for
            // which |fleet@s − Q(s)| ≤ r is exactly
            //   acos((|K|² + D² − r²) / (2·D·|K|)).
            let denom = 2.0 * d_s * k_mag;
            let half_arc = if denom < 1e-12 {
                FRAC_PI_2
            } else {
                let cos_arg = ((k_sq + d_s * d_s - r_sq) / denom).clamp(-1.0, 1.0);
                cos_arg.acos()
            };

            let bearing = ky.atan2(kx);
            let flight_t = t_base + s;

            out.push(BlockerEntry {
                bearing,
                flight_t,
                half_arc,
                aim_min: bearing - half_arc,
                aim_max: bearing + half_arc,
                blocker_id,
            });
        }
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

        let mut best_gap = f64::INFINITY;
        let mut best_s = 0.0_f64;
        for i in 0..LEAD_SAMPLES {
            let s = i as f64 * inv_samples;
            let qx = q0x + s * dqx;
            let qy = q0y + s * dqy;
            let k = (qx - lx).hypot(qy - ly);
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
        if (qx - lx).hypot(qy - ly) < 1e-9 {
            continue;
        }
        let angle = (qy - ly).atan2(qx - lx);
        let flight_time = t as f64 - 1.0 + best_s;
        return Some((angle, t, qx, qy, flight_time));
    }

    None
}

/// Aim from `shooter_id` to `target_id` with `ships`, launching now
/// (`launch_turn_offset == 0`). Quantizes the speed once, then routes the
/// quantized value through both the leader and the cached blocker table so
/// the (angle, turns) result is consistent with what the table tested.
pub fn aim_with_prediction(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
) -> Option<AimResult> {
    // Lead the target at the **exact** engine speed so the (angle, turns) we
    // emit lands on the actual orbital intercept point — quantizing here was
    // a real bug: a ~2.5% speed mismatch is enough for the planet to rotate
    // out from under our predicted hit point on long shots.
    let v_true = fleet_speed(ships.max(1), MAX_SHIP_SPEED);
    let (angle, turns, tx, ty, flight_time) =
        lead_target(cache, shooter_id, target_id, 0, v_true)?;
    // The blocker table is still keyed by the quantized bucket so different
    // ship counts that round to the same speed share a cached table; the
    // angle/flight_time we query it with come from the precise lead solve,
    // which is the conservative direction (slight over-rejection at the
    // boundary, never under-rejection).
    let table = cache.blocker_table(shooter_id, 0, ships);
    if is_blocked(&table, target_id, angle, flight_time) {
        return None;
    }
    Some((angle, turns, tx, ty))
}

/// Cheap revalidation of a previously-computed `(angle, turns)` against the
/// current obstacle set. Used by the aim cache when a comet may have spawned
/// since the result was cached.
pub fn shot_still_clear(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    angle: f64,
    turns: i64,
) -> bool {
    let table = cache.blocker_table(shooter_id, 0, ships);
    !is_blocked(&table, target_id, angle, turns as f64)
}
