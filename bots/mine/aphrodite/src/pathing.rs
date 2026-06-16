//! Lightweight pathing primitives used by Aphrodite's local simulator and
//! value-net fleet extrapolation. Launch-angle planning lives in the vendored
//! Apollo aimer (`apollo/aim.rs`); this module only predicts already-launched
//! fleet motion and shared collision physics.

use crate::{Fleet, GameState, BOARD_SIZE, CENTER_X, CENTER_Y, SUN_RADIUS};

pub const MAX_TIME: i64 = 100;

pub fn fleet_speed(ships: i64, max_speed: f64) -> f64 {
    if ships <= 1 {
        return 1.0;
    }
    let s = 1.0 + (max_speed - 1.0) * ((ships as f64).ln() / 1000.0f64.ln()).powf(1.5);
    s.clamp(1.0, max_speed)
}

#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

pub fn point_to_segment_distance(p: (f64, f64), v: (f64, f64), w: (f64, f64)) -> f64 {
    let l2 = (v.0 - w.0).powi(2) + (v.1 - w.1).powi(2);
    if l2 < 1e-12 {
        return dist(p, v);
    }
    let t = (((p.0 - v.0) * (w.0 - v.0) + (p.1 - v.1) * (w.1 - v.1)) / l2).clamp(0.0, 1.0);
    let proj = (v.0 + t * (w.0 - v.0), v.1 + t * (w.1 - v.1));
    dist(p, proj)
}

pub fn swept_pair_hit(
    a: (f64, f64),
    b: (f64, f64),
    p0: (f64, f64),
    p1: (f64, f64),
    r: f64,
) -> bool {
    let d0x = a.0 - p0.0;
    let d0y = a.1 - p0.1;
    let dvx = (b.0 - a.0) - (p1.0 - p0.0);
    let dvy = (b.1 - a.1) - (p1.1 - p0.1);
    let aq = dvx * dvx + dvy * dvy;
    let bq = 2.0 * (d0x * dvx + d0y * dvy);
    let cq = d0x * d0x + d0y * d0y - r * r;
    if aq < 1e-12 {
        return cq <= 0.0;
    }
    let disc = bq * bq - 4.0 * aq * cq;
    if disc < 0.0 {
        return false;
    }
    let sq = disc.sqrt();
    let t1 = (-bq - sq) / (2.0 * aq);
    let t2 = (-bq + sq) / (2.0 * aq);
    t2 >= 0.0 && t1 <= 1.0
}

#[inline]
fn on_board(p: (f64, f64)) -> bool {
    p.0 >= 0.0 && p.0 <= BOARD_SIZE && p.1 >= 0.0 && p.1 <= BOARD_SIZE
}

/// Predict where a fleet currently in flight will collide. Returns
/// `(planet_id, time)` or `None` if the fleet dies by sun, bounds, or timeout.
/// `time` is counted in future movement ticks from `state`.
pub fn predict_fleet_collision(fleet: &Fleet, state: &GameState) -> Option<(i64, i64)> {
    let speed = fleet_speed(fleet.ships, state.max_speed);
    let dx = speed * fleet.angle.cos();
    let dy = speed * fleet.angle.sin();
    let mut pos = (fleet.x, fleet.y);
    for dt in 1..=MAX_TIME {
        let new_pos = (pos.0 + dx, pos.1 + dy);
        for planet in &state.planets {
            let p_old = match state.planet_pos_at(planet, dt - 1) {
                Some(p) => p,
                None => continue,
            };
            let p_new = match state.planet_pos_at(planet, dt) {
                Some(p) => p,
                None => continue,
            };
            if !on_board(p_old) && !on_board(p_new) {
                continue;
            }
            if swept_pair_hit(pos, new_pos, p_old, p_new, planet.radius) {
                return Some((planet.id, dt));
            }
        }
        if !on_board(new_pos) {
            return None;
        }
        if point_to_segment_distance((CENTER_X, CENTER_Y), pos, new_pos) < SUN_RADIUS {
            return None;
        }
        pos = new_pos;
    }
    None
}
