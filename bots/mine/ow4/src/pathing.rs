//! Pathing: `dir_to_hit(source, target, num_ships, ...)` returns
//! `(angle, time)` where `time` is the arrival tick (production-count
//! convention: the target produces `time` times during the flight).
//!
//! Algorithm (adapted from duck/apollo):
//!   1. For each candidate arrival turn t, project target to its position at t
//!      and build the angular arc where the source-ring `speed*t` intersects.
//!   2. Subtract static obstacles (sun + non-orbiting non-comet planets) using
//!      the cone formula.
//!   3. Subtract moving obstacles: for each k in [1..max_t], if its position
//!      blocks the fleet's ring at k, subtract that arc.
//!   4. Pick the angle closest to the target's current center.
//!   5. Forward-sim along that angle to compute the exact arrival tick.

use crate::game::{Fleet, GameState, Planet, BOARD_SIZE, CENTER_X, CENTER_Y, SUN_RADIUS};
use std::f64::consts::{PI, TAU};

pub const MAX_TIME: i64 = 100;

pub fn fleet_speed(ships: i64, max_speed: f64) -> f64 {
    if ships <= 1 {
        return 1.0;
    }
    let s = 1.0 + (max_speed - 1.0) * ((ships as f64).ln() / 1000.0f64.ln()).powf(1.5);
    s.clamp(1.0, max_speed)
}

#[inline]
pub fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

#[inline]
pub fn normalize_angle(a: f64) -> f64 {
    let mut x = a % TAU;
    if x < 0.0 {
        x += TAU;
    }
    x
}

#[inline]
pub fn angle_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d > PI {
        d -= TAU;
    }
    while d <= -PI {
        d += TAU;
    }
    d
}

#[derive(Clone, Debug, Default)]
pub struct AngleSet {
    pub ivs: Vec<(f64, f64)>,
}

impl AngleSet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.ivs.is_empty()
    }

    pub fn add_arc(&mut self, center: f64, half: f64) {
        if half >= PI {
            self.ivs = vec![(0.0, TAU)];
            return;
        }
        if half <= 0.0 {
            return;
        }
        let lo = normalize_angle(center - half);
        let hi = lo + 2.0 * half;
        let parts: Vec<(f64, f64)> = if hi <= TAU {
            vec![(lo, hi)]
        } else {
            vec![(lo, TAU), (0.0, hi - TAU)]
        };
        let mut all = std::mem::take(&mut self.ivs);
        all.extend(parts);
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut out: Vec<(f64, f64)> = Vec::with_capacity(all.len());
        for iv in all {
            if let Some(last) = out.last_mut() {
                if iv.0 <= last.1 + 1e-9 {
                    last.1 = last.1.max(iv.1);
                    continue;
                }
            }
            out.push(iv);
        }
        self.ivs = out;
    }

    pub fn sub_arc(&mut self, center: f64, half: f64) {
        if half <= 0.0 {
            return;
        }
        if half >= PI {
            self.ivs.clear();
            return;
        }
        let lo = normalize_angle(center - half);
        let hi = lo + 2.0 * half;
        let parts: Vec<(f64, f64)> = if hi <= TAU {
            vec![(lo, hi)]
        } else {
            vec![(lo, TAU), (0.0, hi - TAU)]
        };
        for (s, e) in parts {
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(self.ivs.len() + 2);
            for &(a, b) in &self.ivs {
                if b <= s + 1e-12 || a >= e - 1e-12 {
                    out.push((a, b));
                } else {
                    if a < s {
                        out.push((a, s));
                    }
                    if b > e {
                        out.push((e, b));
                    }
                }
            }
            self.ivs = out.into_iter().filter(|(a, b)| b - a > 1e-9).collect();
        }
    }

    pub fn closest_to(&self, target: f64) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let t = normalize_angle(target);
        let mut best = None;
        let mut best_d = f64::INFINITY;
        for &(a, b) in &self.ivs {
            let cand = if t >= a && t <= b {
                t
            } else if t < a {
                a
            } else {
                b
            };
            let d = angle_diff(cand, t).abs();
            if d < best_d {
                best_d = d;
                best = Some(cand);
            }
        }
        best
    }
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

fn arc_half_angle(d_target: f64, r_target: f64, d_fleet: f64) -> Option<f64> {
    if d_fleet < 1e-9 || d_target < 1e-9 {
        return None;
    }
    if d_target > d_fleet + r_target + 1e-9 {
        return None;
    }
    if d_target + r_target < d_fleet - 1e-9 {
        return None;
    }
    if r_target >= d_target + d_fleet {
        return Some(PI);
    }
    let cos_half =
        (d_fleet * d_fleet + d_target * d_target - r_target * r_target) / (2.0 * d_fleet * d_target);
    let h = cos_half.clamp(-1.0, 1.0).acos();
    Some(h.max(1e-4))
}

#[derive(Debug, Clone, Copy)]
pub struct PathResult {
    pub angle: f64,
    pub time: i64,
}

pub fn dir_to_hit(
    source: &Planet,
    target: &Planet,
    num_ships: i64,
    state: &GameState,
    turns_in_future: i64,
) -> Option<PathResult> {
    let speed = fleet_speed(num_ships, state.max_speed);
    let src_pos = (source.x, source.y);
    let spawn_offset = source.radius + 0.1;

    let mut cand = AngleSet::empty();
    let mut max_target_t = 0i64;
    for t in 1..=MAX_TIME {
        let target_pos = match state.planet_pos_at(target, turns_in_future + t) {
            Some(p) => p,
            None => continue,
        };
        if !on_board(target_pos) {
            continue;
        }
        let d_target = dist(src_pos, target_pos);
        if d_target < 1e-6 {
            continue;
        }
        let angle_t = (target_pos.1 - src_pos.1).atan2(target_pos.0 - src_pos.0);
        let d_fleet = spawn_offset + speed * t as f64;
        if let Some(half) = arc_half_angle(d_target, target.radius, d_fleet) {
            cand.add_arc(angle_t, half);
            if t > max_target_t {
                max_target_t = t;
            }
        }
    }
    if cand.is_empty() {
        return None;
    }

    let d_sun = dist(src_pos, (CENTER_X, CENTER_Y));
    if d_sun <= SUN_RADIUS {
        return None;
    }
    let angle_sun = (CENTER_Y - src_pos.1).atan2(CENTER_X - src_pos.0);
    let h_sun = ((SUN_RADIUS + 0.05) / d_sun).clamp(-1.0, 1.0).asin();
    cand.sub_arc(angle_sun, h_sun);
    for obs in &state.planets {
        if obs.id == source.id || obs.id == target.id {
            continue;
        }
        if obs.is_orbiting || obs.is_comet {
            continue;
        }
        let d_obs = dist(src_pos, (obs.x, obs.y));
        if d_obs < 1e-6 {
            continue;
        }
        let angle_obs = (obs.y - src_pos.1).atan2(obs.x - src_pos.0);
        let h_obs = (((obs.radius + 0.1) / d_obs).clamp(-1.0, 1.0)).asin();
        cand.sub_arc(angle_obs, h_obs);
    }
    if cand.is_empty() {
        return None;
    }

    for k in 1..=max_target_t {
        let fleet_d_k = spawn_offset + speed * k as f64;
        for obs in &state.planets {
            if obs.id == source.id || obs.id == target.id {
                continue;
            }
            if !obs.is_orbiting && !obs.is_comet {
                continue;
            }
            let obs_pos = match state.planet_pos_at(obs, turns_in_future + k) {
                Some(p) => p,
                None => continue,
            };
            if !on_board(obs_pos) {
                continue;
            }
            let d_obs = dist(src_pos, obs_pos);
            if d_obs < 1e-6 {
                continue;
            }
            let buf = obs.radius + 0.25;
            if (fleet_d_k - d_obs).abs() > buf + speed * 0.5 + 0.5 {
                continue;
            }
            let angle_obs = (obs_pos.1 - src_pos.1).atan2(obs_pos.0 - src_pos.0);
            let h = (buf / d_obs).clamp(-1.0, 1.0).asin();
            cand.sub_arc(angle_obs, h);
            if cand.is_empty() {
                return None;
            }
        }
    }

    let target_now = state
        .planet_pos_at(target, turns_in_future)
        .unwrap_or((target.x, target.y));
    let angle_direct = (target_now.1 - src_pos.1).atan2(target_now.0 - src_pos.0);
    let angle = cand.closest_to(angle_direct)?;

    let dx = speed * angle.cos();
    let dy = speed * angle.sin();
    let mut pos = (
        src_pos.0 + spawn_offset * angle.cos(),
        src_pos.1 + spawn_offset * angle.sin(),
    );
    for t in 1..=MAX_TIME {
        let new_pos = (pos.0 + dx, pos.1 + dy);
        let t_old = match state.planet_pos_at(target, turns_in_future + t - 1) {
            Some(p) => p,
            None => break,
        };
        let t_new = match state.planet_pos_at(target, turns_in_future + t) {
            Some(p) => p,
            None => break,
        };
        if swept_pair_hit(pos, new_pos, t_old, t_new, target.radius) {
            return Some(PathResult { angle, time: t });
        }
        if !on_board(new_pos) {
            break;
        }
        pos = new_pos;
    }
    None
}

#[inline]
fn on_board(p: (f64, f64)) -> bool {
    p.0 >= 0.0 && p.0 <= BOARD_SIZE && p.1 >= 0.0 && p.1 <= BOARD_SIZE
}

/// Predict where an in-flight fleet will collide.
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
