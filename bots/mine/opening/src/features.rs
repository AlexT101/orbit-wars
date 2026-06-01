//! First-frame feature extractor — bit-compatible with
//! `bots/mine/alphaow/train/train_first_owned_xgb.py` for the v23 model
//! (34 base features + 4 bandwidth pairs of KDE features = 50 total).

use alphaow_bot::{GameState, Planet, BOARD_SIZE, CENTER_X, CENTER_Y};
use std::f64::consts::{PI, TAU};

pub const N_FEATURES: usize = 50;
const NO_NEIGHBOR_DIST: f64 = 200.0;
const INV_CAP: f64 = 10.0;
const KDE_BANDWIDTHS: [f64; 4] = [20.0, 30.0, 40.0, 50.0];

fn safe_inv(x: f64) -> f64 {
    if x <= 1e-3 { return INV_CAP; }
    let v = 1.0 / x;
    if v <= INV_CAP { v } else { INV_CAP }
}

fn fleet_speed_py(n_ships: f64, max_speed: f64) -> f64 {
    if n_ships <= 1.0 { return 1.0; }
    let s = 1.0 + (max_speed - 1.0)
        * (n_ships.ln() / 1000.0f64.ln()).powf(1.5);
    s.clamp(1.0, max_speed)
}

fn signed_angle_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d > PI { d -= TAU; }
    while d <= -PI { d += TAU; }
    d
}

fn kde_at(target: (f64, f64), planets: &[(f64, f64, f64)], sigma: f64) -> f64 {
    let two_sig2 = 2.0 * sigma * sigma;
    let mut s = 0.0;
    for &(x, y, w) in planets {
        let dx = target.0 - x;
        let dy = target.1 - y;
        let d2 = dx * dx + dy * dy;
        s += w * (-d2 / two_sig2).exp();
    }
    s
}

/// Find each player's home (= the planet they own at t=0).
pub fn homes(state: &GameState) -> Option<((f64, f64, i64, i64), (f64, f64, i64, i64))> {
    let mut h0 = None;
    let mut h1 = None;
    for p in &state.planets {
        if p.owner == 0 && h0.is_none() {
            h0 = Some((p.x, p.y, p.production, p.id));
        } else if p.owner == 1 && h1.is_none() {
            h1 = Some((p.x, p.y, p.production, p.id));
        }
    }
    Some((h0?, h1?))
}

/// Returns a 50-d feature vector for `planet` from `perspective`'s point of view.
pub fn extract(state: &GameState, planet: &Planet, perspective: i32) -> [f32; N_FEATURES] {
    let (h0, h1) = homes(state).expect("homes must exist at turn 0");
    let (my_home, opp_home) = if perspective == 0 { (h0, h1) } else { (h1, h0) };

    // ----- per-game aggregates -----
    let mut n_stat = 0i64;
    let mut n_orb = 0i64;
    let mut sum_stat_prod = 0i64;
    let mut sum_orb_prod = 0i64;
    let mut side_closer: Vec<(i64, i32)> = Vec::with_capacity(state.planets.len());
    for p in &state.planets {
        let d0 = (p.x - h0.0).hypot(p.y - h0.1);
        let d1 = (p.x - h1.0).hypot(p.y - h1.1);
        let side = if d0 <= d1 { 0 } else { 1 };
        side_closer.push((p.id, side));
        if p.is_orbiting {
            n_orb += 1;
            sum_orb_prod += p.production;
        } else if !p.is_comet {
            n_stat += 1;
            sum_stat_prod += p.production;
        }
    }
    let side_of = |id: i64| -> i32 {
        side_closer.iter().find(|(pid, _)| *pid == id).map(|(_, s)| *s).unwrap_or(0)
    };

    // ----- pre-built KDE point lists -----
    let pos_unit: Vec<(f64, f64, f64)> = state.planets.iter()
        .map(|p| (p.x, p.y, 1.0)).collect();
    let pos_prod: Vec<(f64, f64, f64)> = state.planets.iter()
        .map(|p| (p.x, p.y, p.production as f64)).collect();

    // ----- home bookkeeping -----
    let my_home_x = my_home.0;
    let my_home_y = my_home.1;
    let my_home_prod = my_home.2;
    let my_home_pid = my_home.3;
    let opp_home_x = opp_home.0;
    let opp_home_y = opp_home.1;
    let opp_home_pid = opp_home.3;
    let home0_x = h0.0;
    let home0_y = h0.1;
    let my_home_dist_to_edge_game = home0_x.min(home0_y)
        .min(BOARD_SIZE - home0_x)
        .min(BOARD_SIZE - home0_y);

    let my_home_pl = state.planets.iter().find(|p| p.id == my_home_pid).unwrap();
    let opp_home_pl = state.planets.iter().find(|p| p.id == opp_home_pid).unwrap();
    let my_home_stat = !my_home_pl.is_orbiting && !my_home_pl.is_comet;
    let my_home_orb = my_home_pl.is_orbiting;
    let opp_home_stat = !opp_home_pl.is_orbiting && !opp_home_pl.is_comet;
    let opp_home_orb = opp_home_pl.is_orbiting;

    // ----- this planet -----
    let p = planet;
    let x = p.x;
    let y = p.y;
    let prod = p.production;
    let ships = p.ships;
    let is_stat = !p.is_orbiting && !p.is_comet;
    let is_orb = p.is_orbiting;
    let planet_theta = (y - CENTER_Y).atan2(x - CENTER_X);

    let mut closest_side0 = NO_NEIGHBOR_DIST;
    let mut closest_side1 = NO_NEIGHBOR_DIST;
    let mut sum_inv_side0 = 0.0;
    let mut sum_inv_side1 = 0.0;
    let mut cnt_side0 = 0i64;
    let mut cnt_side1 = 0i64;
    for q in &state.planets {
        if q.id == p.id { continue; }
        let d = (x - q.x).hypot(y - q.y);
        if d < 1e-6 { continue; }
        let inv = 1.0 / d;
        if side_of(q.id) == 0 {
            if d < closest_side0 { closest_side0 = d; }
            sum_inv_side0 += inv;
            cnt_side0 += 1;
        } else {
            if d < closest_side1 { closest_side1 = d; }
            sum_inv_side1 += inv;
            cnt_side1 += 1;
        }
    }
    let mean_inv_side0 = if cnt_side0 > 0 { sum_inv_side0 / cnt_side0 as f64 } else { 0.0 };
    let mean_inv_side1 = if cnt_side1 > 0 { sum_inv_side1 / cnt_side1 as f64 } else { 0.0 };

    let mut kde_pairs: Vec<f64> = Vec::with_capacity(2 * KDE_BANDWIDTHS.len());
    let mut inv_kde_pairs: Vec<f64> = Vec::with_capacity(2 * KDE_BANDWIDTHS.len());
    for &sigma in &KDE_BANDWIDTHS {
        let num_d = kde_at((x, y), &pos_unit, sigma) - 1.0;
        let prod_d = kde_at((x, y), &pos_prod, sigma) - prod as f64;
        kde_pairs.push(num_d);
        kde_pairs.push(prod_d);
        inv_kde_pairs.push(safe_inv(num_d));
        inv_kde_pairs.push(safe_inv(prod_d));
    }

    let dist_to_edge = x.min(y).min(BOARD_SIZE - x).min(BOARD_SIZE - y);
    // orbital_radius in python's planet_geom = sqrt((ix-50)² + (iy-50)²),
    // which Rust's parse_state already computes as Planet.orbital_radius
    // for both orbiting and stationary planets.
    let orbital_radius = p.orbital_radius;

    // ----- perspective-dependent block -----
    let d_my = (x - my_home_x).hypot(y - my_home_y);
    let d_opp = (x - opp_home_x).hypot(y - opp_home_y);
    let d_center = (x - CENTER_X).hypot(y - CENTER_Y);
    let my_home_theta = (my_home_y - CENTER_Y).atan2(my_home_x - CENTER_X);
    let opp_home_theta = (opp_home_y - CENTER_Y).atan2(opp_home_x - CENTER_X);
    let ang_to_me = signed_angle_diff(planet_theta, my_home_theta);
    let ang_to_opp = signed_angle_diff(planet_theta, opp_home_theta);

    let (closest_my, closest_opp, mean_inv_my, mean_inv_opp) = if perspective == 0 {
        (closest_side0, closest_side1, mean_inv_side0, mean_inv_side1)
    } else {
        (closest_side1, closest_side0, mean_inv_side1, mean_inv_side0)
    };

    let (ang_div_omega_me, ang_div_omega_opp) =
        if is_orb && state.angular_velocity.abs() > 1e-9 {
            (ang_to_me / state.angular_velocity, ang_to_opp / state.angular_velocity)
        } else {
            (0.0, 0.0)
        };

    let turns_to_cap = if my_home_prod > 0 {
        (ships as f64 - 10.0) / my_home_prod as f64
    } else { 0.0 };
    let fleet_n = (ships + 1).max(1) as f64;
    let travel_t = d_my / fleet_speed_py(fleet_n, state.max_speed);

    // ----- pack -----
    let base: [f64; 34] = [
        d_my, d_opp, d_center,
        if is_stat { 1.0 } else { 0.0 },
        if is_orb { 1.0 } else { 0.0 },
        ang_to_me, ang_to_opp,
        prod as f64, ships as f64,
        n_stat as f64, n_orb as f64,
        sum_stat_prod as f64, sum_orb_prod as f64,
        my_home_prod as f64,
        closest_my, closest_opp,
        dist_to_edge, state.angular_velocity, orbital_radius,
        if my_home_stat { 1.0 } else { 0.0 },
        if my_home_orb { 1.0 } else { 0.0 },
        if opp_home_stat { 1.0 } else { 0.0 },
        if opp_home_orb { 1.0 } else { 0.0 },
        ang_div_omega_me, ang_div_omega_opp,
        turns_to_cap, travel_t,
        mean_inv_my, mean_inv_opp,
        safe_inv(d_my), safe_inv(d_opp),
        safe_inv(closest_my), safe_inv(closest_opp),
        my_home_dist_to_edge_game,
    ];

    let mut out = [0f32; N_FEATURES];
    for (i, &v) in base.iter().enumerate() { out[i] = v as f32; }
    let mut idx = 34;
    for &v in &kde_pairs { out[idx] = v as f32; idx += 1; }
    for &v in &inv_kde_pairs { out[idx] = v as f32; idx += 1; }
    debug_assert_eq!(idx, N_FEATURES);
    out
}
