//! Like `extract_v3` but with 12 extras (user request 2026-05-29):
//!     [tick,
//!      d_my_stat_their_stat_now, d_my_stat_their_orb_now,
//!      d_my_orb_their_stat_now,  d_my_orb_their_orb_now,
//!      d_my_stat_their_stat_ext, d_my_stat_their_orb_ext,
//!      d_my_orb_their_stat_ext,  d_my_orb_their_orb_ext,
//!      n_total_static, n_total_orbit,
//!      angular_velocity]
//!
//! Record format (binary, per observation):
//!     step:i64, player:i32, summary_v2:[f32; 46], extras:[f32; 12]
//!   → 8 + 4 + 184 + 48 = 244 bytes per record.
//!
//! "Static" = non-comet, non-orbiting. "Orbit" = non-comet, orbiting.
//! Comets are excluded from distance buckets. If either side has zero
//! planets of the required type, the distance feature is 0.0 (a placeholder
//! the model can learn from; can't use NaN since the loader is f32 strict).

use alphaow_bot::{parse_state, value_net, Planet};
use serde_json::Value;
use std::io::{self, BufRead, Write};

const EXTRA_DIM: usize = 12;

#[derive(Copy, Clone)]
enum PType { Static, Orbit }

fn matches_type(p: &Planet, t: PType) -> bool {
    if p.is_comet { return false; }
    match t {
        PType::Static => !p.is_orbiting,
        PType::Orbit  =>  p.is_orbiting,
    }
}

fn nearest_pair_dist<F1, F2>(planets: &[Planet], is_a: F1, is_b: F2) -> f32
where F1: Fn(&Planet) -> bool, F2: Fn(&Planet) -> bool {
    let mut best = f32::INFINITY;
    for a in planets.iter().filter(|p| is_a(p)) {
        for b in planets.iter().filter(|p| is_b(p)) {
            let dx = (a.x - b.x) as f32;
            let dy = (a.y - b.y) as f32;
            let d2 = dx * dx + dy * dy;
            if d2 < best { best = d2; }
        }
    }
    if best.is_finite() { best.sqrt() } else { 0.0 }
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut buf = String::new();
    let mut handle = stdin.lock();
    loop {
        buf.clear();
        let n = handle.read_line(&mut buf)?;
        if n == 0 { break; }
        let line = buf.trim_end();
        if line.is_empty() { continue; }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v, Err(_) => continue,
        };
        let state = parse_state(&v);
        let me = state.player;
        let feats = value_net::summary_features_v2::extract(&state, me);

        // ---- 12 extras ----
        let tick = state.step as f32;
        let av   = state.angular_velocity as f32;

        // Counts
        let mut n_static = 0u32;
        let mut n_orbit  = 0u32;
        for p in &state.planets {
            if p.is_comet { continue; }
            if p.is_orbiting { n_orbit += 1; } else { n_static += 1; }
        }

        // Pre-extrap distances (4 type pairs)
        let is_mine    = |p: &Planet| p.owner == me;
        let is_enemy   = |p: &Planet| p.owner != me && p.owner != -1;
        let now_ss = nearest_pair_dist(&state.planets,
            |p| is_mine(p)  && matches_type(p, PType::Static),
            |p| is_enemy(p) && matches_type(p, PType::Static));
        let now_so = nearest_pair_dist(&state.planets,
            |p| is_mine(p)  && matches_type(p, PType::Static),
            |p| is_enemy(p) && matches_type(p, PType::Orbit));
        let now_os = nearest_pair_dist(&state.planets,
            |p| is_mine(p)  && matches_type(p, PType::Orbit),
            |p| is_enemy(p) && matches_type(p, PType::Static));
        let now_oo = nearest_pair_dist(&state.planets,
            |p| is_mine(p)  && matches_type(p, PType::Orbit),
            |p| is_enemy(p) && matches_type(p, PType::Orbit));

        // Post-extrap distances: same planet positions, extrap ownership
        let ext_map = value_net::extrapolate_fleets(&state);
        let ext_owner = |p: &Planet| ext_map.get(&p.id).map(|x| x.0).unwrap_or(p.owner);
        let ext_is_mine  = |p: &Planet| ext_owner(p) == me;
        let ext_is_enemy = |p: &Planet| { let o = ext_owner(p); o != me && o != -1 };
        let ext_ss = nearest_pair_dist(&state.planets,
            |p| ext_is_mine(p)  && matches_type(p, PType::Static),
            |p| ext_is_enemy(p) && matches_type(p, PType::Static));
        let ext_so = nearest_pair_dist(&state.planets,
            |p| ext_is_mine(p)  && matches_type(p, PType::Static),
            |p| ext_is_enemy(p) && matches_type(p, PType::Orbit));
        let ext_os = nearest_pair_dist(&state.planets,
            |p| ext_is_mine(p)  && matches_type(p, PType::Orbit),
            |p| ext_is_enemy(p) && matches_type(p, PType::Static));
        let ext_oo = nearest_pair_dist(&state.planets,
            |p| ext_is_mine(p)  && matches_type(p, PType::Orbit),
            |p| ext_is_enemy(p) && matches_type(p, PType::Orbit));

        let extras: [f32; EXTRA_DIM] = [
            tick,
            now_ss, now_so, now_os, now_oo,
            ext_ss, ext_so, ext_os, ext_oo,
            n_static as f32, n_orbit as f32,
            av,
        ];

        out.write_all(&state.step.to_le_bytes())?;
        out.write_all(&me.to_le_bytes())?;
        unsafe {
            let v2 = std::slice::from_raw_parts(feats.as_ptr() as *const u8, feats.len() * 4);
            out.write_all(v2)?;
            let ex = std::slice::from_raw_parts(extras.as_ptr() as *const u8, EXTRA_DIM * 4);
            out.write_all(ex)?;
        }
        out.flush()?;
    }
    Ok(())
}
