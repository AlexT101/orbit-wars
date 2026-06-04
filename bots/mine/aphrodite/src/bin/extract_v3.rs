//! Like `extract_v2` but emits 5 additional features (user request 2026-05-29):
//!     [tick, nearest_enemy_planet_now, nearest_enemy_planet_ext,
//!      n_total_static, n_total_orbit]
//!
//! Record format (binary, per observation):
//!     step:i64, player:i32, summary_v2:[f32; 46], extras:[f32; 5]
//!   → 8 + 4 + 184 + 20 = 216 bytes per record.
//!
//! All extras are leakage-free (no game-end info). `nearest_enemy_planet_ext`
//! uses the Rust `value_net::extrapolate_fleets` so it's as fast as the
//! existing summary_v2 extrapolation.

use aphrodite::{parse_state, value_net};
use serde_json::Value;
use std::io::{self, BufRead, Write};

const EXTRA_DIM: usize = 5;

fn nearest_enemy_dist(
    planets: &[aphrodite::Planet],
    is_mine: impl Fn(&aphrodite::Planet) -> bool,
    is_enemy: impl Fn(&aphrodite::Planet) -> bool,
) -> f32 {
    let mut best = f32::INFINITY;
    for m in planets.iter().filter(|p| is_mine(p)) {
        for e in planets.iter().filter(|p| is_enemy(p)) {
            let dx = (m.x - e.x) as f32;
            let dy = (m.y - e.y) as f32;
            let d2 = dx * dx + dy * dy;
            if d2 < best {
                best = d2;
            }
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
        if n == 0 {
            break;
        }
        let line = buf.trim_end();
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let state = parse_state(&v);
        let me = state.player;
        let feats = value_net::summary_features_v2::extract(&state, me);

        // ---- extras ----
        // [0] tick (= step)
        let tick = state.step as f32;

        // [1] nearest_enemy_now: min pair dist (mine vs enemy at current state)
        let near_now = nearest_enemy_dist(
            &state.planets,
            |p| p.owner == me,
            |p| p.owner != me && p.owner != -1,
        );

        // [2] nearest_enemy_ext: same but using extrapolated ownership
        let ext_map = value_net::extrapolate_fleets(&state);
        let near_ext = {
            let mut best = f32::INFINITY;
            for m in &state.planets {
                let mo = ext_map.get(&m.id).map(|x| x.0).unwrap_or(m.owner);
                if mo != me {
                    continue;
                }
                for e in &state.planets {
                    let eo = ext_map.get(&e.id).map(|x| x.0).unwrap_or(e.owner);
                    if eo == me || eo == -1 {
                        continue;
                    }
                    let dx = (m.x - e.x) as f32;
                    let dy = (m.y - e.y) as f32;
                    let d2 = dx * dx + dy * dy;
                    if d2 < best {
                        best = d2;
                    }
                }
            }
            if best.is_finite() { best.sqrt() } else { 0.0 }
        };

        // [3,4] n_total_static, n_total_orbit (constants per game — but
        // recomputed per obs for simplicity; constant within a game).
        let mut n_static = 0u32;
        let mut n_orbit = 0u32;
        for p in &state.planets {
            if p.is_comet {
                // skip — comets are their own thing
            } else if p.is_orbiting {
                n_orbit += 1;
            } else {
                n_static += 1;
            }
        }

        let extras: [f32; EXTRA_DIM] = [tick, near_now, near_ext, n_static as f32, n_orbit as f32];

        out.write_all(&state.step.to_le_bytes())?;
        out.write_all(&me.to_le_bytes())?;
        unsafe {
            let v2_bytes = std::slice::from_raw_parts(feats.as_ptr() as *const u8, feats.len() * 4);
            out.write_all(v2_bytes)?;
            let ex_bytes = std::slice::from_raw_parts(extras.as_ptr() as *const u8, EXTRA_DIM * 4);
            out.write_all(ex_bytes)?;
        }
        out.flush()?;
    }
    Ok(())
}
