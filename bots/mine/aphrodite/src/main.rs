//! Orbit Wars bot — daemon mode.
//!
//! Reads one JSON observation per line on stdin, writes one JSON moves
//! array per line on stdout. The Python wrapper spawns the binary once and
//! pipes observations each turn.

use aphrodite::value_net::{self, DIST_BLOCK, INPUT_DIM, PER_BLOCK};
use aphrodite::{duct, parse_state, profiling};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let debug = std::env::var("OW_DEBUG").is_ok();
    let budget_ms: u64 = std::env::var("APHRODITE_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let mut dump: Option<File> = std::env::var("APHRODITE_DUMP_FEATURES_PATH")
        .ok()
        .and_then(|p| File::create(p).ok());
    if dump.is_some() {
        eprintln!(
            "[aphrodite] dumping features (input_dim={}) to APHRODITE_DUMP_FEATURES_PATH",
            INPUT_DIM
        );
    }
    let mut err = io::stderr();
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
            writeln!(out, "[]")?;
            out.flush()?;
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                writeln!(out, "[]")?;
                out.flush()?;
                continue;
            }
        };
        let state = parse_state(&v);
        if let Some(f) = dump.as_mut() {
            let feats = value_net::extract_features(&state, state.player);
            let v2 = value_net::summary_features_v2::extract(&state, state.player);
            // Record: step:i64, player:i32,
            //   current[PER_BLOCK]f32, extrap[PER_BLOCK]f32, dist[DIST_BLOCK]f32,
            //   summary_v2[46]f32
            let _ = f.write_all(&state.step.to_le_bytes());
            let _ = f.write_all(&state.player.to_le_bytes());
            let bytes_per_block = PER_BLOCK * 4;
            let bytes_dist = DIST_BLOCK * 4;
            let bytes_v2 = v2.len() * 4;
            unsafe {
                let cur_bytes = std::slice::from_raw_parts(
                    feats.current.as_ptr() as *const u8,
                    bytes_per_block,
                );
                let ext_bytes =
                    std::slice::from_raw_parts(feats.extrap.as_ptr() as *const u8, bytes_per_block);
                let dst_bytes =
                    std::slice::from_raw_parts(feats.dist.as_ptr() as *const u8, bytes_dist);
                let v2_bytes = std::slice::from_raw_parts(v2.as_ptr() as *const u8, bytes_v2);
                let _ = f.write_all(cur_bytes);
                let _ = f.write_all(ext_bytes);
                let _ = f.write_all(dst_bytes);
                let _ = f.write_all(v2_bytes);
            }
        }
        let prof_enabled = std::env::var("OW_PROFILE").is_ok();
        if prof_enabled {
            profiling::reset();
        }
        let __turn_t0 = std::time::Instant::now();
        let actions = duct::best_move(&state, state.player, budget_ms);
        // Final no-loss reroute pass on the chosen plan — runs after the planner
        // has fully committed (apollo's `redirect_moves` tail, ported via the
        // bridge since `Action` tuples drop the target the pass needs).
        let actions = aphrodite::apollo_bridge::redirect_actions(&state, state.player, actions);
        if prof_enabled {
            profiling::TURN_TOTAL_NS.fetch_add(
                __turn_t0.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            profiling::dump(state.step, state.player);
        }
        let mv: Vec<(i64, f64, i64)> = actions
            .into_iter()
            .filter(|a| a.3 == state.player)
            .map(|a| (a.0, a.1, a.2))
            .collect();
        if debug {
            let me = state.player;
            let my_count = state.planets.iter().filter(|p| p.owner == me).count();
            let my_ships: i64 = state
                .planets
                .iter()
                .filter(|p| p.owner == me)
                .map(|p| p.ships)
                .sum();
            let neutral = state.planets.iter().filter(|p| p.owner == -1).count();
            let enemy = state
                .planets
                .iter()
                .filter(|p| p.owner != me && p.owner != -1)
                .count();
            writeln!(
                err,
                "[aphrodite p{}] step={} planets={}(m)/{}(n)/{}(e) ships={} fleets={} moves={}",
                me,
                state.step,
                my_count,
                neutral,
                enemy,
                my_ships,
                state.fleets.len(),
                mv.len()
            )
            .ok();
        }
        let arr: Vec<Value> = mv
            .into_iter()
            .map(|(fid, ang, ships)| json!([fid, ang, ships]))
            .collect();
        writeln!(out, "{}", Value::Array(arr))?;
        out.flush()?;
    }
    Ok(())
}
