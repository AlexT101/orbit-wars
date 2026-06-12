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
        .unwrap_or(1000);
    // Only dip into the engine's overage pool when explicitly enabled (set by
    // build_submission.py for final submissions; off in dev so local matches
    // stay fast and predictable). See `duct::best_move`'s overage extension.
    let use_overage = std::env::var("APHRODITE_USE_OVERAGE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
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
        if v.get("__cmd").and_then(Value::as_str) == Some("value") {
            let value = value_net::predict(&state, state.player);
            let response = match value {
                Some(y) => json!({"value": y}),
                None => json!({"value": null}),
            };
            writeln!(out, "{}", response)?;
            out.flush()?;
            continue;
        }
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
        // `remainingOverageTime` is reported in SECONDS by the engine. When it
        // is nearly exhausted, shrink the base search cap to leave margin for
        // wrapper/redirect overhead.
        let remaining_overage_s = v["remainingOverageTime"].as_f64().unwrap_or(0.0);
        let turns_left = (500 - state.step).max(0) as f64;
        let low_overage_threshold_s = 1.5 + 0.015 * turns_left;
        let effective_budget_ms = if remaining_overage_s <= low_overage_threshold_s {
            budget_ms.min(900)
        } else {
            budget_ms
        };
        if effective_budget_ms < budget_ms {
            eprintln!(
                "[aphrodite-panic] step={} player={} remaining_overage={:.3}s threshold={:.3}s budget={}ms->{}ms",
                state.step,
                state.player,
                remaining_overage_s,
                low_overage_threshold_s,
                budget_ms,
                effective_budget_ms
            );
        }
        // Convert to ms for the planner. 0.0 when overage use is disabled.
        let overage_ms = if use_overage {
            remaining_overage_s * 1000.0
        } else {
            0.0
        };
        let actions = duct::best_move(&state, state.player, effective_budget_ms, overage_ms);
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
