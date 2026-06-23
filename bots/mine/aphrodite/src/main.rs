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
use std::time::{Duration, Instant};

const DEFAULT_POST_SEARCH_MARGIN_MS: u64 = 80;
const DEFAULT_OVERAGE_HARD_CAP_MS: u64 = 2000;

fn elapsed_ms(t0: Instant) -> u64 {
    t0.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn min_instant(a: Instant, b: Instant) -> Instant {
    if a <= b {
        a
    } else {
        b
    }
}

fn checked_deadline_minus(deadline: Instant, margin_ms: u64, floor: Instant) -> Instant {
    deadline
        .checked_sub(Duration::from_millis(margin_ms))
        .filter(|t| *t >= floor)
        .unwrap_or(floor)
}

fn dominant_enemy_id(state: &aphrodite::GameState, me: i32) -> Option<i32> {
    let mut best: Option<(i32, i64)> = None;
    let mut visit_player = |p: i32| {
        if p == -1 || p == me {
            return;
        }
        let score = aphrodite::sim::player_score(state, p);
        if best.as_ref().map(|(_, s)| score > *s).unwrap_or(true) {
            best = Some((p, score));
        }
    };
    for p in &state.planets {
        visit_player(p.owner);
    }
    for f in &state.fleets {
        visit_player(f.owner);
    }
    best.map(|(p, _)| p)
}

fn parse_il_candidates(v: &Value, key: &str, owner: i32) -> Vec<aphrodite::sim::Action> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let arr = m.as_array()?;
                    let from_id = aphrodite::as_i64(arr.get(0)?)?;
                    let angle = aphrodite::as_f64(arr.get(1)?);
                    let ships = aphrodite::as_i64(arr.get(2)?)?;
                    if ships <= 0 {
                        return None;
                    }
                    Some((from_id, angle, ships, owner))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_f64_array(v: &Value, key: &str) -> Vec<f64> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn parse_i64_array(v: &Value, key: &str) -> Vec<i64> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default()
}

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
        let rust_turn_t0 = Instant::now();
        let parse_t0 = Instant::now();
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                writeln!(out, "[]")?;
                out.flush()?;
                continue;
            }
        };
        let state = parse_state(&v);
        let parse_ms = elapsed_ms(parse_t0);
        // REQUIRED to make sure we set 4p mode correctly before any apollo code runs
        let alive = aphrodite::sim::alive_players(&state);
        aphrodite::apollo::constants::set_mode_for_alive(alive);
        if v.get("__cmd").and_then(Value::as_str) == Some("value") {
            let cache = aphrodite::apollo_bridge::rollout_cache(&state);
            let value = value_net::predict_with_cache(&state, state.player, &cache, None, alive);
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
        // `remainingOverageTime` is reported in SECONDS by the engine. The
        // caller may pass an explicit hard budget; otherwise the daemon derives
        // one from the base budget plus the normal per-turn overage cap.
        let remaining_overage_s = v["remainingOverageTime"].as_f64().unwrap_or(0.0);
        let return_timing = v
            .get("return_timing")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // Optional per-turn budget override. `base_budget_ms` is the preferred
        // field; `budget_ms` remains a simple alias for older wrappers.
        let base_budget_raw_ms = v
            .get("base_budget_ms")
            .or_else(|| v.get("budget_ms"))
            .and_then(Value::as_u64)
            .unwrap_or(budget_ms);
        let effective_base_budget_ms = base_budget_raw_ms;
        let post_search_margin_ms = v
            .get("post_search_margin_ms")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_POST_SEARCH_MARGIN_MS);
        let default_hard_budget_ms = if use_overage {
            effective_base_budget_ms.saturating_add(DEFAULT_OVERAGE_HARD_CAP_MS)
        } else {
            effective_base_budget_ms
        };
        let hard_budget_ms = v
            .get("hard_budget_ms")
            .and_then(Value::as_u64)
            .unwrap_or(default_hard_budget_ms)
            .max(effective_base_budget_ms);
        let hard_deadline = rust_turn_t0 + Duration::from_millis(hard_budget_ms);
        let search_hard_deadline =
            checked_deadline_minus(hard_deadline, post_search_margin_ms, rust_turn_t0);
        let base_deadline = min_instant(
            rust_turn_t0 + Duration::from_millis(effective_base_budget_ms),
            search_hard_deadline,
        );
        // Convert to ms for the planner. 0.0 when overage use is disabled.
        let overage_ms = if use_overage {
            remaining_overage_s * 1000.0
        } else {
            0.0
        };
        // Optional externally proposed root candidates (the chaos wrapper's IL
        // policy moves), each `[from_id, angle, ships]`. Absent in aphrodite's
        // own wrapper payloads, in which case behavior is unchanged.
        let il_candidates = parse_il_candidates(&v, "il_candidates", state.player);
        let il_candidate_probs = parse_f64_array(&v, "il_candidate_probs");
        let il_candidate_indices = parse_i64_array(&v, "il_candidate_indices");
        let opp_player = dominant_enemy_id(&state, state.player).unwrap_or(1 - state.player);
        let opp_il_candidates = parse_il_candidates(&v, "opp_il_candidates", opp_player);
        let opp_il_candidate_probs = parse_f64_array(&v, "opp_il_candidate_probs");
        let opp_il_candidate_indices = parse_i64_array(&v, "opp_il_candidate_indices");
        let search = duct::best_move(
            &state,
            state.player,
            effective_base_budget_ms,
            base_deadline,
            search_hard_deadline,
            &il_candidates,
            &il_candidate_probs,
            &il_candidate_indices,
            &opp_il_candidates,
            &opp_il_candidate_probs,
            &opp_il_candidate_indices,
        );
        // Final no-loss reroute pass on the chosen plan — runs after the planner
        // has fully committed (apollo's `redirect_moves` tail, ported via the
        // bridge since `Action` tuples drop the target the pass needs).
        let redirect_t0 = Instant::now();
        let actions =
            aphrodite::apollo_bridge::redirect_actions(&state, state.player, search.actions);
        let redirect_ms = elapsed_ms(redirect_t0);
        if prof_enabled {
            profiling::TURN_TOTAL_NS.fetch_add(
                rust_turn_t0.elapsed().as_nanos() as u64,
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
        if return_timing {
            let timing = json!({
                "rust_total_ms": elapsed_ms(rust_turn_t0),
                "parse_ms": parse_ms,
                "pre_search_ms": search.timing.pre_search_ms,
                "search_ms": search.timing.search_ms,
                "overage_search_ms": search.timing.overage_search_ms,
                "redirect_ms": redirect_ms,
                "iters": search.timing.iters,
                "overage_ms": search.timing.overage_used_ms,
                "root_visits": search.timing.root_visits,
                "base_budget_ms": effective_base_budget_ms,
                "hard_budget_ms": hard_budget_ms,
                "post_search_margin_ms": post_search_margin_ms,
                "remaining_overage_ms": overage_ms,
            });
            writeln!(out, "{}", json!({"moves": arr, "timing": timing}))?;
        } else {
            writeln!(out, "{}", Value::Array(arr))?;
        }
        out.flush()?;
    }
    Ok(())
}
