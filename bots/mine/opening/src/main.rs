//! Opening bot — predict the top ⌈N/4⌉ planets with v23, then race to
//! capture them via permutation search over orderings. After all sends are
//! issued, idles (this is a clean experiment first; chaining to apollo
//! comes later).

mod features;
mod planner;

use alphaow_bot::{parse_state, GameState};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, Write};

fn read_model() -> alphaow_bot::xgb::XgbModel {
    let path = std::env::var("OPENING_MODEL_PATH")
        .unwrap_or_else(|_| "weights/first_owned_v23.json".to_string());
    let bytes = fs::read(&path)
        .unwrap_or_else(|e| panic!("opening: cannot read model at {}: {}", path, e));
    alphaow_bot::xgb::load(&bytes)
        .unwrap_or_else(|| panic!("opening: failed to parse XGB model at {}", path))
}

/// Pick top ⌈N/4⌉ non-home planets by predicted probability under `model`.
fn select_targets(
    state: &GameState,
    model: &alphaow_bot::xgb::XgbModel,
    my_player: i32,
) -> Vec<i64> {
    let n = state.planets.len();
    let k = ((n as f64) / 4.0).ceil() as usize;
    let mut scored: Vec<(f64, i64)> = state.planets.iter()
        .filter(|p| p.owner != my_player) // exclude my home
        .map(|p| {
            let feats = features::extract(state, p, my_player);
            (model.predict_logit(&feats) as f64, p.id)
        })
        .collect();
    // Sort by logit descending (proxy for probability — sigmoid is monotone).
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut err = io::stderr();
    let debug = std::env::var("OW_DEBUG").is_ok();

    let model = read_model();
    if debug {
        writeln!(err, "[opening] loaded model with {} features", model.num_feature).ok();
    }

    // Plan is computed once at turn 0 and cached. It's only rebuilt when
    // one of the locked targets changes ownership (captured by us or the
    // enemy), since that invalidates the source/timing assumptions baked
    // into the schedule.
    let mut targets_locked: Option<Vec<i64>> = None;
    let mut cached_schedule: Vec<planner::Send> = Vec::new();
    let mut target_owners_at_plan: HashMap<i64, i32> = HashMap::new();
    // Each emitted dispatch is keyed by (src, tgt, send_turn) so a replan
    // that re-emits an identical send isn't double-issued.
    let mut emitted: HashSet<(i64, i64, i64)> = HashSet::new();

    let mut buf = String::new();
    let mut handle = stdin.lock();
    loop {
        buf.clear();
        let n = handle.read_line(&mut buf)?;
        if n == 0 { break; }
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
        let me = state.player;

        // Lock targets on turn 0.
        if targets_locked.is_none() {
            let tgts = select_targets(&state, &model, me);
            if debug {
                writeln!(err, "[opening p{}] step={} targets={:?} (k={})",
                    me, state.step, tgts, tgts.len()).ok();
            }
            targets_locked = Some(tgts);
        }
        let targets = targets_locked.as_ref().unwrap();

        let current_target_owners: HashMap<i64, i32> = targets.iter()
            .filter_map(|&tid| {
                state.planets.iter().find(|p| p.id == tid).map(|p| (tid, p.owner))
            })
            .collect();

        // Decide if we need to (re)plan: first turn (no cache yet) or any
        // target's owner has changed since the cache was built.
        let needs_plan = cached_schedule.is_empty() && target_owners_at_plan.is_empty()
            || current_target_owners != target_owners_at_plan;
        if needs_plan {
            let pending: Vec<i64> = targets.iter()
                .copied()
                .filter(|tid| current_target_owners.get(tid) != Some(&me))
                .collect();
            if !pending.is_empty() {
                match planner::plan(&state, &pending, me) {
                    Some(sch) => {
                        cached_schedule = sch;
                        if debug {
                            writeln!(err, "[opening p{}] step={} REPLAN ({} sends, owners changed)",
                                me, state.step, cached_schedule.len()).ok();
                        }
                    }
                    None => {
                        cached_schedule.clear();
                        if debug {
                            writeln!(err, "[opening p{}] step={} planner returned None for targets={:?}",
                                me, state.step, pending).ok();
                        }
                    }
                }
            } else {
                cached_schedule.clear();
            }
            target_owners_at_plan = current_target_owners;
        }

        // Emit any scheduled sends due this turn.
        let mut moves: Vec<Value> = Vec::new();
        for s in &cached_schedule {
            let key = (s.src_pid, s.tgt_pid, s.send_global_turn);
            if emitted.contains(&key) { continue; }
            if s.send_global_turn == state.step {
                let ok = state.planets.iter().any(|p|
                    p.id == s.src_pid && p.owner == me && p.ships >= s.ships
                );
                if !ok { continue; }
                moves.push(json!([s.src_pid, s.angle, s.ships]));
                emitted.insert(key);
                if debug {
                    writeln!(err, "[opening p{}] step={} SEND {}->{} ships={} angle={:.3} (arrive≈{})",
                        me, state.step, s.src_pid, s.tgt_pid, s.ships, s.angle, s.arrival_global_turn).ok();
                }
            }
        }
        writeln!(out, "{}", Value::Array(moves))?;
        out.flush()?;
    }
    Ok(())
}
