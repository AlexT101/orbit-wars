//! Sim validator: replays a recorded engine trace through `sim::tick` and
//! reports the first divergence.
//!
//! Usage: `validator path/to/trace.json [--from N] [--to M]`
//!
//! Trace format (produced by `dump_trace.py`):
//!   { "seed": int, "config": {shipSpeed, cometSpeed},
//!     "states": [obs_at_step_0, obs_at_step_1, ...],
//!     "actions": [null, [a0_step1, a1_step1], ...] }
//!
//! For each step K from 1..N:
//!   1. parse states[K-1] -> state
//!   2. apply actions[K] for each agent
//!   3. tick_no_spawn (engine's RNG-spawned comets are injected manually below
//!      if K is a spawn-result step)
//!   4. compare to parse_state(states[K])

use duck_bot::policy::XorRng;
use duck_bot::sim::{tick_no_spawn, Action};
use duck_bot::{parse_state, Fleet, GameState, Planet};
use serde_json::Value;
use std::env;
use std::fs;

const FP_EPS: f64 = 1e-4;
const SPAWN_STEPS: &[i64] = &[50, 150, 250, 350, 450];

fn parse_actions(v: &Value, player: i32) -> Vec<Action> {
    let mut out = Vec::new();
    if let Some(arr) = v.as_array() {
        for m in arr {
            if let Some(arr) = m.as_array() {
                if arr.len() != 3 {
                    continue;
                }
                let from_id = arr[0].as_i64().unwrap_or(-1);
                let angle = arr[1].as_f64().unwrap_or(0.0);
                let ships = arr[2].as_i64().unwrap_or(0);
                out.push((from_id, angle, ships, player));
            }
        }
    }
    out
}

#[derive(Debug)]
struct Diff {
    section: String,
    detail: String,
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

fn diff_planets(got: &[Planet], want: &[Planet], skip_comets: bool) -> Vec<Diff> {
    let mut out = Vec::new();
    let mut got_map: std::collections::HashMap<i64, &Planet> = got.iter().map(|p| (p.id, p)).collect();
    let want_map: std::collections::HashMap<i64, &Planet> = want.iter().map(|p| (p.id, p)).collect();
    // Missing/extra
    for w in want {
        if !got_map.contains_key(&w.id) {
            if skip_comets && w.is_comet {
                continue;
            }
            out.push(Diff {
                section: "planets".into(),
                detail: format!("missing planet id={} owner={} is_comet={}", w.id, w.owner, w.is_comet),
            });
        }
    }
    for g in got {
        if !want_map.contains_key(&g.id) {
            if skip_comets && g.is_comet {
                continue;
            }
            out.push(Diff {
                section: "planets".into(),
                detail: format!("extra planet id={} owner={} is_comet={}", g.id, g.owner, g.is_comet),
            });
        }
    }
    // Field diffs
    for w in want {
        if skip_comets && w.is_comet {
            continue;
        }
        let g = match got_map.remove(&w.id) {
            Some(p) => p,
            None => continue,
        };
        if g.owner != w.owner {
            out.push(Diff { section: "planets".into(), detail: format!("planet {} owner got={} want={}", w.id, g.owner, w.owner) });
        }
        if g.ships != w.ships {
            out.push(Diff { section: "planets".into(), detail: format!("planet {} ships got={} want={}", w.id, g.ships, w.ships) });
        }
        if !approx_eq(g.x, w.x, FP_EPS) || !approx_eq(g.y, w.y, FP_EPS) {
            out.push(Diff { section: "planets".into(), detail: format!("planet {} pos got=({:.6},{:.6}) want=({:.6},{:.6}) (dx={:.6},dy={:.6})", w.id, g.x, g.y, w.x, w.y, g.x-w.x, g.y-w.y) });
        }
    }
    out
}

fn diff_fleets(got: &[Fleet], want: &[Fleet]) -> Vec<Diff> {
    let mut out = Vec::new();
    // Match fleets by (owner, from_planet_id, ships, angle) since IDs may
    // differ between engine (global counter) and sim (re-assigned each turn).
    // Position should match exactly though.
    let mut want_idx: Vec<bool> = vec![false; want.len()];
    let mut got_idx: Vec<bool> = vec![false; got.len()];
    for (gi, g) in got.iter().enumerate() {
        for (wi, w) in want.iter().enumerate() {
            if want_idx[wi] {
                continue;
            }
            if g.owner == w.owner
                && g.from_planet_id == w.from_planet_id
                && g.ships == w.ships
                && approx_eq(g.angle, w.angle, 1e-6)
                && approx_eq(g.x, w.x, FP_EPS)
                && approx_eq(g.y, w.y, FP_EPS)
            {
                want_idx[wi] = true;
                got_idx[gi] = true;
                break;
            }
        }
    }
    for (wi, w) in want.iter().enumerate() {
        if !want_idx[wi] {
            out.push(Diff { section: "fleets".into(), detail: format!("missing fleet owner={} from={} ships={} pos=({:.4},{:.4}) angle={:.4}", w.owner, w.from_planet_id, w.ships, w.x, w.y, w.angle) });
        }
    }
    for (gi, g) in got.iter().enumerate() {
        if !got_idx[gi] {
            out.push(Diff { section: "fleets".into(), detail: format!("extra fleet owner={} from={} ships={} pos=({:.4},{:.4}) angle={:.4}", g.owner, g.from_planet_id, g.ships, g.x, g.y, g.angle) });
        }
    }
    out
}

fn diff_states(got: &GameState, want: &GameState, skip_comets: bool) -> Vec<Diff> {
    let mut out = Vec::new();
    if got.step != want.step {
        out.push(Diff { section: "step".into(), detail: format!("got={} want={}", got.step, want.step) });
    }
    out.extend(diff_planets(&got.planets, &want.planets, skip_comets));
    out.extend(diff_fleets(&got.fleets, &want.fleets));
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: validator <trace.json> [--from N] [--to M] [--quiet]");
        std::process::exit(2);
    }
    let path = &args[1];
    let mut from = 1i64;
    let mut to = i64::MAX;
    let mut quiet = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--from" => {
                from = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--to" => {
                to = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            _ => panic!("unknown arg {}", args[i]),
        }
    }
    let raw = fs::read_to_string(path).expect("read trace");
    let trace: Value = serde_json::from_str(&raw).expect("parse trace");
    let states = trace["states"].as_array().expect("states arr");
    let actions = trace["actions"].as_array().expect("actions arr");
    let config = trace["config"].clone();
    let n = states.len();
    eprintln!("[validator] trace has {} states", n);

    let mut total_diverged = 0u64;
    let mut diverged_steps: Vec<i64> = Vec::new();
    let mut diverge_breakdown: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();

    let mut rng = XorRng(0xdeadbeef);

    for k in 1..n {
        let step_k = k as i64;
        if step_k < from || step_k > to {
            continue;
        }
        // Inject config into prev so parse_state picks up max_speed.
        let mut prev = states[k - 1].clone();
        prev["config"] = config.clone();
        let next_expected = {
            let mut n = states[k].clone();
            n["config"] = config.clone();
            n
        };

        let mut state = parse_state(&prev);
        let want_state = parse_state(&next_expected);
        // Apply actions per agent
        if let Some(arr) = actions[k].as_array() {
            for (i, a) in arr.iter().enumerate() {
                let player = i as i32;
                let acts = parse_actions(a, player);
                duck_bot::sim::apply_launches(&mut state, &acts);
            }
        }
        // Skip comet diff on spawn-result steps; engine spawns via RNG we don't have.
        let skip_comets = SPAWN_STEPS.contains(&step_k);
        tick_no_spawn(&mut state, &mut rng);

        let diffs = diff_states(&state, &want_state, skip_comets);
        if !diffs.is_empty() {
            total_diverged += 1;
            diverged_steps.push(step_k);
            for d in &diffs {
                *diverge_breakdown.entry(d.section.clone()).or_insert(0) += 1;
            }
            if !quiet && total_diverged <= 20 {
                println!("=== DIVERGENCE at step {} -> {} ({} diffs) ===", step_k - 1, step_k, diffs.len());
                for d in diffs.iter().take(10) {
                    println!("  [{}] {}", d.section, d.detail);
                }
            }
        }
    }
    println!();
    println!("Total: {} diverged steps out of {}", total_diverged, (to.min(n as i64 - 1) - from + 1).max(0));
    println!("Breakdown by section: {:?}", diverge_breakdown);
    if total_diverged > 0 {
        println!("First 30 diverged step indices: {:?}", &diverged_steps[..30.min(diverged_steps.len())]);
    }
}
