//! Quick benchmark of ow2_plan vs the older policy on a single state.
use alphaow_bot::{parse_state, ow2_plan, policy};
use std::time::Instant;
use std::fs;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let trace_path = args.get(1).cloned().unwrap_or_else(|| "/tmp/trace_main_vs_main_s1.json".to_string());
    let step: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(80);
    let raw = fs::read_to_string(&trace_path).expect("read");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let mut s = v["states"][step].clone();
    s["config"] = v["config"].clone();
    let state = parse_state(&s);
    println!("State step {}: planets={} fleets={} comets={}", state.step, state.planets.len(), state.fleets.len(), state.comets.len());

    let n = 1000;
    let mut rng = policy::XorRng(1);
    ow2_plan::plan_profile_reset();
    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n {
        let a = ow2_plan::plan(&state, state.player, false);
        sink += a.len() as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("ow2_plan coop x{}: {:.3}ms each ({:.1}ms total), sink={}", n, dt*1000.0/(n as f64), dt*1000.0, sink);
    if std::env::var("OW_PROFILE_PLAN").is_ok() {
        ow2_plan::plan_profile_report();
        ow2_plan::plan_profile_reset();
    }

    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n {
        let a = ow2_plan::plan(&state, state.player, true);
        sink += a.len() as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("ow2_plan no_coop x{}: {:.3}ms each ({:.1}ms total), sink={}", n, dt*1000.0/(n as f64), dt*1000.0, sink);

    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n {
        let a = policy::sample_joint_action(&state, state.player, &mut rng);
        sink += a.len() as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("sample_joint_action x{}: {:.3}ms each, sink={}", n, dt*1000.0/(n as f64), sink);

    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n {
        let a = policy::sample_joint_action_fast(&state, state.player, &mut rng);
        sink += a.len() as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("sample_joint_action_fast x{}: {:.3}ms each, sink={}", n, dt*1000.0/(n as f64), sink);

    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n {
        let a = policy::rollout_policy_fast_top_n(&state, state.player, 4);
        sink += a.iter().map(|p| p.len() as i64).sum::<i64>();
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("rollout_policy_fast_top_n(4) x{}: {:.3}ms each, sink={}", n, dt*1000.0/(n as f64), sink);

    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n {
        let a = policy::rollout_policy_fast(&state, state.player);
        sink += a.len() as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("rollout_policy_fast x{}: {:.3}ms each, sink={}", n, dt*1000.0/(n as f64), sink);

    // Bench: full tick from this state.
    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n {
        let mut s = state.clone();
        alphaow_bot::sim::tick(&mut s);
        sink += s.fleets.len() as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("clone + tick x{}: {:.3}ms each, sink={}", n, dt*1000.0/(n as f64), sink);

    // Bench: state.clone() alone
    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n {
        let s = state.clone();
        sink += s.fleets.len() as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("state.clone() x{}: {:.3}ms each, sink={}", n, dt*1000.0/(n as f64), sink);

    // Bench: a full mock rollout = 8 × (plan + plan + tick)
    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..(n/10) {
        let mut s = state.clone();
        for _ in 0..8 {
            let a = ow2_plan::plan(&s, s.player, false);
            let b = ow2_plan::plan(&s, 1 - s.player, false);
            alphaow_bot::sim::apply_launches(&mut s, &a);
            alphaow_bot::sim::apply_launches(&mut s, &b);
            alphaow_bot::sim::tick(&mut s);
            sink += s.fleets.len() as i64;
        }
    }
    let rollout_dt = t0.elapsed().as_secs_f64();
    println!("full 8-step rollout x{}: {:.3}ms each, sink={}", n/10, rollout_dt*1000.0/((n/10) as f64), sink);

    let t0 = Instant::now();
    let mut sink2 = 0i64;
    for _ in 0..n {
        let mut s = state.clone();
        for _ in 0..15 { alphaow_bot::sim::tick(&mut s); }
        sink2 += s.fleets.len() as i64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("eval bench (clone + 15 ticks) x{}: {:.3}ms each, sink={}", n, dt*1000.0/(n as f64), sink2);
}
