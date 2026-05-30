use std::time::Instant;

use super::reference_engine::{PyRandom, RefEngine};
use crate::constants::CENTER;
use crate::engine::{Configuration, MoveAction};
#[cfg(feature = "profile")]
use super::reference_engine::prof;

#[test]
#[ignore] // run with: cargo test --release pure_sim_throughput -- --ignored --nocapture
fn pure_sim_throughput() {
    let noop: Vec<Vec<MoveAction>> = vec![Vec::new(), Vec::new()];
    // Measure step_with_actions only (no PyO3); reset/regeneration excluded.
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let mut steps: u64 = 0;
    let target: u64 = 2_000_000;
    let mut seed = 42u64;
    let mut sim_time = std::time::Duration::ZERO;
    while steps < target {
        let t = Instant::now();
        let done = state.step_with_actions(&noop).unwrap();
        sim_time += t.elapsed();
        steps += 1;
        if done {
            seed += 1;
            state = RefEngine::new(seed, 2, Configuration::default());
        }
    }
    let dt = sim_time.as_secs_f64();
    println!(
        "pure sim 2p: {steps} steps in {dt:.3}s -> {:.0} steps/s ({:.1} ns/step)",
        steps as f64 / dt,
        dt * 1e9 / steps as f64,
    );
}

#[test]
#[ignore] // run with: cargo test --release pure_sim_with_fleets -- --ignored --nocapture
fn pure_sim_with_fleets() {
    // Only the step itself is timed; action construction (fleet_actions)
    // is excluded so we measure step_with_actions, not the test harness.
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let mut steps: u64 = 0;
    let target: u64 = 2_000_000;
    let mut seed = 42u64;
    let mut max_fleets = 0usize;
    let mut fleet_step_sum: u64 = 0;
    let mut sim_time = std::time::Duration::ZERO;
    while steps < target {
        let actions = fleet_actions(&state);
        let t = Instant::now();
        let done = state.step_with_actions(&actions).unwrap();
        sim_time += t.elapsed();
        max_fleets = max_fleets.max(state.fleets.len());
        fleet_step_sum += state.fleets.len() as u64;
        steps += 1;
        if done {
            seed += 1;
            state = RefEngine::new(seed, 2, Configuration::default());
        }
    }
    let dt = sim_time.as_secs_f64();
    println!(
        "pure sim 2p w/fleets: {steps} steps in {dt:.3}s -> {:.0} steps/s ({:.1} ns/step), avg_fleets={:.1}, max_fleets={max_fleets}",
        steps as f64 / dt,
        dt * 1e9 / steps as f64,
        fleet_step_sum as f64 / steps as f64,
    );
}

// Launch from every owned planet so fleets stay in flight, exercising the
// collision loop. Shared by the fleet throughput and profiling benchmarks.
fn fleet_actions(state: &RefEngine) -> Vec<Vec<MoveAction>> {
    let mut actions: Vec<Vec<MoveAction>> = vec![Vec::new(); state.num_players];
    for planet in &state.planets {
        let owner = planet.owner;
        if owner >= 0 && (owner as usize) < state.num_players && planet.ships >= 2 {
            let angle = (planet.y - CENTER).atan2(planet.x - CENTER) + 0.3;
            actions[owner as usize].push(MoveAction {
                from_id: planet.id,
                angle,
                ships: planet.ships / 2,
            });
        }
    }
    actions
}

#[test]
#[cfg(feature = "profile")]
// cargo test --release --features profile profile_sim_sections -- --ignored --nocapture
#[ignore]
fn profile_sim_sections() {
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let mut seed = 42u64;
    // warmup
    for _ in 0..2000 {
        let acts = fleet_actions(&state);
        if state.step_with_actions(&acts).unwrap() {
            seed += 1;
            state = RefEngine::new(seed, 2, Configuration::default());
        }
    }
    prof::reset();
    let target: u64 = 1_000_000;
    let mut steps: u64 = 0;
    while steps < target {
        let acts = fleet_actions(&state);
        let done = state.step_with_actions(&acts).unwrap();
        steps += 1;
        if done {
            seed += 1;
            state = RefEngine::new(seed, 2, Configuration::default());
        }
    }
    let buckets = prof::snapshot();
    let total: f64 = buckets.iter().map(|d| d.as_secs_f64()).sum();
    println!("--- step_with_actions section profile ({steps} steps, 2p w/fleets) ---");
    for (label, d) in prof::LABELS.iter().zip(buckets.iter()) {
        let secs = d.as_secs_f64();
        println!(
            "  {label:<16} {:>7.1} ns/step  {:>5.1}%",
            secs * 1e9 / steps as f64,
            100.0 * secs / total,
        );
    }
    println!(
        "  {:<16} {:>7.1} ns/step  (sum of timed sections)",
        "TOTAL",
        total * 1e9 / steps as f64
    );
}

fn assert_close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-15,
        "expected {a} ~= {b}, diff={}",
        (a - b).abs()
    );
}

#[test]
fn python_random_seed_42_matches() {
    let mut rng = PyRandom::new_from_u64(42);
    assert_close(rng.random(), 0.6394267984578837);
    assert_close(rng.random(), 0.025010755222666936);
    assert_close(rng.random(), 0.27502931836911926);
}

#[test]
fn python_randint_seed_42_matches() {
    let mut rng = PyRandom::new_from_u64(42);
    let values = (0..10).map(|_| rng.randint(1, 5)).collect::<Vec<_>>();
    assert_eq!(values, vec![1, 1, 3, 2, 2, 2, 1, 5, 1, 5]);
}

#[test]
fn python_randbelow_power_of_two_matches() {
    let mut rng = PyRandom::new_from_u64(2024);
    let values = (0..10).map(|_| rng.randbelow(8)).collect::<Vec<_>>();
    assert_eq!(values, vec![7, 2, 4, 3, 6, 4, 3, 7, 5, 6]);
}

#[test]
fn python_string_seed_matches() {
    let mut rng = PyRandom::new_from_py_str_seed("orbit_wars-comet-42-50");
    assert_close(rng.random(), 0.17795165247434586);
    assert_close(rng.random(), 0.34984897997304576);
    assert_close(rng.random(), 0.4498678045067438);
}

#[test]
fn reset_seed_42_matches_reference_snapshot_shape() {
    let state = RefEngine::new(42, 2, Configuration::default());
    assert_eq!(state.step, 0);
    assert_close(state.angular_velocity, 0.04098566996144709);
    assert_eq!(state.planets.len(), 20);
    assert_eq!(state.initial_planets.len(), 20);
    assert_eq!(state.next_fleet_id, 0);
    assert!(state.fleets.is_empty());
    assert!(state.comets.is_empty());
    assert!(state.comet_planet_ids.is_empty());

    let p0 = &state.planets[0];
    assert_eq!(p0.id, 0);
    assert_eq!(p0.owner, 0);
    assert_close(p0.x, 68.17313810856307);
    assert_close(p0.y, 94.88924011533776);
    assert_close(p0.radius, 2.09861228866811);
    assert_eq!(p0.ships, 10);
    assert_eq!(p0.production, 3);

    let init0 = &state.initial_planets[0];
    assert_eq!(init0.owner, -1);
    assert_eq!(init0.ships, 18);
}
