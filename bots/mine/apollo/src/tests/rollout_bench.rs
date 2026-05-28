//! Throughput benchmarks for the bot's planning + rollout hot path.
//!
//! These exercise the code that gets faster from sharing `ArrivalLedger`
//! across players inside `rollout_score` and `opponent_turn0_actions`.

use std::time::Instant;

use super::reference_engine::RefEngine;
use crate::engine::{Configuration, MoveAction};
use crate::cache::EntityCache;
use crate::strategy;
use crate::rollout::{opponent_turn0_actions, pick_plan_by_rollout, rollout_score};
use crate::world::WorldState;

fn cache_for(state: &RefEngine) -> EntityCache {
    EntityCache::build(
        &state.initial_planets,
        &state.comets,
        &state.comet_planet_ids,
        state.angular_velocity,
        state.step,
    )
}

/// Step `state` forward with greedy hellburner play for both players so the
/// benchmark exercises rollout against a non-trivial mid-game position
/// (fleets in flight, mixed ownership). Returns the populated cache too.
fn advance(state: &mut RefEngine, cache: &mut EntityCache, turns: i64) {
    for _ in 0..turns {
        cache.set_current_turn(state.step);
        let snap = state.snapshot();
        let mut actions: Vec<Vec<MoveAction>> = vec![Vec::new(); state.num_players];
        for p in 0..state.num_players {
            let ws = WorldState::from_engine(p as i64, &snap, cache);
            for ma in strategy::plan(&ws) {
                actions[p].push(ma);
            }
        }
        if state.step_with_actions(&actions).is_err() {
            break;
        }
    }
    cache.set_current_turn(state.step);
}

#[test]
#[ignore] // cargo test --release rollout_score_throughput -- --ignored --nocapture
fn rollout_score_throughput() {
    let seeds: &[u64] = &[42, 7, 99, 314, 271];
    let iters_per_seed: u64 = 40;
    let mut total = std::time::Duration::ZERO;
    let mut runs: u64 = 0;

    for &seed in seeds {
        let mut state = RefEngine::new(seed, 2, Configuration::default());
        let mut cache = cache_for(&state);
        advance(&mut state, &mut cache, 30);
        // `state` is unchanged below, so snapshot once into the production
        // `EngineState` the rollout helpers consume.
        let snap = state.snapshot();

        // One candidate + one opponent action set, so we measure rollout_score
        // in isolation (no candidate × variant fan-out).
        let my_player = 0i64;
        let candidate = {
            let ws = WorldState::from_engine(my_player, &snap, &cache);
            strategy::plan(&ws)
        };
        let opp_actions = opponent_turn0_actions(
            &snap,
            my_player,
            strategy::plan,
            &mut cache,
            f64::INFINITY,
            None,
            None,
        );

        crate::aim::counters::reset();
        for _ in 0..iters_per_seed {
            let t = Instant::now();
            let _ = rollout_score(
                &snap,
                my_player,
                &candidate,
                &opp_actions,
                strategy::plan,
                &mut cache,
                f64::INFINITY,
                None,
            );
            total += t.elapsed();
            runs += 1;
        }
        println!(
            "  seed {seed} TIMED-LOOP counters ({iters_per_seed} iters): {}",
            crate::aim::counters::report()
        );
    }

    let dt = total.as_secs_f64();
    println!(
        "rollout_score 2p: {runs} runs in {dt:.3}s -> {:.0} runs/s ({:.2} ms/run)",
        runs as f64 / dt,
        dt * 1000.0 / runs as f64,
    );
}

#[test]
#[ignore] // cargo test --release pick_plan_throughput -- --ignored --nocapture
fn pick_plan_throughput() {
    let seeds: &[u64] = &[42, 7, 99, 314, 271];
    let iters_per_seed: u64 = 10;
    let mut total = std::time::Duration::ZERO;
    let mut runs: u64 = 0;

    for &seed in seeds {
        let mut state = RefEngine::new(seed, 2, Configuration::default());
        let mut cache = cache_for(&state);
        advance(&mut state, &mut cache, 30);
        let snap = state.snapshot();

        let my_player = 0i64;
        let candidates = {
            let ws = WorldState::from_engine(my_player, &snap, &cache);
            strategy::search_candidates(&ws)
        };

        for _ in 0..iters_per_seed {
            let t = Instant::now();
            let _ = pick_plan_by_rollout(
                &snap,
                my_player,
                candidates.clone(),
                strategy::plan,
                strategy::search_candidates,
                &mut cache,
                f64::INFINITY,
                None,
                None,
            );
            total += t.elapsed();
            runs += 1;
        }
    }

    let dt = total.as_secs_f64();
    println!(
        "pick_plan 2p: {runs} runs in {dt:.3}s -> {:.0} runs/s ({:.2} ms/run)",
        runs as f64 / dt,
        dt * 1000.0 / runs as f64,
    );
}

#[test]
#[ignore] // cargo test --release rollout_score_throughput_4p -- --ignored --nocapture
fn rollout_score_throughput_4p() {
    let seeds: &[u64] = &[42, 7, 99, 314, 271];
    let iters_per_seed: u64 = 40;
    let mut total = std::time::Duration::ZERO;
    let mut runs: u64 = 0;

    for &seed in seeds {
        let mut state = RefEngine::new(seed, 4, Configuration::default());
        let mut cache = cache_for(&state);
        advance(&mut state, &mut cache, 30);
        let snap = state.snapshot();

        let my_player = 0i64;
        let candidate = {
            let ws = WorldState::from_engine(my_player, &snap, &cache);
            strategy::plan(&ws)
        };
        let opp_actions = opponent_turn0_actions(
            &snap,
            my_player,
            strategy::plan,
            &mut cache,
            f64::INFINITY,
            None,
            None,
        );

        for _ in 0..iters_per_seed {
            let t = Instant::now();
            let _ = rollout_score(
                &snap,
                my_player,
                &candidate,
                &opp_actions,
                strategy::plan,
                &mut cache,
                f64::INFINITY,
                None,
            );
            total += t.elapsed();
            runs += 1;
        }
    }

    let dt = total.as_secs_f64();
    println!(
        "rollout_score 4p: {runs} runs in {dt:.3}s -> {:.0} runs/s ({:.2} ms/run)",
        runs as f64 / dt,
        dt * 1000.0 / runs as f64,
    );
}
