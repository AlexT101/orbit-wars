//! Early-game expansion pre-pass tests.

use crate::cache::EntityCache;
use crate::constants::EARLY_GAME_END;
use crate::early_game::plan_opening;
use crate::engine::{CometGroup, Planet};
use crate::strategy::HellburnerModel;
use crate::world::WorldState;

fn planet(id: i64, owner: i64, x: f64, y: f64, ships: i64, production: i64) -> Planet {
    Planet {
        id,
        owner,
        x,
        y,
        radius: 2.0,
        ships,
        production,
    }
}

fn cache_for(planets: &[Planet], step: i64) -> EntityCache {
    let comets: Vec<CometGroup> = Vec::new();
    let comet_ids: Vec<i64> = Vec::new();
    EntityCache::build(planets, &comets, &comet_ids, 0.03, step)
}

fn world_for<'a>(planets: &[Planet], cache: &'a EntityCache, step: i64) -> WorldState<'a> {
    WorldState::build(
        0,
        step,
        planets.to_vec(),
        Vec::new(),
        planets.to_vec(),
        Vec::new(),
        Vec::new(),
        0.03,
        cache,
    )
}

/// Captured target ids of the opening plan (any offset).
fn opening_targets(world: &WorldState) -> Vec<i64> {
    let model = HellburnerModel::build(world);
    plan_opening(&model).iter().map(|e| e.target).collect()
}

/// All planets sit in the static zone (orbital radius + planet radius ≥ 50),
/// so positions are turn-constant and travel times are easy to reason about.
///
/// Home (12 ships, prod 5) can afford exactly one opening at turn 0: the
/// greedy bait D alone (prod 3, garrison 11 → costs 12), or the chain — relay
/// B (prod 1, garrison 5) funded with C's down payment on board, then B→C
/// (prod 5, garrison 5). A greedy per-target pass takes D; the set-search
/// must find the B→C chain.
#[test]
fn early_game_finds_chain_captures() {
    let planets = vec![
        planet(0, 0, 95.0, 92.0, 12, 5),  // home
        planet(1, -1, 95.0, 82.0, 5, 1),  // B — cheap relay toward C
        planet(2, -1, 95.0, 72.0, 5, 5),  // C — the prize, far from home
        planet(3, -1, 89.0, 95.0, 11, 3), // D — greedy bait next to home
        planet(4, 1, 5.0, 8.0, 10, 1),    // far-away enemy so plan() engages
    ];
    let cache = cache_for(&planets, 0);
    let world = world_for(&planets, &cache, 0);

    let targets = opening_targets(&world);
    assert!(
        targets.contains(&1),
        "plan should capture the relay B, got {targets:?}"
    );
    assert!(
        targets.contains(&2),
        "plan should capture the chain prize C, got {targets:?}"
    );

    // The emitted turn-0 moves must be legal: launched from our planet, with
    // positive ship counts that don't oversubscribe the source.
    let moves = crate::strategy::plan(&world);
    assert!(!moves.is_empty(), "chain plan should launch this turn");
    let mut spent = 0;
    for m in &moves {
        assert_eq!(m.from_id, 0, "only the home planet can launch at turn 0");
        assert!(m.ships >= 1);
        spent += m.ships;
    }
    assert!(spent <= 12, "moves oversubscribe home: {spent} > 12");
}

/// Rollout-internal worlds must bypass the opening pre-pass entirely — the
/// rollout's reply policy has to stay cheap.
#[test]
fn early_game_disabled_inside_rollout() {
    let planets = vec![
        planet(0, 0, 95.0, 92.0, 12, 5),
        planet(1, -1, 95.0, 82.0, 5, 1),
        planet(4, 1, 5.0, 8.0, 10, 1),
    ];
    let cache = cache_for(&planets, 0);
    let mut world = world_for(&planets, &cache, 0);
    world.rollout_internal = true;
    let model = HellburnerModel::build(&world);
    assert!(plan_opening(&model).is_empty());
}

/// Past the phase boundary the pre-pass must stand down and leave the
/// pipeline to the greedy planner alone.
#[test]
fn early_game_inactive_after_phase() {
    let planets = vec![
        planet(0, 0, 95.0, 92.0, 50, 5),
        planet(1, -1, 95.0, 82.0, 5, 1),
        planet(4, 1, 5.0, 8.0, 10, 1),
    ];
    let cache = cache_for(&planets, EARLY_GAME_END);
    let world = world_for(&planets, &cache, EARLY_GAME_END);
    let model = HellburnerModel::build(&world);
    assert!(plan_opening(&model).is_empty());
}

/// A capture whose ships-at-horizon value is negative (the garrison costs
/// more than the planet can produce back inside the window) must lose to the
/// empty plan. The greedy scorer rejects the same capture as score-negative,
/// so agreeing here keeps the phase boundary cliff-free.
#[test]
fn early_game_rejects_negative_value_captures() {
    let planets = vec![
        planet(0, 0, 95.0, 92.0, 200, 5), // home can easily afford the capture
        planet(1, -1, 95.0, 82.0, 90, 1), // value ≤ 1·(window−1) − 90 < 0
        planet(4, 1, 5.0, 8.0, 10, 1),
    ];
    let cache = cache_for(&planets, 0);
    let world = world_for(&planets, &cache, 0);
    let targets = opening_targets(&world);
    assert!(
        targets.is_empty(),
        "negative-value capture must be rejected, got {targets:?}"
    );
}

/// Future-offset opening events are reservations: home (10 ships, prod 1) can
/// only afford N (garrison 12 → 13 ships) after saving up production, so the
/// opening commits a deferred launch. The greedy pipeline runs on top but
/// must not spend the reserved ships — not even on the tempting enemy planet
/// next door.
#[test]
fn early_game_reservations_block_greedy() {
    let planets = vec![
        planet(0, 0, 95.0, 92.0, 10, 1),  // home
        planet(1, -1, 95.0, 82.0, 12, 5), // N — affordable only at offset ≥ 3
        planet(4, 1, 89.0, 95.0, 5, 2),   // enemy in range of home
    ];
    let cache = cache_for(&planets, 0);
    let world = world_for(&planets, &cache, 0);

    let model = HellburnerModel::build(&world);
    let opening = plan_opening(&model);
    assert_eq!(
        opening.len(),
        1,
        "expected a single deferred capture, got {opening:?}"
    );
    assert_eq!(opening[0].target, 1);
    assert!(
        opening[0].offset > 0,
        "capture should wait for production, got {opening:?}"
    );

    let moves = crate::strategy::plan(&world);
    assert!(
        moves.is_empty(),
        "greedy spent ships reserved for the deferred capture: {moves:?}"
    );
}

/// Opening-search cost/quality profile across self-played opening turns on
/// real maps. Informs the EARLY_GAME_* constants: node-budget exhaustion
/// rate, wall time per turn, and whether late opening turns still produce
/// plans. Run with:
/// `cargo test --release early_game_search_profile -- --ignored --nocapture`
#[test]
#[ignore]
fn early_game_search_profile() {
    use std::time::Instant;

    use super::reference_engine::RefEngine;
    use crate::constants::EARLY_GAME_NODE_BUDGET;
    use crate::engine::{Configuration, MoveAction};

    let seeds: &[u64] = &[42, 7, 99, 314, 271];
    for &num_players in &[2usize, 4] {
        // Per turn: runs, plans found, total/max nodes, exhausted count,
        // total/max ms, total events.
        let mut per_turn = vec![(0u64, 0u64, 0u64, 0u64, 0u64, 0.0f64, 0.0f64, 0u64); EARLY_GAME_END as usize];
        for &seed in seeds {
            let mut state = RefEngine::new(seed, num_players, Configuration::default());
            let mut cache = EntityCache::build(
                &state.initial_planets,
                &state.comets,
                &state.comet_planet_ids,
                state.angular_velocity,
                state.step,
            );
            for turn in 0..EARLY_GAME_END {
                cache.set_current_turn(state.step);
                let snap = state.snapshot();
                for p in 0..num_players {
                    let ws = WorldState::from_engine(p as i64, &snap, &cache);
                    let model = HellburnerModel::build(&ws);
                    let t = Instant::now();
                    let stats = crate::early_game::opening_search_stats(&model);
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    let e = &mut per_turn[turn as usize];
                    e.0 += 1;
                    e.5 += ms;
                    e.6 = e.6.max(ms);
                    if let Some((nodes, events, _value)) = stats {
                        e.2 += nodes;
                        e.3 = e.3.max(nodes);
                        if nodes >= EARLY_GAME_NODE_BUDGET {
                            e.4 += 1;
                        }
                        if events > 0 {
                            e.1 += 1;
                            e.7 += events as u64;
                        }
                    }
                }
                // Self-play one engine step (every player runs the full plan).
                let mut actions: Vec<Vec<MoveAction>> = vec![Vec::new(); num_players];
                for (p, slot) in actions.iter_mut().enumerate() {
                    let ws = WorldState::from_engine(p as i64, &snap, &cache);
                    *slot = crate::strategy::plan(&ws);
                }
                if state.step_with_actions(&actions).is_err() {
                    break;
                }
            }
        }
        println!("--- {num_players}p, {} seeds ---", seeds.len());
        for (turn, e) in per_turn.iter().enumerate() {
            if e.0 == 0 {
                continue;
            }
            println!(
                "turn {turn:>2}: runs {:>2}  plans {:>2}  nodes avg {:>7.0} max {:>7}  exhausted {:>2}  ms avg {:>7.2} max {:>8.2}  events avg {:.1}",
                e.0,
                e.1,
                e.2 as f64 / e.0 as f64,
                e.3,
                e.4,
                e.5 / e.0 as f64,
                e.6,
                e.7 as f64 / e.1.max(1) as f64,
            );
        }
    }
}

/// The pre-pass composes with the greedy pipeline: the opening capture and a
/// greedy combat strike both happen on the same turn, funded from disjoint
/// ship pools (home ferries the opening; F attacks the enemy).
#[test]
fn early_game_composes_with_greedy_combat() {
    let planets = vec![
        planet(0, 0, 95.0, 92.0, 30, 5), // home funds the opening capture
        planet(1, -1, 95.0, 82.0, 5, 5), // B — the opening target, nearest home
        planet(5, 0, 85.0, 95.0, 20, 2), // F — second owned planet, free ships
        planet(4, 1, 80.0, 95.0, 3, 2),  // E — weak enemy next to F
    ];
    let cache = cache_for(&planets, 0);
    let world = world_for(&planets, &cache, 0);

    let moves = crate::strategy::plan(&world);
    assert!(
        moves.iter().any(|m| m.from_id == 0 && m.target == 1),
        "opening capture of B missing: {moves:?}"
    );
    assert!(
        moves.iter().any(|m| m.from_id == 5 && m.target == 4),
        "greedy attack on E missing: {moves:?}"
    );
    for &(src, cap) in &[(0i64, 30i64), (5, 20)] {
        let spent: i64 = moves
            .iter()
            .filter(|m| m.from_id == src)
            .map(|m| m.ships)
            .sum();
        assert!(spent <= cap, "source {src} oversubscribed: {spent} > {cap}");
    }
}
