//! Smoke tests for the hellburner port.

use crate::engine::{Configuration, EngineState};
use crate::entity_cache::EntityCache;
use crate::hellburner;
use crate::world::WorldState;

fn cache_for(state: &EngineState) -> EntityCache {
    EntityCache::build(
        &state.initial_planets,
        &state.comets,
        &state.comet_planet_ids,
        state.angular_velocity,
        state.step,
    )
}

fn build_world<'a>(state: &EngineState, cache: &'a EntityCache, player: i64) -> WorldState<'a> {
    WorldState::build(
        player,
        state.step,
        state.planets.clone(),
        state.fleets.clone(),
        state.initial_planets.clone(),
        state.comets.clone(),
        state.comet_planet_ids.clone(),
        state.angular_velocity,
        cache,
    )
}

/// Plan must not oversubscribe any source planet or emit launches from planets
/// we don't own.
fn assert_plan_is_legal(world: &WorldState, moves: &[(i64, f64, i64)]) {
    let mut spent: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    for (src_id, _angle, ships) in moves {
        assert!(*ships >= 1, "move with non-positive ships");
        let src = world
            .planets
            .iter()
            .find(|p| p.id == *src_id)
            .expect("move from unknown planet id");
        assert_eq!(src.owner, world.player, "move from non-owned planet");
        *spent.entry(*src_id).or_insert(0) += *ships;
        assert!(
            spent[src_id] <= src.ships,
            "move oversubscribed planet {src_id}: {} > {}",
            spent[src_id],
            src.ships
        );
    }
}

#[test]
fn plan_runs_on_initial_state() {
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let world = build_world(&state, &cache, 0);
    let moves = hellburner::plan(&world);
    assert_plan_is_legal(&world, &moves);
}

#[test]
fn plan_after_early_game_phase() {
    // Step past EARLY_ROUNDS so the main loop (not the DFS) is exercised.
    let mut state = EngineState::new(42, 2, Configuration::default());
    let mut cache = cache_for(&state);
    let noop: Vec<Vec<crate::engine::MoveAction>> = vec![Vec::new(), Vec::new()];
    for _ in 0..5 {
        state.step_with_actions(&noop).unwrap();
    }
    cache.set_current_turn(state.step);
    let world = build_world(&state, &cache, 0);
    let moves = hellburner::plan(&world);
    assert_plan_is_legal(&world, &moves);
}

#[test]
fn search_candidates_includes_greedy_plan() {
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let world = build_world(&state, &cache, 0);
    let greedy = hellburner::plan(&world);
    let candidates = hellburner::search_candidates(&world);
    assert!(!candidates.is_empty());
    assert!(candidates.iter().any(|moves| moves == &greedy));
}

#[test]
fn plan_returns_empty_when_no_enemies() {
    // Build a 4-player state, then give player 0 ownership of every planet so
    // they have no enemy planets. plan() must short-circuit to empty.
    let mut state = EngineState::new(7, 4, Configuration::default());
    for p in state.planets.iter_mut() {
        p.owner = 0;
    }
    let cache = cache_for(&state);
    let world = build_world(&state, &cache, 0);
    let moves = hellburner::plan(&world);
    assert!(moves.is_empty());
}
