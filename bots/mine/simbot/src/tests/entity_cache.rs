use crate::engine::{Configuration, EngineState};
use crate::entity_cache::{EntityCache, EntityKind};

fn cache_for(state: &EngineState) -> EntityCache {
    EntityCache::build(
        &state.initial_planets,
        &state.comets,
        &state.comet_planet_ids,
        state.angular_velocity,
        state.step,
    )
}

#[test]
fn static_planet_positions_dont_change() {
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let static_ent = cache
        .entities
        .values()
        .find(|e| e.is_static())
        .expect("seed 42 should have at least one static planet");
    let first = static_ent.positions[0];
    for p in &static_ent.positions {
        assert_eq!(*p, first);
    }
}

#[test]
fn orbiting_planet_matches_engine_after_n_turns() {
    let mut state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let orb_id = cache
        .entities
        .values()
        .find(|e| matches!(e.kind, EntityKind::OrbitingPlanet))
        .map(|e| e.id)
        .expect("seed 42 should have at least one orbiting planet");

    let noop: Vec<Vec<crate::engine::MoveAction>> =
        vec![Vec::new(), Vec::new()];
    for _ in 0..25 {
        state.step_with_actions(&noop).unwrap();
    }
    let engine_planet = state
        .planets
        .iter()
        .find(|p| p.id == orb_id)
        .expect("planet should still exist");

    let predicted = cache
        .position(orb_id, 25)
        .expect("position should be in range");
    assert!(
        (predicted[0] - engine_planet.x).abs() < 1e-9,
        "x: predicted={} engine={}",
        predicted[0],
        engine_planet.x
    );
    assert!(
        (predicted[1] - engine_planet.y).abs() < 1e-9,
        "y: predicted={} engine={}",
        predicted[1],
        engine_planet.y
    );
}
