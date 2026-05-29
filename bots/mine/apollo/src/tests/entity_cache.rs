use super::reference_engine::RefEngine;
use crate::constants::EPISODE_STEPS;
use crate::engine::Configuration;
use crate::entity_cache::{EntityCache, EntityKind};

fn cache_for(state: &RefEngine) -> EntityCache {
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
    let state = RefEngine::new(42, 2, Configuration::default());
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
    let mut state = RefEngine::new(42, 2, Configuration::default());
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

/// The O(1) `remaining_life` (precomputed `off_board_turn`) must match the
/// original linear scan over the `positions` table, for a real comet, at every
/// `current_turn`.
#[test]
fn comet_remaining_life_matches_linear_scan() {
    // The pre-optimization implementation, scanning the public positions table.
    fn reference(positions: &[Option<[f64; 2]>], current: i64) -> i64 {
        let start = current.max(0) as usize;
        for (t, p) in positions.iter().enumerate().skip(start) {
            if p.is_none() {
                return t as i64 - current;
            }
        }
        (EPISODE_STEPS - current).max(0)
    }

    // Advance until comets spawn (they appear on fixed game steps).
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let noop: Vec<Vec<crate::engine::MoveAction>> = vec![Vec::new(), Vec::new()];
    let mut guard = 0;
    while state.comet_planet_ids.is_empty() && guard < EPISODE_STEPS {
        state.step_with_actions(&noop).unwrap();
        guard += 1;
    }
    assert!(!state.comet_planet_ids.is_empty(), "expected comets to spawn");

    let mut cache = cache_for(&state);
    let comet_ids: Vec<i64> = cache
        .entities
        .values()
        .filter(|e| e.is_comet())
        .map(|e| e.id)
        .collect();
    assert!(!comet_ids.is_empty(), "cache should hold the spawned comets");

    for current in 0..EPISODE_STEPS {
        cache.set_current_turn(current);
        for &id in &comet_ids {
            let want = reference(&cache.entities[&id].positions, current);
            let got = cache.remaining_life(id);
            assert_eq!(got, want, "comet {id} at current_turn {current}");
        }
    }
}
