//! Focused tests for the obnext port and its `WorldState` foundation.
//!
//! Verifies the trickier port-specific bits — merging planned commitments,
//! the WorldState aggregates, projection under hypothetical arrivals, and a
//! plan() smoke test against a deterministic engine state.

use crate::engine::{Configuration, EngineState};
use crate::entity_cache::EntityCache;
use crate::helpers::ArrivalEvent;
use crate::obnext;
use crate::world::{merge_arrivals, WorldState};

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

#[test]
fn merge_arrivals_combines_and_filters_by_cutoff() {
    let base = [ArrivalEvent { turns: 3, owner: 0, ships: 5 }];
    let planned = [ArrivalEvent { turns: 7, owner: 0, ships: 4 }];
    let extra = [
        ArrivalEvent { turns: 5, owner: 1, ships: 8 },
        // Past cutoff — must be dropped.
        ArrivalEvent { turns: 20, owner: 1, ships: 100 },
    ];

    let merged = merge_arrivals(&base, &planned, &extra, 10);

    assert_eq!(merged.len(), 3);
    assert!(merged.iter().any(|ev| ev.turns == 3 && ev.ships == 5));
    assert!(merged.iter().any(|ev| ev.turns == 7 && ev.ships == 4));
    assert!(merged.iter().any(|ev| ev.turns == 5 && ev.ships == 8));
    assert!(merged.iter().all(|ev| ev.turns <= 10));
}

#[test]
fn world_state_aggregates_match_observation() {
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let world = build_world(&state, &cache, 0);

    let total = world.my_planets.len() + world.enemy_planets.len() + world.neutral_planets.len();
    assert_eq!(total, state.planets.len());

    let direct_my_ships: i64 = state
        .planets
        .iter()
        .filter(|p| p.owner == 0)
        .map(|p| p.ships)
        .sum();
    let direct_my_prod: i64 = state
        .planets
        .iter()
        .filter(|p| p.owner == 0)
        .map(|p| p.production)
        .sum();
    assert_eq!(world.my_total, direct_my_ships);
    assert_eq!(world.my_prod, direct_my_prod);

    for sp in &world.static_neutral_planets {
        assert!(world.neutral_planets.iter().any(|n| n.id == sp.id));
    }
}

#[test]
fn projected_state_uses_planned_arrivals() {
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let world = build_world(&state, &cache, 0);

    let target = world
        .neutral_planets
        .first()
        .expect("seed 42 should leave at least one neutral");

    // Baseline: still neutral 5 turns out (no in-flight fleets at step 0).
    let (owner_base, _) = world.projected_state(target.id, 5, &[], &[]);
    assert_eq!(owner_base, -1);

    // Planned commitment of an overwhelming arrival should flip ownership.
    let big_attack = target.ships + 5 * target.production + 50;
    let planned = [ArrivalEvent {
        turns: 2,
        owner: 0,
        ships: big_attack,
    }];
    let (owner_after, _) = world.projected_state(target.id, 5, &planned, &[]);
    assert_eq!(owner_after, 0);
}

#[test]
fn min_ships_to_own_at_drops_with_planned_commitments() {
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let world_state = build_world(&state, &cache, 0);
    // Use a WorldModel to access the obnext memoized min_ships_to_own_at.
    let world = obnext::WorldModel::build(&world_state);

    let target = world
        .neutral_planets
        .first()
        .expect("seed 42 should leave at least one neutral");

    let empty_planned = rustc_hash::FxHashMap::default();
    let need_alone = world.min_ships_to_own_at(target.id, 6, 0, &empty_planned, &[], None);
    assert!(need_alone >= 1);

    let half = (need_alone / 2).max(1);
    let mut planned = rustc_hash::FxHashMap::default();
    planned.insert(
        target.id,
        vec![ArrivalEvent {
            turns: 3,
            owner: 0,
            ships: half,
        }],
    );
    let need_after = world.min_ships_to_own_at(target.id, 6, 0, &planned, &[], None);
    assert!(
        need_after < need_alone,
        "expected planned commitments to reduce need ({need_after} < {need_alone})"
    );
}

#[test]
fn plan_smoke_returns_valid_moves() {
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let world = build_world(&state, &cache, 0);

    let moves = obnext::plan(&world);

    let mut spent: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    for (src_id, _angle, ships) in &moves {
        assert!(*ships >= 1, "move with non-positive ships");
        let src = state
            .planets
            .iter()
            .find(|p| p.id == *src_id)
            .expect("move from unknown planet id");
        assert_eq!(src.owner, 0, "move from non-owned planet");
        *spent.entry(*src_id).or_insert(0) += *ships;
        assert!(
            spent[src_id] <= src.ships,
            "move oversubscribed planet {src_id}"
        );
    }
}

#[test]
fn search_candidates_include_greedy_plan() {
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);
    let world = build_world(&state, &cache, 0);

    let greedy = obnext::plan(&world);
    let candidates = obnext::search_candidates(&world);

    assert!(!candidates.is_empty(), "search should emit at least one candidate");
    assert!(
        candidates.iter().any(|moves| moves == &greedy),
        "search candidates should include the plain greedy obnext plan"
    );
}

#[test]
fn plan_runs_under_advancing_engine() {
    let mut state = EngineState::new(42, 2, Configuration::default());
    let mut cache = cache_for(&state);
    let noop: Vec<Vec<crate::engine::MoveAction>> = vec![Vec::new(), Vec::new()];
    for _ in 0..3 {
        state.step_with_actions(&noop).unwrap();
    }
    cache.set_current_turn(state.step);

    let world = build_world(&state, &cache, 0);
    let _ = obnext::plan(&world).len();
}
