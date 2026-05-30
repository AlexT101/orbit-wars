use super::reference_engine::RefEngine;
use crate::blockers::aim_with_prediction;
use crate::constants::EPISODE_STEPS;
use crate::engine::Configuration;
use crate::entity_cache::{rot_sibling, AimCacheVerdict, EntityCache, EntityKind};

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

/// Wrap an angle difference into `(-π, π]` for tolerant bearing comparison.
fn wrap_pi(a: f64) -> f64 {
    use std::f64::consts::PI;
    a - 2.0 * PI * ((a + PI) * (1.0 / (2.0 * PI))).floor()
}

/// Storing an aim result must populate its three quartet siblings with the
/// rotated solution, and that rotated solution must equal an *independent*
/// direct solve of the sibling problem. This is the core guard against a
/// tag/rotation-direction sign error: the symmetry is real only if the
/// fanned-out entry matches what `aim_with_prediction` computes from scratch
/// for the rotated endpoints. Because the sibling board is an exact 90°·k
/// rotation of the original, the deterministic solver lands on the same
/// intercept turn and point, so equality holds to floating tolerance.
fn assert_sibling_aim_matches(cache: &EntityCache, ids: &[i64], ships_grid: &[i64]) {
    let mut checked = 0usize;
    for &src in ids {
        for &target in ids {
            if src == target {
                continue;
            }
            for &ships in ships_grid {
                let direct = aim_with_prediction(cache, src, target, ships, 0);
                cache.aim_cache_store(src, target, ships, 0, direct);

                for k in 1..=3 {
                    let sib_src = rot_sibling(src, k);
                    let sib_target = rot_sibling(target, k);
                    // The fan-out only stores siblings whose endpoints exist.
                    if cache.get(sib_src).is_none() || cache.get(sib_target).is_none() {
                        continue;
                    }

                    let cached = match cache.aim_cache_lookup(sib_src, sib_target, ships, 0) {
                        AimCacheVerdict::Hit(r) => r,
                        v => panic!(
                            "sibling ({sib_src},{sib_target}) ships={ships} k={k} not a cache hit \
                             (got {})",
                            match v {
                                AimCacheVerdict::Miss => "Miss",
                                AimCacheVerdict::Stale => "Stale",
                                AimCacheVerdict::Hit(_) => unreachable!(),
                            }
                        ),
                    };
                    let fresh = aim_with_prediction(cache, sib_src, sib_target, ships, 0);

                    match (cached, fresh) {
                        (None, None) => {}
                        (Some(c), Some(f)) => {
                            assert_eq!(
                                c.1, f.1,
                                "turns mismatch for sibling ({sib_src},{sib_target}) k={k}"
                            );
                            assert!(
                                wrap_pi(c.0 - f.0).abs() < 1e-6,
                                "angle mismatch for sibling ({sib_src},{sib_target}) k={k}: \
                                 cached={} fresh={}",
                                c.0,
                                f.0
                            );
                            assert!(
                                (c.2 - f.2).abs() < 1e-6 && (c.3 - f.3).abs() < 1e-6,
                                "target point mismatch for sibling ({sib_src},{sib_target}) k={k}: \
                                 cached=({},{}) fresh=({},{})",
                                c.2,
                                c.3,
                                f.2,
                                f.3
                            );
                            assert!(
                                (c.4 - f.4).abs() < 1e-6,
                                "flight_time mismatch for sibling ({sib_src},{sib_target}) k={k}"
                            );
                        }
                        (c, f) => panic!(
                            "feasibility mismatch for sibling ({sib_src},{sib_target}) k={k}: \
                             cached.is_some()={} fresh.is_some()={}",
                            c.is_some(),
                            f.is_some()
                        ),
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "expected at least one sibling pair to be checked");
}

/// Planet quartets: the rotated sibling stored on `aim_cache_store` matches a
/// direct solve, across several seeds and fleet sizes.
#[test]
fn quartet_aim_siblings_match_direct_solve() {
    let seeds = [42u64, 7, 100, 1234, 2025];
    let ships_grid = [5i64, 50, 500];
    for seed in seeds {
        let state = RefEngine::new(seed, 2, Configuration::default());
        let cache = cache_for(&state);
        let mut ids: Vec<i64> = cache
            .entities
            .keys()
            .copied()
            .filter(|&id| cache.position(id, 0).is_some())
            .collect();
        ids.sort();
        assert_sibling_aim_matches(&cache, &ids, &ships_grid);
    }
}

/// Comet quartets share the identical 4-member rotation structure, so comet
/// aim (and aim that must dodge comets) reuses across siblings too. Advance the
/// engine until a comet group spawns, then run the same round-trip check
/// restricted to pairs that involve a comet member.
#[test]
fn quartet_aim_siblings_match_direct_solve_with_comets() {
    let ships_grid = [5i64, 50];
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let noop: Vec<Vec<crate::engine::MoveAction>> = vec![Vec::new(), Vec::new()];
    let mut guard = 0;
    while state.comet_planet_ids.is_empty() && guard < EPISODE_STEPS {
        state.step_with_actions(&noop).unwrap();
        guard += 1;
    }
    assert!(!state.comet_planet_ids.is_empty(), "expected comets to spawn");

    let cache = cache_for(&state);
    let comet_ids: std::collections::HashSet<i64> =
        state.comet_planet_ids.iter().copied().collect();
    let mut ids: Vec<i64> = cache
        .entities
        .keys()
        .copied()
        .filter(|&id| cache.position(id, 0).is_some())
        .collect();
    ids.sort();

    // Keep the case set focused on comet-involving pairs (comet ↔ comet and
    // comet ↔ planet) while still exercising the shared rotation machinery.
    let comet_list: Vec<i64> = ids.iter().copied().filter(|id| comet_ids.contains(id)).collect();
    assert!(!comet_list.is_empty(), "cache should hold spawned comets");

    let mut checked = 0usize;
    for &src in &comet_list {
        for &target in &ids {
            if src == target {
                continue;
            }
            for &ships in &ships_grid {
                let direct = aim_with_prediction(&cache, src, target, ships, 0);
                cache.aim_cache_store(src, target, ships, 0, direct);
                for k in 1..=3 {
                    let sib_src = rot_sibling(src, k);
                    let sib_target = rot_sibling(target, k);
                    if cache.get(sib_src).is_none() || cache.get(sib_target).is_none() {
                        continue;
                    }
                    let cached = match cache.aim_cache_lookup(sib_src, sib_target, ships, 0) {
                        AimCacheVerdict::Hit(r) => r,
                        _ => panic!("comet sibling ({sib_src},{sib_target}) k={k} not a hit"),
                    };
                    let fresh = aim_with_prediction(&cache, sib_src, sib_target, ships, 0);
                    match (cached, fresh) {
                        (None, None) => {}
                        (Some(c), Some(f)) => {
                            assert_eq!(c.1, f.1, "turns for comet sibling k={k}");
                            assert!(wrap_pi(c.0 - f.0).abs() < 1e-6, "angle for comet sibling k={k}");
                            assert!(
                                (c.2 - f.2).abs() < 1e-6 && (c.3 - f.3).abs() < 1e-6,
                                "point for comet sibling k={k}"
                            );
                            assert!((c.4 - f.4).abs() < 1e-6, "flight_time for comet sibling k={k}");
                        }
                        (c, f) => panic!(
                            "feasibility mismatch comet sibling ({sib_src},{sib_target}) k={k}: \
                             {} vs {}",
                            c.is_some(),
                            f.is_some()
                        ),
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "expected comet sibling pairs to be checked");
}
