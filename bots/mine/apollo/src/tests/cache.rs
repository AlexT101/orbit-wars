use super::reference_engine::RefEngine;
use crate::aim::{aim_ignoring_comets, aim_with_prediction};
use crate::constants::EPISODE_STEPS;
use crate::engine::Configuration;
use crate::cache::{
    rot_sibling, AimCacheVerdict, EntityCache, EntityKind, InvariantVerdict,
};

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
        .iter()
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
        .iter()
        .find(|e| matches!(e.kind, EntityKind::OrbitingPlanet))
        .map(|e| e.id)
        .expect("seed 42 should have at least one orbiting planet");

    let noop: Vec<Vec<crate::engine::MoveAction>> = vec![Vec::new(), Vec::new()];
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

/// The precomputed `off_board_turn` (read by the planner to zero a comet's value
/// past its lifetime) must match an independent scan of the public `positions`
/// table: one past the last on-board index, clamped to `EPISODE_STEPS`.
#[test]
fn comet_off_board_turn_matches_linear_scan() {
    // Independent re-derivation of `off_board_turn` from the positions table.
    fn reference(positions: &[Option<[f64; 2]>]) -> i64 {
        let mut last_on_board: i64 = -1;
        for (t, p) in positions.iter().enumerate() {
            if p.is_some() {
                last_on_board = t as i64;
            }
        }
        if last_on_board < 0 {
            0
        } else {
            (last_on_board + 1).min(EPISODE_STEPS)
        }
    }

    // Advance until comets spawn (they appear on fixed game steps).
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let noop: Vec<Vec<crate::engine::MoveAction>> = vec![Vec::new(), Vec::new()];
    let mut guard = 0;
    while state.comet_planet_ids.is_empty() && guard < EPISODE_STEPS {
        state.step_with_actions(&noop).unwrap();
        guard += 1;
    }
    assert!(
        !state.comet_planet_ids.is_empty(),
        "expected comets to spawn"
    );

    let cache = cache_for(&state);
    let comet_ids: Vec<i64> = cache
        .entities
        .iter()
        .filter(|e| e.is_comet())
        .map(|e| e.id)
        .collect();
    assert!(
        !comet_ids.is_empty(),
        "cache should hold the spawned comets"
    );

    for &id in &comet_ids {
        let ent = cache.get(id).unwrap();
        assert_eq!(
            ent.off_board_turn,
            reference(&ent.positions),
            "comet {id} off_board_turn"
        );
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
    assert!(
        checked > 0,
        "expected at least one sibling pair to be checked"
    );
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
            .iter()
            .map(|e| e.id)
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
    assert!(
        !state.comet_planet_ids.is_empty(),
        "expected comets to spawn"
    );

    let cache = cache_for(&state);
    let comet_ids: std::collections::HashSet<i64> =
        state.comet_planet_ids.iter().copied().collect();
    let mut ids: Vec<i64> = cache
        .entities
        .iter()
        .map(|e| e.id)
        .filter(|&id| cache.position(id, 0).is_some())
        .collect();
    ids.sort();

    // Keep the case set focused on comet-involving pairs (comet ↔ comet and
    // comet ↔ planet) while still exercising the shared rotation machinery.
    let comet_list: Vec<i64> = ids
        .iter()
        .copied()
        .filter(|id| comet_ids.contains(id))
        .collect();
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
                            assert!(
                                wrap_pi(c.0 - f.0).abs() < 1e-6,
                                "angle for comet sibling k={k}"
                            );
                            assert!(
                                (c.2 - f.2).abs() < 1e-6 && (c.3 - f.3).abs() < 1e-6,
                                "point for comet sibling k={k}"
                            );
                            assert!(
                                (c.4 - f.4).abs() < 1e-6,
                                "flight_time for comet sibling k={k}"
                            );
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

/// Compare two aim results to floating tolerance (bearing wrapped).
fn assert_aim_eq(a: crate::cache::AimResult, b: crate::cache::AimResult, ctx: &str) {
    assert_eq!(a.1, b.1, "turns mismatch ({ctx})");
    assert!(
        wrap_pi(a.0 - b.0).abs() < 1e-6,
        "angle mismatch ({ctx}): {} vs {}",
        a.0,
        b.0
    );
    assert!(
        (a.2 - b.2).abs() < 1e-6 && (a.3 - b.3).abs() < 1e-6,
        "point mismatch ({ctx}): ({},{}) vs ({},{})",
        a.2,
        a.3,
        b.2,
        b.3
    );
    assert!((a.4 - b.4).abs() < 1e-6, "flight_time mismatch ({ctx})");
}

/// When no comet blocks it, the comet-free base equals the full solve: with no
/// comets on the board `aim_ignoring_comets` is bit-identical to
/// `aim_with_prediction`. Guards that the two share the same scan/cone logic.
#[test]
fn aim_ignoring_comets_matches_full_solve_without_comets() {
    let state = RefEngine::new(42, 2, Configuration::default());
    let mut cache = cache_for(&state);
    assert!(cache.comet_ids.is_empty(), "seed 42 starts with no comets");
    let mut ids: Vec<i64> = cache.entities.iter().map(|e| e.id).collect();
    ids.sort();
    let ships_grid = [5i64, 50, 500];

    let mut checked = 0usize;
    for t in [1i64, 7, 20] {
        cache.set_current_turn(t);
        for &src in &ids {
            for &target in &ids {
                if src == target {
                    continue;
                }
                for &ships in &ships_grid {
                    let cf = aim_ignoring_comets(&cache, src, target, ships, 0);
                    let full = aim_with_prediction(&cache, src, target, ships, 0);
                    match (cf, full) {
                        (None, None) => {}
                        (Some(a), Some(b)) => {
                            assert_eq!(a.0.to_bits(), b.0.to_bits(), "angle {src}->{target}");
                            assert_eq!(a.1, b.1, "turns {src}->{target}");
                        }
                        (a, b) => panic!(
                            "feasibility differ {src}->{target} s={ships}: {} vs {}",
                            a.is_some(),
                            b.is_some()
                        ),
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0);
}

/// Core invariant-cache guarantee: a static→static / orbiting→orbiting aim
/// solved at one (base) turn and carried to a later turn via
/// `invariant_aim_lookup` must equal an independent `aim_with_prediction` solve
/// at that later turn. Built at step 0 (no comets), so the comet gate never
/// fires and this isolates the static-fixed / orbiting-rotating carry.
#[test]
fn invariant_aim_matches_direct_solve_across_turns() {
    let seeds = [42u64, 7, 100, 1234, 2025];
    let ships_grid = [5i64, 50, 500];
    let mut nudged_carries = 0usize;
    for seed in seeds {
        let state = RefEngine::new(seed, 2, Configuration::default());
        let mut cache = cache_for(&state);
        let mut ids: Vec<i64> = cache
            .entities
            .iter()
            .filter(|e| !e.is_comet())
            .map(|e| e.id)
            .collect();
        ids.sort();

        // Solve + store every same-kind pair at the base turn, feeding the
        // comet-free base to the cache exactly as `plan_shot` does.
        let base_turn = 5;
        cache.set_current_turn(base_turn);
        for &src in &ids {
            for &target in &ids {
                if src == target {
                    continue;
                }
                for &ships in &ships_grid {
                    let base = aim_ignoring_comets(&cache, src, target, ships, 0);
                    cache.invariant_aim_store(src, target, ships, 0, base);
                }
            }
        }

        // Carry to later turns and compare against fresh solves.
        let mut checked = 0usize;
        for t in [6i64, 11, 23, 47, 88] {
            cache.set_current_turn(t);
            for &src in &ids {
                for &target in &ids {
                    if src == target {
                        continue;
                    }
                    for &ships in &ships_grid {
                        let InvariantVerdict::Use(inv) =
                            cache.invariant_aim_lookup(src, target, ships, 0)
                        else {
                            continue;
                        };
                        let fresh = aim_with_prediction(&cache, src, target, ships, 0).expect(
                            "invariant returned Use but fresh solve says None — feasibility drift",
                        );
                        assert_aim_eq(
                            inv,
                            fresh,
                            &format!("seed={seed} t={t} {src}->{target} s={ships}"),
                        );
                        checked += 1;
                        // Detect a cone-scanned (nudged) carry: bearing no longer
                        // points straight at the intercept point. With no comets,
                        // such a nudge is around a fixed/rotating planet and must
                        // still carry — the capability this whole change adds.
                        if let Some([lx, ly]) = cache.position(src, 0) {
                            let bearing = (inv.3 - ly).atan2(inv.2 - lx);
                            if wrap_pi(inv.0 - bearing).abs() > 1e-6 {
                                nudged_carries += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            checked > 0,
            "seed {seed}: expected at least one invariant hit"
        );
    }
    assert!(
        nudged_carries > 0,
        "expected at least one planet-nudged shot to be cached and carried across turns"
    );
}

/// With comets actually on the board, the invariant cache must stay sound: any
/// `Use` it returns (comet gate passed) must still match a fresh solve that
/// also sees those comets. Cases where a comet blocks the carried path surface
/// as `SingleSolve` (fall-back) and are simply skipped here.
#[test]
fn invariant_aim_sound_with_comets_present() {
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let noop: Vec<Vec<crate::engine::MoveAction>> = vec![Vec::new(), Vec::new()];
    let mut guard = 0;
    while state.comet_planet_ids.is_empty() && guard < EPISODE_STEPS {
        state.step_with_actions(&noop).unwrap();
        guard += 1;
    }
    assert!(
        !state.comet_planet_ids.is_empty(),
        "expected comets to spawn"
    );

    let mut cache = cache_for(&state);
    let comet_ids: std::collections::HashSet<i64> =
        state.comet_planet_ids.iter().copied().collect();
    let mut ids: Vec<i64> = cache
        .entities
        .iter()
        .filter(|e| !e.is_comet())
        .map(|e| e.id)
        .collect();
    ids.sort();

    let ships_grid = [5i64, 50];
    let base_turn = state.step.max(1);
    cache.set_current_turn(base_turn);
    for &src in &ids {
        for &target in &ids {
            if src == target {
                continue;
            }
            for &ships in &ships_grid {
                let base = aim_ignoring_comets(&cache, src, target, ships, 0);
                cache.invariant_aim_store(src, target, ships, 0, base);
            }
        }
    }

    for dt in [1i64, 3, 6, 10] {
        cache.set_current_turn(base_turn + dt);
        for &src in &ids {
            for &target in &ids {
                if src == target {
                    continue;
                }
                for &ships in &ships_grid {
                    let InvariantVerdict::Use(inv) =
                        cache.invariant_aim_lookup(src, target, ships, 0)
                    else {
                        continue;
                    };
                    // A returned shot's carried path is comet-clear, so a fresh
                    // full solve (sun+planets+comets) must agree.
                    let fresh = aim_with_prediction(&cache, src, target, ships, 0)
                        .expect("invariant Use but fresh None with comets present");
                    assert_aim_eq(
                        inv,
                        fresh,
                        &format!("comets t={} {src}->{target} s={ships}", base_turn + dt),
                    );
                }
            }
        }
    }
    let _ = comet_ids;
}
