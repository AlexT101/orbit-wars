//! Simulation-oriented tests for the parametric blocker tester in
//! [`crate::aim`]. For both verdicts of [`aim_with_prediction`] we
//! independently re-simulate the fleet trajectory and the per-turn swept-pair
//! collision against every obstacle, and require the two to agree.

use super::reference_engine::RefEngine;
use crate::aim::{aim_with_prediction, lead_target_from, shot_blocked_exact};
use crate::cache::{EntityCache, EntityKind};
use crate::constants::{BOARD_SIZE, CENTER, EPISODE_STEPS, HORIZON, LAUNCH_CLEARANCE, SUN_RADIUS};
use crate::engine::{fleet_speed, swept_pair_hit, Configuration, MoveAction};

fn cache_for(state: &RefEngine) -> EntityCache {
    EntityCache::build(
        &state.initial_planets,
        &state.comets,
        &state.comet_planet_ids,
        state.angular_velocity,
        state.step,
    )
}

/// Ground truth: at the **exact engine** fleet speed `fleet_speed(ships)` (the
/// speed the simulator actually flies the fleet at, and the speed
/// [`aim_with_prediction`] now decides against — there is no longer a quantized
/// blocker table), step the fleet one turn at a time and run the engine's own
/// [`swept_pair_hit`] against every non-target obstacle and the sun. Returns the
/// first colliding `(turn, blocker_id)`, where `blocker_id == -1` represents the
/// sun (which is not stored in [`EntityCache::entities`]).
fn ground_truth_collision(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    angle: f64,
    ships: i64,
    flight_time: f64,
) -> Option<(i64, i64)> {
    let v = fleet_speed(ships.max(1), 6.0);
    let [lx, ly] = cache.position(shooter_id, 0)?;
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
    let cosa = angle.cos();
    let sina = angle.sin();
    let fleet_at = |k: f64| {
        (
            lx + (launch_offset + k * v) * cosa,
            ly + (launch_offset + k * v) * sina,
        )
    };

    // Simulate full per-turn chords through the arrival turn `last_t`. The engine
    // moves the fleet a *full* tick each turn (it isn't truncated at the target's
    // arrival fraction) and resolves it against the lowest-id planet hit anywhere
    // in the tick. So a lower-id obstacle struck even *past* the target on the
    // arrival tick still kills the fleet — sweep the full tick and let the
    // id/sun skips below model the arrival-turn tiebreak.
    let full_turns = flight_time.floor() as i64;
    let frac = flight_time - full_turns as f64;
    let last_t = if frac > 1e-9 {
        full_turns + 1
    } else {
        full_turns
    };

    for t in 1..=last_t {
        let s_end = 1.0;
        let a = fleet_at((t - 1) as f64);
        let b = fleet_at((t - 1) as f64 + s_end);

        // Sun is a stationary disk at the board center; it isn't represented
        // as an Entity but the blocker table seeds it explicitly, so we have
        // to test it explicitly here too. The engine checks the sun only after
        // the planet loop and only when no planet was hit that tick, so on the
        // arrival turn the target wins and the sun can't block — skip it there.
        if t != last_t && swept_pair_hit(a, b, (CENTER, CENTER), (CENTER, CENTER), SUN_RADIUS) {
            return Some((t, -1));
        }

        for ent in &cache.entities {
            let bid = ent.id;
            // Skip only the target (reaching it is success). The source is
            // *not* skipped: the engine checks the fleet against its own source
            // planet every turn, so an orbiting source that sweeps back into a
            // slow fleet's path is a real collision the aimer must predict.
            if bid == target_id {
                continue;
            }
            // Engine id-order tiebreak: on the arrival turn (`last_t`) the target
            // is resolved before any higher-id planet (planets are swept in id
            // order, first hit wins), so any planet — static or moving — with
            // id >= target_id cannot consume the fleet there. The aimer is allowed
            // to ignore it, so the ground truth must too.
            if t == last_t && bid >= target_id {
                continue;
            }
            let Some([q0x, q0y]) = cache.position(bid, t - 1) else {
                continue;
            };
            let Some([q1x, q1y]) = cache.position(bid, t) else {
                continue;
            };
            // Same partial-chord truncation for the blocker side: we only
            // need its position over `s ∈ [0, s_end]`, not the full turn.
            let q_end_x = q0x + s_end * (q1x - q0x);
            let q_end_y = q0y + s_end * (q1y - q0y);
            if swept_pair_hit(a, b, (q0x, q0y), (q_end_x, q_end_y), ent.radius) {
                return Some((t, bid));
            }
        }
    }
    None
}

/// All ids present in the cache at turn 0, sorted for reproducible iteration.
fn entity_ids_at_t0(cache: &EntityCache) -> Vec<i64> {
    let mut ids: Vec<i64> = cache
        .entities
        .iter()
        .filter(|e| e.positions[0].is_some())
        .map(|e| e.id)
        .collect();
    ids.sort();
    ids
}

/// Returns true iff the shot eventually collides with the target under the
/// engine's real fleet speed and swept-pair target motion. Stops when the
/// fleet hits, leaves the board, or the turn budget expires.
fn reaches_target_with_engine_speed(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    angle: f64,
    ships: i64,
    max_turns: i64,
) -> bool {
    let [sx, sy] = cache.position(shooter_id, 0).expect("shooter visible");
    let shooter_radius = cache.get(shooter_id).map(|e| e.radius).unwrap_or(0.0);
    let target_radius = cache.get(target_id).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
    let speed = fleet_speed(ships.max(1), 6.0);
    let mut fx = sx + angle.cos() * launch_offset;
    let mut fy = sy + angle.sin() * launch_offset;
    let vx = angle.cos() * speed;
    let vy = angle.sin() * speed;

    let mut prev_pos = cache.position(target_id, 0);
    for t in 1..=max_turns {
        let old_pos = (fx, fy);
        fx += vx;
        fy += vy;
        let new_pos = (fx, fy);
        let cur_pos = cache.position(target_id, t);
        let (px0, py0, px1, py1) = match (prev_pos, cur_pos) {
            (Some([x0, y0]), Some([x1, y1])) => (x0, y0, x1, y1),
            (Some([x0, y0]), None) => (x0, y0, x0, y0),
            _ => {
                prev_pos = cur_pos;
                continue;
            }
        };
        if swept_pair_hit(old_pos, new_pos, (px0, py0), (px1, py1), target_radius) {
            return true;
        }
        if !(0.0..=BOARD_SIZE).contains(&fx) || !(0.0..=BOARD_SIZE).contains(&fy) {
            return false;
        }
        prev_pos = cur_pos;
    }
    false
}

/// For every (shooter, target, ships) case where the aimer says "path clear",
/// re-simulate the fleet and require that no non-target obstacle's swept
/// per-turn pair test fires before the fleet reaches the target.
///
/// This is the "no false negatives" direction: if the aimer green-lights a
/// shot, the shot must really be unobstructed under the engine's own
/// collision primitive.
#[test]
fn clear_verdicts_survive_swept_pair_resimulation() {
    let seeds = [42u64, 7, 100, 1234, 2025];
    let ships_grid = [5i64, 50, 500];

    let mut clear_cases = 0usize;
    for seed in seeds {
        let state = RefEngine::new(seed, 2, Configuration::default());
        let cache = cache_for(&state);
        let ids = entity_ids_at_t0(&cache);

        for &shooter in &ids {
            for &target in &ids {
                if shooter == target {
                    continue;
                }
                for &ships in &ships_grid {
                    let Some((angle, turns, _, _, _)) =
                        aim_with_prediction(&cache, shooter, target, ships, 0)
                    else {
                        continue;
                    };
                    let v = fleet_speed(ships.max(1), 6.0);
                    let Some((_, _, _, _, flight_time)) =
                        lead_target_from(&cache, shooter, target, 0, v, 1)
                    else {
                        continue;
                    };

                    // Cap at HORIZON: the table only sees obstacles within
                    // HORIZON turns of launch by design — past that the
                    // aimer is silent, not wrong.
                    let sim_time = flight_time.min(HORIZON as f64);
                    let collision =
                        ground_truth_collision(&cache, shooter, target, angle, ships, sim_time);
                    if let Some((t, bid)) = collision {
                        panic!(
                            "seed={seed} ships={ships} shooter={shooter} target={target} \
                             angle={angle:.6} turns={turns} flight_time={flight_time:.3}: \
                             aimer says CLEAR but engine swept_pair_hit reports collision \
                             with blocker {bid} on turn {t}"
                        );
                    }
                    clear_cases += 1;
                }
            }
        }
    }
    assert!(
        clear_cases > 200,
        "expected hundreds of clear verdicts across {} seeds, only saw {clear_cases}",
        seeds.len()
    );
}

/// For every case where the aimer rejects a path, take the angle
/// [`lead_target_from`] proposed and require the same swept-pair re-simulation to
/// find a real collision with some non-target obstacle within the same
/// flight-time budget.
///
/// This is the "no false positives" direction: if the aimer red-lights a
/// shot, the engine's own collision primitive at that exact angle must agree
/// that something is in the way.
#[test]
fn blocked_verdicts_correspond_to_real_collisions() {
    let seeds = [42u64, 7, 100, 1234, 2025];
    let ships_grid = [5i64, 50, 500];

    let mut blocked_cases = 0usize;
    let mut missing = Vec::new();

    for seed in seeds {
        let state = RefEngine::new(seed, 2, Configuration::default());
        let cache = cache_for(&state);
        let ids = entity_ids_at_t0(&cache);

        for &shooter in &ids {
            for &target in &ids {
                if shooter == target {
                    continue;
                }
                for &ships in &ships_grid {
                    let v = fleet_speed(ships.max(1), 6.0);
                    let Some((angle, turns, _, _, flight_time)) =
                        lead_target_from(&cache, shooter, target, 0, v, 1)
                    else {
                        continue;
                    };
                    if aim_with_prediction(&cache, shooter, target, ships, 0).is_some() {
                        continue;
                    }

                    let collision =
                        ground_truth_collision(&cache, shooter, target, angle, ships, flight_time);
                    if collision.is_none() {
                        missing.push(format!(
                            "seed={seed} ships={ships} shooter={shooter} target={target} \
                             angle={angle:.6} turns={turns} flight_time={flight_time:.3}: \
                             aimer says BLOCKED but no swept_pair_hit found within \
                             flight_time"
                        ));
                    }
                    blocked_cases += 1;
                }
            }
        }
    }

    assert!(
        missing.is_empty(),
        "{} aimer-blocked cases had no corresponding swept_pair_hit collision:\n  {}",
        missing.len(),
        missing.join("\n  "),
    );
    assert!(
        blocked_cases > 50,
        "expected dozens of blocked verdicts across {} seeds, only saw {blocked_cases}",
        seeds.len()
    );
}

/// Targeted obvious case: any pair of static planets whose connecting chord
/// passes well inside the sun's destruction radius must be rejected by the
/// aimer. Catches catastrophic regressions even if the random sweep above
/// happens to miss the sun-blocking geometry on these seeds.
#[test]
fn sun_blocks_chords_passing_through_center() {
    let state = RefEngine::new(42, 2, Configuration::default());
    let cache = cache_for(&state);

    let statics: Vec<(i64, f64, f64, f64)> = cache
        .entities
        .iter()
        .filter(|e| e.is_static())
        .filter_map(|e| e.positions[0].map(|p| (e.id, p[0], p[1], e.radius)))
        .collect();

    let mut tested = 0usize;
    for &(a_id, ax, ay, _) in &statics {
        for &(b_id, bx, by, _) in &statics {
            if a_id == b_id {
                continue;
            }
            let dx = bx - ax;
            let dy = by - ay;
            let len2 = dx * dx + dy * dy;
            if len2 < 1.0 {
                continue;
            }
            // Closest approach of the (infinite) line a→b to the sun center.
            let t = ((CENTER - ax) * dx + (CENTER - ay) * dy) / len2;
            if !(0.05..=0.95).contains(&t) {
                continue;
            }
            let px = ax + t * dx;
            let py = ay + t * dy;
            let sun_d = ((px - CENTER).powi(2) + (py - CENTER).powi(2)).sqrt();
            // Require the chord to pass at least 2 units inside the sun
            // boundary — well past any floating-point ambiguity.
            if sun_d > SUN_RADIUS - 2.0 {
                continue;
            }

            for ships in [5i64, 50, 500] {
                let verdict = aim_with_prediction(&cache, a_id, b_id, ships, 0);
                assert!(
                    verdict.is_none(),
                    "shooter={a_id} target={b_id} ships={ships}: chord passes \
                     {sun_d:.2} from sun center (sun radius {SUN_RADIUS}) yet aimer says CLEAR \
                     (got {verdict:?})"
                );
                tested += 1;
            }
        }
    }
    assert!(
        tested > 0,
        "seed 42 should contain at least one diametrically opposite static pair"
    );
}

#[test]
fn lead_target_returned_point_matches_returned_turn_for_orbiters() {
    // `lead_target_from` returns the closest-approach point `Q(s*)` on the target's
    // chord during the returned turn — anywhere in `[Q(turns-1), Q(turns)]`.
    // Verify the returned `(tx, ty)` lies on that segment (within numerical
    // tolerance) rather than equalling either endpoint.
    let seeds = [42u64, 7, 100];
    let ships_grid = [5i64, 17, 50, 120, 500];

    let mut mismatches = Vec::new();
    for seed in seeds {
        let state = RefEngine::new(seed, 2, Configuration::default());
        let cache = cache_for(&state);
        let ids = entity_ids_at_t0(&cache);

        for &shooter in &ids {
            for &target in &ids {
                if shooter == target {
                    continue;
                }
                let Some(target_ent) = cache.get(target) else {
                    continue;
                };
                if !matches!(target_ent.kind, EntityKind::OrbitingPlanet) {
                    continue;
                }
                for &ships in &ships_grid {
                    let Some((_, turns, tx, ty, _)) =
                        aim_with_prediction(&cache, shooter, target, ships, 0)
                    else {
                        continue;
                    };
                    let Some([p0x, p0y]) = cache.position(target, turns - 1) else {
                        continue;
                    };
                    let Some([p1x, p1y]) = cache.position(target, turns) else {
                        continue;
                    };
                    let dqx = p1x - p0x;
                    let dqy = p1y - p0y;
                    let len_sq = dqx * dqx + dqy * dqy;
                    let s = if len_sq < 1e-18 {
                        0.0
                    } else {
                        ((tx - p0x) * dqx + (ty - p0y) * dqy) / len_sq
                    };
                    let on_seg = s.clamp(0.0, 1.0);
                    let proj_x = p0x + on_seg * dqx;
                    let proj_y = p0y + on_seg * dqy;
                    let err = (tx - proj_x).hypot(ty - proj_y);
                    if err > 1e-6 || !(-1e-9..=1.0 + 1e-9).contains(&s) {
                        mismatches.push(format!(
                            "seed={seed} shooter={shooter} target={target} ships={ships} turns={turns} \
                             returned=({tx:.3},{ty:.3}) s={s:.6} err={err:.6}"
                        ));
                        if mismatches.len() >= 12 {
                            break;
                        }
                    }
                }
                if mismatches.len() >= 12 {
                    break;
                }
            }
            if mismatches.len() >= 12 {
                break;
            }
        }
        if mismatches.len() >= 12 {
            break;
        }
    }

    assert!(
        mismatches.is_empty(),
        "lead_target returned aim points that did not match their returned ETA:\n  {}",
        mismatches.join("\n  "),
    );
}

#[test]
fn accepted_orbiting_shots_reach_target_under_engine_speed() {
    let seeds = [42u64, 7, 100];
    let ships_grid = [5i64, 17, 50, 120, 500];
    let future_steps = [0i64, 10];
    let noop: Vec<Vec<MoveAction>> = vec![Vec::new(), Vec::new()];

    let mut misses = Vec::new();
    for seed in seeds {
        let mut state = RefEngine::new(seed, 2, Configuration::default());
        let mut cache = cache_for(&state);

        for &step in &future_steps {
            while state.step < step {
                state.step_with_actions(&noop).unwrap();
            }
            cache.set_current_turn(state.step);

            let mut ids: Vec<i64> = cache
                .entities
                .iter()
                .filter(|e| e.positions[state.step as usize].is_some())
                .map(|e| e.id)
                .collect();
            ids.sort_unstable();

            for &shooter in &ids {
                for &target in &ids {
                    if shooter == target {
                        continue;
                    }
                    let Some(target_ent) = cache.get(target) else {
                        continue;
                    };
                    if !matches!(target_ent.kind, EntityKind::OrbitingPlanet) {
                        continue;
                    }
                    for &ships in &ships_grid {
                        let Some((angle, turns, _, _, _)) =
                            aim_with_prediction(&cache, shooter, target, ships, 0)
                        else {
                            continue;
                        };
                        let max_turns = (EPISODE_STEPS - cache.current_turn).max(0);
                        if reaches_target_with_engine_speed(
                            &cache, shooter, target, angle, ships, max_turns,
                        ) {
                            continue;
                        }
                        misses.push(format!(
                            "seed={seed} step={step} shooter={shooter} target={target} ships={ships} \
                             angle={angle:.6} turns={turns}"
                        ));
                        if misses.len() >= 12 {
                            break;
                        }
                    }
                    if misses.len() >= 12 {
                        break;
                    }
                }
                if misses.len() >= 12 {
                    break;
                }
            }
            if misses.len() >= 12 {
                break;
            }
        }
        if misses.len() >= 12 {
            break;
        }
    }

    assert!(
        misses.is_empty(),
        "accepted orbiting shots that never reached their target:\n  {}",
        misses.join("\n  "),
    );
}

#[test]
fn wide_seed_scan_accepted_orbiting_shots_reach_target() {
    // Broader version of `accepted_orbiting_shots_reach_target_under_engine_speed`:
    // 30 seeds × 5 game steps. The original 3-seed × 2-step coverage was
    // insufficient to surface the orbital-target convergence bug that caused
    // fleets to fly off the map; pre-fix this scan produced 20+ misses on
    // seed=0 step=0 alone.
    let seeds: Vec<u64> = (0..30).collect();
    let future_steps = [0i64, 25, 75, 150, 300];
    let ships_grid = [10i64, 100, 500];
    let noop: Vec<Vec<MoveAction>> = vec![Vec::new(), Vec::new()];

    let mut misses = Vec::new();
    for seed in seeds {
        let mut state = RefEngine::new(seed, 2, Configuration::default());
        let mut cache = cache_for(&state);

        for &step in &future_steps {
            while state.step < step {
                if state.step_with_actions(&noop).is_err() {
                    break;
                }
            }
            cache.set_current_turn(state.step);

            let mut ids: Vec<i64> = cache
                .entities
                .iter()
                .filter(|e| e.positions[state.step as usize].is_some())
                .map(|e| e.id)
                .collect();
            ids.sort_unstable();

            for &shooter in &ids {
                for &target in &ids {
                    if shooter == target {
                        continue;
                    }
                    let Some(target_ent) = cache.get(target) else {
                        continue;
                    };
                    if !matches!(target_ent.kind, EntityKind::OrbitingPlanet) {
                        continue;
                    }
                    for &ships in &ships_grid {
                        let Some((angle, turns, _, _, _)) =
                            aim_with_prediction(&cache, shooter, target, ships, 0)
                        else {
                            continue;
                        };
                        let max_turns = (EPISODE_STEPS - cache.current_turn).max(0);
                        if !reaches_target_with_engine_speed(
                            &cache, shooter, target, angle, ships, max_turns,
                        ) {
                            misses.push(format!(
                                "seed={seed} step={step} shooter={shooter} target={target} \
                                 ships={ships} angle={angle:.6} turns={turns}"
                            ));
                            if misses.len() >= 12 {
                                break;
                            }
                        }
                    }
                    if misses.len() >= 12 {
                        break;
                    }
                }
                if misses.len() >= 12 {
                    break;
                }
            }
            if misses.len() >= 12 {
                break;
            }
        }
        if misses.len() >= 12 {
            break;
        }
    }

    assert!(
        misses.is_empty(),
        "accepted orbiting shots that did not reach target (broad scan):\n  {}",
        misses.join("\n  "),
    );
}

/// Bit-identity guard for the binary-searched H-reduction in `blocked_on_path`.
///
/// The reduction skips turns it proves can't contact (the fleet's monotonic
/// outward radius makes the radially-reachable turns a contiguous band) and runs
/// the exact swept test only inside that band — with an exact full-scan fallback
/// when a slow fleet can be out-paced radially (`fleet_speed ≤ obstacle step`).
/// Either way the verdict must equal an exhaustive per-turn scan.
///
/// We advance until a comet group spawns (comets are the fast movers that drive
/// the fallback branch and the windowed clamp), then sweep every bearing from
/// several launchers across the ships grid (`5` → slow fleet/fallback, `500` →
/// fast fleet/binary-search) and require `shot_blocked_exact` to agree with the
/// brute-force `ground_truth_collision` for every angle.
#[test]
fn h_reduction_matches_exhaustive_scan_with_comets() {
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let noop: Vec<Vec<MoveAction>> = vec![Vec::new(), Vec::new()];
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
    let now = cache.current_turn as usize;
    // Entities on board *now* (offset 0) — includes the freshly spawned comets,
    // which `entity_ids_at_t0` (keyed on absolute turn 0) would miss.
    let mut on_board: Vec<i64> = cache
        .entities
        .iter()
        .filter(|e| e.positions.get(now).map(|p| p.is_some()).unwrap_or(false))
        .map(|e| e.id)
        .collect();
    on_board.sort();
    let launchers: Vec<i64> = on_board
        .iter()
        .copied()
        .filter(|id| !cache.get(*id).map(|e| e.is_comet()).unwrap_or(true))
        .take(8)
        .collect();
    assert!(!launchers.is_empty(), "need planet launchers");

    let ships_grid = [5i64, 50, 500];
    let steps = 240usize;
    let mut compared = 0usize;
    for &shooter in &launchers {
        for &target in on_board.iter().take(4) {
            if shooter == target {
                continue;
            }
            for &ships in &ships_grid {
                let v = fleet_speed(ships.max(1), 6.0);
                // A generous flight budget so the swept window spans the comets.
                let flight_time = HORIZON as f64;
                for k in 0..steps {
                    let angle = std::f64::consts::TAU * k as f64 / steps as f64;
                    let band =
                        shot_blocked_exact(&cache, shooter, target, angle, flight_time, v, 0);
                    let brute =
                        ground_truth_collision(&cache, shooter, target, angle, ships, flight_time)
                            .is_some();
                    assert_eq!(
                        band, brute,
                        "verdict mismatch: shooter={shooter} target={target} ships={ships} \
                         v={v:.3} angle={angle:.6} — band={band} exhaustive={brute}"
                    );
                    compared += 1;
                }
            }
        }
    }
    assert!(
        compared > 1000,
        "expected a broad sweep, only {compared} comparisons"
    );
}

/// Soundness guard for the cone-scan with comets present.
///
/// `clear_verdicts_survive_swept_pair_resimulation` already checks this for the
/// comet-free board; this extends it to a comet-spawned cache, requiring every
/// CLEAR verdict from `aim_with_prediction` to survive the exhaustive resim.
#[test]
fn cone_cull_clear_verdicts_sound_with_comets() {
    let mut state = RefEngine::new(42, 2, Configuration::default());
    let noop: Vec<Vec<MoveAction>> = vec![Vec::new(), Vec::new()];
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
    let now = cache.current_turn as usize;
    let mut ids: Vec<i64> = cache
        .entities
        .iter()
        .filter(|e| e.positions.get(now).map(|p| p.is_some()).unwrap_or(false))
        .map(|e| e.id)
        .collect();
    ids.sort();

    let ships_grid = [5i64, 50, 500];
    let mut clear_cases = 0usize;
    for &shooter in &ids {
        if cache.get(shooter).map(|e| e.is_comet()).unwrap_or(true) {
            continue; // launch from planets
        }
        for &target in &ids {
            if shooter == target {
                continue;
            }
            for &ships in &ships_grid {
                let Some((angle, _, _, _, _)) =
                    aim_with_prediction(&cache, shooter, target, ships, 0)
                else {
                    continue;
                };
                let v = fleet_speed(ships.max(1), 6.0);
                let Some((_, _, _, _, flight_time)) =
                    lead_target_from(&cache, shooter, target, 0, v, 1)
                else {
                    continue;
                };
                let sim_time = flight_time.min(HORIZON as f64);
                if let Some((t, bid)) =
                    ground_truth_collision(&cache, shooter, target, angle, ships, sim_time)
                {
                    panic!(
                        "shooter={shooter} target={target} ships={ships} angle={angle:.6}: \
                         aimer says CLEAR but engine reports collision with {bid} on turn {t} \
                         (cone cull dropped a real blocker?)"
                    );
                }
                clear_cases += 1;
            }
        }
    }
    assert!(
        clear_cases > 50,
        "expected many clear verdicts, saw {clear_cases}"
    );
}

/// Regression: a fleet launched on the last actable step (`EPISODE_STEPS - 1`)
/// moves and can collide that same step, so a reachable adjacent target must
/// still yield a turn-1 intercept. The aim horizon used to be capped at
/// `EPISODE_STEPS - 1 - abs_launch`, which collapsed to 0 here and dropped the
/// shot before testing it (and the position table lacked the index the final
/// tick reads). Both are fixed: the cap is `EPISODE_STEPS - abs_launch` and the
/// table carries the trailing `EPISODE_STEPS` slot.
#[test]
fn final_tick_launch_can_still_lead_an_adjacent_target() {
    use crate::engine::Planet;

    // Two static planets (orbital_radius + radius >= ROTATION_LIMIT = 50) so
    // their positions never move — keeps the geometry exact at any step.
    let shooter = Planet {
        id: 1,
        owner: 0,
        x: CENTER + 49.0,
        y: CENTER,
        radius: 2.0,
        ships: 0,
        production: 0,
    };
    // Centers 6 apart; with ~6.0 fleet speed the fleet sweeps from launch
    // offset 2.1 out past the target center in a single tick.
    let target = Planet {
        id: 2,
        owner: 1,
        x: CENTER + 55.0,
        y: CENTER,
        radius: 2.0,
        ships: 0,
        production: 0,
    };
    let planets = vec![shooter, target];

    let mut cache = EntityCache::build(&planets, &[], &[], 0.05, EPISODE_STEPS - 1);
    cache.set_current_turn(EPISODE_STEPS - 1);
    assert_eq!(cache.get(1).map(|e| e.kind), Some(EntityKind::StaticPlanet));
    assert_eq!(cache.get(2).map(|e| e.kind), Some(EntityKind::StaticPlanet));

    let v = fleet_speed(1000, 6.0); // ~6.0, enough to reach in one tick
    let got = lead_target_from(&cache, 1, 2, 0, v, 1);
    let (_, turns, _, _, flight_time) = got.expect("final-tick launch should find an intercept");
    assert_eq!(turns, 1, "intercept should land on the launch step itself");
    assert!(
        flight_time > 0.0 && flight_time <= 1.0,
        "flight_time {flight_time} should be within the single final tick"
    );
}
