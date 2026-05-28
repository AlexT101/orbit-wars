//! Simulation-oriented tests for the parametric blocker tester in
//! [`crate::blockers`]. For both verdicts of [`aim_with_prediction`] we
//! independently re-simulate the fleet trajectory and the per-turn swept-pair
//! collision against every obstacle, and require the two to agree.

use crate::blockers::{aim_with_prediction, bucket_to_speed, lead_target, speed_bucket};
use crate::constants::{BOARD_SIZE, CENTER, EPISODE_STEPS, HORIZON, LAUNCH_CLEARANCE, SUN_RADIUS};
use crate::engine::{fleet_speed, swept_pair_hit, Configuration, EngineState, MoveAction};
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

/// Ground truth: at the **quantized** fleet speed (so we test the same speed
/// the blocker table was built against), step the fleet one turn at a time and
/// run the engine's own [`swept_pair_hit`] against every non-target obstacle
/// and the sun. Returns the first colliding `(turn, blocker_id)`, where
/// `blocker_id == -1` represents the sun (which is not stored in
/// [`EntityCache::entities`]).
fn ground_truth_collision(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    angle: f64,
    ships: i64,
    flight_time: f64,
) -> Option<(i64, i64)> {
    let v = bucket_to_speed(speed_bucket(ships));
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

    // Simulate full per-turn chords up to `floor(flight_time)`, plus a partial
    // chord ending at exactly `flight_time` for the turn containing target
    // arrival. Stopping at `flight_time` matters because once the fleet
    // reaches its target it is consumed by the engine — a swept-pair
    // intersection at `s > (flight_time − floor(flight_time))` of the
    // arrival turn happens *after* the fleet has ceased to exist and is
    // therefore not a real collision the aimer needs to predict.
    let full_turns = flight_time.floor() as i64;
    let frac = flight_time - full_turns as f64;
    let last_t = if frac > 1e-9 {
        full_turns + 1
    } else {
        full_turns
    };

    for t in 1..=last_t {
        let s_end = if t == last_t && frac > 1e-9 {
            frac
        } else {
            1.0
        };
        let a = fleet_at((t - 1) as f64);
        let b = fleet_at((t - 1) as f64 + s_end);

        // Sun is a stationary disk at the board center; it isn't represented
        // as an Entity but the blocker table seeds it explicitly, so we have
        // to test it explicitly here too.
        if swept_pair_hit(a, b, (CENTER, CENTER), (CENTER, CENTER), SUN_RADIUS) {
            return Some((t, -1));
        }

        for (&bid, ent) in &cache.entities {
            if bid == shooter_id || bid == target_id {
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
        .filter(|(_, e)| e.positions[0].is_some())
        .map(|(&id, _)| id)
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

/// Diagnostic: dump geometry for the specific failing case.
#[test]
#[ignore]
fn dump_failing_case_geometry() {
    use crate::blockers::{bucket_to_speed, build_blocker_table, speed_bucket};
    let state = crate::engine::EngineState::new(100, 2, crate::engine::Configuration::default());
    let cache = cache_for(&state);
    let shooter = 8i64;
    let blocker = 12i64;
    let ships = 500i64;
    let v_quant = bucket_to_speed(speed_bucket(ships));
    println!("v_quant = {v_quant}");
    let [lx, ly] = cache.position(shooter, 0).unwrap();
    let shooter_r = cache.get(shooter).map(|e| e.radius).unwrap_or(0.0);
    let launch_offset = shooter_r + crate::constants::LAUNCH_CLEARANCE;
    println!("shooter pos = ({lx:.4}, {ly:.4}), launch_offset = {launch_offset:.4}");
    let blocker_r = cache.get(blocker).map(|e| e.radius).unwrap_or(0.0);
    println!("blocker radius = {blocker_r}");

    let t = 5i64;
    let [q0x, q0y] = cache.position(blocker, t - 1).unwrap();
    let [q1x, q1y] = cache.position(blocker, t).unwrap();
    println!("blocker turn {t}: Q(s=0)=({q0x:.4},{q0y:.4}) Q(s=1)=({q1x:.4},{q1y:.4})");
    let dqx = q1x - q0x;
    let dqy = q1y - q0y;
    let dq_speed = (dqx * dqx + dqy * dqy).sqrt();
    println!("blocker chord speed = {dq_speed:.6}");
    let d0 = launch_offset + (t as f64 - 1.0) * v_quant;
    println!("d0 (fleet ring at s=0 of turn {t}) = {d0:.4}");
    let mut min_diff = f64::INFINITY;
    let mut min_s = 0.0f64;
    for i in 0..=100 {
        let s = i as f64 / 100.0;
        let qx = q0x + s * dqx;
        let qy = q0y + s * dqy;
        let k = ((qx - lx).powi(2) + (qy - ly).powi(2)).sqrt();
        let d = d0 + s * v_quant;
        let diff = d - k;
        if diff < min_diff {
            min_diff = diff;
            min_s = s;
        }
    }
    println!("min D-K = {min_diff:.6} at s = {min_s:.2} (blocker_r = {blocker_r})");
    for i in 0..=20 {
        let s = i as f64 / 20.0;
        let qx = q0x + s * dqx;
        let qy = q0y + s * dqy;
        let k = ((qx - lx).powi(2) + (qy - ly).powi(2)).sqrt();
        let d = d0 + s * v_quant;
        let diff = d - k;
        println!(
            "  s={s:.2}: K={k:.4} D={d:.4} D-K={diff:.6} {}",
            if diff.abs() <= blocker_r { "COLLISION" } else { "" }
        );
    }

    let table = build_blocker_table(&cache, shooter, 0, v_quant);
    let angle = -1.585114f64;
    println!("\nTable entries for blocker {blocker} (turn 5 range s in [4,5]):");
    for e in &table.entries {
        if e.blocker_id == blocker && e.flight_t >= 3.9 && e.flight_t <= 5.1 {
            let aim_w = e.bearing + (angle - e.bearing).rem_euclid(2.0 * std::f64::consts::PI);
            let covers = aim_w >= e.aim_min && aim_w <= e.aim_max;
            println!(
                "  flight_t={:.3} bearing={:.4} half_arc={:.4} [{:.4},{:.4}] covers_angle={}",
                e.flight_t, e.bearing, e.half_arc, e.aim_min, e.aim_max, covers
            );
        }
    }
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
        let state = EngineState::new(seed, 2, Configuration::default());
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
                    let v = bucket_to_speed(speed_bucket(ships));
                    let Some((_, _, _, _, flight_time)) =
                        lead_target(&cache, shooter, target, 0, v)
                    else {
                        continue;
                    };

                    // Cap at HORIZON: the table only sees obstacles within
                    // HORIZON turns of launch by design — past that the
                    // aimer is silent, not wrong.
                    let sim_time = flight_time.min(HORIZON as f64);
                    let collision = ground_truth_collision(
                        &cache, shooter, target, angle, ships, sim_time,
                    );
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
/// [`lead_target`] proposed and require the same swept-pair re-simulation to
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
        let state = EngineState::new(seed, 2, Configuration::default());
        let cache = cache_for(&state);
        let ids = entity_ids_at_t0(&cache);

        for &shooter in &ids {
            for &target in &ids {
                if shooter == target {
                    continue;
                }
                for &ships in &ships_grid {
                    let v = bucket_to_speed(speed_bucket(ships));
                    let Some((angle, turns, _, _, flight_time)) =
                        lead_target(&cache, shooter, target, 0, v)
                    else {
                        continue;
                    };
                    if aim_with_prediction(&cache, shooter, target, ships, 0).is_some() {
                        continue;
                    }

                    let collision = ground_truth_collision(
                        &cache, shooter, target, angle, ships, flight_time,
                    );
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
    let state = EngineState::new(42, 2, Configuration::default());
    let cache = cache_for(&state);

    let statics: Vec<(i64, f64, f64, f64)> = cache
        .entities
        .values()
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
    // `lead_target` returns the closest-approach point `Q(s*)` on the target's
    // chord during the returned turn — anywhere in `[Q(turns-1), Q(turns)]`.
    // Verify the returned `(tx, ty)` lies on that segment (within numerical
    // tolerance) rather than equalling either endpoint.
    let seeds = [42u64, 7, 100];
    let ships_grid = [5i64, 17, 50, 120, 500];

    let mut mismatches = Vec::new();
    for seed in seeds {
        let state = EngineState::new(seed, 2, Configuration::default());
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
        let mut state = EngineState::new(seed, 2, Configuration::default());
        let mut cache = cache_for(&state);

        for &step in &future_steps {
            while state.step < step {
                state.step_with_actions(&noop).unwrap();
            }
            cache.set_current_turn(state.step);

            let mut ids: Vec<i64> = cache
                .entities
                .iter()
                .filter(|(_, e)| e.positions[state.step as usize].is_some())
                .map(|(&id, _)| id)
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
        let mut state = EngineState::new(seed, 2, Configuration::default());
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
                .filter(|(_, e)| e.positions[state.step as usize].is_some())
                .map(|(&id, _)| id)
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
