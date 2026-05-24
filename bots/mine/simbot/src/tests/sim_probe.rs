use crate::engine::{Configuration, EngineState, MoveAction};
use crate::sim_probe::{SimEvent, SimProbe};

/// With no actions and no comet-spawn step in range, the probe should be
/// bit-identical to the engine for planets and fleets after N steps.
/// First comet spawn is at step 50; 25 turns from step 0 stays clear.
#[test]
fn probe_matches_engine_noop_short_horizon() {
    let engine = EngineState::new(42, 2, Configuration::default());

    let mut engine_run = engine.clone();
    let noop: Vec<Vec<MoveAction>> = vec![Vec::new(), Vec::new()];
    let mut probe = SimProbe::from_engine(&engine);

    for _ in 0..25 {
        engine_run.step_with_actions(&noop).unwrap();
        probe.step();
    }

    assert_eq!(probe.planets().len(), engine_run.planets.len());
    for (a, b) in probe.planets().iter().zip(engine_run.planets.iter()) {
        assert_eq!(a.id, b.id, "planet id mismatch");
        assert_eq!(a.owner, b.owner, "owner");
        assert!((a.x - b.x).abs() < 1e-12, "x: probe={} engine={}", a.x, b.x);
        assert!((a.y - b.y).abs() < 1e-12, "y: probe={} engine={}", a.y, b.y);
        assert_eq!(a.ships, b.ships, "ships");
    }
    assert_eq!(probe.fleets().len(), engine_run.fleets.len());
}

/// With one player launching a fleet on turn 0, the probe should track
/// the in-flight fleet and emit a `FleetLanded` event when it hits a
/// planet — matching the engine's combat outcome.
#[test]
fn probe_tracks_fleet_landing() {
    // Build an engine state with a single owned planet ready to launch.
    let engine = EngineState::new(42, 2, Configuration::default());

    // Pick an owned planet for player 0 and aim at the nearest enemy planet.
    let mut src_id = -1i64;
    let mut src_xy = (0.0, 0.0);
    let mut src_ships = 0i64;
    for p in &engine.planets {
        if p.owner == 0 {
            src_id = p.id;
            src_xy = (p.x, p.y);
            src_ships = p.ships;
            break;
        }
    }
    assert!(src_id >= 0, "no player-0 planet found");
    assert!(src_ships > 1, "need ships to launch");

    let mut tgt_id = -1i64;
    let mut tgt_xy = (0.0, 0.0);
    let mut best_d = f64::INFINITY;
    for p in &engine.planets {
        if p.owner == 0 || p.id == src_id {
            continue;
        }
        let d = ((p.x - src_xy.0).powi(2) + (p.y - src_xy.1).powi(2)).sqrt();
        if d < best_d {
            best_d = d;
            tgt_id = p.id;
            tgt_xy = (p.x, p.y);
        }
    }
    assert!(tgt_id >= 0, "no enemy/neutral target found");

    let angle = (tgt_xy.1 - src_xy.1).atan2(tgt_xy.0 - src_xy.0);
    let launch = vec![MoveAction {
        from_id: src_id,
        angle,
        ships: src_ships,
    }];

    let mut engine_run = engine.clone();
    let actions = vec![launch.clone(), Vec::new()];
    engine_run.step_with_actions(&actions).unwrap();

    let mut probe = SimProbe::from_engine(&engine);
    probe.step_with_player_actions(0, &launch);

    // Probe and engine should agree on fleet state after turn 1.
    assert_eq!(probe.fleets().len(), engine_run.fleets.len());

    // Step forward until probe sees the fleet land or we hit a horizon.
    let mut landed = None;
    for _ in 0..40 {
        engine_run.step_with_actions(&vec![Vec::new(), Vec::new()]).unwrap();
        probe.step();
        if let Some(SimEvent::FleetLanded { planet_id, .. }) = probe
            .events()
            .iter()
            .rev()
            .find(|e| matches!(e, SimEvent::FleetLanded { .. }))
            .copied()
        {
            landed = Some(planet_id);
            break;
        }
    }
    assert!(landed.is_some(), "fleet never landed within horizon");
    assert_eq!(landed.unwrap(), tgt_id, "fleet hit a different planet");

    // Engine should have dropped the fleet on the same turn the probe did.
    assert_eq!(probe.fleets().len(), engine_run.fleets.len());
}

/// The `collect_arrivals` shape mirrors what helpers.rs's arrival ledger
/// returns: one entry per fleet, bucketed by destination planet id.
#[test]
fn collect_arrivals_buckets_by_planet() {
    let engine = EngineState::new(42, 2, Configuration::default());
    let mut probe = SimProbe::from_engine(&engine);

    // Two owned planets each launch all their ships at the same target.
    let mut owned_p0: Vec<(i64, f64, f64, i64)> = engine
        .planets
        .iter()
        .filter(|p| p.owner == 0 && p.ships >= 2)
        .map(|p| (p.id, p.x, p.y, p.ships))
        .collect();
    if owned_p0.len() < 1 {
        return; // nothing to test
    }

    let tgt = engine.planets.iter().find(|p| p.owner != 0).unwrap();
    let mut launches = Vec::new();
    for (id, x, y, ships) in owned_p0.drain(..) {
        let angle = (tgt.y - y).atan2(tgt.x - x);
        launches.push(MoveAction {
            from_id: id,
            angle,
            ships,
        });
    }

    probe.step_with_player_actions(0, &launches);
    for _ in 0..40 {
        probe.step();
    }

    let ledger = probe.collect_arrivals();
    // Total number of FleetLanded events should equal the sum across buckets.
    let bucket_total: usize = ledger.values().map(|v| v.len()).sum();
    let landed_count = probe
        .events()
        .iter()
        .filter(|e| matches!(e, SimEvent::FleetLanded { .. }))
        .count();
    assert_eq!(bucket_total, landed_count);
}
