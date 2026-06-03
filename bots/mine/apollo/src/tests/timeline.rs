//! Regression test pinning the refactored `simulate_planet_timeline`
//! (`build_trajectory` + `finish_timeline`, with `Rc`-shared arrays) to the
//! original monolithic implementation. `reference_timeline` below is a
//! verbatim copy of that pre-refactor logic; the split must reproduce it
//! byte-for-byte across a wide grid of planets / arrival schedules / players /
//! horizons / expiries.

use crate::engine::{ArrivalEvent, Planet};
use crate::helpers::{
    normalize_arrivals, resolve_arrival_event, simulate_checkpoint_into, simulate_planet_timeline,
};

/// Verbatim copy of the original `simulate_planet_timeline` rollout, returning
/// the per-turn `owner_at`/`ships_at` arrays plus `horizon` as a tuple so we can
/// compare without depending on `PlanetTimeline`'s (now `Rc`-wrapped) field
/// types. (The old per-player summary fields — `keep_needed`/`min_owned`/
/// `first_enemy`/`fall_turn`/`holds_full` — were unused by the live planner and
/// have been removed, so they are no longer derived or asserted here.)
fn reference_timeline(
    planet: &Planet,
    arrivals: &[ArrivalEvent],
    _player: i64,
    horizon: i64,
    expiry_turn: Option<i64>,
) -> (Vec<i64>, Vec<i64>, i64) {
    let horizon = horizon.max(0);
    let effective_horizon = match expiry_turn {
        Some(exp) => horizon.min((exp - 1).max(0)),
        None => horizon,
    };
    let events = normalize_arrivals(arrivals, effective_horizon);

    let len = (horizon + 1) as usize;
    let mut by_turn: Vec<Vec<ArrivalEvent>> = vec![Vec::new(); len];
    for ev in &events {
        by_turn[ev.turns as usize].push(*ev);
    }

    let mut owner = planet.owner;
    let mut garrison = planet.ships;
    let mut owner_at: Vec<i64> = vec![owner; len];
    let mut ships_at: Vec<i64> = vec![garrison.max(0); len];

    for turn in 1..=effective_horizon {
        if owner != -1 {
            garrison += planet.production;
        }
        let group = &by_turn[turn as usize];
        if !group.is_empty() {
            let (no, ng) = resolve_arrival_event(owner, garrison, group);
            owner = no;
            garrison = ng;
        }
        owner_at[turn as usize] = owner;
        ships_at[turn as usize] = garrison.max(0);
    }

    for turn in (effective_horizon + 1)..=horizon {
        owner_at[turn as usize] = -1;
        ships_at[turn as usize] = 0;
    }

    (owner_at, ships_at, horizon)
}

fn planet(owner: i64, ships: i64, production: i64) -> Planet {
    Planet {
        id: 1,
        owner,
        x: 0.0,
        y: 0.0,
        radius: 1.0,
        ships,
        production,
    }
}

/// Tiny deterministic LCG so the grid is reproducible without `rand`/`Date`.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

#[test]
fn split_timeline_matches_reference() {
    let mut rng = Lcg(0x1234_5678);
    let owners = [-1i64, 0, 1, 2];
    let players = [-1i64, 0, 1, 2];
    let horizons = [0i64, 1, 5, 12, 30];
    let expiries = [None, Some(1i64), Some(3), Some(8), Some(30), Some(100)];

    let mut cases = 0u64;
    for &p_owner in &owners {
        for &ships in &[0i64, 1, 5, 25, 300] {
            for &production in &[0i64, 1, 3] {
                let pl = planet(p_owner, ships, production);
                for &player in &players {
                    for &horizon in &horizons {
                        for &expiry in &expiries {
                            // A few random arrival schedules per parameter combo.
                            for _ in 0..6 {
                                let n = rng.range(0, 5) as usize;
                                let arrivals: Vec<ArrivalEvent> = (0..n)
                                    .map(|_| ArrivalEvent {
                                        turns: rng.range(-1, 32),
                                        owner: rng.range(-1, 3),
                                        ships: rng.range(-2, 200),
                                    })
                                    .collect();

                                let got = simulate_planet_timeline(
                                    &pl, &arrivals, player, horizon, expiry,
                                );
                                let want =
                                    reference_timeline(&pl, &arrivals, player, horizon, expiry);

                                let ctx = format!(
                                    "owner={p_owner} ships={ships} prod={production} \
                                     player={player} horizon={horizon} expiry={expiry:?} \
                                     arrivals={arrivals:?}"
                                );
                                assert_eq!(*got.owner_at, want.0, "owner_at | {ctx}");
                                assert_eq!(*got.ships_at, want.1, "ships_at | {ctx}");
                                assert_eq!(got.horizon, want.2, "horizon | {ctx}");
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 10_000, "expected a broad grid, only ran {cases}");
}

/// The buffer-reusing `simulate_checkpoint_into` must reproduce a full re-sim
/// (`simulate_planet_timeline`) of the same arrival set, given a baseline built
/// from arrivals that match on the unchanged prefix (turns `< start_turn`).
#[test]
fn checkpoint_resim_matches_full_resim() {
    let mut rng = Lcg(0xBEEF_1234);
    let mut owner_buf: Vec<i64> = Vec::new();
    let mut ships_buf: Vec<i64> = Vec::new();
    let mut by_turn_buf: Vec<Vec<ArrivalEvent>> = Vec::new();
    let mut cases = 0u64;

    for &p_owner in &[-1i64, 0, 1, 2] {
        for &ships in &[0i64, 3, 40] {
            for &production in &[0i64, 1, 2] {
                let pl = planet(p_owner, ships, production);
                for &horizon in &[1i64, 6, 30] {
                    for &expiry in &[None, Some(4i64), Some(30)] {
                        for _ in 0..240 {
                            let start_turn = rng.range(1, horizon.max(1));

                            // `base` seeds the baseline; `full` adds extra
                            // arrivals only at turns >= start_turn (the
                            // checkpoint precondition).
                            let nb = rng.range(0, 4) as usize;
                            let base: Vec<ArrivalEvent> = (0..nb)
                                .map(|_| ArrivalEvent {
                                    turns: rng.range(1, horizon + 2),
                                    owner: rng.range(0, 3),
                                    ships: rng.range(1, 80),
                                })
                                .collect();
                            let mut full = base.clone();
                            for _ in 0..rng.range(0, 3) {
                                full.push(ArrivalEvent {
                                    turns: rng.range(start_turn, horizon + 2),
                                    owner: rng.range(0, 3),
                                    ships: rng.range(1, 80),
                                });
                            }

                            let baseline =
                                simulate_planet_timeline(&pl, &base, p_owner, horizon, expiry);
                            let expected =
                                simulate_planet_timeline(&pl, &full, p_owner, horizon, expiry);
                            simulate_checkpoint_into(
                                &pl,
                                &baseline,
                                start_turn,
                                &full,
                                expiry,
                                &mut owner_buf,
                                &mut ships_buf,
                                &mut by_turn_buf,
                            );

                            let ctx = format!(
                                "owner={p_owner} ships={ships} prod={production} \
                                 start={start_turn} horizon={horizon} expiry={expiry:?} \
                                 full={full:?}"
                            );
                            assert_eq!(owner_buf, *expected.owner_at, "owner_at | {ctx}");
                            assert_eq!(ships_buf, *expected.ships_at, "ships_at | {ctx}");
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 5_000, "expected a broad grid, only ran {cases}");
}
