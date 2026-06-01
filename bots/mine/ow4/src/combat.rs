//! Combat resolution per Orbit Wars rules:
//!   - All arrivals on a planet on the same tick are sorted by ship count
//!     descending; the top two fight; survivors = |top1 - top2| owned by top1.
//!   - If top1 == top2, both destroyed; planet unchanged.
//!   - If survivors > 0, they then engage the planet's garrison:
//!       - same owner: garrison += survivors
//!       - different owner: garrison -= survivors; if negative, planet flips
//!         and new garrison = -ships.
//!   - Production is added to owned planets at the start of each tick (before
//!     combat on that tick).

use rustc_hash::FxHashMap;

use crate::game::{GameState, Planet};

/// One arrival's contribution. `dt` is ticks-from-now.
#[derive(Clone, Copy, Debug)]
pub struct Arrival {
    pub dt: i64,
    pub owner: i32,
    pub ships: i64,
}

/// Simulate `planet`'s owner/ships forward up to `horizon` ticks given a list
/// of arrivals (already-known mine + theirs). Production accrues each tick on
/// owned planets. Returns timeline `(dt, owner, ships)` for dt = 0..=horizon.
pub fn simulate_planet(
    planet: &Planet,
    arrivals: &[Arrival],
    horizon: i64,
    state: &GameState,
) -> Vec<(i64, i32, i64)> {
    // Group arrivals by dt.
    let mut by_dt: FxHashMap<i64, Vec<(i32, i64)>> = FxHashMap::default();
    for a in arrivals {
        by_dt.entry(a.dt).or_default().push((a.owner, a.ships));
    }

    let mut timeline = Vec::with_capacity(horizon as usize + 1);
    let mut owner = planet.owner;
    let mut ships = planet.ships;
    timeline.push((0, owner, ships));

    for dt in 1..=horizon {
        // Comet death.
        if planet.is_comet && state.planet_pos_at(planet, dt).is_none() {
            owner = -1;
            ships = 0;
            timeline.push((dt, owner, ships));
            continue;
        }
        // Production (start of tick on owned planets).
        if owner != -1 {
            ships += planet.production;
        }
        // Combat this tick.
        if let Some(arrs) = by_dt.get(&dt) {
            // Sum same-owner contributions.
            let mut by_owner: FxHashMap<i32, i64> = FxHashMap::default();
            for &(o, s) in arrs {
                *by_owner.entry(o).or_insert(0) += s;
            }
            let mut srt: Vec<(i32, i64)> =
                by_owner.into_iter().filter(|&(_, s)| s > 0).collect();
            srt.sort_by(|a, b| b.1.cmp(&a.1));
            // Top-two fight.
            let (sv_o, sv_s) = if srt.len() >= 2 {
                let (t1o, t1s) = srt[0];
                let (_, t2s) = srt[1];
                if t1s == t2s {
                    (-1, 0)
                } else {
                    (t1o, t1s - t2s)
                }
            } else if srt.len() == 1 {
                srt[0]
            } else {
                (-1, 0)
            };
            if sv_s > 0 {
                if owner == sv_o {
                    ships += sv_s;
                } else {
                    // Attacker fights the planet's garrison (whether neutral
                    // or enemy-owned). If they overwhelm it, planet flips.
                    ships -= sv_s;
                    if ships < 0 {
                        owner = sv_o;
                        ships = -ships;
                    }
                }
            }
        }
        timeline.push((dt, owner, ships));
    }
    timeline
}
