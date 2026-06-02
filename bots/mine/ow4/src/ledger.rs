//! Per-planet defense ledger.
//!
//! For each owned planet, compute the minimum ships I must keep on it so
//! ownership is NEVER lost across all future ticks, under this worst /
//! best case model (2p):
//!
//!   * **Enemy worst case** — every enemy planet launches its full current
//!     garrison toward this target *this turn*. Plus any enemy fleets
//!     already in flight to this target.
//!   * **Friendly best case** — every other owned planet launches its full
//!     current garrison toward this target *next turn* (one tick later, so
//!     enemy gets the first opportunity to attack). Plus any friendly
//!     fleets already in flight.
//!
//! Some launches won't have a valid path (sun blocking, no reachable arc)
//! — `dir_to_hit` returns `None`, those don't count.
//!
//! Once we have the arrival schedule, my standing on this planet over time
//! is:
//!
//!     standing(T) = initial + production·T + CUM_my(T) − CUM_enemy(T)
//!
//! Ownership rule: I keep control as long as `standing(T) ≥ 0` (0 ships
//! still counts as mine if I owned the previous tick). So the minimum
//! initial garrison that guarantees this is:
//!
//!     min_garrison = max(0, max over T of (CUM_enemy(T) − CUM_my(T) − production·T))
//!
//! Closed-form, no simulation — sort arrivals by tick and sweep.
//!
//! Surplus = `max(0, current_ships − min_garrison)`.

use rustc_hash::FxHashMap;

use crate::combat::{simulate_planet, Arrival};
use crate::game::{GameState, Planet};
use crate::pathing::{dir_to_hit, predict_fleet_collision};

/// Look this many ticks ahead when projecting in-flight ownership for
/// non-owned planets (used by the attack module to skip targets I'll own
/// anyway). Defense itself uses an event-driven horizon — no fixed limit.
pub const DEFENSE_HORIZON: i64 = 40;

pub struct Ledger<'a> {
    pub state: &'a GameState,
    pub me: i32,
    /// Per-planet in-flight arrival list keyed by planet id (mine + enemy
    /// fleets currently moving). Used by the attack module.
    pub arrivals: FxHashMap<i64, Vec<Arrival>>,
    /// Per-owned-planet surplus — ships I can send this turn without
    /// risking ownership under the worst-case enemy / best-case friendly
    /// race model.
    pub surplus: FxHashMap<i64, i64>,
    /// Per-planet projected end-of-horizon (owner, ships) under just the
    /// existing in-flight arrivals. Used to skip already-mine targets.
    pub timelines: FxHashMap<i64, (i32, i64)>,
}

impl<'a> Ledger<'a> {
    pub fn build(state: &'a GameState) -> Self {
        let me = state.player;
        let enemy = state.enemy_id();

        // Real in-flight arrivals.
        let mut arrivals: FxHashMap<i64, Vec<Arrival>> = FxHashMap::default();
        for f in &state.fleets {
            if let Some((pid, dt)) = predict_fleet_collision(f, state) {
                arrivals.entry(pid).or_default().push(Arrival {
                    dt,
                    owner: f.owner,
                    ships: f.ships,
                });
            }
        }

        // Projected end for each planet under in-flight only (cheap signal
        // for "skip this target — I'll own it without acting").
        let mut timelines: FxHashMap<i64, (i32, i64)> = FxHashMap::default();
        for p in &state.planets {
            let arrs = arrivals.get(&p.id).cloned().unwrap_or_default();
            let tl = simulate_planet(p, &arrs, DEFENSE_HORIZON, state);
            let end = tl.last().map(|&(_, o, s)| (o, s)).unwrap_or((p.owner, p.ships));
            timelines.insert(p.id, end);
        }

        // Defense surplus for each owned planet.
        let mut surplus: FxHashMap<i64, i64> = FxHashMap::default();
        for p in &state.planets {
            if p.owner != me {
                continue;
            }
            let inflight = arrivals.get(&p.id).cloned().unwrap_or_default();
            let mg = min_garrison(state, p, me, enemy, &inflight);
            let surp = (p.ships - mg).max(0);
            surplus.insert(p.id, surp);
        }

        Self {
            state,
            me,
            arrivals,
            surplus,
            timelines,
        }
    }

    pub fn surplus_at(&self, planet_id: i64) -> i64 {
        *self.surplus.get(&planet_id).unwrap_or(&0)
    }

    pub fn spend(&mut self, planet_id: i64, ships: i64) {
        if let Some(s) = self.surplus.get_mut(&planet_id) {
            *s = (*s - ships).max(0);
        }
    }

    pub fn projected_end(&self, planet_id: i64) -> (i32, i64) {
        *self.timelines.get(&planet_id).unwrap_or_else(|| {
            // Fall back to current state if not in map (shouldn't happen).
            panic!("planet {} not in ledger", planet_id)
        })
    }

    pub fn all_arrivals(&self, planet_id: i64) -> Vec<Arrival> {
        self.arrivals.get(&planet_id).cloned().unwrap_or_default()
    }
}

/// Minimum ships to keep on `planet` so I never lose control under the
/// worst-enemy / best-friendly race model. See module docs.
fn min_garrison(
    state: &GameState,
    planet: &Planet,
    me: i32,
    enemy: i32,
    inflight: &[Arrival],
) -> i64 {
    let mut enemy_arrivals: Vec<(i64, i64)> = Vec::new();
    let mut my_arrivals: Vec<(i64, i64)> = Vec::new();

    // In-flight fleets headed to this planet (mine + enemy).
    for a in inflight {
        if a.owner == enemy {
            enemy_arrivals.push((a.dt, a.ships));
        } else if a.owner == me {
            my_arrivals.push((a.dt, a.ships));
        }
    }

    // Worst-case enemy commit: every enemy planet launches its full
    // garrison toward this target *this turn*. Sources with no valid path
    // (dir_to_hit None) can't actually contribute.
    for ep in &state.planets {
        if ep.owner != enemy || ep.ships <= 0 || ep.id == planet.id {
            continue;
        }
        if let Some(pr) = dir_to_hit(ep, planet, ep.ships, state, 0) {
            enemy_arrivals.push((pr.time, ep.ships));
        }
    }

    // Best-case friendly reinforcement: every other owned planet launches
    // its full garrison *next turn* — enemy gets the first move. Arrival
    // tick = 1 (launch delay) + flight time.
    for fp in &state.planets {
        if fp.owner != me || fp.ships <= 0 || fp.id == planet.id {
            continue;
        }
        if let Some(pr) = dir_to_hit(fp, planet, fp.ships, state, 1) {
            my_arrivals.push((1 + pr.time, fp.ships));
        }
    }

    if enemy_arrivals.is_empty() {
        return 0;
    }

    enemy_arrivals.sort_by_key(|x| x.0);
    my_arrivals.sort_by_key(|x| x.0);

    // Sweep over event ticks. At each tick T compute
    //   deficit(T) = CUM_enemy(T) − CUM_my(T) − production·T
    // The minimum initial garrison is `max over T of deficit(T)`, clamped
    // at zero. We only need to evaluate at tick boundaries where CUM_* or
    // production·T changes — i.e. at every arrival tick (and right after).
    let max_t = enemy_arrivals
        .iter()
        .chain(my_arrivals.iter())
        .map(|x| x.0)
        .max()
        .unwrap_or(0);

    let mut cum_e: i64 = 0;
    let mut cum_m: i64 = 0;
    let mut ei = 0;
    let mut mi = 0;
    let mut max_deficit: i64 = 0;

    for t in 1..=max_t {
        while ei < enemy_arrivals.len() && enemy_arrivals[ei].0 <= t {
            cum_e += enemy_arrivals[ei].1;
            ei += 1;
        }
        while mi < my_arrivals.len() && my_arrivals[mi].0 <= t {
            cum_m += my_arrivals[mi].1;
            mi += 1;
        }
        let deficit = cum_e - cum_m - planet.production * t;
        if deficit > max_deficit {
            max_deficit = deficit;
        }
    }

    max_deficit.max(0)
}
