//! Strategy-agnostic per-turn world snapshot.
//!
//! Built once per `Bot::compute_moves` call after the [`EntityCache`] is
//! refreshed for the current step. Strategies receive `&WorldState` and read
//! whichever pieces they need — observation snapshots, planet groupings by
//! owner, aggregate strength/production, and the `TimelineCache`. Per-planet
//! timeline metrics (keep_needed, fall_turn, …) are accessed via
//! `ws.timeline_cache.baseline(planet_id)`.
//!
//! Anything strategy-flavoured (phase windows like "early"/"opening", scoring
//! heuristics, solver memoization) belongs in the strategy module, not here.

#![allow(dead_code)]

use std::cell::RefCell;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::apollo::aim::AimResult;
use crate::apollo::constants::{EPISODE_STEPS, HORIZON};
use crate::apollo::engine::{CometGroup, EngineState, Fleet, Planet, Simulator};
use crate::apollo::cache::EntityCache;

/// Step-scoped, lock-free L1 aim cache shared across every [`WorldState`] and
/// `HellburnerModel` built during one `Bot::compute_moves` call. Keyed by the
/// **absolute** launch turn (`current_turn + launch_turn_offset`) so it stays
/// correct as the rollout walks `current_turn` forward; aim is player- and
/// view-agnostic, so the same key resolves to the same shot for every model.
///
/// Lives on the `compute_moves` stack (one OS thread for the whole step) and is
/// threaded in by reference — never stored in the `Bot` pyclass and never shared
/// across threads — so a bare `RefCell` is sound here without any `Sync` shim.
pub type ShotL1 = RefCell<HashMap<(i64, i64, i64, i64), Option<AimResult>>>;
use crate::apollo::helpers::{
    count_players, simulate_planet_timeline, state_at_timeline, ArrivalEvent, ArrivalLedger,
    PlanetTimeline, TimelineCache,
};

/// Per-turn world snapshot. Strategy code borrows this and never mutates it.
pub struct WorldState<'a> {
    pub player: i64,
    pub step: i64,
    pub angular_velocity: f64,
    pub cache: &'a EntityCache,
    /// Optional step-scoped L1 aim cache (see [`ShotL1`]). `None` for tests and
    /// ad-hoc worlds, which fall back to each `HellburnerModel`'s own per-model
    /// cache. Set by the live `Bot` paths so all models in a step share hits.
    pub shot_l1: Option<&'a ShotL1>,
    pub timeline_cache: TimelineCache,

    pub planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    /// `planet_id → index into `planets`. Avoids a second full copy of every
    /// planet; look up positional data via [`Self::planet`].
    pub planet_by_id: HashMap<i64, usize>,
    pub comet_ids: HashSet<i64>,

    pub my_planets: Vec<Planet>,
    pub enemy_planets: Vec<Planet>,
    pub neutral_planets: Vec<Planet>,

    pub num_players: usize,
    pub is_four_player: bool,
    pub remaining_steps: i64,

    pub owner_strength: HashMap<i64, i64>,
    pub owner_production: HashMap<i64, i64>,
    pub my_total: i64,
    pub enemy_total: i64,
    pub max_enemy_strength: i64,
    pub my_prod: i64,
    pub enemy_prod: i64,
    pub total_visible_ships: i64,
    pub total_production: i64,

    /// defaults to 0.0 for rollout-internal and test-built worlds so they take the cheap
    /// path through cost-gated logic.
    pub remaining_overage_time: f64,
}

impl<'a> WorldState<'a> {
    /// Build the per-turn snapshot from the parsed observation + cache.
    /// `cache.current_turn` should already be set to `step`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        player: i64,
        step: i64,
        planets: Vec<Planet>,
        fleets: Vec<Fleet>,
        initial_planets: Vec<Planet>,
        comets: Vec<CometGroup>,
        comet_planet_ids: Vec<i64>,
        angular_velocity: f64,
        cache: &'a EntityCache,
    ) -> Self {
        let num_players = count_players(&planets, &fleets);
        let next_fleet_id = fleets
            .iter()
            .map(|f| f.id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let engine = EngineState::from_observation_parts(
            step,
            angular_velocity,
            planets,
            initial_planets,
            fleets,
            next_fleet_id,
            comet_planet_ids,
            comets,
            num_players,
        );
        Self::from_engine(player, &engine, cache)
    }

    /// Constructor for callers that hold an `EngineState` (the non-search
    /// `compute_moves` path and tests): spin up a forward [`Simulator`], walk
    /// the `HORIZON`-turn arrival ledger, and snapshot. The rollout/search loop
    /// instead calls [`Self::from_simulator_with_ledger`] directly so the
    /// player-agnostic ledger can be shared across players.
    pub fn from_engine(player: i64, engine: &EngineState, cache: &'a EntityCache) -> Self {
        let sim = Simulator::new(engine);
        let ledger = ArrivalLedger::build(&sim, HORIZON, cache);
        Self::from_simulator_with_ledger(player, &sim, &ledger, cache)
    }

    /// Shared-ledger constructor used during rollout. Skips the sim walk
    /// in favor of reusing `ledger` — caller is responsible for ensuring the
    /// ledger was built from the same sim snapshot. Used by `rollout` to
    /// share the player-agnostic forward sim across every player's WorldState
    /// in a reactive turn.
    pub fn from_simulator_with_ledger(
        player: i64,
        sim: &Simulator,
        ledger: &ArrivalLedger,
        cache: &'a EntityCache,
    ) -> Self {
        let step = sim.step_count();
        let angular_velocity = sim.angular_velocity();
        let num_players = sim.num_players();
        let timeline_cache = TimelineCache::from_ledger(sim.planets(), player, ledger);

        let planets: Vec<Planet> = sim.planets().to_vec();
        let fleets: Vec<Fleet> = sim.fleets().to_vec();
        let comet_planet_ids: Vec<i64> = sim.comet_planet_ids().to_vec();

        let mut planet_by_id: HashMap<i64, usize> =
            HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        for (idx, planet) in planets.iter().enumerate() {
            planet_by_id.insert(planet.id, idx);
        }
        let comet_ids: HashSet<i64> = comet_planet_ids.iter().copied().collect();

        let mut my_planets = Vec::new();
        let mut enemy_planets = Vec::new();
        let mut neutral_planets = Vec::new();
        for planet in &planets {
            if planet.owner == player {
                my_planets.push(*planet);
            } else if planet.owner == -1 {
                neutral_planets.push(*planet);
            } else {
                enemy_planets.push(*planet);
            }
        }

        let remaining_steps = (EPISODE_STEPS - step).max(1);
        let is_four_player = num_players >= 4;

        let mut owner_strength: HashMap<i64, i64> = HashMap::default();
        let mut owner_production: HashMap<i64, i64> = HashMap::default();
        for planet in &planets {
            if planet.owner != -1 {
                *owner_strength.entry(planet.owner).or_insert(0) += planet.ships;
                *owner_production.entry(planet.owner).or_insert(0) += planet.production;
            }
        }
        for fleet in &fleets {
            *owner_strength.entry(fleet.owner).or_insert(0) += fleet.ships;
        }
        let my_total = *owner_strength.get(&player).unwrap_or(&0);
        let enemy_total: i64 = owner_strength
            .iter()
            .filter(|(o, _)| **o != player)
            .map(|(_, s)| *s)
            .sum();
        let max_enemy_strength = owner_strength
            .iter()
            .filter(|(o, _)| **o != player)
            .map(|(_, s)| *s)
            .max()
            .unwrap_or(0);
        let my_prod = *owner_production.get(&player).unwrap_or(&0);
        let enemy_prod: i64 = owner_production
            .iter()
            .filter(|(o, _)| **o != player)
            .map(|(_, s)| *s)
            .sum();

        let total_visible_ships: i64 = planets.iter().map(|p| p.ships).sum::<i64>()
            + fleets.iter().map(|f| f.ships).sum::<i64>();
        let total_production: i64 = planets.iter().map(|p| p.production).sum();

        Self {
            player,
            step,
            angular_velocity,
            cache,
            shot_l1: None,
            timeline_cache,
            planets,
            fleets,
            planet_by_id,
            comet_ids,
            my_planets,
            enemy_planets,
            neutral_planets,
            num_players,
            is_four_player,
            remaining_steps,
            owner_strength,
            owner_production,
            my_total,
            enemy_total,
            max_enemy_strength,
            my_prod,
            enemy_prod,
            total_visible_ships,
            total_production,
            remaining_overage_time: 0.0,
        }
    }

    #[inline]
    pub fn planet(&self, id: i64) -> &Planet {
        &self.planets[self.planet_by_id[&id]]
    }

    pub fn is_static(&self, planet_id: i64) -> bool {
        self.cache
            .get(planet_id)
            .map(|e| e.is_static())
            .unwrap_or(false)
    }

    pub fn comet_life(&self, planet_id: i64) -> i64 {
        self.cache.remaining_life(planet_id)
    }

    pub fn source_inventory_left(&self, source_id: i64, spent_total: &HashMap<i64, i64>) -> i64 {
        let cap = self.planet(source_id).ships;
        (cap - spent_total.get(&source_id).copied().unwrap_or(0)).max(0)
    }

    /// Loose upper bound on how many ships could ever flip a planet — used by
    /// the binary-search solvers as a doubling-search ceiling.
    pub fn ownership_search_cap(&self, eval_turn: i64) -> i64 {
        let productive_cap = self.total_production * 2.max(eval_turn + 2);
        (self.total_visible_ships + productive_cap + 32).max(32)
    }

    /// `(owner, ships)` at `arrival_turn` after merging caller-provided
    /// arrivals into the timeline cache's base ledger. Generic over which
    /// arrival sources are merged so strategy code can keep its planned
    /// commitments separate from one-off "what if" arrivals.
    pub fn projected_state(
        &self,
        target_id: i64,
        arrival_turn: i64,
        planned: &[ArrivalEvent],
        extra: &[ArrivalEvent],
    ) -> (i64, i64) {
        let cutoff = arrival_turn.max(1);
        if planned.is_empty() && extra.is_empty() {
            if let Some(baseline) = self.timeline_cache.baseline(target_id) {
                return state_at_timeline(baseline, cutoff);
            }
        }
        let merged = merge_arrivals(
            self.timeline_cache.arrivals(target_id),
            planned,
            extra,
            cutoff,
        );
        let target = self.planet(target_id);
        let expiry = self.timeline_cache.expiry(target_id);
        let tl = simulate_planet_timeline(target, &merged, self.player, cutoff, expiry);
        state_at_timeline(&tl, cutoff)
    }

    pub fn projected_timeline(
        &self,
        target_id: i64,
        horizon: i64,
        planned: &[ArrivalEvent],
        extra: &[ArrivalEvent],
    ) -> PlanetTimeline {
        let horizon = horizon.max(1);
        let merged = merge_arrivals(
            self.timeline_cache.arrivals(target_id),
            planned,
            extra,
            horizon,
        );
        let target = self.planet(target_id);
        let expiry = self.timeline_cache.expiry(target_id);
        simulate_planet_timeline(target, &merged, self.player, horizon, expiry)
    }

    pub fn hold_status(
        &self,
        target_id: i64,
        planned: &[ArrivalEvent],
        horizon: i64,
    ) -> HoldStatus {
        if !planned.is_empty() {
            let tl = self.projected_timeline(target_id, horizon, planned, &[]);
            HoldStatus {
                keep_needed: tl.keep_needed,
                min_owned: tl.min_owned,
                first_enemy: tl.first_enemy,
                fall_turn: tl.fall_turn,
                holds_full: tl.holds_full,
            }
        } else if let Some(baseline) = self.timeline_cache.baseline(target_id) {
            HoldStatus {
                keep_needed: baseline.keep_needed,
                min_owned: baseline.min_owned,
                first_enemy: baseline.first_enemy,
                fall_turn: baseline.fall_turn,
                holds_full: baseline.holds_full,
            }
        } else {
            HoldStatus::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HoldStatus {
    pub keep_needed: i64,
    pub min_owned: i64,
    pub first_enemy: Option<i64>,
    pub fall_turn: Option<i64>,
    pub holds_full: bool,
}

/// Merge base arrivals + planned commitments + extra arrivals into one
/// arrival list (unsorted — `simulate_planet_timeline` re-normalizes), dropping
/// anything past `cutoff`.
pub fn merge_arrivals(
    base: &[ArrivalEvent],
    planned: &[ArrivalEvent],
    extra: &[ArrivalEvent],
    cutoff: i64,
) -> Vec<ArrivalEvent> {
    let mut out: Vec<ArrivalEvent> = Vec::with_capacity(base.len() + planned.len() + extra.len());
    for ev in base {
        if ev.turns <= cutoff {
            out.push(*ev);
        }
    }
    for ev in planned {
        if ev.turns <= cutoff {
            out.push(*ev);
        }
    }
    for ev in extra {
        if ev.turns <= cutoff {
            out.push(*ev);
        }
    }
    out
}
