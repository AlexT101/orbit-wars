//! Strategy-agnostic per-turn world snapshot.
//!
//! Built once per `compute_moves` call after the [`EntityCache`] is refreshed
//! for the current step. Strategies receive `&WorldState` and read whichever
//! pieces they need — observation snapshots, planet groupings by owner, and the
//! `TimelineCache`. Per-planet timeline metrics are accessed via
//! `ws.timeline_cache.baseline(planet_id)`.
//!
//! Anything strategy-flavoured (phase windows like "early"/"opening", scoring
//! heuristics, solver memoization) belongs in the strategy module, not here.

use std::cell::RefCell;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::aim::AimResult;
use crate::cache::EntityCache;
use crate::constants::Config;
use crate::engine::{CometGroup, EngineState, Fleet, Planet, Simulator};

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
use crate::helpers::{
    count_alive_players, count_players, simulate_planet_timeline, ArrivalEvent, ArrivalLedger,
    PlanetTimeline, TimelineCache,
};

/// Per-turn world snapshot. Strategy code borrows this and never mutates it.
pub struct WorldState<'a> {
    pub player: i64,
    pub cache: &'a EntityCache,
    /// Optional step-scoped L1 aim cache (see [`ShotL1`]). `None` for tests and
    /// ad-hoc worlds, which fall back to each `HellburnerModel`'s own per-model
    /// cache. Set by the live `Bot` paths so all models in a step share hits.
    pub shot_l1: Option<&'a ShotL1>,
    pub timeline_cache: TimelineCache,

    pub planets: Vec<Planet>,
    /// `planet_id → index into `planets`. Avoids a second full copy of every
    /// planet; look up positional data via [`Self::planet`].
    pub planet_by_id: HashMap<i64, usize>,
    pub comet_ids: HashSet<i64>,

    pub my_planets: Vec<Planet>,
    pub enemy_planets: Vec<Planet>,

    /// Runtime tuning profile selected from the alive-player count of this snapshot.
    pub config: Config,

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
        let config = Config::for_alive(count_alive_players(sim.planets(), sim.fleets()));
        let ledger = ArrivalLedger::build(&sim, config.horizon, cache);
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
        let timeline_cache = TimelineCache::from_ledger(sim.planets(), player, ledger);
        let config = Config::for_alive(count_alive_players(sim.planets(), sim.fleets()));

        let planets: Vec<Planet> = sim.planets().to_vec();
        let comet_planet_ids: Vec<i64> = sim.comet_planet_ids().to_vec();

        let mut planet_by_id: HashMap<i64, usize> =
            HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        for (idx, planet) in planets.iter().enumerate() {
            planet_by_id.insert(planet.id, idx);
        }
        let comet_ids: HashSet<i64> = comet_planet_ids.iter().copied().collect();

        let mut my_planets = Vec::new();
        let mut enemy_planets = Vec::new();
        for planet in &planets {
            if planet.owner == player {
                my_planets.push(*planet);
            } else if planet.owner != -1 {
                enemy_planets.push(*planet);
            }
        }

        Self {
            player,
            cache,
            shot_l1: None,
            timeline_cache,
            planets,
            planet_by_id,
            comet_ids,
            my_planets,
            enemy_planets,
            config,
            remaining_overage_time: 0.0,
        }
    }

    #[inline]
    pub fn planet(&self, id: i64) -> &Planet {
        &self.planets[self.planet_by_id[&id]]
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
}

/// Merge base arrivals + planned commitments + extra arrivals into one
/// arrival list (unsorted — `simulate_planet_timeline` re-normalizes), dropping
/// anything past `cutoff`.
fn merge_arrivals(
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
