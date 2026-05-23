//! Strategy-agnostic per-turn world snapshot.
//!
//! Built once per `Bot::compute_moves` call after the [`EntityCache`] is
//! refreshed for the current step. Strategies receive `&WorldState` and read
//! whichever pieces they need — observation snapshots, planet groupings by
//! owner, aggregate strength/production, the `TimelineCache`, and per-planet
//! timeline-derived maps (`keep_needed_map`, `fall_turn_map`, …).
//!
//! Anything strategy-flavoured (phase windows like "early"/"opening", scoring
//! heuristics, solver memoization) belongs in the strategy module, not here.

#![allow(dead_code)]

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::constants::{EPISODE_STEPS, HORIZON};
use crate::engine::{CometGroup, Configuration, EngineState, Fleet, Planet};
use crate::entity_cache::EntityCache;
use crate::helpers::{
    count_players, simulate_planet_timeline, state_at_timeline, ArrivalEvent, PlanetTimeline,
    TimelineCache,
};

/// Per-turn world snapshot. Strategy code borrows this and never mutates it.
pub struct WorldState<'a> {
    pub player: i64,
    pub step: i64,
    pub angular_velocity: f64,
    pub entity_cache: &'a EntityCache,
    pub timeline_cache: TimelineCache,

    pub planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub planet_by_id: HashMap<i64, Planet>,
    pub comet_ids: HashSet<i64>,

    pub my_planets: Vec<Planet>,
    pub enemy_planets: Vec<Planet>,
    pub neutral_planets: Vec<Planet>,
    pub static_neutral_planets: Vec<Planet>,

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

    pub keep_needed_map: HashMap<i64, i64>,
    pub min_owned_map: HashMap<i64, i64>,
    pub first_enemy_map: HashMap<i64, Option<i64>>,
    pub fall_turn_map: HashMap<i64, Option<i64>>,
    pub holds_full_map: HashMap<i64, bool>,
}

impl<'a> WorldState<'a> {
    /// Build the per-turn snapshot from the parsed observation + cache.
    /// `entity_cache.current_turn` should already be set to `step`.
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
        entity_cache: &'a EntityCache,
    ) -> Self {
        let num_players = count_players(&planets, &fleets);
        let next_fleet_id = fleets.iter().map(|f| f.id).max().map(|m| m + 1).unwrap_or(0);
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
            Configuration::default(),
        );
        Self::from_engine(player, &engine, entity_cache)
    }

    /// Fast-path constructor for callers that already have an `EngineState`
    /// (the rollout / search loop). Skips the EngineState reconstruction and
    /// the two-stage Vec clones that `build` does when called with owned
    /// observation parts.
    pub fn from_engine(
        player: i64,
        engine: &EngineState,
        entity_cache: &'a EntityCache,
    ) -> Self {
        let step = engine.step;
        let angular_velocity = engine.angular_velocity;
        let num_players = engine.num_players;
        let timeline_cache = TimelineCache::build(engine, player, HORIZON, entity_cache);

        // Clone the planets/fleets/comet ids once for the owned WorldState
        // fields (vs. twice when the caller routed through `build`).
        let planets: Vec<Planet> = engine.planets.clone();
        let fleets: Vec<Fleet> = engine.fleets.clone();
        let comet_planet_ids: Vec<i64> = engine.comet_planet_ids.clone();

        let mut planet_by_id: HashMap<i64, Planet> = HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        for planet in &planets {
            planet_by_id.insert(planet.id, planet.clone());
        }
        let comet_ids: HashSet<i64> = comet_planet_ids.iter().copied().collect();

        let mut my_planets = Vec::new();
        let mut enemy_planets = Vec::new();
        let mut neutral_planets = Vec::new();
        let mut static_neutral_planets = Vec::new();
        for planet in &planets {
            if planet.owner == player {
                my_planets.push(planet.clone());
            } else if planet.owner == -1 {
                neutral_planets.push(planet.clone());
                if entity_cache
                    .get(planet.id)
                    .map(|e| e.is_static())
                    .unwrap_or(false)
                {
                    static_neutral_planets.push(planet.clone());
                }
            } else {
                enemy_planets.push(planet.clone());
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

        let mut keep_needed_map = HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        let mut min_owned_map = HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        let mut first_enemy_map = HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        let mut fall_turn_map = HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        let mut holds_full_map = HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        for planet in &planets {
            if let Some(baseline) = timeline_cache.baseline(planet.id) {
                keep_needed_map.insert(planet.id, baseline.keep_needed);
                min_owned_map.insert(planet.id, baseline.min_owned);
                first_enemy_map.insert(planet.id, baseline.first_enemy);
                fall_turn_map.insert(planet.id, baseline.fall_turn);
                holds_full_map.insert(planet.id, baseline.holds_full);
            } else {
                keep_needed_map.insert(planet.id, 0);
                min_owned_map.insert(planet.id, 0);
                first_enemy_map.insert(planet.id, None);
                fall_turn_map.insert(planet.id, None);
                holds_full_map.insert(planet.id, true);
            }
        }

        let total_visible_ships: i64 = planets.iter().map(|p| p.ships).sum::<i64>()
            + fleets.iter().map(|f| f.ships).sum::<i64>();
        let total_production: i64 = planets.iter().map(|p| p.production).sum();

        Self {
            player,
            step,
            angular_velocity,
            entity_cache,
            timeline_cache,
            planets,
            fleets,
            planet_by_id,
            comet_ids,
            my_planets,
            enemy_planets,
            neutral_planets,
            static_neutral_planets,
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
            keep_needed_map,
            min_owned_map,
            first_enemy_map,
            fall_turn_map,
            holds_full_map,
        }
    }

    #[inline]
    pub fn planet(&self, id: i64) -> &Planet {
        &self.planet_by_id[&id]
    }

    pub fn is_static(&self, planet_id: i64) -> bool {
        self.entity_cache
            .get(planet_id)
            .map(|e| e.is_static())
            .unwrap_or(false)
    }

    pub fn comet_life(&self, planet_id: i64) -> i64 {
        self.entity_cache.remaining_life(planet_id)
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
        let merged = merge_arrivals(self.timeline_cache.arrivals(target_id), planned, extra, cutoff);
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
        let merged = merge_arrivals(self.timeline_cache.arrivals(target_id), planned, extra, horizon);
        let target = self.planet(target_id);
        let expiry = self.timeline_cache.expiry(target_id);
        simulate_planet_timeline(target, &merged, self.player, horizon, expiry)
    }

    pub fn hold_status(&self, target_id: i64, planned: &[ArrivalEvent], horizon: i64) -> HoldStatus {
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
    let mut out: Vec<ArrivalEvent> =
        Vec::with_capacity(base.len() + planned.len() + extra.len());
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
