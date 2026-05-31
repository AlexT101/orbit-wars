//! The Python wrapper (`main.py`) instantiates one [`Bot`] at import time and
//! forwards every observation through `Bot::compute_moves`.

mod blockers;
mod constants;
mod engine;
mod entity_cache;
mod helpers;
mod hellburner;
mod rollout;
mod world;

#[cfg(test)]
mod tests;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use crate::constants::{COMET_SPAWN_STEPS, HORIZON, TOTAL_OVERAGE_TIME};
use crate::engine::{CometGroup, EngineState, Fleet, Planet, Simulator};
use crate::entity_cache::EntityCache;
use crate::helpers::ArrivalLedger;
use crate::rollout::pick_plan_by_rollout;
use crate::world::WorldState;

#[allow(dead_code)]
struct Observation {
    player: i64,
    planets: Vec<Planet>,
    fleets: Vec<Fleet>,
    initial_planets: Vec<Planet>,
    comets: Vec<CometGroup>,
    comet_planet_ids: Vec<i64>,
    angular_velocity: f64,
    remaining_overage_time: f64,
}

fn get_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("obs missing field '{}'", key)))
}

fn parse_planets(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Planet>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let row = seq.get_item(i)?;
        out.push(Planet {
            id: row.get_item(0)?.extract()?,
            owner: row.get_item(1)?.extract()?,
            x: row.get_item(2)?.extract()?,
            y: row.get_item(3)?.extract()?,
            radius: row.get_item(4)?.extract()?,
            ships: row.get_item(5)?.extract()?,
            production: row.get_item(6)?.extract()?,
        });
    }
    Ok(out)
}

fn parse_fleets(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Fleet>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let row = seq.get_item(i)?;
        out.push(Fleet {
            id: row.get_item(0)?.extract()?,
            owner: row.get_item(1)?.extract()?,
            x: row.get_item(2)?.extract()?,
            y: row.get_item(3)?.extract()?,
            angle: row.get_item(4)?.extract()?,
            from_planet_id: row.get_item(5)?.extract()?,
            ships: row.get_item(6)?.extract()?,
        });
    }
    Ok(out)
}

fn parse_path(seq: &Bound<'_, PyAny>) -> PyResult<Vec<[f64; 2]>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let row = seq.get_item(i)?;
        let x: f64 = row.get_item(0)?.extract()?;
        let y: f64 = row.get_item(1)?.extract()?;
        out.push([x, y]);
    }
    Ok(out)
}

fn parse_comets(seq: &Bound<'_, PyAny>) -> PyResult<Vec<CometGroup>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        let dict = item.downcast::<PyDict>()?;
        let planet_ids: Vec<i64> = get_item(dict, "planet_ids")?.extract()?;
        let paths_any = get_item(dict, "paths")?;
        let paths_seq: Bound<'_, PySequence> = paths_any.downcast::<PySequence>()?.clone();
        let mut paths: Vec<Vec<[f64; 2]>> = Vec::with_capacity(paths_seq.len()?);
        for j in 0..paths_seq.len()? {
            paths.push(parse_path(&paths_seq.get_item(j)?)?);
        }
        let path_index: i64 = get_item(dict, "path_index")?.extract()?;
        out.push(CometGroup {
            planet_ids,
            paths,
            path_index,
        });
    }
    Ok(out)
}

impl Observation {
    fn from_dict(obs: &Bound<'_, PyDict>) -> PyResult<Self> {
        let player: i64 = get_item(obs, "player")?.extract()?;
        let planets = parse_planets(&get_item(obs, "planets")?)?;
        let fleets = parse_fleets(&get_item(obs, "fleets")?)?;
        let initial_planets = parse_planets(&get_item(obs, "initial_planets")?)?;
        let comets = parse_comets(&get_item(obs, "comets")?)?;
        let comet_planet_ids: Vec<i64> =
            get_item(obs, "comet_planet_ids")?.extract()?;
        let angular_velocity: f64 = get_item(obs, "angular_velocity")?.extract()?;
        let remaining_overage_time: f64 = match obs.get_item("remainingOverageTime")? {
            Some(v) => v.extract().unwrap_or(TOTAL_OVERAGE_TIME),
            None => TOTAL_OVERAGE_TIME,
        };
        Ok(Self {
            player,
            planets,
            fleets,
            initial_planets,
            comets,
            comet_planet_ids,
            angular_velocity,
            remaining_overage_time,
        })
    }
}

#[pyclass]
pub struct Bot {
    current_turn: i64,
    cache: Option<EntityCache>,
}

#[pymethods]
impl Bot {
    #[new]
    fn new() -> Self {
        Self {
            current_turn: 0,
            cache: None,
        }
    }

    #[getter]
    fn current_turn(&self) -> i64 {
        self.current_turn
    }

    fn compute_moves(
        &mut self,
        obs: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<(i64, f64, i64)>> {
        let obs = Observation::from_dict(obs)?;
        self.refresh_cache(&obs);
        let cache = self.cache.as_ref().expect("entity cache populated above");

        let mut world = WorldState::build(
            obs.player,
            self.current_turn,
            obs.planets,
            obs.fleets,
            obs.initial_planets,
            obs.comets,
            obs.comet_planet_ids,
            obs.angular_velocity,
            cache,
        );
        world.remaining_overage_time = obs.remaining_overage_time;

        let moves = crate::hellburner::plan(&world);
        self.current_turn += 1;
        Ok(moves.into_iter().map(|m| (m.from_id, m.angle, m.ships)).collect())
    }

    /// Plan with rollout-based multi-candidate selection. Costs ~5-10x more
    /// than `compute_moves` but rejects plans that lose to a modeled opponent.
    fn compute_moves_with_search(
        &mut self,
        obs: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<(i64, f64, i64)>> {
        let obs = Observation::from_dict(obs)?;
        self.refresh_cache(&obs);

        // Build engine state once; reused for candidate WorldState and rollout seed.
        // NOTE: next_fleet_id may recycle destroyed fleets' IDs since we only
        // see currently-visible fleets. Safe while no consumer keys on fleet
        // ID across turns; revisit if any cache/hash ever does.
        let next_fleet_id = obs.fleets.iter().map(|f| f.id).max().map(|m| m + 1).unwrap_or(0);
        let num_players = crate::helpers::count_players(&obs.planets, &obs.fleets);
        let player = obs.player;
        let initial_state = EngineState::from_observation_parts(
            self.current_turn,
            obs.angular_velocity,
            obs.planets,
            obs.initial_planets,
            obs.fleets,
            next_fleet_id,
            obs.comet_planet_ids,
            obs.comets,
            num_players,
        );

        // The turn-0 arrival ledger is player-agnostic, so build it once here and
        // reuse it both for candidate generation and for the opponent turn-0
        // modelling inside the rollout — saving a redundant HORIZON-turn walk.
        // Built in a block so the WorldState's immutable borrow on the cache
        // ends before the rollout reborrows it mutably.
        let (candidates, initial_ledger) = {
            let cache_ref = self.cache.as_ref().expect("entity cache populated above");
            let initial_sim = Simulator::new(&initial_state);
            let ledger = ArrivalLedger::build(&initial_sim, HORIZON, cache_ref);
            let mut world =
                WorldState::from_simulator_with_ledger(player, &initial_sim, &ledger, cache_ref);
            world.remaining_overage_time = obs.remaining_overage_time;
            (crate::hellburner::search_candidates(&world), ledger)
        };

        let cache_mut = self.cache.as_mut().expect("entity cache populated above");
        let moves = pick_plan_by_rollout(
            &initial_state,
            player,
            candidates,
            crate::hellburner::plan,
            crate::hellburner::search_candidates,
            cache_mut,
            obs.remaining_overage_time,
            Some(&initial_ledger),
        );
        self.current_turn += 1;
        Ok(moves.into_iter().map(|m| (m.from_id, m.angle, m.ships)).collect())
    }
}

impl Bot {
    fn refresh_cache(&mut self, obs: &Observation) {
        match &mut self.cache {
            None => {
                self.cache = Some(EntityCache::build(
                    &obs.initial_planets,
                    &obs.comets,
                    &obs.comet_planet_ids,
                    obs.angular_velocity,
                    self.current_turn,
                ));
            }
            Some(cache) if COMET_SPAWN_STEPS.contains(&self.current_turn) => {
                cache.refresh_comets(&obs.comets, &obs.comet_planet_ids, self.current_turn);
            }
            _ => {}
        }
        if let Some(cache) = &mut self.cache {
            cache.set_current_turn(self.current_turn);
            // Drop the prior turn's aim entries; rollout forward-sim slots
            // age out the same way past `current_turn`.
            cache.clear_aim_cache_slot(self.current_turn - 1);
        }
    }
}

#[pymodule]
fn apollo_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bot>()?;
    Ok(())
}

