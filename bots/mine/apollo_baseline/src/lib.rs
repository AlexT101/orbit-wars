//! The Python wrapper (`main.py`) instantiates one [`Bot`] at import time and
//! forwards every observation through `Bot::compute_moves_with_search`.

mod aim;
mod cache;
mod constants;
mod early_game;
mod engine;
mod helpers;
mod rollout;
mod strategy;
mod world;

#[cfg(test)]
mod tests;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use crate::cache::EntityCache;
use crate::constants::{Config, COMET_SPAWN_STEPS, TOTAL_OVERAGE_TIME};
use crate::engine::{CometGroup, EngineState, Fleet, Planet, Simulator};
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

/// Extract an integer field, tolerating whole-number floats.
///
/// The Kaggle engine sanitizes a move's `ships` to `int(...)` but stores some
/// id fields verbatim (`orbit_wars.py`). An opponent who submits a move with a
/// float planet id (e.g. `33.0`) therefore produces an observation whose id
/// field is a float, which a plain `extract::<i64>()` rejects with "'float'
/// object cannot be interpreted as an integer", crashing our agent on its next
/// turn. Falling back to an `f64` extraction makes parsing robust to such
/// poisoned fields.
fn extract_i64(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    match v.extract::<i64>() {
        Ok(i) => Ok(i),
        Err(_) => Ok(v.extract::<f64>()?.round() as i64),
    }
}

fn parse_planets(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Planet>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let row = seq.get_item(i)?;
        out.push(Planet {
            id: extract_i64(&row.get_item(0)?)?,
            owner: extract_i64(&row.get_item(1)?)?,
            x: row.get_item(2)?.extract()?,
            y: row.get_item(3)?.extract()?,
            radius: row.get_item(4)?.extract()?,
            ships: extract_i64(&row.get_item(5)?)?,
            production: extract_i64(&row.get_item(6)?)?,
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
            id: extract_i64(&row.get_item(0)?)?,
            owner: extract_i64(&row.get_item(1)?)?,
            x: row.get_item(2)?.extract()?,
            y: row.get_item(3)?.extract()?,
            angle: row.get_item(4)?.extract()?,
            // Column 5 is the fleet's `from_planet_id` (the source planet of the
            // launch). The planner never reads it, so it is parsed-and-dropped.
            ships: extract_i64(&row.get_item(6)?)?,
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
        let path_index: i64 = extract_i64(&get_item(dict, "path_index")?)?;
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
        let player: i64 = extract_i64(&get_item(obs, "player")?)?;
        let planets = parse_planets(&get_item(obs, "planets")?)?;
        let fleets = parse_fleets(&get_item(obs, "fleets")?)?;
        let initial_planets = parse_planets(&get_item(obs, "initial_planets")?)?;
        let comets = parse_comets(&get_item(obs, "comets")?)?;
        let comet_planet_ids: Vec<i64> = get_item(obs, "comet_planet_ids")?.extract()?;
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

    fn compute_moves(&mut self, obs: &Bound<'_, PyDict>) -> PyResult<Vec<(i64, f64, i64)>> {
        let obs = Observation::from_dict(obs)?;
        self.refresh_cache(&obs);
        // Step-scoped L1 aim cache, shared across every model built this step.
        // Declared before `world` so it outlives the borrow `world` takes on it.
        let shot_l1 = crate::world::ShotL1::default();
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
        world.shot_l1 = Some(&shot_l1);

        let moves = crate::strategy::plan(&world);
        let moves = crate::strategy::redirect_moves(&world, moves);
        self.current_turn += 1;
        Ok(moves
            .into_iter()
            .map(|m| (m.from_id, m.angle, m.ships))
            .collect())
    }

    fn compute_moves_with_search(
        &mut self,
        obs: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<(i64, f64, i64)>> {
        let obs = Observation::from_dict(obs)?;
        self.refresh_cache(&obs);
        let shot_l1 = crate::world::ShotL1::default();

        // Build engine state once; reused for candidate WorldState and rollout seed.
        // NOTE: next_fleet_id may recycle destroyed fleets' IDs since we only
        // see currently-visible fleets. Safe while no consumer keys on fleet
        // ID across turns; revisit if any cache/hash ever does.
        let next_fleet_id = obs
            .fleets
            .iter()
            .map(|f| f.id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
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
            let config = Config::for_alive(crate::helpers::count_alive_players(
                initial_sim.planets(),
                initial_sim.fleets(),
            ));
            let ledger = ArrivalLedger::build(&initial_sim, config.horizon, cache_ref);
            let mut world =
                WorldState::from_simulator_with_ledger(player, &initial_sim, &ledger, cache_ref);
            world.remaining_overage_time = obs.remaining_overage_time;
            world.shot_l1 = Some(&shot_l1);
            (crate::strategy::search_candidates(&world), ledger)
        };

        let cache_mut = self.cache.as_mut().expect("entity cache populated above");
        let moves = pick_plan_by_rollout(
            &initial_state,
            player,
            candidates,
            crate::strategy::plan,
            crate::strategy::search_candidates,
            cache_mut,
            obs.remaining_overage_time,
            Some(&initial_ledger),
            Some(&shot_l1),
        );

        // Final reroute pass on the chosen plan only — after the rollout has
        // scored the untouched policy. Rebuild a WorldState (the rollout's
        // mutable cache borrow has ended) so `redirect_moves` can re-derive
        // travel times and project intermediate planets' ownership.
        let moves = {
            let cache_ref = self.cache.as_ref().expect("entity cache populated above");
            let final_sim = Simulator::new(&initial_state);
            let mut world = WorldState::from_simulator_with_ledger(
                player,
                &final_sim,
                &initial_ledger,
                cache_ref,
            );
            world.remaining_overage_time = obs.remaining_overage_time;
            world.shot_l1 = Some(&shot_l1);
            crate::strategy::redirect_moves(&world, moves)
        };

        self.current_turn += 1;
        Ok(moves
            .into_iter()
            .map(|m| (m.from_id, m.angle, m.ships))
            .collect())
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

/// Cache-build `current_step` for a benchmark obs. The obs planets are the
/// snapshot at `obs["step"]`; treated as the cache's `initial_planets`, that
/// snapshot already sits at the cache's turn-1 slot, so `current_step = 1`
/// reproduces the engine's per-turn positions for any mid-game step (and keeps
/// the full look-ahead horizon, unlike building at the real step near game end).
///
/// The sole exception is **game step 0**: the engine's launch tick does not
/// rotate planets (the game-start seam, `current_angle = init + ω·0`), so the
/// obs planets are the true initial positions and the target is static on the
/// first flight turn. `current_step = 0` engages apollo's matching
/// `positions[0] == positions[1]` seam so the binding reproduces that. Defaults
/// to 1 when `step` is absent (pure-geometry use).
fn obs_current_step(obs: &Bound<'_, PyDict>) -> PyResult<i64> {
    let step = match obs.get_item("step")? {
        Some(s) => extract_i64(&s)?,
        None => return Ok(1),
    };
    Ok(if step == 0 { 0 } else { 1 })
}

/// True iff the obs carries a `player` field and the `source` planet is not
/// owned by that player (or is absent). Used by the benchmark bindings to
/// decline launches the engine could never spawn a fleet for. Returns `false`
/// (don't decline) when `player` isn't present, preserving pure-geometry use.
fn source_unowned(planets: &[Planet], source: i64, obs: &Bound<'_, PyDict>) -> PyResult<bool> {
    let Some(player) = obs.get_item("player")? else {
        return Ok(false);
    };
    let player = extract_i64(&player)?;
    let owned = planets.iter().any(|p| p.id == source && p.owner == player);
    Ok(!owned)
}

/// Standalone aim entry point for the Kaggle aim benchmark
/// (`benchmark-for-aiming-implementation.ipynb`).
///
/// The benchmark hands us a *current* board snapshot and asks for the launch
/// angle from `source` to `target` with `fleet_size` ships, or `None` if the
/// shot can't connect. It only populates the fields the example aimer reads
/// (`planets`, `angular_velocity`, optionally `comets`/`comet_planet_ids`), so
/// this deliberately does **not** go through [`Observation::from_dict`], which
/// requires `player`/`initial_planets`/`fleets`/etc.
///
/// The snapshot's planets are fed to [`EntityCache::build`] as `initial_planets`
/// with `current_step = 1`. [`crate::cache::build_planet_entity`] uses
/// `effective = (t - 1).max(0)`, so `positions[1]` is the observed position and
/// `positions[1 + t]` is that position rotated by `ω·t` — the exact, seam-free
/// model the engine uses mid-game (`current_step = 0` would under-rotate the
/// first lead turn through the game-start 0/1 seam). Comet `path_index` already
/// locates "now", so querying at `launch_turn_offset = 0` reads the present and
/// offset `t` reads `t` turns ahead for every entity kind.
#[pyfunction]
#[pyo3(signature = (obs, source, target, fleet_size))]
fn aim_angle(
    obs: &Bound<'_, PyDict>,
    source: i64,
    target: i64,
    fleet_size: i64,
) -> PyResult<Option<f64>> {
    let planets = parse_planets(&get_item(obs, "planets")?)?;
    let angular_velocity: f64 = get_item(obs, "angular_velocity")?.extract()?;

    // The engine only spawns a fleet from a planet the launching player owns; an
    // unowned source can never hit (no fleet is created), so decline. This
    // mirrors the ownership filter the strategy applies before it ever calls the
    // aimer in a real game — the geometry solver itself is never handed an
    // unowned source. Only enforced when `player` is present so a pure-geometry
    // obs still works.
    if source_unowned(&planets, source, obs)? {
        return Ok(None);
    }

    // Comets are optional in the benchmark obs — default to none.
    let (comets, comet_planet_ids) =
        match (obs.get_item("comets")?, obs.get_item("comet_planet_ids")?) {
            (Some(c), Some(ids)) => (parse_comets(&c)?, ids.extract::<Vec<i64>>()?),
            _ => (Vec::new(), Vec::new()),
        };

    let cache = EntityCache::build(
        &planets,
        &comets,
        &comet_planet_ids,
        angular_velocity,
        obs_current_step(obs)?,
    );

    Ok(
        crate::aim::aim_with_prediction(&cache, source, target, fleet_size, 0)
            .map(|(angle, _turns, _tx, _ty, _flight_time)| angle),
    )
}

#[pymodule]
fn apollo_baseline_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bot>()?;
    m.add_function(wrap_pyfunction!(aim_angle, m)?)?;
    Ok(())
}
