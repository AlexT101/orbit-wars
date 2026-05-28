//! The Python wrapper (`main.py`) instantiates one [`Bot`] at import time and
//! forwards every observation through `Bot::compute_moves`.

mod blockers;
mod constants;
mod engine;
mod entity_cache;
mod helpers;
mod hellburner;
mod rollout;
mod sim_probe;
mod world;

#[cfg(test)]
mod tests;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use crate::constants::COMET_SPAWN_STEPS;
use crate::engine::{CometGroup, Configuration, EngineState, Fleet, Planet};
use crate::entity_cache::EntityCache;
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
        Ok(Self {
            player,
            planets,
            fleets,
            initial_planets,
            comets,
            comet_planet_ids,
            angular_velocity,
        })
    }
}

#[pyclass]
pub struct Bot {
    current_turn: i64,
    cache: Option<EntityCache>,
    /// Pending debug strings emitted during the last `compute_moves*` call.
    /// Python drains this via `take_debug()` and prints each line so the
    /// viewer can overlay `[LINE]`/`[DOT]` indicators (parsed by the runner
    /// and surfaced as `replay.debug.messages`).
    debug: Vec<String>,
}

#[pymethods]
impl Bot {
    #[new]
    fn new() -> Self {
        Self {
            current_turn: 0,
            cache: None,
            debug: Vec::new(),
        }
    }

    #[getter]
    fn current_turn(&self) -> i64 {
        self.current_turn
    }

    /// Drain debug lines emitted by the last planning call. Python prints
    /// these to stdout where the runner picks them up.
    fn take_debug(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug)
    }

    fn compute_moves(
        &mut self,
        obs: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<(i64, f64, i64)>> {
        self.debug.clear();
        let obs = Observation::from_dict(obs)?;
        self.refresh_cache(&obs);
        let cache = self.cache.as_ref().expect("entity cache populated above");

        let world = WorldState::build(
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

        let moves = crate::hellburner::plan(&world);
        self.current_turn += 1;
        Ok(moves)
    }

    /// Plan with rollout-based multi-candidate selection. Costs ~5-10x more
    /// than `compute_moves` but rejects plans that lose to a modeled opponent.
    fn compute_moves_with_search(
        &mut self,
        obs: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<(i64, f64, i64)>> {
        self.debug.clear();
        let obs = Observation::from_dict(obs)?;
        self.refresh_cache(&obs);

        // Construct the engine state once and reuse it for both the candidate
        // WorldState and the rollout seed — avoids parsing/cloning the
        // observation vecs a second time.
        // NOTE: this recycles IDs of destroyed fleets — Kaggle's engine issues
        // monotonically increasing IDs across the whole game, but we only see
        // currently-visible fleets. Safe today because no consumer keys on
        // fleet ID across turns; revisit if any cache/hash ever does.
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
            Configuration::default(),
        );

        // Cache planet (id -> (x, y)) so we can render candidate moves from
        // their source planet to the direction they aim.
        let planet_xy: std::collections::HashMap<i64, (f64, f64)> = initial_state
            .planets
            .iter()
            .map(|p| (p.id, (p.x, p.y)))
            .collect();

        // Plan candidates inside a block so the WorldState's immutable borrow
        // on the cache ends before the rollout reborrows it mutably.
        let candidates = {
            let cache_ref = self.cache.as_ref().expect("entity cache populated above");
            let world = WorldState::from_engine(player, &initial_state, cache_ref);
            crate::hellburner::search_candidates(&world)
        };

        // Snapshot candidates for debug rendering before we hand them off to
        // the rollout (which consumes the Vec).
        let candidate_snapshot: Vec<Vec<(i64, f64, i64)>> = candidates.clone();

        let cache_mut = self.cache.as_mut().expect("entity cache populated above");
        let moves = pick_plan_by_rollout(
            &initial_state,
            player,
            candidates,
            crate::hellburner::plan,
            crate::hellburner::search_candidates,
            cache_mut,
        );

        // Identify which snapshot candidate matches the chosen plan so we can
        // highlight it; rollout returns the actual selected variant.
        let chosen_idx = candidate_snapshot
            .iter()
            .position(|c| c == &moves)
            .unwrap_or(usize::MAX);

        self.emit_candidate_debug(&candidate_snapshot, chosen_idx, &planet_xy, &moves);

        self.current_turn += 1;
        Ok(moves)
    }
}

impl Bot {
    /// Emit `[LINE]` debug messages describing every candidate plan the
    /// search considered. Rejected candidates render dim/translucent (the
    /// frontend reads the color hex); the chosen plan renders in a brighter
    /// hue so it stands out. Each line starts at the source planet and
    /// extends a fixed length along the launch angle — enough to show
    /// direction without overlapping the rest of the board.
    fn emit_candidate_debug(
        &mut self,
        candidates: &[Vec<(i64, f64, i64)>],
        chosen_idx: usize,
        planet_xy: &std::collections::HashMap<i64, (f64, f64)>,
        chosen: &[(i64, f64, i64)],
    ) {
        // Length of the candidate-direction ray in board units.
        const RAY_LEN: f64 = 8.0;

        // Color choice: chosen plan stands out, others are dimmer variants.
        // Hex format keeps the parsing on the frontend trivial.
        const CHOSEN_COLOR: &str = "#00ff88";   // bright green
        const REJECTED_COLOR: &str = "#444466"; // dim slate

        // --- Turn header ---
        self.debug.push(format!(
            "[TEXT] === turn {} ===",
            self.current_turn,
        ));
        self.debug.push(format!(
            "[TEXT] search produced {} candidate plan{}, chose #{}",
            candidates.len(),
            if candidates.len() == 1 { "" } else { "s" },
            if chosen_idx == usize::MAX {
                "?".to_string()
            } else {
                chosen_idx.to_string()
            },
        ));

        // --- Per-candidate summary + the lines/dots overlay ---
        for (idx, plan) in candidates.iter().enumerate() {
            let is_chosen = idx == chosen_idx;
            let marker = if is_chosen { "✓" } else { " " };
            let total_ships: i64 = plan.iter().map(|(_, _, s)| *s).sum();
            self.debug.push(format!(
                "[TEXT] {} cand #{}: {} move{} · {} ships total",
                marker,
                idx,
                plan.len(),
                if plan.len() == 1 { "" } else { "s" },
                total_ships,
            ));
            for &(src_id, angle, ships) in plan {
                // Per-move text log so the panel shows what each candidate
                // actually wants to do — `from planet X · N ships · θ rad`.
                self.debug.push(format!(
                    "[TEXT]    from p{} · {} ships · angle={:.2} rad",
                    src_id, ships, angle,
                ));

                // Visual overlay for the same move.
                let Some(&(x, y)) = planet_xy.get(&src_id) else {
                    continue;
                };
                let x2 = x + angle.cos() * RAY_LEN;
                let y2 = y + angle.sin() * RAY_LEN;
                let color = if is_chosen { CHOSEN_COLOR } else { REJECTED_COLOR };
                self.debug.push(format!(
                    "[LINE] {:.3} {:.3} {:.3} {:.3} {}",
                    x, y, x2, y2, color
                ));
                if is_chosen {
                    // Small dot at the chosen plan's source so it's clear
                    // which planet originated the shot.
                    self.debug.push(format!(
                        "[DOT] {:.3} {:.3} 0.8 {}",
                        x, y, CHOSEN_COLOR
                    ));
                }
            }
        }

        if chosen.is_empty() {
            self.debug.push(
                "[TEXT] chosen plan: pass (no launches this turn)".to_string(),
            );
        }
    }

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
            // Drop the prior bot turn's aim entries. Slots from rollout
            // forward-sim of earlier turns are released the same way as they
            // age past `current_turn`.
            cache.clear_aim_cache_slot(self.current_turn - 1);
        }
    }
}

#[pymodule]
fn apollo_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bot>()?;
    Ok(())
}

