//! Orbit Wars forward model.
//!
//! Same physics as `rust_engine`, minus: planet generation, comet spawning, and
//! the seeded RNG. Caller provides a state dict (typically lifted from a kaggle
//! observation) via `set_state`, then calls `step` to advance one turn.
//!
//! Comet spawn boundaries (steps 50/150/250/350/450) are NOT simulated — we
//! don't have access to the kaggle seed used for comet RNG. The validation
//! script skips these step transitions.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence};

pub mod features;

const BOARD_SIZE: f64 = 100.0;
const CENTER: f64 = BOARD_SIZE / 2.0;
const SUN_RADIUS: f64 = 10.0;
const ROTATION_RADIUS_LIMIT: f64 = 50.0;
const MAX_PLAYERS: usize = 4;

// NOTE: these types and the `EngineState` forward model are `pub` so other
// crates in the workspace can depend on `orbit_wars_model` as an rlib and reuse
// the physics directly (e.g. the `graph` bot builds an `EngineState` from an
// observation to drive its forward model). The PyO3 surface below is unchanged.
#[derive(Clone, Debug)]
pub struct Planet {
    pub id: i64,
    pub owner: i64,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i64,
    pub production: i64,
}

impl Planet {
    fn as_tuple(&self) -> (i64, i64, f64, f64, f64, i64, i64) {
        (
            self.id,
            self.owner,
            self.x,
            self.y,
            self.radius,
            self.ships,
            self.production,
        )
    }
}

#[derive(Clone, Debug)]
pub struct Fleet {
    pub id: i64,
    pub owner: i64,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i64,
    pub ships: i64,
}

impl Fleet {
    fn as_tuple(&self) -> (i64, i64, f64, f64, f64, i64, i64) {
        (
            self.id,
            self.owner,
            self.x,
            self.y,
            self.angle,
            self.from_planet_id,
            self.ships,
        )
    }
}

#[derive(Clone, Debug)]
pub struct CometGroup {
    pub planet_ids: Vec<i64>,
    pub paths: Vec<Vec<[f64; 2]>>,
    pub path_index: i64,
}

#[derive(Clone, Debug)]
pub struct Configuration {
    pub episode_steps: i64,
    pub ship_speed: f64,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            episode_steps: 500,
            ship_speed: 6.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MoveAction {
    pub from_id: i64,
    pub angle: f64,
    pub ships: i64,
}

#[derive(Clone, Debug)]
pub struct EngineState {
    pub step: i64,
    pub angular_velocity: f64,
    pub planets: Vec<Planet>,
    pub initial_planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub next_fleet_id: i64,
    pub comet_planet_ids: Vec<i64>,
    pub comets: Vec<CometGroup>,
    pub done: bool,
    pub num_players: usize,
    pub configuration: Configuration,
    pub planet_index_by_id: HashMap<i64, usize>,
}

#[derive(Clone, Copy, Debug)]
struct PlanetPath {
    old_pos: (f64, f64),
    new_pos: (f64, f64),
    check_collision: bool,
}

impl EngineState {
    /// Build a forward-model state from raw parts (the shape a caller gets after
    /// parsing a kaggle observation). The planet index is built for you.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        step: i64,
        angular_velocity: f64,
        planets: Vec<Planet>,
        initial_planets: Vec<Planet>,
        fleets: Vec<Fleet>,
        next_fleet_id: i64,
        comet_planet_ids: Vec<i64>,
        comets: Vec<CometGroup>,
        num_players: usize,
        configuration: Configuration,
    ) -> Self {
        let mut state = Self {
            step,
            angular_velocity,
            planets,
            initial_planets,
            fleets,
            next_fleet_id,
            comet_planet_ids,
            comets,
            done: false,
            num_players,
            configuration,
            planet_index_by_id: HashMap::new(),
        };
        state.rebuild_planet_index();
        state
    }

    /// Parse an observation dict (kaggle / env_engine / env_model shape) into a
    /// forward-model state. This is the single source of truth for obs→state
    /// parsing, shared by `set_state` and the feature encoder so neither
    /// reimplements it.
    pub fn from_py_obs(
        dict: &Bound<'_, PyDict>,
        num_players: usize,
        configuration: Configuration,
    ) -> PyResult<Self> {
        let step: i64 = dict
            .get_item("step")?
            .ok_or_else(|| PyRuntimeError::new_err("state missing 'step'"))?
            .extract()?;
        let angular_velocity: f64 = dict
            .get_item("angular_velocity")?
            .ok_or_else(|| PyRuntimeError::new_err("state missing 'angular_velocity'"))?
            .extract()?;

        let planets_obj = dict
            .get_item("planets")?
            .ok_or_else(|| PyRuntimeError::new_err("state missing 'planets'"))?;
        let planets_l = planets_obj.downcast::<PyList>()?;
        let mut planets = Vec::with_capacity(planets_l.len());
        for v in planets_l.iter() {
            planets.push(parse_planet(&v)?);
        }

        let initial_obj = dict
            .get_item("initial_planets")?
            .ok_or_else(|| PyRuntimeError::new_err("state missing 'initial_planets'"))?;
        let initial_l = initial_obj.downcast::<PyList>()?;
        let mut initial_planets = Vec::with_capacity(initial_l.len());
        for v in initial_l.iter() {
            initial_planets.push(parse_planet(&v)?);
        }

        let fleets_obj = dict
            .get_item("fleets")?
            .ok_or_else(|| PyRuntimeError::new_err("state missing 'fleets'"))?;
        let fleets_l = fleets_obj.downcast::<PyList>()?;
        let mut fleets = Vec::with_capacity(fleets_l.len());
        for v in fleets_l.iter() {
            fleets.push(parse_fleet(&v)?);
        }

        // next_fleet_id may be missing on raw player observations — default to
        // max(fleet.id)+1, which matches kaggle's behavior when no fleets have
        // been spawned yet.
        let next_fleet_id: i64 = match dict.get_item("next_fleet_id")? {
            Some(v) => v.extract()?,
            None => fleets
                .iter()
                .map(|f| f.id)
                .max()
                .map(|m| m + 1)
                .unwrap_or(0),
        };

        let comet_planet_ids: Vec<i64> = match dict.get_item("comet_planet_ids")? {
            Some(v) => v.extract()?,
            None => Vec::new(),
        };
        let mut comets = Vec::new();
        if let Some(comets_obj) = dict.get_item("comets")? {
            if !comets_obj.is_none() {
                let comets_l = comets_obj.downcast::<PyList>()?;
                for v in comets_l.iter() {
                    comets.push(parse_comet(&v)?);
                }
            }
        }

        Ok(Self::new(
            step,
            angular_velocity,
            planets,
            initial_planets,
            fleets,
            next_fleet_id,
            comet_planet_ids,
            comets,
            num_players,
            configuration,
        ))
    }

    pub fn rebuild_planet_index(&mut self) {
        self.planet_index_by_id.clear();
        self.planet_index_by_id.reserve(self.planets.len());
        for (idx, planet) in self.planets.iter().enumerate() {
            self.planet_index_by_id.insert(planet.id, idx);
        }
    }

    pub fn step_with_actions(&mut self, actions: &[Vec<MoveAction>]) -> Result<bool, String> {
        if self.done {
            return Ok(true);
        }
        if actions.len() != self.num_players {
            return Err(format!(
                "need {} action lists, got {}",
                self.num_players,
                actions.len()
            ));
        }

        // Drop comets whose path has already been exhausted. We do NOT spawn
        // new comets here — that step is owned by the kaggle env and not
        // reproducible without the original seed.
        let expired_prelaunch = self.expired_comet_ids();
        if !expired_prelaunch.is_empty() {
            self.remove_comets(&expired_prelaunch);
        }

        for (player_id, action) in actions.iter().enumerate() {
            self.process_moves(player_id as i64, action);
        }

        for planet in &mut self.planets {
            if planet.owner != -1 {
                planet.ships += planet.production;
            }
        }

        let turn_step = self.step;
        let planet_count = self.planets.len();
        let mut planet_paths: Vec<Option<PlanetPath>> = vec![None; planet_count];

        // Orbital motion for non-comet planets.
        let comet_id_set: HashSet<i64> = self.comet_planet_ids.iter().copied().collect();
        for (idx, planet) in self.planets.iter().enumerate() {
            if comet_id_set.contains(&planet.id) {
                continue;
            }
            let old_pos = (planet.x, planet.y);
            let mut new_pos = old_pos;
            let initial_p = &self.initial_planets[idx];
            let dx = initial_p.x - CENTER;
            let dy = initial_p.y - CENTER;
            let orbital_r = (dx * dx + dy * dy).sqrt();
            if orbital_r + planet.radius < ROTATION_RADIUS_LIMIT {
                let initial_angle = dy.atan2(dx);
                let current_angle = initial_angle + self.angular_velocity * turn_step as f64;
                new_pos = (
                    CENTER + orbital_r * current_angle.cos(),
                    CENTER + orbital_r * current_angle.sin(),
                );
            }
            planet_paths[idx] = Some(PlanetPath {
                old_pos,
                new_pos,
                check_collision: true,
            });
        }

        // Step existing comets along their precomputed paths.
        let mut expired_postmove: Vec<i64> = Vec::new();
        for group in &mut self.comets {
            group.path_index += 1;
            let idx = group.path_index as usize;
            for (i, pid) in group.planet_ids.iter().enumerate() {
                let Some(planet_idx) = self.planet_index_by_id.get(pid).copied() else {
                    continue;
                };
                let planet = &self.planets[planet_idx];
                let old_pos = (planet.x, planet.y);
                let p_path = &group.paths[i];
                if idx >= p_path.len() {
                    expired_postmove.push(*pid);
                    planet_paths[planet_idx] = Some(PlanetPath {
                        old_pos,
                        new_pos: old_pos,
                        check_collision: true,
                    });
                } else {
                    let next = p_path[idx];
                    planet_paths[planet_idx] = Some(PlanetPath {
                        old_pos,
                        new_pos: (next[0], next[1]),
                        check_collision: old_pos.0 >= 0.0,
                    });
                }
            }
        }

        // Fleet movement + collision.
        let fleet_count = self.fleets.len();
        let mut fleets_to_remove = vec![false; fleet_count];
        let mut combat_lists: Vec<Vec<(i64, i64)>> = vec![Vec::new(); planet_count];
        for (fleet_idx, fleet) in self.fleets.iter_mut().enumerate() {
            let old_pos = (fleet.x, fleet.y);
            let speed = fleet_speed(fleet.ships, self.configuration.ship_speed);
            fleet.x += fleet.angle.cos() * speed;
            fleet.y += fleet.angle.sin() * speed;
            let new_pos = (fleet.x, fleet.y);

            let mut hit_planet = false;
            for (planet_idx, planet) in self.planets.iter().enumerate() {
                let Some(path) = &planet_paths[planet_idx] else {
                    continue;
                };
                if !path.check_collision {
                    continue;
                }
                if swept_pair_hit(old_pos, new_pos, path.old_pos, path.new_pos, planet.radius) {
                    combat_lists[planet_idx].push((fleet.owner, fleet.ships));
                    fleets_to_remove[fleet_idx] = true;
                    hit_planet = true;
                    break;
                }
            }
            if hit_planet {
                continue;
            }

            if !(0.0..=BOARD_SIZE).contains(&fleet.x) || !(0.0..=BOARD_SIZE).contains(&fleet.y) {
                fleets_to_remove[fleet_idx] = true;
                continue;
            }
            if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS {
                fleets_to_remove[fleet_idx] = true;
                continue;
            }
        }

        // Apply movement and resolve combat.
        for (idx, planet) in self.planets.iter_mut().enumerate() {
            if let Some(path) = &planet_paths[idx] {
                planet.x = path.new_pos.0;
                planet.y = path.new_pos.1;
            }
        }

        for (idx, planet) in self.planets.iter_mut().enumerate() {
            let planet_fleets = &combat_lists[idx];
            if planet_fleets.is_empty() {
                continue;
            }
            let mut player_ships = [0i64; MAX_PLAYERS];
            for &(owner, ships) in planet_fleets {
                if owner >= 0 && (owner as usize) < MAX_PLAYERS {
                    player_ships[owner as usize] += ships;
                }
            }
            let mut top_player: i64 = -1;
            let mut top_ships: i64 = -1;
            let mut second_ships: i64 = -1;
            let mut entry_count = 0;
            for (player_idx, &ships) in player_ships.iter().enumerate() {
                if ships <= 0 {
                    continue;
                }
                entry_count += 1;
                if ships > top_ships {
                    second_ships = top_ships;
                    top_ships = ships;
                    top_player = player_idx as i64;
                } else if ships > second_ships {
                    second_ships = ships;
                }
            }
            if entry_count == 0 {
                continue;
            }
            let (survivor_owner, survivor_ships) = if entry_count > 1 {
                let survivor_ships = if top_ships == second_ships {
                    0
                } else {
                    top_ships - second_ships
                };
                let survivor_owner = if survivor_ships > 0 { top_player } else { -1 };
                (survivor_owner, survivor_ships)
            } else {
                (top_player, top_ships)
            };
            if survivor_ships > 0 {
                if planet.owner == survivor_owner {
                    planet.ships += survivor_ships;
                } else {
                    planet.ships -= survivor_ships;
                    if planet.ships < 0 {
                        planet.owner = survivor_owner;
                        planet.ships = planet.ships.abs();
                    }
                }
            }
        }

        if !expired_postmove.is_empty() {
            self.remove_comets(&expired_postmove);
        }

        let mut retain_idx = 0usize;
        self.fleets.retain(|_| {
            let keep = !fleets_to_remove[retain_idx];
            retain_idx += 1;
            keep
        });

        let mut terminated = turn_step >= self.configuration.episode_steps - 2;
        let mut alive = [false; MAX_PLAYERS];
        for planet in &self.planets {
            if planet.owner >= 0 && (planet.owner as usize) < MAX_PLAYERS {
                alive[planet.owner as usize] = true;
            }
        }
        for fleet in &self.fleets {
            if fleet.owner >= 0 && (fleet.owner as usize) < MAX_PLAYERS {
                alive[fleet.owner as usize] = true;
            }
        }
        let alive_count: usize = alive.iter().filter(|&&b| b).count();
        if alive_count <= 1 {
            terminated = true;
        }

        self.done = terminated;
        // kaggle keeps `step` incrementing through the terminal turn (does
        // not reset to 0). We match kaggle so terminal observations align.
        self.step += 1;
        Ok(self.done)
    }

    fn expired_comet_ids(&self) -> Vec<i64> {
        if self.comets.is_empty() {
            return Vec::new();
        }
        let mut expired = Vec::new();
        for group in &self.comets {
            let idx = group.path_index;
            for (i, pid) in group.planet_ids.iter().enumerate() {
                if idx >= group.paths[i].len() as i64 {
                    expired.push(*pid);
                }
            }
        }
        expired
    }

    fn remove_comets(&mut self, expired_ids: &[i64]) {
        let expired_set: HashSet<i64> = expired_ids.iter().copied().collect();
        self.planets
            .retain(|planet| !expired_set.contains(&planet.id));
        self.initial_planets
            .retain(|planet| !expired_set.contains(&planet.id));
        self.comet_planet_ids
            .retain(|pid| !expired_set.contains(pid));
        for group in &mut self.comets {
            group.planet_ids.retain(|pid| !expired_set.contains(pid));
        }
        self.comets.retain(|group| !group.planet_ids.is_empty());
        self.rebuild_planet_index();
    }

    fn process_moves(&mut self, player_id: i64, action: &[MoveAction]) {
        for move_action in action {
            let Some(from_planet_idx) = self.planet_index_by_id.get(&move_action.from_id).copied()
            else {
                continue;
            };
            let from_planet = &mut self.planets[from_planet_idx];
            if from_planet.owner != player_id {
                continue;
            }
            if move_action.ships <= 0 || from_planet.ships < move_action.ships {
                continue;
            }
            from_planet.ships -= move_action.ships;
            let start_x = from_planet.x + move_action.angle.cos() * (from_planet.radius + 0.1);
            let start_y = from_planet.y + move_action.angle.sin() * (from_planet.radius + 0.1);
            self.fleets.push(Fleet {
                id: self.next_fleet_id,
                owner: player_id,
                x: start_x,
                y: start_y,
                angle: move_action.angle,
                from_planet_id: move_action.from_id,
                ships: move_action.ships,
            });
            self.next_fleet_id += 1;
        }
    }
}

pub fn distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt()
}

fn point_to_segment_distance(p: (f64, f64), v: (f64, f64), w: (f64, f64)) -> f64 {
    let l2 = (v.0 - w.0).powi(2) + (v.1 - w.1).powi(2);
    if l2 == 0.0 {
        return distance(p, v);
    }
    let t = (((p.0 - v.0) * (w.0 - v.0) + (p.1 - v.1) * (w.1 - v.1)) / l2).clamp(0.0, 1.0);
    let projection = (v.0 + t * (w.0 - v.0), v.1 + t * (w.1 - v.1));
    distance(p, projection)
}

fn swept_pair_hit(
    a: (f64, f64),
    b: (f64, f64),
    p0: (f64, f64),
    p1: (f64, f64),
    radius: f64,
) -> bool {
    let d0x = a.0 - p0.0;
    let d0y = a.1 - p0.1;
    let dvx = (b.0 - a.0) - (p1.0 - p0.0);
    let dvy = (b.1 - a.1) - (p1.1 - p0.1);
    let a_coeff = dvx * dvx + dvy * dvy;
    let b_coeff = 2.0 * (d0x * dvx + d0y * dvy);
    let c_coeff = d0x * d0x + d0y * d0y - radius * radius;
    if a_coeff < 1e-12 {
        return c_coeff <= 0.0;
    }
    let disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
    if disc < 0.0 {
        return false;
    }
    let sq = disc.sqrt();
    let t1 = (-b_coeff - sq) / (2.0 * a_coeff);
    let t2 = (-b_coeff + sq) / (2.0 * a_coeff);
    t2 >= 0.0 && t1 <= 1.0
}

fn fleet_speed(ships: i64, max_speed: f64) -> f64 {
    let speed = 1.0 + (max_speed - 1.0) * ((ships as f64).ln() / 1000.0f64.ln()).powf(1.5);
    speed.min(max_speed)
}

// ---- PyO3 parsing helpers ----------------------------------------------

fn py_any_to_f64(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value
        .extract::<f64>()
        .or_else(|_| value.extract::<i64>().map(|v| v as f64))
}

fn py_any_to_i64(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    value
        .extract::<i64>()
        .or_else(|_| value.extract::<f64>().map(|v| v as i64))
}

fn parse_planet(value: &Bound<'_, PyAny>) -> PyResult<Planet> {
    // Accept either a list (kaggle obs) or a tuple (env_engine/env_model obs,
    // which serialize rows via `as_tuple`). PySequence covers both.
    let parts = value.downcast::<PySequence>()?;
    if parts.len()? != 7 {
        return Err(PyRuntimeError::new_err("planet tuple must have 7 fields"));
    }
    Ok(Planet {
        id: py_any_to_i64(&parts.get_item(0)?)?,
        owner: py_any_to_i64(&parts.get_item(1)?)?,
        x: py_any_to_f64(&parts.get_item(2)?)?,
        y: py_any_to_f64(&parts.get_item(3)?)?,
        radius: py_any_to_f64(&parts.get_item(4)?)?,
        ships: py_any_to_i64(&parts.get_item(5)?)?,
        production: py_any_to_i64(&parts.get_item(6)?)?,
    })
}

fn parse_fleet(value: &Bound<'_, PyAny>) -> PyResult<Fleet> {
    let parts = value.downcast::<PySequence>()?;
    if parts.len()? != 7 {
        return Err(PyRuntimeError::new_err("fleet tuple must have 7 fields"));
    }
    Ok(Fleet {
        id: py_any_to_i64(&parts.get_item(0)?)?,
        owner: py_any_to_i64(&parts.get_item(1)?)?,
        x: py_any_to_f64(&parts.get_item(2)?)?,
        y: py_any_to_f64(&parts.get_item(3)?)?,
        angle: py_any_to_f64(&parts.get_item(4)?)?,
        from_planet_id: py_any_to_i64(&parts.get_item(5)?)?,
        ships: py_any_to_i64(&parts.get_item(6)?)?,
    })
}

fn parse_comet(value: &Bound<'_, PyAny>) -> PyResult<CometGroup> {
    let dict = value.downcast::<PyDict>()?;
    let planet_ids: Vec<i64> = dict
        .get_item("planet_ids")?
        .ok_or_else(|| PyRuntimeError::new_err("comet missing planet_ids"))?
        .extract()?;
    let paths_obj = dict
        .get_item("paths")?
        .ok_or_else(|| PyRuntimeError::new_err("comet missing paths"))?;
    let paths_list = paths_obj.downcast::<PyList>()?;
    let mut paths = Vec::with_capacity(paths_list.len());
    for path_v in paths_list.iter() {
        let path_l = path_v.downcast::<PyList>()?;
        let mut path = Vec::with_capacity(path_l.len());
        for pt_v in path_l.iter() {
            let pt_l = pt_v.downcast::<PyList>().map(|l| {
                let x = py_any_to_f64(&l.get_item(0)?)?;
                let y = py_any_to_f64(&l.get_item(1)?)?;
                Ok::<[f64; 2], PyErr>([x, y])
            });
            let pt = match pt_l {
                Ok(r) => r?,
                Err(_) => {
                    // Tuple fallback.
                    let tup: (f64, f64) = pt_v.extract()?;
                    [tup.0, tup.1]
                }
            };
            path.push(pt);
        }
        paths.push(path);
    }
    let path_index: i64 = dict
        .get_item("path_index")?
        .ok_or_else(|| PyRuntimeError::new_err("comet missing path_index"))?
        .extract()?;
    Ok(CometGroup {
        planet_ids,
        paths,
        path_index,
    })
}

fn parse_configuration(value: Option<&Bound<'_, PyAny>>) -> PyResult<Configuration> {
    let mut cfg = Configuration::default();
    let Some(v) = value else { return Ok(cfg) };
    if v.is_none() {
        return Ok(cfg);
    }
    let dict = v.downcast::<PyDict>()?;
    if let Some(x) = dict.get_item("episodeSteps")? {
        cfg.episode_steps = x.extract()?;
    }
    if let Some(x) = dict.get_item("shipSpeed")? {
        cfg.ship_speed = x.extract()?;
    }
    Ok(cfg)
}

fn parse_actions(actions: &Bound<'_, PyAny>, num_players: usize) -> PyResult<Vec<Vec<MoveAction>>> {
    let actions_list = actions.downcast::<PyList>()?;
    if actions_list.len() != num_players {
        return Err(PyRuntimeError::new_err(format!(
            "need {num_players} action lists, got {}",
            actions_list.len()
        )));
    }
    let mut out = Vec::with_capacity(num_players);
    for player_actions in actions_list.iter() {
        let Ok(moves) = player_actions.downcast::<PyList>() else {
            out.push(Vec::new());
            continue;
        };
        let mut parsed = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            let Ok(parts) = mv.downcast::<PyList>() else {
                continue;
            };
            if parts.len() != 3 {
                continue;
            }
            let from_id = py_any_to_i64(&parts.get_item(0)?)?;
            let angle = py_any_to_f64(&parts.get_item(1)?)?;
            let ships = py_any_to_i64(&parts.get_item(2)?)?;
            parsed.push(MoveAction {
                from_id,
                angle,
                ships,
            });
        }
        out.push(parsed);
    }
    Ok(out)
}

// ---- PyO3 serialization helpers ----------------------------------------

fn py_comets<'py>(py: Python<'py>, comets: &[CometGroup]) -> PyResult<Bound<'py, PyAny>> {
    let items: PyResult<Vec<Py<PyAny>>> = comets
        .iter()
        .map(|c| {
            let d = PyDict::new(py);
            d.set_item("planet_ids", c.planet_ids.clone())?;
            let paths: Vec<Vec<(f64, f64)>> = c
                .paths
                .iter()
                .map(|p| p.iter().map(|pt| (pt[0], pt[1])).collect())
                .collect();
            d.set_item("paths", paths)?;
            d.set_item("path_index", c.path_index)?;
            Ok::<Py<PyAny>, PyErr>(d.into_any().unbind())
        })
        .collect();
    Ok(PyList::new(py, items?)?.into_any())
}

fn build_observation<'py>(
    py: Python<'py>,
    state: &EngineState,
    player: usize,
) -> PyResult<Py<PyAny>> {
    let planets_obj = PyList::new(py, state.planets.iter().map(Planet::as_tuple))?.into_any();
    let initial_obj =
        PyList::new(py, state.initial_planets.iter().map(Planet::as_tuple))?.into_any();
    let fleets_obj = PyList::new(py, state.fleets.iter().map(Fleet::as_tuple))?.into_any();
    let comets_obj = py_comets(py, &state.comets)?;
    let comet_ids_obj = PyList::new(py, state.comet_planet_ids.iter().copied())?.into_any();
    let dict = PyDict::new(py);
    dict.set_item("player", player)?;
    dict.set_item("step", state.step)?;
    dict.set_item("angular_velocity", state.angular_velocity)?;
    dict.set_item("planets", planets_obj)?;
    dict.set_item("initial_planets", initial_obj)?;
    dict.set_item("fleets", fleets_obj)?;
    dict.set_item("comets", comets_obj)?;
    dict.set_item("comet_planet_ids", comet_ids_obj)?;
    dict.set_item("next_fleet_id", state.next_fleet_id)?;
    Ok(dict.into_any().unbind())
}

// ---- Public Python class ------------------------------------------------

#[pyclass]
struct OrbitWarsModel {
    state: Option<EngineState>,
}

#[pymethods]
impl OrbitWarsModel {
    #[new]
    #[pyo3(signature = (num_players=2, configuration=None))]
    fn new(num_players: usize, configuration: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if num_players != 2 && num_players != 4 {
            return Err(PyRuntimeError::new_err(format!(
                "num_players must be 2 or 4, got {num_players}"
            )));
        }
        let configuration = parse_configuration(configuration)?;
        // Empty placeholder — set_state will populate.
        let _ = (num_players, configuration);
        Ok(Self { state: None })
    }

    /// Load engine state from a dict — typically a kaggle observation augmented
    /// with `next_fleet_id` (and optionally `configuration` to override
    /// shipSpeed / episodeSteps).
    #[pyo3(signature = (state, num_players=2, configuration=None))]
    fn set_state(
        &mut self,
        state: &Bound<'_, PyAny>,
        num_players: usize,
        configuration: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if num_players != 2 && num_players != 4 {
            return Err(PyRuntimeError::new_err(format!(
                "num_players must be 2 or 4, got {num_players}"
            )));
        }
        let dict = state.downcast::<PyDict>()?;
        let cfg = parse_configuration(configuration)?;
        self.state = Some(EngineState::from_py_obs(dict, num_players, cfg)?);
        Ok(())
    }

    /// Advance one step. `actions` is `[player0_moves, player1_moves, ...]`
    /// where each move is `[from_planet_id, angle_radians, ships]`.
    /// Returns `{observations: [...], done: bool}`.
    fn step(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let np = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call set_state before step"))?
            .num_players;
        let parsed = parse_actions(actions, np)?;
        let done = {
            let s = self.state.as_mut().expect("state present");
            s.step_with_actions(&parsed)
                .map_err(PyRuntimeError::new_err)?
        };
        let state = self.state.as_ref().expect("state present");
        let mut observations = Vec::with_capacity(np);
        for p in 0..np {
            observations.push(build_observation(py, state, p)?);
        }
        let dict = PyDict::new(py);
        dict.set_item("observations", observations)?;
        dict.set_item("done", done)?;
        Ok(dict.into_any().unbind())
    }

    /// Step without producing per-player observation dicts. Returns just
    /// `{done}`. Use for benchmarking / batched rollouts where the caller
    /// pulls state via `get_state()` only when needed.
    fn step_fast(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let np = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call set_state before step"))?
            .num_players;
        let parsed = parse_actions(actions, np)?;
        let done = {
            let s = self.state.as_mut().expect("state present");
            s.step_with_actions(&parsed)
                .map_err(PyRuntimeError::new_err)?
        };
        let dict = PyDict::new(py);
        dict.set_item("done", done)?;
        Ok(dict.into_any().unbind())
    }

    /// Get the current engine state as a dict (same shape as set_state input,
    /// plus `done`).
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let s = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call set_state first"))?;
        let dict = PyDict::new(py);
        dict.set_item("step", s.step)?;
        dict.set_item("angular_velocity", s.angular_velocity)?;
        dict.set_item(
            "planets",
            s.planets.iter().map(Planet::as_tuple).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "initial_planets",
            s.initial_planets
                .iter()
                .map(Planet::as_tuple)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "fleets",
            s.fleets.iter().map(Fleet::as_tuple).collect::<Vec<_>>(),
        )?;
        dict.set_item("next_fleet_id", s.next_fleet_id)?;
        dict.set_item("comet_planet_ids", s.comet_planet_ids.clone())?;
        dict.set_item("comets", py_comets(py, &s.comets)?)?;
        dict.set_item("done", s.done)?;
        Ok(dict.into_any().unbind())
    }

    /// Encode the current state into model input features from `player`'s
    /// perspective. Returns `{"planet_ids", "distance_matrix", "n"}`. This is
    /// the same `features::encode` the bot uses natively — call this in the
    /// training loop so training and the bot share one feature implementation.
    #[pyo3(signature = (player=0))]
    fn features(&self, py: Python<'_>, player: i64) -> PyResult<Py<PyAny>> {
        let s = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call set_state first"))?;
        features::encode(s, player).into_py_dict(py)
    }

    #[getter]
    fn done(&self) -> PyResult<bool> {
        Ok(self.state.as_ref().map(|s| s.done).unwrap_or(false))
    }

    #[getter]
    fn step_count(&self) -> PyResult<i64> {
        Ok(self.state.as_ref().map(|s| s.step).unwrap_or(0))
    }
}

/// Encode an observation dict directly into features without holding a model —
/// the path the training loop uses on `env_engine` observations. Identical
/// output to `OrbitWarsModel.features` (both call `features::encode`).
#[pyfunction]
#[pyo3(signature = (obs, player=0, num_players=2, configuration=None))]
fn encode_obs(
    py: Python<'_>,
    obs: &Bound<'_, PyAny>,
    player: i64,
    num_players: usize,
    configuration: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dict = obs.downcast::<PyDict>()?;
    let cfg = parse_configuration(configuration)?;
    let state = EngineState::from_py_obs(dict, num_players, cfg)?;
    features::encode(&state, player).into_py_dict(py)
}

#[pymodule]
fn orbit_wars_model(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OrbitWarsModel>()?;
    m.add_function(wrap_pyfunction!(encode_obs, m)?)?;
    Ok(())
}
