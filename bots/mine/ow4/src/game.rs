use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER_X: f64 = 50.0;
pub const CENTER_Y: f64 = 50.0;
pub const SUN_RADIUS: f64 = 10.0;
pub const ROTATION_RADIUS_LIMIT: f64 = 50.0;
pub const DEFAULT_MAX_FLEET_SPEED: f64 = 6.0;

#[derive(Clone, Debug)]
pub struct Planet {
    pub id: i64,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i64,
    pub production: i64,
    pub orbital_radius: f64,
    pub initial_angle: f64,
    pub is_orbiting: bool,
    pub is_comet: bool,
}

#[derive(Clone, Debug)]
pub struct Fleet {
    pub id: i64,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i64,
    pub ships: i64,
}

#[derive(Clone, Debug)]
pub struct CometGroup {
    pub planet_ids: Vec<i64>,
    pub paths: Vec<Vec<(f64, f64)>>,
    pub path_index: i64,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub player: i32,
    pub step: i64,
    pub planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub angular_velocity: f64,
    pub comets: Vec<CometGroup>,
    pub max_speed: f64,
}

#[derive(Clone, Debug)]
pub struct Action {
    pub from_id: i64,
    pub angle: f64,
    pub ships: i64,
}

impl GameState {
    pub fn comet_group_for(&self, comet_id: i64) -> Option<(&CometGroup, usize)> {
        for g in &self.comets {
            if let Some(i) = g.planet_ids.iter().position(|&id| id == comet_id) {
                return Some((g, i));
            }
        }
        None
    }

    pub fn comet_remaining(&self, planet: &Planet) -> i64 {
        if !planet.is_comet {
            return 0;
        }
        if let Some((g, i)) = self.comet_group_for(planet.id) {
            return (g.paths[i].len() as i64 - g.path_index).max(0);
        }
        0
    }

    pub fn planet_pos_at(&self, planet: &Planet, dt: i64) -> Option<(f64, f64)> {
        if planet.is_comet {
            let (g, i) = self.comet_group_for(planet.id)?;
            let idx = g.path_index + dt;
            if idx < 0 || idx as usize >= g.paths[i].len() {
                return None;
            }
            return Some(g.paths[i][idx as usize]);
        }
        if planet.is_orbiting {
            let abs_step = (self.step + dt - 1).max(0);
            let a = planet.initial_angle + self.angular_velocity * abs_step as f64;
            Some((
                CENTER_X + planet.orbital_radius * a.cos(),
                CENTER_Y + planet.orbital_radius * a.sin(),
            ))
        } else {
            Some((planet.x, planet.y))
        }
    }

    pub fn planet_by_id(&self, id: i64) -> Option<&Planet> {
        self.planets.iter().find(|p| p.id == id)
    }

    pub fn enemy_id(&self) -> i32 {
        // 2p assumption: opponent is the other id present in player 0/1.
        // If the engine ever uses different ids, fall back to "any non-me, non-neutral".
        for p in &self.planets {
            if p.owner != self.player && p.owner != -1 {
                return p.owner;
            }
        }
        for f in &self.fleets {
            if f.owner != self.player && f.owner != -1 {
                return f.owner;
            }
        }
        1 - self.player
    }
}

fn get_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("obs missing '{}'", key)))
}

fn get_opt<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(d.get_item(key)?)
}

fn parse_planets(seq: &Bound<'_, PyAny>, comet_ids: &[i64], initial: &[(i64, f64, f64)]) -> PyResult<Vec<Planet>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let row = seq.get_item(i)?;
        let id: i64 = row.get_item(0)?.extract()?;
        let owner: i32 = row.get_item(1)?.extract()?;
        let x: f64 = row.get_item(2)?.extract()?;
        let y: f64 = row.get_item(3)?.extract()?;
        let radius: f64 = row.get_item(4)?.extract()?;
        let ships: i64 = row.get_item(5)?.extract()?;
        let production: i64 = row.get_item(6)?.extract()?;
        let is_comet = comet_ids.contains(&id);
        let (ix, iy) = initial
            .iter()
            .find(|(pid, _, _)| *pid == id)
            .map(|(_, x, y)| (*x, *y))
            .unwrap_or((x, y));
        let dx = ix - CENTER_X;
        let dy = iy - CENTER_Y;
        let orbital_radius = (dx * dx + dy * dy).sqrt();
        let initial_angle = dy.atan2(dx);
        let is_orbiting = !is_comet && orbital_radius + radius < ROTATION_RADIUS_LIMIT;
        out.push(Planet {
            id,
            owner,
            x,
            y,
            radius,
            ships,
            production,
            orbital_radius,
            initial_angle,
            is_orbiting,
            is_comet,
        });
    }
    Ok(out)
}

fn parse_initial_pos(seq: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, f64, f64)>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let row = seq.get_item(i)?;
        let id: i64 = row.get_item(0)?.extract()?;
        let x: f64 = row.get_item(2)?.extract()?;
        let y: f64 = row.get_item(3)?.extract()?;
        out.push((id, x, y));
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

fn parse_path(seq: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let row = seq.get_item(i)?;
        let x: f64 = row.get_item(0)?.extract()?;
        let y: f64 = row.get_item(1)?.extract()?;
        out.push((x, y));
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
        let mut paths: Vec<Vec<(f64, f64)>> = Vec::with_capacity(paths_seq.len()?);
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

pub fn parse_observation(obs: &Bound<'_, PyDict>, step: i64) -> PyResult<GameState> {
    let player: i32 = get_item(obs, "player")?.extract()?;
    let angular_velocity: f64 = get_item(obs, "angular_velocity")?.extract()?;
    let comet_planet_ids: Vec<i64> = match get_opt(obs, "comet_planet_ids")? {
        Some(v) => v.extract()?,
        None => Vec::new(),
    };
    let initial = parse_initial_pos(&get_item(obs, "initial_planets")?)?;
    let planets = parse_planets(&get_item(obs, "planets")?, &comet_planet_ids, &initial)?;
    let fleets = parse_fleets(&get_item(obs, "fleets")?)?;
    let comets = match get_opt(obs, "comets")? {
        Some(v) => parse_comets(&v)?,
        None => Vec::new(),
    };

    let max_speed = match get_opt(obs, "config")? {
        Some(cfg) => {
            let d = cfg.downcast::<PyDict>().ok();
            d.and_then(|d| d.get_item("shipSpeed").ok().flatten())
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(DEFAULT_MAX_FLEET_SPEED)
        }
        None => DEFAULT_MAX_FLEET_SPEED,
    };

    // Engine convention: provided step if present, else caller-tracked turn.
    let step = match get_opt(obs, "step")? {
        Some(v) => v.extract::<i64>().unwrap_or(step),
        None => step,
    };

    Ok(GameState {
        player,
        step,
        planets,
        fleets,
        angular_velocity,
        comets,
        max_speed,
    })
}
