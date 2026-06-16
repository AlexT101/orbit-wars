//! Shared library code for the aphrodite bot and extraction tools.

pub mod apollo;
pub mod apollo_bridge;
pub mod duct;
pub mod pathing;
pub mod profiling;
pub mod sim;
pub mod value_net;
pub mod xgb;

use serde_json::Value;
use std::collections::HashMap;

// ---- Game constants ----
pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER_X: f64 = 50.0;
pub const CENTER_Y: f64 = 50.0;
pub const SUN_RADIUS: f64 = 10.0;
pub const ROTATION_RADIUS_LIMIT: f64 = 50.0;
pub const COMET_RADIUS: f64 = 1.0;
pub const COMET_PRODUCTION: i64 = 1;
pub const DEFAULT_MAX_FLEET_SPEED: f64 = 6.0;
pub const DEFAULT_COMET_SPEED: f64 = 4.0;

// ---- Types ----
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
    pub comet_speed: f64,
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
            if idx < 0 {
                return None;
            }
            let path = &g.paths[i];
            if idx as usize == path.len() && !path.is_empty() {
                return Some(path[path.len() - 1]);
            }
            if idx as usize > path.len() || path.is_empty() {
                return None;
            }
            return Some(path[idx as usize]);
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
}

// ---- Parsing ----
pub fn as_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| v.as_i64().unwrap_or(0) as f64)
}

/// Parse an integer field, tolerating whole-number floats. Mirrors [`as_f64`].
///
/// The Kaggle engine sanitizes a move's `ships` to `int(...)` but stores the
/// move's `from_id` verbatim into the resulting fleet's `from_planet_id`
/// (`orbit_wars.py`). An opponent who submits a move with a float planet id
/// (e.g. `33.0`) therefore produces a fleet whose `from_planet_id` serializes
/// to JSON as `33.0`, for which serde_json's `as_i64()` returns `None`. With
/// the `as_i64()?` pattern inside a `filter_map` that would silently drop the
/// whole fleet, blinding us to it; rounding the float keeps the fleet visible.
pub fn as_i64(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_f64().map(|f| f.round() as i64))
}

pub fn parse_state(v: &Value) -> GameState {
    let player = as_i64(&v["player"]).unwrap_or(0) as i32;
    let step = as_i64(&v["step"]).unwrap_or(0);
    let angular_velocity = v["angular_velocity"].as_f64().unwrap_or(0.0);

    let comet_ids: std::collections::HashSet<i64> = v["comet_planet_ids"]
        .as_array()
        .map(|a| a.iter().filter_map(as_i64).collect())
        .unwrap_or_default();

    let initial_pos: HashMap<i64, (f64, f64)> = v["initial_planets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let arr = p.as_array()?;
                    let id = as_i64(arr.get(0)?)?;
                    let x = as_f64(arr.get(2)?);
                    let y = as_f64(arr.get(3)?);
                    Some((id, (x, y)))
                })
                .collect()
        })
        .unwrap_or_default();

    let planets: Vec<Planet> = v["planets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let arr = p.as_array()?;
                    let id = as_i64(arr.get(0)?)?;
                    let owner = as_i64(arr.get(1)?)? as i32;
                    let x = as_f64(arr.get(2)?);
                    let y = as_f64(arr.get(3)?);
                    let radius = as_f64(arr.get(4)?);
                    let ships = as_i64(arr.get(5)?).unwrap_or(0);
                    let production = as_i64(arr.get(6)?).unwrap_or(0);
                    let is_comet = comet_ids.contains(&id);
                    let (ix, iy) = *initial_pos.get(&id).unwrap_or(&(x, y));
                    let dx = ix - CENTER_X;
                    let dy = iy - CENTER_Y;
                    let orbital_radius = (dx * dx + dy * dy).sqrt();
                    let initial_angle = dy.atan2(dx);
                    let is_orbiting = !is_comet && orbital_radius + radius < ROTATION_RADIUS_LIMIT;
                    Some(Planet {
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
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let fleets: Vec<Fleet> = v["fleets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|f| {
                    let arr = f.as_array()?;
                    Some(Fleet {
                        id: as_i64(arr.get(0)?)?,
                        owner: as_i64(arr.get(1)?)? as i32,
                        x: as_f64(arr.get(2)?),
                        y: as_f64(arr.get(3)?),
                        angle: as_f64(arr.get(4)?),
                        from_planet_id: as_i64(arr.get(5)?)?,
                        ships: as_i64(arr.get(6)?).unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let comets: Vec<CometGroup> = v["comets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|g| {
                    let pids = g["planet_ids"]
                        .as_array()?
                        .iter()
                        .filter_map(as_i64)
                        .collect();
                    let paths = g["paths"]
                        .as_array()?
                        .iter()
                        .filter_map(|p| {
                            Some(
                                p.as_array()?
                                    .iter()
                                    .filter_map(|pt| {
                                        let arr = pt.as_array()?;
                                        Some((as_f64(arr.get(0)?), as_f64(arr.get(1)?)))
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    let path_index = as_i64(&g["path_index"])?;
                    Some(CometGroup {
                        planet_ids: pids,
                        paths,
                        path_index,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let cfg = v.get("config");
    let max_speed = cfg
        .and_then(|c| c.get("shipSpeed"))
        .and_then(|v| v.as_f64())
        .unwrap_or(DEFAULT_MAX_FLEET_SPEED);
    let comet_speed = cfg
        .and_then(|c| c.get("cometSpeed"))
        .and_then(|v| v.as_f64())
        .unwrap_or(DEFAULT_COMET_SPEED);

    GameState {
        player,
        step,
        planets,
        fleets,
        angular_velocity,
        comets,
        max_speed,
        comet_speed,
    }
}

#[cfg(test)]
mod parse_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_state_accepts_float_shaped_integer_metadata() {
        let v = json!({
            "player": 1.0,
            "step": 12.0,
            "angular_velocity": 0.03,
            "comet_planet_ids": [33.0],
            "initial_planets": [
                [0.0, 0.0, 60.0, 50.0, 1.5, 10.0, 1.0],
                [33.0, -1.0, 5.0, 6.0, 1.0, 1.0, 1.0]
            ],
            "planets": [
                [0.0, 0.0, 50.0, 60.0, 1.5, 10.0, 1.0],
                [33.0, -1.0, 5.0, 6.0, 1.0, 1.0, 1.0]
            ],
            "fleets": [],
            "comets": [{
                "planet_ids": [33.0],
                "paths": [[[5.0, 6.0], [7.0, 8.0]]],
                "path_index": 1.0
            }]
        });

        let state = parse_state(&v);
        assert_eq!(state.player, 1);
        assert_eq!(state.step, 12);

        let orbiting = state.planets.iter().find(|p| p.id == 0).unwrap();
        assert!((orbiting.orbital_radius - 10.0).abs() < 1e-9);
        assert!(orbiting.initial_angle.abs() < 1e-9);

        let comet = state.planets.iter().find(|p| p.id == 33).unwrap();
        assert!(comet.is_comet);
        assert_eq!(state.comets.len(), 1);
        assert_eq!(state.comets[0].planet_ids, vec![33]);
        assert_eq!(state.comets[0].path_index, 1);
    }
}
