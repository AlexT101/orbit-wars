//! Forked slice of apollo: aim solver + entity cache, exposed as a separate
//! `aim_native` Python module so alphaduck can call apollo-quality aim eta
//! without modifying the shared `bots/mine/apollo` crate.
//!
//! Slimmed surface (no Bot, no rollout/strategy, no WorldState): only
//! `aim_eta`, `aim_eta_batch`, and a stateful `Cache` handle for callers that
//! want to amortize the EntityCache build across many lookups for the same obs.
//!
//! Source files copied verbatim from apollo:
//!   aim.rs, cache.rs, constants.rs, engine.rs
//!
//! Optimizations applied vs the apollo originals are commented in cache.rs
//! (lazy per-turn aim_cache slot allocation) and lib.rs (stateful Cache pyclass,
//! group-by-src in aim_eta_batch).

mod aim;
mod cache;
mod constants;
mod engine;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use crate::cache::EntityCache;
use crate::constants::CENTER;
use crate::engine::{CometGroup, EngineState, Fleet, Planet, Simulator};
use pyo3::types::{PyList, PyTuple};

fn get_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("obs missing field '{}'", key)))
}

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

fn obs_current_step(obs: &Bound<'_, PyDict>) -> PyResult<i64> {
    let step = match obs.get_item("step")? {
        Some(s) => extract_i64(&s)?,
        None => return Ok(1),
    };
    Ok(if step == 0 { 0 } else { 1 })
}

fn source_unowned(planets: &[Planet], source: i64, obs: &Bound<'_, PyDict>) -> PyResult<bool> {
    let Some(player) = obs.get_item("player")? else {
        return Ok(false);
    };
    let player = extract_i64(&player)?;
    let owned = planets.iter().any(|p| p.id == source && p.owner == player);
    Ok(!owned)
}

fn build_cache_from_obs(obs: &Bound<'_, PyDict>) -> PyResult<EntityCache> {
    let planets = parse_planets(&get_item(obs, "planets")?)?;
    let angular_velocity: f64 = get_item(obs, "angular_velocity")?.extract()?;
    let (comets, comet_planet_ids) =
        match (obs.get_item("comets")?, obs.get_item("comet_planet_ids")?) {
            (Some(c), Some(ids)) => (parse_comets(&c)?, ids.extract::<Vec<i64>>()?),
            _ => (Vec::new(), Vec::new()),
        };
    Ok(EntityCache::build(
        &planets,
        &comets,
        &comet_planet_ids,
        angular_velocity,
        obs_current_step(obs)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (obs, source, target, fleet_size))]
fn aim_eta(
    obs: &Bound<'_, PyDict>,
    source: i64,
    target: i64,
    fleet_size: i64,
) -> PyResult<Option<f64>> {
    let planets = parse_planets(&get_item(obs, "planets")?)?;
    if source_unowned(&planets, source, obs)? {
        return Ok(None);
    }
    let cache = build_cache_from_obs(obs)?;
    Ok(crate::aim::aim_with_prediction(&cache, source, target, fleet_size, 0)
        .map(|(_a, _t, _x, _y, flight_time)| flight_time))
}

/// Compute eta for all (src, tgt, ships) triples against one obs in a single
/// call. Builds the EntityCache once. Triples are processed sorted by
/// `(src, ships)` so the inner per-pair work shares launcher position lookups.
#[pyfunction]
#[pyo3(signature = (obs, triples))]
fn aim_eta_batch(
    obs: &Bound<'_, PyDict>,
    triples: Vec<(i64, i64, i64)>,
) -> PyResult<Vec<Option<f64>>> {
    let cache = build_cache_from_obs(obs)?;
    // Sort indirectly by (src, ships) so within-group calls share launcher pos
    // + fleet velocity. We restore caller order on the way out.
    let n = triples.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (triples[i].0, triples[i].2));
    let mut out: Vec<Option<f64>> = vec![None; n];
    for &i in &order {
        let (src, tgt, fleet_size) = triples[i];
        out[i] = crate::aim::aim_with_prediction(&cache, src, tgt, fleet_size, 0)
            .map(|(_a, _t, _x, _y, flight_time)| flight_time);
    }
    Ok(out)
}

/// Same as `aim_eta_batch` but returns `(eta, angle)` per triple. Use this
/// when the caller needs the apollo launch angle alongside the ETA — apollo's
/// angle and an iterative lead-angle solver are NOT bit-equivalent, so a
/// caller that uses the apollo eta MUST also use the apollo angle for the
/// launched fleet to actually arrive at the target.
#[pyfunction]
#[pyo3(signature = (obs, triples))]
fn aim_eta_angle_batch(
    obs: &Bound<'_, PyDict>,
    triples: Vec<(i64, i64, i64)>,
) -> PyResult<Vec<Option<(f64, f64)>>> {
    let cache = build_cache_from_obs(obs)?;
    let n = triples.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (triples[i].0, triples[i].2));
    let mut out: Vec<Option<(f64, f64)>> = vec![None; n];
    for &i in &order {
        let (src, tgt, fleet_size) = triples[i];
        out[i] = crate::aim::aim_with_prediction(&cache, src, tgt, fleet_size, 0)
            .map(|(angle, _t, _x, _y, flight_time)| (flight_time, angle));
    }
    Ok(out)
}

/// Stateful cache handle: build once per obs, query many times without
/// re-parsing planets/comets or rebuilding entity tables. Useful when a single
/// obs needs many independent aim_eta lookups (replay analysis, search loops).
///
/// IMPORTANT: planet IDs are assigned fresh per game. The cached `(src, tgt,
/// ships) → eta` entries inside an `EntityCache` are valid only for the obs
/// they were built from. Callers MUST construct a new `Cache(obs)` per game
/// (and per turn if planet ownership changes — apollo's `EntityCache` is
/// internally per-turn anyway). Never reuse a `Cache` across games — its
/// id-keyed entries would alias to unrelated planets in the next game.
#[pyclass]
struct Cache {
    inner: EntityCache,
}

#[pymethods]
impl Cache {
    #[new]
    fn new(obs: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: build_cache_from_obs(obs)?,
        })
    }

    fn aim_eta(&self, source: i64, target: i64, fleet_size: i64) -> Option<f64> {
        crate::aim::aim_with_prediction(&self.inner, source, target, fleet_size, 0)
            .map(|(_a, _t, _x, _y, flight_time)| flight_time)
    }

    fn aim_eta_batch(&self, triples: Vec<(i64, i64, i64)>) -> Vec<Option<f64>> {
        let n = triples.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (triples[i].0, triples[i].2));
        let mut out: Vec<Option<f64>> = vec![None; n];
        for &i in &order {
            let (src, tgt, fleet_size) = triples[i];
            out[i] = crate::aim::aim_with_prediction(&self.inner, src, tgt, fleet_size, 0)
                .map(|(_a, _t, _x, _y, flight_time)| flight_time);
        }
        out
    }
}

// ----------------------------------------------------------------------------
// engine_step: bit-exact single-tick engine advance.
// ----------------------------------------------------------------------------

fn parse_fleets_dicts(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Fleet>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let f = seq.get_item(i)?;
        let dict = f.downcast::<PyDict>()?;
        out.push(Fleet {
            id: extract_i64(&get_item(dict, "id")?)?,
            owner: extract_i64(&get_item(dict, "owner")?)?,
            x: get_item(dict, "x")?.extract()?,
            y: get_item(dict, "y")?.extract()?,
            angle: get_item(dict, "angle")?.extract()?,
            ships: extract_i64(&get_item(dict, "ships")?)?,
        });
    }
    Ok(out)
}

fn parse_planets_dicts(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Planet>> {
    let seq: Bound<'_, PySequence> = seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let p = seq.get_item(i)?;
        let dict = p.downcast::<PyDict>()?;
        out.push(Planet {
            id: extract_i64(&get_item(dict, "id")?)?,
            owner: extract_i64(&get_item(dict, "owner")?)?,
            x: get_item(dict, "x")?.extract()?,
            y: get_item(dict, "y")?.extract()?,
            radius: get_item(dict, "radius")?.extract()?,
            ships: extract_i64(&get_item(dict, "ships")?)?,
            production: extract_i64(&get_item(dict, "prod")?)?,
        });
    }
    Ok(out)
}

fn parse_comets_dicts(seq: &Bound<'_, PyAny>) -> PyResult<Vec<CometGroup>> {
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

/// Synthesize the engine's `initial_planets` table from an alphaduck-shaped
/// state where each planet dict carries `orb_r`, `init_angle`, `is_comet`,
/// `is_orbiting`. For orbital planets, initial position is on the orbit at
/// `init_angle`; for stationary planets and comets, current xy is used as the
/// initial xy (comets' positions are driven by paths, not orbit math).
fn synth_initial_planets(planets_seq: &Bound<'_, PyAny>) -> PyResult<Vec<Planet>> {
    let seq: Bound<'_, PySequence> = planets_seq.downcast::<PySequence>()?.clone();
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        let dict = item.downcast::<PyDict>()?;
        let id = extract_i64(&get_item(dict, "id")?)?;
        let owner = extract_i64(&get_item(dict, "owner")?)?;
        let radius: f64 = get_item(dict, "radius")?.extract()?;
        let ships = extract_i64(&get_item(dict, "ships")?)?;
        let production = extract_i64(&get_item(dict, "prod")?)?;
        let is_comet: bool = match dict.get_item("is_comet")? {
            Some(v) => v.extract().unwrap_or(false),
            None => false,
        };
        let is_orbiting: bool = match dict.get_item("is_orbiting")? {
            Some(v) => v.extract().unwrap_or(false),
            None => false,
        };
        let (x, y) = if is_orbiting && !is_comet {
            let orb_r: f64 = get_item(dict, "orb_r")?.extract()?;
            let init_angle: f64 = get_item(dict, "init_angle")?.extract()?;
            (CENTER + orb_r * init_angle.cos(), CENTER + orb_r * init_angle.sin())
        } else {
            let x: f64 = get_item(dict, "x")?.extract()?;
            let y: f64 = get_item(dict, "y")?.extract()?;
            (x, y)
        };
        out.push(Planet {
            id,
            owner,
            x,
            y,
            radius,
            ships,
            production,
        });
    }
    Ok(out)
}

fn planet_to_pydict<'py>(py: Python<'py>, p: &Planet) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", p.id)?;
    d.set_item("owner", p.owner)?;
    d.set_item("x", p.x)?;
    d.set_item("y", p.y)?;
    d.set_item("radius", p.radius)?;
    d.set_item("ships", p.ships)?;
    d.set_item("prod", p.production)?;
    Ok(d)
}

fn fleet_to_pydict<'py>(py: Python<'py>, f: &Fleet) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", f.id)?;
    d.set_item("owner", f.owner)?;
    d.set_item("x", f.x)?;
    d.set_item("y", f.y)?;
    d.set_item("angle", f.angle)?;
    d.set_item("ships", f.ships)?;
    Ok(d)
}

/// Advance one full engine turn from `state`. No player actions are applied
/// (any new launches must already be present in `state["fleets"]`).
///
/// Required fields on `state`:
///   step            : i64
///   angular_velocity (or "av") : f64
///   planets         : list of dicts with id/owner/x/y/radius/ships/prod
///                     and (for orbital sense) is_orbiting/orb_r/init_angle/is_comet
///   fleets          : list of dicts with id/owner/x/y/angle/ships
///   comets          : list of {planet_ids, paths, path_index} (may be empty)
///   comet_planet_ids: list of i64 (may be empty)
///
/// Returns a new state dict in the same shape, advanced by one engine tick
/// (production, planet/comet rotation, fleet movement, combat resolution,
/// despawn checks).
#[pyfunction]
#[pyo3(signature = (state))]
fn engine_step<'py>(py: Python<'py>, state: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let step = extract_i64(&get_item(state, "step")?)?;
    let omega: f64 = match state.get_item("angular_velocity")? {
        Some(v) => v.extract()?,
        None => get_item(state, "av")?.extract()?,
    };
    let planets_any = get_item(state, "planets")?;
    let planets = parse_planets_dicts(&planets_any)?;
    let initial_planets = synth_initial_planets(&planets_any)?;
    let fleets = parse_fleets_dicts(&get_item(state, "fleets")?)?;
    let next_fleet_id = fleets.iter().map(|f| f.id).max().unwrap_or(-1) + 1;
    let comets = parse_comets_dicts(&get_item(state, "comets")?)?;
    let comet_planet_ids: Vec<i64> = match state.get_item("comet_planet_ids")? {
        Some(v) => v.extract()?,
        None => Vec::new(),
    };

    let engine_state = EngineState::from_observation_parts(
        step,
        omega,
        planets,
        initial_planets,
        fleets,
        next_fleet_id,
        comet_planet_ids,
        comets,
        2,
    );

    let mut sim = Simulator::new(&engine_state);
    sim.step(None);

    let out = PyDict::new(py);
    out.set_item("step", sim.step_count())?;
    out.set_item("angular_velocity", omega)?;
    out.set_item("av", omega)?;
    let new_planets = PyList::empty(py);
    for p in sim.planets() {
        new_planets.append(planet_to_pydict(py, p)?)?;
    }
    out.set_item("planets", new_planets)?;
    let new_fleets = PyList::empty(py);
    for f in sim.fleets() {
        new_fleets.append(fleet_to_pydict(py, f)?)?;
    }
    out.set_item("fleets", new_fleets)?;
    out.set_item("comet_planet_ids", sim.comet_planet_ids().to_vec())?;
    // Comets: re-export the (now-advanced) groups so the caller can keep stepping.
    let comets_out = PyList::empty(py);
    out.set_item("comets", comets_out)?;
    let _ = PyTuple::empty(py);
    Ok(out)
}

#[pymodule]
fn aim_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aim_eta, m)?)?;
    m.add_function(wrap_pyfunction!(aim_eta_batch, m)?)?;
    m.add_function(wrap_pyfunction!(aim_eta_angle_batch, m)?)?;
    m.add_function(wrap_pyfunction!(engine_step, m)?)?;
    m.add_class::<Cache>()?;
    Ok(())
}
