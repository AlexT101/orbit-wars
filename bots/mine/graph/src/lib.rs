//! Graph bot — a thin native agent in the spirit of `apollo2`, but whose model
//! of the environment is `orbit_wars_model` (the `experimental_arch/env_model`
//! forward model) instead of a vendored engine clone.
//!
//! The Python wrapper (`main.py`) instantiates one [`Bot`] at import time and
//! forwards every observation through `Bot::compute_moves`, then drains the
//! `[LINE]`/`[DOT]`/`[TEXT]` debug strings (same grammar apollo2 uses) so the
//! viewer can overlay them.
//!
//! What's special here: every turn it computes the full pairwise distance
//! matrix between planets and draws an edge between each pair of planets,
//! colored by how far apart they are. That same matrix is exposed to Python via
//! [`Bot::distances_matrix`] — the intended RL input feature. The forward model
//! itself (`EngineState`) is built from the observation and kept around so that,
//! at test time, the RL policy can roll the *same* model used during training.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use orbit_wars_model::{
    distance, CometGroup, Configuration, EngineState, Fleet, Planet,
};

/// Board is 100x100, so the longest possible edge is the diagonal. We normalize
/// edge distances against this so the edge colors are comparable across turns
/// (rather than rescaling to each turn's own min/max).
const BOARD_DIAGONAL: f64 = 141.421_356_237; // sqrt(100^2 + 100^2)

/// Length-1 radius for the small node dot drawn at each planet.
const NODE_DOT_RADIUS: f64 = 0.6;

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
        let comet_planet_ids: Vec<i64> = get_item(obs, "comet_planet_ids")?.extract()?;
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

/// Number of distinct non-neutral owners, clamped to the 2/4-player options the
/// model accepts. Used only to populate `EngineState.num_players`.
fn count_players(planets: &[Planet], fleets: &[Fleet]) -> usize {
    let mut seen = [false; 4];
    for p in planets {
        if (0..4).contains(&p.owner) {
            seen[p.owner as usize] = true;
        }
    }
    for f in fleets {
        if (0..4).contains(&f.owner) {
            seen[f.owner as usize] = true;
        }
    }
    let n = seen.iter().filter(|&&b| b).count();
    if n > 2 {
        4
    } else {
        2
    }
}

/// Map an edge distance to a heat-map hex color: short edges are green, mid
/// edges yellow, long edges red. `t` is `dist / BOARD_DIAGONAL`, clamped.
fn distance_to_color(dist: f64) -> String {
    let t = (dist / BOARD_DIAGONAL).clamp(0.0, 1.0);
    let (r, g) = if t < 0.5 {
        // green -> yellow
        ((t * 2.0 * 255.0) as u32, 255u32)
    } else {
        // yellow -> red
        (255u32, ((1.0 - (t - 0.5) * 2.0) * 255.0) as u32)
    };
    format!("#{:02x}{:02x}00", r.min(255), g.min(255))
}

#[pyclass]
pub struct Bot {
    current_turn: i64,
    /// The forward model built from the latest observation. Kept so a future RL
    /// policy can step the exact same model (`orbit_wars_model`) at test time.
    model: Option<EngineState>,
    /// Pending debug strings emitted during the last `compute_moves` call.
    /// Python drains this via `take_debug()`.
    debug: Vec<String>,
}

#[pymethods]
impl Bot {
    #[new]
    fn new() -> Self {
        Self {
            current_turn: 0,
            model: None,
            debug: Vec::new(),
        }
    }

    #[getter]
    fn current_turn(&self) -> i64 {
        self.current_turn
    }

    /// Drain debug lines emitted by the last planning call. Python prints these
    /// to stdout where the runner picks them up.
    fn take_debug(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug)
    }

    /// Plan a turn. Loads the observation into the `orbit_wars_model` forward
    /// model, emits the planet-graph debug overlay, and returns baseline moves.
    fn compute_moves(&mut self, obs: &Bound<'_, PyDict>) -> PyResult<Vec<(i64, f64, i64)>> {
        self.debug.clear();
        let obs = Observation::from_dict(obs)?;
        let player = obs.player;

        // Build the forward model from this observation (env_model semantics:
        // load arbitrary state, ready to step). We keep it on the Bot for reuse.
        let num_players = count_players(&obs.planets, &obs.fleets);
        let next_fleet_id = obs
            .fleets
            .iter()
            .map(|f| f.id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let model = EngineState::new(
            self.current_turn,
            obs.angular_velocity,
            obs.planets.clone(),
            obs.initial_planets.clone(),
            obs.fleets.clone(),
            next_fleet_id,
            obs.comet_planet_ids.clone(),
            obs.comets.clone(),
            num_players,
            Configuration::default(),
        );

        // Distance matrix over the model's planets — the RL input feature.
        let matrix = pairwise_distances(&model.planets);
        self.emit_graph_debug(&model.planets, &matrix);

        let moves = baseline_moves(player, &model.planets);

        self.model = Some(model);
        self.current_turn += 1;
        Ok(moves)
    }

    /// Return the pairwise planet distance matrix for an observation without
    /// planning a turn — the feature an RL pipeline feeds in. Returns a dict
    /// `{"planet_ids": [...], "matrix": [[...], ...]}` where `matrix[i][j]` is
    /// the distance between `planet_ids[i]` and `planet_ids[j]`.
    fn distances_matrix(&self, py: Python<'_>, obs: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let planets = parse_planets(&get_item(obs, "planets")?)?;
        let matrix = pairwise_distances(&planets);
        let ids: Vec<i64> = planets.iter().map(|p| p.id).collect();
        let dict = PyDict::new(py);
        dict.set_item("planet_ids", ids)?;
        dict.set_item("matrix", matrix)?;
        Ok(dict.into_any().unbind())
    }
}

impl Bot {
    /// Emit the planet adjacency graph: one `[LINE]` per pair of planets colored
    /// by distance, a `[DOT]` node marker at each planet, and a `[TEXT]` summary.
    fn emit_graph_debug(&mut self, planets: &[Planet], matrix: &[Vec<f64>]) {
        const NODE_COLOR: &str = "#cccccc";

        self.debug.push(format!(
            "[TEXT] === turn {} · {} planets ===",
            self.current_turn,
            planets.len()
        ));

        // Edge stats for the header (min / mean / max distance).
        let mut min_d = f64::INFINITY;
        let mut max_d = 0.0f64;
        let mut sum_d = 0.0f64;
        let mut count = 0usize;
        for i in 0..planets.len() {
            for j in (i + 1)..planets.len() {
                let d = matrix[i][j];
                min_d = min_d.min(d);
                max_d = max_d.max(d);
                sum_d += d;
                count += 1;
            }
        }
        if count > 0 {
            self.debug.push(format!(
                "[TEXT] edges: {} · dist min={:.1} mean={:.1} max={:.1}",
                count,
                min_d,
                sum_d / count as f64,
                max_d
            ));
        }

        // Edges (each unordered pair once).
        for i in 0..planets.len() {
            for j in (i + 1)..planets.len() {
                let a = &planets[i];
                let b = &planets[j];
                let color = distance_to_color(matrix[i][j]);
                self.debug.push(format!(
                    "[LINE] {:.3} {:.3} {:.3} {:.3} {}",
                    a.x, a.y, b.x, b.y, color
                ));
            }
        }

        // Node markers.
        for p in planets {
            self.debug.push(format!(
                "[DOT] {:.3} {:.3} {:.1} {}",
                p.x, p.y, NODE_DOT_RADIUS, NODE_COLOR
            ));
        }
    }
}

/// Full symmetric pairwise distance matrix between planet centers, using the
/// env_model `distance` function so the metric matches the forward model.
fn pairwise_distances(planets: &[Planet]) -> Vec<Vec<f64>> {
    let n = planets.len();
    let mut matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distance(
                (planets[i].x, planets[i].y),
                (planets[j].x, planets[j].y),
            );
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }
    matrix
}

/// Nearest-planet-sniper baseline (same logic as the `nearest-sniper`
/// baseline): for each owned planet, send just enough ships to take the closest
/// planet we don't own, if we can afford it. Placeholder until the RL policy
/// replaces it.
fn baseline_moves(player: i64, planets: &[Planet]) -> Vec<(i64, f64, i64)> {
    let mut moves = Vec::new();
    let targets: Vec<&Planet> = planets.iter().filter(|p| p.owner != player).collect();
    if targets.is_empty() {
        return moves;
    }
    for mine in planets.iter().filter(|p| p.owner == player) {
        let mut nearest: Option<&Planet> = None;
        let mut min_dist = f64::INFINITY;
        for t in &targets {
            let d = distance((mine.x, mine.y), (t.x, t.y));
            if d < min_dist {
                min_dist = d;
                nearest = Some(t);
            }
        }
        let Some(target) = nearest else { continue };
        let ships_needed = target.ships + 1;
        if mine.ships >= ships_needed {
            let angle = (target.y - mine.y).atan2(target.x - mine.x);
            moves.push((mine.id, angle, ships_needed));
        }
    }
    moves
}

#[pymodule]
fn graph_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bot>()?;
    Ok(())
}
