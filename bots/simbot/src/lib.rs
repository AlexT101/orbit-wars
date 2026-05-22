//! Native Orbit Wars agent for `simbot`.
//!
//! Cloned from the `nearest-sniper-rust` bot: for each planet we own, find the
//! closest planet we don't own and send `garrison + 1` ships if we can afford
//! the takeover, otherwise accumulate. This is the starting point for a stronger
//! strategy driven by in-bot simulation (see [`engine`]).
//!
//! The Python wrapper (`main.py`) parses the observation and forwards the
//! player id plus the raw planet tuples here; we return the moves as
//! `(from_planet_id, angle_radians, ships)` tuples.

mod constants;
mod engine;
mod helpers;
mod sim_probe;

use pyo3::prelude::*;

/// One planet as the engine serializes it:
/// `(id, owner, x, y, radius, ships, production)`. `owner == -1` is neutral.
type PlanetTuple = (i64, i64, f64, f64, f64, i64, i64);

/// Pick moves for `player` given the current `planets`.
///
/// Mirrors `bots/_open_source/nearest-sniper/main.py` field-for-field, including
/// iteration order and the strict `<` tie-break (first nearest target wins), so
/// the output is identical to the Python baseline.
#[pyfunction]
fn compute_moves(player: i64, planets: Vec<PlanetTuple>) -> Vec<(i64, f64, i64)> {
    let mut moves = Vec::new();

    let my_planets: Vec<&PlanetTuple> = planets.iter().filter(|p| p.1 == player).collect();
    let targets: Vec<&PlanetTuple> = planets.iter().filter(|p| p.1 != player).collect();

    if targets.is_empty() {
        return moves;
    }

    for mine in &my_planets {
        // Find the nearest planet we don't own. Strict `<` keeps the first
        // target on ties, matching the Python loop.
        let mut nearest: Option<&PlanetTuple> = None;
        let mut min_dist = f64::INFINITY;
        for t in &targets {
            let dx = mine.2 - t.2;
            let dy = mine.3 - t.3;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < min_dist {
                min_dist = dist;
                nearest = Some(t);
            }
        }

        let Some(nearest) = nearest else {
            continue;
        };

        // garrison + 1 guarantees the takeover; only launch if affordable.
        let ships_needed = nearest.5 + 1;
        if mine.5 >= ships_needed {
            let angle = (nearest.3 - mine.3).atan2(nearest.2 - mine.2);
            moves.push((mine.0, angle, ships_needed));
        }
    }

    moves
}

#[pymodule]
fn simbot_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_moves, m)?)?;
    Ok(())
}
