//! ow4 — heuristic 2p Orbit Wars bot.
//!
//! Pipeline per turn:
//!   1. Build per-planet defense ledger (arrivals + surplus).
//!   2. Run snipe pass (capture neutrals that enemy is about to capture).
//!   3. Run attack pass (assign remaining surplus to ROI-best targets).

mod attack;
mod combat;
mod game;
mod ledger;
mod opening;
mod pathing;
mod plan;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::game::{parse_observation, Action};

#[pyclass]
pub struct Bot {
    turn: i64,
}

#[pymethods]
impl Bot {
    #[new]
    fn new() -> Self {
        Self { turn: 0 }
    }

    fn compute_moves(
        &mut self,
        obs: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<(i64, f64, i64)>> {
        let state = parse_observation(obs, self.turn)?;
        let moves: Vec<Action> = crate::plan::plan(&state);
        self.turn += 1;
        Ok(moves
            .into_iter()
            .map(|a| (a.from_id, a.angle, a.ships))
            .collect())
    }
}

#[pymodule]
fn ow4_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bot>()?;
    Ok(())
}
