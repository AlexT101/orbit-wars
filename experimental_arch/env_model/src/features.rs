//! Model input features for Orbit Wars.
//!
//! The single source of truth for turning an [`EngineState`] into the numeric
//! features a policy/value network consumes. The bot calls [`encode`] natively
//! on its `env_model` state; the training loop calls it via the `encode_obs`
//! pyo3 wrapper (see `lib.rs`) on `env_engine` observations. One implementation,
//! two callers — no train/test skew by construction.
//!
//! Current features:
//!   - `distance_matrix`: symmetric N×N L2 distance between planet centers,
//!     row-major, in board units (a 100×100 board, so values lie in
//!     `[0, 100·√2]`). The diagonal is exactly zero.
//!
//! As features grow, add a field to [`Features`], populate it in [`encode`], and
//! document it in [`feature_info`] so the Python side stays in lock-step.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{distance, EngineState};

/// Encoded features for one observation.
#[derive(Clone, Debug)]
pub struct Features {
    /// Planet ids, in the row/column order of `distance_matrix`.
    pub planet_ids: Vec<i64>,
    /// Symmetric L2 distance matrix between planet centers, row-major, length
    /// `n * n`. `distance_matrix[i * n + j]` is the distance between
    /// `planet_ids[i]` and `planet_ids[j]`.
    pub distance_matrix: Vec<f32>,
    /// Side length of the (square) distance matrix == number of planets.
    pub n: usize,
}

impl Features {
    /// `distance_matrix[i][j]`.
    #[inline]
    pub fn dist(&self, i: usize, j: usize) -> f32 {
        self.distance_matrix[i * self.n + j]
    }

    /// Distance matrix as nested Vecs (convenient for Python / assertions).
    pub fn distance_matrix_nested(&self) -> Vec<Vec<f32>> {
        (0..self.n)
            .map(|i| self.distance_matrix[i * self.n..(i + 1) * self.n].to_vec())
            .collect()
    }

    /// Serialize to a Python dict: `{"planet_ids", "distance_matrix", "n"}`,
    /// where `distance_matrix` is a list-of-lists (np.asarray-ready).
    pub fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("planet_ids", self.planet_ids.clone())?;
        d.set_item("distance_matrix", self.distance_matrix_nested())?;
        d.set_item("n", self.n)?;
        Ok(d.into_any().unbind())
    }
}

/// Encode an [`EngineState`] into model features from `player`'s perspective.
///
/// The distance matrix is player-invariant; `player` is accepted now so the
/// signature stays stable as player-relative features are added.
pub fn encode(state: &EngineState, _player: i64) -> Features {
    let planets = &state.planets;
    let n = planets.len();
    let mut distance_matrix = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distance(
                (planets[i].x, planets[i].y),
                (planets[j].x, planets[j].y),
            ) as f32;
            distance_matrix[i * n + j] = d;
            distance_matrix[j * n + i] = d;
        }
    }
    Features {
        planet_ids: planets.iter().map(|p| p.id).collect(),
        distance_matrix,
        n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Configuration, Planet};
    use std::collections::HashMap;

    fn planet(id: i64, x: f64, y: f64) -> Planet {
        Planet { id, owner: -1, x, y, radius: 1.0, ships: 0, production: 0 }
    }

    fn state_from(coords: &[(i64, f64, f64)]) -> EngineState {
        let planets: Vec<Planet> = coords.iter().map(|&(id, x, y)| planet(id, x, y)).collect();
        let initial = planets.clone();
        EngineState::new(
            0,
            0.0,
            planets,
            initial,
            Vec::new(),
            0,
            Vec::new(),
            Vec::new(),
            2,
            Configuration::default(),
        )
    }

    /// Tiny deterministic LCG so property tests need no external crate and stay
    /// reproducible across runs (no `rand`, no proptest dependency to fetch).
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        /// Uniform in [0, 1).
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
        /// Uniform in [0, n).
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Random valid state with 1..=max_planets planets at positions in the
    /// board, ids 0..n (unique, so id↔index maps are well-defined).
    fn random_state(rng: &mut Lcg, max_planets: usize) -> EngineState {
        let n = 1 + rng.below(max_planets);
        let coords: Vec<(i64, f64, f64)> = (0..n)
            .map(|i| (i as i64, rng.range(0.0, 100.0), rng.range(0.0, 100.0)))
            .collect();
        state_from(&coords)
    }

    // ---- example-based unit tests --------------------------------------

    #[test]
    fn empty_state_has_empty_matrix() {
        let f = encode(&state_from(&[]), 0);
        assert_eq!(f.n, 0);
        assert!(f.distance_matrix.is_empty());
        assert!(f.planet_ids.is_empty());
    }

    #[test]
    fn single_planet_is_one_by_one_zero() {
        let f = encode(&state_from(&[(7, 12.0, 34.0)]), 0);
        assert_eq!(f.n, 1);
        assert_eq!(f.planet_ids, vec![7]);
        assert_eq!(f.dist(0, 0), 0.0);
    }

    #[test]
    fn known_3_4_5_distance() {
        let f = encode(&state_from(&[(0, 0.0, 0.0), (1, 3.0, 4.0)]), 0);
        assert!((f.dist(0, 1) - 5.0).abs() < 1e-6);
        assert!((f.dist(1, 0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn planet_ids_track_order() {
        let f = encode(&state_from(&[(5, 0.0, 0.0), (2, 1.0, 0.0), (9, 0.0, 1.0)]), 0);
        assert_eq!(f.planet_ids, vec![5, 2, 9]);
    }

    #[test]
    fn diagonal_is_zero_and_symmetric_example() {
        let f = encode(&state_from(&[(0, 10.0, 10.0), (1, 90.0, 20.0), (2, 50.0, 70.0)]), 0);
        for i in 0..f.n {
            assert_eq!(f.dist(i, i), 0.0);
            for j in 0..f.n {
                assert_eq!(f.dist(i, j), f.dist(j, i));
            }
        }
    }

    // ---- property tests (randomized, deterministic) --------------------

    #[test]
    fn prop_symmetric_zero_diag_finite_nonneg() {
        let mut rng = Lcg::new(0xC0FFEE);
        for _ in 0..500 {
            let f = encode(&random_state(&mut rng, 12), 0);
            assert_eq!(f.distance_matrix.len(), f.n * f.n);
            for i in 0..f.n {
                assert_eq!(f.dist(i, i), 0.0, "diagonal must be zero");
                for j in 0..f.n {
                    let d = f.dist(i, j);
                    assert!(d.is_finite(), "distance must be finite");
                    assert!(d >= 0.0, "distance must be non-negative");
                    assert_eq!(d, f.dist(j, i), "distance must be symmetric");
                }
            }
        }
    }

    #[test]
    fn prop_within_board_diagonal() {
        let max = 100.0f32 * std::f32::consts::SQRT_2 + 1e-3;
        let mut rng = Lcg::new(0x1234);
        for _ in 0..500 {
            let f = encode(&random_state(&mut rng, 12), 0);
            for &d in &f.distance_matrix {
                assert!(d <= max, "distance {d} exceeds board diagonal");
            }
        }
    }

    #[test]
    fn prop_triangle_inequality() {
        let mut rng = Lcg::new(0x5EED);
        for _ in 0..300 {
            let f = encode(&random_state(&mut rng, 8), 0);
            for i in 0..f.n {
                for j in 0..f.n {
                    for k in 0..f.n {
                        assert!(
                            f.dist(i, k) <= f.dist(i, j) + f.dist(j, k) + 1e-3,
                            "triangle inequality violated"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prop_permutation_equivariance() {
        // The distance between two planet *ids* must be invariant to the order
        // planets appear in — catches accidental index-vs-id confusion.
        let mut rng = Lcg::new(0xABCD);
        for _ in 0..300 {
            let s = random_state(&mut rng, 10);
            let f1 = encode(&s, 0);

            let mut planets = s.planets.clone();
            for i in (1..planets.len()).rev() {
                let j = rng.below(i + 1);
                planets.swap(i, j);
            }
            let coords: Vec<(i64, f64, f64)> =
                planets.iter().map(|p| (p.id, p.x, p.y)).collect();
            let f2 = encode(&state_from(&coords), 0);

            let idx1: HashMap<i64, usize> =
                f1.planet_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let idx2: HashMap<i64, usize> =
                f2.planet_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            for &a in &f1.planet_ids {
                for &b in &f1.planet_ids {
                    assert_eq!(
                        f1.dist(idx1[&a], idx1[&b]),
                        f2.dist(idx2[&a], idx2[&b]),
                        "distance between ids {a},{b} changed under permutation"
                    );
                }
            }
        }
    }

    #[test]
    fn prop_player_invariant() {
        // The distance matrix does not depend on whose perspective we encode.
        let mut rng = Lcg::new(0xFEED);
        for _ in 0..200 {
            let s = random_state(&mut rng, 10);
            assert_eq!(encode(&s, 0).distance_matrix, encode(&s, 1).distance_matrix);
        }
    }
}
