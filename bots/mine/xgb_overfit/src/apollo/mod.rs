//! Vendored apollo engine + planner modules (non-pyo3 subset of the apollo
//! crate). Used to generate MCTS child candidates via
//! `strategy::search_candidates`. The `crate::` paths in the original were
//! rewritten to `crate::apollo::`, and all pyo3 bindings were stripped.
pub mod constants;
pub mod engine;
pub mod world;
pub mod cache;
pub mod helpers;
pub mod aim;
pub mod strategy;
pub mod rollout;
