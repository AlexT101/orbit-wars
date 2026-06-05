//! Per-turn target-predictor prior stash.
//!
//! The Python wrapper computes a transformer per-planet probability vector for
//! the *current* observation and serializes it as a `{"target_priors": {pid:
//! prob, ...}, "target_priors_player": i32}` field on the JSON sent over
//! stdin. `main.rs` parses both fields once per turn and stashes them here;
//! `apollo_bridge::apollo_candidates` reads the stash and, when the candidate
//! player matches the stashed player, attaches the priors to the apollo
//! `WorldState` so `apollo::strategy::run_strategy` can filter and re-weight
//! candidate targets.
//!
//! Thread-local because the bot binary is single-threaded per turn and we want
//! zero synchronization in the hot planner loops. Cleared at the start of each
//! turn by `main.rs` (`reset()` then `set()`).

use rustc_hash::FxHashMap;
use std::cell::RefCell;

thread_local! {
    /// `Some((player, priors))` set by `main.rs` after parsing the per-turn
    /// JSON; `None` when the wrapper did not include `target_priors` (so the
    /// bot behaves identically to upstream prometheus).
    static TURN_PRIORS: RefCell<Option<(i32, FxHashMap<i64, f64>)>> = const { RefCell::new(None) };
}

/// Stash the priors for this turn. Call before any planner work begins.
pub fn set(player: i32, priors: FxHashMap<i64, f64>) {
    TURN_PRIORS.with(|cell| {
        *cell.borrow_mut() = Some((player, priors));
    });
}

/// Drop the per-turn stash. Called at the start of each turn so a stale stash
/// from a prior turn can't leak in if the wrapper stops sending priors mid-game.
pub fn reset() {
    TURN_PRIORS.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Run `f` with a borrow of the priors map iff (a) a stash exists for this
/// turn AND (b) the stashed player matches `player`. The borrow is read-only
/// and stays inside the planner call, so no lifetimes escape.
///
/// This is the only access point. `apollo_bridge::apollo_candidates` uses it
/// to thread `Some(&map)` into the `WorldState` we just built; every other
/// apollo call site (rollout opponent replay, MCTS expansion) skips this and
/// gets `target_priors = None`, preserving stock behavior.
pub fn with_priors_for<F, R>(player: i32, f: F) -> R
where
    F: FnOnce(Option<&FxHashMap<i64, f64>>) -> R,
{
    TURN_PRIORS.with(|cell| {
        let borrowed = cell.borrow();
        let arg = borrowed.as_ref().and_then(|(p, m)| {
            if *p == player {
                Some(m)
            } else {
                None
            }
        });
        f(arg)
    })
}
