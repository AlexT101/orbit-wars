//! Top-level orchestrator: build ledger, run attack pass.
//!
//! Snipe is no longer a separate module — the capture decision in
//! `attack.rs` already accounts for enemy fleets weakening neutrals
//! through the `captures` simulation, so finishing-with-1-ship attacks
//! emerge from the standard scoring without dedicated logic.

use crate::attack::apply_attacks;
use crate::game::{Action, GameState};
use crate::ledger::Ledger;
use crate::opening::opening_action;

pub fn plan(state: &GameState) -> Vec<Action> {
    let mut ledger = Ledger::build(state);
    let mut out: Vec<Action> = Vec::new();
    if let Some(act) = opening_action(state, &ledger) {
        if ledger.surplus_at(act.from_id) >= act.ships {
            ledger.spend(act.from_id, act.ships);
            out.push(act);
        }
    }
    apply_attacks(state, &mut ledger, &mut out);
    out
}
