//! Opponent-modeled rollout for scoring candidate plans.
//!
//! Structure: 4 turns of full simulation where every player plans with the
//! fast obnext profile, then 20 turns of "ballistic" stepping with no new
//! launches (in-flight fleets keep moving, combat resolves, planets produce).
//!
//! This catches the failure mode where a plan looks good against a no-op
//! opponent but loses to a real reaction — and skips planning during the
//! ballistic phase where decisions don't change the outcome inside our scoring
//! window.

#![allow(dead_code)]

use crate::engine::{EngineState, MoveAction};
use crate::entity_cache::EntityCache;
use crate::obnext::{plan_with_profile, PlanProfile};
use crate::world::WorldState;

pub const REACTIVE_TURNS: i64 = 5;
pub const BALLISTIC_TURNS: i64 = 20;

/// Multiplier on the opponent's total contribution to the score. Compensates
/// for the rollout using stripped obnext (`PlanProfile::fast`) for opponents
/// rather than the full planner — real opponents play stronger than our
/// model. Applied symmetrically to opponent production and ship terms; ships
/// themselves use weight 1.0 on both sides.
const OPPONENT_PESSIMISM: f64 = 1.0;

/// Score a candidate plan by simulating it forward.
///
/// The caller passes a pre-built `EngineState` reflecting the current turn;
/// this function clones it and steps the clone, leaving the caller's state
/// untouched. `cache.current_turn` is restored before returning.
pub fn rollout_score(
    initial_state: &EngineState,
    my_player: i64,
    my_moves: &[(i64, f64, i64)],
    cache: &mut EntityCache,
) -> f64 {
    let saved_turn = cache.current_turn;
    let num_players = initial_state.num_players;
    let mut engine = initial_state.clone();

    for t in 0..REACTIVE_TURNS {
        if engine.done {
            break;
        }
        cache.set_current_turn(engine.step);
        let mut actions: Vec<Vec<MoveAction>> = vec![Vec::new(); num_players];
        for p in 0..num_players {
            let pid = p as i64;
            if pid == my_player && t == 0 {
                actions[p] = to_move_actions(my_moves);
                continue;
            }
            let ws = WorldState::build(
                pid,
                engine.step,
                engine.planets.clone(),
                engine.fleets.clone(),
                engine.initial_planets.clone(),
                engine.comets.clone(),
                engine.comet_planet_ids.clone(),
                engine.angular_velocity,
                cache,
            );
            actions[p] = to_move_actions(&plan_with_profile(&ws, PlanProfile::full()));
        }
        if engine.step_with_actions(&actions).is_err() {
            break;
        }
    }

    // Ballistic phase: no new launches. Fleets in flight still resolve.
    let empty: Vec<Vec<MoveAction>> = vec![Vec::new(); num_players];
    for _ in 0..BALLISTIC_TURNS {
        if engine.done {
            break;
        }
        if engine.step_with_actions(&empty).is_err() {
            break;
        }
    }

    cache.set_current_turn(saved_turn);
    score_state(&engine, my_player)
}

fn to_move_actions(moves: &[(i64, f64, i64)]) -> Vec<MoveAction> {
    moves
        .iter()
        .map(|&(from_id, angle, ships)| MoveAction {
            from_id,
            angle,
            ships,
        })
        .collect()
}

/// Production-weighted board control delta from `my_player`'s perspective.
/// Counts owned-planet production over the remaining game, current ship
/// inventories on planets, and ships in flight. Opponent contribution is
/// scaled by `OPPONENT_PESSIMISM` to compensate for under-modeling them with
/// stripped obnext.
fn score_state(engine: &EngineState, my_player: i64) -> f64 {
    let remaining = (engine.configuration.episode_steps - engine.step).max(0) as f64;
    let mut my_score = 0.0;
    let mut enemy_score = 0.0;
    for planet in &engine.planets {
        if planet.owner == my_player {
            my_score += planet.production as f64 * remaining;
            my_score += planet.ships as f64;
        } else if planet.owner != -1 {
            enemy_score += planet.production as f64 * remaining;
            enemy_score += planet.ships as f64;
        }
    }
    for fleet in &engine.fleets {
        if fleet.owner == my_player {
            my_score += fleet.ships as f64;
        } else {
            enemy_score += fleet.ships as f64;
        }
    }
    my_score - OPPONENT_PESSIMISM * enemy_score
}
