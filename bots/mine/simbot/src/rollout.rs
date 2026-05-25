//! Strategy-agnostic rollout/search infrastructure for scoring candidate plans.
//!
//! Structure: 2 turns of full simulation where every player replans via the
//! supplied planner hook, then 20 turns of "ballistic" stepping with no new
//! launches (in-flight fleets keep moving, combat resolves, planets produce).

#![allow(dead_code)]

use crate::engine::{EngineState, MoveAction};
use crate::entity_cache::EntityCache;
use crate::world::WorldState;

pub const REACTIVE_TURNS: i64 = 2;
pub const BALLISTIC_TURNS: i64 = 20;

pub type PlannedMove = (i64, f64, i64);
pub type PlanFn = for<'a> fn(&WorldState<'a>) -> Vec<PlannedMove>;
pub type CandidateFn = for<'a> fn(&WorldState<'a>) -> Vec<Vec<PlannedMove>>;

/// Score pre-built candidate plans via rollout and return the best. The
/// planner hooks make the rollout/search layer strategy-agnostic: any policy
/// that can produce a greedy reply plan and (optionally) a wider candidate set
/// for 2-player opponent turn-0 minimax can plug in here.
pub fn pick_plan_by_rollout(
    initial_state: &EngineState,
    my_player: i64,
    candidates: Vec<Vec<PlannedMove>>,
    reply_plan_fn: PlanFn,
    opponent_candidate_fn: CandidateFn,
    cache: &mut EntityCache,
) -> Vec<PlannedMove> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let opp_variants = opponent_turn0_variants(
        initial_state,
        my_player,
        reply_plan_fn,
        opponent_candidate_fn,
        cache,
    );

    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, moves) in candidates.iter().enumerate() {
        let mut worst = f64::INFINITY;
        for opp in &opp_variants {
            let score = rollout_score(initial_state, my_player, moves, opp, reply_plan_fn, cache);
            if score < worst {
                worst = score;
            }
        }
        if worst > best_score {
            best_score = worst;
            best_idx = i;
        }
    }
    candidates.into_iter().nth(best_idx).unwrap_or_default()
}

/// Score a candidate plan by simulating it forward.
///
/// The caller passes a pre-built `EngineState` reflecting the current turn;
/// this function clones it and steps the clone, leaving the caller's state
/// untouched. `cache.current_turn` is restored before returning.
///
/// `turn0_opponents` is the per-player opponent action list for the very
/// first rollout step. The initial engine state is identical across every
/// candidate, so opponent turn-0 plans don't depend on `my_moves` — the
/// caller (`pick_plan_by_rollout`) computes the variant roster once via
/// [`opponent_turn0_variants`] and reuses it across all candidates. The slot
/// for `my_player` is ignored.
pub fn rollout_score(
    initial_state: &EngineState,
    my_player: i64,
    my_moves: &[PlannedMove],
    turn0_opponents: &[Vec<MoveAction>],
    reply_plan_fn: PlanFn,
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
            if pid == my_player {
                if t == 0 {
                    actions[p] = to_move_actions(my_moves);
                    continue;
                }
            } else if t == 0 {
                actions[p] = turn0_opponents[p].clone();
                continue;
            }
            let ws = WorldState::from_engine(pid, &engine, cache);
            actions[p] = to_move_actions(&reply_plan_fn(&ws));
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

/// Precompute opponent turn-0 plans from the shared initial state. `my_player`'s
/// slot is left empty (the caller fills it per-candidate). `cache.current_turn`
/// is restored before returning.
pub fn opponent_turn0_actions(
    initial_state: &EngineState,
    my_player: i64,
    reply_plan_fn: PlanFn,
    cache: &mut EntityCache,
) -> Vec<Vec<MoveAction>> {
    let saved_turn = cache.current_turn;
    cache.set_current_turn(initial_state.step);
    let num_players = initial_state.num_players;
    let mut actions: Vec<Vec<MoveAction>> = vec![Vec::new(); num_players];
    for p in 0..num_players {
        let pid = p as i64;
        if pid == my_player {
            continue;
        }
        let ws = WorldState::from_engine(pid, initial_state, cache);
        actions[p] = to_move_actions(&reply_plan_fn(&ws));
    }
    cache.set_current_turn(saved_turn);
    actions
}

/// Build up to K distinct opponent turn-0 action sets for minimax-style
/// scoring. In 2-player games we reuse the same full-search candidate builder
/// as our own side, evaluated from the opponent's POV, and return one
/// per-player action layout per variant. In games with more players we skip
/// the combinatorial opponent options and fall back to a single greedy variant
/// for every opponent. `cache.current_turn` is restored before returning.
pub fn opponent_turn0_variants(
    initial_state: &EngineState,
    my_player: i64,
    reply_plan_fn: PlanFn,
    opponent_candidate_fn: CandidateFn,
    cache: &mut EntityCache,
) -> Vec<Vec<Vec<MoveAction>>> {
    let num_players = initial_state.num_players;
    if num_players != 2 {
        return vec![opponent_turn0_actions(
            initial_state,
            my_player,
            reply_plan_fn,
            cache,
        )];
    }
    let Some(opp_player) = (0..num_players as i64).find(|&p| p != my_player) else {
        return vec![opponent_turn0_actions(
            initial_state,
            my_player,
            reply_plan_fn,
            cache,
        )];
    };

    let saved_turn = cache.current_turn;
    cache.set_current_turn(initial_state.step);

    let opp_ws = WorldState::from_engine(opp_player, initial_state, cache);

    if opp_ws.my_planets.is_empty() {
        cache.set_current_turn(saved_turn);
        let mut single = vec![Vec::new(); num_players];
        single[opp_player as usize] = Vec::new();
        return vec![single];
    }
    let variants = opponent_candidate_fn(&opp_ws);

    cache.set_current_turn(saved_turn);

    variants
        .into_iter()
        .map(|opp_moves| {
            let mut per_player: Vec<Vec<MoveAction>> = vec![Vec::new(); num_players];
            per_player[opp_player as usize] = to_move_actions(&opp_moves);
            per_player
        })
        .collect()
}

fn to_move_actions(moves: &[PlannedMove]) -> Vec<MoveAction> {
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
/// inventories on planets, and ships in flight.
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
    my_score - enemy_score
}
