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

use rustc_hash::FxHashSet as HashSet;

use crate::engine::{EngineState, MoveAction};
use crate::entity_cache::EntityCache;
use crate::obnext::{
    build_mission_artifacts, plan_from_artifacts, plan_with_profile, PlanProfile, WorldModel,
};
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
///
/// `turn0_opponents` is the per-player opponent action list for the very
/// first rollout step. The initial engine state is identical across every
/// candidate, so opponent turn-0 plans don't depend on `my_moves` — the
/// caller (`pick_plan_by_rollout`) computes them once via
/// [`opponent_turn0_actions`] and reuses them, skipping ~N-1 obnext-full
/// plans per candidate. The slot for `my_player` is ignored.
pub fn rollout_score(
    initial_state: &EngineState,
    my_player: i64,
    my_moves: &[(i64, f64, i64)],
    turn0_opponents: &[Vec<MoveAction>],
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

/// Precompute opponent turn-0 plans from the shared initial state. `my_player`'s
/// slot is left empty (the caller fills it per-candidate). `cache.current_turn`
/// is restored before returning.
pub fn opponent_turn0_actions(
    initial_state: &EngineState,
    my_player: i64,
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
        actions[p] = to_move_actions(&plan_with_profile(&ws, PlanProfile::full()));
    }
    cache.set_current_turn(saved_turn);
    actions
}

/// Build up to K distinct opponent turn-0 action sets for minimax-style
/// scoring. In 2-player games we generate ~5 opp plan variants from the
/// opponent's POV (greedy, forbid-top1, only-top1, defense-only, no-op),
/// dedup by move-list, and return one per-player action layout per variant.
/// In games with more players we fall back to a single variant (the greedy
/// plan for every opponent) — varying multiple opponents jointly explodes
/// combinatorially. The caller minimaxes our candidates against the returned
/// variants. `cache.current_turn` is restored before returning.
pub fn opponent_turn0_variants(
    initial_state: &EngineState,
    my_player: i64,
    cache: &mut EntityCache,
) -> Vec<Vec<Vec<MoveAction>>> {
    let num_players = initial_state.num_players;
    if num_players != 2 {
        return vec![opponent_turn0_actions(initial_state, my_player, cache)];
    }
    let Some(opp_player) = (0..num_players as i64).find(|&p| p != my_player) else {
        return vec![opponent_turn0_actions(initial_state, my_player, cache)];
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

    let opp_model = WorldModel::build(&opp_ws);
    let opp_artifacts = build_mission_artifacts(&opp_model);

    let plan_greedy =
        plan_from_artifacts(&opp_model, &opp_artifacts, PlanProfile::full(), &HashSet::default());
    let opp_targets = plan_greedy.offensive_targets.clone();
    let opposing: HashSet<i64> = opp_ws
        .planets
        .iter()
        .filter(|p| p.owner != opp_player)
        .map(|p| p.id)
        .collect();

    let mut seen: HashSet<Vec<(i64, u64, i64)>> = HashSet::default();
    let mut variants: Vec<Vec<(i64, f64, i64)>> = Vec::new();
    let push = |moves: Vec<(i64, f64, i64)>,
                    seen: &mut HashSet<Vec<(i64, u64, i64)>>,
                    variants: &mut Vec<Vec<(i64, f64, i64)>>| {
        let mut key: Vec<(i64, u64, i64)> = moves
            .iter()
            .map(|&(src, angle, ships)| (src, angle.to_bits(), ships))
            .collect();
        key.sort_unstable();
        if seen.insert(key) {
            variants.push(moves);
        }
    };

    push(plan_greedy.moves, &mut seen, &mut variants);
    if let Some(t1) = opp_targets.first().copied() {
        let mut forbid_t1 = HashSet::default();
        forbid_t1.insert(t1);
        let p = plan_from_artifacts(&opp_model, &opp_artifacts, PlanProfile::full(), &forbid_t1);
        push(p.moves, &mut seen, &mut variants);

        let mut only_t1 = opposing.clone();
        only_t1.remove(&t1);
        let p = plan_from_artifacts(&opp_model, &opp_artifacts, PlanProfile::full(), &only_t1);
        push(p.moves, &mut seen, &mut variants);
    }
    let defense_only =
        plan_from_artifacts(&opp_model, &opp_artifacts, PlanProfile::full(), &opposing);
    push(defense_only.moves, &mut seen, &mut variants);
    push(Vec::new(), &mut seen, &mut variants);

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
