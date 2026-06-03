//! Strategy-agnostic rollout/search infrastructure for scoring candidate plans.
//!
//! Structure: 2 turns of full simulation where every player replans via the
//! supplied planner hook, then HORIZON turns of "ballistic" stepping with no new
//! launches (in-flight fleets keep moving, combat resolves, planets produce).

use crate::cache::EntityCache;
use crate::constants::{EPISODE_STEPS, HORIZON, REACTIVE_TURNS};
use crate::engine::Simulator;
use crate::engine::{EngineState, Fleet, MoveAction, Planet};
use crate::helpers::ArrivalLedger;
use crate::world::{ShotL1, WorldState};

pub type PlanFn = for<'a> fn(&WorldState<'a>) -> Vec<MoveAction>;
pub type CandidateFn = for<'a> fn(&WorldState<'a>) -> Vec<Vec<MoveAction>>;

/// Score pre-built candidate plans via rollout and return the best. The
/// planner hooks make the rollout/search layer strategy-agnostic: any policy
/// that can produce a greedy reply plan and (optionally) a wider candidate set
/// for 2-player opponent turn-0 minimax can plug in here.
pub fn pick_plan_by_rollout(
    initial_state: &EngineState,
    my_player: i64,
    candidates: Vec<Vec<MoveAction>>,
    reply_plan_fn: PlanFn,
    opponent_candidate_fn: CandidateFn,
    cache: &mut EntityCache,
    remaining_overage_time: f64,
    shared_ledger: Option<&ArrivalLedger>,
    shot_l1: Option<&ShotL1>,
) -> Vec<MoveAction> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let opp_variants = opponent_turn0_variants(
        initial_state,
        my_player,
        reply_plan_fn,
        opponent_candidate_fn,
        cache,
        remaining_overage_time,
        shared_ledger,
        shot_l1,
    );

    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, moves) in candidates.iter().enumerate() {
        let mut worst = f64::INFINITY;
        for opp in &opp_variants {
            let score = rollout_score(
                initial_state,
                my_player,
                moves,
                opp,
                reply_plan_fn,
                cache,
                remaining_overage_time,
                shot_l1,
            );
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

/// Score a candidate plan by simulating it forward. `cache.current_turn` is
/// restored before returning.
///
/// `turn0_opponents` is reused across all candidates since the initial state
/// is shared; the slot for `my_player` is ignored.
pub fn rollout_score(
    initial_state: &EngineState,
    my_player: i64,
    my_moves: &[MoveAction],
    turn0_opponents: &[Vec<MoveAction>],
    reply_plan_fn: PlanFn,
    cache: &mut EntityCache,
    remaining_overage_time: f64,
    shot_l1: Option<&ShotL1>,
) -> f64 {
    let saved_turn = cache.current_turn;
    let num_players = initial_state.num_players;
    let mut sim = Simulator::new(initial_state);
    // Scoring reads only planets/fleets; the per-turn ledger forks the sim
    // (forks always record), so the scoring sim itself needn't log events.
    sim.set_record_events(false);

    for t in 0..REACTIVE_TURNS {
        if sim.step_count() >= EPISODE_STEPS {
            break;
        }
        cache.set_current_turn(sim.step_count());
        let mut actions: Vec<Vec<MoveAction>> = vec![Vec::new(); num_players];

        // Share one arrival ledger across all players (sim-walk is
        // player-agnostic). Turn 0 uses pre-baked actions, so skip it.
        let ledger: Option<ArrivalLedger> = if t == 0 {
            None
        } else {
            Some(ArrivalLedger::build(&sim, HORIZON, cache))
        };

        for p in 0..num_players {
            let pid = p as i64;
            if pid == my_player {
                if t == 0 {
                    actions[p] = my_moves.to_vec();
                    continue;
                }
            } else if t == 0 {
                actions[p] = turn0_opponents[p].clone();
                continue;
            }
            let ledger = ledger.as_ref().expect("ledger built for t >= 1");
            let mut ws = WorldState::from_simulator_with_ledger(pid, &sim, ledger, cache);
            ws.remaining_overage_time = remaining_overage_time;
            ws.shot_l1 = shot_l1;
            actions[p] = reply_plan_fn(&ws);
        }
        let action_slices: Vec<&[MoveAction]> = actions.iter().map(|v| v.as_slice()).collect();
        sim.step_with_actions(&action_slices, Some(&*cache));
    }

    // Ballistic phase: no new launches. Fleets in flight still resolve.
    for _ in 0..HORIZON {
        if sim.step_count() >= EPISODE_STEPS {
            break;
        }
        sim.step(Some(&*cache));
    }

    cache.set_current_turn(saved_turn);
    score_simulation(&sim, my_player)
}

/// Precompute opponent turn-0 plans from the shared initial state. `my_player`'s
/// slot is left empty (the caller fills it per-candidate). `cache.current_turn`
/// is restored before returning.
pub fn opponent_turn0_actions(
    initial_state: &EngineState,
    my_player: i64,
    reply_plan_fn: PlanFn,
    cache: &mut EntityCache,
    remaining_overage_time: f64,
    shared_ledger: Option<&ArrivalLedger>,
    shot_l1: Option<&ShotL1>,
) -> Vec<Vec<MoveAction>> {
    let saved_turn = cache.current_turn;
    cache.set_current_turn(initial_state.step);
    let num_players = initial_state.num_players;
    let sim = Simulator::new(initial_state);
    // The turn-0 ledger is player-agnostic; reuse the caller's if supplied so we
    // don't repeat the O(HORIZON * planets) forward walk.
    let owned_ledger;
    let ledger = match shared_ledger {
        Some(l) => l,
        None => {
            owned_ledger = ArrivalLedger::build(&sim, HORIZON, cache);
            &owned_ledger
        }
    };
    let mut actions: Vec<Vec<MoveAction>> = vec![Vec::new(); num_players];
    for p in 0..num_players {
        let pid = p as i64;
        if pid == my_player {
            continue;
        }
        let mut ws = WorldState::from_simulator_with_ledger(pid, &sim, ledger, cache);
        ws.remaining_overage_time = remaining_overage_time;
        ws.shot_l1 = shot_l1;
        actions[p] = reply_plan_fn(&ws);
    }
    cache.set_current_turn(saved_turn);
    actions
}

/// Build up to K distinct opponent turn-0 action sets for minimax-style
/// scoring. 2-player games reuse the full-search candidate builder from the
/// opponent's POV; more-player games fall back to a single greedy variant per
/// opponent to avoid combinatorial blow-up. `cache.current_turn` is restored.
pub fn opponent_turn0_variants(
    initial_state: &EngineState,
    my_player: i64,
    reply_plan_fn: PlanFn,
    opponent_candidate_fn: CandidateFn,
    cache: &mut EntityCache,
    remaining_overage_time: f64,
    shared_ledger: Option<&ArrivalLedger>,
    shot_l1: Option<&ShotL1>,
) -> Vec<Vec<Vec<MoveAction>>> {
    let num_players = initial_state.num_players;
    if num_players != 2 {
        return vec![opponent_turn0_actions(
            initial_state,
            my_player,
            reply_plan_fn,
            cache,
            remaining_overage_time,
            shared_ledger,
            shot_l1,
        )];
    }
    let Some(opp_player) = (0..num_players as i64).find(|&p| p != my_player) else {
        return vec![opponent_turn0_actions(
            initial_state,
            my_player,
            reply_plan_fn,
            cache,
            remaining_overage_time,
            shared_ledger,
            shot_l1,
        )];
    };

    let saved_turn = cache.current_turn;
    cache.set_current_turn(initial_state.step);

    // Player-agnostic turn-0 ledger; reuse the caller's if supplied.
    let sim = Simulator::new(initial_state);
    let owned_ledger;
    let ledger = match shared_ledger {
        Some(l) => l,
        None => {
            owned_ledger = ArrivalLedger::build(&sim, HORIZON, cache);
            &owned_ledger
        }
    };
    let mut opp_ws = WorldState::from_simulator_with_ledger(opp_player, &sim, ledger, cache);
    opp_ws.remaining_overage_time = remaining_overage_time;
    opp_ws.shot_l1 = shot_l1;

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
            per_player[opp_player as usize] = opp_moves;
            per_player
        })
        .collect()
}

/// Production-weighted board control delta from `my_player`'s perspective.
/// Counts owned-planet production over the remaining game, current ship
/// inventories on planets, and ships in flight.
fn score_simulation(sim: &Simulator, my_player: i64) -> f64 {
    score_snapshot(sim.planets(), sim.fleets(), sim.step_count(), my_player)
}

fn score_snapshot(planets: &[Planet], fleets: &[Fleet], step: i64, my_player: i64) -> f64 {
    let remaining = (EPISODE_STEPS - step).max(0) as f64;
    let mut my_score = 0.0;
    let mut enemy_score = 0.0;
    for planet in planets {
        if planet.owner == my_player {
            my_score += planet.production as f64 * remaining;
            my_score += planet.ships as f64;
        } else if planet.owner != -1 {
            enemy_score += planet.production as f64 * remaining;
            enemy_score += planet.ships as f64;
        }
    }
    for fleet in fleets {
        if fleet.owner == my_player {
            my_score += fleet.ships as f64;
        } else {
            enemy_score += fleet.ships as f64;
        }
    }
    my_score - enemy_score
}
