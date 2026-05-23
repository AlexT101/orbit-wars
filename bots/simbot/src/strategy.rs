//! Strategy entry points exposed to the PyO3 layer. Concrete strategy logic
//! lives in dedicated modules (e.g. [`crate::obnext`]); functions here are
//! thin orchestrators over the strategy-agnostic [`WorldState`].

#![allow(dead_code)]

use std::collections::HashSet;

use crate::engine::{EngineState, Planet};
use crate::entity_cache::EntityCache;
use crate::obnext::{plan_moves_full, PlanProfile, WorldModel};
use crate::rollout::rollout_score;
use crate::world::WorldState;

/// Nearest-sniper baseline: for each owned planet, send `garrison + 1` ships
/// at the closest non-owned planet when affordable.
pub fn nearest_sniper(world: &WorldState) -> Vec<(i64, f64, i64)> {
    let mut moves = Vec::new();
    if world.my_planets.is_empty() {
        return moves;
    }
    let targets: Vec<&Planet> = world
        .enemy_planets
        .iter()
        .chain(world.neutral_planets.iter())
        .collect();
    if targets.is_empty() {
        return moves;
    }
    for m in &world.my_planets {
        let mut nearest: Option<&Planet> = None;
        let mut best = f64::INFINITY;
        for t in &targets {
            let dx = m.x - t.x;
            let dy = m.y - t.y;
            let d = (dx * dx + dy * dy).sqrt();
            if d < best {
                best = d;
                nearest = Some(*t);
            }
        }
        let Some(t) = nearest else { continue };
        let needed = t.ships + 1;
        if m.ships >= needed {
            let angle = (t.y - m.y).atan2(t.x - m.x);
            moves.push((m.id, angle, needed));
        }
    }
    moves
}

pub fn obnext(world: &WorldState) -> Vec<(i64, f64, i64)> {
    crate::obnext::plan(world)
}

/// Score pre-built candidate plans via rollout and return the best. Kept
/// separate from plan generation because `WorldState` borrows `EntityCache`,
/// so plans must be generated and dropped before the rollout reborrows the
/// cache mutably.
pub fn pick_plan_by_rollout(
    initial_state: &EngineState,
    my_player: i64,
    candidates: Vec<Vec<(i64, f64, i64)>>,
    cache: &mut EntityCache,
) -> Vec<(i64, f64, i64)> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, moves) in candidates.iter().enumerate() {
        let score = rollout_score(initial_state, my_player, moves, cache);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    candidates.into_iter().nth(best_idx).unwrap_or_default()
}

/// Beam width: number of candidate plans evaluated by the rollout-based
/// search. Currently supports 3, 5, 6, or 8.
///
/// At 3: greedy-full, forbid-top1-full, fast.
/// At 5: adds forbid-top2-full (deeper alternative attack focus) and
///   forbid-top1-fast (conservative variant on a different attack target).
/// At 6: adds no-op (hold all ships) — catches turns where active play
///   actually loses to a no-action baseline.
/// At 8: adds only-top-of-A and only-top-of-B — commits the entire top
///   offensive mission (full swarm fragments if it's a Swarm) plus all
///   defensive moves, skipping every other offensive mission. Tests
///   "focus exclusively on the top attack, hold everything else."
pub const BEAM_WIDTH: usize = 8;

/// Build a small diverse candidate set from a WorldState. Candidates explore
/// two axes: which offensive target is forbidden (forces different primary
/// attacks) and which profile is used (full vs fast). Empty `top_offensive_target`
/// just shortens the returned set rather than padding with duplicates.
pub fn obnext_candidates(world: &WorldState) -> Vec<Vec<(i64, f64, i64)>> {
    if world.my_planets.is_empty() {
        return vec![Vec::new()];
    }
    let model = WorldModel::build(world);

    // Plan A: greedy full.
    let a = plan_moves_full(&model, PlanProfile::full(), &HashSet::new());
    let top1 = a.top_offensive_target;
    let a_moves = a.moves;

    // Plan B: full, forbid top-1. The returned top_offensive_target here is
    // the *second*-highest scoring offensive target after top-1 is removed —
    // reused below for the forbid-top-2 plan.
    let mut forbid1: HashSet<i64> = HashSet::new();
    let mut top2: Option<i64> = None;
    let mut b_moves: Option<Vec<(i64, f64, i64)>> = None;
    if let Some(t) = top1 {
        forbid1.insert(t);
        let b = plan_moves_full(&model, PlanProfile::full(), &forbid1);
        top2 = b.top_offensive_target;
        b_moves = Some(b.moves);
    }

    let mut candidates = vec![a_moves];
    if let Some(b_m) = b_moves {
        candidates.push(b_m);
    }

    if BEAM_WIDTH >= 5 {
        // Plan C: full, forbid top-1 and top-2 — deeper alternative focus.
        if let Some(t2) = top2 {
            let mut forbid2 = forbid1.clone();
            forbid2.insert(t2);
            let c = plan_moves_full(&model, PlanProfile::full(), &forbid2);
            candidates.push(c.moves);
        }
        // Plan D: fast, forbid top-1 — conservative variant on a different attack.
        if !forbid1.is_empty() {
            let d = plan_moves_full(&model, PlanProfile::fast(), &forbid1);
            candidates.push(d.moves);
        }
    }

    // Plan E: fast, no forbid — conservative fallback.
    let last = plan_moves_full(&model, PlanProfile::fast(), &HashSet::new());
    candidates.push(last.moves);

    if BEAM_WIDTH >= 6 {
        // No-op: hold all ships this turn. Wins on turns where any active play
        // makes us worse off (e.g. ill-timed commits against a stronger board).
        candidates.push(Vec::new());
    }
    if BEAM_WIDTH >= 8 {
        // Only-top-of-X: forbid every opposing planet *except* the chosen
        // target so the commit walk's offensive options are restricted to
        // exactly that one mission (including all swarm fragments if it's a
        // Swarm). Defensive missions (reinforce/rescue/recapture) target our
        // own planets and are untouched by the filter.
        let opposing: HashSet<i64> = world
            .planets
            .iter()
            .filter(|p| p.owner != world.player)
            .map(|p| p.id)
            .collect();

        if let Some(t) = top1 {
            let mut forbid_except_t = opposing.clone();
            forbid_except_t.remove(&t);
            let only_a = plan_moves_full(&model, PlanProfile::full(), &forbid_except_t);
            candidates.push(only_a.moves);
        }
        if let Some(t2) = top2 {
            let mut forbid_except_t2 = opposing.clone();
            forbid_except_t2.remove(&t2);
            let only_b = plan_moves_full(&model, PlanProfile::full(), &forbid_except_t2);
            candidates.push(only_b.moves);
        }
    }

    candidates
}
