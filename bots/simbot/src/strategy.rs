//! Strategy entry points exposed to the PyO3 layer. Concrete strategy logic
//! lives in dedicated modules (e.g. [`crate::obnext`]); functions here are
//! thin orchestrators over the strategy-agnostic [`WorldState`].

#![allow(dead_code)]

use std::collections::HashSet;

use crate::engine::{EngineState, Planet};
use crate::entity_cache::EntityCache;
use crate::obnext::{
    build_mission_artifacts, plan_from_artifacts, MissionArtifacts, PlanProfile, WorldModel,
};
use crate::rollout::{opponent_turn0_variants, rollout_score};
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
    // In 2-player we score each of our candidates against up to 5 distinct
    // opponent turn-0 plans (greedy, forbid-top1, only-top1, defense-only,
    // no-op) and pick the candidate with the best worst-case score (minimax).
    // In 4-player `opponent_turn0_variants` returns a single greedy variant
    // for every opponent — same behavior as the prior shared turn-0 path.
    let opp_variants = opponent_turn0_variants(initial_state, my_player, cache);

    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, moves) in candidates.iter().enumerate() {
        let mut worst = f64::INFINITY;
        for opp in &opp_variants {
            let score = rollout_score(initial_state, my_player, moves, opp, cache);
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

/// Build a diverse candidate set (~25) for the rollout to score.
///
/// One Plan A pass over the full mission list exposes a score-ranked list of
/// offensive targets `T = [t1, t2, …]`; every other candidate is just a
/// `(profile, forbidden_targets)` permutation expressed in terms of that list.
/// All candidates share one `WorldModel` AND one `MissionArtifacts` — the
/// heavy mission generation (modes, policy, the O(my × all) sweep, multi-source
/// swarm pairing, crash exploits) runs exactly once; each candidate is then
/// just a filter + sort + commit-loop + followup + doomed + rear pass.
///
/// Candidate roster (`T_k` = first k elements of T, `opposing` = all planets
/// with owner != us, `\` = set difference):
///   * Forbid-prefix:    {}, {t1}, {t1,t2}, {t1,t2,t3}, {t1,t2,t3,t4}        (5)
///   * Only-one:         opposing\{tk} for k in 1..=5                        (5)
///   * Only-pair:        opposing\{tk,tj} for chosen (k,j) ∈ {(1,2),(1,3),
///                       (2,3),(1,4),(1,5)}                                  (5)
///   * Owner partition:  forbid neutrals, forbid hostiles, defense-only
///                       (forbid all opposing)                               (3)
///   * Skip-second:      {t2}                                                (1)
///   * No-op:            empty moves                                         (1)
///   * Fast profile:     greedy, forbid {t1}, only-t1, defense-only,
///                       only-{t1,t2}                                        (5)
pub fn obnext_candidates(world: &WorldState) -> Vec<Vec<(i64, f64, i64)>> {
    if world.my_planets.is_empty() {
        return vec![Vec::new()];
    }
    let model = WorldModel::build(world);
    // Shared across every candidate: modes, policy, full mission list. The
    // expensive O(my × all) sweep + swarm pairing + crash exploit only runs
    // here, not per candidate.
    let artifacts = build_mission_artifacts(&model);

    // Plan A: greedy full. Drives the offensive-target ordering for every
    // forbid/only candidate below.
    let plan_a = plan_from_artifacts(&model, &artifacts, PlanProfile::full(), &HashSet::new());
    let targets = &plan_a.offensive_targets;
    let take = |k: usize| -> Option<i64> { targets.get(k).copied() };

    let opposing: HashSet<i64> = world
        .planets
        .iter()
        .filter(|p| p.owner != world.player)
        .map(|p| p.id)
        .collect();
    let neutrals: HashSet<i64> = world
        .planets
        .iter()
        .filter(|p| p.owner == -1)
        .map(|p| p.id)
        .collect();
    let hostiles: HashSet<i64> = world
        .planets
        .iter()
        .filter(|p| p.owner != world.player && p.owner != -1)
        .map(|p| p.id)
        .collect();

    let forbid_prefix = |k: usize| -> HashSet<i64> { targets.iter().take(k).copied().collect() };
    let only = |ids: &[i64]| -> HashSet<i64> {
        let mut f = opposing.clone();
        for id in ids {
            f.remove(id);
        }
        f
    };

    let mut emitter = CandidateEmitter::new(plan_a.moves);

    // Forbid-prefix: 1..=4 (k=0 is Plan A, already emitted).
    for k in 1..=4 {
        emitter.run(&model, &artifacts, PlanProfile::full(), &forbid_prefix(k));
    }

    // Only-one: focus all offense on a single top target.
    for k in 0..5 {
        if let Some(t) = take(k) {
            emitter.run(&model, &artifacts, PlanProfile::full(), &only(&[t]));
        }
    }

    // Only-pair: two-target focus across a few useful combinations.
    let pairs: [(usize, usize); 5] = [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4)];
    for (i, j) in pairs {
        if let (Some(ti), Some(tj)) = (take(i), take(j)) {
            emitter.run(&model, &artifacts, PlanProfile::full(), &only(&[ti, tj]));
        }
    }

    // Owner-type partition. `neutrals` and `hostiles` may be subsets of
    // `opposing`; emitting all three still exercises distinct strategies.
    emitter.run(&model, &artifacts, PlanProfile::full(), &neutrals);
    emitter.run(&model, &artifacts, PlanProfile::full(), &hostiles);
    emitter.run(&model, &artifacts, PlanProfile::full(), &opposing);

    // Skip-second: keep t1 but force a different runner-up. Distinct from
    // forbid-prefix variants because t1 remains the top attack.
    if let Some(t2) = take(1) {
        let mut f = HashSet::new();
        f.insert(t2);
        emitter.run(&model, &artifacts, PlanProfile::full(), &f);
    }

    // No-op: hold all ships.
    emitter.push_raw(Vec::new());

    // Fast-profile variants on the most useful structural choices.
    emitter.run(&model, &artifacts, PlanProfile::fast(), &HashSet::new());
    if let Some(t1) = take(0) {
        let mut f = HashSet::new();
        f.insert(t1);
        emitter.run(&model, &artifacts, PlanProfile::fast(), &f);
        emitter.run(&model, &artifacts, PlanProfile::fast(), &only(&[t1]));
    }
    emitter.run(&model, &artifacts, PlanProfile::fast(), &opposing);
    if let (Some(t1), Some(t2)) = (take(0), take(1)) {
        emitter.run(&model, &artifacts, PlanProfile::fast(), &only(&[t1, t2]));
    }

    emitter.into_candidates()
}

/// Accumulates candidate plans while dropping move-list duplicates. Two
/// candidates that produce the same plan (common when `offensive_targets` is
/// short — Only t1 and forbid {t2} can both collapse to Plan A) skip the
/// rollout entirely.
struct CandidateEmitter {
    candidates: Vec<Vec<(i64, f64, i64)>>,
    seen: HashSet<Vec<(i64, u64, i64)>>,
}

impl CandidateEmitter {
    fn new(first: Vec<(i64, f64, i64)>) -> Self {
        let mut e = Self {
            candidates: Vec::new(),
            seen: HashSet::new(),
        };
        e.push_raw(first);
        e
    }

    fn run(
        &mut self,
        model: &WorldModel,
        artifacts: &MissionArtifacts,
        profile: PlanProfile,
        forbid: &HashSet<i64>,
    ) {
        let plan = plan_from_artifacts(model, artifacts, profile, forbid);
        self.push_raw(plan.moves);
    }

    fn push_raw(&mut self, moves: Vec<(i64, f64, i64)>) {
        if self.seen.insert(dedup_key(&moves)) {
            self.candidates.push(moves);
        }
    }

    fn into_candidates(self) -> Vec<Vec<(i64, f64, i64)>> {
        self.candidates
    }
}

/// Order-insensitive fingerprint for move-list dedup. `to_bits` keeps the
/// angle comparison exact: two plans built off the same `WorldModel` hit the
/// same `plan_shot` cache entries and produce bit-identical angles.
fn dedup_key(moves: &[(i64, f64, i64)]) -> Vec<(i64, u64, i64)> {
    let mut key: Vec<(i64, u64, i64)> = moves
        .iter()
        .map(|&(src, angle, ships)| (src, angle.to_bits(), ships))
        .collect();
    key.sort_unstable();
    key
}
