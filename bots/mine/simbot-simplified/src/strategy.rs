//! Strategy entry points exposed to the PyO3 layer. Concrete strategy logic
//! lives in dedicated modules (e.g. [`crate::obnext`]); functions here are
//! thin orchestrators over the strategy-agnostic [`WorldState`].

#![allow(dead_code)]

use rustc_hash::FxHashSet as HashSet;

use crate::engine::EngineState;
use crate::entity_cache::EntityCache;
use crate::obnext::{
    build_mission_artifacts, patient_targets, plan_from_artifacts, MissionArtifacts,
    PlanConstraints, PlanProfile, WorldModel,
};
use crate::rollout::{opponent_turn0_variants, rollout_score};
use crate::world::WorldState;

const OFFENSE_REMOVE_POOL: usize = 5;
const OFFENSE_REMOVE_DEPTH: usize = 2;
const OFFENSE_KEEP_ONLY_POOL: usize = 4;
const OFFENSE_KEEP_ONLY_DEPTH: usize = 2;

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
    // In 2-player we score each of our candidates against a wider roster of
    // opponent turn-0 plans and pick the candidate with the best worst-case
    // score (minimax).
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

/// Build a rollout candidate set by perturbing the offensive missions the
/// greedy planner actually accepted. We try two neighborhoods from a shared
/// artifact build:
///   * remove-and-refill: block some accepted missions and let greedy refill
///   * keep-only: allow only a small accepted subset (including the empty set)
///
/// We run both an eager branch and a patience-gated branch. All candidates
/// still share one `WorldModel` and one `MissionArtifacts`, so the extra cost
/// is just the additional commit/filter passes and rollout scoring.
pub fn obnext_candidates(world: &WorldState) -> Vec<Vec<(i64, f64, i64)>> {
    if world.my_planets.is_empty() {
        return vec![Vec::new()];
    }
    let model = WorldModel::build(world);
    let artifacts = build_mission_artifacts(&model);

    let eager_constraints = PlanConstraints::default();
    let eager_plan = plan_from_artifacts(&model, &artifacts, PlanProfile::full(), &eager_constraints);

    let patient_constraints = PlanConstraints {
        forbidden_targets: patient_targets(&model, &artifacts),
        ..PlanConstraints::default()
    };
    let patient_plan =
        plan_from_artifacts(&model, &artifacts, PlanProfile::full(), &patient_constraints);

    let mut emitter = CandidateEmitter::new(eager_plan.moves);
    emit_bundle_branch(
        &mut emitter,
        &model,
        &artifacts,
        &eager_plan.accepted_offense,
        &eager_constraints,
        true,
    );

    emitter.push_raw(patient_plan.moves);
    emit_bundle_branch(
        &mut emitter,
        &model,
        &artifacts,
        &patient_plan.accepted_offense,
        &patient_constraints,
        false,
    );

    emitter.into_candidates()
}

fn emit_bundle_branch(
    emitter: &mut CandidateEmitter,
    model: &WorldModel,
    artifacts: &MissionArtifacts,
    accepted_offense: &[usize],
    base_constraints: &PlanConstraints,
    include_noop: bool,
) {
    emit_remove_and_refill_variants(emitter, model, artifacts, accepted_offense, base_constraints);
    emit_keep_only_variants(emitter, model, artifacts, accepted_offense, base_constraints);
    if include_noop {
        emitter.push_raw(Vec::new());
    }
}

fn emit_remove_and_refill_variants(
    emitter: &mut CandidateEmitter,
    model: &WorldModel,
    artifacts: &MissionArtifacts,
    accepted_offense: &[usize],
    base_constraints: &PlanConstraints,
) {
    let pool = &accepted_offense[..accepted_offense.len().min(OFFENSE_REMOVE_POOL)];
    for blocked_ids in mission_subsets(pool, 1, OFFENSE_REMOVE_DEPTH) {
        let mut constraints = base_constraints.clone();
        constraints.blocked_offense.extend(blocked_ids);
        emitter.run(model, artifacts, PlanProfile::full(), &constraints);
    }
}

fn emit_keep_only_variants(
    emitter: &mut CandidateEmitter,
    model: &WorldModel,
    artifacts: &MissionArtifacts,
    accepted_offense: &[usize],
    base_constraints: &PlanConstraints,
) {
    let pool = &accepted_offense[..accepted_offense.len().min(OFFENSE_KEEP_ONLY_POOL)];
    for allowed_ids in mission_subsets(pool, 0, OFFENSE_KEEP_ONLY_DEPTH) {
        let mut constraints = base_constraints.clone();
        constraints.allowed_offense = Some(allowed_ids);
        emitter.run(model, artifacts, PlanProfile::full(), &constraints);
    }
}

fn mission_subsets(
    pool: &[usize],
    min_size: usize,
    max_size: usize,
) -> Vec<HashSet<usize>> {
    let upper = pool.len().min(max_size);
    let mut out: Vec<HashSet<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::with_capacity(upper);
    for size in min_size..=upper {
        collect_subsets(pool, size, 0, &mut current, &mut out);
    }
    out
}

fn collect_subsets(
    pool: &[usize],
    target_size: usize,
    start: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<HashSet<usize>>,
) {
    if current.len() == target_size {
        out.push(current.iter().copied().collect());
        return;
    }
    let remaining_needed = target_size - current.len();
    if pool.len().saturating_sub(start) < remaining_needed {
        return;
    }
    for idx in start..pool.len() {
        current.push(pool[idx]);
        collect_subsets(pool, target_size, idx + 1, current, out);
        current.pop();
    }
}

/// Accumulates candidate plans while dropping move-list duplicates. Two
/// candidates that produce the same plan skip the rollout entirely.
struct CandidateEmitter {
    candidates: Vec<Vec<(i64, f64, i64)>>,
    seen: HashSet<Vec<(i64, u64, i64)>>,
}

impl CandidateEmitter {
    fn new(first: Vec<(i64, f64, i64)>) -> Self {
        let mut e = Self {
            candidates: Vec::new(),
            seen: HashSet::default(),
        };
        e.push_raw(first);
        e
    }

    fn run(
        &mut self,
        model: &WorldModel,
        artifacts: &MissionArtifacts,
        profile: PlanProfile,
        constraints: &PlanConstraints,
    ) {
        let plan = plan_from_artifacts(model, artifacts, profile, constraints);
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

/// Order-insensitive exact key for move-list dedup. `to_bits` keeps the
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
