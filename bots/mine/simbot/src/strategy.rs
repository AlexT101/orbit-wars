//! Strategy entry points exposed to the PyO3 layer. Concrete strategy logic
//! lives in dedicated modules (e.g. [`crate::obnext`]); functions here are
//! thin orchestrators over the strategy-agnostic [`WorldState`].

#![allow(dead_code)]

use rustc_hash::FxHashSet as HashSet;

use crate::engine::{EngineState};
use crate::entity_cache::EntityCache;
use crate::obnext::{
    build_mission_artifacts, patient_targets, plan_from_artifacts, MissionArtifacts, PlanProfile,
    WorldModel,
};
use crate::rollout::{opponent_turn0_variants, rollout_score};
use crate::world::WorldState;

const EAGER_FORBID_PREFIX_CAP: usize = 8;
const EAGER_ONLY_ONE_CAP: usize = 8;
const EAGER_ONLY_PAIR_POOL: usize = 6;
const EAGER_ONLY_TRIPLE_POOL: usize = 5;
const EAGER_SKIP_CAP: usize = 5;

const PATIENT_FORBID_PREFIX_CAP: usize = 3;
const PATIENT_ONLY_ONE_CAP: usize = 3;
const PATIENT_ONLY_PAIR_POOL: usize = 3;

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

/// Build a wide full-search candidate set for the rollout to score.
///
/// We emit two branches from one shared artifact build:
///   * an eager branch with a wider structural roster
///   * a patience-gated branch with a smaller roster
///
/// Both branches use `PlanProfile::full()`. All candidates still share one
/// `WorldModel` AND one `MissionArtifacts`, so widening the roster mostly pays
/// for extra commit/filter passes and rollout scoring rather than repeating the
/// expensive mission sweep.
pub fn obnext_candidates(world: &WorldState) -> Vec<Vec<(i64, f64, i64)>> {
    if world.my_planets.is_empty() {
        return vec![Vec::new()];
    }
    let model = WorldModel::build(world);
    // Shared across every candidate: policy, full mission list. The
    // expensive O(my × all) sweep + swarm pairing + crash exploit only runs
    // here, not per candidate.
    let artifacts = build_mission_artifacts(&model);

    let wait_set = patient_targets(&model, &artifacts);

    let eager_forbid: HashSet<i64> = HashSet::default();
    let eager_plan = plan_from_artifacts(&model, &artifacts, PlanProfile::full(), &eager_forbid);
    let eager_targets = eager_plan.offensive_targets.clone();

    let patient_plan = plan_from_artifacts(&model, &artifacts, PlanProfile::full(), &wait_set);
    let patient_targets = patient_plan.offensive_targets.clone();

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

    let mut emitter = CandidateEmitter::new(eager_plan.moves);

    emit_candidate_branch(
        &mut emitter,
        &model,
        &artifacts,
        &eager_targets,
        &eager_forbid,
        &opposing,
        &neutrals,
        &hostiles,
        CandidateBranchSpec {
            prefix_cap: EAGER_FORBID_PREFIX_CAP,
            only_one_cap: EAGER_ONLY_ONE_CAP,
            only_pair_pool: EAGER_ONLY_PAIR_POOL,
            only_triple_pool: EAGER_ONLY_TRIPLE_POOL,
            skip_cap: EAGER_SKIP_CAP,
            include_owner_partitions: true,
            include_noop: true,
        },
    );

    emitter.push_raw(patient_plan.moves);
    emit_candidate_branch(
        &mut emitter,
        &model,
        &artifacts,
        &patient_targets,
        &wait_set,
        &opposing,
        &neutrals,
        &hostiles,
        CandidateBranchSpec {
            prefix_cap: PATIENT_FORBID_PREFIX_CAP,
            only_one_cap: PATIENT_ONLY_ONE_CAP,
            only_pair_pool: PATIENT_ONLY_PAIR_POOL,
            only_triple_pool: 0,
            skip_cap: 0,
            include_owner_partitions: false,
            include_noop: false,
        },
    );

    emitter.into_candidates()
}

#[derive(Clone, Copy)]
struct CandidateBranchSpec {
    prefix_cap: usize,
    only_one_cap: usize,
    only_pair_pool: usize,
    only_triple_pool: usize,
    skip_cap: usize,
    include_owner_partitions: bool,
    include_noop: bool,
}

fn emit_candidate_branch(
    emitter: &mut CandidateEmitter,
    model: &WorldModel,
    artifacts: &MissionArtifacts,
    targets: &[i64],
    base_forbid: &HashSet<i64>,
    opposing: &HashSet<i64>,
    neutrals: &HashSet<i64>,
    hostiles: &HashSet<i64>,
    spec: CandidateBranchSpec,
) {
    emit_forbid_prefix(
        emitter,
        model,
        artifacts,
        PlanProfile::full(),
        targets,
        base_forbid,
        spec.prefix_cap,
    );
    emit_only_k_variants(
        emitter,
        model,
        artifacts,
        PlanProfile::full(),
        targets,
        base_forbid,
        opposing,
        1,
        spec.only_one_cap,
    );
    emit_only_k_variants(
        emitter,
        model,
        artifacts,
        PlanProfile::full(),
        targets,
        base_forbid,
        opposing,
        2,
        spec.only_pair_pool,
    );
    emit_only_k_variants(
        emitter,
        model,
        artifacts,
        PlanProfile::full(),
        targets,
        base_forbid,
        opposing,
        3,
        spec.only_triple_pool,
    );
    emit_skip_variants(
        emitter,
        model,
        artifacts,
        PlanProfile::full(),
        targets,
        base_forbid,
        spec.skip_cap,
    );

    if spec.include_owner_partitions {
        emitter.run(
            model,
            artifacts,
            PlanProfile::full(),
            &merged_forbid(base_forbid, neutrals.iter().copied()),
        );
        emitter.run(
            model,
            artifacts,
            PlanProfile::full(),
            &merged_forbid(base_forbid, hostiles.iter().copied()),
        );
        emitter.run(model, artifacts, PlanProfile::full(), opposing);
    }

    if spec.include_noop {
        emitter.push_raw(Vec::new());
    }
}

fn emit_forbid_prefix(
    emitter: &mut CandidateEmitter,
    model: &WorldModel,
    artifacts: &MissionArtifacts,
    profile: PlanProfile,
    targets: &[i64],
    base_forbid: &HashSet<i64>,
    cap: usize,
) {
    for k in 1..=cap.min(targets.len()) {
        emitter.run(
            model,
            artifacts,
            profile,
            &merged_forbid(base_forbid, targets.iter().take(k).copied()),
        );
    }
}

fn emit_only_k_variants(
    emitter: &mut CandidateEmitter,
    model: &WorldModel,
    artifacts: &MissionArtifacts,
    profile: PlanProfile,
    targets: &[i64],
    base_forbid: &HashSet<i64>,
    opposing: &HashSet<i64>,
    keep_count: usize,
    pool: usize,
) {
    if keep_count == 0 || pool < keep_count {
        return;
    }
    let limited_targets = &targets[..targets.len().min(pool)];
    match keep_count {
        1 => {
            for &t in limited_targets {
                emitter.run(
                    model,
                    artifacts,
                    profile,
                    &forbid_all_except(base_forbid, opposing, &[t]),
                );
            }
        }
        2 => {
            for i in 0..limited_targets.len() {
                for j in (i + 1)..limited_targets.len() {
                    emitter.run(
                        model,
                        artifacts,
                        profile,
                        &forbid_all_except(base_forbid, opposing, &[limited_targets[i], limited_targets[j]]),
                    );
                }
            }
        }
        3 => {
            for i in 0..limited_targets.len() {
                for j in (i + 1)..limited_targets.len() {
                    for k in (j + 1)..limited_targets.len() {
                        emitter.run(
                            model,
                            artifacts,
                            profile,
                            &forbid_all_except(
                                base_forbid,
                                opposing,
                                &[limited_targets[i], limited_targets[j], limited_targets[k]],
                            ),
                        );
                    }
                }
            }
        }
        _ => {}
    }
}

fn emit_skip_variants(
    emitter: &mut CandidateEmitter,
    model: &WorldModel,
    artifacts: &MissionArtifacts,
    profile: PlanProfile,
    targets: &[i64],
    base_forbid: &HashSet<i64>,
    skip_cap: usize,
) {
    if targets.len() <= 1 || skip_cap == 0 {
        return;
    }
    let upper = targets.len().min(skip_cap.saturating_add(1));
    for &target_id in &targets[1..upper] {
        emitter.run(
            model,
            artifacts,
            profile,
            &merged_forbid(base_forbid, std::iter::once(target_id)),
        );
    }
}

fn merged_forbid<I>(base_forbid: &HashSet<i64>, extra: I) -> HashSet<i64>
where
    I: IntoIterator<Item = i64>,
{
    let mut forbid = base_forbid.clone();
    forbid.extend(extra);
    forbid
}

fn forbid_all_except(
    base_forbid: &HashSet<i64>,
    opposing: &HashSet<i64>,
    keep: &[i64],
) -> HashSet<i64> {
    let mut forbid = opposing.clone();
    forbid.extend(base_forbid.iter().copied());
    for &id in keep {
        forbid.remove(&id);
    }
    forbid
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
