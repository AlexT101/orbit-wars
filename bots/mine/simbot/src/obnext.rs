//! Port of the open-source `obnext` strategy
//! ([bots/_open_source/obnext/main.py](../../_open_source/obnext/main.py)).
//!
//! Layered on top of:
//!   * Solvers in [`crate::helpers`] — aim, timeline simulation, ownership solver.
//!   * [`crate::world::WorldState`] — strategy-agnostic per-turn snapshot.
//!
//! Layout:
//!   1. Strategy constants
//!   2. Mission / option / commitment types
//!   3. `WorldModel` — obnext-specific wrapper around `&WorldState`
//!      (phase flags, indirect-wealth heuristic, solver memoization caches)
//!   4. Scoring & filter helpers
//!   5. Policy builder
//!   6. `settle_plan` / `settle_reinforce_plan` iterative solvers
//!   7. Mission builders (snipe, rescue, recapture, reinforce, crash_exploit)
//!   8. `plan_moves` — the main loop
//!   9. `plan(...)` — entry point that wraps a `WorldState` and runs the loop
//!
//! Scoring is intentionally thin: mission value is `production × turns_profit`
//! plus a small indirect-wealth term. Mode/phase tilts (is_behind/ahead/
//! finishing, early/late multipliers, mission-kind multipliers) have been
//! removed — the rollout in [`crate::strategy`] runs full forward simulations
//! against opponent variants and picks the winning candidate, so per-mission
//! scoring only needs to set commit *order* under budget pressure.
//! [`preferred_send`]'s margin pile is retained: rollout judges plans as a
//! whole and cannot tune individual send sizes.

#![allow(dead_code)]

use std::cell::RefCell;
use std::ops::Deref;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::constants::HORIZON;
use crate::engine::Planet;
use crate::entity_cache::AimCacheVerdict;
use crate::helpers::{self, aim_with_prediction, dist, nearest_distance_to_set, ArrivalEvent};
use crate::world::{HoldStatus, WorldState};

// ── 1. Strategy constants ─────────────────────────────────────────────────

const OPENING_TURN_LIMIT: i64 = 80;
const LATE_REMAINING_TURNS: i64 = 60;
const VERY_LATE_REMAINING_TURNS: i64 = 25;

const SAFE_NEUTRAL_MARGIN: i64 = 2;
const CONTESTED_NEUTRAL_MARGIN: i64 = 2;

const SAFE_OPENING_PROD_THRESHOLD: i64 = 4;
const SAFE_OPENING_TURN_LIMIT: i64 = 10;
const ROTATING_OPENING_MAX_TURNS: i64 = 13;
const ROTATING_OPENING_LOW_PROD: i64 = 2;
const FOUR_PLAYER_ROTATING_REACTION_GAP: i64 = 1;
const FOUR_PLAYER_ROTATING_SEND_RATIO: f64 = 0.72;
const FOUR_PLAYER_ROTATING_TURN_LIMIT: i64 = 14;

const COMET_MAX_CHASE_TURNS: i64 = 15;

const ATTACK_COST_TURN_WEIGHT: f64 = 0.55;
const SNIPE_COST_TURN_WEIGHT: f64 = 0.45;
const INDIRECT_VALUE_SCALE: f64 = 0.15;
const INDIRECT_FRIENDLY_WEIGHT: f64 = 0.35;
const INDIRECT_NEUTRAL_WEIGHT: f64 = 0.9;
const INDIRECT_ENEMY_WEIGHT: f64 = 1.25;

const NEUTRAL_MARGIN_BASE: i64 = 2;
const NEUTRAL_MARGIN_PROD_WEIGHT: i64 = 2;
const NEUTRAL_MARGIN_CAP: i64 = 8;
const HOSTILE_MARGIN_BASE: i64 = 3;
const HOSTILE_MARGIN_PROD_WEIGHT: i64 = 2;
const HOSTILE_MARGIN_CAP: i64 = 12;
const STATIC_TARGET_MARGIN: i64 = 4;
const CONTESTED_TARGET_MARGIN: i64 = 5;
const FOUR_PLAYER_TARGET_MARGIN: i64 = 2;
const LONG_TRAVEL_MARGIN_START: i64 = 18;
const LONG_TRAVEL_MARGIN_DIVISOR: i64 = 3;
const LONG_TRAVEL_MARGIN_CAP: i64 = 8;
const COMET_MARGIN_RELIEF: i64 = 6;

const FOLLOWUP_MIN_SHIPS: i64 = 8;
const LOW_VALUE_COMET_PRODUCTION: i64 = 1;
const LATE_CAPTURE_BUFFER: i64 = 5;
const VERY_LATE_CAPTURE_BUFFER: i64 = 3;

const DEFENSE_LOOKAHEAD_TURNS: i64 = 28;
const DEFENSE_COST_TURN_WEIGHT: f64 = 0.4;
const DEFENSE_SEND_MARGIN_BASE: i64 = 1;
const DEFENSE_SEND_MARGIN_PROD_WEIGHT: i64 = 1;
const DEFENSE_SHIP_VALUE: f64 = 0.55;

const REINFORCE_ENABLED: bool = true;
const REINFORCE_MIN_PRODUCTION: i64 = 2;
const REINFORCE_MAX_TRAVEL_TURNS: i64 = 22;
const REINFORCE_SAFETY_MARGIN: i64 = 2;
const REINFORCE_MAX_SOURCE_FRACTION: f64 = 0.75;
const REINFORCE_MIN_FUTURE_TURNS: i64 = 40;
const REINFORCE_HOLD_LOOKAHEAD: i64 = 20;
const REINFORCE_COST_TURN_WEIGHT: f64 = 0.35;

const RECAPTURE_LOOKAHEAD_TURNS: i64 = 10;
const RECAPTURE_COST_TURN_WEIGHT: f64 = 0.52;
const RECAPTURE_PRODUCTION_WEIGHT: f64 = 0.6;
const RECAPTURE_IMMEDIATE_WEIGHT: f64 = 0.4;

const REAR_SOURCE_MIN_SHIPS: i64 = 16;
const REAR_DISTANCE_RATIO: f64 = 1.25;
const REAR_STAGE_PROGRESS: f64 = 0.78;
const REAR_SEND_RATIO_TWO_PLAYER: f64 = 0.62;
const REAR_SEND_RATIO_FOUR_PLAYER: f64 = 0.7;
const REAR_SEND_MIN_SHIPS: i64 = 10;
const REAR_MAX_TRAVEL_TURNS: i64 = 40;

const PARTIAL_SOURCE_MIN_SHIPS: i64 = 6;
const MULTI_SOURCE_TOP_K: usize = 10;
const MULTI_SOURCE_ETA_TOLERANCE: i64 = 2;
const MULTI_SOURCE_PLAN_PENALTY: f64 = 0.97;
const HOSTILE_SWARM_ETA_TOLERANCE: i64 = 1;
const THREE_SOURCE_SWARM_ENABLED: bool = true;
const THREE_SOURCE_MIN_TARGET_SHIPS: i64 = 20;
const THREE_SOURCE_ETA_TOLERANCE: i64 = 2;
const THREE_SOURCE_PLAN_PENALTY: f64 = 0.94;

const PROACTIVE_DEFENSE_HORIZON: i64 = 12;
const PROACTIVE_DEFENSE_RATIO: f64 = 0.18;
const MULTI_ENEMY_PROACTIVE_HORIZON: i64 = 14;
const MULTI_ENEMY_PROACTIVE_RATIO: f64 = 0.22;
const MULTI_ENEMY_STACK_WINDOW: i64 = 3;
const REACTION_SOURCE_TOP_K_MY: usize = 4;
const REACTION_SOURCE_TOP_K_ENEMY: usize = 4;
const PROACTIVE_ENEMY_TOP_K: usize = 3;

const CRASH_EXPLOIT_ENABLED: bool = true;
const CRASH_EXPLOIT_MIN_TOTAL_SHIPS: i64 = 10;
const CRASH_EXPLOIT_ETA_WINDOW: i64 = 2;
const CRASH_EXPLOIT_POST_CRASH_DELAY: i64 = 1;

const DOOMED_EVAC_TURN_LIMIT: i64 = 24;
const DOOMED_MIN_SHIPS: i64 = 8;

// ── 2. PlanProfile ────────────────────────────────────────────────────────

/// Toggles optimization features. Defensive features (reinforce/rescue/
/// recapture/doomed evac) and core offense (Single + Snipe) are always on.
/// `heavy = false` is used by the rollout opponent model to keep per-turn
/// cost down.
#[derive(Debug, Clone, Copy)]
pub struct PlanProfile {
    /// Enables 2-/3-source swarms, crash exploit, follow-up pass, rear staging.
    pub heavy: bool,
}

impl PlanProfile {
    pub const fn full() -> Self {
        Self { heavy: true }
    }
    pub const fn fast() -> Self {
        Self { heavy: false }
    }
}

impl Default for PlanProfile {
    fn default() -> Self {
        Self::full()
    }
}

// ── 2. Mission / option / commitment types ────────────────────────────────

/// What a single shot is trying to do. Used both on `ShotOption.mission` (per
/// individual shot) and on `Mission.kind` (per accepted plan); `Single`
/// appears only on the latter — it's the kind name obnext gives to a one-shot
/// capture wrapped as a mission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MissionKind {
    Capture,
    Single,
    Snipe,
    Swarm,
    Reinforce,
    Rescue,
    Recapture,
    CrashExploit,
}

#[derive(Debug, Clone, Copy)]
pub struct ShotOption {
    pub score: f64,
    pub src_id: i64,
    pub target_id: i64,
    pub angle: f64,
    pub turns: i64,
    pub needed: i64,
    pub send_cap: i64,
    pub mission: MissionKind,
    pub anchor_turn: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct Mission {
    pub kind: MissionKind,
    pub score: f64,
    pub target_id: i64,
    pub turns: i64,
    pub options: Vec<ShotOption>,
}

/// Friendly arrivals decided this turn but not yet flown. Combined with
/// `TimelineCache::arrivals` whenever the planner asks a hypothetical
/// "if we add these arrivals, who owns the planet at turn T?" question.
pub type PlannedCommitments = HashMap<i64, Vec<ArrivalEvent>>;

/// `(angle, turns, target_x, target_y)` — return shape of every solver here.
type AimResult = (f64, i64, f64, f64);

/// Inline-storage hint key. Hints in practice are 0–3 elements; capping at 4
/// (and storing on the stack) lets the probe caches use a `Copy` key instead
/// of allocating a `Vec<i64>` per lookup/insert.
///
/// This path intentionally supports up to 4 hints only. Callers that exceed
/// that bound should fail fast rather than silently changing solver behavior.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct HintsKey {
    data: [i64; 4],
    len: u8,
}

impl HintsKey {
    fn from_slice(hints: &[i64]) -> Self {
        assert!(hints.len() <= 4, "HintsKey supports at most 4 hints");
        let mut data = [0i64; 4];
        let len = hints.len();
        data[..len].copy_from_slice(&hints[..len]);
        Self { data, len: len as u8 }
    }

    #[inline]
    fn as_slice(&self) -> &[i64] {
        &self.data[..self.len as usize]
    }
}

/// Cache key for `WorldModel::best_probe_aim` (mirrors obnext's tuple key).
type BestProbeKey = (
    i64,            // src_id
    i64,            // target_id
    i64,            // source_cap
    HintsKey,       // hints (inline, up to 4)
    Option<i64>,    // min_turn
    Option<i64>,    // max_turn
    Option<i64>,    // anchor_turn
    Option<i64>,    // max_anchor_diff
);

// ── 3. WorldModel ─────────────────────────────────────────────────────────

/// Obnext-specific wrapper over a [`WorldState`]. Adds phase flags, the
/// indirect-wealth heuristic, and the solver memoization caches; everything
/// else lives in `state` and is reached via `Deref` (so `world.my_planets`
/// still works as it did before).
pub struct WorldModel<'a> {
    pub state: &'a WorldState<'a>,
    pub is_opening: bool,
    pub is_late: bool,
    pub is_very_late: bool,
    /// Per-planet `(friendly, neutral, enemy)` production-weighted nearness.
    /// Drives `target_value`'s indirect-wealth term.
    pub indirect_feature_map: HashMap<i64, (f64, f64, f64)>,

    shot_cache: RefCell<HashMap<(i64, i64, i64), Option<AimResult>>>,
    probe_candidate_cache: RefCell<HashMap<(i64, i64, i64, HintsKey), Vec<i64>>>,
    best_probe_cache: RefCell<HashMap<BestProbeKey, Option<(i64, AimResult)>>>,
    reaction_cache: RefCell<HashMap<i64, (i64, i64)>>,
    exact_need_cache: RefCell<HashMap<(i64, i64, i64), i64>>,
}

impl<'a> Deref for WorldModel<'a> {
    type Target = WorldState<'a>;
    fn deref(&self) -> &Self::Target {
        self.state
    }
}

impl<'a> WorldModel<'a> {
    pub fn build(state: &'a WorldState<'a>) -> Self {
        let is_opening = state.step < OPENING_TURN_LIMIT;
        let is_late = state.remaining_steps < LATE_REMAINING_TURNS;
        let is_very_late = state.remaining_steps < VERY_LATE_REMAINING_TURNS;

        let mut indirect_feature_map =
            HashMap::with_capacity_and_hasher(state.planets.len(), Default::default());
        for planet in &state.planets {
            indirect_feature_map
                .insert(planet.id, indirect_features(planet, &state.planets, state.player));
        }

        Self {
            state,
            is_opening,
            is_late,
            is_very_late,
            indirect_feature_map,
            shot_cache: RefCell::new(HashMap::default()),
            probe_candidate_cache: RefCell::new(HashMap::default()),
            best_probe_cache: RefCell::new(HashMap::default()),
            reaction_cache: RefCell::new(HashMap::default()),
            exact_need_cache: RefCell::new(HashMap::default()),
        }
    }

    /// Cached `aim_with_prediction`. Two-level cache:
    ///   * L1 — per-`WorldModel` `shot_cache`: avoids repeated traffic to the
    ///     L2 `RefCell` inside the hot probe loops.
    ///   * L2 — `EntityCache::aim_cache`: shares results across every
    ///     `WorldModel` built during the same bot turn (candidate plans,
    ///     opponent-rollout rebuilds) and across turns when geometry is still
    ///     valid. Stale entries are re-verified or evicted lazily.
    pub fn plan_shot(&self, src_id: i64, target_id: i64, ships: i64) -> Option<AimResult> {
        let ships = ships.max(1);
        let key = (src_id, target_id, ships);
        if let Some(cached) = self.shot_cache.borrow().get(&key) {
            return *cached;
        }

        let result = match self.entity_cache.aim_cache_lookup(src_id, target_id, ships) {
            AimCacheVerdict::Hit(r) => r,
            AimCacheVerdict::Miss | AimCacheVerdict::Stale => {
                let r = aim_with_prediction(self.entity_cache, src_id, target_id, ships);
                self.entity_cache.aim_cache_store(src_id, target_id, ships, r);
                r
            }
        };

        self.shot_cache.borrow_mut().insert(key, result);
        result
    }

    /// Generate a set of candidate send sizes for binary-search-style ship
    /// tuning. Mirrors obnext's `probe_ship_candidates` so the same hints land
    /// the same values.
    pub fn probe_ship_candidates(
        &self,
        src_id: i64,
        target_id: i64,
        source_cap: i64,
        hints: &[i64],
    ) -> Vec<i64> {
        let source_cap = source_cap.max(1);
        assert!(hints.len() <= 4, "probe_ship_candidates supports at most 4 hints");
        // Normalize in place on a stack-friendly buffer (hints are 0–4 long
        // in practice, matching the inline cache key contract).
        let mut normalized_buf = [0i64; 4];
        let mut normalized_len = 0usize;
        for &h in hints {
            if h > 0 && normalized_len < normalized_buf.len() {
                normalized_buf[normalized_len] = h;
                normalized_len += 1;
            }
        }
        normalized_buf[..normalized_len].sort_unstable();
        // dedup in place
        let mut write = 0usize;
        let mut prev: Option<i64> = None;
        for i in 0..normalized_len {
            let v = normalized_buf[i];
            if Some(v) != prev {
                normalized_buf[write] = v;
                write += 1;
                prev = Some(v);
            }
        }
        normalized_len = write;
        let normalized_hints = &normalized_buf[..normalized_len];

        let cache_key = (
            src_id,
            target_id,
            source_cap,
            HintsKey::from_slice(normalized_hints),
        );
        if let Some(cached) = self.probe_candidate_cache.borrow().get(&cache_key) {
            return cached.clone();
        }

        let target = self.planet(target_id);
        let target_ships = target.ships.max(1);

        // The set has at most ~12 entries (8 fixed slots + up to 5 deltas per
        // hint, all clamped to `[1, source_cap]`). A Vec + sort + dedup beats
        // a HashSet at this size, avoiding allocation + hashing overhead.
        let mut values: Vec<i64> = Vec::with_capacity(16);
        let push = |v: i64, values: &mut Vec<i64>| {
            if (1..=source_cap).contains(&v) {
                values.push(v);
            }
        };
        for v in 1..=source_cap.min(6) {
            push(v, &mut values);
        }
        push(source_cap, &mut values);
        push((source_cap / 2).max(1), &mut values);
        push((source_cap / 3).max(1), &mut values);
        push(source_cap.min(PARTIAL_SOURCE_MIN_SHIPS), &mut values);
        push(source_cap.min(target_ships + 1), &mut values);
        push(source_cap.min(target_ships + 2), &mut values);
        push(source_cap.min(target_ships + 4), &mut values);
        push(source_cap.min(target_ships + 8), &mut values);

        for &hint in normalized_hints {
            let base = hint.clamp(1, source_cap);
            for delta in [-2, -1, 0, 1, 2] {
                let candidate = base + delta;
                push(candidate, &mut values);
            }
        }

        values.sort_unstable();
        values.dedup();
        let result = values;
        self.probe_candidate_cache
            .borrow_mut()
            .insert(cache_key, result.clone());
        result
    }

    /// Find the smallest-/closest-to-anchor send that produces a valid aim
    /// within the given timing window. Cached on the full filter tuple.
    #[allow(clippy::too_many_arguments)]
    pub fn best_probe_aim(
        &self,
        src_id: i64,
        target_id: i64,
        source_cap: i64,
        hints: &[i64],
        min_turn: Option<i64>,
        max_turn: Option<i64>,
        anchor_turn: Option<i64>,
        max_anchor_diff: Option<i64>,
    ) -> Option<(i64, AimResult)> {
        let source_cap = source_cap.max(1);
        let hints_key = HintsKey::from_slice(hints);
        let cache_key: BestProbeKey = (
            src_id,
            target_id,
            source_cap,
            hints_key,
            min_turn,
            max_turn,
            anchor_turn,
            max_anchor_diff,
        );
        if let Some(cached) = self.best_probe_cache.borrow().get(&cache_key) {
            return *cached;
        }

        let mut best: Option<(i64, AimResult)> = None;
        // best_key uses turn count + (optionally) anchor distance, identical
        // ordering to obnext.
        let mut best_key: Option<(i64, i64, i64)> = None;

        for ships in self.probe_ship_candidates(src_id, target_id, source_cap, hints_key.as_slice()) {
            let Some(aim) = self.plan_shot(src_id, target_id, ships) else {
                continue;
            };
            let (_, turns, _, _) = aim;
            if let Some(mn) = min_turn {
                if turns < mn {
                    continue;
                }
            }
            if let Some(mx) = max_turn {
                if turns > mx {
                    continue;
                }
            }
            if let (Some(anchor), Some(diff)) = (anchor_turn, max_anchor_diff) {
                if (turns - anchor).abs() > diff {
                    continue;
                }
            }

            let key = match anchor_turn {
                None => (0, turns, ships),
                Some(anchor) => ((turns - anchor).abs(), turns, ships),
            };
            if best_key.map_or(true, |bk| key < bk) {
                best_key = Some(key);
                best = Some((ships, aim));
            }
        }

        self.best_probe_cache.borrow_mut().insert(cache_key, best);
        best
    }

    /// `(my_first_arrival_turn, enemy_first_arrival_turn)` against `target_id`.
    pub fn reaction_times(&self, target_id: i64) -> (i64, i64) {
        if let Some(&v) = self.reaction_cache.borrow().get(&target_id) {
            return v;
        }
        let mut my_t: i64 = i64::MAX / 4;
        for planet in &self.my_planets {
            if let Some((_, aim)) =
                self.best_probe_aim(planet.id, target_id, planet.ships.max(1), &[], None, None, None, None)
            {
                my_t = my_t.min(aim.1);
            }
        }
        let mut enemy_t: i64 = i64::MAX / 4;
        for planet in &self.enemy_planets {
            if let Some((_, aim)) =
                self.best_probe_aim(planet.id, target_id, planet.ships.max(1), &[], None, None, None, None)
            {
                enemy_t = enemy_t.min(aim.1);
            }
        }
        let v = (my_t, enemy_t);
        self.reaction_cache.borrow_mut().insert(target_id, v);
        v
    }

    /// Strategy-side projection wrapper: extracts the planned-commitments
    /// slice for `target_id` and defers to [`WorldState::projected_state`].
    pub fn projected_state(
        &self,
        target_id: i64,
        arrival_turn: i64,
        planned: &PlannedCommitments,
        extra: &[ArrivalEvent],
    ) -> (i64, i64) {
        let planned_for = planned_for(planned, target_id);
        self.state
            .projected_state(target_id, arrival_turn, planned_for, extra)
    }

    pub fn hold_status(
        &self,
        target_id: i64,
        planned: &PlannedCommitments,
        horizon: i64,
    ) -> HoldStatus {
        let planned_for = planned_for(planned, target_id);
        self.state.hold_status(target_id, planned_for, horizon)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn min_ships_to_own_by(
        &self,
        target_id: i64,
        eval_turn: i64,
        attacker_owner: i64,
        arrival_turn: Option<i64>,
        planned: &PlannedCommitments,
        extra: &[ArrivalEvent],
        upper_bound: Option<i64>,
    ) -> i64 {
        let eval_turn = eval_turn.max(1);
        let arrival_turn = arrival_turn.unwrap_or(eval_turn).max(1);
        let ub = upper_bound.unwrap_or_else(|| self.ownership_search_cap(eval_turn));
        if arrival_turn > eval_turn {
            return ub + 1;
        }

        let extras = merged_extras(planned_for(planned, target_id), extra, eval_turn);
        let cache_key = if arrival_turn == eval_turn && extras.is_empty() {
            Some((target_id, eval_turn, attacker_owner))
        } else {
            None
        };
        if let Some(key) = cache_key.as_ref() {
            if let Some(&v) = self.exact_need_cache.borrow().get(key) {
                return v;
            }
        }

        let result = helpers::min_ships_to_own_by(
            &self.timeline_cache,
            self.planet(target_id),
            attacker_owner,
            arrival_turn,
            eval_turn,
            ub,
            &extras,
        );

        if let Some(key) = cache_key {
            self.exact_need_cache.borrow_mut().insert(key, result);
        }
        result
    }

    pub fn min_ships_to_own_at(
        &self,
        target_id: i64,
        arrival_turn: i64,
        attacker_owner: i64,
        planned: &PlannedCommitments,
        extra: &[ArrivalEvent],
        upper_bound: Option<i64>,
    ) -> i64 {
        self.min_ships_to_own_by(
            target_id,
            arrival_turn,
            attacker_owner,
            Some(arrival_turn),
            planned,
            extra,
            upper_bound,
        )
    }

    pub fn reinforcement_needed_to_hold_until(
        &self,
        planet_id: i64,
        arrival_turn: i64,
        hold_until: i64,
        planned: &PlannedCommitments,
        upper_bound: Option<i64>,
    ) -> i64 {
        let arrival_turn = arrival_turn.max(1);
        let hold_until = hold_until.max(arrival_turn);
        let ub = upper_bound.unwrap_or_else(|| self.ownership_search_cap(hold_until));
        let extras = merged_extras(planned_for(planned, planet_id), &[], hold_until);
        helpers::reinforcement_needed_to_hold_until(
            &self.timeline_cache,
            self.planet(planet_id),
            arrival_turn,
            hold_until,
            ub,
            &extras,
        )
    }

    pub fn ships_needed_to_capture(
        &self,
        target_id: i64,
        arrival_turn: i64,
        planned: &PlannedCommitments,
        extra: &[ArrivalEvent],
    ) -> i64 {
        self.min_ships_to_own_at(target_id, arrival_turn, self.player, planned, extra, None)
    }
}

/// Pull the planned-commitments slice for `target_id` (empty if none planned).
#[inline]
fn planned_for(planned: &PlannedCommitments, target_id: i64) -> &[ArrivalEvent] {
    planned.get(&target_id).map(|v| v.as_slice()).unwrap_or(&[])
}

/// Normalize and merge `planned + extra` arrivals into a single slice the
/// helpers can append to the cache's base arrivals. Filters out non-positive
/// ship counts, clamps turns to `>= 1`, and drops anything past `cutoff`.
fn merged_extras(
    planned: &[ArrivalEvent],
    extra: &[ArrivalEvent],
    cutoff: i64,
) -> Vec<ArrivalEvent> {
    let mut out = Vec::with_capacity(planned.len() + extra.len());
    for ev in planned.iter().chain(extra.iter()) {
        if ev.ships <= 0 {
            continue;
        }
        let turns = ev.turns.max(1);
        if turns > cutoff {
            continue;
        }
        out.push(ArrivalEvent {
            turns,
            owner: ev.owner,
            ships: ev.ships,
        });
    }
    out
}

// ── 4. Scoring & filter helpers ───────────────────────────────────────────

fn planet_distance(a: &Planet, b: &Planet) -> f64 {
    dist(a.x, a.y, b.x, b.y)
}

fn nearest_sources_to_target(target: &Planet, sources: &[Planet], top_k: usize) -> Vec<Planet> {
    if top_k == 0 || sources.len() <= top_k {
        return sources.to_vec();
    }
    let mut ordered: Vec<Planet> = sources.to_vec();
    ordered.sort_by(|a, b| {
        planet_distance(a, target)
            .partial_cmp(&planet_distance(b, target))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.ships.cmp(&a.ships))
            .then(a.id.cmp(&b.id))
    });
    ordered.truncate(top_k);
    ordered
}

fn min_legal_reaction_time(target_id: i64, sources: &[Planet], world: &WorldModel) -> i64 {
    let mut best = i64::MAX / 4;
    for src in sources {
        if let Some((_, aim)) =
            world.best_probe_aim(src.id, target_id, src.ships.max(1), &[], None, None, None, None)
        {
            best = best.min(aim.1);
        }
    }
    best
}

fn indirect_features(planet: &Planet, planets: &[Planet], player: i64) -> (f64, f64, f64) {
    let mut friendly = 0.0;
    let mut neutral = 0.0;
    let mut enemy = 0.0;
    for other in planets {
        if other.id == planet.id {
            continue;
        }
        let d = dist(planet.x, planet.y, other.x, other.y);
        if d < 1.0 {
            continue;
        }
        let factor = other.production as f64 / (d + 12.0);
        if other.owner == player {
            friendly += factor;
        } else if other.owner == -1 {
            neutral += factor;
        } else {
            enemy += factor;
        }
    }
    (friendly, neutral, enemy)
}

fn policy_reaction_times(target_id: i64, policy: &PolicyState) -> (i64, i64) {
    policy
        .reaction_time_map
        .get(&target_id)
        .copied()
        .unwrap_or((i64::MAX / 4, i64::MAX / 4))
}

fn is_safe_neutral(target: &Planet, policy: &PolicyState) -> bool {
    if target.owner != -1 {
        return false;
    }
    let (my_t, enemy_t) = policy_reaction_times(target.id, policy);
    my_t <= enemy_t - SAFE_NEUTRAL_MARGIN
}

fn is_contested_neutral(target: &Planet, policy: &PolicyState) -> bool {
    if target.owner != -1 {
        return false;
    }
    let (my_t, enemy_t) = policy_reaction_times(target.id, policy);
    (my_t - enemy_t).abs() <= CONTESTED_NEUTRAL_MARGIN
}

fn candidate_time_valid(target: &Planet, turns: i64, world: &WorldModel, remaining_buffer: i64) -> bool {
    if turns > world.remaining_steps - remaining_buffer {
        return false;
    }
    if world.comet_ids.contains(&target.id) {
        let life = world.comet_life(target.id);
        if turns >= life || turns > COMET_MAX_CHASE_TURNS {
            return false;
        }
    }
    true
}

fn stacked_enemy_proactive_keep(planet: &Planet, world: &WorldModel) -> i64 {
    let mut threats: Vec<(i64, i64)> = Vec::new();
    for enemy in &world.enemy_planets {
        let Some((_, aim)) = world.best_probe_aim(
            enemy.id,
            planet.id,
            enemy.ships.max(1),
            &[],
            None,
            None,
            None,
            None,
        ) else {
            continue;
        };
        let eta = aim.1;
        if eta > MULTI_ENEMY_PROACTIVE_HORIZON {
            continue;
        }
        threats.push((eta, enemy.ships));
    }
    if threats.is_empty() {
        return 0;
    }
    threats.sort_by_key(|t| t.0);
    let mut best_stacked: i64 = 0;
    let mut left = 0usize;
    let mut running: i64 = 0;
    for right in 0..threats.len() {
        running += threats[right].1;
        while threats[right].0 - threats[left].0 > MULTI_ENEMY_STACK_WINDOW {
            running -= threats[left].1;
            left += 1;
        }
        if running > best_stacked {
            best_stacked = running;
        }
    }
    (best_stacked as f64 * MULTI_ENEMY_PROACTIVE_RATIO) as i64
}

fn swarm_eta_tolerance(options: &[ShotOption], target: &Planet, world: &WorldModel) -> i64 {
    if options.len() >= 3 {
        return THREE_SOURCE_ETA_TOLERANCE;
    }
    if target.owner != -1 && target.owner != world.player {
        return HOSTILE_SWARM_ETA_TOLERANCE;
    }
    MULTI_SOURCE_ETA_TOLERANCE
}

#[derive(Debug, Clone)]
struct EnemyCrash {
    target_id: i64,
    crash_turn: i64,
}

fn detect_enemy_crashes(world: &WorldModel) -> Vec<EnemyCrash> {
    let mut crashes = Vec::new();
    for target in &world.planets {
        let arrivals = world.timeline_cache.arrivals(target.id);
        let mut enemy_events: Vec<ArrivalEvent> = arrivals
            .iter()
            .filter(|ev| ev.owner != -1 && ev.owner != world.player && ev.ships > 0)
            .copied()
            .collect();
        enemy_events.sort_by_key(|ev| ev.turns);
        for i in 0..enemy_events.len() {
            for j in (i + 1)..enemy_events.len() {
                let a = enemy_events[i];
                let b = enemy_events[j];
                if a.owner == b.owner {
                    continue;
                }
                if (a.turns - b.turns).abs() > CRASH_EXPLOIT_ETA_WINDOW {
                    break;
                }
                if a.ships + b.ships < CRASH_EXPLOIT_MIN_TOTAL_SHIPS {
                    continue;
                }
                crashes.push(EnemyCrash {
                    target_id: target.id,
                    crash_turn: a.turns.max(b.turns),
                });
            }
        }
    }
    crashes
}

fn opening_filter(
    target: &Planet,
    arrival_turns: i64,
    needed: i64,
    src_available: i64,
    world: &WorldModel,
    policy: &PolicyState,
) -> bool {
    if !world.is_opening || target.owner != -1 {
        return false;
    }
    if world.comet_ids.contains(&target.id) {
        return false;
    }
    if world.is_static(target.id) {
        return false;
    }

    let (my_t, enemy_t) = policy_reaction_times(target.id, policy);
    let reaction_gap = enemy_t - my_t;
    if target.production >= SAFE_OPENING_PROD_THRESHOLD
        && arrival_turns <= SAFE_OPENING_TURN_LIMIT
        && reaction_gap >= SAFE_NEUTRAL_MARGIN
    {
        return false;
    }

    if world.is_four_player {
        let affordable_cap =
            PARTIAL_SOURCE_MIN_SHIPS.max((src_available as f64 * FOUR_PLAYER_ROTATING_SEND_RATIO) as i64);
        let affordable = needed <= affordable_cap;
        if affordable
            && arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT
            && reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP
        {
            return false;
        }
        return true;
    }

    arrival_turns > ROTATING_OPENING_MAX_TURNS || target.production <= ROTATING_OPENING_LOW_PROD
}

/// Flat mission value: production over the remaining horizon, plus a small
/// indirect-wealth contribution. No mode/phase tilts, no mission-kind
/// multipliers — rollout is the final judge of plan quality; this only sets
/// commit order under budget pressure.
fn target_value(
    target: &Planet,
    arrival_turns: i64,
    world: &WorldModel,
    policy: &PolicyState,
) -> f64 {
    let mut turns_profit = (world.remaining_steps - arrival_turns).max(1);
    if world.comet_ids.contains(&target.id) {
        let life = world.comet_life(target.id);
        turns_profit = turns_profit.min((life - arrival_turns).max(0)).max(0);
        if turns_profit <= 0 {
            return -1.0;
        }
    }

    let mut value = (target.production * turns_profit) as f64;
    let indirect = policy
        .indirect_wealth_map
        .get(&target.id)
        .copied()
        .unwrap_or(0.0);
    value += indirect * (turns_profit as f64) * INDIRECT_VALUE_SCALE;
    value
}

fn reinforce_value(target: &Planet, hold_until: i64, world: &WorldModel, policy: &PolicyState) -> f64 {
    let saved_turns = (world.remaining_steps - hold_until).max(1);
    let mut value = (target.production * saved_turns) as f64 + target.ships.max(0) as f64 * DEFENSE_SHIP_VALUE;
    let indirect = policy.indirect_wealth_map.get(&target.id).copied().unwrap_or(0.0);
    value += indirect * saved_turns as f64 * INDIRECT_VALUE_SCALE * 0.35;
    value
}

/// Send-sizing: takes the exact `min_ships_to_own_by` count and adds a
/// geometric margin pile (base + production-weighted + structural bumps for
/// static / contested / four-player / long-travel; relief for comets). Rollout
/// judges whole plans as a unit and cannot tune individual send sizes, so
/// these margins are the *only* mechanism for robustness against
/// opponent reactions and timing slop.
fn preferred_send(
    target: &Planet,
    base_needed: i64,
    arrival_turns: i64,
    src_available: i64,
    world: &WorldModel,
    policy: &PolicyState,
) -> i64 {
    let mut margin: i64 = 0;
    if target.owner == -1 {
        margin += NEUTRAL_MARGIN_CAP.min(NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD_WEIGHT);
    } else {
        margin += HOSTILE_MARGIN_CAP.min(HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT);
    }
    if world.is_static(target.id) {
        margin += STATIC_TARGET_MARGIN;
    }
    if is_contested_neutral(target, policy) {
        margin += CONTESTED_TARGET_MARGIN;
    }
    if world.is_four_player {
        margin += FOUR_PLAYER_TARGET_MARGIN;
    }
    if arrival_turns > LONG_TRAVEL_MARGIN_START {
        margin += LONG_TRAVEL_MARGIN_CAP.min(arrival_turns / LONG_TRAVEL_MARGIN_DIVISOR);
    }
    if world.comet_ids.contains(&target.id) {
        margin = (margin - COMET_MARGIN_RELIEF).max(0);
    }
    base_needed.saturating_add(margin).min(src_available)
}

// ── 5. Policy ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PolicyState {
    pub indirect_wealth_map: HashMap<i64, f64>,
    pub reserve: HashMap<i64, i64>,
    pub attack_budget: HashMap<i64, i64>,
    pub reaction_time_map: HashMap<i64, (i64, i64)>,
}

fn build_policy_state(world: &WorldModel) -> PolicyState {
    let mut indirect_wealth_map: HashMap<i64, f64> = HashMap::default();
    for (&id, &(friendly, neutral, enemy)) in &world.indirect_feature_map {
        indirect_wealth_map.insert(
            id,
            friendly * INDIRECT_FRIENDLY_WEIGHT
                + neutral * INDIRECT_NEUTRAL_WEIGHT
                + enemy * INDIRECT_ENEMY_WEIGHT,
        );
    }

    let mut reaction_time_map: HashMap<i64, (i64, i64)> = HashMap::default();
    for target in &world.planets {
        if target.owner == world.player {
            continue;
        }
        let my_sources = nearest_sources_to_target(target, &world.my_planets, REACTION_SOURCE_TOP_K_MY);
        let enemy_sources =
            nearest_sources_to_target(target, &world.enemy_planets, REACTION_SOURCE_TOP_K_ENEMY);
        let my_t = min_legal_reaction_time(target.id, &my_sources, world);
        let enemy_t = min_legal_reaction_time(target.id, &enemy_sources, world);
        reaction_time_map.insert(target.id, (my_t, enemy_t));
    }

    let mut reserve: HashMap<i64, i64> = HashMap::default();
    let mut attack_budget: HashMap<i64, i64> = HashMap::default();
    for planet in &world.my_planets {
        let exact_keep = world.keep_needed_map.get(&planet.id).copied().unwrap_or(0);

        let mut proactive_keep: i64 = 0;
        for enemy in nearest_sources_to_target(planet, &world.enemy_planets, PROACTIVE_ENEMY_TOP_K) {
            let Some(aim) = world.plan_shot(enemy.id, planet.id, enemy.ships.max(1)) else {
                continue;
            };
            let enemy_eta = aim.1;
            if enemy_eta > PROACTIVE_DEFENSE_HORIZON {
                continue;
            }
            proactive_keep = proactive_keep.max((enemy.ships as f64 * PROACTIVE_DEFENSE_RATIO) as i64);
        }
        proactive_keep = proactive_keep.max(stacked_enemy_proactive_keep(planet, world));

        let r = planet.ships.min(exact_keep.max(proactive_keep));
        reserve.insert(planet.id, r);
        attack_budget.insert(planet.id, (planet.ships - r).max(0));
    }

    PolicyState {
        indirect_wealth_map,
        reserve,
        attack_budget,
        reaction_time_map,
    }
}

// ── 6. Settlement solvers ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct SettleResult {
    angle: f64,
    turns: i64,
    eval_turn: i64,
    need: i64,
    send: i64,
}

/// Mirrors obnext's `settle_plan`. `eval_turn_fn` lets snipe/rescue/crash
/// missions evaluate ownership at a later anchor turn than the actual
/// arrival.
#[allow(clippy::too_many_arguments)]
fn settle_plan(
    world: &WorldModel,
    src: &Planet,
    target: &Planet,
    src_cap: i64,
    send_guess: i64,
    planned: &PlannedCommitments,
    policy: &PolicyState,
    mission: MissionKind,
    eval_turn_fn: impl Fn(i64) -> i64,
    anchor_turn: Option<i64>,
    anchor_tolerance: Option<i64>,
    max_iter: usize,
) -> Option<SettleResult> {
    if src_cap < 1 {
        return None;
    }
    let seed_hint = send_guess.clamp(1, src_cap);
    let anchor_tolerance = anchor_tolerance.or_else(|| {
        if matches!(mission, MissionKind::Snipe) {
            Some(1)
        } else {
            None
        }
    });

    // tested.insert(send → Option<EvalRow>). None means "this send doesn't
    // settle for some reason"; Some means "this is a candidate".
    let mut tested: HashMap<i64, Option<EvalRow>> = HashMap::default();
    let mut tested_order: Vec<i64> = Vec::new();

    let evaluate = |send: i64,
                    tested: &mut HashMap<i64, Option<EvalRow>>,
                    tested_order: &mut Vec<i64>|
     -> Option<EvalRow> {
        let send = send.clamp(1, src_cap);
        if let Some(cached) = tested.get(&send) {
            return *cached;
        }

        let Some(aim) = world.plan_shot(src.id, target.id, send) else {
            tested.insert(send, None);
            return None;
        };
        let (angle, turns, _, _) = aim;
        if matches!(mission, MissionKind::CrashExploit) {
            if let Some(anchor) = anchor_turn {
                if turns < anchor {
                    tested.insert(send, None);
                    return None;
                }
            }
        }
        let raw_eval_turn = eval_turn_fn(turns);
        if raw_eval_turn < turns {
            tested.insert(send, None);
            return None;
        }
        let eval_turn = raw_eval_turn;
        let need = world.min_ships_to_own_by(
            target.id,
            eval_turn,
            world.player,
            Some(turns),
            planned,
            &[],
            Some(src_cap),
        );
        if need <= 0 || need > src_cap {
            tested.insert(send, None);
            return None;
        }

        let desired = match mission {
            MissionKind::Snipe | MissionKind::CrashExploit => need,
            MissionKind::Rescue => {
                let safety = need + DEFENSE_SEND_MARGIN_BASE + target.production * DEFENSE_SEND_MARGIN_PROD_WEIGHT;
                need.max(safety).min(src_cap)
            }
            _ => {
                let p = preferred_send(target, need, turns, src_cap, world, policy);
                need.max(p).min(src_cap)
            }
        };

        let row = EvalRow {
            angle,
            turns,
            eval_turn,
            need,
            send,
            desired,
        };
        tested.insert(send, Some(row));
        tested_order.push(send);
        Some(row)
    };

    let initial_candidates = {
        let mut cands = world.probe_ship_candidates(src.id, target.id, src_cap, &[seed_hint]);
        cands.sort_by(|a, b| {
            (a - seed_hint)
                .abs()
                .cmp(&(b - seed_hint).abs())
                .then(a.cmp(b))
        });
        cands
    };

    let mut current_send: Option<i64> = None;
    for seed in &initial_candidates {
        let Some(row) = evaluate(*seed, &mut tested, &mut tested_order) else {
            continue;
        };
        if let (Some(anchor), Some(tol)) = (anchor_turn, anchor_tolerance) {
            if (row.turns - anchor).abs() > tol {
                continue;
            }
        }
        current_send = Some(*seed);
        break;
    }
    let mut current_send = current_send?;

    for _ in 0..max_iter {
        let row = match evaluate(current_send, &mut tested, &mut tested_order) {
            Some(r) => r,
            None => break,
        };
        if row.desired == row.send {
            if let (Some(anchor), Some(tol)) = (anchor_turn, anchor_tolerance) {
                if (row.turns - anchor).abs() > tol {
                    return None;
                }
            }
            if matches!(mission, MissionKind::Rescue) && row.turns > row.eval_turn {
                return None;
            }
            return Some(SettleResult {
                angle: row.angle,
                turns: row.turns,
                eval_turn: row.eval_turn,
                need: row.need,
                send: row.send,
            });
        }

        let next_send = row.desired.clamp(1, src_cap);
        if tested.contains_key(&next_send) {
            break;
        }
        current_send = next_send;
    }

    let mut candidate_sends: Vec<i64> = tested_order
        .iter()
        .filter(|s| tested.get(s).and_then(|v| *v).is_some())
        .copied()
        .collect();
    let anchor_for_sort = if matches!(mission, MissionKind::Snipe) {
        anchor_turn
    } else {
        None
    };
    candidate_sends.sort_by(|a, b| {
        let ra = tested[a].unwrap();
        let rb = tested[b].unwrap();
        let ka_anchor = anchor_for_sort.map(|an| (ra.turns - an).abs()).unwrap_or(0);
        let kb_anchor = anchor_for_sort.map(|an| (rb.turns - an).abs()).unwrap_or(0);
        ka_anchor
            .cmp(&kb_anchor)
            .then((a - seed_hint).abs().cmp(&(b - seed_hint).abs()))
            .then(ra.turns.cmp(&rb.turns))
            .then(a.cmp(b))
    });

    let mut seen = HashSet::default();
    for send in candidate_sends {
        if !seen.insert(send) {
            continue;
        }
        let row = tested[&send].unwrap();
        if row.send < row.need {
            continue;
        }
        if let (Some(anchor), Some(tol)) = (anchor_turn, anchor_tolerance) {
            if (row.turns - anchor).abs() > tol {
                continue;
            }
        }
        if matches!(mission, MissionKind::Rescue) && row.turns > row.eval_turn {
            continue;
        }
        return Some(SettleResult {
            angle: row.angle,
            turns: row.turns,
            eval_turn: row.eval_turn,
            need: row.need,
            send: row.send,
        });
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct EvalRow {
    angle: f64,
    turns: i64,
    eval_turn: i64,
    need: i64,
    send: i64,
    desired: i64,
}

fn settle_reinforce_plan(
    world: &WorldModel,
    src: &Planet,
    target: &Planet,
    src_cap: i64,
    send_guess: i64,
    planned: &PlannedCommitments,
    hold_until: i64,
    max_arrival_turn: i64,
    max_iter: usize,
) -> Option<SettleResult> {
    if src_cap < 1 {
        return None;
    }
    let seed_hint = send_guess.clamp(1, src_cap);

    let mut tested: HashMap<i64, Option<EvalRow>> = HashMap::default();
    let mut tested_order: Vec<i64> = Vec::new();

    let evaluate = |send: i64,
                    tested: &mut HashMap<i64, Option<EvalRow>>,
                    tested_order: &mut Vec<i64>|
     -> Option<EvalRow> {
        let send = send.clamp(1, src_cap);
        if let Some(cached) = tested.get(&send) {
            return *cached;
        }
        let Some(aim) = world.plan_shot(src.id, target.id, send) else {
            tested.insert(send, None);
            return None;
        };
        let (angle, turns, _, _) = aim;
        if turns > max_arrival_turn {
            tested.insert(send, None);
            return None;
        }
        let need = world.reinforcement_needed_to_hold_until(
            target.id,
            turns,
            hold_until,
            planned,
            Some(src_cap),
        );
        if need <= 0 || need > src_cap {
            tested.insert(send, None);
            return None;
        }
        let desired = src_cap.min(need + REINFORCE_SAFETY_MARGIN);
        let row = EvalRow {
            angle,
            turns,
            eval_turn: hold_until,
            need,
            send,
            desired,
        };
        tested.insert(send, Some(row));
        tested_order.push(send);
        Some(row)
    };

    let initial_candidates = {
        let mut cands = world.probe_ship_candidates(src.id, target.id, src_cap, &[seed_hint]);
        cands.sort_by(|a, b| {
            (a - seed_hint)
                .abs()
                .cmp(&(b - seed_hint).abs())
                .then(a.cmp(b))
        });
        cands
    };

    let mut current_send: Option<i64> = None;
    for seed in &initial_candidates {
        if evaluate(*seed, &mut tested, &mut tested_order).is_some() {
            current_send = Some(*seed);
            break;
        }
    }
    let mut current_send = current_send?;

    for _ in 0..max_iter {
        let row = match evaluate(current_send, &mut tested, &mut tested_order) {
            Some(r) => r,
            None => break,
        };
        if row.desired == row.send {
            return Some(SettleResult {
                angle: row.angle,
                turns: row.turns,
                eval_turn: row.eval_turn,
                need: row.need,
                send: row.send,
            });
        }
        let next_send = row.desired.clamp(1, src_cap);
        if tested.contains_key(&next_send) {
            break;
        }
        current_send = next_send;
    }

    let mut candidate_sends: Vec<i64> = tested_order
        .iter()
        .filter(|s| tested.get(s).and_then(|v| *v).is_some())
        .copied()
        .collect();
    candidate_sends.sort_by(|a, b| {
        let ra = tested[a].unwrap();
        let rb = tested[b].unwrap();
        (a - seed_hint)
            .abs()
            .cmp(&(b - seed_hint).abs())
            .then(ra.turns.cmp(&rb.turns))
            .then(a.cmp(b))
    });
    for send in candidate_sends {
        let row = tested[&send].unwrap();
        if row.send < row.need || row.turns > max_arrival_turn {
            continue;
        }
        return Some(SettleResult {
            angle: row.angle,
            turns: row.turns,
            eval_turn: row.eval_turn,
            need: row.need,
            send: row.send,
        });
    }
    None
}

/// Re-settle a single-source mission against the current `PlanState`. Picks
/// the right source budget (attack vs inventory for Reinforce), then
/// dispatches to `settle_plan`/`settle_reinforce_plan` with the mission's
/// anchor/tolerance/eval-turn rule. Returns `(plan, left)` so the commit
/// caller can validate `plan.need <= left` before launching.
fn settle_single_source_mission(
    world: &WorldModel,
    mission: &Mission,
    option: &ShotOption,
    src: &Planet,
    target: &Planet,
    state: &PlanState,
    policy: &PolicyState,
) -> Option<(SettleResult, i64)> {
    let left = if matches!(mission.kind, MissionKind::Reinforce) {
        state
            .source_inventory_left(world, option.src_id)
            .min((src.ships as f64 * REINFORCE_MAX_SOURCE_FRACTION) as i64)
    } else {
        state.source_attack_left(policy, option.src_id)
    };
    if left <= 0 {
        return None;
    }
    let send_cap = left.min(option.send_cap);
    let planned = &state.planned_commitments;
    let plan = match mission.kind {
        MissionKind::Reinforce => settle_reinforce_plan(
            world,
            src,
            target,
            left,
            send_cap,
            planned,
            option.anchor_turn.unwrap_or(mission.turns),
            mission.turns,
            4,
        ),
        MissionKind::Rescue => {
            let hold_turn = mission.turns;
            settle_plan(
                world,
                src,
                target,
                left,
                send_cap,
                planned,
                policy,
                MissionKind::Rescue,
                move |_| hold_turn,
                option.anchor_turn,
                None,
                4,
            )
        }
        MissionKind::Snipe => {
            let enemy_eta = option.anchor_turn.unwrap_or(mission.turns);
            settle_plan(
                world,
                src,
                target,
                left,
                send_cap,
                planned,
                policy,
                MissionKind::Snipe,
                move |turns| turns.max(enemy_eta),
                option.anchor_turn,
                None,
                4,
            )
        }
        MissionKind::CrashExploit => {
            let desired_arrival = option.anchor_turn.unwrap_or(mission.turns);
            settle_plan(
                world,
                src,
                target,
                left,
                send_cap,
                planned,
                policy,
                MissionKind::CrashExploit,
                move |turns| turns.max(desired_arrival),
                option.anchor_turn,
                Some(CRASH_EXPLOIT_ETA_WINDOW),
                4,
            )
        }
        // Single / Recapture / fallthrough → plain capture.
        _ => settle_plan(
            world,
            src,
            target,
            left,
            send_cap,
            planned,
            policy,
            MissionKind::Capture,
            |turns| turns,
            None,
            None,
            4,
        ),
    };
    plan.map(|p| (p, left))
}

// ── 7. Mission builders ───────────────────────────────────────────────────

fn build_snipe_mission(
    world: &WorldModel,
    src: &Planet,
    target: &Planet,
    src_available: i64,
    planned: &PlannedCommitments,
    policy: &PolicyState,
) -> Option<Mission> {
    if target.owner != -1 {
        return None;
    }
    let mut enemy_etas: Vec<i64> = world
        .timeline_cache
        .arrivals(target.id)
        .iter()
        .filter(|ev| ev.owner != -1 && ev.owner != world.player && ev.ships > 0)
        .map(|ev| ev.turns)
        .collect();
    enemy_etas.sort_unstable();
    enemy_etas.dedup();
    if enemy_etas.is_empty() {
        return None;
    }

    let mut best: Option<Mission> = None;
    for &enemy_eta in enemy_etas.iter().take(3) {
        let hints = [target.ships + 1, target.ships + 8];
        let Some((probe, rough)) = world.best_probe_aim(
            src.id,
            target.id,
            src_available,
            &hints,
            None,
            None,
            Some(enemy_eta),
            Some(1),
        ) else {
            continue;
        };
        let sync_turn = rough.1.max(enemy_eta);
        if world.comet_ids.contains(&target.id) {
            let life = world.comet_life(target.id);
            if sync_turn >= life || sync_turn > COMET_MAX_CHASE_TURNS {
                continue;
            }
        }

        let Some(plan) = settle_plan(
            world,
            src,
            target,
            src_available,
            probe,
            planned,
            policy,
            MissionKind::Snipe,
            |turns| turns.max(enemy_eta),
            Some(enemy_eta),
            None,
            4,
        ) else {
            continue;
        };

        if world.comet_ids.contains(&target.id) {
            let life = world.comet_life(target.id);
            if plan.eval_turn >= life || plan.eval_turn > COMET_MAX_CHASE_TURNS {
                continue;
            }
        }

        let value = target_value(target, plan.eval_turn, world, policy);
        if value <= 0.0 {
            continue;
        }
        let score = value / (plan.send as f64 + plan.eval_turn as f64 * SNIPE_COST_TURN_WEIGHT + 1.0);
        let option = ShotOption {
            score,
            src_id: src.id,
            target_id: target.id,
            angle: plan.angle,
            turns: plan.turns,
            needed: plan.need,
            send_cap: plan.send,
            mission: MissionKind::Snipe,
            anchor_turn: Some(enemy_eta),
        };
        let mission = Mission {
            kind: MissionKind::Snipe,
            score,
            target_id: target.id,
            turns: plan.eval_turn,
            options: vec![option],
        };
        if best.as_ref().map_or(true, |b| mission.score > b.score) {
            best = Some(mission);
        }
    }
    best
}

fn build_rescue_missions(
    world: &WorldModel,
    policy: &PolicyState,
    planned: &PlannedCommitments,
) -> Vec<Mission> {
    let mut missions = Vec::new();
    for target in &world.my_planets {
        let Some(fall_turn) = world.fall_turn_map.get(&target.id).copied().flatten() else {
            continue;
        };
        if fall_turn > DEFENSE_LOOKAHEAD_TURNS {
            continue;
        }
        for src in &world.my_planets {
            if src.id == target.id {
                continue;
            }
            let src_available = policy.attack_budget.get(&src.id).copied().unwrap_or(0);
            if src_available < PARTIAL_SOURCE_MIN_SHIPS {
                continue;
            }
            let hints = [target.production + DEFENSE_SEND_MARGIN_BASE + 2];
            let Some((probe, _)) = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                &hints,
                None,
                Some(fall_turn),
                None,
                None,
            ) else {
                continue;
            };
            let Some(plan) = settle_plan(
                world,
                src,
                target,
                src_available,
                probe,
                planned,
                policy,
                MissionKind::Rescue,
                move |_| fall_turn,
                Some(fall_turn),
                None,
                4,
            ) else {
                continue;
            };
            let saved_turns = (world.remaining_steps - fall_turn).max(1);
            let value =
                (target.production * saved_turns) as f64 + target.ships.max(0) as f64 * DEFENSE_SHIP_VALUE;
            let score = value / (plan.send as f64 + plan.turns as f64 * DEFENSE_COST_TURN_WEIGHT + 1.0);
            let option = ShotOption {
                score,
                src_id: src.id,
                target_id: target.id,
                angle: plan.angle,
                turns: plan.turns,
                needed: plan.need,
                send_cap: plan.send,
                mission: MissionKind::Rescue,
                anchor_turn: Some(fall_turn),
            };
            missions.push(Mission {
                kind: MissionKind::Rescue,
                score,
                target_id: target.id,
                turns: fall_turn,
                options: vec![option],
            });
        }
    }
    missions
}

fn build_recapture_missions(
    world: &WorldModel,
    policy: &PolicyState,
    planned: &PlannedCommitments,
) -> Vec<Mission> {
    let mut missions = Vec::new();
    for target in &world.my_planets {
        let Some(fall_turn) = world.fall_turn_map.get(&target.id).copied().flatten() else {
            continue;
        };
        if fall_turn > DEFENSE_LOOKAHEAD_TURNS {
            continue;
        }
        for src in &world.my_planets {
            if src.id == target.id {
                continue;
            }
            let src_available = policy.attack_budget.get(&src.id).copied().unwrap_or(0);
            if src_available < PARTIAL_SOURCE_MIN_SHIPS {
                continue;
            }
            let hints = [target.production + DEFENSE_SEND_MARGIN_BASE + 2];
            let Some((probe, _)) = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                &hints,
                Some(fall_turn + 1),
                Some(fall_turn + RECAPTURE_LOOKAHEAD_TURNS),
                None,
                None,
            ) else {
                continue;
            };
            let Some(plan) = settle_plan(
                world,
                src,
                target,
                src_available,
                probe,
                planned,
                policy,
                MissionKind::Capture,
                |turns| turns,
                None,
                None,
                4,
            ) else {
                continue;
            };
            if plan.turns <= fall_turn || plan.turns - fall_turn > RECAPTURE_LOOKAHEAD_TURNS {
                continue;
            }
            let saved_turns = (world.remaining_steps - plan.turns).max(1);
            let value = RECAPTURE_PRODUCTION_WEIGHT * (target.production * saved_turns) as f64
                + RECAPTURE_IMMEDIATE_WEIGHT * target.ships.max(0) as f64;
            let score = value / (plan.send as f64 + plan.turns as f64 * RECAPTURE_COST_TURN_WEIGHT + 1.0);
            let option = ShotOption {
                score,
                src_id: src.id,
                target_id: target.id,
                angle: plan.angle,
                turns: plan.turns,
                needed: plan.need,
                send_cap: plan.send,
                mission: MissionKind::Recapture,
                anchor_turn: Some(fall_turn),
            };
            missions.push(Mission {
                kind: MissionKind::Recapture,
                score,
                target_id: target.id,
                turns: plan.turns,
                options: vec![option],
            });
        }
    }
    missions
}

fn build_reinforce_missions(
    world: &WorldModel,
    policy: &PolicyState,
    planned: &PlannedCommitments,
    spent_total: &HashMap<i64, i64>,
) -> Vec<Mission> {
    if !REINFORCE_ENABLED {
        return Vec::new();
    }
    let mut missions = Vec::new();
    if world.remaining_steps < REINFORCE_MIN_FUTURE_TURNS {
        return missions;
    }
    for target in &world.my_planets {
        let Some(fall_turn) = world.fall_turn_map.get(&target.id).copied().flatten() else {
            continue;
        };
        if target.production < REINFORCE_MIN_PRODUCTION {
            continue;
        }
        let hold_until = HORIZON.min(fall_turn + REINFORCE_HOLD_LOOKAHEAD);
        let max_arrival_turn = fall_turn.min(REINFORCE_MAX_TRAVEL_TURNS);

        for src in &world.my_planets {
            if src.id == target.id {
                continue;
            }
            let budget = world.source_inventory_left(src.id, spent_total);
            let source_cap = budget.min((src.ships as f64 * REINFORCE_MAX_SOURCE_FRACTION) as i64);
            if source_cap < PARTIAL_SOURCE_MIN_SHIPS {
                continue;
            }
            let hints = [target.production + REINFORCE_SAFETY_MARGIN + 2];
            let Some((probe, _)) = world.best_probe_aim(
                src.id,
                target.id,
                source_cap,
                &hints,
                None,
                Some(max_arrival_turn),
                None,
                None,
            ) else {
                continue;
            };
            let Some(plan) = settle_reinforce_plan(
                world,
                src,
                target,
                source_cap,
                probe,
                planned,
                hold_until,
                max_arrival_turn,
                4,
            ) else {
                continue;
            };
            let value = reinforce_value(target, hold_until, world, policy);
            let score = value / (plan.send as f64 + plan.turns as f64 * REINFORCE_COST_TURN_WEIGHT + 1.0);
            let option = ShotOption {
                score,
                src_id: src.id,
                target_id: target.id,
                angle: plan.angle,
                turns: plan.turns,
                needed: plan.need,
                send_cap: plan.send,
                mission: MissionKind::Reinforce,
                anchor_turn: Some(hold_until),
            };
            missions.push(Mission {
                kind: MissionKind::Reinforce,
                score,
                target_id: target.id,
                turns: fall_turn,
                options: vec![option],
            });
        }
    }
    missions
}

fn build_crash_exploit_missions(
    world: &WorldModel,
    policy: &PolicyState,
    planned: &PlannedCommitments,
) -> Vec<Mission> {
    if !CRASH_EXPLOIT_ENABLED || !world.is_four_player {
        return Vec::new();
    }
    let mut missions = Vec::new();
    for crash in detect_enemy_crashes(world) {
        let target = world.planet(crash.target_id);
        if target.owner == world.player {
            continue;
        }
        let desired_arrival = crash.crash_turn + CRASH_EXPLOIT_POST_CRASH_DELAY;
        for src in &world.my_planets {
            let src_available = policy.attack_budget.get(&src.id).copied().unwrap_or(0);
            if src_available < PARTIAL_SOURCE_MIN_SHIPS {
                continue;
            }
            let hints = [12, target.ships + 1];
            let Some((probe, _)) = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                &hints,
                None,
                None,
                Some(desired_arrival),
                Some(CRASH_EXPLOIT_ETA_WINDOW),
            ) else {
                continue;
            };
            let Some(plan) = settle_plan(
                world,
                src,
                target,
                src_available,
                probe,
                planned,
                policy,
                MissionKind::CrashExploit,
                move |turns| turns.max(desired_arrival),
                Some(desired_arrival),
                Some(CRASH_EXPLOIT_ETA_WINDOW),
                4,
            ) else {
                continue;
            };
            if !candidate_time_valid(target, plan.turns, world, LATE_CAPTURE_BUFFER) {
                continue;
            }
            let value = target_value(target, plan.turns, world, policy);
            if value <= 0.0 {
                continue;
            }
            let score = value / (plan.send as f64 + plan.turns as f64 * SNIPE_COST_TURN_WEIGHT + 1.0);
            let option = ShotOption {
                score,
                src_id: src.id,
                target_id: target.id,
                angle: plan.angle,
                turns: plan.turns,
                needed: plan.need,
                send_cap: plan.send,
                mission: MissionKind::CrashExploit,
                anchor_turn: Some(desired_arrival),
            };
            missions.push(Mission {
                kind: MissionKind::CrashExploit,
                score,
                target_id: target.id,
                turns: plan.turns,
                options: vec![option],
            });
        }
    }
    missions
}

// ── 8. Main plan_moves ────────────────────────────────────────────────────

struct PlanState {
    planned_commitments: PlannedCommitments,
    moves: Vec<(i64, f64, i64)>,
    spent_total: HashMap<i64, i64>,
}

impl PlanState {
    fn new() -> Self {
        Self {
            planned_commitments: HashMap::default(),
            moves: Vec::new(),
            spent_total: HashMap::default(),
        }
    }

    fn source_inventory_left(&self, world: &WorldModel, source_id: i64) -> i64 {
        world.source_inventory_left(source_id, &self.spent_total)
    }

    fn source_attack_left(&self, policy: &PolicyState, source_id: i64) -> i64 {
        let budget = policy.attack_budget.get(&source_id).copied().unwrap_or(0);
        (budget - self.spent_total.get(&source_id).copied().unwrap_or(0)).max(0)
    }

    fn append_move(&mut self, world: &WorldModel, src_id: i64, angle: f64, ships: i64) -> i64 {
        let send = ships.min(self.source_inventory_left(world, src_id));
        if send < 1 {
            return 0;
        }
        self.moves.push((src_id, angle, send));
        *self.spent_total.entry(src_id).or_insert(0) += send;
        send
    }

    fn finalize_moves(&self, world: &WorldModel) -> Vec<(i64, f64, i64)> {
        let mut used_final: HashMap<i64, i64> = HashMap::default();
        let mut out = Vec::with_capacity(self.moves.len());
        for &(src_id, angle, ships) in &self.moves {
            let source = world.planet(src_id);
            let max_allowed = source.ships - used_final.get(&src_id).copied().unwrap_or(0);
            let send = ships.min(max_allowed);
            if send >= 1 {
                out.push((src_id, angle, send));
                *used_final.entry(src_id).or_insert(0) += send;
            }
        }
        out
    }
}

fn compute_live_doomed(world: &WorldModel, state: &PlanState) -> HashSet<i64> {
    let mut doomed = HashSet::default();
    for planet in &world.my_planets {
        let status = world.hold_status(planet.id, &state.planned_commitments, DOOMED_EVAC_TURN_LIMIT);
        if !status.holds_full
            && status.fall_turn.is_some()
            && status.fall_turn.unwrap() <= DOOMED_EVAC_TURN_LIMIT
            && state.source_inventory_left(world, planet.id) >= DOOMED_MIN_SHIPS
        {
            doomed.insert(planet.id);
        }
    }
    doomed
}

/// Frontier-facing geometry shared between the doomed-evac and rear-staging
/// phases: where the action is (`targets`), how far each of our planets sits
/// from it (`distance`), and which of our planets are still healthy enough to
/// be a fallback (`safe_fronts`).
struct FrontierContext {
    targets: Vec<Planet>,
    distance: HashMap<i64, f64>,
    safe_fronts: Vec<Planet>,
}

impl FrontierContext {
    fn build(world: &WorldModel, live_doomed: &HashSet<i64>) -> Self {
        let targets: Vec<Planet> = if !world.enemy_planets.is_empty() {
            world.enemy_planets.clone()
        } else if !world.static_neutral_planets.is_empty() {
            world.static_neutral_planets.clone()
        } else {
            world.neutral_planets.clone()
        };
        let distance: HashMap<i64, f64> = world
            .my_planets
            .iter()
            .map(|p| {
                let d = if targets.is_empty() {
                    f64::INFINITY
                } else {
                    nearest_distance_to_set(p.x, p.y, &targets)
                };
                (p.id, d)
            })
            .collect();
        let safe_fronts: Vec<Planet> = world
            .my_planets
            .iter()
            .filter(|p| !live_doomed.contains(&p.id))
            .cloned()
            .collect();
        Self {
            targets,
            distance,
            safe_fronts,
        }
    }

    #[inline]
    fn dist_of(&self, planet_id: i64) -> f64 {
        self.distance.get(&planet_id).copied().unwrap_or(f64::INFINITY)
    }
}

fn time_filters_pass(
    target: &Planet,
    turns: i64,
    needed: i64,
    src_cap: i64,
    world: &WorldModel,
    policy: &PolicyState,
) -> bool {
    let buf = if world.is_very_late {
        VERY_LATE_CAPTURE_BUFFER
    } else {
        LATE_CAPTURE_BUFFER
    };
    if !candidate_time_valid(target, turns, world, buf) {
        return false;
    }
    if opening_filter(target, turns, needed, src_cap, world, policy) {
        return false;
    }
    true
}

pub fn plan_moves(world: &WorldModel) -> Vec<(i64, f64, i64)> {
    plan_moves_with_profile(world, PlanProfile::full())
}

pub fn plan_moves_with_profile(world: &WorldModel, profile: PlanProfile) -> Vec<(i64, f64, i64)> {
    plan_moves_full(world, profile, &HashSet::default()).moves
}

/// Per-plan output. `offensive_targets` is the score-descending unique list of
/// targets among offensive missions (Capture/Single/Snipe/Swarm/CrashExploit);
/// the caller uses it to drive forbid-prefix / only-target beam variants.
/// `top_offensive_target` is the head of that list, kept for older callers.
pub struct PlanOutput {
    pub moves: Vec<(i64, f64, i64)>,
    pub top_offensive_target: Option<i64>,
    pub offensive_targets: Vec<i64>,
}

#[inline]
fn is_offensive_mission(kind: MissionKind) -> bool {
    matches!(
        kind,
        MissionKind::Capture
            | MissionKind::Single
            | MissionKind::Snipe
            | MissionKind::Swarm
            | MissionKind::CrashExploit
    )
}

/// Per-WorldModel artifacts shared across candidate plans:
///   * `policy` is a pure function of the WorldModel.
///   * `missions` is the *heavy-superset* mission list. Swarm/CrashExploit
///     missions are present even though they used to be gated by
///     `profile.heavy`; [`plan_from_artifacts`] filters them out when running
///     a fast-profile candidate. Building one list lets 25 candidates share
///     the expensive O(my × all) sweep + multi-source swarm pairing instead
///     of repeating it per candidate.
///
/// The reinforce/rescue/recapture/crash-exploit builders and the offensive
/// sweep here all read `state.planned_commitments` / `state.spent_total` —
/// which are empty at this stage — so mission scores reflect the pre-commit
/// world. Per-candidate `plan_from_artifacts` runs its own `PlanState` for
/// the commit loop.
pub struct MissionArtifacts {
    pub policy: PolicyState,
    pub missions: Vec<Mission>,
    /// Pre-collected once so candidate commit loops don't rebuild them.
    pub my_planet_ids: Vec<i64>,
    pub all_planet_ids: Vec<i64>,
}

pub fn build_mission_artifacts(world: &WorldModel) -> MissionArtifacts {
    let policy = build_policy_state(world);
    let state = PlanState::new();

    // Per-target option lists for multi-source swarm consideration.
    let mut source_options_by_target: HashMap<i64, Vec<ShotOption>> = HashMap::default();
    let mut missions: Vec<Mission> = Vec::new();

    // Reinforce + rescue + recapture (defensive — target our own planets,
    // independent of profile).
    missions.extend(build_reinforce_missions(
        world,
        &policy,
        &state.planned_commitments,
        &state.spent_total,
    ));
    missions.extend(build_rescue_missions(
        world,
        &policy,
        &state.planned_commitments,
    ));
    missions.extend(build_recapture_missions(
        world,
        &policy,
        &state.planned_commitments,
    ));

    // Main per-source × per-target sweep.
    let my_planet_ids: Vec<i64> = world.my_planets.iter().map(|p| p.id).collect();
    let all_planet_ids: Vec<i64> = world.planets.iter().map(|p| p.id).collect();

    for src_id in &my_planet_ids {
        let src_available = state.source_attack_left(&policy, *src_id);
        if src_available <= 0 {
            continue;
        }
        let src = world.planet(*src_id);
        for target_id in &all_planet_ids {
            if *target_id == *src_id {
                continue;
            }
            let target = world.planet(*target_id);
            if target.owner == world.player {
                continue;
            }
            let hints = [target.ships + 1];
            let Some((_, rough_aim)) = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                &hints,
                None,
                None,
                None,
                None,
            ) else {
                continue;
            };
            let rough_turns = rough_aim.1;
            let buf = if world.is_very_late {
                VERY_LATE_CAPTURE_BUFFER
            } else {
                LATE_CAPTURE_BUFFER
            };
            if !candidate_time_valid(target, rough_turns, world, buf) {
                continue;
            }
            let global_needed = world.min_ships_to_own_at(
                target.id,
                rough_turns,
                world.player,
                &state.planned_commitments,
                &[],
                None,
            );
            if global_needed <= 0 {
                continue;
            }
            if opening_filter(target, rough_turns, global_needed, src_available, world, &policy) {
                continue;
            }

            // Swarm fragment (heavy-superset — fast candidates drop these in
            // plan_from_artifacts).
            let partial_send_cap = src_available
                .min(preferred_send(target, global_needed, rough_turns, src_available, world, &policy));
            if partial_send_cap >= PARTIAL_SOURCE_MIN_SHIPS {
                let partial_hints = [partial_send_cap, global_needed, target.ships + 1];
                if let Some((_, partial_aim)) = world.best_probe_aim(
                    src.id,
                    target.id,
                    partial_send_cap,
                    &partial_hints,
                    None,
                    None,
                    None,
                    None,
                ) {
                    let (p_angle, p_turns, _, _) = partial_aim;
                    if time_filters_pass(target, p_turns, global_needed, src_available, world, &policy) {
                        let partial_value =
                            target_value(target, p_turns, world, &policy);
                        if partial_value > 0.0 {
                            let partial_score = partial_value
                                / (partial_send_cap as f64
                                    + p_turns as f64 * ATTACK_COST_TURN_WEIGHT
                                    + 1.0);
                            source_options_by_target
                                .entry(target.id)
                                .or_default()
                                .push(ShotOption {
                                    score: partial_score,
                                    src_id: src.id,
                                    target_id: target.id,
                                    angle: p_angle,
                                    turns: p_turns,
                                    needed: global_needed,
                                    send_cap: partial_send_cap,
                                    mission: MissionKind::Swarm,
                                    anchor_turn: None,
                                });
                        }
                    }
                }
            }

            if global_needed <= src_available {
                let send_guess =
                    preferred_send(target, global_needed, rough_turns, src_available, world, &policy);
                if let Some(plan) = settle_plan(
                    world,
                    src,
                    target,
                    src_available,
                    send_guess,
                    &state.planned_commitments,
                    &policy,
                    MissionKind::Capture,
                    |turns| turns,
                    None,
                    None,
                    4,
                ) {
                    if time_filters_pass(target, plan.turns, plan.need, src_available, world, &policy)
                        && plan.send >= 1
                    {
                        let value =
                            target_value(target, plan.turns, world, &policy);
                        if value > 0.0 {
                            let score = value
                                / (plan.send as f64
                                    + plan.turns as f64 * ATTACK_COST_TURN_WEIGHT
                                    + 1.0);
                            let option = ShotOption {
                                score,
                                src_id: src.id,
                                target_id: target.id,
                                angle: plan.angle,
                                turns: plan.turns,
                                needed: plan.need,
                                send_cap: plan.send,
                                mission: MissionKind::Capture,
                                anchor_turn: None,
                            };
                            if plan.send >= plan.need {
                                missions.push(Mission {
                                    kind: MissionKind::Single,
                                    score,
                                    target_id: target.id,
                                    turns: plan.turns,
                                    options: vec![option],
                                });
                            }
                        }
                    }
                }
            }

            if let Some(snipe) =
                build_snipe_mission(world, src, target, src_available, &state.planned_commitments, &policy)
            {
                missions.push(snipe);
            }
        }
    }

    // Multi-source swarm pairing (heavy-superset).
    let target_ids_with_options: Vec<i64> = source_options_by_target.keys().copied().collect();
    for target_id in &target_ids_with_options {
        // `options` is only used to populate `top_options`; build `top_options`
        // directly from the source map and skip the intermediate clone.
        let Some(options_ref) = source_options_by_target.get(target_id) else {
            continue;
        };
        if options_ref.len() < 2 {
            continue;
        }
        let target = world.planet(*target_id);
        let mut top_options: Vec<ShotOption> = options_ref.clone();
        top_options.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        top_options.truncate(MULTI_SOURCE_TOP_K);

        for i in 0..top_options.len() {
            for j in (i + 1)..top_options.len() {
                let first = top_options[i];
                let second = top_options[j];
                if first.src_id == second.src_id {
                    continue;
                }
                let pair_tol = swarm_eta_tolerance(&[first, second], target, world);
                if (first.turns - second.turns).abs() > pair_tol {
                    continue;
                }
                let joint_turn = first.turns.max(second.turns);
                let total_cap = first.send_cap + second.send_cap;
                let need = world.min_ships_to_own_at(
                    *target_id,
                    joint_turn,
                    world.player,
                    &state.planned_commitments,
                    &[],
                    Some(total_cap),
                );
                if need <= 0 {
                    continue;
                }
                if first.send_cap >= need || second.send_cap >= need {
                    continue;
                }
                if total_cap < need {
                    continue;
                }
                let value = target_value(target, joint_turn, world, &policy);
                if value <= 0.0 {
                    continue;
                }
                let pair_score = value
                    / (need as f64 + joint_turn as f64 * ATTACK_COST_TURN_WEIGHT + 1.0)
                    * MULTI_SOURCE_PLAN_PENALTY;
                missions.push(Mission {
                    kind: MissionKind::Swarm,
                    score: pair_score,
                    target_id: *target_id,
                    turns: joint_turn,
                    options: vec![first, second],
                });
            }
        }

        // 3-source swarms only for sufficiently fat hostile targets.
        if THREE_SOURCE_SWARM_ENABLED
            && target.owner != -1
            && target.owner != world.player
            && target.ships >= THREE_SOURCE_MIN_TARGET_SHIPS
            && top_options.len() >= 3
        {
            for i in 0..top_options.len() {
                for j in (i + 1)..top_options.len() {
                    for k in (j + 1)..top_options.len() {
                        let trio = [top_options[i], top_options[j], top_options[k]];
                        let mut src_ids = HashSet::default();
                        for opt in &trio {
                            src_ids.insert(opt.src_id);
                        }
                        if src_ids.len() < 3 {
                            continue;
                        }
                        let trio_vec = trio.to_vec();
                        let trio_tol = swarm_eta_tolerance(&trio_vec, target, world);
                        let turns: [i64; 3] = [trio[0].turns, trio[1].turns, trio[2].turns];
                        let max_t = *turns.iter().max().unwrap();
                        let min_t = *turns.iter().min().unwrap();
                        if max_t - min_t > trio_tol {
                            continue;
                        }
                        let joint_turn = max_t;
                        let total_cap: i64 = trio.iter().map(|o| o.send_cap).sum();
                        let need = world.min_ships_to_own_at(
                            *target_id,
                            joint_turn,
                            world.player,
                            &state.planned_commitments,
                            &[],
                            Some(total_cap),
                        );
                        if need <= 0 || total_cap < need {
                            continue;
                        }
                        let pair_covers = (0..3)
                            .any(|a| ((a + 1)..3).any(|b| trio[a].send_cap + trio[b].send_cap >= need));
                        if pair_covers {
                            continue;
                        }
                        let value =
                            target_value(target, joint_turn, world, &policy);
                        if value <= 0.0 {
                            continue;
                        }
                        let trio_score = value
                            / (need as f64 + joint_turn as f64 * ATTACK_COST_TURN_WEIGHT + 1.0)
                            * THREE_SOURCE_PLAN_PENALTY;
                        missions.push(Mission {
                            kind: MissionKind::Swarm,
                            score: trio_score,
                            target_id: *target_id,
                            turns: joint_turn,
                            options: trio_vec,
                        });
                    }
                }
            }
        }
    }

    // Crash exploit (heavy-superset; build_crash_exploit_missions itself
    // gates on CRASH_EXPLOIT_ENABLED + 4-player mode).
    missions.extend(build_crash_exploit_missions(
        world,
        &policy,
        &state.planned_commitments,
    ));

    MissionArtifacts {
        policy,
        missions,
        my_planet_ids,
        all_planet_ids,
    }
}

/// Thin wrapper: build artifacts and immediately commit. Use this for one-off
/// plans; the rollout-search path uses [`build_mission_artifacts`] +
/// [`plan_from_artifacts`] directly so all candidates share one artifact set.
pub fn plan_moves_full(
    world: &WorldModel,
    profile: PlanProfile,
    forbidden_targets: &HashSet<i64>,
) -> PlanOutput {
    let artifacts = build_mission_artifacts(world);
    plan_from_artifacts(world, &artifacts, profile, forbidden_targets)
}

/// Per-candidate commit pass over a shared `MissionArtifacts`. Filters the
/// heavy-superset mission list by `forbidden_targets` (and drops Swarm/
/// CrashExploit when `profile.heavy` is false), then runs the same commit /
/// followup / doomed-evac / rear-staging pipeline that used to live inside
/// `plan_moves_full`.
pub fn plan_from_artifacts(
    world: &WorldModel,
    artifacts: &MissionArtifacts,
    profile: PlanProfile,
    forbidden_targets: &HashSet<i64>,
) -> PlanOutput {
    let policy = &artifacts.policy;
    let mut state = PlanState::new();
    let my_planet_ids: &[i64] = &artifacts.my_planet_ids;
    let all_planet_ids: &[i64] = &artifacts.all_planet_ids;

    // Heavy-superset filter: drop Swarm + CrashExploit for fast profile.
    // Iterate by reference (no Mission clones, no PolicyState clone).
    let mut missions: Vec<&Mission> = artifacts
        .missions
        .iter()
        .filter(|m| {
            !forbidden_targets.contains(&m.target_id)
                && (profile.heavy
                    || !matches!(m.kind, MissionKind::Swarm | MissionKind::CrashExploit))
        })
        .collect();
    missions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen_offensive: HashSet<i64> = HashSet::default();
    let offensive_targets: Vec<i64> = missions
        .iter()
        .filter(|m| is_offensive_mission(m.kind))
        .filter_map(|m| seen_offensive.insert(m.target_id).then_some(m.target_id))
        .collect();
    let top_offensive_target = offensive_targets.first().copied();

    for mission in &missions {
        let mission: &Mission = *mission;
        let target = world.planet(mission.target_id);
        match mission.kind {
            MissionKind::Single
            | MissionKind::Snipe
            | MissionKind::Rescue
            | MissionKind::Recapture
            | MissionKind::Reinforce
            | MissionKind::CrashExploit => {
                let option = mission.options[0];
                let src = world.planet(option.src_id);
                let Some((plan, left)) = settle_single_source_mission(
                    world, mission, &option, src, target, &state, &policy,
                ) else {
                    continue;
                };
                if plan.send < plan.need || plan.need > left {
                    continue;
                }
                let sent = state.append_move(world, option.src_id, plan.angle, plan.send);
                if sent < plan.need {
                    continue;
                }
                state
                    .planned_commitments
                    .entry(target.id)
                    .or_default()
                    .push(ArrivalEvent {
                        turns: plan.turns,
                        owner: world.player,
                        ships: sent,
                    });
            }
            MissionKind::Swarm => {
                // Per-source effective send cap given current spent_total.
                let mut limits: Vec<i64> = Vec::with_capacity(mission.options.len());
                for option in &mission.options {
                    let left = state.source_attack_left(&policy, option.src_id);
                    limits.push(left.min(option.send_cap));
                }
                if *limits.iter().min().unwrap_or(&0) <= 0 {
                    continue;
                }
                let total_cap: i64 = limits.iter().sum();
                let missing = world.min_ships_to_own_at(
                    target.id,
                    mission.turns,
                    world.player,
                    &state.planned_commitments,
                    &[],
                    Some(total_cap),
                );
                if missing <= 0 || total_cap < missing {
                    continue;
                }
                // Walk options sorted by (turns, -limit, src_id), distributing
                // need greedily while leaving room for later sources.
                let mut ordered: Vec<(ShotOption, i64)> = mission
                    .options
                    .iter()
                    .copied()
                    .zip(limits.iter().copied())
                    .collect();
                ordered.sort_by(|a, b| {
                    a.0.turns
                        .cmp(&b.0.turns)
                        .then(b.1.cmp(&a.1))
                        .then(a.0.src_id.cmp(&b.0.src_id))
                });
                let mut remaining = missing;
                let mut sends: HashMap<i64, i64> = HashMap::default();
                for idx in 0..ordered.len() {
                    let (option, limit) = ordered[idx];
                    let remaining_other: i64 =
                        ordered[(idx + 1)..].iter().map(|(_, l)| *l).sum();
                    let send = limit.min((remaining - remaining_other).max(0));
                    sends.insert(option.src_id, send);
                    remaining -= send;
                }
                if remaining > 0 {
                    continue;
                }
                let mut reaimed: Vec<(i64, f64, i64, i64)> = Vec::new();
                let mut aim_failed = false;
                for (option, _) in &ordered {
                    let send = sends.get(&option.src_id).copied().unwrap_or(0);
                    if send <= 0 {
                        continue;
                    }
                    let src = world.planet(option.src_id);
                    let Some(aim) = world.plan_shot(src.id, target.id, send) else {
                        aim_failed = true;
                        break;
                    };
                    let (angle, turns, _, _) = aim;
                    reaimed.push((option.src_id, angle, turns, send));
                }
                if aim_failed || reaimed.is_empty() {
                    continue;
                }
                let turns_only: Vec<i64> = reaimed.iter().map(|r| r.2).collect();
                let eta_tol = swarm_eta_tolerance(&mission.options, target, world);
                let max_t = *turns_only.iter().max().unwrap();
                let min_t = *turns_only.iter().min().unwrap();
                if max_t - min_t > eta_tol {
                    continue;
                }
                let actual_joint_turn = max_t;
                let extra: Vec<ArrivalEvent> = reaimed
                    .iter()
                    .map(|(_, _, t, s)| ArrivalEvent {
                        turns: *t,
                        owner: world.player,
                        ships: *s,
                    })
                    .collect();
                let (owner_after, _) = world.projected_state(
                    target.id,
                    actual_joint_turn,
                    &state.planned_commitments,
                    &extra,
                );
                if owner_after != world.player {
                    continue;
                }
                let mut committed: Vec<ArrivalEvent> = Vec::new();
                for (src_id, angle, turns, send) in &reaimed {
                    let actual = state.append_move(world, *src_id, *angle, *send);
                    if actual <= 0 {
                        continue;
                    }
                    committed.push(ArrivalEvent {
                        turns: *turns,
                        owner: world.player,
                        ships: actual,
                    });
                }
                let committed_total: i64 = committed.iter().map(|e| e.ships).sum();
                if committed_total < missing {
                    continue;
                }
                state
                    .planned_commitments
                    .entry(target.id)
                    .or_default()
                    .extend(committed);
            }
            MissionKind::Capture => {
                // Plain Capture mission shouldn't appear at top-level — sweeps
                // promote it to Single. Ignore defensively.
                continue;
            }
        }
    }

    // Optional follow-up pass: use remaining attack budget for one extra shot
    // per source. Skipped in very-late games and in the fast profile.
    if profile.heavy && !world.is_very_late {
        for src_id in my_planet_ids {
            let src_left = state.source_attack_left(&policy, *src_id);
            if src_left < FOLLOWUP_MIN_SHIPS {
                continue;
            }
            let src = world.planet(*src_id);
            // Stash only the chosen target's id; we re-resolve it via
            // `world.planet(...)` after the inner search to avoid carrying an
            // owned Planet across the loop boundary.
            let mut best: Option<(f64, i64, SettleResult)> = None;
            for target_id in all_planet_ids {
                if *target_id == *src_id {
                    continue;
                }
                let target = world.planet(*target_id);
                if target.owner == world.player {
                    continue;
                }
                if world.comet_ids.contains(&target.id)
                    && target.production <= LOW_VALUE_COMET_PRODUCTION
                {
                    continue;
                }
                let hints = [target.ships + 1];
                let Some((_, rough_aim)) = world.best_probe_aim(
                    src.id,
                    target.id,
                    src_left,
                    &hints,
                    None,
                    None,
                    None,
                    None,
                ) else {
                    continue;
                };
                let est_turns = rough_aim.1;
                if world.is_late && est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER {
                    continue;
                }
                let rough_needed = world.min_ships_to_own_at(
                    target.id,
                    est_turns,
                    world.player,
                    &state.planned_commitments,
                    &[],
                    Some(src_left),
                );
                if rough_needed <= 0 || rough_needed > src_left {
                    continue;
                }
                if opening_filter(target, est_turns, rough_needed, src_left, world, &policy) {
                    continue;
                }
                let send = preferred_send(target, rough_needed, est_turns, src_left, world, &policy);
                if send < rough_needed {
                    continue;
                }
                let Some(plan) = settle_plan(
                    world,
                    src,
                    target,
                    src_left,
                    send,
                    &state.planned_commitments,
                    &policy,
                    MissionKind::Capture,
                    |turns| turns,
                    None,
                    None,
                    4,
                ) else {
                    continue;
                };
                if world.is_late && plan.turns > world.remaining_steps - LATE_CAPTURE_BUFFER {
                    continue;
                }
                if plan.send < plan.need {
                    continue;
                }
                let value = target_value(target, plan.turns, world, &policy);
                if value <= 0.0 {
                    continue;
                }
                let score = value / (plan.send as f64 + plan.turns as f64 * ATTACK_COST_TURN_WEIGHT + 1.0);
                if best.as_ref().map_or(true, |b| score > b.0) {
                    best = Some((score, target.id, plan));
                }
            }
            let Some((_, target_id, plan)) = best else { continue };
            let target = world.planet(target_id);
            let src_left = state.source_attack_left(&policy, *src_id);
            if plan.need > src_left {
                continue;
            }
            let Some(plan2) = settle_plan(
                world,
                src,
                target,
                src_left,
                src_left.min(plan.send),
                &state.planned_commitments,
                &policy,
                MissionKind::Capture,
                |turns| turns,
                None,
                None,
                4,
            ) else {
                continue;
            };
            if plan2.send < plan2.need {
                continue;
            }
            let actual = state.append_move(world, src.id, plan2.angle, plan2.send);
            if actual < plan2.need {
                continue;
            }
            state
                .planned_commitments
                .entry(target.id)
                .or_default()
                .push(ArrivalEvent {
                    turns: plan2.turns,
                    owner: world.player,
                    ships: actual,
                });
        }
    }

    // `live_doomed` is invariant across the doomed-evac pass below: evac only
    // mutates `state.spent_total` and pushes commitments to captured targets
    // (never our own planets), neither of which feed back into the
    // `hold_status` query that defines doom for our own planets. So we build
    // it once here and reuse it for rear staging too.
    let live_doomed = compute_live_doomed(world, &state);
    let frontier = FrontierContext::build(world, &live_doomed);

    // Doomed evac: planets that look lost get one last capture, else retreat.
    if !live_doomed.is_empty() {
        for planet in &world.my_planets {
            if !live_doomed.contains(&planet.id) {
                continue;
            }
            let available_now = state.source_inventory_left(world, planet.id);
            if available_now < policy.reserve.get(&planet.id).copied().unwrap_or(0) {
                continue;
            }
            let mut best_capture: Option<(f64, i64, f64, i64, i64)> = None;
            for target in &world.planets {
                if target.id == planet.id || target.owner == world.player {
                    continue;
                }
                let hints = [available_now, target.ships + 1];
                let Some((_, probe_aim)) = world.best_probe_aim(
                    planet.id,
                    target.id,
                    available_now,
                    &hints,
                    None,
                    None,
                    None,
                    None,
                ) else {
                    continue;
                };
                let probe_turns = probe_aim.1;
                if probe_turns > world.remaining_steps - 2 {
                    continue;
                }
                let need = world.min_ships_to_own_at(
                    target.id,
                    probe_turns,
                    world.player,
                    &state.planned_commitments,
                    &[],
                    Some(available_now),
                );
                if need <= 0 || need > available_now {
                    continue;
                }
                let Some(plan) = settle_plan(
                    world,
                    planet,
                    target,
                    available_now,
                    available_now.min(need.max(target.ships + 1)),
                    &state.planned_commitments,
                    &policy,
                    MissionKind::Capture,
                    |turns| turns,
                    None,
                    None,
                    4,
                ) else {
                    continue;
                };
                if plan.send < plan.need {
                    continue;
                }
                let mut score = target_value(target, plan.turns, world, &policy)
                    / (plan.send as f64 + plan.turns as f64 + 1.0);
                if target.owner != -1 && target.owner != world.player {
                    score *= 1.05;
                }
                if best_capture.as_ref().map_or(true, |b| score > b.0) {
                    best_capture = Some((score, target.id, plan.angle, plan.turns, plan.send));
                }
            }
            if let Some((_, tid, angle, turns, send)) = best_capture {
                let actual = state.append_move(world, planet.id, angle, send);
                if actual >= 1 {
                    state
                        .planned_commitments
                        .entry(tid)
                        .or_default()
                        .push(ArrivalEvent {
                            turns,
                            owner: world.player,
                            ships: actual,
                        });
                }
                continue;
            }

            // Retreat: pick the safest ally to fall back to. We only need its
            // id, so iterate by reference and skip the safe-allies Vec clone.
            let Some(retreat_target_id) = world
                .my_planets
                .iter()
                .filter(|ally| ally.id != planet.id && !live_doomed.contains(&ally.id))
                .min_by(|a, b| {
                    frontier
                        .dist_of(a.id)
                        .partial_cmp(&frontier.dist_of(b.id))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            planet_distance(planet, a)
                                .partial_cmp(&planet_distance(planet, b))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                })
                .map(|p| p.id)
            else {
                continue;
            };
            let Some(aim) = world.plan_shot(planet.id, retreat_target_id, available_now) else {
                continue;
            };
            let (angle, _, _, _) = aim;
            state.append_move(world, planet.id, angle, available_now);
        }
    }

    // Rear staging: deep-back planets feed forward. Reuses the `live_doomed`
    // / `frontier` built before the evac pass (see invariant note above).
    if profile.heavy
        && (!world.enemy_planets.is_empty() || !world.neutral_planets.is_empty())
        && world.my_planets.len() > 1
        && !world.is_late
    {
        if !frontier.targets.is_empty() && !frontier.safe_fronts.is_empty() {
            let front_anchor_id = frontier
                .safe_fronts
                .iter()
                .min_by(|a, b| {
                    frontier
                        .dist_of(a.id)
                        .partial_cmp(&frontier.dist_of(b.id))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
                .id;
            let send_ratio = if world.is_four_player {
                REAR_SEND_RATIO_FOUR_PLAYER
            } else {
                REAR_SEND_RATIO_TWO_PLAYER
            };
            // Sort references rather than cloning the whole Vec<Planet>.
            let mut sorted_rears: Vec<&Planet> = world.my_planets.iter().collect();
            sorted_rears.sort_by(|a, b| {
                frontier
                    .dist_of(b.id)
                    .partial_cmp(&frontier.dist_of(a.id))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for rear in &sorted_rears {
                let rear: &Planet = *rear;
                if rear.id == front_anchor_id || live_doomed.contains(&rear.id) {
                    continue;
                }
                if state.source_attack_left(&policy, rear.id) < REAR_SOURCE_MIN_SHIPS {
                    continue;
                }
                let rear_dist = frontier.dist_of(rear.id);
                let anchor_dist = frontier.dist_of(front_anchor_id);
                if rear_dist < anchor_dist * REAR_DISTANCE_RATIO {
                    continue;
                }
                // Pick the staging front id (closest staged ally, or fallback
                // to ally closest to the rear's objective). Only the id is
                // needed downstream — avoid cloning the Planet.
                let stage_pick: Option<&Planet> = frontier
                    .safe_fronts
                    .iter()
                    .filter(|p| {
                        p.id != rear.id && frontier.dist_of(p.id) < rear_dist * REAR_STAGE_PROGRESS
                    })
                    .min_by(|a, b| {
                        planet_distance(rear, a)
                            .partial_cmp(&planet_distance(rear, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                let front_id = if let Some(p) = stage_pick {
                    p.id
                } else {
                    // No stage closer than REAR_STAGE_PROGRESS; pick the
                    // ally closest to the rear's objective.
                    let Some(objective) = frontier.targets.iter().min_by(|a, b| {
                        planet_distance(rear, a)
                            .partial_cmp(&planet_distance(rear, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }) else {
                        continue;
                    };
                    let Some(fallback) = frontier
                        .safe_fronts
                        .iter()
                        .filter(|p| p.id != rear.id)
                        .min_by(|a, b| {
                            planet_distance(a, objective)
                                .partial_cmp(&planet_distance(b, objective))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    else {
                        continue;
                    };
                    fallback.id
                };
                if front_id == rear.id {
                    continue;
                }
                let send = (state.source_attack_left(&policy, rear.id) as f64 * send_ratio) as i64;
                if send < REAR_SEND_MIN_SHIPS {
                    continue;
                }
                let Some(aim) = world.plan_shot(rear.id, front_id, send) else {
                    continue;
                };
                let (angle, turns, _, _) = aim;
                if turns > REAR_MAX_TRAVEL_TURNS {
                    continue;
                }
                state.append_move(world, rear.id, angle, send);
            }
        }
    }

    PlanOutput {
        moves: state.finalize_moves(world),
        top_offensive_target,
        offensive_targets,
    }
}

// ── 8b. Patience analysis ────────────────────────────────────────────────

/// Identify offensive targets where deferring one turn yields a strictly
/// better mission for that target. Returned set is meant to be force-gated
/// into every candidate's forbid set so the rollout cannot select an
/// eager-on-T plan under variance — patience is decided here, structurally.
///
/// Two improvement axes are checked against today's best mission for each
/// target:
///
///   1. **Swarm collapse** — today's mission is a multi-source swarm, but a
///      single member's ship count plus one turn of production is enough to
///      solo the capture by the same (or earlier) eval turn.
///   2. **Faster travel** — the dominant source's best aim with a bumped cap
///      reaches the target with `1 + travel' < today_eval`, i.e. the shorter
///      travel more than compensates for the one-turn launch delay.
///
/// Snipe / CrashExploit / Rescue / Recapture / Reinforce are skipped: their
/// scheduling is anchored to opponent or own-falling timing, so a one-turn
/// delay risks missing the window entirely.
pub fn patient_targets(world: &WorldModel, artifacts: &MissionArtifacts) -> HashSet<i64> {
    let mut wait_set: HashSet<i64> = HashSet::default();
    let mut considered: HashSet<i64> = HashSet::default();
    let planned: PlannedCommitments = HashMap::default();

    // Score-descending walk so the first mission we see for each target is
    // the one we'd actually commit (matches plan_from_artifacts' ordering).
    let mut missions = artifacts.missions.clone();
    missions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    for mission in &missions {
        if !considered.insert(mission.target_id) {
            continue;
        }
        // The top-scoring mission for this target is what the planner will
        // commit. If it isn't a Single/Swarm (Snipe/CrashExploit/Reinforce/
        // Rescue/Recapture), deferring risks missing its timing window — so
        // skip patience for this target entirely rather than re-evaluating
        // against a lower-score Single/Swarm we'd never actually choose.
        if !matches!(mission.kind, MissionKind::Single | MissionKind::Swarm) {
            continue;
        }
        if delaying_dominates(world, mission, &planned) {
            wait_set.insert(mission.target_id);
        }
    }
    wait_set
}

fn delaying_dominates(
    world: &WorldModel,
    mission: &Mission,
    planned: &PlannedCommitments,
) -> bool {
    let target_id = mission.target_id;
    let target_ships = world.planet(target_id).ships;
    let today_eval = mission.turns;

    if matches!(mission.kind, MissionKind::Swarm) {
        for option in &mission.options {
            if can_solo_after_delay(world, option.src_id, target_id, target_ships, today_eval, planned) {
                return true;
            }
        }
    }

    if let Some(dom) = mission.options.iter().max_by_key(|o| o.send_cap) {
        if can_travel_faster_after_delay(world, dom.src_id, target_id, target_ships, today_eval, planned) {
            return true;
        }
    }

    false
}

/// True iff `src` with one turn of production accrued can solo-capture
/// `target` by `today_eval` (i.e., arrival = 1 + travel' ≤ today_eval).
fn can_solo_after_delay(
    world: &WorldModel,
    src_id: i64,
    target_id: i64,
    target_ships: i64,
    today_eval: i64,
    planned: &PlannedCommitments,
) -> bool {
    let src = world.planet(src_id);
    let bumped_cap = src.ships + src.production;
    if bumped_cap < 1 {
        return false;
    }
    let hints = [target_ships + 1, target_ships + 4];
    let Some((aim_ships, aim)) = world.best_probe_aim(
        src_id, target_id, bumped_cap, &hints, None, None, None, None,
    ) else {
        return false;
    };
    let arrival = 1 + aim.1;
    if arrival > today_eval {
        return false;
    }
    let needed = world.min_ships_to_own_at(
        target_id, arrival, world.player, planned, &[], Some(bumped_cap),
    );
    needed > 0 && needed <= bumped_cap && aim_ships >= needed
}

/// True iff `src` with one turn of production accrued can capture `target`
/// with strictly fewer turns of game-time elapsed (`1 + travel' < today_eval`).
fn can_travel_faster_after_delay(
    world: &WorldModel,
    src_id: i64,
    target_id: i64,
    target_ships: i64,
    today_eval: i64,
    planned: &PlannedCommitments,
) -> bool {
    let src = world.planet(src_id);
    let bumped_cap = src.ships + src.production;
    if bumped_cap < 1 {
        return false;
    }
    let hints = [target_ships + 1, target_ships + 4];
    let Some((aim_ships, aim)) = world.best_probe_aim(
        src_id, target_id, bumped_cap, &hints, None, None, None, None,
    ) else {
        return false;
    };
    let arrival = 1 + aim.1;
    if arrival >= today_eval {
        return false;
    }
    let needed = world.min_ships_to_own_at(
        target_id, arrival, world.player, planned, &[], Some(bumped_cap),
    );
    needed > 0 && needed <= bumped_cap && aim_ships >= needed
}

// ── 9. Entry point ────────────────────────────────────────────────────────

/// Run the obnext-style planner end-to-end against a [`WorldState`] that the
/// caller already built.
pub fn plan(world: &WorldState) -> Vec<(i64, f64, i64)> {
    plan_with_profile(world, PlanProfile::full())
}

pub fn plan_with_profile(world: &WorldState, profile: PlanProfile) -> Vec<(i64, f64, i64)> {
    if world.my_planets.is_empty() {
        return Vec::new();
    }
    let model = WorldModel::build(world);
    plan_moves_with_profile(&model, profile)
}
