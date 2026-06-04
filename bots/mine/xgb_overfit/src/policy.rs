//! Simplified ow2-style action enumeration with uniqueness constraint
//! ("no two of my planets pick the same target this turn") and randomized
//! scoring so each sample yields a different joint action.
//!
//! Per the user: drop ow2's time-search loop, iterate over (source, target)
//! pairs directly. Each planet acts solo (no coordination), so cost-to-capture
//! is determined by a single fleet.

use crate::pathing;
use crate::sim::Action;
use crate::{GameState, Planet};
use std::collections::HashSet;

/// Cheap RNG (xorshift) we control end-to-end so MCTS is repeatable.
#[derive(Clone, Copy)]
pub struct XorRng(pub u64);

impl XorRng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_f64(&mut self) -> f64 {
        // Uniform [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Estimate "ships available to send" without immediately losing the planet
/// to known in-flight fleets. Approximates ow2's safe_excess but cheap:
/// subtract any incoming enemy fleets aimed straight at this planet
/// (within a small angular cone) up to the production cushion.
pub fn avail(state: &GameState, planet: &Planet, player: i32) -> i64 {
    let mut incoming: i64 = 0;
    for f in &state.fleets {
        if f.owner == player {
            continue;
        }
        let dx = planet.x - f.x;
        let dy = planet.y - f.y;
        let d2 = dx * dx + dy * dy;
        if d2 < 1e-9 {
            continue;
        }
        let target_dir = dy.atan2(dx);
        let mut diff = (f.angle - target_dir).abs();
        while diff > std::f64::consts::PI {
            diff = (diff - 2.0 * std::f64::consts::PI).abs();
        }
        // Within ~10 degrees and not pointing away.
        if diff < 0.18 {
            incoming += f.ships;
        }
    }
    let need = (incoming - planet.production * 5).max(0); // 5 turns of production buffer
    (planet.ships - need - 1).max(0)
}

/// Required ships from one source to capture one target solo.
/// Single iteration to estimate arrival-time inflation for enemy-owned.
fn required_for_target(
    state: &GameState,
    src: &Planet,
    tgt: &Planet,
    player: i32,
) -> Option<(i64, f64, i64)> {
    // First pass: send target.ships + 1.
    let init_required = tgt.ships.max(0) + 1;
    let mut ships = init_required;
    let mut path = pathing::dir_to_hit(src, tgt, ships, state, 0)?;
    // For enemy-owned targets, factor in their production over flight time.
    if tgt.owner != -1 && tgt.owner != player {
        let inflated = tgt.ships + tgt.production * path.time + 1;
        if inflated > ships {
            ships = inflated;
            path = pathing::dir_to_hit(src, tgt, ships, state, 0)?;
        }
    }
    Some((ships, path.angle, path.time))
}

/// Race filter: with all-sendable ships from each side, do I arrive sooner?
/// Skip target itself when the target belongs to an opponent (the target
/// being closest-to-itself doesn't count as reinforcement). Same rule as
/// ow2 but using full `avail` per planet rather than 10.
fn race_pass(state: &GameState, target: &Planet, player: i32) -> bool {
    let mut my_t = i64::MAX;
    let mut their_t = i64::MAX;
    for p in &state.planets {
        if p.owner == player {
            let a = avail(state, p, player);
            if a <= 0 {
                continue;
            }
            if let Some(r) = pathing::dir_to_hit(p, target, a, state, 0) {
                if r.time < my_t {
                    my_t = r.time;
                }
            }
        } else if p.owner != -1 && p.id != target.id {
            let a = avail(state, p, p.owner);
            if a <= 0 {
                continue;
            }
            if let Some(r) = pathing::dir_to_hit(p, target, a, state, 0) {
                if r.time < their_t {
                    their_t = r.time;
                }
            }
        }
    }
    my_t <= their_t
}

/// Fast joint-action sampler used inside rollouts. Skips `dir_to_hit` and
/// the race filter — uses straight-line distance/angle. Much faster than the
/// full policy at some accuracy cost. Suitable when sampling cost dominates
/// the rollout depth.
pub fn sample_joint_action_fast(state: &GameState, player: i32, rng: &mut XorRng) -> Vec<Action> {
    use crate::pathing::fleet_speed;
    let max_speed = state.max_speed;
    let _ = max_speed;
    let mut per_src: Vec<(i64, Vec<(i64, f64, f64, i64)>)> = Vec::new();
    for src in state.planets.iter().filter(|p| p.owner == player) {
        let a = (src.ships - 1).max(0);
        if a <= 0 {
            continue;
        }
        let mut cands: Vec<(i64, f64, f64, i64)> = Vec::new();
        for tgt in &state.planets {
            if tgt.id == src.id || tgt.owner == player {
                continue;
            }
            let required = tgt.ships.max(0) + 1;
            if required > a {
                continue;
            }
            let dx = tgt.x - src.x;
            let dy = tgt.y - src.y;
            let angle = dy.atan2(dx);
            let prod = tgt.production.max(1) as f64;
            let base = prod / (required as f64);
            cands.push((tgt.id, base, angle, required));
        }
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(TOP_K_PER_SRC);
        if !cands.is_empty() {
            per_src.push((src.id, cands));
        }
    }
    let mut order: Vec<usize> = (0..per_src.len()).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut used_tgt: HashSet<i64> = HashSet::new();
    let mut out: Vec<Action> = Vec::new();
    for &i in &order {
        let (src_id, cands) = &per_src[i];
        let remaining: Vec<&(i64, f64, f64, i64)> =
            cands.iter().filter(|c| !used_tgt.contains(&c.0)).collect();
        if remaining.is_empty() {
            continue;
        }
        let pick = if rng.next_f64() < 0.5 {
            remaining[0]
        } else {
            let k = (rng.next_u64() as usize) % remaining.len();
            remaining[k]
        };
        used_tgt.insert(pick.0);
        out.push((*src_id, pick.2, pick.3, player));
    }
    out
}

const TOP_K_PER_SRC: usize = 5;

/// Fast rollout policy: for each non-mine target, sort owned sources by
/// arrival time and either (a) launch from the closest source that can
/// solo-capture, or (b) accumulate ships from sources FURTHEST-first
/// (per user spec) until cumulative `safe_send` covers `target.ships + 1`,
/// trimming the last source's contribution to avoid waste.
///
/// `safe_send` per source uses the END-STATE check (per user spec): max
/// ships I can send now such that, considering all in-flight fleets and
/// production for `horizon` turns, the source still has ≥1 ship at horizon.
/// Ignores intermediate ownership flips. Cheap, single formula per source.
///
/// Designed for rollouts (≈50× faster than ow2_plan with caches warm).
pub fn rollout_policy_fast(state: &GameState, player: i32) -> Vec<Action> {
    rollout_policy_fast_excluded(state, player, None)
}

/// Generate up to N distinct top alternatives in one call. Shares the
/// expensive precomputation (extrapolation, race filter, dir_to_hit cache)
/// across all N plans. Each variant excludes a different non-mine target,
/// giving K mostly-similar-but-not-identical plans.
///
/// Order: result[0] = greedy (no exclusion); result[1..] = exclude target
/// in planet-id order.
pub fn rollout_policy_fast_top_n(state: &GameState, player: i32, n: usize) -> Vec<Vec<Action>> {
    if n == 0 {
        return Vec::new();
    }
    let (extrap_base, race_ok, dh_cache_owned) = compute_fast_context(state, player);
    let mut dh_cache = dh_cache_owned;
    let mut out: Vec<Vec<Action>> = Vec::new();
    let greedy = generate_plan(state, player, &extrap_base, &race_ok, &mut dh_cache, None);
    out.push(greedy);
    for tgt in &state.planets {
        if tgt.owner == player {
            continue;
        }
        if out.len() >= n {
            break;
        }
        let alt = generate_plan(state, player, &extrap_base, &race_ok, &mut dh_cache, Some(tgt.id));
        if !out.iter().any(|p| {
            // Same dedup as enumerate_alternatives*: by (source, quantized
            // angle, ships).
            if p.len() != alt.len() {
                return false;
            }
            let key = |x: &Action| (x.0, (x.1 * 100.0).round() as i64, x.2);
            let mut a: Vec<_> = p.iter().map(key).collect();
            let mut b: Vec<_> = alt.iter().map(key).collect();
            a.sort();
            b.sort();
            a == b
        }) {
            out.push(alt);
        }
    }
    if out.len() < n && !out.iter().any(|a| a.is_empty()) {
        out.push(Vec::new());
    }
    out
}

/// Build the (extrap, race_ok, empty dh_cache) precomputed context.
fn compute_fast_context(
    state: &GameState,
    player: i32,
) -> (
    std::collections::HashMap<i64, i64>,
    std::collections::HashMap<i64, bool>,
    std::collections::HashMap<(i64, i64, i64), Option<pathing::PathResult>>,
) {
    use std::collections::HashMap;
    const HORIZON: i64 = 20;
    let mut extrap: HashMap<i64, i64> = HashMap::new();
    let owner_of: HashMap<i64, i32> = state.planets.iter().map(|p| (p.id, p.owner)).collect();
    for p in &state.planets {
        let prod = if p.owner != -1 { p.production } else { 0 };
        extrap.insert(p.id, p.ships + prod * HORIZON);
    }
    for fleet in &state.fleets {
        if let Some((pid, dt)) = crate::ow2_plan::cached_predict_fleet_collision(fleet, state) {
            if dt > HORIZON {
                continue;
            }
            let towner = match owner_of.get(&pid) {
                Some(o) => *o,
                None => continue,
            };
            let signed = if fleet.owner == towner { fleet.ships } else { -fleet.ships };
            *extrap.entry(pid).or_insert(0) += signed;
        }
    }
    let my_total: i64 = state.planets.iter().filter(|p| p.owner == player).map(|p| p.ships).sum::<i64>()
        + state.fleets.iter().filter(|f| f.owner == player).map(|f| f.ships).sum::<i64>();
    let opp_total: i64 = state.planets.iter().filter(|p| p.owner != player && p.owner != -1).map(|p| p.ships).sum::<i64>()
        + state.fleets.iter().filter(|f| f.owner != player && f.owner != -1).map(|f| f.ships).sum::<i64>();
    let dominating = my_total >= 2 * opp_total.max(1);
    let max_speed = state.max_speed;
    let straight_time = |src: &Planet, tgt: &Planet, ships: i64| -> f64 {
        let dx = src.x - tgt.x;
        let dy = src.y - tgt.y;
        let dist = (dx * dx + dy * dy).sqrt();
        dist / pathing::fleet_speed(ships, max_speed)
    };
    let mut race_ok: HashMap<i64, bool> = HashMap::new();
    for tgt in &state.planets {
        if tgt.owner == player {
            continue;
        }
        if dominating {
            race_ok.insert(tgt.id, true);
            continue;
        }
        let my_t = state.planets.iter()
            .filter(|p| p.owner == player && p.ships > 0)
            .map(|p| straight_time(p, tgt, p.ships))
            .fold(f64::INFINITY, f64::min);
        let opp_t = state.planets.iter()
            .filter(|p| p.owner != player && p.owner != -1 && p.id != tgt.id && p.ships > 0)
            .map(|p| straight_time(p, tgt, p.ships))
            .fold(f64::INFINITY, f64::min);
        race_ok.insert(tgt.id, my_t < opp_t);
    }
    let dh_cache: HashMap<(i64, i64, i64), Option<pathing::PathResult>> = HashMap::new();
    (extrap, race_ok, dh_cache)
}

/// Per-target launch loop. Operates on a CLONE of extrap so each plan
/// gets its own incremental state.
fn generate_plan(
    state: &GameState,
    player: i32,
    extrap_base: &std::collections::HashMap<i64, i64>,
    race_ok: &std::collections::HashMap<i64, bool>,
    dh_cache: &mut std::collections::HashMap<(i64, i64, i64), Option<pathing::PathResult>>,
    excluded_target: Option<i64>,
) -> Vec<Action> {
    use std::collections::HashSet;
    let mut extrap = extrap_base.clone();
    let mut used_src: HashSet<i64> = HashSet::new();
    let mut out: Vec<Action> = Vec::new();
    for target in &state.planets {
        if target.owner == player {
            continue;
        }
        if Some(target.id) == excluded_target {
            continue;
        }
        if !*race_ok.get(&target.id).unwrap_or(&false) {
            continue;
        }
        let target_extrap = *extrap.get(&target.id).unwrap_or(&0);
        if target_extrap <= 0 {
            continue;
        }
        let need = target_extrap + 1;
        let mut srcs: Vec<(i64, i64, f64, i64)> = Vec::new();
        for src in &state.planets {
            if src.owner != player || used_src.contains(&src.id) {
                continue;
            }
            let src_extrap = *extrap.get(&src.id).unwrap_or(&0);
            let ss = (src_extrap - 1).max(0).min(src.ships);
            if ss <= 0 {
                continue;
            }
            let key = (src.id, target.id, ss.min(need));
            let r = if let Some(v) = dh_cache.get(&key) {
                *v
            } else {
                let v = pathing::dir_to_hit(src, target, ss.min(need), state, 0);
                dh_cache.insert(key, v);
                v
            };
            if let Some(r) = r {
                srcs.push((r.time, src.id, r.angle, ss));
            }
        }
        if srcs.is_empty() {
            continue;
        }
        srcs.sort_by_key(|x| x.0);
        let mut solo: Option<(i64, f64, i64)> = None;
        for &(_, sid, ang, safe_send) in &srcs {
            if safe_send >= need {
                solo = Some((sid, ang, need));
                break;
            }
        }
        if let Some((sid, ang, n)) = solo {
            used_src.insert(sid);
            *extrap.entry(sid).or_insert(0) -= n;
            *extrap.entry(target.id).or_insert(0) -= n;
            out.push((sid, ang, n, player));
            continue;
        }
        srcs.sort_by_key(|x| std::cmp::Reverse(x.0));
        let mut committed: Vec<(i64, f64, i64)> = Vec::new();
        let mut total = 0i64;
        for &(_, sid, ang, safe_send) in &srcs {
            committed.push((sid, ang, safe_send));
            total += safe_send;
            if total >= need {
                break;
            }
        }
        if total >= need {
            let extra = total - need;
            if let Some(last) = committed.last_mut() {
                last.2 -= extra;
            }
            for (sid, ang, n) in committed {
                used_src.insert(sid);
                if n > 0 {
                    *extrap.entry(sid).or_insert(0) -= n;
                    *extrap.entry(target.id).or_insert(0) -= n;
                    out.push((sid, ang, n, player));
                }
            }
        }
    }
    out
}

pub fn rollout_policy_fast_excluded(
    state: &GameState,
    player: i32,
    excluded_target: Option<i64>,
) -> Vec<Action> {
    use std::collections::{HashMap, HashSet};
    const HORIZON: i64 = 20;

    // Per-planet extrapolated "ships in favor of current owner" at HORIZON.
    // = current ships + production*HORIZON + (signed arrivals where +ve
    // means favorable to current owner).
    //
    // Interpretation: positive = owner holds the planet at horizon with
    // that many ships. Negative would mean owner loses control by horizon.
    // For a NEUTRAL planet, production=0 and net is signed by attacker.
    //
    // Incrementally updated: when I commit a launch from A→B with N ships,
    // A.extrap -= N (A loses N ships to flight) AND
    // B.extrap -= N (enemy B's defense weakens by my N arrival).
    let mut extrap: HashMap<i64, i64> = HashMap::new();
    let owner_of: HashMap<i64, i32> = state.planets.iter().map(|p| (p.id, p.owner)).collect();
    for p in &state.planets {
        let prod = if p.owner != -1 { p.production } else { 0 };
        extrap.insert(p.id, p.ships + prod * HORIZON);
    }
    for fleet in &state.fleets {
        if let Some((pid, dt)) = crate::ow2_plan::cached_predict_fleet_collision(fleet, state) {
            if dt > HORIZON {
                continue;
            }
            let towner = match owner_of.get(&pid) {
                Some(o) => *o,
                None => continue,
            };
            // Fleet arrives at planet pid. If fleet owner == planet owner:
            // reinforcement (+ favor). Else: attack (- favor).
            let signed = if fleet.owner == towner { fleet.ships } else { -fleet.ships };
            *extrap.entry(pid).or_insert(0) += signed;
        }
    }

    let mut used_src: HashSet<i64> = HashSet::new();
    let mut out: Vec<Action> = Vec::new();

    // Per-(src,tgt,ships) dir_to_hit cache local to this call.
    let mut dh_cache: HashMap<(i64, i64, i64), Option<pathing::PathResult>> = HashMap::new();
    let mut dir_to_hit_cached = |src: &Planet, tgt: &Planet, ships: i64| -> Option<pathing::PathResult> {
        let key = (src.id, tgt.id, ships);
        if let Some(v) = dh_cache.get(&key) {
            return *v;
        }
        let v = pathing::dir_to_hit(src, tgt, ships, state, 0);
        dh_cache.insert(key, v);
        v
    };

    // Race filter: skip targets where I can't hit them faster than the enemy.
    // Uses straight-line distance / fleet_speed (no obstacle dodging) — O(1)
    // per pair, sub-microsecond per call. Loses accuracy on sun-blocked or
    // orbiting targets but adequate for a rollout filter.
    //
    // Shortcut: if my total ships ≥ 2× enemy total, skip the per-target
    // race check (all targets are winnable when dominating).
    let my_total: i64 = state.planets.iter().filter(|p| p.owner == player).map(|p| p.ships).sum::<i64>()
        + state.fleets.iter().filter(|f| f.owner == player).map(|f| f.ships).sum::<i64>();
    let opp_total: i64 = state.planets.iter().filter(|p| p.owner != player && p.owner != -1).map(|p| p.ships).sum::<i64>()
        + state.fleets.iter().filter(|f| f.owner != player && f.owner != -1).map(|f| f.ships).sum::<i64>();
    let dominating = my_total >= 2 * opp_total.max(1);

    let straight_time = |src: &Planet, tgt: &Planet, ships: i64| -> f64 {
        let dx = src.x - tgt.x;
        let dy = src.y - tgt.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let speed = pathing::fleet_speed(ships, state.max_speed);
        dist / speed
    };

    let mut race_ok: HashMap<i64, bool> = HashMap::new();
    for tgt in &state.planets {
        if tgt.owner == player {
            continue;
        }
        if dominating {
            race_ok.insert(tgt.id, true);
            continue;
        }
        let my_t = state
            .planets
            .iter()
            .filter(|p| p.owner == player && p.ships > 0)
            .map(|p| straight_time(p, tgt, p.ships))
            .fold(f64::INFINITY, f64::min);
        let opp_t = state
            .planets
            .iter()
            .filter(|p| p.owner != player && p.owner != -1 && p.id != tgt.id && p.ships > 0)
            .map(|p| straight_time(p, tgt, p.ships))
            .fold(f64::INFINITY, f64::min);
        race_ok.insert(tgt.id, my_t < opp_t);
    }

    for target in &state.planets {
        if target.owner == player {
            continue;
        }
        if Some(target.id) == excluded_target {
            continue;
        }
        if !*race_ok.get(&target.id).unwrap_or(&false) {
            continue;
        }
        // Extrapolated defense at HORIZON for this target.
        // If target's owner already loses control (extrap <= 0), we don't
        // need to send anything — it's effectively ours.
        // Otherwise need ships > extrapolated defense.
        let target_extrap = *extrap.get(&target.id).unwrap_or(&0);
        if target_extrap <= 0 {
            continue;
        }
        let need = target_extrap + 1;

        // Gather owned sources sorted by arrival time + safe_send capacity.
        // safe_send for src = src.extrap - 1 (must keep ≥1 ship at horizon).
        // Capped at src.ships (can't send more than currently held).
        let mut srcs: Vec<(i64, i64, f64, i64)> = Vec::new(); // (time, src_id, angle, safe_send)
        for src in &state.planets {
            if src.owner != player || used_src.contains(&src.id) {
                continue;
            }
            let src_extrap = *extrap.get(&src.id).unwrap_or(&0);
            let ss = (src_extrap - 1).max(0).min(src.ships);
            if ss <= 0 {
                continue;
            }
            if let Some(r) = dir_to_hit_cached(src, target, ss.min(need)) {
                srcs.push((r.time, src.id, r.angle, ss));
            }
        }
        if srcs.is_empty() {
            continue;
        }

        // (a) Solo: closest source whose safe_send covers need.
        srcs.sort_by_key(|x| x.0);
        let mut solo: Option<(i64, f64, i64)> = None;
        for &(_, sid, ang, safe_send) in &srcs {
            if safe_send >= need {
                solo = Some((sid, ang, need));
                break;
            }
        }
        if let Some((sid, ang, n)) = solo {
            used_src.insert(sid);
            // Incremental extrap update: src loses N ships (becomes worse for
            // me); target enemy loses N defense (becomes better for me).
            *extrap.entry(sid).or_insert(0) -= n;
            *extrap.entry(target.id).or_insert(0) -= n;
            out.push((sid, ang, n, player));
            continue;
        }

        // (b) Multi-source: furthest-first accumulation until total covers need.
        srcs.sort_by_key(|x| std::cmp::Reverse(x.0));
        let mut committed: Vec<(i64, f64, i64)> = Vec::new();
        let mut total = 0i64;
        for &(_, sid, ang, safe_send) in &srcs {
            committed.push((sid, ang, safe_send));
            total += safe_send;
            if total >= need {
                break;
            }
        }
        if total >= need {
            let extra = total - need;
            if let Some(last) = committed.last_mut() {
                last.2 -= extra;
            }
            for (sid, ang, n) in committed {
                used_src.insert(sid);
                if n > 0 {
                    *extrap.entry(sid).or_insert(0) -= n;
                    *extrap.entry(target.id).or_insert(0) -= n;
                    out.push((sid, ang, n, player));
                }
            }
        }
    }
    out
}


/// Sample one joint action for `player`. Mode is controlled by env var
/// `OW_SAMPLER`:
///   - "ow2_alt" (default): 40% ow2_plan as-is, 50% ow2_plan with one
///     non-mine target excluded (forces a real alternative), 10% empty.
///     Accurate but slow (~30ms per call) — yields ~1-3 MCTS iters/turn.
///   - "legacy": cheap race-filter+random sampler (~0.5ms per call).
pub fn sample_joint_action(state: &GameState, player: i32, rng: &mut XorRng) -> Vec<Action> {
    let mode = std::env::var("OW_SAMPLER").unwrap_or_else(|_| "ow2_alt".to_string());
    if mode == "ow2_alt" {
        let nc = no_coop_default();
        let r = rng.next_f64();
        if r < 0.40 {
            return crate::ow2_plan::plan(state, player, nc);
        }
        if r < 0.70 {
            let candidates: Vec<i64> = state
                .planets
                .iter()
                .filter(|p| p.owner != player)
                .map(|p| p.id)
                .collect();
            if !candidates.is_empty() {
                let excluded = candidates[(rng.next_u64() as usize) % candidates.len()];
                return crate::ow2_plan::plan_with_exclusion(state, player, nc, Some(excluded));
            }
        }
        // Remaining 30%: free random — any source to any target with random
        // ship count. Lets MCTS explore moves ow2_plan never generates.
        return random_free_action(state, player, rng);
    }
    sample_joint_action_legacy(state, player, rng)
}

fn sample_joint_action_legacy(state: &GameState, player: i32, rng: &mut XorRng) -> Vec<Action> {
    // Precompute race-eligibility per non-mine target.
    let mut race_ok: std::collections::HashMap<i64, bool> = std::collections::HashMap::new();
    for t in &state.planets {
        if t.owner != player {
            race_ok.insert(t.id, race_pass(state, t, player));
        }
    }

    // Per-source list of top-K candidate (tgt_id, score, angle, ships),
    // sorted by score desc. Score is deterministic — variety comes from the
    // sampling step below.
    let mut per_src: Vec<(i64, Vec<(i64, f64, f64, i64)>)> = Vec::new();
    for src in state.planets.iter().filter(|p| p.owner == player) {
        let a = avail(state, src, player);
        if a <= 0 {
            continue;
        }
        let mut cands: Vec<(i64, f64, f64, i64)> = Vec::new();
        for tgt in &state.planets {
            if tgt.id == src.id || tgt.owner == player {
                continue;
            }
            if !*race_ok.get(&tgt.id).unwrap_or(&false) {
                continue;
            }
            let (ships, angle, _) = match required_for_target(state, src, tgt, player) {
                Some(v) => v,
                None => continue,
            };
            if ships > a {
                continue;
            }
            let prod = tgt.production.max(1) as f64;
            let base = prod / (ships.max(1) as f64);
            cands.push((tgt.id, base, angle, ships));
        }
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(TOP_K_PER_SRC);
        if !cands.is_empty() {
            per_src.push((src.id, cands));
        }
    }

    // Randomize source assignment order so different samples shuffle who
    // grabs which target first.
    let mut order: Vec<usize> = (0..per_src.len()).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }

    let mut used_tgt: HashSet<i64> = HashSet::new();
    let mut actions: Vec<Action> = Vec::new();
    for &i in &order {
        let (src_id, cands) = &per_src[i];
        let remaining: Vec<&(i64, f64, f64, i64)> =
            cands.iter().filter(|c| !used_tgt.contains(&c.0)).collect();
        if remaining.is_empty() {
            continue;
        }
        // 50% pick best remaining; 50% pick random remaining. Keeps a strong
        // bias toward the high-quality candidate while still exploring.
        let pick = if rng.next_f64() < 0.5 {
            remaining[0]
        } else {
            let k = (rng.next_u64() as usize) % remaining.len();
            remaining[k]
        };
        used_tgt.insert(pick.0);
        actions.push((*src_id, pick.2, pick.3, player));
    }
    actions
}

fn no_coop_default() -> bool {
    std::env::var("OW_NO_COOP").is_ok()
}

/// Sample a free-form joint action: for each of my planets independently,
/// with probability p decide to act, then pick any other planet as target
/// and a random ship count (1..ships-1). Uses pathing::dir_to_hit for the
/// angle so the fleet has a chance of actually arriving. Returns 0..N
/// moves (one per acting source). Gives MCTS access to actions ow2_plan
/// would never generate.
fn random_free_action(state: &GameState, player: i32, rng: &mut XorRng) -> Vec<Action> {
    let my_planets: Vec<&Planet> = state
        .planets
        .iter()
        .filter(|p| p.owner == player && p.ships > 1)
        .collect();
    if my_planets.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for src in &my_planets {
        // 60% chance this planet acts. Otherwise hold.
        if rng.next_f64() > 0.6 {
            continue;
        }
        let targets: Vec<&Planet> = state.planets.iter().filter(|p| p.id != src.id).collect();
        if targets.is_empty() {
            continue;
        }
        let tgt = targets[(rng.next_u64() as usize) % targets.len()];
        let max_ships = (src.ships - 1).max(1);
        let ships = 1 + (rng.next_u64() as i64).rem_euclid(max_ships);
        if let Some(p) = crate::pathing::dir_to_hit(src, tgt, ships, state, 0) {
            out.push((src.id, p.angle, ships, player));
        }
    }
    out
}

/// Deterministic strong policy — full ow2 plan. Cooperation is on by
/// default; set OW_NO_COOP=1 to force one dispatch per target.
pub fn greedy_joint_action(state: &GameState, player: i32) -> Vec<Action> {
    crate::ow2_plan::plan(state, player, no_coop_default())
}

#[allow(dead_code)]
fn greedy_joint_action_legacy(state: &GameState, player: i32) -> Vec<Action> {
    let mut race_ok: std::collections::HashMap<i64, bool> = std::collections::HashMap::new();
    for t in &state.planets {
        if t.owner != player {
            race_ok.insert(t.id, race_pass(state, t, player));
        }
    }
    let mut per_src: Vec<(i64, Vec<(i64, f64, f64, i64)>)> = Vec::new();
    for src in state.planets.iter().filter(|p| p.owner == player) {
        let a = avail(state, src, player);
        if a <= 0 {
            continue;
        }
        let mut cands: Vec<(i64, f64, f64, i64)> = Vec::new();
        for tgt in &state.planets {
            if tgt.id == src.id || tgt.owner == player {
                continue;
            }
            if !*race_ok.get(&tgt.id).unwrap_or(&false) {
                continue;
            }
            let (ships, angle, _) = match required_for_target(state, src, tgt, player) {
                Some(v) => v,
                None => continue,
            };
            if ships > a {
                continue;
            }
            let prod = tgt.production.max(1) as f64;
            cands.push((tgt.id, prod / (ships as f64), angle, ships));
        }
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if !cands.is_empty() {
            per_src.push((src.id, cands));
        }
    }
    // Greedy: sort src by best-score desc and let each grab its top
    // unused target.
    per_src.sort_by(|a, b| {
        b.1[0].1.partial_cmp(&a.1[0].1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut used_tgt: HashSet<i64> = HashSet::new();
    let mut out: Vec<Action> = Vec::new();
    for (src_id, cands) in &per_src {
        for c in cands {
            if !used_tgt.contains(&c.0) {
                used_tgt.insert(c.0);
                out.push((*src_id, c.2, c.3, player));
                break;
            }
        }
    }
    out
}
