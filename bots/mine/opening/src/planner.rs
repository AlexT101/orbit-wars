//! Opening planner: given a set of target planet IDs, find the schedule
//! that captures them all as fast as possible.
//!
//! Approach: Held–Karp-style bitmask DP. State = `(captured_mask,
//! last_added_target_idx)`. Value = `(last_arrival, full source state,
//! schedule)`. For each (mask, i), iterate over `(prev_j, source)` and
//! keep the transition with the smallest resulting `last_arrival`.
//!
//! This collapses the `k!` orderings DFS would explore down to `2^k * k`
//! distinct cells. Each transition does one `best_for_pair` lookup,
//! which still scans wait_offsets × production-aligned ship counts.

use alphaow_bot::pathing::{dir_to_hit, PathResult};
use alphaow_bot::{GameState, Planet};
use std::time::{Duration, Instant};

/// One scheduled launch.
#[derive(Clone, Debug)]
pub struct Send {
    pub src_pid: i64,
    pub tgt_pid: i64,
    pub send_global_turn: i64,
    pub ships: i64,
    pub angle: f64,
    pub arrival_global_turn: i64,
}

#[derive(Clone, Debug)]
struct Source {
    pid: i64,
    ships: i64,
    production: i64,
    /// Global turn from which this source becomes available.
    avail: i64,
}

/// How many turns past min-ships-turn to consider waiting. Bigger fleets
/// travel faster (`fleet_speed` ∝ ln(ships)^1.5), so a few turns of
/// production can pay back many turns of travel time. Past this the
/// linearly-growing wait dominates the diminishing speed-up.
const WAIT_HORIZON: i64 = 25;

/// How many extra production-step values to try beyond the smallest
/// production-aligned send size. (n_aligned, n_aligned + prod,
/// n_aligned + 2*prod, …)
const PROD_STEPS_BEYOND_ALIGNED: i64 = 5;

fn min_send_turn_for_ships(src: &Source, ships_needed: i64) -> i64 {
    if src.ships >= ships_needed { return src.avail; }
    if src.production <= 0 { return i64::MAX; }
    let deficit = ships_needed - src.ships;
    let extra_turns = (deficit + src.production - 1) / src.production;
    src.avail + extra_turns
}

/// Ship counts to try for a launch.
///
/// Always include the minimum `need` (anything less wouldn't capture, so
/// sending less than enough is wasted). Beyond that, pick sizes that leave
/// the source with a ship count divisible by `production` — so future
/// production keeps the source on clean production-aligned increments
/// rather than accumulating an off-by-k remainder forever. We try the
/// smallest production-aligned n ≥ need, then a few more in production-
/// sized steps.
fn ship_count_candidates(need: i64, avail: i64, production: i64) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(PROD_STEPS_BEYOND_ALIGNED as usize + 2);
    out.push(need);
    if production > 0 {
        // Smallest n ≥ need with (avail - n) divisible by production.
        // Equivalent to n ≡ avail  (mod production).
        let need_mod = need.rem_euclid(production);
        let avail_mod = avail.rem_euclid(production);
        let delta = (avail_mod - need_mod).rem_euclid(production);
        let mut n = need + delta;
        for _ in 0..=PROD_STEPS_BEYOND_ALIGNED {
            if n > avail { break; }
            if n >= need { out.push(n); }
            n += production;
        }
    }
    out.retain(|&n| n >= need && n <= avail);
    out.sort();
    out.dedup();
    out
}

/// Best (send_turn, path, ships, arrival) for a SINGLE (source, target).
/// Scans wait_offsets × ship counts; returns the earliest-arriving combo.
fn best_for_pair(
    state: &GameState,
    src: &Source,
    target: &Planet,
) -> Option<(i64, PathResult, i64, i64)> {
    let need = (target.ships + 1).max(1);
    let src_planet = state.planets.iter().find(|p| p.id == src.pid)?;
    let min_send = min_send_turn_for_ships(src, need);
    if min_send == i64::MAX { return None; }
    let mut best: Option<(i64, PathResult, i64, i64)> = None;
    for wait in 0..=WAIT_HORIZON {
        let send_turn = min_send + wait;
        let dt = send_turn - state.step;
        if dt < 0 { continue; }
        let ships_avail = src.ships + (send_turn - src.avail) * src.production;
        if ships_avail < need { continue; }
        for ships in ship_count_candidates(need, ships_avail, src.production) {
            let pr = match dir_to_hit(src_planet, target, ships, state, dt) {
                Some(r) => r,
                None => continue,
            };
            let arrival = send_turn + pr.time;
            if best.as_ref().map_or(true, |b| arrival < b.3) {
                best = Some((send_turn, pr, ships, arrival));
            }
        }
    }
    best
}

/// One DP cell: minimum-arrival way to reach a specific `(mask, last)`.
#[derive(Clone)]
struct DpCell {
    last_arrival: i64,
    sources: Vec<Source>,
    schedule: Vec<Send>,
}

/// Apply a transition: take `prev` cell + chosen source index + target +
/// best_for_pair output and produce a new DpCell representing the state
/// after that capture. Mutates the cloned sources list and schedule.
fn apply_transition(
    prev: &DpCell,
    src_idx: usize,
    target: &Planet,
    send_turn: i64,
    pr: &PathResult,
    ships: i64,
    arrival: i64,
) -> DpCell {
    let mut sources = prev.sources.clone();
    let src = &mut sources[src_idx];
    let ships_at_send = src.ships + (send_turn - src.avail) * src.production;
    src.ships = ships_at_send - ships;
    src.avail = send_turn;
    let src_pid = sources[src_idx].pid;
    sources.push(Source {
        pid: target.id,
        ships: 1,
        production: target.production,
        avail: arrival,
    });
    let mut schedule = prev.schedule.clone();
    schedule.push(Send {
        src_pid,
        tgt_pid: target.id,
        send_global_turn: send_turn,
        ships,
        angle: pr.angle,
        arrival_global_turn: arrival,
    });
    let new_last = prev.last_arrival.max(arrival);
    DpCell { last_arrival: new_last, sources, schedule }
}

/// Bitmask DP over the captured set. State `(mask, i)` = "exactly the
/// targets in `mask` are captured, and target `i` was the most recent
/// addition". Transition: from `(mask\{i}, j)` over all sources in that
/// cell's source state, pick the best target-i dispatch.
pub fn plan(
    state: &GameState,
    targets: &[i64],
    my_player: i32,
) -> Option<Vec<Send>> {
    let initial_sources: Vec<Source> = state.planets.iter()
        .filter(|p| p.owner == my_player)
        .map(|p| Source {
            pid: p.id,
            ships: p.ships,
            production: p.production,
            avail: state.step,
        })
        .collect();
    if initial_sources.is_empty() { return None; }

    let k = targets.len();
    if k == 0 { return None; }
    if k > 16 { return None; } // 2^16 cells would be 65k * 16 = 1M cells; cap

    let budget_ms: u64 = std::env::var("OPENING_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let deadline = Instant::now() + Duration::from_millis(budget_ms);

    // Resolve target planet refs once.
    let target_planets: Vec<Planet> = targets.iter()
        .filter_map(|&tid| state.planets.iter().find(|p| p.id == tid).cloned())
        .collect();
    if target_planets.len() != k { return None; }

    // dp[mask * k + i] holds the best cell for that (mask, last=i) pair.
    let n_cells = (1usize << k) * k;
    let mut dp: Vec<Option<DpCell>> = (0..n_cells).map(|_| None).collect();

    // Seed singletons: capture target i first from one of the initial sources.
    let seed = DpCell { last_arrival: state.step, sources: initial_sources.clone(),
                        schedule: Vec::new() };
    for i in 0..k {
        let target = &target_planets[i];
        let mut best: Option<DpCell> = None;
        for (si, src) in seed.sources.iter().enumerate() {
            let (send_turn, pr, ships, arrival) = match best_for_pair(state, src, target) {
                Some(v) => v,
                None => continue,
            };
            let new = apply_transition(&seed, si, target, send_turn, &pr, ships, arrival);
            if best.as_ref().map_or(true, |c| new.last_arrival < c.last_arrival) {
                best = Some(new);
            }
        }
        dp[(1usize << i) * k + i] = best;
    }

    // Iterate masks in popcount order so dp[mask\{i}][j] is filled before
    // dp[mask][i] is computed.
    let mut masks_by_size: Vec<Vec<u32>> = vec![vec![]; k + 1];
    for m in 1..(1u32 << k) {
        masks_by_size[m.count_ones() as usize].push(m);
    }

    for size in 2..=k {
        for &mask in &masks_by_size[size] {
            if Instant::now() >= deadline { break; }
            for i in 0..k {
                if mask & (1u32 << i) == 0 { continue; }
                let target = &target_planets[i];
                let prev_mask = mask & !(1u32 << i);
                let cell_idx = (mask as usize) * k + i;
                let mut best: Option<DpCell> = None;
                // For each j ∈ prev_mask, try all sources in dp[prev_mask][j].
                let mut j = 0usize;
                while j < k {
                    if prev_mask & (1u32 << j) != 0 {
                        if let Some(prev) = dp[(prev_mask as usize) * k + j].clone() {
                            for (si, src) in prev.sources.iter().enumerate() {
                                let (send_turn, pr, ships, arrival) =
                                    match best_for_pair(state, src, target) {
                                        Some(v) => v,
                                        None => continue,
                                    };
                                let new_last = prev.last_arrival.max(arrival);
                                if best.as_ref().map_or(true, |c| new_last < c.last_arrival) {
                                    best = Some(apply_transition(
                                        &prev, si, target, send_turn, &pr, ships, arrival));
                                }
                            }
                        }
                    }
                    j += 1;
                }
                dp[cell_idx] = best;
            }
        }
    }

    // Read out: among full-mask cells, pick the smallest last_arrival.
    let full_mask = (1u32 << k) - 1;
    let mut best: Option<DpCell> = None;
    for i in 0..k {
        if let Some(c) = &dp[(full_mask as usize) * k + i] {
            if best.as_ref().map_or(true, |b| c.last_arrival < b.last_arrival) {
                best = Some(c.clone());
            }
        }
    }
    best.map(|c| c.schedule)
}
