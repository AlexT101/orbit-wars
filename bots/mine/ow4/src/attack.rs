//! Attack planner with global-balance scoring + look-ahead delay search.
//!
//! For each candidate `(source, target)`:
//!
//!  1. Score acting AT DELAY 0 — exactly per the previous attack module.
//!  2. Score acting AT DELAY T (for T ∈ {2, 5, 10}). To do that we build a
//!     `future_state` where everyone accrued T turns of production and any
//!     in-flight fleets are gone (assumed to have arrived/resolved). Then
//!     compute the same scenario-A-vs-B score against that future state.
//!  3. Pick the best delay. If the best is `delay == 0`, queue this
//!     candidate for the current turn. Otherwise, NO-OP for this (src, tgt)
//!     — waiting genuinely beats acting now.
//!
//! Scenario A (per candidate):
//!  * I send the chosen ships now (or at the chosen delay).
//!  * Every enemy planet launches its full garrison at target one tick later.
//!  * Every other owned planet launches its full garrison at target one tick
//!    after that.
//!  * Existing in-flight fleets keep flying.
//!
//! Scenario B: nothing happens; extrapolate to scenario-A's end-tick.
//!
//! `score = (ΣmyShips − ΣenemyShips)(A) − (ΣmyShips − ΣenemyShips)(B)`.
//!
//! NOTE: the future-state model assumes the opponent ALSO no-ops during the
//! wait (mirrors production but doesn't capture). With apollo as the real
//! opponent this is optimistic for the "wait" choice — the bot may
//! over-no-op until we model opponent moves explicitly.

use std::collections::HashSet;

use crate::combat::{simulate_planet, Arrival};
use crate::game::{Action, GameState, Planet};
use crate::ledger::Ledger;
use crate::pathing::dir_to_hit;

const LOOKAHEAD_DELAYS: &[i64] = &[0, 2, 5, 10];
/// Absolute end-tick for scoring. All scenarios (act now, wait then act,
/// no-action baseline) are compared at this same tick so waiting's cost
/// shows up.
const SCORE_END_TICK: i64 = 30;

pub struct ScoredCandidate {
    pub from_id: i64,
    pub target_id: i64,
    pub angle: f64,
    pub ships: i64,
    pub score: f64,
    pub chosen_delay: i64,
    /// Arrival tick (relative to now) for tie-breaking — earlier wins.
    pub arrival_dt: i64,
}

/// Future state model: everyone's ships grow by `turns * production`,
/// in-flight fleets are assumed resolved (cleared), AND the opponent
/// captures neutrals during the wait — approximating apollo's greedy
/// expansion. The opponent grabs roughly one neutral per `OPP_CAPTURE_PERIOD`
/// turns, picking targets by `production / dist_to_nearest_opp_planet`.
fn future_state(state: &GameState, turns: i64) -> GameState {
    const OPP_CAPTURE_PERIOD: i64 = 5;
    let me = state.player;
    let enemy = state.enemy_id();
    let mut s = state.clone();
    s.step += turns;
    for p in &mut s.planets {
        if p.owner != -1 {
            p.ships += p.production * turns;
        }
    }
    s.fleets.clear();

    let n_captures = (turns / OPP_CAPTURE_PERIOD) as usize;
    if n_captures > 0 {
        // Rank neutrals by apollo-ish heuristic: high production, close to
        // an existing opp planet.
        let mut targets: Vec<(f64, i64, i64)> = Vec::new(); // (score, pid, cost)
        for p in &s.planets {
            if p.owner != -1 {
                continue;
            }
            let nearest_opp_dist = s
                .planets
                .iter()
                .filter(|q| q.owner == enemy && q.ships > 0)
                .map(|q| ((q.x - p.x).powi(2) + (q.y - p.y).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            if !nearest_opp_dist.is_finite() {
                continue;
            }
            let score = (p.production as f64) / (nearest_opp_dist + 1.0);
            targets.push((score, p.id, p.ships + 1));
        }
        targets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (_, pid, cost) in targets.into_iter().take(n_captures) {
            // Find an opp planet with enough ships to fund this capture.
            let mut funded = false;
            for q in s.planets.iter_mut() {
                if q.owner == enemy && q.ships >= cost {
                    q.ships -= cost;
                    funded = true;
                    break;
                }
            }
            if funded {
                if let Some(p) = s.planets.iter_mut().find(|p| p.id == pid) {
                    p.owner = enemy;
                    p.ships = 1;
                }
            }
        }
    }
    let _ = me;
    s
}

fn owned_at_end(
    state: &GameState,
    tgt: &Planet,
    inflight: &[Arrival],
    me: i32,
    arrival_dt: i64,
    send: i64,
) -> bool {
    let mut arrs = inflight.to_vec();
    arrs.push(Arrival {
        dt: arrival_dt,
        owner: me,
        ships: send,
    });
    let horizon = arrs.iter().map(|a| a.dt).max().unwrap_or(arrival_dt);
    let tl = simulate_planet(tgt, &arrs, horizon, state);
    tl.last().map(|x| x.1 == me).unwrap_or(false)
}

/// Smallest ship count from `src` that captures `tgt` under `inflight`.
fn min_capture(
    state: &GameState,
    src: &Planet,
    tgt: &Planet,
    inflight: &[Arrival],
    me: i32,
    max_send: i64,
) -> Option<(i64, f64, i64)> {
    if max_send < 1 {
        return None;
    }
    let pr_max = dir_to_hit(src, tgt, max_send, state, 0)?;
    if !owned_at_end(state, tgt, inflight, me, pr_max.time, max_send) {
        return None;
    }
    let mut best_n = max_send;
    let mut best_ang = pr_max.angle;
    let mut best_dt = pr_max.time;
    let mut lo = 1i64;
    let mut hi = max_send - 1;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        if let Some(pr) = dir_to_hit(src, tgt, mid, state, 0) {
            if owned_at_end(state, tgt, inflight, me, pr.time, mid) {
                best_n = mid;
                best_ang = pr.angle;
                best_dt = pr.time;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        } else {
            lo = mid + 1;
        }
    }
    Some((best_n, best_ang, best_dt))
}

/// Smallest ships from `src` whose `dir_to_hit` arrival is ≤ `target_dt`.
/// (Arrival time decreases monotonically with ship count up to `max_ships`.)
fn min_ships_for_arrival(
    state: &GameState,
    src: &Planet,
    tgt: &Planet,
    target_dt: i64,
    max_ships: i64,
) -> Option<i64> {
    if max_ships < 1 {
        return None;
    }
    let mut lo = 1i64;
    let mut hi = max_ships;
    let mut found = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        if let Some(pr) = dir_to_hit(src, tgt, mid, state, 0) {
            if pr.time <= target_dt {
                found = Some(mid);
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        } else {
            lo = mid + 1;
        }
    }
    found
}

/// Scenario A score: I commit `ships` from `src` to `tgt` (arriving at
/// `arrival_dt`). For each enemy planet, model apollo-like response: send
/// the **minimum** ships that arrive on my arrival tick — enough to deny
/// my capture without committing the full garrison. My phantom launches
/// at tick 2 with full garrison from every other owned planet.
///
/// Returns `(my_total, enemy_total)` at `end_t`.
fn scenario_a_totals(
    state: &GameState,
    src: &Planet,
    tgt: &Planet,
    ships: i64,
    arrival_dt: i64,
    inflight: &[Arrival],
    me: i32,
    enemy: i32,
    end_t: i64,
) -> (i64, i64) {
    let mut arrs: Vec<Arrival> = inflight.to_vec();
    arrs.push(Arrival {
        dt: arrival_dt,
        owner: me,
        ships,
    });
    // Track each enemy planet's actual snipe cost so we can subtract it
    // from their planet's residual when totalling at end_t.
    let mut enemy_snipe_cost: Vec<(i64, i64)> = Vec::new(); // (planet_id, ships_spent)
    for ep in &state.planets {
        if ep.owner != enemy || ep.ships <= 0 || ep.id == tgt.id {
            continue;
        }
        // Minimum ships from this enemy planet that arrive by my arrival
        // tick. If they can't reach in time, they don't snipe.
        if let Some(n) = min_ships_for_arrival(state, ep, tgt, arrival_dt, ep.ships) {
            arrs.push(Arrival {
                dt: arrival_dt,
                owner: enemy,
                ships: n,
            });
            enemy_snipe_cost.push((ep.id, n));
        }
    }
    for mp in &state.planets {
        if mp.owner != me || mp.ships <= 0 || mp.id == tgt.id || mp.id == src.id {
            continue;
        }
        if let Some(pr) = dir_to_hit(mp, tgt, mp.ships, state, 2) {
            arrs.push(Arrival {
                dt: 2 + pr.time,
                owner: me,
                ships: mp.ships,
            });
        }
    }

    let tl = simulate_planet(tgt, &arrs, end_t, state);
    let (tgt_owner, tgt_ships) = tl
        .last()
        .map(|&(_, o, s)| (o, s))
        .unwrap_or((tgt.owner, tgt.ships));

    let mut a_my: i64 = 0;
    let mut a_en: i64 = 0;
    if tgt_owner == me {
        a_my += tgt_ships;
    } else if tgt_owner == enemy {
        a_en += tgt_ships;
    }
    for p in &state.planets {
        if p.id == tgt.id {
            continue;
        }
        if p.owner == me {
            if p.id == src.id {
                a_my += (p.ships - ships).max(0) + p.production * end_t;
            } else {
                a_my += p.production * (end_t - 2).max(0);
            }
        } else if p.owner == enemy {
            // Enemy only "lost" the minimum snipe cost from this planet,
            // not its full garrison. Residual + production accrual.
            let snipe = enemy_snipe_cost
                .iter()
                .find(|(pid, _)| *pid == p.id)
                .map(|(_, n)| *n)
                .unwrap_or(0);
            a_en += (p.ships - snipe).max(0) + p.production * end_t;
        }
    }
    (a_my, a_en)
}

/// Scenario B totals at `end_t`: no action by me, no phantom launches —
/// just production accruing on every owned planet. This is the FIXED
/// baseline that every delay's A is measured against.
fn scenario_b_totals(state: &GameState, me: i32, enemy: i32, end_t: i64) -> (i64, i64) {
    let mut b_my: i64 = 0;
    let mut b_en: i64 = 0;
    for p in &state.planets {
        let s = p.ships + p.production * end_t;
        if p.owner == me {
            b_my += s;
        } else if p.owner == enemy {
            b_en += s;
        }
    }
    (b_my, b_en)
}

/// For a given (src, tgt) pair, evaluate the best action across all
/// `LOOKAHEAD_DELAYS`. Returns the candidate IF the best delay is 0
/// (otherwise waiting beats acting now and we should NOT queue an action).
fn evaluate_pair(
    state: &GameState,
    ledger: &Ledger,
    src: &Planet,
    tgt: &Planet,
    me: i32,
    enemy: i32,
) -> Option<ScoredCandidate> {
    let mut best_score: i64 = i64::MIN;
    let mut best_delay: i64 = -1;
    let mut best_now: Option<(f64, i64, i64)> = None; // (angle, ships, arrival_dt) at delay 0

    // Fixed baseline B at SCORE_END_TICK — no actions, no combat. Same for
    // every delay so waiting's cost (opp making progress in scenario A's
    // future-state) shows up as a worse A-score, not a quieter B.
    let (b_my, b_en) = scenario_b_totals(state, me, enemy, SCORE_END_TICK);

    for &delay in LOOKAHEAD_DELAYS {
        let s_eval = if delay == 0 {
            state.clone()
        } else {
            future_state(state, delay)
        };
        let src_eval = match s_eval.planet_by_id(src.id) {
            Some(p) => p.clone(),
            None => continue,
        };
        let tgt_eval = match s_eval.planet_by_id(tgt.id) {
            Some(p) => p.clone(),
            None => continue,
        };
        let surp = if delay == 0 {
            ledger.surplus_at(src.id)
        } else {
            src_eval.ships
        };
        if surp < 1 {
            continue;
        }
        let inflight = if delay == 0 {
            ledger.all_arrivals(tgt.id)
        } else {
            Vec::new()
        };

        let (ships, angle, dt) =
            match min_capture(&s_eval, &src_eval, &tgt_eval, &inflight, me, surp) {
                Some(x) => x,
                None => continue,
            };
        // The action arrives at absolute tick `delay + dt`. End_t for the
        // simulation is SCORE_END_TICK − delay (in s_eval's local frame).
        let local_end_t = (SCORE_END_TICK - delay).max(1);
        let (a_my, a_en) = scenario_a_totals(
            &s_eval, &src_eval, &tgt_eval, ships, dt, &inflight, me, enemy, local_end_t,
        );
        // For delay > 0, s_eval already advanced `delay` ticks of opp/my
        // production. To make A and B comparable, add the production that
        // happened DURING the wait to s_eval's a_my/a_en.
        let (a_my_total, a_en_total) = if delay == 0 {
            (a_my, a_en)
        } else {
            // s_eval's planets already had +delay*prod baked in. The a_my
            // counts "ships at end_t starting from s_eval" which is the
            // future-state ships + production-from-delay-to-end_t. So total
            // ships at absolute SCORE_END_TICK = a_my (s_eval already
            // included the +delay production). Same for a_en.
            (a_my, a_en)
        };

        let score = (a_my_total - a_en_total) - (b_my - b_en);
        if score > best_score {
            best_score = score;
            best_delay = delay;
            if delay == 0 {
                best_now = Some((angle, ships, dt));
            }
        }
    }

    let debug = std::env::var("OW4_DEBUG").ok().as_deref() == Some("1");
    if debug && state.step == 0 && best_score > i64::MIN {
        // Find nearest enemy planet to tgt for context
        let nearest_e_dist = state
            .planets
            .iter()
            .filter(|p| p.owner == enemy && p.ships > 0)
            .map(|p| ((p.x - tgt.x).powi(2) + (p.y - tgt.y).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        let my_dist = ((src.x - tgt.x).powi(2) + (src.y - tgt.y).powi(2)).sqrt();
        eprintln!(
            "    pair src={} tgt={} (g={} prod={} my_dist={:.1} enemy_dist={:.1}) best_score={} best_delay={}",
            src.id, tgt.id, tgt.ships, tgt.production, my_dist, nearest_e_dist,
            best_score, best_delay
        );
    }
    if best_delay == 0 && best_score > 0 {
        let (angle, ships, arrival_dt) = best_now?;
        Some(ScoredCandidate {
            from_id: src.id,
            target_id: tgt.id,
            angle,
            ships,
            score: best_score as f64,
            chosen_delay: 0,
            arrival_dt,
        })
    } else {
        None
    }
}

pub fn build_candidates(state: &GameState, ledger: &Ledger) -> Vec<ScoredCandidate> {
    let me = state.player;
    let enemy = state.enemy_id();
    let mut out: Vec<ScoredCandidate> = Vec::new();
    for tgt in &state.planets {
        let (end_o, _) = ledger.projected_end(tgt.id);
        if end_o == me {
            continue;
        }
        for src in &state.planets {
            if src.owner != me || src.id == tgt.id {
                continue;
            }
            if ledger.surplus_at(src.id) < 1 {
                continue;
            }
            if let Some(cand) = evaluate_pair(state, ledger, src, tgt, me, enemy) {
                out.push(cand);
            }
        }
    }
    // Sort by score desc; tiebreak by arrival_dt asc (faster delivery first).
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap()
            .then_with(|| a.arrival_dt.cmp(&b.arrival_dt))
    });
    out
}

pub fn apply_attacks(state: &GameState, ledger: &mut Ledger, out: &mut Vec<Action>) {
    let debug = std::env::var("OW4_DEBUG").ok().as_deref() == Some("1");
    let cands = build_candidates(state, ledger);
    if debug && state.step <= 1 {
        eprintln!("  [all_cands at step={}]", state.step);
        for c in &cands {
            eprintln!(
                "    tgt={} ships={} dt={} score={:.0}",
                c.target_id, c.ships, c.arrival_dt, c.score
            );
        }
    }
    let mut committed: HashSet<i64> = HashSet::new();
    for c in cands {
        if c.score <= 0.0 {
            break;
        }
        if committed.contains(&c.target_id) {
            continue;
        }
        if ledger.surplus_at(c.from_id) < c.ships {
            continue;
        }
        if debug {
            if let Some(tgt) = state.planet_by_id(c.target_id) {
                let enemy = state.enemy_id();
                let nearest_e = state
                    .planets
                    .iter()
                    .filter(|p| p.owner == enemy && p.ships > 0)
                    .map(|p| ((p.x - tgt.x).powi(2) + (p.y - tgt.y).powi(2)).sqrt())
                    .fold(f64::INFINITY, f64::min);
                let src = state.planet_by_id(c.from_id).unwrap();
                let my_d = ((src.x - tgt.x).powi(2) + (src.y - tgt.y).powi(2)).sqrt();
                eprintln!(
                    "  [t={}] tgt={} (my_d={:.1} enemy_d={:.1}) ships={} score={:.0}",
                    state.step, c.target_id, my_d, nearest_e, c.ships, c.score,
                );
            }
        }
        out.push(Action {
            from_id: c.from_id,
            angle: c.angle,
            ships: c.ships,
        });
        ledger.spend(c.from_id, c.ships);
        committed.insert(c.target_id);
    }
}
