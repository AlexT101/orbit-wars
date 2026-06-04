//! Focused single-target candidate generator for DUCT.
//!
//! For each "target" (= planet I won't own at end of `extrapolate_fleets`),
//! generate ONE candidate plan whose attack orders come from
//! `ow2_plan::plan_for_target` (the proper ow2 policy restricted to that
//! single target), then layered with "healing fleets" from rear-line
//! "support" planets back toward the planets that just sent ships.
//!
//! Algorithm per turn:
//!
//!   1. Run `value_net::extrapolate_fleets` → per-planet (owner, ships)
//!      after in-flight fleets resolve.
//!   2. Targets = planets whose extrapolated owner is not me, sorted by
//!      closest-to-my-current-planets first; cap at MAX_TARGETS.
//!   3. For each target T:
//!        a. attack_orders = `ow2_plan::plan_for_target(state, me, no_coop, T)`
//!           — ow2 commits cooperating sources via binary-search-per-source
//!           ship sizing + `simulates_safe` verification, all launches THIS
//!           turn (no offset reservations).
//!        b. Frontline = sources that appear in `attack_orders` with
//!           ships_sent > 0.
//!        c. Support pool = my planets NOT in frontline AND NOT a target,
//!           sorted by distance to nearest enemy planet DESC.
//!        d. For each support in order, compute `safe_drain` (max ships
//!           it can send and still be ours at the extrapolation horizon);
//!           distribute up to that budget among frontline planets sorted
//!           closest-to-this-support first, capped per-frontline at how
//!           many ships it sent out this turn.
//!        e. Plan = attack_orders ++ healing_orders.
//!   4. Return list of plans (one per target). Empty plans / non-capturable
//!      targets are skipped.
//!
//! Why this is different from raw `ow2_plan::plan`:
//!
//!   * Raw `plan` runs the greedy target loop and emits one multi-target
//!     plan. DUCT's candidate set then has only ~3-4 plans (1 per
//!     `SelectionStrategy` after dedup). For ~80% of turns those all
//!     collapse to the same plan, so DUCT's `my_K = 1`.
//!   * This module emits ONE plan per target → DUCT's `my_K` = number of
//!     viable targets (often 3-10) → MCTS gets to choose what to attack.
//!   * The healing fleets also give the bot "second-order" candidates
//!     that build up its rear infrastructure between attacks.

use rustc_hash::{FxHashMap as HashMap, FxHasher};
use std::cell::RefCell;
use std::hash::Hasher;

use crate::sim::Action;
use crate::{GameState, Planet};

thread_local! {
    static CACHE: RefCell<HashMap<u64, Vec<Vec<Action>>>> = RefCell::new(HashMap::default());
}

/// Clear the per-turn memoization cache. Call at the start of each turn.
pub fn reset_cache() {
    CACHE.with(|c| c.borrow_mut().clear());
}

fn state_key(state: &GameState, player: i32, is_root: bool) -> u64 {
    let mut h = FxHasher::default();
    h.write_i64(state.step);
    h.write_i32(player);
    h.write_u8(is_root as u8);
    for p in &state.planets {
        h.write_i64(p.id);
        h.write_i32(p.owner);
        h.write_i64(p.ships);
    }
    // Sort fleets so hash is invariant to launch-order permutations of
    // commutative actions.
    let mut fleet_keys: Vec<(i64, i32, i64, u64, u64)> = state.fleets.iter()
        .map(|f| (f.from_planet_id, f.owner, f.ships, f.angle.to_bits(), f.x.to_bits()))
        .collect();
    fleet_keys.sort_unstable();
    for (a, b, c, d, e) in fleet_keys {
        h.write_i64(a); h.write_i32(b); h.write_i64(c);
        h.write_u64(d); h.write_u64(e);
    }
    h.finish()
}

const MAX_TARGETS: usize = 24;       // hard cap on candidate count per call
const ROOT_KEEP: usize = 8;          // root attacks kept after race+quality filter (+ 1 no-op slot)
const NON_ROOT_KEEP: usize = 4;      // non-root attacks kept after race+quality filter (+ 1 no-op slot)
const MAX_SHIP_SPEED: f64 = 6.0;     // physics cap, matches engine

fn dist2(a: &Planet, b: &Planet) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

/// Race filter (cousin of `ow2_plan`'s `race_ok`): can I reach this target
/// before the opponent can? Uses straight-line distance / MAX_SHIP_SPEED as a
/// lower-bound time-to-hit per planet, taking the min across each side.
/// This is a *looser* version of ow2's filter (which uses `cached_time_to_hit`
/// with per-fleet swept geometry), but the inequality `min_my_LB <= min_their_LB`
/// is enough to exclude clearly-unreachable targets.
fn race_ok(state: &GameState, player: i32, target_id: i64) -> bool {
    let target = match state.planets.iter().find(|p| p.id == target_id) {
        Some(t) => t,
        None => return false,
    };
    let min_t_for = |is_mine: bool| -> f64 {
        state
            .planets
            .iter()
            .filter(|p| {
                if p.id == target_id || p.ships <= 0 {
                    return false;
                }
                if is_mine {
                    p.owner == player
                } else {
                    p.owner != player && p.owner != -1
                }
            })
            .map(|p| dist2(p, target).sqrt() / MAX_SHIP_SPEED)
            .fold(f64::INFINITY, f64::min)
    };
    let my_t = min_t_for(true);
    let their_t = min_t_for(false);
    my_t <= their_t
}

/// Quality score for non-root ranking: target.production / total_ships_committed.
/// Higher = more production per ship spent = better target.
fn quality_score(production: i64, orders: &[Action]) -> f64 {
    let total_ships: i64 = orders.iter().map(|(_, _, s, _)| *s).sum();
    if total_ships <= 0 {
        return 0.0;
    }
    production as f64 / total_ships as f64
}

pub fn focused_candidates(state: &GameState, player: i32, is_root: bool) -> Vec<Vec<Action>> {
    let __fc_t0 = std::time::Instant::now();
    crate::profiling::inc(&crate::profiling::FOCUSED_CANDIDATES_CALLS);

    // Memoization: focused_candidates is a pure function of (state, player,
    // is_root). DUCT expands many nodes that converge to identical states
    // (commutative joint actions), so the hit rate is high.
    let key = state_key(state, player, is_root);
    if let Some(cached) = CACHE.with(|c| c.borrow().get(&key).cloned()) {
        crate::profiling::add(&crate::profiling::FOCUSED_CANDIDATES_NS, __fc_t0);
        return cached;
    }

    let result = focused_candidates_uncached(state, player, is_root);
    CACHE.with(|c| c.borrow_mut().insert(key, result.clone()));
    crate::profiling::add(&crate::profiling::FOCUSED_CANDIDATES_NS, __fc_t0);
    result
}

fn focused_candidates_uncached(state: &GameState, player: i32, is_root: bool) -> Vec<Vec<Action>> {
    use crate::value_net::extrapolate_fleets;

    let __ex_t0 = std::time::Instant::now();
    let extrap: HashMap<i64, (i32, i64)> = extrapolate_fleets(state).into_iter().collect();
    crate::profiling::add(&crate::profiling::EXTRAPOLATE_NS, __ex_t0);
    crate::profiling::inc(&crate::profiling::EXTRAPOLATE_CALLS);

    // Targets = planets I won't own at end of extrapolation (excluding own).
    // Sort by closest-to-me first; cap at MAX_TARGETS.
    let target_ids: Vec<i64> = {
        let mut ids: Vec<(i64, f64)> = state
            .planets
            .iter()
            .filter_map(|p| {
                let (owner, _) = extrap.get(&p.id).copied().unwrap_or((p.owner, p.ships));
                if owner == player {
                    None
                } else {
                    let nearest_mine = state
                        .planets
                        .iter()
                        .filter(|q| q.owner == player)
                        .map(|q| dist2(p, q))
                        .fold(f64::INFINITY, f64::min);
                    Some((p.id, nearest_mine))
                }
            })
            .collect();
        ids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ids.into_iter().take(MAX_TARGETS).map(|(id, _)| id).collect()
    };

    if target_ids.is_empty() {
        return vec![Vec::new()];
    }

    let no_coop = std::env::var("OW_NO_COOP").is_ok();

    // Optional verbose debug on the very first focused_candidates call of
    // the game (step 0, root). Logged to stderr so it's visible without
    // affecting bot stdout. Enabled by OW_DEBUG=1.
    let debug_first = is_root
        && state.step == 0
        && std::env::var("OW_DEBUG").is_ok();
    let object_type = |p: &Planet| -> &'static str {
        if p.is_comet { "comet" } else if p.is_orbiting { "orbit" } else { "static" }
    };
    let owner_str = |o: i32| -> String {
        if o == -1 { "neutral".into() } else if o == player { format!("me(p{})", o) } else { format!("enemy(p{})", o) }
    };

    if debug_first {
        eprintln!("[focused] === step=0 player={} initial candidates from {} targets ===", player, target_ids.len());
    }

    // Per-target: try ow2_plan; unaffordable targets contribute a SINGLE
    // shared no-op candidate (so MCTS can "pick no-op" but the choice
    // doesn't get 20x prior mass starving the attack branches of depth).
    //
    // Dedup over ALL plans (empties dedup to one no-op; identical attacks
    // from distinct targets collapse to one slot).
    // Build ow2's heavy precompute (arrivals/safe/enemy_safe/race_ok) ONCE
    // for this (state, player) pair, then run only the cheap per-target
    // greedy body for each candidate.
    let ctx = crate::ow2_plan::PlanContext::build(state, player, no_coop);

    // Affordability prefilter: max total ships we can send (sum of safe-drain
    // across all my planets). Any target requiring more is unaffordable and
    // would otherwise waste the full t=1..60 plan_for_time loop returning
    // empty. Race filter inside plan_target_with_ctx catches a separate
    // case; this one catches "not enough ships to overcome target garrison".
    let total_capacity: i64 = ctx.safe.values().sum();
    let unaffordable = |tgt: i64| -> bool {
        let target = match state.planets.iter().find(|p| p.id == tgt) { Some(t) => t, None => return true };
        if target.owner == player { return true; }
        // Neutral targets don't grow (no production accrues to owner=-1).
        // Enemy targets grow at target.production per tick — use worst-case
        // t=60 horizon as a lower bound on ships needed.
        let need = if target.owner == -1 {
            target.ships + 1
        } else {
            target.ships + 60 * target.production + 1
        };
        need > total_capacity
    };

    let mut tagged: Vec<(Vec<Action>, i64)> = Vec::with_capacity(target_ids.len());
    let mut have_noop = false;
    for &tgt in &target_ids {
        let __pft_t0 = std::time::Instant::now();
        let target = state.planets.iter().find(|p| p.id == tgt).unwrap();
        let attack_orders: Vec<Action> = if unaffordable(tgt) {
            // Affordability prefilter said we can't capture this turn.
            // For high-production targets, emit an APPROACH plan instead
            // of folding to no-op: send safe-drain ships from the closest
            // source toward T so MCTS can evaluate "commit toward big
            // target this turn, complete capture next turn." Low-prod
            // unaffordable targets fall through to the no-op slot since
            // they're not worth the deposit.
            if target.production >= 3 {
                crate::ow2_plan::approach_plan_for_target(&ctx, tgt).unwrap_or_default()
            } else {
                Vec::new()
            }
        } else {
            crate::ow2_plan::plan_target_with_ctx(&ctx, tgt)
        };
        crate::profiling::add(&crate::profiling::PLAN_FOR_TARGET_NS, __pft_t0);
        crate::profiling::inc(&crate::profiling::PLAN_FOR_TARGET_CALLS);
        let total_ships: i64 = attack_orders.iter().map(|(_, _, s, _)| *s).sum();
        if debug_first {
            let status = if attack_orders.is_empty() { "EMPTY → no-op".into() } else { format!("{} orders, {} ships", attack_orders.len(), total_ships) };
            eprintln!(
                "  target id={} x={:.1} y={:.1} prod={} owner={} type={} → {}",
                tgt, target.x, target.y, target.production,
                owner_str(target.owner), object_type(target), status
            );
        }
        if attack_orders.is_empty() {
            if have_noop {
                if debug_first { eprintln!("    (no-op already in pool, folded)"); }
                continue;
            }
            have_noop = true;
            tagged.push((attack_orders, tgt));
            continue;
        }
        if tagged.iter().any(|(p, _)| !p.is_empty() && actions_equal(p, &attack_orders)) {
            if debug_first { eprintln!("    (duplicate attack, skipped)"); }
            continue;
        }
        tagged.push((attack_orders, tgt));
    }

    if debug_first {
        let n_attack = tagged.iter().filter(|(p, _)| !p.is_empty()).count();
        let n_noop = tagged.iter().filter(|(p, _)| p.is_empty()).count();
        eprintln!("[focused] after generation: {} candidates ({} attack + {} no-op)", tagged.len(), n_attack, n_noop);
    }

    // ow-style "closer to me" race filter — applied to ATTACK plans only
    // (no-ops have no target reachability to verify). Drops attacks where
    // the opponent reaches the target before any of my sources.
    if debug_first {
        let pre = tagged.len();
        tagged.retain(|(plan, tgt_id)| {
            if plan.is_empty() { return true; }
            let ok = race_ok(state, player, *tgt_id);
            if !ok {
                let t = state.planets.iter().find(|p| p.id == *tgt_id).unwrap();
                eprintln!("  race-DROP attack id={} x={:.1} y={:.1} prod={} (opponent reaches first)", tgt_id, t.x, t.y, t.production);
            }
            ok
        });
        eprintln!("[focused] after race filter: {} -> {}", pre, tagged.len());
    } else {
        tagged.retain(|(plan, tgt_id)| plan.is_empty() || race_ok(state, player, *tgt_id));
    }

    // Sort: attack plans first by production/ships DESC, then no-op slots.
    tagged.sort_by(|a, b| {
        let prod_for = |id: i64| -> i64 {
            state.planets.iter().find(|p| p.id == id).map(|p| p.production).unwrap_or(0)
        };
        match (a.0.is_empty(), b.0.is_empty()) {
            (false, false) => quality_score(prod_for(b.1), &b.0)
                .partial_cmp(&quality_score(prod_for(a.1), &a.0))
                .unwrap_or(std::cmp::Ordering::Equal),
            (false, true) => std::cmp::Ordering::Less,   // attack before no-op
            (true, false) => std::cmp::Ordering::Greater,
            (true, true) => std::cmp::Ordering::Equal,
        }
    });

    // Truncate ATTACKS only (the no-op slot is preserved). Since attacks
    // are sorted before no-ops, take_while gives us their count.
    let keep_attacks = if is_root { ROOT_KEEP } else { NON_ROOT_KEEP };
    let n_attack = tagged.iter().take_while(|(p, _)| !p.is_empty()).count();
    if n_attack > keep_attacks {
        if debug_first {
            for (i, (orders, tgt_id)) in tagged.iter().enumerate().take(n_attack).skip(keep_attacks) {
                let t = state.planets.iter().find(|p| p.id == *tgt_id).unwrap();
                let ships: i64 = orders.iter().map(|(_, _, s, _)| *s).sum();
                eprintln!("  rank-DROP attack idx={} id={} x={:.1} y={:.1} prod={} ships={} q={:.4}",
                    i, tgt_id, t.x, t.y, t.production, ships, quality_score(t.production, orders));
            }
        }
        tagged.drain(keep_attacks..n_attack);
    }

    if debug_first {
        eprintln!("[focused] final {} candidates (root={} keep={}):", tagged.len(), is_root, keep_attacks);
        for (i, (orders, tgt_id)) in tagged.iter().enumerate() {
            let t = state.planets.iter().find(|p| p.id == *tgt_id).unwrap();
            let ships: i64 = orders.iter().map(|(_, _, s, _)| *s).sum();
            let kind = if orders.is_empty() { "NO-OP " } else { "ATTACK" };
            eprintln!("  rank {:>2}: {} id={} x={:.1} y={:.1} fleet={} prod={} owner={} type={}",
                i, kind, tgt_id, t.x, t.y, ships, t.production, owner_str(t.owner), object_type(t));
        }
    }

    let mut plans: Vec<Vec<Action>> = tagged.into_iter().map(|(p, _)| p).collect();
    // The sort placed any no-op at the END of the list, which would give it
    // ~0.2% PUCT prior (rank_prior = 0.5^N). We actually want the no-op at
    // rank 1 — ~25% prior — so MCTS gives "wait/save up" real visit weight.
    // Strip any existing no-ops, then re-insert exactly one at rank 1.
    plans.retain(|p| !p.is_empty());
    if plans.is_empty() {
        return vec![Vec::new()];
    }
    plans.insert(plans.len().min(1), Vec::new());
    plans
}

fn actions_equal(a: &[Action], b: &[Action]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    // Action = (i64, f64, i64, i32). Direct PartialEq compares all fields
    // including the f64 angle, which is bit-equal for plans produced by
    // the same ow2_plan call from the same state.
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}
