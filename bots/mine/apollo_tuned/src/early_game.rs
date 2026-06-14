//! Early-game expansion pre-pass: brute-force capture scheduling for the
//! first [`EARLY_GAME_END`] steps.
//!
//! The greedy pipeline in [`crate::strategy`] commits the best-scoring
//! capture per iteration, independently. In the opening that throws away
//! chained expansions — sending A→B and then B→C two turns later can beat
//! A→D even when D alone outscores B, because B unlocks C. This module
//! instead searches over *sets* of capture events.
//!
//! This is a *pre-pass*, not a separate regime: [`plan_opening`] returns
//! capture events that `run_strategy` commits into its `PlanState` before the
//! greedy iterations run. Offset-0 events become fleet moves; future-offset
//! events are reservations (spent ships the greedy loop cannot poach, targets
//! it sees as already won), re-derived next turn exactly like greedy's own
//! delayed orders. Greedy combat, defense, and reinforcement still run on
//! whatever the opening leaves over, and `search_candidates` adds a
//! no-opening candidate so the rollout minimax can reject a bad opening
//! wholesale.
//!
//! The phase gate ([`EARLY_GAME_END`]) is a hard stop on *running* the DFS, not
//! a valuation cliff: the objective extends to the full horizon and greedy runs
//! on top, so there is no scoring discontinuity. There is, however, a behavioral
//! limitation at the boundary — only offset-0 hops are emitted each turn, and
//! once the gate closes the DFS no longer re-derives the chain, so a chain whose
//! later hops would launch at/after [`EARLY_GAME_END`] is handed to the greedy
//! planner (which does not model chains) for its tail. In practice first hops
//! launch early and per-turn re-planning keeps this rare, but a chain set up
//! late in the window can lose its continuation.
//!
//! Objective — the same quantity the greedy scorer prices
//! (`timeline_delta_score`), restricted to uncontested neutral captures where
//! it has a closed form: Σ `production·(window − arrival) − garrison` is our
//! ship-count delta at the horizon (a capture trades `garrison` of our ships
//! against the neutral garrison, then produces until the horizon). Plans with
//! negative total value lose to the empty plan, so captures the greedy would
//! reject as score-negative are rejected here too — no valuation cliff at the
//! phase boundary. Ties prefer Σ production (value beyond the horizon), then
//! fewer events.
//!
//! Enemy *pressure* (what opponents could send) is deliberately not modeled
//! here — per-turn re-planning plus the rollout's no-opening alternative
//! cover interference. Observed reality is modeled: in-flight fleets already
//! in the baseline timeline rule out neutrals that flip to an enemy inside
//! the window and size garrisons from the baseline's worst case.
//!
//! Model:
//!   * Targets are neutral planets only. Neutral garrisons don't produce (the
//!     engine only adds production to owned planets), so a capture's cost is
//!     exact: `garrison + 1` ships arriving on any turn within the window.
//!   * A plan is a set of events `(src, target, ships, launch_offset)`. Ship
//!     amounts per edge come in three variants: *minimal* (`G+1`), *ferry*
//!     (send everything available — funds deeper chains and rides the
//!     log-shaped speed curve), and *min+child* (`G_t+1 + G_c+1` for a nearby
//!     remaining neutral `c` — funds exactly one downstream hop with zero
//!     waste).
//!   * Launch timing is searched, not assumed: per `(src, target, ships)` we
//!     scan every admissible offset and keep only strictly-improving arrivals
//!     (waiting can shorten travel via orbital geometry, cleared blockers, or
//!     a bigger-therefore-faster fleet). The ferry variant runs on its own
//!     frontier, since the bigger fleet's arrivals order differently.
//!
//! Search = DFS over canonically ordered event sequences (non-decreasing
//! `(offset, src, target)` keys, so each event *set* is enumerated once; for
//! a fixed source, availability grows with offset while debits accumulate,
//! making offset-ascending the most permissive feasibility order) with
//! branch-and-bound (optimistic bound: each remaining candidate's value at
//! its earliest probe arrival, clamped at zero; candidates unreachable even
//! by an achievable-fleet probe were dropped up front) and a hard node
//! budget.

use rustc_hash::FxHashMap as HashMap;

use crate::constants::{
    EARLY_GAME_END, EARLY_GAME_FERRY_PROBES, EARLY_GAME_MAX_CANDIDATES, EARLY_GAME_MAX_CHILD_FUND,
    EARLY_GAME_NODE_BUDGET, EARLY_GAME_PROBE_SHIPS, EARLY_GAME_VALUE_PICKS,
};
use crate::engine::ArrivalEvent;
use crate::helpers::dist;
use crate::strategy::{
    available_at_timeline_for_owner, baseline_available_at_for_owner, HellburnerModel,
};
use crate::world::WorldState;

/// Travel-only arrival turn of a `ships`-fleet launched from `src` toward
/// `target` at launch `offset`, or `None` when the shot is blocked or lands
/// past `window`. Centralizes the `(offset + turns).max(1)` / `≤ window`
/// invariant shared by the reachability probe, the geometry row cache, and the
/// ferry frontier.
fn arrival_within(
    model: &HellburnerModel,
    src: i64,
    target: i64,
    ships: i64,
    offset: i64,
    window: i64,
) -> Option<i64> {
    model
        .plan_shot(src, target, ships, offset)
        .map(|(_, turns, _, _, _)| (offset + turns).max(1))
        .filter(|&arrival| arrival <= window)
}

/// Lower `best` by the earliest in-window arrival of a `ships`-fleet from `src`
/// to `target` launched at any offset in `[min_launch, window)`. Used by the
/// reachability fixpoint; the `offset ≥ best` short-circuit holds because an
/// arrival is always `≥` its launch offset.
fn relax_arrival(
    model: &HellburnerModel,
    src: i64,
    target: i64,
    ships: i64,
    min_launch: i64,
    window: i64,
    best: i64,
) -> i64 {
    let mut best = best;
    for o in min_launch.max(0)..window {
        if o >= best {
            break;
        }
        if let Some(arrival) = arrival_within(model, src, target, ships, o, window) {
            best = best.min(arrival);
        }
    }
    best
}

/// One planned capture in the vocabulary the strategy layer commits: launch
/// `ships` from `src` at `offset` turns from now, arriving at planet `target`
/// at `arrival` turns from now.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OpeningEvent {
    pub src: i64,
    pub target: i64,
    pub ships: i64,
    pub offset: i64,
    pub arrival: i64,
}

/// Internal search event — like [`OpeningEvent`] but indexing into the
/// candidate set. `src_idx` is the launching source's position in the DFS
/// node's `srcs` vector at the moment the option was generated (stable across
/// the node's sibling options, since `srcs` is restored between them), so the
/// commit step debits the right source without an id scan.
#[derive(Clone, Copy, Debug)]
struct Event {
    src: i64,
    src_idx: usize,
    target_idx: usize,
    ships: i64,
    offset: i64,
    arrival: i64,
}

/// Candidate stats before reachability probing and final selection.
#[derive(Clone, Copy)]
struct RawCandidate {
    id: i64,
    garrison: i64,
    production: i64,
}

struct Candidate {
    id: i64,
    garrison: i64,
    production: i64,
    /// Optimistic value of capturing this candidate:
    /// `production·(window − earliest probe arrival) − garrison`, clamped at
    /// zero. Feeds the branch-and-bound completion bound.
    value_bound: i64,
    /// Nearby candidate indices (highest value bound first, capped), used by
    /// the min+child funding variant.
    children: Vec<usize>,
}

/// Opening pre-pass entry point. Returns the best capture schedule for the
/// caller to commit ahead of its greedy run; empty when the phase is over,
/// when no positive-value capture plan exists, or inside rollout forward
/// simulation (reply policies must stay cheap — a DFS per simulated reply
/// would blow the turn budget).
pub(crate) fn plan_opening(model: &HellburnerModel) -> Vec<OpeningEvent> {
    let world = model.state;
    if world.rollout_internal || world.cache.current_turn >= EARLY_GAME_END {
        return Vec::new();
    }
    let Some(s) = run_search(world, model) else {
        return Vec::new();
    };
    s.best_events
        .iter()
        .map(|e| OpeningEvent {
            src: e.src,
            target: s.candidates[e.target_idx].id,
            ships: e.ships,
            offset: e.offset,
            arrival: e.arrival,
        })
        .collect()
}

/// Ships-at-horizon value of capturing a `production`/`garrison` neutral at
/// `arrival`: the closed form of the greedy `timeline_delta_score` for an
/// uncontested capture (we trade `garrison` of our ships against the neutral
/// garrison, then produce until the horizon).
fn capture_value(production: i64, garrison: i64, window: i64, arrival: i64) -> i64 {
    production * (window - arrival) - garrison
}

/// Bench hook: run the opening search and report
/// `(nodes used, best-plan events, best-plan value)`. `None` when the phase
/// gate is closed or there is nothing to search.
#[cfg(test)]
pub(crate) fn opening_search_stats(model: &HellburnerModel) -> Option<(u64, usize, i64)> {
    let world = model.state;
    if world.rollout_internal || world.cache.current_turn >= EARLY_GAME_END {
        return None;
    }
    run_search(world, model).map(|s| (s.nodes, s.best_events.len(), s.best_value))
}

/// Per-source launch-availability state threaded through the DFS. `raw[o]`
/// is the source's base availability at launch offset `o` minus every ship
/// already committed from it on the current branch — kept *unclamped* so a
/// commit and its undo are exact flat additions. Consumers only ever compare
/// availability against amounts ≥ 1, where negative values behave
/// identically to a zero clamp.
struct SrcState {
    id: i64,
    /// Earliest admissible launch offset (turn after a pending or in-branch
    /// capture lands; 0 for planets owned now).
    min_launch: i64,
    raw: Vec<i64>,
}

/// Base availability vector (launch offsets `0..window`) for `id`: the
/// forward-min launchable ships per offset, read from the baseline timeline —
/// or, for a candidate captured on the current branch, from one planet sim
/// folding our capture arrival into the baseline (paid once per commit
/// instead of once per node).
fn avail_vector(
    world: &WorldState,
    id: i64,
    capture: Option<ArrivalEvent>,
    window: i64,
) -> Vec<i64> {
    let player = world.player;
    if let Some(ev) = capture {
        let tl = world.projected_timeline(id, world.timeline_cache.horizon, &[ev], &[]);
        return (0..window)
            .map(|o| available_at_timeline_for_owner(&tl, player, player, o))
            .collect();
    }
    // Plan-free baseline availability (forward-min from the cached timeline, or
    // linear growth when no trajectory is cached) — shared with the greedy
    // planner's `ships_available_at`.
    (0..window)
        .map(|o| baseline_available_at_for_owner(world, id, player, o))
        .collect()
}

fn run_search(world: &WorldState, model: &HellburnerModel) -> Option<Search> {
    // Objective window: ship delta at the end of the timeline horizon (the
    // phase gate already passed in `plan_opening`).
    let window = world.timeline_cache.horizon.max(1);
    let player = world.player;

    // Launch sources: planets owned now, plus planets the baseline timeline
    // already shows us capturing (fleets emitted on previous turns, still in
    // flight). Pending captures must not be re-targeted, but they can relay
    // chains once they land — their earliest launch is the turn after the
    // baseline first shows us owning them.
    let mut sources: Vec<(i64, i64)> = Vec::new(); // (id, earliest launch offset)
    let mut raw: Vec<RawCandidate> = Vec::new();
    for p in &world.planets {
        if !model.non_comet_ids.contains(&p.id) {
            continue;
        }
        if p.owner == player {
            sources.push((p.id, 0));
            continue;
        }
        let pending_owned_at = world.timeline_cache.baseline(p.id).and_then(|b| {
            if b.owner_at[window as usize] == player {
                (0..=window).find(|&t| b.owner_at[t as usize] == player)
            } else {
                None
            }
        });
        if let Some(t) = pending_owned_at {
            sources.push((p.id, t + 1));
            continue;
        }
        if p.owner != -1 {
            continue;
        }
        // Baseline-aware candidate stats. In-flight fleets are observed
        // reality, not enemy pressure: skip neutrals the baseline already
        // shows flipping to an enemy inside the window (racing a landed
        // capture is the mid-game planner's job), and size the garrison from
        // the baseline's worst case (sub-critical arrivals can change a
        // neutral's ships without flipping it).
        let mut garrison = p.ships;
        let mut contested = false;
        if let Some(b) = world.timeline_cache.baseline(p.id) {
            for t in 0..=window as usize {
                if b.owner_at[t] != -1 {
                    contested = true;
                    break;
                }
                garrison = garrison.max(b.ships_at[t]);
            }
        }
        if contested {
            continue;
        }
        // Skip neutrals an enemy fleet is already inbound to within the window.
        // The objective assumes pure post-capture production, but a baseline
        // enemy arrival — which hits the *neutral* in the baseline, yet would
        // hit *us* once our capture lands — is not in that closed form. Racing
        // a landed capture is the mid-game planner's job; modelling it here
        // would make the score optimistic for exactly the targets most likely
        // to be retaken.
        let enemy_inbound = world
            .timeline_cache
            .arrivals(p.id)
            .iter()
            .any(|a| a.turns <= window && a.owner != -1 && a.owner != player);
        if enemy_inbound {
            continue;
        }
        raw.push(RawCandidate {
            id: p.id,
            garrison,
            production: p.production,
        });
    }
    if sources.is_empty() || raw.is_empty() {
        return None;
    }

    // Distance pre-filter, purely to bound probe cost: keep twice the final
    // cap so the arrival/value ranking below still has slack to differ from
    // raw distance order.
    let mut ranked: Vec<(f64, usize)> = raw
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let p = world.planet(c.id);
            let d = sources
                .iter()
                .map(|&(sid, _)| {
                    let s = world.planet(sid);
                    dist(s.x, s.y, p.x, p.y)
                })
                .fold(f64::INFINITY, f64::min);
            (d, i)
        })
        .collect();
    ranked.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(raw[a.1].id.cmp(&raw[b.1].id))
    });
    ranked.truncate(EARLY_GAME_MAX_CANDIDATES * 2);
    let pool: Vec<RawCandidate> = ranked.iter().map(|&(_, i)| raw[i]).collect();

    // Achievable-fleet probe size: every ship we own plus everything that
    // could be produced by planets we own, are about to own, or could
    // capture over the window. Garrisons only ever remove ships from the
    // pool, so this bounds any single fleet we could field — much tighter
    // than a max-speed constant, which would classify planets as reachable
    // at speeds we can never attain — while staying optimistic, which the
    // branch-and-bound's per-candidate earliest arrivals require. Clamped
    // where the speed curve saturates (a bigger probe can't be faster); the
    // exact size is used as-is, since the monotone early-break below keeps
    // the probe cheap without cross-turn aim-cache key stabilization.
    let mut achievable: i64 = 0;
    for &(sid, _) in &sources {
        let s = world.planet(sid);
        if s.owner == player {
            achievable += s.ships;
        }
        achievable += s.production * window;
    }
    for c in &pool {
        achievable += c.production * window;
    }
    let probe_ships = achievable.clamp(1, EARLY_GAME_PROBE_SHIPS);

    // Reachability probe + earliest possible arrival per candidate, computed as
    // a fixpoint relaxation: a candidate is reachable from a base source, or as
    // a relay from another candidate that is *itself* reachable, launching no
    // earlier than that relay's own arrival. Sources seed the relaxation at
    // their launch offsets. Probing with a fixed achievable-fleet size keeps it
    // optimistic (ignores ship cost / relay-launch-the-turn-after) so the value
    // bounds derived from these arrivals stay valid upper bounds for the
    // branch-and-bound, while no longer routing through candidates that are
    // themselves unreachable (which would spend candidate slots on planets no
    // feasible plan can take). The probe geometry is L1-cached, so the repeated
    // relaxations are cheap re-reads after the first sweep.
    let n = pool.len();
    let mut earliest = vec![i64::MAX; n];
    loop {
        let mut changed = false;
        for ci in 0..n {
            let cid = pool[ci].id;
            let mut best = earliest[ci];
            for &(sid, min_launch) in &sources {
                if sid == cid {
                    continue;
                }
                best = relax_arrival(model, sid, cid, probe_ships, min_launch, window, best);
            }
            for oi in 0..n {
                if oi == ci || earliest[oi] >= window {
                    continue;
                }
                best = relax_arrival(model, pool[oi].id, cid, probe_ships, earliest[oi], window, best);
            }
            if best < earliest[ci] {
                earliest[ci] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let reachable: Vec<(i64, RawCandidate)> = pool
        .iter()
        .enumerate()
        .filter(|&(i, _)| earliest[i] <= window)
        .map(|(i, c)| (earliest[i], *c))
        .collect();
    if reachable.is_empty() {
        return None;
    }

    // Final selection: nearest by earliest probe arrival (prices rotation and
    // blockers, unlike raw distance; near candidates also serve as chain
    // relays), unioned with the highest value-bound reachable planets so a
    // fat target just outside the nearest set still gets considered.
    let value_bound = |earliest: i64, c: &RawCandidate| {
        capture_value(c.production, c.garrison, window, earliest).max(0)
    };
    let mut by_arrival = reachable.clone();
    by_arrival.sort_by_key(|&(e, c)| (e, c.id));
    let mut kept: Vec<(i64, RawCandidate)> = by_arrival
        .iter()
        .take(EARLY_GAME_MAX_CANDIDATES)
        .copied()
        .collect();
    let mut by_value = reachable;
    by_value.sort_by_key(|&(e, c)| (std::cmp::Reverse(value_bound(e, &c)), e, c.id));
    for &(e, c) in by_value.iter().take(EARLY_GAME_VALUE_PICKS) {
        if !kept.iter().any(|&(_, k)| k.id == c.id) {
            kept.push((e, c));
        }
    }

    let index: HashMap<i64, usize> = kept
        .iter()
        .enumerate()
        .map(|(i, &(_, c))| (c.id, i))
        .collect();
    let candidates: Vec<Candidate> = kept
        .iter()
        .map(|&(earliest, c)| {
            // Children ranked by what the chain is for — the value of the
            // downstream hop — not by distance.
            let mut near: Vec<(i64, f64, usize)> = model
                .outbound_edges
                .get(&c.id)
                .map(|v| {
                    v.iter()
                        .filter_map(|&(dst, d)| {
                            index
                                .get(&dst)
                                .map(|&i| (value_bound(kept[i].0, &kept[i].1), d, i))
                        })
                        .collect()
                })
                .unwrap_or_default();
            near.sort_by(|a, b| {
                b.0.cmp(&a.0)
                    .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .then(kept[a.2].1.id.cmp(&kept[b.2].1.id))
            });
            Candidate {
                id: c.id,
                garrison: c.garrison,
                production: c.production,
                value_bound: value_bound(earliest, &c),
                children: near
                    .into_iter()
                    .take(EARLY_GAME_MAX_CHILD_FUND)
                    .map(|(_, _, i)| i)
                    .collect(),
            }
        })
        .collect();

    let total_value: i64 = candidates.iter().map(|c| c.value_bound).sum();
    let mut search = Search {
        window,
        candidates,
        rows: HashMap::default(),
        nodes: 0,
        best_value: 0,
        best_prod: 0,
        best_events: Vec::new(),
        option_pool: Vec::new(),
    };
    // Per-source availability vectors, maintained incrementally across the
    // DFS (commits apply flat debits; captured targets pay one planet sim
    // when they join) instead of being re-derived at every node.
    let mut srcs: Vec<SrcState> = sources
        .iter()
        .map(|&(id, min_launch)| SrcState {
            id,
            min_launch,
            raw: avail_vector(world, id, None, window),
        })
        .collect();
    let mut events: Vec<Event> = Vec::new();
    let mut remaining = vec![true; search.candidates.len()];
    search.dfs(
        world, model, &mut srcs, &mut events, &mut remaining, 0, 0, total_value,
    );
    // `best_events` may be empty (no positive-value plan); the caller's map
    // over it yields the empty schedule naturally.
    Some(search)
}

struct Search {
    window: i64,
    candidates: Vec<Candidate>,
    /// `(src, target, ships) → arrival per launch offset` (`None` = blocked
    /// or lands past the window). Geometry only — affordability is checked
    /// per node against the maintained per-source availability vectors.
    rows: HashMap<(i64, i64, i64), Vec<Option<i64>>>,
    nodes: u64,
    /// Best plan value so far: Σ `production·(window − arrival) − garrison`.
    best_value: i64,
    /// Production tie-break of the best plan (value beyond the horizon).
    best_prod: i64,
    best_events: Vec<Event>,
    /// Reusable per-node option buffers, recycled across DFS nodes to avoid a
    /// fresh allocation per node.
    option_pool: Vec<Vec<Event>>,
}

/// Arrival row for one `(src, target, ships)` edge, lazily built and cached
/// across the whole search (free function so callers can borrow `rows`
/// without freezing the rest of the `Search` struct).
fn row_for<'r>(
    rows: &'r mut HashMap<(i64, i64, i64), Vec<Option<i64>>>,
    window: i64,
    model: &HellburnerModel,
    src: i64,
    target: i64,
    ships: i64,
) -> &'r [Option<i64>] {
    rows.entry((src, target, ships)).or_insert_with(|| {
        (0..window)
            .map(|o| arrival_within(model, src, target, ships, o, window))
            .collect()
    })
}

impl Search {
    #[allow(clippy::too_many_arguments)]
    fn dfs(
        &mut self,
        world: &WorldState,
        model: &HellburnerModel,
        srcs: &mut Vec<SrcState>,
        events: &mut Vec<Event>,
        remaining: &mut Vec<bool>,
        cur_value: i64,
        cur_prod: i64,
        remaining_value: i64,
    ) {
        self.nodes += 1;

        // Every prefix is a valid plan; record improvements as we go so the
        // node budget degrades gracefully into "best found so far". The empty
        // plan starts at value 0, so net-negative plans (captures the greedy
        // scorer would also reject) never win.
        if cur_value > self.best_value
            || (cur_value == self.best_value && cur_prod > self.best_prod)
            || (cur_value == self.best_value
                && cur_prod == self.best_prod
                && events.len() < self.best_events.len())
        {
            self.best_value = cur_value;
            self.best_prod = cur_prod;
            self.best_events = events.clone();
        }
        if self.nodes >= EARLY_GAME_NODE_BUDGET {
            return;
        }
        // Optimistic completion bound: every remaining candidate captured at
        // its earliest probe arrival. Equal-bound branches continue — they
        // can still improve the production tie-break.
        if cur_value + remaining_value < self.best_value {
            return;
        }

        let last_key = events
            .last()
            .map(|e| (e.offset, e.src, self.candidates[e.target_idx].id));

        let mut options = self.option_pool.pop().unwrap_or_default();
        options.clear();
        for si in 0..srcs.len() {
            let src = srcs[si].id;
            // Canonical floor: only offsets that can still exceed `last_key`
            // are admissible. Folding the floor into the frontier (instead of
            // generating from the raw earliest launch and filtering after)
            // skips sources with no admissible offset, and keeps the frontier
            // honest — an inadmissible earlier offset must not suppress an
            // admissible later one that reaches the same arrival.
            let floor = match last_key {
                None => 0,
                Some((lo, ls, _)) => {
                    if src >= ls {
                        lo
                    } else {
                        lo + 1
                    }
                }
            };
            let min_launch = srcs[si].min_launch.max(floor);
            if min_launch >= self.window {
                continue;
            }
            for ti in 0..self.candidates.len() {
                if !remaining[ti] {
                    continue;
                }
                self.options_for(
                    model,
                    src,
                    si,
                    min_launch,
                    &srcs[si].raw,
                    ti,
                    remaining,
                    &mut options,
                );
            }
        }

        // Canonical ordering: only extend with events whose key is strictly
        // greater than the last committed one, so every event *set* is
        // enumerated exactly once (chain launches always have a larger offset
        // than the capture that enables them, so no feasible set is lost).
        // The per-source floor above already enforces the offset component;
        // this filters the same-(offset, src) target tie it can't express.
        options.retain(|e| match last_key {
            None => true,
            Some(k) => (e.offset, e.src, self.candidates[e.target_idx].id) > k,
        });

        // Visit promising branches first so the bound bites early. Full key
        // chain keeps the order (and therefore budget-truncated results)
        // deterministic. No dedup is needed: each (src, target) pair gets one
        // `options_for` call per node, which emits unique (ships, offset)
        // keys — amounts are deduped, each frontier pushes at most one event
        // per offset, and the ferry skips amount collisions.
        options.sort_by(|a, b| {
            let ca = &self.candidates[a.target_idx];
            let cb = &self.candidates[b.target_idx];
            let ga = capture_value(ca.production, ca.garrison, self.window, a.arrival);
            let gb = capture_value(cb.production, cb.garrison, self.window, b.arrival);
            gb.cmp(&ga)
                .then(a.arrival.cmp(&b.arrival))
                .then(a.offset.cmp(&b.offset))
                .then(a.src.cmp(&b.src))
                .then(ca.id.cmp(&cb.id))
                .then(a.ships.cmp(&b.ships))
        });

        for idx in 0..options.len() {
            let ev = options[idx];
            let target_id = self.candidates[ev.target_idx].id;
            let prod = self.candidates[ev.target_idx].production;
            let garrison = self.candidates[ev.target_idx].garrison;
            let bound = self.candidates[ev.target_idx].value_bound;
            // Commit: flat-debit the launching source (`raw` is unclamped, so
            // the debit and its undo below are exact). `src_idx` was recorded
            // when the option was generated this node; `srcs` is restored to the
            // node's base state between sibling options, so it still indexes the
            // launching source. Add the captured target as a new launch source —
            // the one planet sim folding our capture arrival into its baseline.
            let si = ev.src_idx;
            debug_assert_eq!(srcs[si].id, ev.src, "src_idx no longer indexes ev.src");
            for r in &mut srcs[si].raw {
                *r -= ev.ships;
            }
            srcs.push(SrcState {
                id: target_id,
                min_launch: ev.arrival + 1,
                raw: avail_vector(
                    world,
                    target_id,
                    Some(ArrivalEvent {
                        turns: ev.arrival.max(1),
                        owner: world.player,
                        ships: ev.ships,
                    }),
                    self.window,
                ),
            });
            events.push(ev);
            remaining[ev.target_idx] = false;
            self.dfs(
                world,
                model,
                srcs,
                events,
                remaining,
                cur_value + capture_value(prod, garrison, self.window, ev.arrival),
                cur_prod + prod,
                remaining_value - bound,
            );
            remaining[ev.target_idx] = true;
            events.pop();
            srcs.pop();
            for r in &mut srcs[si].raw {
                *r += ev.ships;
            }
            if self.nodes >= EARLY_GAME_NODE_BUDGET {
                break;
            }
        }
        options.clear();
        self.option_pool.push(options);
    }

    /// All launch options from `src` against one candidate target: the three
    /// ship-amount variants, each reduced to its strictly-improving arrival
    /// frontier over the affordable offsets. `avail` is the source's
    /// unclamped [`SrcState::raw`] vector — every comparison here is against
    /// an amount ≥ 1, so negative entries read as unaffordable, exactly like
    /// a zero clamp.
    #[allow(clippy::too_many_arguments)]
    fn options_for(
        &mut self,
        model: &HellburnerModel,
        src: i64,
        src_idx: usize,
        min_launch: i64,
        avail: &[i64],
        target_idx: usize,
        remaining: &[bool],
        out: &mut Vec<Event>,
    ) {
        let window = self.window;
        let target = self.candidates[target_idx].id;
        if src == target {
            return;
        }
        let min_ships = self.candidates[target_idx].garrison + 1;

        // Fixed amounts: minimal, plus min+child per nearby remaining
        // candidate (fund exactly one downstream hop through the target;
        // `children` never contains the target itself — the proximity graph
        // has no self-edges).
        let mut amounts: Vec<i64> = vec![min_ships];
        for &ci in &self.candidates[target_idx].children {
            if remaining[ci] {
                amounts.push(min_ships + self.candidates[ci].garrison + 1);
            }
        }
        amounts.sort_unstable();
        amounts.dedup();

        for &amount in &amounts {
            let row = row_for(&mut self.rows, window, model, src, target, amount);
            let mut best_arrival = i64::MAX;
            for o in min_launch..window {
                // Arrival from offset `o` is ≥ o, so no later offset can
                // extend the strictly-improving frontier.
                if o >= best_arrival {
                    break;
                }
                let Some(arrival) = row[o as usize] else {
                    continue;
                };
                if avail[o as usize] < amount {
                    continue;
                }
                if arrival < best_arrival {
                    best_arrival = arrival;
                    out.push(Event {
                        src,
                        src_idx,
                        target_idx,
                        ships: amount,
                        offset: o,
                        arrival,
                    });
                }
            }
        }

        // Ferry: send everything available, on its *own* arrival frontier —
        // the bigger fleet is faster, so an offset dominated for the minimal
        // fleet can still be the ferry's best. Plan-dependent ship counts
        // can't use the geometry row cache, so the affordable offsets are
        // subsampled (first affordable kept, rest evenly spread) to bound
        // per-node cost.
        let mut afford: Vec<i64> = (min_launch..window)
            .filter(|&o| avail[o as usize] > min_ships)
            .collect();
        if afford.len() > EARLY_GAME_FERRY_PROBES {
            let n = afford.len();
            afford = (0..EARLY_GAME_FERRY_PROBES)
                .map(|i| afford[i * n / EARLY_GAME_FERRY_PROBES])
                .collect();
        }
        let mut best_arrival = i64::MAX;
        for o in afford {
            // `afford` is ascending and an arrival is ≥ its launch offset, so
            // once an offset reaches the best arrival so far no later one in the
            // (already subsampled) list can improve the frontier.
            if o >= best_arrival {
                break;
            }
            let ferry = avail[o as usize];
            if amounts.contains(&ferry) {
                continue;
            }
            let Some(arrival) = arrival_within(model, src, target, ferry, o, window) else {
                continue;
            };
            if arrival < best_arrival {
                best_arrival = arrival;
                out.push(Event {
                    src,
                    src_idx,
                    target_idx,
                    ships: ferry,
                    offset: o,
                    arrival,
                });
            }
        }
    }
}
