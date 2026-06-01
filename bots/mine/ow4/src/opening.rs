//! Opening search.
//!
//! At the start of the game the greedy single-turn planner is myopic — a
//! slightly worse first capture can lock in a better 2nd/3rd/4th capture
//! (closer-to-it neutrals, more total production by t=50). This module
//! runs a small beam search over the next `OPENING_DEPTH` captures and
//! returns the action that maximises the summed
//! `production × (HORIZON − arrival)` over the whole sequence.
//!
//! Beam width × depth × ~20 candidates × ~4 sources = a few thousand
//! evaluations per call — well inside the 1-second turn budget.
//!
//! Only runs in the genuine opening (first few turns). After that the
//! regular planner takes over.

use crate::game::{Action, GameState, Planet};
use crate::ledger::Ledger;
use crate::pathing::dir_to_hit;

const OPENING_DEPTH: usize = 4;
const BEAM_WIDTH: usize = 6;
/// Short horizon for "fast start" — we care about production accumulated
/// over the opening phase, not the full game. A long horizon would equally
/// weight every capture and just reproduce greedy's per-step pick.
const HORIZON: i64 = 50;
/// Only consider this many nearest-to-me candidate neutrals — far ones
/// won't be in a "fastest start" sequence anyway.
const TOP_NEUTRALS: usize = 10;
/// Only run during the first few turns; after that the regular planner is
/// good enough and the search would burn time.
/// Opening search is currently disabled (set to -1) — empirically it
/// matched or slightly underperformed the per-turn greedy planner on
/// every opponent. Kept in the tree to revisit when the search criterion
/// or constraint model improves.
const OPENING_STEP_LIMIT: i64 = -1;

#[derive(Clone)]
struct Slot {
    id: i64,
    production: i64,
    captured_at: i64,
    ships_at_capture: i64,
    sends: Vec<(i64, i64)>,
}

impl Slot {
    fn ships_at(&self, tick: i64) -> i64 {
        if tick < self.captured_at {
            return 0;
        }
        let elapsed = tick - self.captured_at;
        let mut s = self.ships_at_capture + self.production * elapsed;
        for &(t, r) in &self.sends {
            if t <= tick {
                s -= r;
            }
        }
        s.max(0)
    }
}

#[derive(Clone)]
struct Node {
    owned: Vec<Slot>,
    captured: Vec<i64>,
    /// Sum of `production × (HORIZON − arrival)` across captures so far.
    total_reward: f64,
    /// Sum of ships committed across captures so far (for normalization).
    total_ships: i64,
    /// Sum of arrival ticks across captures so far (lower = faster start).
    total_arrivals: i64,
    /// Number of captures planned so far.
    n_captures: i64,
    /// (src_id, angle, ships, launch_tick) of the *very first* action in
    /// the sequence.
    first_action: Option<(i64, f64, i64, i64)>,
}

impl Node {
    /// Fast-start score: reward gained, penalised by total arrival time.
    /// A plan that gets 4 captures by tick 20 beats one that gets 4 by
    /// tick 40 with the same production. Includes a small per-ship cost
    /// so wasteful big-fleet plans don't dominate.
    fn fast_start_score(&self) -> f64 {
        if self.n_captures == 0 {
            return 0.0;
        }
        // Reward each tick of "earliness" by production weight.
        // total_reward already encodes prod × (HORIZON − arrival), so a
        // late arrival gives less reward. Combine with ship efficiency.
        self.total_reward / (self.total_ships as f64).max(1.0)
    }
}

/// Top neutrals by approximate distance from my closest owned planet (so
/// the search only considers plausible opening targets).
fn pick_top_neutrals<'a>(state: &'a GameState, me: i32) -> Vec<&'a Planet> {
    let owned: Vec<&Planet> = state.planets.iter().filter(|p| p.owner == me).collect();
    let mut scored: Vec<(f64, &Planet)> = state
        .planets
        .iter()
        .filter(|p| p.owner == -1)
        .map(|n| {
            let d = owned
                .iter()
                .map(|o| ((o.x - n.x).powi(2) + (o.y - n.y).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            (d, n)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(TOP_NEUTRALS).map(|(_, p)| p).collect()
}

fn expand(
    state: &GameState,
    node: &Node,
    neutrals: &[&Planet],
    must_launch_now: bool,
) -> Vec<Node> {
    let mut out = Vec::new();
    for tgt in neutrals {
        if node.captured.contains(&tgt.id) {
            continue;
        }
        let needed = tgt.ships + 1;
        for src_idx in 0..node.owned.len() {
            let src_slot = &node.owned[src_idx];
            let src_planet = match state.planet_by_id(src_slot.id) {
                Some(p) => p,
                None => continue,
            };
            // Earliest launch tick when `needed` ships exist at src.
            let base = src_slot.ships_at_capture;
            let extra = (needed - base).max(0);
            let earliest = if extra == 0 {
                src_slot.captured_at
            } else if src_slot.production > 0 {
                src_slot.captured_at
                    + ((extra + src_slot.production - 1) / src_slot.production)
            } else {
                continue;
            };
            if src_slot.ships_at(earliest) < needed {
                continue;
            }
            // At the very first expansion level we MUST launch this turn —
            // delayed plans aren't actionable now (they'd be picked up in
            // future turns once production catches up).
            if must_launch_now && earliest > 0 {
                continue;
            }
            let pr = match dir_to_hit(src_planet, tgt, needed, state, 0) {
                Some(p) => p,
                None => continue,
            };
            let arrival = earliest + pr.time;
            if arrival >= HORIZON {
                continue;
            }
            let reward = (tgt.production as f64) * ((HORIZON - arrival) as f64);

            let mut child = node.clone();
            child.owned[src_idx].sends.push((earliest, needed));
            child.owned.push(Slot {
                id: tgt.id,
                production: tgt.production,
                captured_at: arrival,
                ships_at_capture: 0,
                sends: Vec::new(),
            });
            child.captured.push(tgt.id);
            child.total_reward += reward;
            child.total_ships += needed;
            child.total_arrivals += arrival;
            child.n_captures += 1;
            if child.first_action.is_none() {
                child.first_action = Some((src_slot.id, pr.angle, needed, earliest));
            }
            out.push(child);
        }
    }
    out
}

pub fn opening_action(state: &GameState, ledger: &Ledger) -> Option<Action> {
    if state.step > OPENING_STEP_LIMIT {
        return None;
    }
    let me = state.player;
    let owned: Vec<&Planet> = state.planets.iter().filter(|p| p.owner == me).collect();
    if owned.is_empty() {
        return None;
    }
    // Filter out neutrals whose projected end-owner is already me — sending
    // again would be a wasteful double-commit (the in-flight fleet already
    // captures it).
    let mut neutrals = pick_top_neutrals(state, me);
    neutrals.retain(|p| {
        let (end_o, _) = ledger.projected_end(p.id);
        end_o != me
    });
    if neutrals.len() < 2 {
        return None;
    }

    // Initial node: my real owned planets, accounting for surplus already
    // spoken-for by in-flight commitments.
    let mut root = Node {
        owned: owned
            .iter()
            .map(|p| {
                let surp = ledger.surplus_at(p.id);
                let mut slot = Slot {
                    id: p.id,
                    production: p.production,
                    captured_at: 0,
                    ships_at_capture: p.ships,
                    sends: Vec::new(),
                };
                let gap = (p.ships - surp).max(0);
                if gap > 0 {
                    slot.sends.push((0, gap));
                }
                slot
            })
            .collect(),
        captured: Vec::new(),
        total_reward: 0.0,
        total_ships: 0,
        total_arrivals: 0,
        n_captures: 0,
        first_action: None,
    };
    let _ = &mut root;

    let mut frontier: Vec<Node> = vec![root];
    let mut best: Option<Node> = None;

    for depth_idx in 0..OPENING_DEPTH {
        let must_now = depth_idx == 0;
        let mut next: Vec<Node> = Vec::new();
        for n in &frontier {
            next.extend(expand(state, n, &neutrals, must_now));
        }
        if next.is_empty() {
            break;
        }
        // Track best across all depths by normalised score (reward per
        // ship spent) so a 1-capture plan competes fairly with a 4-capture
        // plan.
        for n in &next {
            if best
                .as_ref()
                .map(|b| n.fast_start_score() > b.fast_start_score())
                .unwrap_or(true)
            {
                best = Some(n.clone());
            }
        }
        next.sort_by(|a, b| b.fast_start_score().partial_cmp(&a.fast_start_score()).unwrap());
        next.truncate(BEAM_WIDTH);
        frontier = next;
    }

    let debug = std::env::var("OW4_DEBUG").ok().as_deref() == Some("1");
    // Walk best→worst across the final frontier and pick the first plan
    // whose first action is launchable RIGHT NOW (launch_tick == 0). A
    // plan that says "wait 4 turns then fire 23 ships" is not actionable
    // this turn — return None and let the greedy planner spend the small
    // surplus now.
    let mut all: Vec<&Node> = frontier.iter().collect();
    if let Some(b) = best.as_ref() {
        all.push(b);
    }
    all.sort_by(|a, b| b.fast_start_score().partial_cmp(&a.fast_start_score()).unwrap());
    let actionable = all.iter().find_map(|n| {
        n.first_action.and_then(|(sid, ang, ships, launch)| {
            if launch == 0 {
                Some((sid, ang, ships, n.fast_start_score()))
            } else {
                None
            }
        })
    });
    if debug {
        eprintln!(
            "  [opening] step={} owned={} neutrals={} actionable={:?}",
            state.step,
            state.planets.iter().filter(|p| p.owner == state.player).count(),
            neutrals.len(),
            actionable,
        );
    }
    actionable.map(|(src_id, ang, ships, _)| Action {
        from_id: src_id,
        angle: ang,
        ships,
    })
}
