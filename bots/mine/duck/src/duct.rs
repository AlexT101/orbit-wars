//! Decoupled UCT for simultaneous-move games.
//!
//! Models the game correctly: at each node, both players commit actions
//! privately, then the joint action is applied. Each player picks their
//! action by PUCT on their OWN MARGINAL stats (summed across what the
//! opponent might play). Backprop updates both players' marginal stats
//! plus the joint child.
//!
//! Compared to the sequential MCTS in `mcts.rs`:
//!   - No opp-sees-my-action info leak
//!   - Branching at each node is my_K × opp_K (joint), but marginal stats
//!     accumulate at N/my_K (my) and N/opp_K (opp) — faster signal than
//!     fully joint
//!   - Tree depth is half (no MyTurn/EnemyTurn alternation)

use crate::policy::XorRng;
use crate::sim::{alive_players, apply_launches, tick, Action};
use crate::{ow2_plan, GameState};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

const EXPLORATION: f64 = 0.3;
const TERMINAL_STEP: i64 = 500;
const K_ROOT: usize = 5;
const K_NON_ROOT: usize = 4;

thread_local! {
    /// Stash the root after each turn, so next turn's matching state can
    /// reuse the joint subtree corresponding to (my_chosen, opp_actual).
    static LAST_TREE: RefCell<Option<(i64, Box<Node>)>> = RefCell::new(None);
}

#[derive(Clone)]
struct ActionStats {
    visits: u32,
    sum_value: f64, // from MY perspective
}

#[derive(Clone)]
struct Node {
    state: GameState,
    visits: u32,
    my_candidates: Vec<Vec<Action>>,
    my_priors: Vec<f64>,
    my_stats: Vec<ActionStats>,
    opp_candidates: Vec<Vec<Action>>,
    opp_priors: Vec<f64>,
    opp_stats: Vec<ActionStats>,
    /// (my_idx, opp_idx) -> joint child subtree
    children: HashMap<(usize, usize), Box<Node>>,
    candidates_initialized: bool,
}

fn rank_prior(rank: usize, total: usize) -> f64 {
    let raw = 0.5_f64.powi(rank as i32);
    let z: f64 = (0..total).map(|i| 0.5_f64.powi(i as i32)).sum();
    raw / z
}

fn enumerate_alternatives_strong(state: &GameState, player: i32, k: usize) -> Vec<Vec<Action>> {
    let nc = std::env::var("OW_NO_COOP").is_ok();
    let greedy = ow2_plan::plan(state, player, nc);
    let mut out: Vec<Vec<Action>> = vec![greedy];
    for tgt in &state.planets {
        if tgt.owner == player {
            continue;
        }
        if out.len() >= k {
            break;
        }
        let alt = ow2_plan::plan_with_exclusion(state, player, nc, Some(tgt.id));
        if !out.iter().any(|a| actions_equal(a, &alt)) {
            out.push(alt);
        }
    }
    if out.len() < k && !out.iter().any(|a| a.is_empty()) {
        out.push(Vec::new());
    }
    out
}

/// Fast alternative enumeration using the shared-precompute `top_n` function.
fn enumerate_alternatives_fast(state: &GameState, player: i32, k: usize) -> Vec<Vec<Action>> {
    crate::policy::rollout_policy_fast_top_n(state, player, k)
}

fn enumerate_alternatives(state: &GameState, player: i32, k: usize, _is_root: bool) -> Vec<Vec<Action>> {
    // Default to STRONG everywhere (match v4-reusefix policy). The fast
    // variant lost 0-6 vs v4 in sequential MCTS — same trade-off applies
    // here. Override: OW_DUCT_ENUMERATE=fast forces fast (mostly for
    // diagnostics).
    let mode = std::env::var("OW_DUCT_ENUMERATE").unwrap_or_default();
    let use_strong = match mode.as_str() {
        "fast" => false,
        _ => true,
    };
    if use_strong {
        enumerate_alternatives_strong(state, player, k)
    } else {
        enumerate_alternatives_fast(state, player, k)
    }
}

fn actions_equal(a: &[Action], b: &[Action]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let key = |x: &Action| (x.0, (x.1 * 100.0).round() as i64, x.2);
    let mut ax: Vec<_> = a.iter().map(key).collect();
    let mut bx: Vec<_> = b.iter().map(key).collect();
    ax.sort();
    bx.sort();
    ax == bx
}

fn dominant_enemy(state: &GameState, me: i32) -> Option<i32> {
    let mut best: Option<(i32, i64)> = None;
    let mut visit_player = |p: i32| {
        if p == -1 || p == me {
            return;
        }
        let s = crate::sim::player_score(state, p);
        if best.as_ref().map(|b| s > b.1).unwrap_or(true) {
            best = Some((p, s));
        }
    };
    for p in &state.planets {
        visit_player(p.owner);
    }
    for f in &state.fleets {
        visit_player(f.owner);
    }
    best.map(|b| b.0)
}

fn puct_c() -> f64 {
    std::env::var("OW_PUCT_C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(EXPLORATION)
}

fn select_my(node: &Node) -> usize {
    let c = puct_c();
    let parent_n = node.visits.max(1) as f64;
    let mut best_i = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..node.my_candidates.len() {
        let st = &node.my_stats[i];
        let exploit = if st.visits == 0 {
            0.0
        } else {
            st.sum_value / st.visits as f64
        };
        let explore = c * node.my_priors[i] * parent_n.sqrt() / (1.0 + st.visits as f64);
        let s = exploit + explore;
        if s > best_score {
            best_score = s;
            best_i = i;
        }
    }
    best_i
}

fn select_opp(node: &Node) -> usize {
    // Opp wants to minimize MY value → negate exploit.
    let c = puct_c();
    let parent_n = node.visits.max(1) as f64;
    let mut best_i = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..node.opp_candidates.len() {
        let st = &node.opp_stats[i];
        let exploit = if st.visits == 0 {
            0.0
        } else {
            -st.sum_value / st.visits as f64
        };
        let explore = c * node.opp_priors[i] * parent_n.sqrt() / (1.0 + st.visits as f64);
        let s = exploit + explore;
        if s > best_score {
            best_score = s;
            best_i = i;
        }
    }
    best_i
}

fn ensure_candidates(node: &mut Node, me: i32, root: bool) {
    if node.candidates_initialized {
        return;
    }
    let k = if root { K_ROOT } else { K_NON_ROOT };
    let opp = dominant_enemy(&node.state, me).unwrap_or(1 - me);
    let my_alts = enumerate_alternatives(&node.state, me, k, root);
    let opp_alts = enumerate_alternatives(&node.state, opp, k, root);
    let my_n = my_alts.len();
    let opp_n = opp_alts.len();
    node.my_priors = (0..my_n).map(|i| rank_prior(i, my_n)).collect();
    node.opp_priors = (0..opp_n).map(|i| rank_prior(i, opp_n)).collect();
    node.my_stats = (0..my_n).map(|_| ActionStats { visits: 0, sum_value: 0.0 }).collect();
    node.opp_stats = (0..opp_n).map(|_| ActionStats { visits: 0, sum_value: 0.0 }).collect();
    node.my_candidates = my_alts;
    node.opp_candidates = opp_alts;
    node.candidates_initialized = true;
}

fn rollout(mut state: GameState, me: i32, rng: &mut XorRng) -> f64 {
    let mode = std::env::var("OW_ROLLOUT").unwrap_or_else(|_| "ow2_full".to_string());
    let depth: i64 = std::env::var("OW_ROLLOUT_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(match mode.as_str() {
            "none" => 0,
            "fast" => 30,
            "ow2_short" => 2,
            "ow2_full" => 8,
            "ow2_fast" => 12,
            _ => 2,
        });
    for _ in 0..depth {
        if state.step >= TERMINAL_STEP || alive_players(&state) <= 1 {
            break;
        }
        let my_score = crate::sim::player_score(&state, me) as f64;
        let opp_score: f64 = state
            .planets
            .iter()
            .filter(|p| p.owner != -1 && p.owner != me)
            .map(|p| p.ships as f64)
            .sum::<f64>()
            + state
                .fleets
                .iter()
                .filter(|f| f.owner != me)
                .map(|f| f.ships as f64)
                .sum::<f64>();
        let tot = my_score + opp_score;
        if tot > 30.0 && (my_score / tot < 0.05 || opp_score / tot < 0.05) {
            break;
        }
        let opp = dominant_enemy(&state, me);
        let (my_act, opp_act) = if mode == "ow2_fast" {
            (
                crate::policy::rollout_policy_fast(&state, me),
                opp.map(|o| crate::policy::rollout_policy_fast(&state, o)).unwrap_or_default(),
            )
        } else {
            let nc = std::env::var("OW_NO_COOP").is_ok();
            (
                ow2_plan::plan(&state, me, nc),
                opp.map(|o| ow2_plan::plan(&state, o, nc)).unwrap_or_default(),
            )
        };
        apply_launches(&mut state, &my_act);
        apply_launches(&mut state, &opp_act);
        tick(&mut state, rng);
    }
    let v = evaluate(&state, me);
    // Rollout noise REGRESSED DUCT 2-4 in head-to-head (vs benefit in v4).
    // Likely because DUCT applies the same noisy value to both my and opp
    // marginal stats, amplifying noise into opp's selection — opp seems
    // weaker than it should, inflating my perceived value.
    // Default OFF for DUCT. Set OW_ROLLOUT_NOISE > 0 to re-enable.
    let noise: f64 = std::env::var("OW_ROLLOUT_NOISE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    if noise > 0.0 {
        let jitter = (rng.next_f64() * 2.0 - 1.0) * noise;
        v + jitter
    } else {
        v
    }
}

fn evaluate(state: &GameState, me: i32) -> f64 {
    crate::mcts::evaluate_external(state, me)
}

fn select_and_expand(node: &mut Node, me: i32, rng: &mut XorRng, is_root: bool) -> f64 {
    if node.state.step >= TERMINAL_STEP || alive_players(&node.state) <= 1 {
        let v = evaluate(&node.state, me);
        node.visits += 1;
        return v;
    }
    ensure_candidates(node, me, is_root);
    let my_idx = select_my(node);
    let opp_idx = select_opp(node);
    let value: f64;
    if !node.children.contains_key(&(my_idx, opp_idx)) {
        // Expand: apply both actions, tick, create new node, rollout.
        let mut s = node.state.clone();
        apply_launches(&mut s, &node.my_candidates[my_idx]);
        apply_launches(&mut s, &node.opp_candidates[opp_idx]);
        tick(&mut s, rng);
        let rollout_value = rollout(s.clone(), me, rng);
        let child = Node {
            state: s,
            visits: 1,
            my_candidates: Vec::new(),
            my_priors: Vec::new(),
            my_stats: Vec::new(),
            opp_candidates: Vec::new(),
            opp_priors: Vec::new(),
            opp_stats: Vec::new(),
            children: HashMap::new(),
            candidates_initialized: false,
        };
        node.children.insert((my_idx, opp_idx), Box::new(child));
        value = rollout_value;
    } else {
        // Recurse.
        let child = node.children.get_mut(&(my_idx, opp_idx)).unwrap();
        value = select_and_expand(child, me, rng, false);
    }
    // Backprop: update both marginal stats + joint node.
    node.visits += 1;
    node.my_stats[my_idx].visits += 1;
    node.my_stats[my_idx].sum_value += value;
    node.opp_stats[opp_idx].visits += 1;
    node.opp_stats[opp_idx].sum_value += value;
    value
}

fn state_hash(state: &GameState) -> u64 {
    let mut h: u64 = state.step as u64;
    for p in &state.planets {
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(p.id as u64);
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add((p.owner as i64 + 1) as u64);
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(p.ships as u64);
    }
    for f in &state.fleets {
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(f.from_planet_id as u64);
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add((f.owner + 1) as u64);
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(f.ships as u64);
    }
    h
}

pub fn best_move(state: &GameState, me: i32, budget_ms: u64) -> Vec<Action> {
    let reuse_disabled = std::env::var("OW_NO_REUSE").is_ok();
    let target_hash = state_hash(state);
    let mut reused: Option<Box<Node>> = None;
    if !reuse_disabled {
        LAST_TREE.with(|cell| {
            let mut slot = cell.borrow_mut();
            if let Some((expected_step, prev_root)) = slot.take() {
                if expected_step == state.step {
                    // Find joint child whose state matches.
                    for (_key, child) in prev_root.children.iter() {
                        if state_hash(&child.state) == target_hash {
                            reused = Some(child.clone());
                            break;
                        }
                    }
                }
            }
        });
    }
    let mut root = match reused {
        Some(r) => *r,
        None => Node {
            state: state.clone(),
            visits: 0,
            my_candidates: Vec::new(),
            my_priors: Vec::new(),
            my_stats: Vec::new(),
            opp_candidates: Vec::new(),
            opp_priors: Vec::new(),
            opp_stats: Vec::new(),
            children: HashMap::new(),
            candidates_initialized: false,
        },
    };
    let seed = (state.step as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        ^ 0xdeadbeefcafebabe;
    let mut rng = XorRng(seed | 1);
    ensure_candidates(&mut root, me, true);

    let deadline = Instant::now() + std::time::Duration::from_millis(budget_ms);
    let mut iters = 0u32;
    while Instant::now() < deadline {
        select_and_expand(&mut root, me, &mut rng, true);
        iters += 1;
        if iters > 100_000 {
            break;
        }
    }

    if std::env::var("OW_DEBUG").is_ok() {
        let mut child_info: Vec<String> = Vec::new();
        for i in 0..root.my_candidates.len() {
            let st = &root.my_stats[i];
            let avg = if st.visits > 0 { st.sum_value / st.visits as f64 } else { 0.0 };
            let target_summary: Vec<String> = root.my_candidates[i]
                .iter()
                .take(3)
                .map(|x| format!("(s={},t≈{:.2})", x.0, x.1))
                .collect();
            child_info.push(format!("v={}/avg={:.3} {}", st.visits, avg, target_summary.join(",")));
        }
        eprintln!(
            "[duck] step={} iters={} root_visits={} my_K={} | {}",
            state.step, iters, root.visits, root.my_candidates.len(),
            child_info.join(" || ")
        );
    }

    if root.my_candidates.is_empty() {
        return Vec::new();
    }
    // Pick by raw max in my marginal stats. Margin override REGRESSED DUCT
    // 0-6 vs base (opp's independent PUCT exploits greedy; alts have
    // less-explored opp responses → genuinely better; forcing stay-with-
    // greedy via margin defeats this). Default to raw max.
    // To re-enable margin: set OW_MARGIN > 0.
    let robust_margin: f64 = std::env::var("OW_MARGIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let min_visits: u32 = std::env::var("OW_MIN_OVERRIDE_VISITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let mut best_i = 0usize;
    let mut best_val = if root.my_stats[0].visits > 0 {
        root.my_stats[0].sum_value / root.my_stats[0].visits as f64
    } else {
        f64::NEG_INFINITY
    };
    for i in 1..root.my_candidates.len() {
        let st = &root.my_stats[i];
        if st.visits < min_visits {
            continue;
        }
        let avg = st.sum_value / st.visits as f64;
        if avg > best_val + robust_margin {
            best_val = avg;
            best_i = i;
        }
    }
    let chosen = root.my_candidates[best_i].clone();

    // Stash root for next turn's reuse (so we can match observed state to
    // a joint child).
    if !reuse_disabled {
        let next_step = state.step + 1;
        LAST_TREE.with(|cell| {
            *cell.borrow_mut() = Some((next_step, Box::new(root)));
        });
    }
    chosen
}
