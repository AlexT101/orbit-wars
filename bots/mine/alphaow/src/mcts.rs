//! MCTS over per-turn joint actions. Alternating: my full turn, then enemy's
//! full turn (with my action visible). The engine `tick` advances only after
//! both players have committed.
//!
//! Leaf evaluation is a light rollout using the same `sample_joint_action`
//! policy with scoring noise (so each rollout is somewhat different).

use crate::policy::{
    greedy_joint_action, sample_joint_action, sample_joint_action_fast, XorRng,
};
use crate::sim::{alive_players, apply_launches, player_score, tick, Action};
use crate::{ow2_plan, GameState};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// Profile counters (only used when OW_PROFILE=1). Accumulate nanoseconds.
static PROF_ROLLOUT_NS: AtomicU64 = AtomicU64::new(0);
static PROF_EXPAND_NS: AtomicU64 = AtomicU64::new(0);  // state.clone + apply_launches + tick
static PROF_CANDIDATES_NS: AtomicU64 = AtomicU64::new(0);  // ensure_candidates first call
static PROF_SELECT_NS: AtomicU64 = AtomicU64::new(0);  // PUCT pick + recurse
static PROF_EVAL_NS: AtomicU64 = AtomicU64::new(0);  // eval (lookahead + score)
static PROF_ROLLOUT_COUNT: AtomicU64 = AtomicU64::new(0);
static PROF_EXPAND_COUNT: AtomicU64 = AtomicU64::new(0);
static PROF_CANDIDATES_COUNT: AtomicU64 = AtomicU64::new(0);

fn profile_enabled() -> bool {
    std::env::var("OW_PROFILE").is_ok()
}

fn prof_reset() {
    PROF_ROLLOUT_NS.store(0, Ordering::Relaxed);
    PROF_EXPAND_NS.store(0, Ordering::Relaxed);
    PROF_CANDIDATES_NS.store(0, Ordering::Relaxed);
    PROF_SELECT_NS.store(0, Ordering::Relaxed);
    PROF_EVAL_NS.store(0, Ordering::Relaxed);
    PROF_ROLLOUT_COUNT.store(0, Ordering::Relaxed);
    PROF_EXPAND_COUNT.store(0, Ordering::Relaxed);
    PROF_CANDIDATES_COUNT.store(0, Ordering::Relaxed);
}

fn prof_report(total_ms: f64) {
    let rollout_ms = PROF_ROLLOUT_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let expand_ms = PROF_EXPAND_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let cands_ms = PROF_CANDIDATES_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let select_ms = PROF_SELECT_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let eval_ms = PROF_EVAL_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let r_count = PROF_ROLLOUT_COUNT.load(Ordering::Relaxed);
    let e_count = PROF_EXPAND_COUNT.load(Ordering::Relaxed);
    let c_count = PROF_CANDIDATES_COUNT.load(Ordering::Relaxed);
    eprintln!("[profile] total={:.1}ms", total_ms);
    eprintln!("[profile]   rollout: {:.1}ms ({} calls, avg {:.2}ms)", rollout_ms, r_count, rollout_ms / r_count.max(1) as f64);
    eprintln!("[profile]   expand:  {:.1}ms ({} calls, avg {:.2}ms)", expand_ms, e_count, expand_ms / e_count.max(1) as f64);
    eprintln!("[profile]   cands:   {:.1}ms ({} calls, avg {:.2}ms)", cands_ms, c_count, cands_ms / c_count.max(1) as f64);
    eprintln!("[profile]   select:  {:.1}ms", select_ms);
    eprintln!("[profile]   eval:    {:.1}ms", eval_ms);
}

thread_local! {
    /// Stash the subtree under our chosen action after each turn, keyed by
    /// (step+1). On the next turn, if the observed state matches one of
    /// that subtree's MyTurn grandchildren, we reuse it instead of starting
    /// fresh. Compounds search effort across turns.
    static LAST_TREE: RefCell<Option<(i64, Box<Node>)>> = RefCell::new(None);
}

/// Deterministic content hash of the parts of state that change turn-to-turn.
/// Used to match an observed state against a stored MyTurn subtree's state.
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

const EXPLORATION: f64 = 0.3;
const MAX_EXPANSIONS_PER_NODE: u32 = 8;
const TERMINAL_STEP: i64 = 500;

#[derive(Clone)]
enum NodeKind {
    MyTurn,
    EnemyTurn { my_action: Vec<Action> },
}

#[derive(Clone)]
struct Node {
    state: GameState,
    kind: NodeKind,
    visits: u32,
    total_value: f64,
    /// (action, prior, subtree). Children are added in candidate order so
    /// `children[i]` corresponds to `candidates[i]` with prior `priors[i]`.
    children: Vec<(Vec<Action>, Box<Node>)>,
    /// Lazy-init candidate actions for this node. Populated on first
    /// expansion call via `ensure_candidates`. Greedy ow2_plan is always
    /// candidates[0]; the rest are top-K-1 alternatives.
    candidates: Vec<Vec<Action>>,
    /// Prior probability per candidate (geometric decay by rank). Used by
    /// PUCT to bias exploration toward high-quality moves.
    priors: Vec<f64>,
    candidates_initialized: bool,
}

impl Node {
    fn clone_box(&self) -> Box<Node> {
        Box::new(self.clone())
    }
}

const K_NON_ROOT: usize = 4;

fn dominant_enemy(state: &GameState, me: i32) -> Option<i32> {
    let mut best: Option<(i32, i64)> = None;
    let mut visit_player = |p: i32| {
        if p == -1 || p == me {
            return;
        }
        let s = player_score(state, p);
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

pub fn evaluate_external(state: &GameState, me: i32) -> f64 {
    evaluate(state, me)
}

fn evaluate(state: &GameState, me: i32) -> f64 {
    // Simulate forward N turns with NO new actions, letting in-flight fleets
    // resolve and production tick. This makes the eval reflect the "value"
    // of the current trajectory, not just instantaneous ship counts.
    let lookahead: i64 = std::env::var("OW_EVAL_LOOKAHEAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let mut s = state.clone();
    let mut rng = XorRng(0xdeadbeef ^ (s.step as u64));
    for _ in 0..lookahead {
        if alive_players(&s) <= 1 || s.step >= TERMINAL_STEP {
            break;
        }
        crate::sim::tick(&mut s, &mut rng);
    }
    raw_score(&s, me)
}

fn raw_score(state: &GameState, me: i32) -> f64 {
    // Legacy (ship-count + flat planet bonus) is the default — empirically
    // beats prod-weighted 6-0 head-to-head despite both sweeping ow2.
    // Use OW_EVAL_PRODWEIGHT=1 to switch to the prod-weighted variant.
    if !std::env::var("OW_EVAL_PRODWEIGHT").is_ok() {
        return raw_score_legacy(state, me);
    }
    // Three components, weighted:
    //   1. Current ships (planets + fleets).
    //   2. Future production: each owned planet contributes
    //      `production × min(50, comet_remaining)` — i.e., the next 50 turns
    //      of income (capped for comets at their remaining lifespan).
    //   3. Endgame: huge bonus if opponent is eliminated.
    let mut my_ships = 0.0_f64;
    let mut opp_ships = 0.0_f64;
    let mut my_future = 0.0_f64;
    let mut opp_future = 0.0_f64;
    let horizon = 50i64;
    for p in &state.planets {
        if p.owner == -1 {
            continue;
        }
        let remaining = if p.is_comet {
            state.comet_remaining(p).max(0).min(horizon)
        } else {
            horizon
        };
        let future = (p.production as f64) * (remaining as f64);
        if p.owner == me {
            my_ships += p.ships as f64;
            my_future += future;
        } else {
            opp_ships += p.ships as f64;
            opp_future += future;
        }
    }
    for f in &state.fleets {
        if f.owner == me {
            my_ships += f.ships as f64;
        } else {
            opp_ships += f.ships as f64;
        }
    }
    let my_total = my_ships + my_future;
    let opp_total = opp_ships + opp_future;
    let denom = my_total + opp_total;
    if denom < 1.0 {
        return 0.0;
    }
    let base = (my_total - opp_total) / denom;
    // Hard endgame bonus
    let my_planets = state.planets.iter().filter(|p| p.owner == me).count();
    let opp_planets = state
        .planets
        .iter()
        .filter(|p| p.owner != -1 && p.owner != me)
        .count();
    let opp_fleets = state.fleets.iter().filter(|f| f.owner != me).count();
    let me_fleets = state.fleets.iter().filter(|f| f.owner == me).count();
    if opp_planets == 0 && opp_fleets == 0 && my_planets > 0 {
        return 1.0;
    }
    if my_planets == 0 && me_fleets == 0 && opp_planets > 0 {
        return -1.0;
    }
    base
}

/// Old eval (pre-prodweight) — kept for A/B testing via OW_EVAL_LEGACY=1.
fn raw_score_legacy(state: &GameState, me: i32) -> f64 {
    let my = player_score(state, me) as f64;
    let mut opp = 0.0_f64;
    for p in &state.planets {
        if p.owner != -1 && p.owner != me {
            opp += p.ships as f64;
        }
    }
    for f in &state.fleets {
        if f.owner != me {
            opp += f.ships as f64;
        }
    }
    let my_planets: i64 = state.planets.iter().filter(|p| p.owner == me).count() as i64;
    let opp_planets: i64 = state
        .planets
        .iter()
        .filter(|p| p.owner != me && p.owner != -1)
        .count() as i64;
    let planet_bonus = (my_planets - opp_planets) as f64 * 5.0;
    let tot = my + opp;
    if tot < 1.0 {
        return planet_bonus / 10.0;
    }
    ((my - opp) + planet_bonus) / (tot + 10.0)
}

/// Rollout-only noise amount, scaled to eval range ~[-1, 1]. Adds random
/// jitter to the final value so MCTS sees variance across iterations
/// targeting the same subtree (helpful for exploration when rollouts are
/// deterministic given state). Default 0.05; tunable via OW_ROLLOUT_NOISE.
fn rollout_noise_amount() -> f64 {
    std::env::var("OW_ROLLOUT_NOISE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05)
}

fn rollout(mut state: GameState, me: i32, rng: &mut XorRng, _depth: i64) -> f64 {
    // Rollout policy mode set by OW_ROLLOUT env:
    //   "none"      — no rollout, just evaluate the leaf
    //   "fast"      — old fast sampler
    //   "ow2_short" — ow2_plan for ~3 steps then evaluate (default)
    //   "ow2_full"  — ow2_plan for N steps (slow, accurate)
    let mode = std::env::var("OW_ROLLOUT").unwrap_or_else(|_| "ow2_full".to_string());
    let depth: i64 = std::env::var("OW_ROLLOUT_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(match mode.as_str() {
            "none" => 0,
            "fast" => 30,
            "ow2_short" => 2,
            "ow2_full" => 8,
            "ow2_fast" => 12, // user's suggested target-centric heuristic
            "mixed" => 6,
            _ => 2,
        });
    for k in 0..depth {
        if state.step >= TERMINAL_STEP || alive_players(&state) <= 1 {
            break;
        }
        // Truncate if game is effectively decided: one side has < 5% of
        // total ships AND total ships > 30 (i.e., not just an early-game
        // imbalance). Frees rollout iters for harder states.
        let my_score = player_score(&state, me) as f64;
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
        let use_fast = mode == "fast" || (mode == "mixed" && k >= 2);
        let (my_act, opp_act) = if use_fast {
            (
                sample_joint_action_fast(&state, me, rng),
                opp.map(|o| sample_joint_action_fast(&state, o, rng)).unwrap_or_default(),
            )
        } else if mode == "ow2_fast" {
            (
                crate::policy::rollout_policy_fast(&state, me),
                opp.map(|o| crate::policy::rollout_policy_fast(&state, o)).unwrap_or_default(),
            )
        } else {
            (
                {
                    let nc = std::env::var("OW_NO_COOP").is_ok();
                    ow2_plan::plan(&state, me, nc)
                },
                {
                    let nc = std::env::var("OW_NO_COOP").is_ok();
                    opp.map(|o| ow2_plan::plan(&state, o, nc)).unwrap_or_default()
                },
            )
        };
        apply_launches(&mut state, &my_act);
        apply_launches(&mut state, &opp_act);
        tick(&mut state, rng);
    }
    let v = evaluate(&state, me);
    // Add small noise ONLY to rollout-returned values (not to terminal-state
    // eval in select_and_expand). Helps MCTS explore when multiple iterations
    // hit similar leaf positions.
    let noise = rollout_noise_amount();
    if noise > 0.0 {
        // rng.next_f64() in [0, 1); shift to [-1, 1) then scale by `noise`.
        let jitter = (rng.next_f64() * 2.0 - 1.0) * noise;
        v + jitter
    } else {
        v
    }
}

#[allow(dead_code)]
fn rollout_legacy(state: GameState, me: i32, rng: &mut XorRng) -> f64 {
    let _ = (state, me, rng, sample_joint_action);
    0.0
}

/// PUCT score (AlphaZero-style): exploit + c × prior × sqrt(N_parent) / (1 + N_child).
/// `sign` flips at enemy-turn nodes so we always pick what's best for `me`.
fn puct_score(child_value: f64, child_visits: u32, parent_visits: u32, prior: f64, sign: f64) -> f64 {
    let c = puct_c();
    let exploit = if child_visits == 0 {
        0.0
    } else {
        sign * (child_value / child_visits as f64)
    };
    let explore = c * prior * (parent_visits as f64).sqrt() / (1.0 + child_visits as f64);
    exploit + explore
}

fn puct_c() -> f64 {
    std::env::var("OW_PUCT_C").ok().and_then(|s| s.parse().ok()).unwrap_or(EXPLORATION)
}

#[allow(dead_code)]
fn ucb_score(child_value: f64, child_visits: u32, parent_visits: u32, sign: f64) -> f64 {
    if child_visits == 0 {
        return f64::INFINITY;
    }
    let exploit = sign * (child_value / child_visits as f64);
    let explore = EXPLORATION * ((parent_visits as f64).ln().max(0.0) / child_visits as f64).sqrt();
    exploit + explore
}

/// Geometric-decay prior for the i-th candidate (0-indexed). Greedy gets
/// the largest mass; alternatives decay quickly.
fn rank_prior(rank: usize, total: usize) -> f64 {
    let raw = 0.5_f64.powi(rank as i32);
    // Normalize so priors sum to 1 over the visible candidates.
    let z: f64 = (0..total).map(|i| 0.5_f64.powi(i as i32)).sum();
    raw / z
}

fn enumerate_alternatives_strong(state: &GameState, player: i32, k: usize) -> Vec<Vec<Action>> {
    let nc = std::env::var("OW_NO_COOP").is_ok();
    let greedy = crate::ow2_plan::plan(state, player, nc);
    let mut out: Vec<Vec<Action>> = vec![greedy];
    for tgt in &state.planets {
        if tgt.owner == player {
            continue;
        }
        if out.len() >= k {
            break;
        }
        let alt = crate::ow2_plan::plan_with_exclusion(state, player, nc, Some(tgt.id));
        if !out.iter().any(|a| actions_equal(a, &alt)) {
            out.push(alt);
        }
    }
    if out.len() < k && !out.iter().any(|a| a.is_empty()) {
        out.push(Vec::new());
    }
    out
}

/// Same shape as `enumerate_alternatives_strong` but uses `rollout_policy_fast`
/// (~0.05ms/call vs ow2_plan's ~2ms). Lower-quality candidates, much faster.
/// Uses `rollout_policy_fast_top_n` which shares precomputed extrapolation
/// + race filter + dir_to_hit cache across all K plans.
fn enumerate_alternatives_fast(state: &GameState, player: i32, k: usize) -> Vec<Vec<Action>> {
    crate::policy::rollout_policy_fast_top_n(state, player, k)
}

/// Route between strong (ow2_plan) and fast (rollout_policy_fast) enumeration.
/// Root always uses strong (it's the most-visited node; quality matters most).
/// Non-root: controlled by OW_MCTS_NONROOT_FAST=1 (when set, uses fast).
fn enumerate_alternatives(state: &GameState, player: i32, k: usize, is_root: bool) -> Vec<Vec<Action>> {
    if !is_root && std::env::var("OW_MCTS_NONROOT_FAST").is_ok() {
        enumerate_alternatives_fast(state, player, k)
    } else {
        enumerate_alternatives_strong(state, player, k)
    }
}

fn ensure_candidates(node: &mut Node, me: i32) {
    if node.candidates_initialized {
        return;
    }
    let to_act = match node.kind {
        NodeKind::MyTurn => me,
        NodeKind::EnemyTurn { .. } => dominant_enemy(&node.state, me).unwrap_or(me),
    };
    // Non-root only — root explicitly bypasses this via the seed phase in
    // best_move using the strong enumerator.
    let alts = enumerate_alternatives(&node.state, to_act, K_NON_ROOT, false);
    let n = alts.len();
    let priors: Vec<f64> = (0..n).map(|i| rank_prior(i, n)).collect();
    node.candidates = alts;
    node.priors = priors;
    node.candidates_initialized = true;
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

fn same_action(a: &[Action], b: &[Action]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    // Compare by (source, angle_quantized, ships). Two different targets
    // give different angles, so this catches "different target" properly.
    let key = |x: &Action| (x.0, (x.1 * 100.0).round() as i64, x.2);
    let mut ax: Vec<_> = a.iter().map(key).collect();
    let mut bx: Vec<_> = b.iter().map(key).collect();
    ax.sort();
    bx.sort();
    ax == bx
}

fn select_and_expand(node: &mut Node, me: i32, rng: &mut XorRng) -> f64 {
    if node.state.step >= TERMINAL_STEP || alive_players(&node.state) <= 1 {
        let t0 = Instant::now();
        let v = evaluate(&node.state, me);
        if profile_enabled() {
            PROF_EVAL_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        node.visits += 1;
        node.total_value += v;
        return v;
    }
    if !node.candidates_initialized {
        let t0 = Instant::now();
        ensure_candidates(node, me);
        if profile_enabled() {
            PROF_CANDIDATES_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            PROF_CANDIDATES_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    } else {
        ensure_candidates(node, me);
    }
    let value: f64;
    // If there's an un-expanded candidate, expand the next one in rank order.
    if node.children.len() < node.candidates.len() {
        let idx = node.children.len();
        let action = node.candidates[idx].clone();
        let t_expand = Instant::now();
        let (child_state, child_kind) = match &node.kind {
            NodeKind::MyTurn => (
                node.state.clone(),
                NodeKind::EnemyTurn { my_action: action.clone() },
            ),
            NodeKind::EnemyTurn { my_action } => {
                let mut s = node.state.clone();
                apply_launches(&mut s, my_action);
                apply_launches(&mut s, &action);
                tick(&mut s, rng);
                (s, NodeKind::MyTurn)
            }
        };
        let rollout_start = match &child_kind {
            NodeKind::MyTurn => child_state.clone(),
            NodeKind::EnemyTurn { my_action } => {
                let opp = dominant_enemy(&child_state, me);
                let mut s = child_state.clone();
                apply_launches(&mut s, my_action);
                if let Some(o) = opp {
                    let nc = std::env::var("OW_NO_COOP").is_ok();
                    let opp_act = crate::ow2_plan::plan(&s, o, nc);
                    apply_launches(&mut s, &opp_act);
                }
                tick(&mut s, rng);
                s
            }
        };
        if profile_enabled() {
            PROF_EXPAND_NS.fetch_add(t_expand.elapsed().as_nanos() as u64, Ordering::Relaxed);
            PROF_EXPAND_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        let t_rollout = Instant::now();
        let v = rollout(rollout_start, me, rng, 0);
        if profile_enabled() {
            PROF_ROLLOUT_NS.fetch_add(t_rollout.elapsed().as_nanos() as u64, Ordering::Relaxed);
            PROF_ROLLOUT_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        let child = Node {
            state: child_state,
            kind: child_kind,
            visits: 1,
            total_value: v,
            children: Vec::new(),
            candidates: Vec::new(),
            priors: Vec::new(),
            candidates_initialized: false,
        };
        node.children.push((action, Box::new(child)));
        value = v;
    } else {
        value = select_existing(node, me, rng);
    }
    node.visits += 1;
    node.total_value += value;
    value
}

fn select_existing(node: &mut Node, me: i32, rng: &mut XorRng) -> f64 {
    let sign = match node.kind {
        NodeKind::MyTurn => 1.0,
        NodeKind::EnemyTurn { .. } => -1.0,
    };
    let parent_visits = node.visits.max(1);
    let mut best_i = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, (_, c)) in node.children.iter().enumerate() {
        let prior = node.priors.get(i).copied().unwrap_or(1.0 / node.children.len() as f64);
        let s = puct_score(c.total_value, c.visits, parent_visits, prior, sign);
        if s > best_score {
            best_score = s;
            best_i = i;
        }
    }
    let child = &mut node.children[best_i].1;
    select_and_expand(child, me, rng)
}

/// Find the best joint action for `me` within `budget_ms`. Returns the actions
/// to play this turn from MY planets.
pub fn best_move(state: &GameState, me: i32, budget_ms: u64) -> Vec<Action> {
    if std::env::var("OW_BYPASS_MCTS").is_ok() {
        return greedy_joint_action(state, me);
    }
    if profile_enabled() {
        prof_reset();
    }
    let prof_start = Instant::now();
    // Try to reuse a stashed subtree from the previous turn. Match by
    // (step, state_hash) against the EnemyTurn subtree's MyTurn grandchildren.
    let reuse_disabled = std::env::var("OW_NO_REUSE").is_ok();
    let target_hash = state_hash(state);
    let mut reused_root: Option<Box<Node>> = None;
    if !reuse_disabled {
        LAST_TREE.with(|cell| {
            let mut slot = cell.borrow_mut();
            if let Some((expected_step, enemy_node)) = slot.take() {
                if expected_step == state.step {
                    // enemy_node is an EnemyTurn node; its children are MyTurn
                    // candidates for various opp actions. Find one matching obs.
                    for (_a, child) in enemy_node.children.iter() {
                        if state_hash(&child.state) == target_hash {
                            // Found a match — clone the subtree as new root.
                            // (We can't move out of Box<Node> via iteration,
                            // so we'll do a second pass to take ownership.)
                            reused_root = Some(child.clone_box());
                            break;
                        }
                    }
                }
            }
        });
    }
    let reused = reused_root.is_some();
    let mut root = match reused_root {
        Some(r) => *r,
        None => Node {
            state: state.clone(),
            kind: NodeKind::MyTurn,
            visits: 0,
            total_value: 0.0,
            children: Vec::new(),
            candidates: Vec::new(),
            priors: Vec::new(),
            candidates_initialized: false,
        },
    };
    let seed = (state.step as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        ^ 0xdeadbeefcafebabe;
    let mut rng = XorRng(seed | 1);

    // Always re-enumerate the root with STRONG (ow2_plan-based) candidates
    // regardless of reuse — the root is the most-visited node and its
    // returned action is what we play. Without this, a reused root would
    // use FAST candidates inherited from when it was a non-root grandchild
    // (under OW_MCTS_NONROOT_FAST=1), and the bot would return fast-quality
    // actions.
    //
    // Alignment: for each new strong candidate, reuse the matching existing
    // child if any (preserves accumulated visits/value). Unmatched existing
    // children are APPENDED as extra options at the end — accumulated value
    // not thrown away, just deprioritized in PUCT priors (geometric decay
    // by index). MCTS can still pick them if their explored value beats the
    // freshly-seeded strong candidates.
    let k_alts: usize = std::env::var("OW_ROOT_ALTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let new_strong_cands = enumerate_alternatives_strong(state, me, k_alts);
    let mut existing_children: Vec<(Vec<Action>, Box<Node>)> =
        std::mem::take(&mut root.children);
    let mut final_cands: Vec<Vec<Action>> = Vec::new();
    let mut final_children: Vec<(Vec<Action>, Box<Node>)> = Vec::new();
    for cand in &new_strong_cands {
        let pos = existing_children
            .iter()
            .position(|(a, _)| actions_equal(a, cand));
        let child = if let Some(p) = pos {
            let (_, c) = existing_children.swap_remove(p);
            c
        } else {
            Box::new(Node {
                state: state.clone(),
                kind: NodeKind::EnemyTurn { my_action: cand.clone() },
                visits: 0,
                total_value: 0.0,
                children: Vec::new(),
                candidates: Vec::new(),
                priors: Vec::new(),
                candidates_initialized: false,
            })
        };
        final_cands.push(cand.clone());
        final_children.push((cand.clone(), child));
    }
    // Append leftover existing children (their actions weren't in the new
    // strong set — keep them as extra candidates so prior search effort
    // isn't lost).
    for (action, child) in existing_children {
        final_cands.push(action.clone());
        final_children.push((action, child));
    }
    let n = final_cands.len();
    root.candidates = final_cands;
    root.priors = (0..n).map(|i| rank_prior(i, n)).collect();
    root.children = final_children;
    root.candidates_initialized = true;
    let _ = reused; // suppress unused warning
    let deadline = Instant::now() + std::time::Duration::from_millis(budget_ms);
    let mut iters = 0u32;
    while Instant::now() < deadline {
        select_and_expand(&mut root, me, &mut rng);
        iters += 1;
        if iters > 100_000 {
            break;
        }
    }
    if std::env::var("OW_DEBUG").is_ok() {
        let mut child_info: Vec<String> = Vec::new();
        for (a, c) in &root.children {
            let avg = if c.visits > 0 { c.total_value / c.visits as f64 } else { 0.0 };
            let target_summary: Vec<String> = a.iter().take(3).map(|x| format!("(s={},t≈{:.2})", x.0, x.1)).collect();
            child_info.push(format!("v={}/avg={:.3} {}", c.visits, avg, target_summary.join(",")));
        }
        eprintln!(
            "[duck-mcts] step={} iters={} root_visits={} children={} | {}",
            state.step, iters, root.visits, root.children.len(),
            child_info.join(" || ")
        );
    }
    if root.children.is_empty() {
        return Vec::new();
    }
    // Always seed root child 0 with the greedy policy action. Pick a
    // non-greedy alternative ONLY if its avg value beats greedy's by a
    // clear margin AND has enough visits to be trusted. Otherwise the bot
    // oscillates between near-equal-value targets and never accumulates
    // enough ships at any one target to capture it.
    // Default margin 0.05 — empirically (see REPORT.md), MCTS with cached
    // ow2_plan and ow2_full rollouts at depth 8 beats ow2 4-2 across 6
    // games at this margin. Higher margins lock to ow2 baseline (3-3
    // mirror). Lower margins regressed before the speedup.
    let robust_margin: f64 = std::env::var("OW_MARGIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);
    let min_visits_to_override: u32 = std::env::var("OW_MIN_OVERRIDE_VISITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let greedy_avg = {
        let c = &root.children[0].1;
        if c.visits > 0 {
            c.total_value / c.visits as f64
        } else {
            f64::NEG_INFINITY
        }
    };
    let mut best_i = 0usize;
    let mut best_val = greedy_avg;
    for (i, (_, c)) in root.children.iter().enumerate().skip(1) {
        if c.visits < min_visits_to_override {
            continue;
        }
        let avg = c.total_value / c.visits as f64;
        if avg > best_val + robust_margin {
            best_val = avg;
            best_i = i;
        }
    }
    // Stash the EnemyTurn subtree under the chosen action for next turn's
    // reuse. Its MyTurn grandchildren are the candidate next roots.
    if !reuse_disabled {
        let next_step = state.step + 1;
        let enemy_subtree = root.children[best_i].1.clone_box();
        LAST_TREE.with(|cell| {
            *cell.borrow_mut() = Some((next_step, enemy_subtree));
        });
    }
    if profile_enabled() {
        prof_report(prof_start.elapsed().as_secs_f64() * 1000.0);
    }
    root.children[best_i].0.clone()
}
