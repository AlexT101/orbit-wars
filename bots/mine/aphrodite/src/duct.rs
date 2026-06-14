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

use crate::sim::{alive_players, apply_launches, tick, Action};
use crate::{ow2_plan, GameState};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::OnceLock;
use std::time::Instant;

const EXPLORATION: f64 = 0.3;
const TERMINAL_STEP: i64 = 500;
const K_ROOT_DEFAULT: usize = 5;
const K_NON_ROOT_DEFAULT: usize = 4;

// ── overage-time budgeting ────────────────────────────────────────────────
// When enabled (a nonzero `overage_remaining_ms` is passed into `best_move`, DUCT may
// keep searching past the per-turn base budget by dipping into the engine's
// shared overage pool, but ONLY on turns where extra computation can plausibly
// change the chosen move:
//   * the position is still CONTESTED (no player dominates), and
//   * the root decision is still CLOSE (top-two candidate values nearly tied).
// Spend is bounded three ways: a per-turn hard cap, a safety reserve that must
// remain in the pool, and small chunks so the decision gap is re-checked
// between them (we stop early once one move clearly separates). All in ms.

/// Hard cap on overage spent beyond the base budget on any single turn.
const OVERAGE_PER_TURN_CAP_MS: u64 = 2000;
/// Safety reserve to keep untouched in the engine's overage pool, computed as
/// a base amount plus a per-turn multiplier
const OVERAGE_SAFETY_BASE_MS: f64 = 2000.0;
const OVERAGE_SAFETY_PER_TURN_MS: f64 = 50.0;
/// Grant overage in chunks this size, re-checking the decision gap between each.
const OVERAGE_CHUNK_MS: u64 = 200;
/// The root decision counts as "close" (worth more search) while the gap
/// between the best and second-best candidate average values is below this.
/// Values are clamped predictions in [-1, 1].
const OVERAGE_CLOSE_GAP: f64 = 0.05;
/// The position counts as "decided" (not worth extra search) once any single
/// player controls at least this share of total ship strength.
const OVERAGE_DECIDED_SHARE: f64 = 0.70;

thread_local! {
    /// Stash the root after each turn, so next turn's matching state can
    /// reuse the joint subtree corresponding to (my_chosen, opp_actual).
    static LAST_TREE: RefCell<Option<(i64, Box<Node>)>> = RefCell::new(None);

    /// Persistent apollo `EntityCache`, shared across every candidate-generation
    /// call, every rollout, and every real turn of a game. The cache holds no
    /// strategy/player-specific state — only game-static orbiter geometry and an
    /// owner-agnostic aim cache keyed by absolute launch turn — so a single
    /// instance is correct everywhere. Paired with a `GeometryKey` so a process
    /// that serves multiple games (benchmarks, harness reuse) rebuilds on a new
    /// map instead of reusing stale geometry. Mirrors apollo's `Bot.cache`
    /// (`bots/mine/apollo/src/lib.rs`).
    static CACHE: RefCell<Option<(GeometryKey, crate::apollo::cache::EntityCache)>> =
        RefCell::new(None);

    /// Search-scoped hot L1 aim cache, shared across all value-net leaf evals in a
    /// turn (cleared per turn in `refresh_cache`). The value net re-queries the
    /// same planet-pair pressures across thousands of leaves; this RefCell fronts
    /// the `Mutex`-locked L2 `EntityCache::aim_cache`. Per-thread, so no
    /// cross-thread contention. Keyed by `(src,dst,ships,abs_launch)` — entries
    /// for different node steps coexist safely.
    static EVAL_L1: crate::apollo::world::ShotL1 = RefCell::new(Default::default());
}

/// Fingerprint of a game's fixed geometry: angular velocity plus the static
/// orbiter layout (id + initial radius/angle of every non-comet planet). Two
/// observations of the same game share this; a new game (different map or
/// angular velocity) does not, triggering a cache rebuild.
#[derive(Clone, PartialEq)]
struct GeometryKey {
    av_bits: u64,
    planets: Vec<(i64, u64, u64)>,
}

fn geometry_key(state: &GameState) -> GeometryKey {
    let mut planets: Vec<(i64, u64, u64)> = state
        .planets
        .iter()
        .filter(|p| !p.is_comet)
        .map(|p| (p.id, p.orbital_radius.to_bits(), p.initial_angle.to_bits()))
        .collect();
    planets.sort_unstable_by_key(|t| t.0);
    GeometryKey {
        av_bits: state.angular_velocity.to_bits(),
        planets,
    }
}

/// Rebuild-if-needed + per-turn refresh of the persistent [`CACHE`],
/// mirroring apollo's `Bot::refresh_cache`: build once per game, refresh comets
/// only on a spawn step, then set the current turn and drop the now-unqueryable
/// prior turn's aim slot. Run once at the top of [`best_move`].
fn refresh_cache(state: &GameState) {
    use crate::apollo::constants::COMET_SPAWN_STEPS;
    let key = geometry_key(state);
    CACHE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_build = match slot.as_ref() {
            Some((k, _)) => *k != key,
            None => true,
        };
        if needs_build {
            *slot = Some((key, crate::apollo_bridge::rollout_cache(state)));
        } else if COMET_SPAWN_STEPS.contains(&state.step) {
            if let Some((_, cache)) = slot.as_mut() {
                crate::apollo_bridge::refresh_cache_comets(cache, state);
            }
        }
        if let Some((_, cache)) = slot.as_mut() {
            cache.set_current_turn(state.step);
            cache.clear_aim_cache_slot(state.step - 1);
        }
    });
    // Drop last turn's value-net L1 entries (bounds memory; L2/L3 persist).
    EVAL_L1.with(|l1| l1.borrow_mut().clear());
}

/// Run `f` with the shared entity cache's `current_turn` set to `turn`. Used by
/// candidate generation (one call per node, whose step may differ from the real
/// turn). The cache is interior-mutable for its aim table, so `f` takes `&_`.
fn with_cache_at<R>(turn: i64, f: impl FnOnce(&crate::apollo::cache::EntityCache) -> R) -> R {
    CACHE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let (_, cache) = slot.as_mut().expect("entity cache built in best_move");
        cache.set_current_turn(turn);
        f(cache)
    })
}

// ── search instrumentation (optional, env-gated) ──────────────────────────
// Two independent dumps, both off unless their env var names a file:
//   APHRODITE_DUMP_LEAVES_PATH      — one binary record per value-net leaf eval:
//                                     search_step:i32, leaf_step:i32,
//                                     summary_v2:[f32; DIM]. Lets a probe measure
//                                     each feature's *within-search* variance (its
//                                     ability to rank sibling leaves) vs global
//                                     variance.
//   APHRODITE_DUMP_TREE_STATS_PATH  — one CSV row per real turn with the DUCT
//                                     tree shape (nodes/leaves/depth/iters).
// `APHRODITE_DUMP_LEAVES_MAX_PER_SEARCH` caps leaves dumped per search (0 = all).
thread_local! {
    static LEAF_DUMP: RefCell<Option<BufWriter<File>>> =
        RefCell::new(open_env_file("APHRODITE_DUMP_LEAVES_PATH").map(BufWriter::new));
    static TREE_STATS: RefCell<Option<File>> = RefCell::new(open_tree_stats());
    /// Root step of the in-progress search, tagged onto each leaf record so the
    /// probe can group leaves by the search they belong to.
    static SEARCH_STEP: std::cell::Cell<i64> = std::cell::Cell::new(0);
    static LEAVES_THIS_SEARCH: std::cell::Cell<u64> = std::cell::Cell::new(0);
    /// Monotonic per-search id (bumped each `best_move`), used as the v3 leaf-dump
    /// cohort key so searches stay separable even when root steps collide across
    /// probe positions from different games.
    static SEARCH_SEQ: std::cell::Cell<i32> = std::cell::Cell::new(0);
    /// `APHRODITE_DUMP_FEATURES=v3` makes the leaf dump emit the 145-d
    /// `summary_v3` (4p probe) instead of the 65-d `summary_v2`.
    static DUMP_V3: bool =
        std::env::var("APHRODITE_DUMP_FEATURES").map(|v| v == "v3").unwrap_or(false);
}

fn open_env_file(var: &str) -> Option<File> {
    let p = std::env::var(var).ok()?;
    match File::create(&p) {
        Ok(f) => {
            eprintln!("[aphrodite] {} -> {}", var, p);
            Some(f)
        }
        Err(e) => {
            eprintln!("[aphrodite] could not create {} at {}: {}", var, p, e);
            None
        }
    }
}

fn open_tree_stats() -> Option<File> {
    let mut f = open_env_file("APHRODITE_DUMP_TREE_STATS_PATH")?;
    let _ = writeln!(
        f,
        "step,iters,root_visits,my_K,opp_K,nodes,leaves,max_depth"
    );
    Some(f)
}

fn leaf_dump_cap() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("APHRODITE_DUMP_LEAVES_MAX_PER_SEARCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// Append one leaf's summary-v2 feature row to the leaf dump, if enabled. Called
/// at every value-net leaf evaluation. Cheap no-op (one thread-local borrow) when
/// the dump is off.
fn maybe_dump_leaf(state: &GameState, me: i32) {
    LEAF_DUMP.with(|cell| {
        let mut slot = cell.borrow_mut();
        let w = match slot.as_mut() {
            Some(w) => w,
            None => return,
        };
        let cap = leaf_dump_cap();
        if cap != 0 {
            let n = LEAVES_THIS_SEARCH.with(|c| c.get());
            if n >= cap {
                return;
            }
            LEAVES_THIS_SEARCH.with(|c| c.set(n + 1));
        }
        let leaf_step = state.step as i32;
        if DUMP_V3.with(|v| *v) {
            // 4p probe: cohort key = monotonic search id; feats = 145-d summary_v3.
            let search_id = SEARCH_SEQ.with(|c| c.get());
            let v3 = with_cache_at(state.step, |cache| {
                EVAL_L1.with(|l1| {
                    crate::value_net::summary_features_v3::extract_with_cache(
                        state,
                        me,
                        cache,
                        Some(l1),
                    )
                })
            });
            let _ = w.write_all(&search_id.to_le_bytes());
            let _ = w.write_all(&leaf_step.to_le_bytes());
            let bytes =
                unsafe { std::slice::from_raw_parts(v3.as_ptr() as *const u8, v3.len() * 4) };
            let _ = w.write_all(bytes);
        } else {
            let search_step = SEARCH_STEP.with(|c| c.get()) as i32;
            let v2 = with_cache_at(state.step, |cache| {
                crate::value_net::summary_features_v2::extract_with_cache(state, me, cache)
            });
            let _ = w.write_all(&search_step.to_le_bytes());
            let _ = w.write_all(&leaf_step.to_le_bytes());
            let bytes =
                unsafe { std::slice::from_raw_parts(v2.as_ptr() as *const u8, v2.len() * 4) };
            let _ = w.write_all(bytes);
        }
    });
}

fn count_nodes(n: &Node) -> usize {
    1 + n.children.values().map(|c| count_nodes(c)).sum::<usize>()
}

fn count_leaves(n: &Node) -> usize {
    if n.children.is_empty() {
        1
    } else {
        n.children.values().map(|c| count_leaves(c)).sum()
    }
}

fn max_depth(n: &Node) -> usize {
    if n.children.is_empty() {
        1
    } else {
        1 + n.children.values().map(|c| max_depth(c)).max().unwrap_or(0)
    }
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
    /// Assumed launches for the non-branched minor players (4p only): each
    /// alive player other than `me`/`opp` contributes their single apollo
    /// `ScorePerShip` greedy plan. A pure function of `state` (every player
    /// commits privately from the same observed node), so it is computed once
    /// per node in `ensure_candidates` and replayed at every expansion. Empty
    /// in 2p — that path is then identical to before.
    other_launches: Vec<Action>,
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
    let greedy = ow2_plan::plan(state, player, false);
    let mut out: Vec<Vec<Action>> = vec![greedy];
    for tgt in &state.planets {
        if tgt.owner == player {
            continue;
        }
        if out.len() >= k {
            break;
        }
        let alt = ow2_plan::plan_with_exclusion(state, player, false, Some(tgt.id));
        if !out.iter().any(|a| actions_equal(a, &alt)) {
            out.push(alt);
        }
    }
    if out.len() < k && !out.iter().any(|a| a.is_empty()) {
        out.push(Vec::new());
    }
    out
}

fn enumerate_alternatives(
    state: &GameState,
    player: i32,
    k: usize,
    rollout_internal: bool,
) -> Vec<Vec<Action>> {
    let __apollo_t0 = std::time::Instant::now();
    let mut alts = with_cache_at(state.step, |cache| {
        crate::apollo_bridge::apollo_candidates(state, player, cache, rollout_internal)
    });
    crate::profiling::add(&crate::profiling::APOLLO_CANDIDATES_NS, __apollo_t0);
    crate::profiling::inc(&crate::profiling::APOLLO_CANDIDATES_CALLS);
    if !alts.is_empty() {
        alts.truncate(k);
        return alts;
    }
    enumerate_alternatives_strong(state, player, k)
}

/// Candidate sets for both players at `state`. Fast path: when apollo candidates
/// are the active generator for both (the default — no focused override), build
/// the player-agnostic `Simulator` + arrival ledger once and derive both sets
/// from it via [`crate::apollo_bridge::apollo_candidates_pair`], so the
/// `HORIZON`-turn ledger walk is paid once instead of per player. Falls back to
/// the per-player [`enumerate_alternatives`] when focused candidates are enabled
/// or apollo yields nothing for a side.
///
/// `rollout_internal` is forwarded to apollo so the early-game opening DFS only
/// runs at the genuine root node and stands down at every non-root expansion
/// (the DUCT analog of apollo suppressing it inside rollouts).
fn enumerate_pair(
    state: &GameState,
    me: i32,
    opp: i32,
    k: usize,
    rollout_internal: bool,
) -> (Vec<Vec<Action>>, Vec<Vec<Action>>) {
    let (mut my, mut op) = with_cache_at(state.step, |cache| {
        crate::apollo_bridge::apollo_candidates_pair(state, me, opp, cache, rollout_internal)
    });
    if !my.is_empty() && !op.is_empty() {
        my.truncate(k);
        op.truncate(k);
        return (my, op);
    }
    // One side empty: fall back to per-player generation so the ow2 fallback can
    // fill the missing side.
    (
        enumerate_alternatives(state, me, k, rollout_internal),
        enumerate_alternatives(state, opp, k, rollout_internal),
    )
}

/// Splice externally supplied single-launch candidates (the chaos IL policy's
/// top-k moves) into the root's MY candidate set. Entries are appended, never
/// interleaved in place: `children` is keyed by candidate index, so existing
/// entries — including a reused subtree's — must keep their positions. The
/// interleave instead happens in the prior vector: weights decay by sqrt(0.5)
/// per virtual slot (apollo#0, il#0, apollo#1, il#1, …), which preserves the
/// apollo candidates' existing 0.5-per-rank ratios exactly while giving il#j
/// the geometric mean of apollo#j and apollo#j+1.
fn inject_root_candidates(node: &mut Node, extra: &[Action]) -> usize {
    let base_n = node.my_candidates.len();
    for &a in extra {
        let plan = vec![a];
        if node.my_candidates.iter().any(|c| actions_equal(c, &plan)) {
            continue;
        }
        node.my_candidates.push(plan);
        node.my_stats.push(ActionStats {
            visits: 0,
            sum_value: 0.0,
        });
    }
    let added = node.my_candidates.len() - base_n;
    if added == 0 {
        return 0;
    }
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let mut raw: Vec<f64> = Vec::with_capacity(node.my_candidates.len());
    for r in 0..base_n {
        raw.push(half.powi((2 * r) as i32));
    }
    for j in 0..added {
        raw.push(half.powi((2 * j + 1) as i32));
    }
    let z: f64 = raw.iter().sum();
    node.my_priors = raw.into_iter().map(|w| w / z).collect();
    added
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

/// Alive players (planet/fleet owners) other than `me` and `opp`, sorted for
/// determinism. In a 2p game this is empty. These are the minor players whose
/// replies DUCT does not branch over but instead fixes to a single greedy plan.
fn other_players(state: &GameState, me: i32, opp: i32) -> Vec<i32> {
    let mut seen: u32 = 0;
    let mut note = |p: i32| {
        if p >= 0 && p < 32 && p != me && p != opp {
            seen |= 1 << p;
        }
    };
    for p in &state.planets {
        note(p.owner);
    }
    for f in &state.fleets {
        note(f.owner);
    }
    (0..32).filter(|i| seen & (1 << i) != 0).collect()
}

pub(crate) fn dominant_enemy(state: &GameState, me: i32) -> Option<i32> {
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
    EXPLORATION
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
    let __ec_t0 = std::time::Instant::now();
    let k = if root {
        K_ROOT_DEFAULT
    } else {
        K_NON_ROOT_DEFAULT
    };
    let opp = dominant_enemy(&node.state, me).unwrap_or(1 - me);
    // Only the genuine root expansion runs apollo's early-game opening DFS;
    // every non-root node suppresses it (rollout_internal), mirroring apollo's
    // in-rollout reply policy.
    let (my_alts, opp_alts) = enumerate_pair(&node.state, me, opp, k, !root);
    // Fix each minor player's reply to their single apollo greedy plan, computed
    // once here from the node state (empty in 2p). Replayed at every expansion.
    node.other_launches = {
        let others = other_players(&node.state, me, opp);
        let mut launches: Vec<Action> = Vec::new();
        if !others.is_empty() {
            with_cache_at(node.state.step, |cache| {
                for p in others {
                    launches.extend(crate::apollo_bridge::apollo_greedy(&node.state, p, cache));
                }
            });
        }
        launches
    };
    let my_n = my_alts.len();
    let opp_n = opp_alts.len();
    node.my_priors = (0..my_n).map(|i| rank_prior(i, my_n)).collect();
    node.opp_priors = (0..opp_n).map(|i| rank_prior(i, opp_n)).collect();
    node.my_stats = (0..my_n)
        .map(|_| ActionStats {
            visits: 0,
            sum_value: 0.0,
        })
        .collect();
    node.opp_stats = (0..opp_n)
        .map(|_| ActionStats {
            visits: 0,
            sum_value: 0.0,
        })
        .collect();
    node.my_candidates = my_alts;
    node.opp_candidates = opp_alts;
    node.candidates_initialized = true;
    crate::profiling::add(&crate::profiling::ENSURE_CANDIDATES_NS, __ec_t0);
    crate::profiling::inc(&crate::profiling::ENSURE_CANDIDATES_CALLS);
}

// ── profiling (OW_PROFILE) ───────────────────────────────────────────────
// Per-turn cumulative timing of leaf evaluation.
fn prof_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("OW_PROFILE").is_ok())
}
thread_local! {
    static PROF_EVAL_NS: std::cell::Cell<u64> = std::cell::Cell::new(0);
    static PROF_EVAL_N: std::cell::Cell<u64> = std::cell::Cell::new(0);
}
fn prof_reset() {
    PROF_EVAL_NS.with(|c| c.set(0));
    PROF_EVAL_N.with(|c| c.set(0));
}

pub(crate) fn evaluate(state: &GameState, me: i32) -> f64 {
    if prof_enabled() {
        let t = Instant::now();
        let v = evaluate_inner(state, me);
        let ns = t.elapsed().as_nanos() as u64;
        PROF_EVAL_NS.with(|c| c.set(c.get() + ns));
        PROF_EVAL_N.with(|c| c.set(c.get() + 1));
        return v;
    }
    evaluate_inner(state, me)
}

fn evaluate_inner(state: &GameState, me: i32) -> f64 {
    maybe_dump_leaf(state, me);
    let __vn_t0 = std::time::Instant::now();
    // Reuse the persistent per-search EntityCache (geometry/aim) for value-net
    // feature extraction instead of building one per leaf. `with_cache_at` sets
    // the cache's current turn to this leaf's step before scoring.
    let __pred = with_cache_at(state.step, |cache| {
        EVAL_L1.with(|l1| crate::value_net::predict_with_cache(state, me, cache, Some(l1)))
    });
    crate::profiling::add(&crate::profiling::VALUE_NET_NS, __vn_t0);
    crate::profiling::inc(&crate::profiling::VALUE_NET_CALLS);
    // The value net is always loaded in production; if weights are somehow
    // absent, fall back to a neutral score rather than a heuristic.
    __pred.map(|v| v.clamp(-1.0, 1.0)).unwrap_or(0.0)
}

fn select_and_expand(node: &mut Node, me: i32, is_root: bool) -> f64 {
    if node.state.step >= TERMINAL_STEP || alive_players(&node.state) <= 1 {
        let v = evaluate(&node.state, me);
        node.visits += 1;
        return v;
    }
    ensure_candidates(node, me, is_root);
    let __sel_t0 = std::time::Instant::now();
    let my_idx = select_my(node);
    let opp_idx = select_opp(node);
    crate::profiling::add(&crate::profiling::SELECTION_NS, __sel_t0);
    crate::profiling::inc(&crate::profiling::SELECTION_CALLS);
    let value: f64;
    if !node.children.contains_key(&(my_idx, opp_idx)) {
        let __tree_t0 = std::time::Instant::now();
        let mut s = node.state.clone();
        crate::profiling::add(&crate::profiling::TREE_OPS_NS, __tree_t0);
        crate::profiling::inc(&crate::profiling::TREE_OPS_CALLS);
        // Expand: apply both actions, tick, create new node, rollout.
        let __al_t0 = std::time::Instant::now();
        apply_launches(&mut s, &node.my_candidates[my_idx]);
        apply_launches(&mut s, &node.opp_candidates[opp_idx]);
        // Minor players' fixed greedy replies (empty in 2p).
        apply_launches(&mut s, &node.other_launches);
        crate::profiling::add(&crate::profiling::APPLY_LAUNCHES_NS, __al_t0);
        let __tick_t0 = std::time::Instant::now();
        tick(&mut s);
        crate::profiling::add(&crate::profiling::TICK_NS, __tick_t0);
        crate::profiling::inc(&crate::profiling::TICK_CALLS);
        let rollout_value = evaluate(&s, me);
        let __tree_t1 = std::time::Instant::now();
        let child = Node {
            state: s,
            visits: 1,
            my_candidates: Vec::new(),
            my_priors: Vec::new(),
            my_stats: Vec::new(),
            opp_candidates: Vec::new(),
            opp_priors: Vec::new(),
            opp_stats: Vec::new(),
            other_launches: Vec::new(),
            children: HashMap::new(),
            candidates_initialized: false,
        };
        node.children.insert((my_idx, opp_idx), Box::new(child));
        crate::profiling::add(&crate::profiling::TREE_OPS_NS, __tree_t1);
        value = rollout_value;
    } else {
        // Recurse.
        let child = node.children.get_mut(&(my_idx, opp_idx)).unwrap();
        value = select_and_expand(child, me, false);
    }
    // Backprop: update both marginal stats + joint node.
    let __bp_t0 = std::time::Instant::now();
    node.visits += 1;
    node.my_stats[my_idx].visits += 1;
    node.my_stats[my_idx].sum_value += value;
    node.opp_stats[opp_idx].visits += 1;
    node.opp_stats[opp_idx].sum_value += value;
    crate::profiling::add(&crate::profiling::BACKPROP_NS, __bp_t0);
    crate::profiling::inc(&crate::profiling::BACKPROP_CALLS);
    value
}

fn state_hash(state: &GameState) -> u64 {
    let mut h: u64 = state.step as u64;
    for p in &state.planets {
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(p.id as u64);
        h = h
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add((p.owner as i64 + 1) as u64);
        h = h
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(p.ships as u64);
    }
    for f in &state.fleets {
        h = h
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(f.from_planet_id as u64);
        h = h
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add((f.owner + 1) as u64);
        h = h
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(f.ships as u64);
    }
    h
}

/// Largest single-player share of total ship strength (garrisons + in-flight),
/// across all owned planets/fleets (neutrals excluded). 0.0 when nobody has
/// ships. Used to gate overage spend: a blowout is not worth extra search, and
/// this works for both 2p and 4p (it asks "does anyone dominate", not "is it
/// 50/50"). Cheap O(planets+fleets) scan.
fn max_player_ship_share(state: &GameState) -> f64 {
    let mut by: HashMap<i32, i64> = HashMap::new();
    let mut total = 0i64;
    for pl in &state.planets {
        if pl.owner >= 0 && pl.ships > 0 {
            *by.entry(pl.owner).or_insert(0) += pl.ships;
            total += pl.ships;
        }
    }
    for f in &state.fleets {
        if f.owner >= 0 && f.ships > 0 {
            *by.entry(f.owner).or_insert(0) += f.ships;
            total += f.ships;
        }
    }
    if total <= 0 {
        return 0.0;
    }
    by.values().copied().max().unwrap_or(0) as f64 / total as f64
}

/// Gap between the best and second-best candidate average values at the root,
/// over candidates that have been visited. Returns +inf when fewer than two
/// candidates have stats (treated as "not close" so we never extend on a
/// degenerate root). Used to decide whether overage search could still flip the
/// chosen move.
fn root_top2_gap(root: &Node) -> f64 {
    let mut avgs: Vec<f64> = root
        .my_stats
        .iter()
        .filter(|st| st.visits > 0)
        .map(|st| st.sum_value / st.visits as f64)
        .collect();
    if avgs.len() < 2 {
        return f64::INFINITY;
    }
    avgs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    avgs[0] - avgs[1]
}

/// `overage_remaining_ms` is the engine's `remainingOverageTime` (seconds,
/// converted to ms by the caller) for THIS turn, or 0.0 when overage use is
/// disabled (dev) — in which case the extension below is skipped entirely.
pub fn best_move(
    state: &GameState,
    me: i32,
    budget_ms: u64,
    overage_remaining_ms: f64,
    il_candidates: &[Action],
) -> Vec<Action> {
    // Build/refresh the persistent shared apollo cache before any candidate
    // generation or rollout reads it.
    refresh_cache(state);
    let reuse_disabled = false;
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
            other_launches: Vec::new(),
            children: HashMap::new(),
            candidates_initialized: false,
        },
    };
    ensure_candidates(&mut root, me, true);
    if !il_candidates.is_empty() {
        let added = inject_root_candidates(&mut root, il_candidates);
        if std::env::var("OW_DEBUG").is_ok() {
            eprintln!(
                "[chaos-il] step={} player={} il_offered={} il_added={} root_K={}",
                state.step,
                me,
                il_candidates.len(),
                added,
                root.my_candidates.len()
            );
        }
    }

    if prof_enabled() {
        prof_reset();
    }
    // Tag leaves dumped during this search with the root step, and reset the
    // per-search leaf cap counter.
    SEARCH_STEP.with(|c| c.set(state.step));
    SEARCH_SEQ.with(|c| c.set(c.get() + 1));
    LEAVES_THIS_SEARCH.with(|c| c.set(0));
    let deadline = Instant::now() + std::time::Duration::from_millis(budget_ms);
    let mut iters = 0u32;
    while Instant::now() < deadline {
        select_and_expand(&mut root, me, true);
        iters += 1;
        if iters > 100_000 {
            break;
        }
    }

    // ── overage extension ────────────────────────────────────────────────
    // Past the base budget, optionally keep searching by dipping into the
    // engine's overage pool — but only on a contested position with a still-
    // close root decision, bounded by a per-turn cap and a pool safety reserve.
    // Granted in chunks so the gap is re-checked (and search stops early once a
    // move separates). `overage_remaining_ms == 0.0` (disabled) skips all of it.
    let mut overage_used_ms: u64 = 0;
    let remaining_turns = (TERMINAL_STEP - state.step).max(0) as f64;
    let safety_buffer_ms = OVERAGE_SAFETY_BASE_MS + OVERAGE_SAFETY_PER_TURN_MS * remaining_turns;
    let mut overage_turn_cap_ms: u64 = 0;
    let mut overage_initial_gap = f64::INFINITY;
    let mut overage_final_gap = f64::INFINITY;
    if overage_remaining_ms > safety_buffer_ms {
        let available = overage_remaining_ms - safety_buffer_ms;
        let turn_cap = (available.floor() as u64).min(OVERAGE_PER_TURN_CAP_MS);
        overage_turn_cap_ms = turn_cap;
        let contested = max_player_ship_share(&root.state) < OVERAGE_DECIDED_SHARE;
        if turn_cap >= OVERAGE_CHUNK_MS && contested {
            overage_initial_gap = root_top2_gap(&root);
            while overage_used_ms + OVERAGE_CHUNK_MS <= turn_cap
                && root_top2_gap(&root) < OVERAGE_CLOSE_GAP
            {
                let ext_deadline =
                    Instant::now() + std::time::Duration::from_millis(OVERAGE_CHUNK_MS);
                while Instant::now() < ext_deadline {
                    select_and_expand(&mut root, me, true);
                    iters += 1;
                    if iters > 1_000_000 {
                        break;
                    }
                }
                overage_used_ms += OVERAGE_CHUNK_MS;
            }
            overage_final_gap = root_top2_gap(&root);
        }
    }
    if overage_used_ms > 0 {
        eprintln!(
            "[duck-overage] step={} player={} spent={}ms remaining={:.0}ms safety={:.0}ms cap={}ms gap={:.3}->{:.3} iters={} root_visits={}",
            state.step,
            me,
            overage_used_ms,
            overage_remaining_ms,
            safety_buffer_ms,
            overage_turn_cap_ms,
            overage_initial_gap,
            overage_final_gap,
            iters,
            root.visits
        );
    }

    crate::profiling::ITERATIONS.fetch_add(iters as u64, std::sync::atomic::Ordering::Relaxed);

    // Flush the leaf dump for this search and append a tree-shape row (both
    // no-ops unless their env var named a file).
    LEAF_DUMP.with(|cell| {
        if let Some(w) = cell.borrow_mut().as_mut() {
            let _ = w.flush();
        }
    });
    TREE_STATS.with(|cell| {
        if let Some(f) = cell.borrow_mut().as_mut() {
            let _ = writeln!(
                f,
                "{},{},{},{},{},{},{},{}",
                state.step,
                iters,
                root.visits,
                root.my_candidates.len(),
                root.opp_candidates.len(),
                count_nodes(&root),
                count_leaves(&root),
                max_depth(&root),
            );
            let _ = f.flush();
        }
    });

    if std::env::var("OW_DEBUG").is_ok() {
        // Walk the tree to measure unique node count + max depth.
        let nodes = count_nodes(&root);
        let depth = max_depth(&root);

        let mut child_info: Vec<String> = Vec::new();
        for i in 0..root.my_candidates.len() {
            let st = &root.my_stats[i];
            let avg = if st.visits > 0 {
                st.sum_value / st.visits as f64
            } else {
                0.0
            };
            let target_summary: Vec<String> = root.my_candidates[i]
                .iter()
                .take(3)
                .map(|x| format!("(s={},t≈{:.2})", x.0, x.1))
                .collect();
            child_info.push(format!(
                "v={}/avg={:.3} {}",
                st.visits,
                avg,
                target_summary.join(",")
            ));
        }
        eprintln!(
            "[duck] step={} iters={} overage_ms={} root_visits={} my_K={} nodes={} max_depth={} | {}",
            state.step,
            iters,
            overage_used_ms,
            root.visits,
            root.my_candidates.len(),
            nodes,
            depth,
            child_info.join(" || ")
        );
    }

    if prof_enabled() {
        let eval_ns = PROF_EVAL_NS.with(|c| c.get());
        let eval_n = PROF_EVAL_N.with(|c| c.get());
        let eval_ms = eval_ns as f64 / 1e6;
        let budget = budget_ms as f64;
        let pct = |x: f64| {
            if budget > 0.0 {
                x / budget * 100.0
            } else {
                0.0
            }
        };
        let per = |ns: u64, n: u64| {
            if n > 0 {
                ns as f64 / 1e3 / n as f64
            } else {
                0.0
            }
        };
        eprintln!(
            "[prof] step={} iters={} budget={}ms | leaf_eval={:.1}ms ({:.0}%, n={}, {:.2}µs/call); rest=tree+candidates",
            state.step, iters, budget_ms,
            eval_ms, pct(eval_ms), eval_n, per(eval_ns, eval_n),
        );
    }

    if root.my_candidates.is_empty() {
        return Vec::new();
    }
    // Pick by raw max in my marginal stats.
    let mut best_i = 0usize;
    let mut best_val = if root.my_stats[0].visits > 0 {
        root.my_stats[0].sum_value / root.my_stats[0].visits as f64
    } else {
        f64::NEG_INFINITY
    };
    for i in 1..root.my_candidates.len() {
        let st = &root.my_stats[i];
        if st.visits == 0 {
            continue;
        }
        let avg = st.sum_value / st.visits as f64;
        if avg > best_val {
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
