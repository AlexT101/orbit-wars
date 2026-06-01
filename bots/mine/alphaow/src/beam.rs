//! Beam search planner with minimax-style backup.
//!
//! Per turn:
//!   1. Enumerate root candidates for me (`my_root`) and opp (`opp_root`)
//!      via the same apollo bridge used by DUCT.
//!   2. For every joint (my_root_i, opp_root_j) pair: apply both moves,
//!      tick one engine step, score with the XGB value net. Each beam
//!      trace remembers WHICH root pair it descended from.
//!   3. Beam prune to top `width` traces by score.
//!   4. For each subsequent depth (up to `max_depth`, time-bounded):
//!      expand every beam trace by every (my, opp) joint action drawn
//!      from non-root apollo candidates. The trace's root pair is
//!      preserved as it deepens.
//!   5. Propagate the best leaf score reached by each trace back to its
//!      root pair: `pair_best[my_i][opp_j]` is the max score I can
//!      achieve given I committed to `my_i` and opp committed to `opp_j`.
//!   6. **Minimax pick**: root my-action value =
//!      `min over opp_j of pair_best[my_i][opp_j]` — opp gets the worst
//!      reply for me. I pick the my-action with the highest such value.
//!
//! This is the pessimistic correction to the previous max-over-all-leaves
//! backup, which was optimistic (assumed opp let me reach the best
//! position). Now opp picks the response that's worst for me, mirroring
//! DUCT's joint-action averaging shape.
//!
//! Tunables (env vars):
//!   * `OW_BEAM_WIDTH` (default 4)
//!   * `OW_BEAM_DEPTH` (default 200)

use std::time::{Duration, Instant};

use crate::duct;
use crate::sim::{alive_players, apply_launches, tick, Action};
use crate::GameState;

const TERMINAL_STEP: i64 = 500;

fn beam_width() -> usize {
    std::env::var("OW_BEAM_WIDTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

fn max_depth() -> usize {
    std::env::var("OW_BEAM_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200)
}

struct Node {
    state: GameState,
    root_my_idx: usize,
    root_opp_idx: usize,
    score: f64,
}

pub fn best_move(state: &GameState, me: i32, budget_ms: u64) -> Vec<Action> {
    if state.step >= TERMINAL_STEP || alive_players(state) <= 1 {
        return Vec::new();
    }
    let opp = duct::dominant_enemy(state, me).unwrap_or(1 - me);
    let my_root = duct::enumerate_alternatives(state, me, duct::k_root(), true);
    let opp_root = duct::enumerate_alternatives(state, opp, duct::k_root(), true);
    if my_root.is_empty() {
        return Vec::new();
    }
    if opp_root.is_empty() {
        // Opp has no responses; just pick the first my-action.
        return my_root.into_iter().next().unwrap_or_default();
    }

    let n_my = my_root.len();
    let n_opp = opp_root.len();

    // Per-root-pair best-reachable score. Defaults to -inf so an
    // unreachable pair acts like "opp gets a free win" under MIN backup.
    let mut pair_best: Vec<Vec<f64>> = vec![vec![f64::NEG_INFINITY; n_opp]; n_my];
    let mut beam: Vec<Node> = Vec::new();

    let deadline = Instant::now() + Duration::from_millis(budget_ms);
    let width = beam_width();
    let depth_max = max_depth();

    // --- Depth 1: every (my_root, opp_root) pair.
    for (my_i, my_a) in my_root.iter().enumerate() {
        for (opp_i, opp_a) in opp_root.iter().enumerate() {
            if Instant::now() >= deadline {
                break;
            }
            let mut s = state.clone();
            apply_launches(&mut s, my_a);
            apply_launches(&mut s, opp_a);
            tick(&mut s);
            let score = duct::evaluate(&s, me);
            if score > pair_best[my_i][opp_i] {
                pair_best[my_i][opp_i] = score;
            }
            beam.push(Node {
                state: s,
                root_my_idx: my_i,
                root_opp_idx: opp_i,
                score,
            });
        }
    }
    beam.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    beam.truncate(width);

    // --- Depths 2..max: expand each beam trace.
    let mut depth = 2usize;
    while depth <= depth_max && Instant::now() < deadline && !beam.is_empty() {
        let mut next: Vec<Node> = Vec::new();
        for node in &beam {
            if Instant::now() >= deadline {
                break;
            }
            if node.state.step >= TERMINAL_STEP || alive_players(&node.state) <= 1 {
                if node.score > pair_best[node.root_my_idx][node.root_opp_idx] {
                    pair_best[node.root_my_idx][node.root_opp_idx] = node.score;
                }
                continue;
            }
            let my_alts = duct::enumerate_alternatives(&node.state, me, duct::k_non_root(), false);
            let opp_alts = duct::enumerate_alternatives(&node.state, opp, duct::k_non_root(), false);
            for my_a in &my_alts {
                for opp_a in &opp_alts {
                    let mut s = node.state.clone();
                    apply_launches(&mut s, my_a);
                    apply_launches(&mut s, opp_a);
                    tick(&mut s);
                    let score = duct::evaluate(&s, me);
                    if score > pair_best[node.root_my_idx][node.root_opp_idx] {
                        pair_best[node.root_my_idx][node.root_opp_idx] = score;
                    }
                    next.push(Node {
                        state: s,
                        root_my_idx: node.root_my_idx,
                        root_opp_idx: node.root_opp_idx,
                        score,
                    });
                }
            }
        }
        if next.is_empty() {
            break;
        }
        next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        next.truncate(width);
        beam = next;
        depth += 1;
    }

    // --- Minimax pick: opp chooses the response that minimises my best.
    let mut best_my_i = 0usize;
    let mut best_my_val = f64::NEG_INFINITY;
    for my_i in 0..n_my {
        let min_over_opp = pair_best[my_i]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if min_over_opp > best_my_val {
            best_my_val = min_over_opp;
            best_my_i = my_i;
        }
    }

    if std::env::var("OW_BEAM_DEBUG").ok().as_deref() == Some("1") {
        eprintln!(
            "[beam p{}] step={} depth={} width={} my_root={} opp_root={} chose={} min_value={:.4}",
            me, state.step, depth - 1, width, n_my, n_opp, best_my_i, best_my_val
        );
    }
    my_root.into_iter().nth(best_my_i).unwrap_or_default()
}
