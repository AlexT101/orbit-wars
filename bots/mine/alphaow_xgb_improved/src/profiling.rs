//! Lightweight per-turn timing accumulators for alphaow_ow profiling.
//! Each hot path adds its elapsed nanoseconds to an atomic counter.
//! `reset()` zeros all counters at the start of a turn; `dump()` prints
//! the breakdown to stderr at the end.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub static FOCUSED_CANDIDATES_NS: AtomicU64 = AtomicU64::new(0);
pub static FOCUSED_CANDIDATES_CALLS: AtomicU64 = AtomicU64::new(0);
pub static PLAN_FOR_TARGET_NS: AtomicU64 = AtomicU64::new(0);
pub static PLAN_FOR_TARGET_CALLS: AtomicU64 = AtomicU64::new(0);
pub static APPLY_LAUNCHES_NS: AtomicU64 = AtomicU64::new(0);
pub static TICK_NS: AtomicU64 = AtomicU64::new(0);
pub static TICK_CALLS: AtomicU64 = AtomicU64::new(0);
pub static VALUE_NET_NS: AtomicU64 = AtomicU64::new(0);
pub static VALUE_NET_CALLS: AtomicU64 = AtomicU64::new(0);
pub static EXTRAPOLATE_NS: AtomicU64 = AtomicU64::new(0);
pub static EXTRAPOLATE_CALLS: AtomicU64 = AtomicU64::new(0);
pub static TURN_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
// DUCT-internal breakdown.
pub static ENSURE_CANDIDATES_NS: AtomicU64 = AtomicU64::new(0);
pub static ENSURE_CANDIDATES_CALLS: AtomicU64 = AtomicU64::new(0);
pub static APOLLO_CANDIDATES_NS: AtomicU64 = AtomicU64::new(0);
pub static APOLLO_CANDIDATES_CALLS: AtomicU64 = AtomicU64::new(0);
pub static SELECTION_NS: AtomicU64 = AtomicU64::new(0);
pub static SELECTION_CALLS: AtomicU64 = AtomicU64::new(0);
pub static TREE_OPS_NS: AtomicU64 = AtomicU64::new(0);
pub static TREE_OPS_CALLS: AtomicU64 = AtomicU64::new(0);
pub static BACKPROP_NS: AtomicU64 = AtomicU64::new(0);
pub static BACKPROP_CALLS: AtomicU64 = AtomicU64::new(0);
pub static ITERATIONS: AtomicU64 = AtomicU64::new(0);

pub fn reset() {
    for c in [
        &FOCUSED_CANDIDATES_NS, &FOCUSED_CANDIDATES_CALLS,
        &PLAN_FOR_TARGET_NS, &PLAN_FOR_TARGET_CALLS,
        &APPLY_LAUNCHES_NS,
        &TICK_NS, &TICK_CALLS,
        &VALUE_NET_NS, &VALUE_NET_CALLS,
        &EXTRAPOLATE_NS, &EXTRAPOLATE_CALLS,
        &TURN_TOTAL_NS,
        &ENSURE_CANDIDATES_NS, &ENSURE_CANDIDATES_CALLS,
        &APOLLO_CANDIDATES_NS, &APOLLO_CANDIDATES_CALLS,
        &SELECTION_NS, &SELECTION_CALLS,
        &TREE_OPS_NS, &TREE_OPS_CALLS,
        &BACKPROP_NS, &BACKPROP_CALLS,
        &ITERATIONS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

#[inline]
pub fn add(counter: &AtomicU64, t0: Instant) {
    counter.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

#[inline]
pub fn inc(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

pub fn dump(step: i64, player: i32) {
    let load = |c: &AtomicU64| c.load(Ordering::Relaxed);
    let ms = |ns: u64| ns as f64 / 1_000_000.0;
    let total = load(&TURN_TOTAL_NS);
    let total_ms = ms(total);
    let pct = |ns: u64| if total > 0 { 100.0 * ns as f64 / total as f64 } else { 0.0 };

    let fc = (load(&FOCUSED_CANDIDATES_NS), load(&FOCUSED_CANDIDATES_CALLS));
    let pft = (load(&PLAN_FOR_TARGET_NS), load(&PLAN_FOR_TARGET_CALLS));
    let al = load(&APPLY_LAUNCHES_NS);
    let tick = (load(&TICK_NS), load(&TICK_CALLS));
    let vn = (load(&VALUE_NET_NS), load(&VALUE_NET_CALLS));
    let ex = (load(&EXTRAPOLATE_NS), load(&EXTRAPOLATE_CALLS));
    let ec = (load(&ENSURE_CANDIDATES_NS), load(&ENSURE_CANDIDATES_CALLS));
    let ac = (load(&APOLLO_CANDIDATES_NS), load(&APOLLO_CANDIDATES_CALLS));
    let sel = (load(&SELECTION_NS), load(&SELECTION_CALLS));
    let tr = (load(&TREE_OPS_NS), load(&TREE_OPS_CALLS));
    let bp = (load(&BACKPROP_NS), load(&BACKPROP_CALLS));
    let iters = load(&ITERATIONS);

    let accounted = ec.0 + sel.0 + tr.0 + bp.0 + al + tick.0 + vn.0 + ex.0;
    let other = total.saturating_sub(accounted);

    eprintln!(
        "[prof p{} step={} iters={}] total={:.1}ms  \
         ensure_cands={:.1}ms({:.1}%, n={})  \
         apollo_cands={:.1}ms({:.1}%, n={})  \
         focused_cands={:.1}ms({:.1}%, n={})  \
         plan_for_target={:.1}ms({:.1}%, n={})  \
         selection={:.1}ms({:.1}%, n={})  \
         tree_ops={:.1}ms({:.1}%, n={})  \
         apply_launches={:.1}ms({:.1}%)  \
         tick={:.1}ms({:.1}%, n={})  \
         value_net={:.1}ms({:.1}%, n={})  \
         extrapolate={:.1}ms({:.1}%, n={})  \
         backprop={:.1}ms({:.1}%, n={})  \
         other={:.1}ms({:.1}%)",
        player, step, iters, total_ms,
        ms(ec.0), pct(ec.0), ec.1,
        ms(ac.0), pct(ac.0), ac.1,
        ms(fc.0), pct(fc.0), fc.1,
        ms(pft.0), pct(pft.0), pft.1,
        ms(sel.0), pct(sel.0), sel.1,
        ms(tr.0), pct(tr.0), tr.1,
        ms(al), pct(al),
        ms(tick.0), pct(tick.0), tick.1,
        ms(vn.0), pct(vn.0), vn.1,
        ms(ex.0), pct(ex.0), ex.1,
        ms(bp.0), pct(bp.0), bp.1,
        ms(other), pct(other),
    );
}
