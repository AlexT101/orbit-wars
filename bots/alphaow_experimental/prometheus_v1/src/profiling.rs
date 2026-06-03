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

pub fn reset() {
    for c in [
        &FOCUSED_CANDIDATES_NS, &FOCUSED_CANDIDATES_CALLS,
        &PLAN_FOR_TARGET_NS, &PLAN_FOR_TARGET_CALLS,
        &APPLY_LAUNCHES_NS,
        &TICK_NS, &TICK_CALLS,
        &VALUE_NET_NS, &VALUE_NET_CALLS,
        &EXTRAPOLATE_NS, &EXTRAPOLATE_CALLS,
        &TURN_TOTAL_NS,
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

    let fc_ns = load(&FOCUSED_CANDIDATES_NS);
    let fc_calls = load(&FOCUSED_CANDIDATES_CALLS);
    let pft_ns = load(&PLAN_FOR_TARGET_NS);
    let pft_calls = load(&PLAN_FOR_TARGET_CALLS);
    let al_ns = load(&APPLY_LAUNCHES_NS);
    let tick_ns = load(&TICK_NS);
    let tick_calls = load(&TICK_CALLS);
    let vn_ns = load(&VALUE_NET_NS);
    let vn_calls = load(&VALUE_NET_CALLS);
    let ex_ns = load(&EXTRAPOLATE_NS);
    let ex_calls = load(&EXTRAPOLATE_CALLS);

    let accounted = fc_ns + al_ns + tick_ns + vn_ns + ex_ns;
    let other = total.saturating_sub(accounted);

    eprintln!(
        "[prof p{} step={}] total={:.1}ms | focused={:.1}ms({:.0}%, calls={}, plan_for_target={:.1}ms in {} calls) | tick={:.1}ms({:.0}%, {} calls) | apply_launches={:.1}ms({:.0}%) | value_net={:.1}ms({:.0}%, {} calls) | extrapolate={:.1}ms({:.0}%, {} calls) | other/duct={:.1}ms({:.0}%)",
        player, step, total_ms,
        ms(fc_ns), pct(fc_ns), fc_calls, ms(pft_ns), pft_calls,
        ms(tick_ns), pct(tick_ns), tick_calls,
        ms(al_ns), pct(al_ns),
        ms(vn_ns), pct(vn_ns), vn_calls,
        ms(ex_ns), pct(ex_ns), ex_calls,
        ms(other), pct(other),
    );
}
