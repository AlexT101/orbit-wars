//! Port of ow2's strong `plan()` function to use as a baseline policy for
//! duck. Output is the joint set of moves for `state.player` derived from
//! ow2's arrivals-aware planner (greedy target loop + plan_for_time with
//! min/max ship binary search per source + cooperation between sources).

use crate::pathing;
use crate::pathing::PathResult;
use crate::sim::Action;
use crate::{GameState, Planet, CENTER_X, CENTER_Y};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// Profile counters (only used when OW_PROFILE_PLAN=1).
static PLAN_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static PLAN_CALLS: AtomicU64 = AtomicU64::new(0);
static PLAN_DIR_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static PLAN_DIR_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static PLAN_DIR_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static PLAN_SIMULATE_AT_NS: AtomicU64 = AtomicU64::new(0);
static PLAN_SIMULATE_AT_CALLS: AtomicU64 = AtomicU64::new(0);
static PLAN_OUTER_ITERS: AtomicU64 = AtomicU64::new(0);
static PLAN_FOR_TIME_CALLS: AtomicU64 = AtomicU64::new(0);

// Cached env-var lookup (the per-call check was costing 17M × 100ns = 1.7s
// in tight hot loops). Checked once at process start.
use std::sync::OnceLock;
static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
fn plan_profile_enabled() -> bool {
    *PROFILE_ENABLED.get_or_init(|| std::env::var("OW_PROFILE_PLAN").is_ok())
}

pub fn plan_profile_reset() {
    PLAN_TOTAL_NS.store(0, Ordering::Relaxed);
    PLAN_CALLS.store(0, Ordering::Relaxed);
    PLAN_DIR_CACHE_HITS.store(0, Ordering::Relaxed);
    PLAN_DIR_CACHE_MISSES.store(0, Ordering::Relaxed);
    PLAN_DIR_TOTAL_NS.store(0, Ordering::Relaxed);
    PLAN_SIMULATE_AT_NS.store(0, Ordering::Relaxed);
    PLAN_SIMULATE_AT_CALLS.store(0, Ordering::Relaxed);
    PLAN_OUTER_ITERS.store(0, Ordering::Relaxed);
    PLAN_FOR_TIME_CALLS.store(0, Ordering::Relaxed);
}

pub fn plan_profile_report() {
    let total_ms = PLAN_TOTAL_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let calls = PLAN_CALLS.load(Ordering::Relaxed);
    let hits = PLAN_DIR_CACHE_HITS.load(Ordering::Relaxed);
    let misses = PLAN_DIR_CACHE_MISSES.load(Ordering::Relaxed);
    let dir_ms = PLAN_DIR_TOTAL_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let sim_ms = PLAN_SIMULATE_AT_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let sim_calls = PLAN_SIMULATE_AT_CALLS.load(Ordering::Relaxed);
    let outer = PLAN_OUTER_ITERS.load(Ordering::Relaxed);
    let pft = PLAN_FOR_TIME_CALLS.load(Ordering::Relaxed);
    eprintln!(
        "[plan-profile] total={:.2}ms across {} calls (avg {:.3}ms/call)",
        total_ms,
        calls,
        total_ms / calls.max(1) as f64
    );
    eprintln!(
        "[plan-profile]   dir_to_hit total: {:.2}ms ({} hits + {} misses)",
        dir_ms, hits, misses
    );
    eprintln!(
        "[plan-profile]     hit avg {:.0}ns, miss avg {:.0}ns",
        if hits > 0 {
            (dir_ms * 1e6 - misses as f64 * 1000.0) / hits as f64
        } else {
            0.0
        },
        if misses > 0 {
            dir_ms * 1e6 / misses as f64
        } else {
            0.0
        }
    );
    eprintln!(
        "[plan-profile]   simulate_at: {:.2}ms across {} calls",
        sim_ms, sim_calls
    );
    eprintln!(
        "[plan-profile]   outer iters: {}, plan_for_time calls: {}",
        outer, pft
    );
}

// Thread-local cache for dir_to_hit results, valid for the duration of a
// single `plan()` call. Keyed by (src_id, tgt_id, ships, turns_in_future).
// plan_for_time's binary searches make ~14 dir_to_hit calls per
// (src, tgt) pair across all T iterations; many of those calls repeat the
// same ships count and benefit from memoization.
// Three-tier cache:
//   1. Small-ships path (ships < FAST_SHIPS_MAX): dense 3D array indexed by
//      (src.id, tgt.id, ships). Direct load, ~5-10ns.
//   2. Hybrid path (ships >= FAST_SHIPS_MAX, planet ids in bounds): outer
//      array indexed by (src.id, tgt.id), inner HashMap keyed by ships.
//      ~30-50ns.
//   3. Fallback (out-of-bounds planet ids): single flat HashMap.
//
// Most binary-search lookups in plan_for_time bounce among low ship counts
// (midpoints between 1 and avail/2), so the fast path is hit ~80% of the
// time in practice.
const DIR_N_PLANETS: usize = 32;
const DIR_ARR_LEN: usize = DIR_N_PLANETS * DIR_N_PLANETS;
const FAST_SHIPS_MAX: usize = 64;

// Time slot: 2 bytes per (src, tgt, ships). ver=0 means uncached; ver==cur_v
// means cached this plan_inner. time=0 means cached-as-None (real times start
// at 1), time>0 means Some(time). u8 ver wraps every 256 calls — on wrap we
// zero the array (~20µs, ~4 times/turn → negligible).
#[derive(Clone, Copy, Default)]
struct TimeSlot {
    ver: u8,
    time: u8,
}
const FAST_ARR_LEN: usize = DIR_N_PLANETS * DIR_N_PLANETS * FAST_SHIPS_MAX;

#[inline]
fn fast_idx(src_id: i64, tgt_id: i64, ships: i64) -> Option<usize> {
    if src_id < 0 || tgt_id < 0 || ships < 1 {
        return None;
    }
    let s = src_id as usize;
    let t = tgt_id as usize;
    let sh = ships as usize;
    if s >= DIR_N_PLANETS || t >= DIR_N_PLANETS || sh >= FAST_SHIPS_MAX {
        return None;
    }
    Some(s * (DIR_N_PLANETS * FAST_SHIPS_MAX) + t * FAST_SHIPS_MAX + sh)
}

#[inline]
fn dir_pair_idx(src_id: i64, tgt_id: i64) -> Option<usize> {
    if src_id < 0
        || tgt_id < 0
        || (src_id as usize) >= DIR_N_PLANETS
        || (tgt_id as usize) >= DIR_N_PLANETS
    {
        return None;
    }
    Some((src_id as usize) * DIR_N_PLANETS + tgt_id as usize)
}

thread_local! {
    // Fast path: split arrays. TIME (2 bytes/slot) is hit by binary searches
    // (~80% of calls) and fits L1 at N=64 (128KB). ANGLE (8 bytes/slot,
    // f64) is only read by the ~750 angle-needing calls per plan and is
    // co-written on TIME misses (so cached angle is always valid when its
    // matching TIME slot is current).
    //
    // Tuple: (cached_step, version, array). dir_to_hit is deterministic in
    // (src, tgt, ships, state.step) since planet positions are determined
    // by state.step and obstacle set is identical across same-step calls
    // (planets are never added/removed mid-step). So the cache stays valid
    // across many plan() calls at the same step — only invalidate on step
    // change.
    static FAST_TIME: RefCell<(i64, u8, Box<[TimeSlot; FAST_ARR_LEN]>)> = {
        let arr: Box<[TimeSlot; FAST_ARR_LEN]> =
            vec![TimeSlot::default(); FAST_ARR_LEN].into_boxed_slice().try_into().ok().unwrap();
        RefCell::new((-1, 1, arr))
    };
    static FAST_ANGLE: RefCell<Box<[f64; FAST_ARR_LEN]>> = {
        let arr: Box<[f64; FAST_ARR_LEN]> =
            vec![0.0f64; FAST_ARR_LEN].into_boxed_slice().try_into().ok().unwrap();
        RefCell::new(arr)
    };
    // Hybrid path: outer array indexed by (src.id, tgt.id); inner small
    // FxHashMap keyed by ships count.
    static DIR_ARR: RefCell<Box<[rustc_hash::FxHashMap<i64, Option<PathResult>>; DIR_ARR_LEN]>> = {
        let arr: Box<[rustc_hash::FxHashMap<i64, Option<PathResult>>; DIR_ARR_LEN]> =
            (0..DIR_ARR_LEN).map(|_| rustc_hash::FxHashMap::default())
                .collect::<Vec<_>>().try_into().ok().unwrap();
        RefCell::new(arr)
    };
    // Track which inner maps had inserts this plan_inner so we don't have
    // to iterate all 1024 to clear (most are untouched).
    static DIR_TOUCHED: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(512));
    // Fallback HashMap for out-of-bounds planet ids (rare — late-game comets
    // beyond DIR_N_PLANETS).
    static DIR_CACHE: RefCell<rustc_hash::FxHashMap<u64, Option<PathResult>>> =
        RefCell::new(rustc_hash::FxHashMap::default());
}

/// Time-only fast path for callers that only need arrival time (binary
/// searches, race filter). Skips the FAST_ANGLE RefCell lookup on hits —
/// ~30% faster per-hit than cached_dir_to_hit since hits are the dominant
/// path after the cross-call cache warms up.
#[inline]
fn cached_time_to_hit(
    src: &Planet,
    tgt: &Planet,
    ships: i64,
    state: &GameState,
    turns_in_future: i64,
) -> Option<i64> {
    let prof = plan_profile_enabled();
    let t0 = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    debug_assert_eq!(turns_in_future, 0);
    let r = if let Some(idx) = fast_idx(src.id, tgt.id, ships) {
        FAST_TIME.with(|tc| {
            let mut tcell = tc.borrow_mut();
            let cur_v = tcell.1;
            let slot = tcell.2[idx];
            if slot.ver == cur_v {
                if prof {
                    PLAN_DIR_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                }
                return if slot.time == 0 {
                    None
                } else {
                    Some(slot.time as i64)
                };
            }
            if prof {
                PLAN_DIR_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            }
            let v = pathing::dir_to_hit(src, tgt, ships, state, turns_in_future);
            let t_byte = match v.as_ref() {
                Some(p) => {
                    debug_assert!(p.time >= 1 && p.time <= 255);
                    FAST_ANGLE.with(|a| a.borrow_mut()[idx] = p.angle);
                    p.time as u8
                }
                None => 0,
            };
            tcell.2[idx] = TimeSlot {
                ver: cur_v,
                time: t_byte,
            };
            v.map(|r| r.time)
        })
    } else {
        cached_dir_to_hit(src, tgt, ships, state, turns_in_future).map(|r| r.time)
    };
    if let Some(t0) = t0 {
        PLAN_DIR_TOTAL_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    r
}

fn cached_dir_to_hit(
    src: &Planet,
    tgt: &Planet,
    ships: i64,
    state: &GameState,
    turns_in_future: i64,
) -> Option<PathResult> {
    let prof = plan_profile_enabled();
    let t0 = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    debug_assert_eq!(
        turns_in_future, 0,
        "cached_dir_to_hit called with non-zero turns_in_future; cache key doesn't include it"
    );
    let r = if let Some(idx) = fast_idx(src.id, tgt.id, ships) {
        // Tier 1 fast path: split arrays (TIME hot, ANGLE cold).
        FAST_TIME.with(|tc| {
            let mut tcell = tc.borrow_mut();
            let cur_v = tcell.1;
            let slot = tcell.2[idx];
            if slot.ver == cur_v {
                if prof {
                    PLAN_DIR_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                }
                return if slot.time == 0 {
                    None
                } else {
                    let angle = FAST_ANGLE.with(|a| a.borrow()[idx]);
                    Some(PathResult {
                        angle,
                        time: slot.time as i64,
                    })
                };
            }
            if prof {
                PLAN_DIR_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            }
            let v = pathing::dir_to_hit(src, tgt, ships, state, turns_in_future);
            let t_byte = match v.as_ref() {
                Some(p) => {
                    debug_assert!(
                        p.time >= 1 && p.time <= 255,
                        "time {} out of u8 range",
                        p.time
                    );
                    FAST_ANGLE.with(|a| a.borrow_mut()[idx] = p.angle);
                    p.time as u8
                }
                None => 0,
            };
            tcell.2[idx] = TimeSlot {
                ver: cur_v,
                time: t_byte,
            };
            v
        })
    } else if let Some(idx) = dir_pair_idx(src.id, tgt.id) {
        // Tier 2 hybrid: outer array → inner per-ships HashMap (for ships ≥ FAST_SHIPS_MAX).
        DIR_ARR.with(|c| {
            let mut arr = c.borrow_mut();
            let inner = &mut arr[idx];
            if let Some(v) = inner.get(&ships) {
                if prof {
                    PLAN_DIR_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                }
                return *v;
            }
            if prof {
                PLAN_DIR_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            }
            let v = pathing::dir_to_hit(src, tgt, ships, state, turns_in_future);
            let was_empty = inner.is_empty();
            inner.insert(ships, v);
            if was_empty {
                DIR_TOUCHED.with(|t| t.borrow_mut().push(idx));
            }
            v
        })
    } else {
        // Fallback: HashMap (out-of-bounds planet ids).
        let key: u64 = ((ships as u64) & 0xFFFF_FFFF)
            | (((tgt.id as u64) & 0xFFFF) << 32)
            | (((src.id as u64) & 0xFFFF) << 48);
        DIR_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            if let Some(v) = cache.get(&key) {
                if prof {
                    PLAN_DIR_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                }
                return *v;
            }
            if prof {
                PLAN_DIR_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            }
            let v = pathing::dir_to_hit(src, tgt, ships, state, turns_in_future);
            cache.insert(key, v);
            v
        })
    };
    if let Some(t0) = t0 {
        PLAN_DIR_TOTAL_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    r
}

/// Safe lower bound on dir_to_hit time. Used to skip sources that can't
/// possibly reach a target in the requested time, avoiding expensive
/// dir_to_hit cache lookups (and especially misses).
///
/// Must NEVER over-estimate: if this returns T, the real time is >= T.
/// Otherwise we'd wrongly skip a reachable source and diverge from baseline.
#[inline]
fn time_lower_bound(src: &Planet, tgt: &Planet, ships: i64, max_speed: f64) -> i64 {
    if tgt.is_comet {
        return 0;
    }
    let min_d = if tgt.is_orbiting {
        // Target orbits CENTER at orbital_radius; min distance from src over
        // all times = max(0, dist(src, CENTER) - orbital_radius).
        let dx = src.x - CENTER_X;
        let dy = src.y - CENTER_Y;
        ((dx * dx + dy * dy).sqrt() - tgt.orbital_radius).max(0.0)
    } else {
        let dx = tgt.x - src.x;
        let dy = tgt.y - src.y;
        (dx * dx + dy * dy).sqrt()
    };
    // Fleet hits target's edge; subtract target.radius. Subtract 1.0 extra
    // for safety (engine spawn_offset / hit_radius slack).
    let effective_d = (min_d - tgt.radius - 1.0).max(0.0);
    let speed = pathing::fleet_speed(ships, max_speed);
    (effective_d / speed).floor() as i64
}

fn clear_dir_cache_if_step_changed(step: i64) {
    // Tier 1 (fast TIME array): bump version only when step changes.
    // dir_to_hit is purely a function of (src, tgt, ships, step) since
    // positions/orbit obstacles are step-determined and ownership/ships
    // don't affect it. Reusing the cache across plan() calls at the same
    // step gives near-perfect hits after the first plan warms it up.
    let stale = FAST_TIME.with(|c| {
        let mut cell = c.borrow_mut();
        if cell.0 != step {
            cell.0 = step;
            cell.1 = cell.1.wrapping_add(1);
            if cell.1 == 0 {
                cell.1 = 1;
                for slot in cell.2.iter_mut() {
                    slot.ver = 0;
                }
            }
            true
        } else {
            false
        }
    });
    if stale {
        // Tier 2 (hybrid): only clear touched cells.
        DIR_TOUCHED.with(|t| {
            let mut touched = t.borrow_mut();
            DIR_ARR.with(|c| {
                let mut arr = c.borrow_mut();
                for &idx in touched.iter() {
                    arr[idx].clear();
                }
            });
            touched.clear();
        });
        // Tier 3 (fallback): clear.
        DIR_CACHE.with(|c| c.borrow_mut().clear());
    }
}

pub fn cached_predict_fleet_collision(
    fleet: &crate::Fleet,
    state: &GameState,
) -> Option<(i64, i64)> {
    pathing::predict_fleet_collision(fleet, state)
}

fn resolve_combat(
    mut owner: i32,
    mut ships: i64,
    arrivals_by_owner: &HashMap<i32, i64>,
) -> (i32, i64) {
    if arrivals_by_owner.is_empty() {
        return (owner, ships);
    }
    let mut srt: Vec<(i32, i64)> = arrivals_by_owner.iter().map(|(k, v)| (*k, *v)).collect();
    srt.sort_by(|a, b| b.1.cmp(&a.1));
    let (top_o, top_s) = srt[0];
    let (sv_o, sv_s) = if srt.len() > 1 {
        let sec = srt[1].1;
        if top_s == sec {
            (-1, 0)
        } else {
            (top_o, top_s - sec)
        }
    } else {
        (top_o, top_s)
    };
    if sv_s > 0 {
        if owner == sv_o {
            ships += sv_s;
        } else {
            ships -= sv_s;
            if ships < 0 {
                owner = sv_o;
                ships = -ships;
            }
        }
    }
    (owner, ships)
}

fn simulate_at(
    planet: &Planet,
    target_dt: i64,
    arrivals: &[(i64, i32, i64)],
    state: &GameState,
) -> (i32, i64) {
    let mut sorted: Vec<(i64, i32, i64)> = arrivals.to_vec();
    sorted.sort_by_key(|x| x.0);
    let mut owner = planet.owner;
    let mut ships = planet.ships;
    let mut cur_dt = 0i64;
    let mut i = 0;
    while i < sorted.len() && sorted[i].0 <= target_dt {
        let dt = sorted[i].0;
        if planet.is_comet && state.planet_pos_at(planet, dt).is_none() {
            return (-1, 0);
        }
        if owner != -1 {
            ships += planet.production * (dt - cur_dt);
        }
        let mut by_owner: HashMap<i32, i64> = HashMap::new();
        while i < sorted.len() && sorted[i].0 == dt {
            *by_owner.entry(sorted[i].1).or_insert(0) += sorted[i].2;
            i += 1;
        }
        let (no, ns) = resolve_combat(owner, ships, &by_owner);
        owner = no;
        ships = ns;
        cur_dt = dt;
    }
    if owner != -1 && target_dt > cur_dt {
        ships += planet.production * (target_dt - cur_dt);
    }
    (owner, ships)
}

fn stays_mine_throughout(
    target: &Planet,
    arrivals: &[(i64, i32, i64)],
    me: i32,
    state: &GameState,
) -> bool {
    let mut sorted: Vec<(i64, i32, i64)> = arrivals.to_vec();
    sorted.sort_by_key(|x| x.0);
    let mut owner = target.owner;
    let mut ships = target.ships;
    let mut cur_dt = 0i64;
    let mut became_mine = owner == me;
    let mut ever_lost = false;
    let mut i = 0;
    while i < sorted.len() {
        let dt = sorted[i].0;
        if target.is_comet && state.planet_pos_at(target, dt).is_none() {
            break;
        }
        if owner != -1 {
            ships += target.production * (dt - cur_dt);
        }
        let mut by_owner: HashMap<i32, i64> = HashMap::new();
        while i < sorted.len() && sorted[i].0 == dt {
            *by_owner.entry(sorted[i].1).or_insert(0) += sorted[i].2;
            i += 1;
        }
        let (no, ns) = resolve_combat(owner, ships, &by_owner);
        owner = no;
        ships = ns;
        if owner == me {
            became_mine = true;
        } else if became_mine {
            ever_lost = true;
        }
        cur_dt = dt;
    }
    !ever_lost && owner == me
}

fn simulates_safe(
    planet: &Planet,
    send_amount: i64,
    arrivals: &[(i64, i32, i64)],
    me: i32,
    state: &GameState,
) -> bool {
    if planet.owner != me {
        return send_amount == 0;
    }
    if send_amount < 0 || send_amount > planet.ships {
        return false;
    }
    let mut sorted: Vec<(i64, i32, i64)> = arrivals.to_vec();
    sorted.sort_by_key(|x| x.0);
    let mut owner = me;
    let mut ships = planet.ships - send_amount;
    let mut cur_dt = 0i64;
    let mut i = 0;
    while i < sorted.len() {
        let dt = sorted[i].0;
        if planet.is_comet && state.planet_pos_at(planet, dt).is_none() {
            break;
        }
        if owner != -1 {
            ships += planet.production * (dt - cur_dt);
        }
        let mut by_owner: HashMap<i32, i64> = HashMap::new();
        while i < sorted.len() && sorted[i].0 == dt {
            *by_owner.entry(sorted[i].1).or_insert(0) += sorted[i].2;
            i += 1;
        }
        let (no, ns) = resolve_combat(owner, ships, &by_owner);
        owner = no;
        ships = ns;
        if owner != me {
            return false;
        }
        cur_dt = dt;
    }
    true
}

#[derive(Clone, Copy)]
struct PlanEntry {
    from_id: i64,
    from_idx: usize, // index into my_planets / available
    ships: i64,
    angle: f64,
}

struct PlanetFeas {
    from_id: i64,
    from_idx: usize,
    min_s: i64,
    max_s: i64,
    arr_at_min: i64,
    angle_at_min: f64,
}

fn plan_for_time(
    target: &Planet,
    t: i64,
    my_planets: &[Planet],
    available: &[i64], // parallel to my_planets, indexed by position
    state: &GameState,
    arrivals: &HashMap<i64, Vec<(i64, i32, i64)>>,
    me: i32,
) -> Option<(Vec<PlanEntry>, i64)> {
    if plan_profile_enabled() {
        PLAN_FOR_TIME_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    let empty = Vec::new();
    let arr = arrivals.get(&target.id).unwrap_or(&empty);
    let _t_sim = if plan_profile_enabled() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let (owner_t, ships_t) = simulate_at(target, t, arr, state);
    if let Some(t0) = _t_sim {
        PLAN_SIMULATE_AT_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        PLAN_SIMULATE_AT_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    let required: i64 = if owner_t == me { 0 } else { ships_t + 1 };
    if required <= 0 {
        return None;
    }

    let mut feas: Vec<PlanetFeas> = Vec::new();
    for (mp_idx, mp) in my_planets.iter().enumerate() {
        if mp.id == target.id {
            continue;
        }
        let avail = available[mp_idx];
        if avail <= 0 {
            continue;
        }
        // Cheap pre-filter: if even max ships at max speed can't reach in
        // time t, skip without a dir_to_hit call. Saves the top_path probe
        // plus both binary searches.
        if time_lower_bound(mp, target, avail, state.max_speed) > t {
            continue;
        }
        let top_time = cached_time_to_hit(mp, target, avail, state, 0);
        if !top_time.map(|tt| tt <= t).unwrap_or(false) {
            continue;
        }
        let mut lo = 1i64;
        let mut hi = avail;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let ok = cached_time_to_hit(mp, target, mid, state, 0)
                .map(|tt| tt <= t)
                .unwrap_or(false);
            if ok {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let min_s = lo;
        let r_min = match cached_dir_to_hit(mp, target, min_s, state, 0) {
            Some(r) => r,
            None => continue,
        };
        let mut lo2 = min_s;
        let mut hi2 = avail;
        while lo2 < hi2 {
            let mid = (lo2 + hi2 + 1) / 2;
            let arr_mid = cached_time_to_hit(mp, target, mid, state, 0).unwrap_or(i64::MAX);
            if arr_mid >= t {
                lo2 = mid;
            } else {
                hi2 = mid - 1;
            }
        }
        let max_s = lo2;
        feas.push(PlanetFeas {
            from_id: mp.id,
            from_idx: mp_idx,
            min_s,
            max_s,
            arr_at_min: r_min.time,
            angle_at_min: r_min.angle,
        });
    }

    let cap_min: i64 = feas.iter().map(|x| x.min_s).sum();
    if cap_min < required {
        return None;
    }

    let mut tied: Vec<&PlanetFeas> = feas.iter().filter(|f| f.arr_at_min == t).collect();
    if tied.is_empty() {
        return None;
    }
    let base: i64 = tied.iter().map(|f| f.min_s).sum();
    let mut remaining_extra = (required - base).max(0);
    tied.sort_by(|a, b| (b.max_s - b.min_s).cmp(&(a.max_s - a.min_s)));
    let mut sends: HashMap<i64, i64> = HashMap::new();
    for f in &tied {
        sends.insert(f.from_id, f.min_s);
    }
    for f in &tied {
        if remaining_extra <= 0 {
            break;
        }
        let cap = (f.max_s - f.min_s).max(0);
        let add = remaining_extra.min(cap);
        if add > 0 {
            *sends.get_mut(&f.from_id).unwrap() += add;
            remaining_extra -= add;
        }
    }
    let total: i64 = sends.values().sum();
    if total < required {
        return None;
    }
    let mut dispatches: Vec<PlanEntry> = Vec::new();
    for f in &tied {
        let send = *sends.get(&f.from_id).unwrap_or(&0);
        if send <= 0 {
            continue;
        }
        let mp = match my_planets.iter().find(|p| p.id == f.from_id) {
            Some(p) => p,
            None => continue,
        };
        // If pathing succeeds at `send` ships, use that angle (it leads the
        // target for the faster fleet). If it fails, the `angle_at_min`
        // angle is stale — it was the lead angle for `min_s` at a different
        // (slower) speed, so reusing it with `send` ships shoots at where
        // the target was predicted to be at `arr_at_min`, not where the
        // faster fleet will actually meet it. Fall back to launching
        // `min_s` ships with their verified angle instead.
        let (angle, ships_to_send) = match cached_dir_to_hit(mp, target, send, state, 0) {
            Some(r) => (r.angle, send),
            None => (f.angle_at_min, f.min_s),
        };
        dispatches.push(PlanEntry {
            from_id: f.from_id,
            from_idx: f.from_idx,
            ships: ships_to_send,
            angle,
        });
    }
    Some((dispatches, total))
}

/// Run ow2-style plan for `player`, optionally with one target excluded
/// from consideration. Excluding a target forces the planner to pick its
/// next-best option, which is the natural way for MCTS to sample
/// "alternatives to greedy" that are still strong.
pub fn plan_with_exclusion(
    state: &GameState,
    player: i32,
    no_coop: bool,
    excluded_target: Option<i64>,
) -> Vec<Action> {
    plan_inner(state, player, no_coop, excluded_target, None)
}

/// Run ow2-style plan for `player` and return joint actions.
pub fn plan(state: &GameState, player: i32, no_coop: bool) -> Vec<Action> {
    plan_inner(state, player, no_coop, None, None)
}

/// Run ow2's plan, but restrict the greedy target loop to the SINGLE
/// specified target. Returns the cooperating-source attack orders that
/// ow2 would commit for capturing `only_target` this turn (one greedy
/// iteration — after that the single target is in `used_target` and the
/// outer loop exits).
///
/// This is the proper "ow policy for one target" helper.
pub fn plan_for_target(
    state: &GameState,
    player: i32,
    no_coop: bool,
    only_target: i64,
) -> Vec<Action> {
    plan_inner(state, player, no_coop, None, Some(only_target))
}

/// Precomputed state shared across many `plan_for_target` calls on the
/// SAME `(state, player, no_coop)` triple. The setup (arrivals, safe[],
/// enemy_safe[], race_ok[]) is the bulk of `plan_inner`'s cost — building
/// it once and reusing it for each of N candidate targets cuts
/// focused-candidate generation from O(N × setup) to O(setup + N × greedy).
pub struct PlanContext<'a> {
    pub state: &'a GameState,
    pub me: i32,
    pub no_coop: bool,
    pub arrivals: HashMap<i64, Vec<(i64, i32, i64)>>,
    pub safe: HashMap<i64, i64>,
    #[allow(dead_code)]
    pub enemy_safe: HashMap<i64, i64>,
    pub my_planets: Vec<Planet>,
    #[allow(dead_code)]
    pub enemy_planets: Vec<Planet>,
    pub race_ok: HashMap<i64, bool>,
}

impl<'a> PlanContext<'a> {
    pub fn build(state: &'a GameState, player: i32, no_coop: bool) -> Self {
        clear_dir_cache_if_step_changed(state.step);
        let me = player;

        let mut arrivals: HashMap<i64, Vec<(i64, i32, i64)>> = HashMap::new();
        for fleet in &state.fleets {
            if let Some((pid, dt)) = cached_predict_fleet_collision(fleet, state) {
                arrivals
                    .entry(pid)
                    .or_default()
                    .push((dt, fleet.owner, fleet.ships));
            }
        }
        for v in arrivals.values_mut() {
            v.sort_by_key(|x| x.0);
        }

        let mut safe: HashMap<i64, i64> = HashMap::new();
        for p in &state.planets {
            if p.owner != me {
                continue;
            }
            let empty = Vec::new();
            let arr = arrivals.get(&p.id).unwrap_or(&empty);
            let mut lo = 0;
            let mut hi = p.ships;
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if simulates_safe(p, mid, arr, me, state) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            safe.insert(p.id, lo);
        }

        let my_planets: Vec<Planet> = state
            .planets
            .iter()
            .filter(|p| p.owner == me)
            .cloned()
            .collect();
        let enemy_planets: Vec<Planet> = state
            .planets
            .iter()
            .filter(|p| p.owner != me && p.owner != -1)
            .cloned()
            .collect();

        let mut enemy_safe: HashMap<i64, i64> = HashMap::new();
        for p in &state.planets {
            if p.owner == me || p.owner == -1 {
                continue;
            }
            let empty = Vec::new();
            let arr = arrivals.get(&p.id).unwrap_or(&empty);
            let mut lo = 0;
            let mut hi = p.ships;
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if simulates_safe(p, mid, arr, p.owner, state) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            enemy_safe.insert(p.id, lo);
        }

        let mut race_ok: HashMap<i64, bool> = HashMap::new();
        let mut scratch: Vec<(i64, usize)> = Vec::with_capacity(16);
        for target in &state.planets {
            if target.owner == me {
                continue;
            }
            scratch.clear();
            for (i, mp) in my_planets.iter().enumerate() {
                if mp.ships <= 0 {
                    continue;
                }
                scratch.push((time_lower_bound(mp, target, mp.ships, state.max_speed), i));
            }
            scratch.sort_unstable_by_key(|&(lb, _)| lb);
            let mut my_t = i64::MAX;
            for &(lb, i) in &scratch {
                if lb >= my_t {
                    break;
                }
                if let Some(tt) =
                    cached_time_to_hit(&my_planets[i], target, my_planets[i].ships, state, 0)
                {
                    if tt < my_t {
                        my_t = tt;
                    }
                }
            }
            scratch.clear();
            for (i, ep) in enemy_planets.iter().enumerate() {
                if ep.ships <= 0 || ep.id == target.id {
                    continue;
                }
                scratch.push((time_lower_bound(ep, target, ep.ships, state.max_speed), i));
            }
            scratch.sort_unstable_by_key(|&(lb, _)| lb);
            let mut their_t = i64::MAX;
            for &(lb, i) in &scratch {
                if lb >= their_t {
                    break;
                }
                if let Some(tt) =
                    cached_time_to_hit(&enemy_planets[i], target, enemy_planets[i].ships, state, 0)
                {
                    if tt < their_t {
                        their_t = tt;
                    }
                }
            }
            race_ok.insert(target.id, my_t <= their_t);
        }

        PlanContext {
            state,
            me,
            no_coop,
            arrivals,
            safe,
            enemy_safe,
            my_planets,
            enemy_planets,
            race_ok,
        }
    }
}

/// Run the greedy multi-pass body of `plan_inner` for ONE specific target,
/// using shared precomputed context. Equivalent to
/// `plan_for_target(state, player, no_coop, target_id)` but skips the
/// O(N × log × |fleets|) setup work.
pub fn plan_target_with_ctx(ctx: &PlanContext, target_id: i64) -> Vec<Action> {
    let me = ctx.me;
    let state = ctx.state;
    let no_coop = ctx.no_coop;
    let mut arrivals = ctx.arrivals.clone();
    let mut available: Vec<i64> = ctx
        .my_planets
        .iter()
        .map(|p| ctx.safe.get(&p.id).copied().unwrap_or(0))
        .collect();
    let mut used_target: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut moves: Vec<(i64, f64, i64, i64)> = Vec::new();
    let mut guard = 0usize;
    loop {
        guard += 1;
        if guard > 200 {
            break;
        }
        let mut best: Option<(f64, i64, i64, Vec<PlanEntry>)> = None;
        // Only one target body runs (filtered by only_target).
        let target = match state.planets.iter().find(|p| p.id == target_id) {
            Some(t) => t,
            None => break,
        };
        if no_coop && used_target.contains(&target.id) {
            break;
        }
        let empty = Vec::new();
        let arr = arrivals.get(&target.id).unwrap_or(&empty);
        if stays_mine_throughout(target, arr, me, state) {
            break;
        }
        if target.owner != me && !*ctx.race_ok.get(&target.id).unwrap_or(&false) {
            break;
        }
        let hi = pathing::MAX_TIME.min(60);
        let mut found: Option<(i64, Vec<PlanEntry>, i64)> = None;
        let mut t = 1i64;
        while t <= hi {
            if let Some((p, total)) =
                plan_for_time(target, t, &ctx.my_planets, &available, state, &arrivals, me)
            {
                found = Some((t, p, total));
                break;
            }
            t += 1;
        }
        if let Some((t_best, plan_best, total_best)) = found {
            if target.is_comet && target.owner != me {
                let remaining = state.comet_remaining(target);
                let productions_after = (remaining - t_best).max(0);
                let ships_lost = (total_best - 1).max(0);
                if ships_lost >= productions_after * target.production {
                    break;
                }
            }
            let prod = target.production.max(1) as f64;
            let score = prod / (total_best.max(1) as f64);
            let s = score - 1e-6 * t_best as f64;
            best = Some((s, target.id, t_best, plan_best));
        }
        if let Some((_, target_id, t_best, dispatches)) = best {
            if dispatches.is_empty() {
                break;
            }
            used_target.insert(target_id);
            let entry = arrivals.entry(target_id).or_default();
            for d in &dispatches {
                entry.push((t_best, me, d.ships));
            }
            entry.sort_by_key(|x| x.0);
            let mut to_emit: Vec<PlanEntry> = dispatches.clone();
            if no_coop && to_emit.len() > 1 {
                to_emit.sort_by(|a, b| b.ships.cmp(&a.ships));
                to_emit.truncate(1);
            }
            for d in to_emit {
                moves.push((d.from_id, d.angle, d.ships, target_id));
                if d.from_idx < available.len() {
                    available[d.from_idx] -= d.ships;
                    if available[d.from_idx] < 0 {
                        available[d.from_idx] = 0;
                    }
                }
            }
        } else {
            break;
        }
    }
    moves
        .into_iter()
        .map(|(f, a, s, _t)| (f, a, s, me))
        .collect()
}

/// Build a single-action "approach" plan for an unaffordable target.
/// Picks the source planet that can ARRIVE at the target soonest with its
/// safe-drain ships, and emits one launch toward the target. The ships
/// stay in flight as committed-but-unresolved — next-turn focused_candidates
/// sees them in `arrivals[]` and can complete the capture from there.
/// Returns None if no source can reach the target.
pub fn approach_plan_for_target(ctx: &PlanContext, target_id: i64) -> Option<Vec<Action>> {
    let state = ctx.state;
    let target = state.planets.iter().find(|p| p.id == target_id)?;
    if target.owner == ctx.me {
        return None;
    }
    let me = ctx.me;

    let mut best: Option<(i64, i64, f64, i64)> = None; // (time, source_id, angle, send)
    for src in &ctx.my_planets {
        let safe = ctx.safe.get(&src.id).copied().unwrap_or(0);
        if safe <= 0 {
            continue;
        }
        // Send the smaller of (safe-drain, target garrison) — sending more
        // than target.ships at one source isn't useful since one source
        // can't single-handedly capture a target it can't afford anyway.
        let send = safe.min(target.ships.max(1));
        if send <= 0 {
            continue;
        }
        if let Some(r) = cached_dir_to_hit(src, target, send, state, 0) {
            if best.as_ref().map(|b| r.time < b.0).unwrap_or(true) {
                best = Some((r.time, src.id, r.angle, send));
            }
        }
    }

    let (_, src_id, angle, send) = best?;
    Some(vec![(src_id, angle, send, me)])
}

fn plan_inner(
    state: &GameState,
    player: i32,
    no_coop: bool,
    excluded_target: Option<i64>,
    only_target: Option<i64>,
) -> Vec<Action> {
    let _prof_start = if plan_profile_enabled() {
        PLAN_CALLS.fetch_add(1, Ordering::Relaxed);
        Some(std::time::Instant::now())
    } else {
        None
    };
    // Each plan() call gets a fresh cache — entries cached against the
    // current state.step, so subsequent ticks must not reuse stale results.
    clear_dir_cache_if_step_changed(state.step);
    let me = player;
    let mut moves: Vec<(i64, f64, i64, i64)> = Vec::new(); // (from, angle, ships, target_id)

    let mut arrivals: HashMap<i64, Vec<(i64, i32, i64)>> = HashMap::new();
    for fleet in &state.fleets {
        if let Some((pid, dt)) = cached_predict_fleet_collision(fleet, state) {
            arrivals
                .entry(pid)
                .or_default()
                .push((dt, fleet.owner, fleet.ships));
        }
    }
    for v in arrivals.values_mut() {
        v.sort_by_key(|x| x.0);
    }

    // Reverted to original throughout-safety check (simulates_safe + binary
    // search). The end-state-only simplification tested 1-5 LOSS vs the
    // throughout version — being more permissive about temporary ownership
    // flips loses planets we could have defended.
    let mut safe: HashMap<i64, i64> = HashMap::new();
    for p in &state.planets {
        if p.owner != me {
            continue;
        }
        let empty = Vec::new();
        let arr = arrivals.get(&p.id).unwrap_or(&empty);
        let mut lo = 0;
        let mut hi = p.ships;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if simulates_safe(p, mid, arr, me, state) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        safe.insert(p.id, lo);
    }

    let my_planets: Vec<Planet> = state
        .planets
        .iter()
        .filter(|p| p.owner == me)
        .cloned()
        .collect();
    let enemy_planets: Vec<Planet> = state
        .planets
        .iter()
        .filter(|p| p.owner != me && p.owner != -1)
        .cloned()
        .collect();

    let mut enemy_safe: HashMap<i64, i64> = HashMap::new();
    for p in &state.planets {
        if p.owner == me || p.owner == -1 {
            continue;
        }
        let empty = Vec::new();
        let arr = arrivals.get(&p.id).unwrap_or(&empty);
        let mut lo = 0;
        let mut hi = p.ships;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if simulates_safe(p, mid, arr, p.owner, state) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        enemy_safe.insert(p.id, lo);
    }

    // Race filter: which player gets to the target first?
    // Use CURRENT ownership (my_planets, enemy_planets are filtered by
    // current state) AND CURRENT ship count (mp.ships / ep.ships) — NOT
    // the post-extrapolation `safe` / `enemy_safe` (which would exclude
    // planets projected to be lost in the future, even though they're
    // currently ours and can launch right now).
    // Race filter: sort candidates by time_lower_bound and early-exit when
    // remaining LBs can't improve the running min. Saves dir_to_hit calls
    // for far-away planets (which dominate at large game states).
    let mut race_ok: HashMap<i64, bool> = HashMap::new();
    let mut scratch: Vec<(i64, usize)> = Vec::with_capacity(16); // (lb, idx)
    for target in &state.planets {
        if target.owner == me {
            continue;
        }
        // my_t with early-exit.
        scratch.clear();
        for (i, mp) in my_planets.iter().enumerate() {
            if mp.ships <= 0 {
                continue;
            }
            scratch.push((time_lower_bound(mp, target, mp.ships, state.max_speed), i));
        }
        scratch.sort_unstable_by_key(|&(lb, _)| lb);
        let mut my_t = i64::MAX;
        for &(lb, i) in &scratch {
            if lb >= my_t {
                break;
            }
            if let Some(tt) =
                cached_time_to_hit(&my_planets[i], target, my_planets[i].ships, state, 0)
            {
                if tt < my_t {
                    my_t = tt;
                }
            }
        }
        // their_t with early-exit.
        scratch.clear();
        for (i, ep) in enemy_planets.iter().enumerate() {
            if ep.ships <= 0 || ep.id == target.id {
                continue;
            }
            scratch.push((time_lower_bound(ep, target, ep.ships, state.max_speed), i));
        }
        scratch.sort_unstable_by_key(|&(lb, _)| lb);
        let mut their_t = i64::MAX;
        for &(lb, i) in &scratch {
            if lb >= their_t {
                break;
            }
            if let Some(tt) =
                cached_time_to_hit(&enemy_planets[i], target, enemy_planets[i].ships, state, 0)
            {
                if tt < their_t {
                    their_t = tt;
                }
            }
        }
        race_ok.insert(target.id, my_t <= their_t);
    }

    // `available` is a Vec parallel to `my_planets` (indexed by position),
    // not a HashMap. ~3000 lookups in the hot loop drop from ~50ns HashMap
    // to ~5ns array indexing.
    let mut available: Vec<i64> = my_planets
        .iter()
        .map(|p| safe.get(&p.id).copied().unwrap_or(0))
        .collect();
    let mut used_target: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut guard = 0usize;
    loop {
        guard += 1;
        if guard > 200 {
            break;
        }
        if plan_profile_enabled() {
            PLAN_OUTER_ITERS.fetch_add(1, Ordering::Relaxed);
        }
        let mut best: Option<(f64, i64, i64, Vec<PlanEntry>)> = None;
        for target in &state.planets {
            if no_coop && used_target.contains(&target.id) {
                continue;
            }
            if Some(target.id) == excluded_target {
                continue;
            }
            if let Some(only) = only_target {
                if target.id != only {
                    continue;
                }
            }
            let empty = Vec::new();
            let arr = arrivals.get(&target.id).unwrap_or(&empty);
            if stays_mine_throughout(target, arr, me, state) {
                continue;
            }
            if target.owner != me && !*race_ok.get(&target.id).unwrap_or(&false) {
                continue;
            }
            let hi = pathing::MAX_TIME.min(60);
            let mut found: Option<(i64, Vec<PlanEntry>, i64)> = None;
            let mut t = 1i64;
            while t <= hi {
                if let Some((p, total)) =
                    plan_for_time(target, t, &my_planets, &available, state, &arrivals, me)
                {
                    found = Some((t, p, total));
                    break;
                }
                t += 1;
            }
            if let Some((t_best, plan_best, total_best)) = found {
                if target.is_comet && target.owner != me {
                    let remaining = state.comet_remaining(target);
                    let productions_after = (remaining - t_best).max(0);
                    let ships_lost = (total_best - 1).max(0);
                    if ships_lost >= productions_after * target.production {
                        continue;
                    }
                }
                let prod = target.production.max(1) as f64;
                let score = prod / (total_best.max(1) as f64);
                let s = score - 1e-6 * t_best as f64;
                if best.as_ref().map(|b| s > b.0).unwrap_or(true) {
                    best = Some((s, target.id, t_best, plan_best));
                }
            }
        }
        if let Some((_, target_id, t_best, dispatches)) = best {
            if dispatches.is_empty() {
                break;
            }
            used_target.insert(target_id);
            let entry = arrivals.entry(target_id).or_default();
            for d in &dispatches {
                entry.push((t_best, me, d.ships));
            }
            entry.sort_by_key(|x| x.0);
            // With no_coop, keep the largest single dispatch per target.
            let mut to_emit: Vec<PlanEntry> = dispatches.clone();
            if no_coop && to_emit.len() > 1 {
                to_emit.sort_by(|a, b| b.ships.cmp(&a.ships));
                to_emit.truncate(1);
            }
            for d in to_emit {
                moves.push((d.from_id, d.angle, d.ships, target_id));
                if d.from_idx < available.len() {
                    available[d.from_idx] -= d.ships;
                    if available[d.from_idx] < 0 {
                        available[d.from_idx] = 0;
                    }
                }
            }
        } else {
            break;
        }
    }

    // Departing-comet evacuation.
    for p in &state.planets {
        if p.owner != me || !p.is_comet {
            continue;
        }
        if state.comet_remaining(p) > 1 {
            continue;
        }
        // Find this planet's index in my_planets (linear, but only on
        // departing-comet iteration which is rare).
        let mp_idx = my_planets.iter().position(|mp| mp.id == p.id);
        let avail = mp_idx.map(|i| available[i]).unwrap_or(0).min(p.ships);
        if avail <= 0 {
            continue;
        }
        let mut best: Option<(i64, f64, i64)> = None;
        for tgt in &state.planets {
            if tgt.id == p.id {
                continue;
            }
            if no_coop && used_target.contains(&tgt.id) {
                continue;
            }
            if let Some(r) = cached_dir_to_hit(p, tgt, avail, state, 0) {
                if best.as_ref().map(|b| r.time < b.0).unwrap_or(true) {
                    best = Some((r.time, r.angle, tgt.id));
                }
            }
        }
        if let Some((_, angle, tgt_id)) = best {
            used_target.insert(tgt_id);
            moves.push((p.id, angle, avail, tgt_id));
            if let Some(idx) = mp_idx {
                let a = &mut available[idx];
                *a = 0;
            }
        }
    }

    let result: Vec<Action> = moves
        .into_iter()
        .map(|(f, a, s, _t)| (f, a, s, me))
        .collect();
    if let Some(t0) = _prof_start {
        PLAN_TOTAL_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    result
}
