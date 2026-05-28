//! Value network for alphaow leaf evaluation.
//!
//! Two-stream input per leaf state:
//!   1. CURRENT — `(owner, ships, radius, type, production)` for every
//!      object (planets + comets).
//!   2. EXTRAPOLATED — same per-object features but with every in-flight
//!      fleet already resolved into its predicted target (no production
//!      added — per user spec, this is "ships extrapolated", not a
//!      timed forward sim).
//!
//! Plus a pairwise Euclidean-distance matrix between all objects.
//!
//! ## Feature layout
//!
//! Up to `MAX_OBJECTS = 44` objects (40 planets + 4 comets). Each object
//! contributes `PER_OBJECT = 9` floats (see `pack_object`). Missing
//! objects are zero-padded.
//!
//! ```text
//! [current:  MAX_OBJECTS*PER_OBJECT = 396 f32]
//! [extrap:   MAX_OBJECTS*PER_OBJECT = 396 f32]
//! [dist:     MAX_OBJECTS*MAX_OBJECTS = 1936 f32]
//! total input dim = 2728
//! ```
//!
//! ## Weight format (`ALPHAOW_VALUE_NET_PATH`)
//!
//! Little-endian binary file:
//!
//! ```text
//! magic     u32 = 0x564f4157  ("AOWV")
//! version   u32 = 1
//! input_dim u32 (must equal INPUT_DIM)
//! hidden    u32
//! w1        f32[hidden * input_dim]   row-major (hidden first)
//! b1        f32[hidden]
//! w2        f32[hidden]
//! b2        f32
//! ```
//!
//! Forward pass: `y = tanh(b2 + w2 · ReLU(w1 · x + b1))`. Output is a
//! scalar in `[-1, 1]` interpreted as MCTS value from MY perspective.
//!
//! If no weights file is found or the file is malformed, `predict`
//! returns `None`. Callers should fall back to the duck heuristic.

use crate::ow2_plan::cached_predict_fleet_collision;
use crate::{GameState, Planet};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const MAX_OBJECTS: usize = 44;
pub const PER_OBJECT: usize = 9;
pub const PER_BLOCK: usize = MAX_OBJECTS * PER_OBJECT;
pub const DIST_BLOCK: usize = MAX_OBJECTS * MAX_OBJECTS;
pub const INPUT_DIM: usize = 2 * PER_BLOCK + DIST_BLOCK;

const WEIGHTS_MAGIC: u32 = 0x564f_4157; // "AOWV" little-endian

/// Per-object feature slot. 9 floats per planet/comet:
/// `[is_me, is_opp, is_neutral, log1p(ships), radius, is_static, is_orbit, is_comet, production]`
pub struct Features {
    pub current: Box<[f32; PER_BLOCK]>,
    pub extrap: Box<[f32; PER_BLOCK]>,
    pub dist: Box<[f32; DIST_BLOCK]>,
}

#[inline]
fn one_hot_owner(owner: i32, me: i32) -> (f32, f32, f32) {
    if owner == -1 {
        (0.0, 0.0, 1.0)
    } else if owner == me {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    }
}

#[inline]
fn one_hot_type(p: &Planet) -> (f32, f32, f32) {
    if p.is_comet {
        (0.0, 0.0, 1.0)
    } else if p.is_orbiting {
        (0.0, 1.0, 0.0)
    } else {
        (1.0, 0.0, 0.0)
    }
}

fn pack_object(buf: &mut [f32; PER_BLOCK], slot: usize, p: &Planet, ships: i64, owner: i32, me: i32) {
    let base = slot * PER_OBJECT;
    let (om, oo, on_) = one_hot_owner(owner, me);
    let (st, ob, co) = one_hot_type(p);
    buf[base] = om;
    buf[base + 1] = oo;
    buf[base + 2] = on_;
    buf[base + 3] = (ships.max(0) as f32 + 1.0).ln();
    buf[base + 4] = p.radius as f32;
    buf[base + 5] = st;
    buf[base + 6] = ob;
    buf[base + 7] = co;
    buf[base + 8] = p.production as f32;
}

/// Resolve all in-flight fleets into a predicted `(owner, ships)` per
/// planet. NO production is applied — purely "what does the board look
/// like once the current flights land". Fleets that die in the sun or
/// fly off the board contribute nothing.
///
/// Combat at each planet: process arrivals in arrival-time order. Mirror
/// the simple combat rule (attacker > defender → flip; attacker ≤
/// defender → defender shaves attacker's ships off). Ignores
/// same-step multi-fleet engine logic for simplicity (the value net is
/// an approximation anyway).
pub fn extrapolate_fleets(state: &GameState) -> HashMap<i64, (i32, i64)> {
    let mut arrivals: HashMap<i64, Vec<(i64, i32, i64)>> = HashMap::new();
    for fleet in &state.fleets {
        if let Some((pid, dt)) = cached_predict_fleet_collision(fleet, state) {
            arrivals.entry(pid).or_default().push((dt, fleet.owner, fleet.ships));
        }
    }
    let mut result: HashMap<i64, (i32, i64)> = state
        .planets
        .iter()
        .map(|p| (p.id, (p.owner, p.ships)))
        .collect();
    for (pid, mut arrs) in arrivals {
        arrs.sort_by_key(|x| x.0);
        let entry = result.entry(pid).or_insert((-1, 0));
        let (mut owner, mut ships) = *entry;
        for (_t, f_owner, f_ships) in arrs {
            if f_owner == owner {
                ships += f_ships;
            } else if f_ships > ships {
                owner = f_owner;
                ships = f_ships - ships;
            } else {
                ships -= f_ships;
            }
        }
        *entry = (owner, ships);
    }
    result
}

/// Build the feature tensors for `state` from `me`'s perspective.
pub fn extract_features(state: &GameState, me: i32) -> Features {
    let mut current = Box::new([0f32; PER_BLOCK]);
    let mut extrap = Box::new([0f32; PER_BLOCK]);
    let mut dist = Box::new([0f32; DIST_BLOCK]);

    let extrap_map = extrapolate_fleets(state);
    // Stable ordering by planet.id so feature slot k always maps to the
    // same object across calls (lets the net learn positional priors).
    let mut planets: Vec<&Planet> = state.planets.iter().collect();
    planets.sort_by_key(|p| p.id);
    let n = planets.len().min(MAX_OBJECTS);
    for i in 0..n {
        let p = planets[i];
        pack_object(&mut current, i, p, p.ships, p.owner, me);
        let (eo, es) = extrap_map.get(&p.id).copied().unwrap_or((p.owner, p.ships));
        pack_object(&mut extrap, i, p, es, eo, me);
    }
    for i in 0..n {
        for j in 0..n {
            let dx = (planets[i].x - planets[j].x) as f32;
            let dy = (planets[i].y - planets[j].y) as f32;
            dist[i * MAX_OBJECTS + j] = (dx * dx + dy * dy).sqrt();
        }
    }
    Features { current, extrap, dist }
}

/// Input variant for the loaded weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum InputKind {
    /// 2728-d raw two-stream + distance matrix.
    Full,
    /// 19-d handcrafted summary (see `summary_features::extract`).
    Summary,
    /// 46-d v2 summary (per-player + extrap + neutral block;
    /// see `summary_features_v2::extract`).
    SummaryV2,
}

struct MlpWeights {
    hidden: usize,
    input_dim: usize,
    kind: InputKind,
    w1: Vec<f32>, // [hidden, input_dim]
    b1: Vec<f32>, // [hidden]
    w2: Vec<f32>, // [hidden]
    b2: f32,
}

fn read_u32_le(slice: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(slice.get(..4)?.try_into().ok()?))
}

fn parse_weights(bytes: &[u8]) -> Option<MlpWeights> {
    if bytes.len() < 16 {
        return None;
    }
    if read_u32_le(&bytes[0..4])? != WEIGHTS_MAGIC {
        return None;
    }
    let version = read_u32_le(&bytes[4..8])?;
    if version != 1 {
        return None;
    }
    let input_dim = read_u32_le(&bytes[8..12])? as usize;
    let hidden = read_u32_le(&bytes[12..16])? as usize;
    if hidden == 0 {
        return None;
    }
    let kind = if input_dim == INPUT_DIM {
        InputKind::Full
    } else if input_dim == summary_features::DIM {
        InputKind::Summary
    } else if input_dim == summary_features_v2::DIM {
        InputKind::SummaryV2
    } else {
        eprintln!(
            "[alphaow] unknown input_dim={} (expected {} full / {} summary / {} summary_v2)",
            input_dim,
            INPUT_DIM,
            summary_features::DIM,
            summary_features_v2::DIM
        );
        return None;
    };
    let need = 16 + 4 * (hidden * input_dim + hidden + hidden + 1);
    if bytes.len() < need {
        return None;
    }
    let mut cursor = 16usize;
    let mut read_f32_block = |count: usize, c: &mut usize| -> Vec<f32> {
        let end = *c + 4 * count;
        let slice = &bytes[*c..end];
        let mut out = Vec::with_capacity(count);
        for chunk in slice.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        *c = end;
        out
    };
    let w1 = read_f32_block(hidden * input_dim, &mut cursor);
    let b1 = read_f32_block(hidden, &mut cursor);
    let w2 = read_f32_block(hidden, &mut cursor);
    let b2 = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().ok()?);
    Some(MlpWeights { hidden, input_dim, kind, w1, b1, w2, b2 })
}

fn load_weights() -> Option<MlpWeights> {
    let path = match std::env::var("ALPHAOW_VALUE_NET_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("[alphaow] ALPHAOW_VALUE_NET_PATH not set; using duck heuristic");
            return None;
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[alphaow] could not read weights at {}: {}", path, e);
            return None;
        }
    };
    let w = parse_weights(&bytes);
    match &w {
        Some(mw) => eprintln!(
            "[alphaow] loaded value net (kind={:?}, hidden={}, input_dim={}) from {}",
            mw.kind, mw.hidden, mw.input_dim, path
        ),
        None => eprintln!("[alphaow] failed to parse value net weights at {}", path),
    }
    w
}

static WEIGHTS: OnceLock<Option<MlpWeights>> = OnceLock::new();

fn weights() -> Option<&'static MlpWeights> {
    WEIGHTS.get_or_init(load_weights).as_ref()
}

/// True iff weights are loaded and ready for inference.
pub fn is_ready() -> bool {
    weights().is_some()
}

/// Inner-product `row · input + bias`. On aarch64+NEON we use 4× FMA
/// accumulators over 16-element chunks; falls back to an 8-accumulator
/// scalar loop elsewhere. `dim` is the length of both slices.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn dot_neon(row: &[f32], input: &[f32], bias: f32, dim: usize) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(row.len(), dim);
    debug_assert_eq!(input.len(), dim);
    unsafe {
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let chunks = dim / 16;
        let r_ptr = row.as_ptr();
        let i_ptr = input.as_ptr();
        for c in 0..chunks {
            let b = c * 16;
            a0 = vfmaq_f32(a0, vld1q_f32(r_ptr.add(b)), vld1q_f32(i_ptr.add(b)));
            a1 = vfmaq_f32(a1, vld1q_f32(r_ptr.add(b + 4)), vld1q_f32(i_ptr.add(b + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(r_ptr.add(b + 8)), vld1q_f32(i_ptr.add(b + 8)));
            a3 = vfmaq_f32(a3, vld1q_f32(r_ptr.add(b + 12)), vld1q_f32(i_ptr.add(b + 12)));
        }
        let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) + bias;
        for i in (chunks * 16)..dim {
            acc += row.get_unchecked(i) * input.get_unchecked(i);
        }
        acc
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline(always)]
fn dot_neon(row: &[f32], input: &[f32], bias: f32, dim: usize) -> f32 {
    debug_assert_eq!(row.len(), dim);
    debug_assert_eq!(input.len(), dim);
    let mut s0 = bias;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let chunks = dim / 4;
    for c in 0..chunks {
        let b = c * 4;
        unsafe {
            s0 += row.get_unchecked(b) * input.get_unchecked(b);
            s1 += row.get_unchecked(b + 1) * input.get_unchecked(b + 1);
            s2 += row.get_unchecked(b + 2) * input.get_unchecked(b + 2);
            s3 += row.get_unchecked(b + 3) * input.get_unchecked(b + 3);
        }
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..dim {
        s += row[i] * input[i];
    }
    s
}

fn forward_raw(w: &MlpWeights, input: &[f32]) -> f32 {
    debug_assert_eq!(input.len(), w.input_dim);
    let mut hidden = [0f32; 512];
    debug_assert!(w.hidden <= hidden.len(), "hidden > 512 not supported");
    for h in 0..w.hidden {
        let row = &w.w1[h * w.input_dim..(h + 1) * w.input_dim];
        let s = dot_neon(row, input, w.b1[h], w.input_dim);
        hidden[h] = if s > 0.0 { s } else { 0.0 };
    }
    let mut out = w.b2;
    for h in 0..w.hidden {
        out += w.w2[h] * hidden[h];
    }
    out.tanh()
}

fn forward_full(w: &MlpWeights, features: &Features) -> f32 {
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<f32>> =
            std::cell::RefCell::new(vec![0.0f32; INPUT_DIM]);
    }
    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch[..PER_BLOCK].copy_from_slice(features.current.as_ref());
        scratch[PER_BLOCK..2 * PER_BLOCK].copy_from_slice(features.extrap.as_ref());
        scratch[2 * PER_BLOCK..].copy_from_slice(features.dist.as_ref());
        forward_raw(w, &scratch[..])
    })
}

/// Run the value net on `state` from `me`'s perspective. Returns `None`
/// if no weights are loaded (caller should fall back to the heuristic).
/// Output is in `[-1, 1]` — MY perspective.
pub fn predict(state: &GameState, me: i32) -> Option<f64> {
    let w = weights()?;
    let y = match w.kind {
        InputKind::Full => {
            let features = extract_features(state, me);
            forward_full(w, &features)
        }
        InputKind::Summary => {
            let feats = summary_features::extract(state, me);
            forward_raw(w, &feats)
        }
        InputKind::SummaryV2 => {
            let feats = summary_features_v2::extract(state, me);
            forward_raw(w, &feats)
        }
    };
    Some(y as f64)
}

/// Handcrafted scalar summary features. Permutation-invariant by
/// construction — does not depend on planet ordering. Matches the
/// Python reference in `train/summary_features.py` byte-for-byte
/// (verified by a feature-parity unit test in
/// `tests/summary_parity.rs`, see CI).
pub mod summary_features {
    use super::*;

    pub const DIM: usize = 23;

    /// Build the 19-d summary feature vector directly from the game
    /// state. Avoids the cost of materializing the 2728-d raw feature
    /// tensor — important when the bot uses a summary-only value net
    /// (no need to compute the distance matrix).
    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        // Resolve in-flight fleets for the extrapolated counts.
        let extrap_map = extrapolate_fleets(state);

        let mut my_ships = 0.0f32;
        let mut opp_ships = 0.0f32;
        let mut neutral_ships = 0.0f32;
        let mut my_ships_ext = 0.0f32;
        let mut opp_ships_ext = 0.0f32;
        let mut my_planets = 0.0f32;
        let mut opp_planets = 0.0f32;
        let mut neutral_planets = 0.0f32;
        let mut my_planets_ext = 0.0f32;
        let mut opp_planets_ext = 0.0f32;
        let mut my_production = 0.0f32;
        let mut opp_production = 0.0f32;
        let mut my_radius = 0.0f32;
        let mut opp_radius = 0.0f32;
        let mut pressure_me_to_opp = 0.0f32;
        let mut pressure_opp_to_me = 0.0f32;
        let mut pressure_me_to_neutral = 0.0f32;
        let mut pressure_opp_to_neutral = 0.0f32;
        let mut max_my_ships = 0.0f32;
        let mut max_opp_ships = 0.0f32;
        let n_planets = state.planets.len() as f32;

        // Precompute per-planet owner classification (current + extrap).
        let n = state.planets.len();
        let mut cur_owner: Vec<i32> = Vec::with_capacity(n);
        let mut ext_owner: Vec<i32> = Vec::with_capacity(n);
        let mut ext_ships: Vec<f32> = Vec::with_capacity(n);
        for p in &state.planets {
            cur_owner.push(p.owner);
            let (eo, es) = extrap_map.get(&p.id).copied().unwrap_or((p.owner, p.ships));
            ext_owner.push(eo);
            ext_ships.push(es as f32);
        }

        for (idx, p) in state.planets.iter().enumerate() {
            let ships = p.ships as f32;
            let radius = p.radius as f32;
            let prod = p.production as f32;
            let owner = cur_owner[idx];
            let ext_o = ext_owner[idx];
            let ext_s = ext_ships[idx];
            if owner == -1 {
                neutral_ships += ships;
                neutral_planets += 1.0;
            } else if owner == me {
                my_ships += ships;
                my_planets += 1.0;
                my_production += prod;
                my_radius += radius;
                if ships > max_my_ships {
                    max_my_ships = ships;
                }
            } else {
                opp_ships += ships;
                opp_planets += 1.0;
                opp_production += prod;
                opp_radius += radius;
                if ships > max_opp_ships {
                    max_opp_ships = ships;
                }
            }
            if ext_o == -1 {
                // no contribution to my/opp ships ext
            } else if ext_o == me {
                my_ships_ext += ext_s;
                my_planets_ext += 1.0;
            } else {
                opp_ships_ext += ext_s;
                opp_planets_ext += 1.0;
            }
        }

        // Pairwise pressure + frontline distance.
        let mut front_dist = 200.0f32;
        for i in 0..n {
            let pi = &state.planets[i];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = &state.planets[j];
                let dx = (pi.x - pj.x) as f32;
                let dy = (pi.y - pj.y) as f32;
                let d = (dx * dx + dy * dy).sqrt();
                let inv = 1.0 / (1.0 + d);
                let oi = cur_owner[i];
                let oj = cur_owner[j];
                if oi == me && oj != me && oj != -1 {
                    pressure_me_to_opp += inv;
                    if d < front_dist {
                        front_dist = d;
                    }
                }
                if oi != me && oi != -1 && oj == me {
                    pressure_opp_to_me += inv;
                }
                if oi == me && oj == -1 {
                    pressure_me_to_neutral += inv;
                }
                if oi != me && oi != -1 && oj == -1 {
                    pressure_opp_to_neutral += inv;
                }
            }
        }
        let log_ratio = (1.0 + my_ships).ln() - (1.0 + opp_ships).ln();

        [
            my_ships,
            opp_ships,
            neutral_ships,
            my_ships_ext,
            opp_ships_ext,
            my_planets,
            opp_planets,
            neutral_planets,
            my_planets_ext - my_planets,
            opp_planets_ext - opp_planets,
            my_production,
            opp_production,
            my_radius,
            opp_radius,
            pressure_me_to_opp,
            pressure_opp_to_me,
            pressure_me_to_neutral,
            pressure_opp_to_neutral,
            n_planets,
            max_my_ships,
            max_opp_ships,
            front_dist,
            log_ratio,
        ]
    }
}

/// v2 summary feature set per user spec:
///   per-player (×2 for me + enemy):
///     [10 current]: ships_on_planets, ships_flying, n_static, n_orbit,
///                   n_comet, prod_static, prod_orbit, prod_comet,
///                   n_neutrals_closer_to_me, n_enemies_closer_to_me
///     [ 9 extrap ]: same as above minus ships_flying
///   neutral block (8 features):
///     n_ships, n_static, n_orbit, n_comet,
///     prod_static, prod_orbit, prod_comet, comet_time_before_gone
///
/// Total: 10 + 10 + 9 + 9 + 8 = 46.
///
/// "Enemy" = any owner that is not me AND not -1 (handles 2P and 4P).
/// Distances use the planet's current position (`planet.x`, `planet.y`)
/// — orbiting planets' future positions are intentionally NOT used for
/// the "closer to me" features, matching the user's wording.
pub mod summary_features_v2 {
    use super::*;

    pub const DIM: usize = 46;

    /// Compute min distance from object `o` to any planet whose
    /// classification function returns true. Returns `INF` if no such
    /// planet exists.
    fn min_dist_to<F: Fn(&Planet) -> bool>(
        planets: &[Planet],
        o_x: f64,
        o_y: f64,
        pred: F,
    ) -> f32 {
        let mut best = f32::INFINITY;
        for p in planets {
            if !pred(p) {
                continue;
            }
            let dx = (p.x - o_x) as f32;
            let dy = (p.y - o_y) as f32;
            let d = (dx * dx + dy * dy).sqrt();
            if d < best {
                best = d;
            }
        }
        best
    }

    /// 10-d per-player feature row for the CURRENT state.
    fn current_player_block(state: &GameState, p: i32) -> [f32; 10] {
        let mut ships_on_planets = 0.0f32;
        let mut ships_flying = 0.0f32;
        let mut n_static = 0.0f32;
        let mut n_orbit = 0.0f32;
        let mut n_comet = 0.0f32;
        let mut prod_static = 0.0f32;
        let mut prod_orbit = 0.0f32;
        let mut prod_comet = 0.0f32;
        for planet in &state.planets {
            if planet.owner != p {
                continue;
            }
            ships_on_planets += planet.ships as f32;
            let prod = planet.production as f32;
            if planet.is_comet {
                n_comet += 1.0;
                prod_comet += prod;
            } else if planet.is_orbiting {
                n_orbit += 1.0;
                prod_orbit += prod;
            } else {
                n_static += 1.0;
                prod_static += prod;
            }
        }
        for fleet in &state.fleets {
            if fleet.owner == p {
                ships_flying += fleet.ships as f32;
            }
        }
        // "closer to me" counts use current owners.
        let mut n_neutrals_closer = 0.0f32;
        let mut n_enemies_closer = 0.0f32;
        for o in &state.planets {
            if o.owner == -1 {
                let d_me = min_dist_to(&state.planets, o.x, o.y, |q| q.owner == p);
                let d_en = min_dist_to(&state.planets, o.x, o.y, |q| q.owner != p && q.owner != -1);
                if d_me < d_en {
                    n_neutrals_closer += 1.0;
                }
            } else if o.owner != p {
                let d_me = min_dist_to(&state.planets, o.x, o.y, |q| q.owner == p);
                let d_other =
                    min_dist_to(&state.planets, o.x, o.y, |q| q.owner != p && q.owner != -1 && q.id != o.id);
                if d_me < d_other {
                    n_enemies_closer += 1.0;
                }
            }
        }
        [
            ships_on_planets,
            ships_flying,
            n_static,
            n_orbit,
            n_comet,
            prod_static,
            prod_orbit,
            prod_comet,
            n_neutrals_closer,
            n_enemies_closer,
        ]
    }

    /// 9-d per-player feature row for the EXTRAPOLATED state
    /// (ships_flying is omitted — by construction, no fleets are still
    /// in flight after extrapolation).
    fn extrap_player_block(
        state: &GameState,
        p: i32,
        extrap: &std::collections::HashMap<i64, (i32, i64)>,
    ) -> [f32; 9] {
        let mut ships_on_planets = 0.0f32;
        let mut n_static = 0.0f32;
        let mut n_orbit = 0.0f32;
        let mut n_comet = 0.0f32;
        let mut prod_static = 0.0f32;
        let mut prod_orbit = 0.0f32;
        let mut prod_comet = 0.0f32;
        // Build extrap-owner-by-planet lookup once.
        // (extrap already covers all planet IDs.)
        // For "closer to me" we evaluate using extrap owners, but the
        // planet positions are unchanged (extrapolation is about owner
        // and ship counts only).
        for planet in &state.planets {
            let (eo, es) = extrap.get(&planet.id).copied().unwrap_or((planet.owner, planet.ships));
            if eo != p {
                continue;
            }
            ships_on_planets += es as f32;
            let prod = planet.production as f32;
            if planet.is_comet {
                n_comet += 1.0;
                prod_comet += prod;
            } else if planet.is_orbiting {
                n_orbit += 1.0;
                prod_orbit += prod;
            } else {
                n_static += 1.0;
                prod_static += prod;
            }
        }
        let owner_of = |id: i64| -> i32 {
            extrap
                .get(&id)
                .map(|x| x.0)
                .unwrap_or_else(|| state.planets.iter().find(|p| p.id == id).map(|p| p.owner).unwrap_or(-1))
        };
        let mut n_neutrals_closer = 0.0f32;
        let mut n_enemies_closer = 0.0f32;
        for o in &state.planets {
            let eo = owner_of(o.id);
            if eo == -1 {
                let d_me = min_dist_to(&state.planets, o.x, o.y, |q| owner_of(q.id) == p);
                let d_en = min_dist_to(&state.planets, o.x, o.y, |q| {
                    let qo = owner_of(q.id);
                    qo != p && qo != -1
                });
                if d_me < d_en {
                    n_neutrals_closer += 1.0;
                }
            } else if eo != p {
                let d_me = min_dist_to(&state.planets, o.x, o.y, |q| owner_of(q.id) == p);
                let d_other = min_dist_to(&state.planets, o.x, o.y, |q| {
                    let qo = owner_of(q.id);
                    qo != p && qo != -1 && q.id != o.id
                });
                if d_me < d_other {
                    n_enemies_closer += 1.0;
                }
            }
        }
        [
            ships_on_planets,
            n_static,
            n_orbit,
            n_comet,
            prod_static,
            prod_orbit,
            prod_comet,
            n_neutrals_closer,
            n_enemies_closer,
        ]
    }

    fn neutral_block(state: &GameState) -> [f32; 8] {
        let mut ships = 0.0f32;
        let mut n_static = 0.0f32;
        let mut n_orbit = 0.0f32;
        let mut n_comet = 0.0f32;
        let mut prod_static = 0.0f32;
        let mut prod_orbit = 0.0f32;
        let mut prod_comet = 0.0f32;
        let mut comet_time = 0.0f32;
        for planet in &state.planets {
            if planet.owner == -1 {
                ships += planet.ships as f32;
                let prod = planet.production as f32;
                if planet.is_comet {
                    n_comet += 1.0;
                    prod_comet += prod;
                } else if planet.is_orbiting {
                    n_orbit += 1.0;
                    prod_orbit += prod;
                } else {
                    n_static += 1.0;
                    prod_static += prod;
                }
            }
            // comet_time_before_gone — sum across all comets regardless
            // of owner (per user spec it's a "neutral block" stat,
            // grouped with other neutral-side info).
            if planet.is_comet {
                comet_time += state.comet_remaining(planet) as f32;
            }
        }
        [
            ships,
            n_static,
            n_orbit,
            n_comet,
            prod_static,
            prod_orbit,
            prod_comet,
            comet_time,
        ]
    }

    /// Pick the "enemy" player for `me`. In 2-player games it's the
    /// only other slot. In 4-player it's the player with the highest
    /// total ships (most threatening); per-player aggregates already
    /// sum across all opponents, so the enemy choice only affects the
    /// 10-d per-player block columns 1, 2, 3, 4, 5, 6, 7 (those are
    /// per-player counts) — i.e., every column. To keep features
    /// well-defined and to make the network see a consistent slot
    /// layout, we just emit the per-player block for `me` and for
    /// `dominant_enemy(me)`.
    fn dominant_enemy(state: &GameState, me: i32) -> i32 {
        let mut best: Option<(i32, i64)> = None;
        for p in &state.planets {
            if p.owner == -1 || p.owner == me {
                continue;
            }
            let entry = best.get_or_insert((p.owner, 0));
            // recount ships for this owner
            let total: i64 = state
                .planets
                .iter()
                .filter(|q| q.owner == p.owner)
                .map(|q| q.ships)
                .sum::<i64>()
                + state
                    .fleets
                    .iter()
                    .filter(|f| f.owner == p.owner)
                    .map(|f| f.ships)
                    .sum::<i64>();
            if total > entry.1 || best.map(|b| b.0 == p.owner).unwrap_or(false) {
                best = Some((p.owner, total));
            }
        }
        best.map(|b| b.0).unwrap_or(if me == 0 { 1 } else { 0 })
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let opp = dominant_enemy(state, me);
        let extrap = extrapolate_fleets(state);
        let me_cur = current_player_block(state, me);
        let opp_cur = current_player_block(state, opp);
        let me_ext = extrap_player_block(state, me, &extrap);
        let opp_ext = extrap_player_block(state, opp, &extrap);
        let neut = neutral_block(state);
        let mut out = [0f32; DIM];
        out[..10].copy_from_slice(&me_cur);
        out[10..20].copy_from_slice(&opp_cur);
        out[20..29].copy_from_slice(&me_ext);
        out[29..38].copy_from_slice(&opp_ext);
        out[38..46].copy_from_slice(&neut);
        out
    }
}
