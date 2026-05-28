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

/// Record a real incoming observation for live tempo/history features.
pub fn observe_root_state(state: &GameState) {
    summary_features_v9::observe_root_state(state);
}

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
    /// 58-d v3: v2 + 12 extras (tick, 8 split distances, n_static, n_orbit, AV).
    /// See `summary_features_v3::extract`.
    SummaryV3,
    /// 96-d v4: v3 + engineered matchup deltas/shares.
    /// See `summary_features_v4::extract`.
    SummaryV4,
    /// 112-d v5: v4 + time-aware forecast/pressure features.
    /// See `summary_features_v5::extract`.
    SummaryV5,
    /// 140-d v6: v5 + horizon, speed, comet timing, and rotation features.
    /// See `summary_features_v6::extract`.
    SummaryV6,
    /// 156-d v7: curated v6 without brittle sign/abs columns, plus strategy
    /// area/density/payback proxies.
    SummaryV7,
    /// 146-d v8: curated v6 without brittle sign/abs columns, plus focused
    /// phase interaction features.
    SummaryV8,
    /// 157-d v9: v8 plus causal tempo/history slopes.
    SummaryV9,
}

/// One dense layer: `out_dim` rows of length `in_dim` (row-major) plus a
/// per-output bias. Hidden layers apply ReLU; the final layer (out_dim==1)
/// is followed by tanh in `forward_raw`.
struct Layer {
    in_dim: usize,
    out_dim: usize,
    w: Vec<f32>, // [out_dim, in_dim] row-major
    b: Vec<f32>, // [out_dim]
}

struct MlpWeights {
    input_dim: usize,
    kind: InputKind,
    layers: Vec<Layer>,
}

fn read_u32_le(slice: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(slice.get(..4)?.try_into().ok()?))
}

fn detect_kind(input_dim: usize) -> Option<InputKind> {
    if input_dim == INPUT_DIM {
        Some(InputKind::Full)
    } else if input_dim == summary_features::DIM {
        Some(InputKind::Summary)
    } else if input_dim == summary_features_v2::DIM {
        Some(InputKind::SummaryV2)
    } else if input_dim == summary_features_v3::DIM {
        Some(InputKind::SummaryV3)
    } else if input_dim == summary_features_v4::DIM {
        Some(InputKind::SummaryV4)
    } else if input_dim == summary_features_v5::DIM {
        Some(InputKind::SummaryV5)
    } else if input_dim == summary_features_v6::DIM {
        Some(InputKind::SummaryV6)
    } else if input_dim == summary_features_v7::DIM {
        Some(InputKind::SummaryV7)
    } else if input_dim == summary_features_v8::DIM {
        Some(InputKind::SummaryV8)
    } else if input_dim == summary_features_v9::DIM {
        Some(InputKind::SummaryV9)
    } else {
        eprintln!(
            "[alphaow] unknown input_dim={} (expected {} full / {} summary / {} summary_v2 / {} summary_v3 / {} summary_v4 / {} summary_v5 / {} summary_v6 / {} summary_v7 / {} summary_v8 / {} summary_v9)",
            input_dim,
            INPUT_DIM,
            summary_features::DIM,
            summary_features_v2::DIM,
            summary_features_v3::DIM,
            summary_features_v4::DIM,
            summary_features_v5::DIM,
            summary_features_v6::DIM,
            summary_features_v7::DIM,
            summary_features_v8::DIM,
            summary_features_v9::DIM
        );
        None
    }
}

fn read_f32_vec(bytes: &[u8], cursor: &mut usize, count: usize) -> Option<Vec<f32>> {
    let end = cursor.checked_add(4usize.checked_mul(count)?)?;
    let slice = bytes.get(*cursor..end)?;
    let mut out = Vec::with_capacity(count);
    for chunk in slice.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    *cursor = end;
    Some(out)
}

fn parse_weights(bytes: &[u8]) -> Option<MlpWeights> {
    if bytes.len() < 16 {
        return None;
    }
    if read_u32_le(&bytes[0..4])? != WEIGHTS_MAGIC {
        return None;
    }
    let version = read_u32_le(&bytes[4..8])?;
    let input_dim = read_u32_le(&bytes[8..12])? as usize;
    let kind = detect_kind(input_dim)?;
    match version {
        1 => parse_v1(bytes, input_dim, kind),
        2 => parse_v2(bytes, input_dim, kind),
        v => {
            eprintln!("[alphaow] unsupported weights version {}", v);
            None
        }
    }
}

/// v1 (legacy): single hidden layer. Header word at [12..16] is `hidden`,
/// then w1[hidden*input], b1[hidden], w2[hidden], b2. Lifted into a
/// two-layer stack: input->hidden (relu), hidden->1 (tanh).
fn parse_v1(bytes: &[u8], input_dim: usize, kind: InputKind) -> Option<MlpWeights> {
    let hidden = read_u32_le(&bytes[12..16])? as usize;
    if hidden == 0 {
        return None;
    }
    let mut cursor = 16usize;
    let w1 = read_f32_vec(bytes, &mut cursor, hidden * input_dim)?;
    let b1 = read_f32_vec(bytes, &mut cursor, hidden)?;
    let w2 = read_f32_vec(bytes, &mut cursor, hidden)?;
    let b2 = read_f32_vec(bytes, &mut cursor, 1)?;
    Some(MlpWeights {
        input_dim,
        kind,
        layers: vec![
            Layer { in_dim: input_dim, out_dim: hidden, w: w1, b: b1 },
            Layer { in_dim: hidden, out_dim: 1, w: w2, b: b2 },
        ],
    })
}

/// v2 (deep): arbitrary dense stack. Header word at [12..16] is `n_layers`,
/// followed by `n_layers` u32 out-dims, then per layer `w[out*in]`
/// (row-major) and `b[out]`. `in_dim` chains: input_dim for the first
/// layer, the previous out_dim thereafter. The last out_dim must be 1.
fn parse_v2(bytes: &[u8], input_dim: usize, kind: InputKind) -> Option<MlpWeights> {
    let n_layers = read_u32_le(&bytes[12..16])? as usize;
    if n_layers == 0 || n_layers > 16 {
        return None;
    }
    let mut cursor = 16usize;
    let mut out_dims = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let d = read_u32_le(bytes.get(cursor..cursor + 4)?)? as usize;
        if d == 0 {
            return None;
        }
        out_dims.push(d);
        cursor += 4;
    }
    if *out_dims.last()? != 1 {
        eprintln!("[alphaow] v2 weights: final layer out_dim must be 1");
        return None;
    }
    let mut layers = Vec::with_capacity(n_layers);
    let mut in_dim = input_dim;
    for &out_dim in &out_dims {
        let w = read_f32_vec(bytes, &mut cursor, out_dim * in_dim)?;
        let b = read_f32_vec(bytes, &mut cursor, out_dim)?;
        layers.push(Layer { in_dim, out_dim, w, b });
        in_dim = out_dim;
    }
    Some(MlpWeights { input_dim, kind, layers })
}

/// Loaded model: either the legacy MLP (AOWV binary) or an XGBoost gbtree
/// dump (`bst.save_model('*.json')`). Auto-detected by leading byte.
enum Model {
    Mlp(MlpWeights),
    Xgb {
        model: crate::xgb::XgbModel,
        kind: InputKind,
    },
}

fn load_weights() -> Option<Model> {
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
    // Auto-detect: JSON XGB dump (leading '{') vs legacy AOWV binary.
    if crate::xgb::looks_like_json(&bytes) {
        let model = match crate::xgb::load(&bytes) {
            Some(m) => m,
            None => {
                eprintln!("[alphaow] failed to parse XGB JSON at {}", path);
                return None;
            }
        };
        let kind = detect_kind(model.num_feature)?;
        eprintln!(
            "[alphaow] loaded XGB value net (kind={:?}, num_feature={}, base_score_logit={:.4}) from {}",
            kind, model.num_feature, model.base_score_logit, path
        );
        return Some(Model::Xgb { model, kind });
    }
    let w = parse_weights(&bytes);
    match &w {
        Some(mw) => {
            let arch: Vec<String> = std::iter::once(mw.input_dim)
                .chain(mw.layers.iter().map(|l| l.out_dim))
                .map(|d| d.to_string())
                .collect();
            eprintln!(
                "[alphaow] loaded value net (kind={:?}, arch={}) from {}",
                mw.kind,
                arch.join("->"),
                path
            );
        }
        None => eprintln!("[alphaow] failed to parse value net weights at {}", path),
    }
    w.map(Model::Mlp)
}

static WEIGHTS: OnceLock<Option<Model>> = OnceLock::new();

fn weights() -> Option<&'static Model> {
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
    // Ping-pong scratch buffers reused across calls (the deep stack means a
    // fixed-size array no longer fits). All but the last layer apply ReLU;
    // the final layer (out_dim==1) passes through tanh below.
    thread_local! {
        static CUR: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
        static NXT: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }
    CUR.with(|cur_cell| {
        NXT.with(|nxt_cell| {
            let mut cur = cur_cell.borrow_mut();
            let mut nxt = nxt_cell.borrow_mut();
            cur.clear();
            cur.extend_from_slice(input);
            let n = w.layers.len();
            for (li, layer) in w.layers.iter().enumerate() {
                let is_last = li + 1 == n;
                nxt.clear();
                nxt.resize(layer.out_dim, 0.0);
                for o in 0..layer.out_dim {
                    let row = &layer.w[o * layer.in_dim..(o + 1) * layer.in_dim];
                    let s = dot_neon(row, &cur[..], layer.b[o], layer.in_dim);
                    nxt[o] = if is_last || s > 0.0 { s } else { 0.0 };
                }
                std::mem::swap(&mut *cur, &mut *nxt);
            }
            cur[0].tanh()
        })
    })
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
    let m = weights()?;
    let y = match m {
        Model::Mlp(w) => match w.kind {
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
            InputKind::SummaryV3 => {
                let feats = summary_features_v3::extract(state, me);
                forward_raw(w, &feats)
            }
            InputKind::SummaryV4 => {
                let feats = summary_features_v4::extract(state, me);
                forward_raw(w, &feats)
            }
            InputKind::SummaryV5 => {
                let feats = summary_features_v5::extract(state, me);
                forward_raw(w, &feats)
            }
            InputKind::SummaryV6 => {
                let feats = summary_features_v6::extract(state, me);
                forward_raw(w, &feats)
            }
            InputKind::SummaryV7 => {
                let feats = summary_features_v7::extract(state, me);
                forward_raw(w, &feats)
            }
            InputKind::SummaryV8 => {
                let feats = summary_features_v8::extract(state, me);
                forward_raw(w, &feats)
            }
            InputKind::SummaryV9 => {
                let feats = summary_features_v9::extract(state, me);
                forward_raw(w, &feats)
            }
        },
        Model::Xgb { model, kind } => match kind {
            InputKind::Full => {
                let features = extract_features(state, me);
                // Concatenate to one flat slice for XGB.
                let mut scratch = Vec::with_capacity(INPUT_DIM);
                scratch.extend_from_slice(features.current.as_ref());
                scratch.extend_from_slice(features.extrap.as_ref());
                scratch.extend_from_slice(features.dist.as_ref());
                model.predict_value(&scratch)
            }
            InputKind::Summary => {
                let feats = summary_features::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::SummaryV2 => {
                let feats = summary_features_v2::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::SummaryV3 => {
                let feats = summary_features_v3::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::SummaryV4 => {
                let feats = summary_features_v4::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::SummaryV5 => {
                let feats = summary_features_v5::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::SummaryV6 => {
                let feats = summary_features_v6::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::SummaryV7 => {
                let feats = summary_features_v7::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::SummaryV8 => {
                let feats = summary_features_v8::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::SummaryV9 => {
                let feats = summary_features_v9::extract(state, me);
                model.predict_value(&feats)
            }
        },
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

/// 58-d: 46-d summary_v2 + 12-d extras (user-requested, leak-aware):
///   [46]: same as summary_v2
///   [46]: tick
///   [47..51]: 4 split distances NOW
///       my_static→their_static, my_static→their_orb,
///       my_orb→their_static,    my_orb→their_orb
///   [51..55]: same 4 distances after `extrapolate_fleets`
///   [55]: n_total_static
///   [56]: n_total_orbit
///   [57]: angular_velocity
///
/// Matches the `bin/extract_v4` offline extractor byte-for-byte (same
/// underlying `extrapolate_fleets` + same type buckets).
pub mod summary_features_v3 {
    use super::*;

    pub const DIM: usize = 58;
    const EXTRA_DIM: usize = 12;

    #[derive(Copy, Clone)]
    enum PType { Static, Orbit }

    fn matches(p: &Planet, t: PType) -> bool {
        if p.is_comet { return false; }
        match t {
            PType::Static => !p.is_orbiting,
            PType::Orbit  =>  p.is_orbiting,
        }
    }

    fn min_pair_dist<F1, F2>(planets: &[Planet], is_a: F1, is_b: F2) -> f32
    where F1: Fn(&Planet) -> bool, F2: Fn(&Planet) -> bool {
        let mut best = f32::INFINITY;
        for a in planets.iter().filter(|p| is_a(p)) {
            for b in planets.iter().filter(|p| is_b(p)) {
                let dx = (a.x - b.x) as f32;
                let dy = (a.y - b.y) as f32;
                let d2 = dx * dx + dy * dy;
                if d2 < best { best = d2; }
            }
        }
        if best.is_finite() { best.sqrt() } else { 0.0 }
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let v2 = summary_features_v2::extract(state, me);
        let mut out = [0f32; DIM];
        out[..summary_features_v2::DIM].copy_from_slice(&v2);

        let mut n_static = 0u32;
        let mut n_orbit  = 0u32;
        for p in &state.planets {
            if p.is_comet { continue; }
            if p.is_orbiting { n_orbit += 1; } else { n_static += 1; }
        }

        let is_mine  = |p: &Planet| p.owner == me;
        let is_enemy = |p: &Planet| p.owner != me && p.owner != -1;
        let now_ss = min_pair_dist(&state.planets,
            |p| is_mine(p)  && matches(p, PType::Static),
            |p| is_enemy(p) && matches(p, PType::Static));
        let now_so = min_pair_dist(&state.planets,
            |p| is_mine(p)  && matches(p, PType::Static),
            |p| is_enemy(p) && matches(p, PType::Orbit));
        let now_os = min_pair_dist(&state.planets,
            |p| is_mine(p)  && matches(p, PType::Orbit),
            |p| is_enemy(p) && matches(p, PType::Static));
        let now_oo = min_pair_dist(&state.planets,
            |p| is_mine(p)  && matches(p, PType::Orbit),
            |p| is_enemy(p) && matches(p, PType::Orbit));

        let ext_map = extrapolate_fleets(state);
        let ext_owner = |p: &Planet| ext_map.get(&p.id).map(|x| x.0).unwrap_or(p.owner);
        let ext_is_mine  = |p: &Planet| ext_owner(p) == me;
        let ext_is_enemy = |p: &Planet| { let o = ext_owner(p); o != me && o != -1 };
        let ext_ss = min_pair_dist(&state.planets,
            |p| ext_is_mine(p)  && matches(p, PType::Static),
            |p| ext_is_enemy(p) && matches(p, PType::Static));
        let ext_so = min_pair_dist(&state.planets,
            |p| ext_is_mine(p)  && matches(p, PType::Static),
            |p| ext_is_enemy(p) && matches(p, PType::Orbit));
        let ext_os = min_pair_dist(&state.planets,
            |p| ext_is_mine(p)  && matches(p, PType::Orbit),
            |p| ext_is_enemy(p) && matches(p, PType::Static));
        let ext_oo = min_pair_dist(&state.planets,
            |p| ext_is_mine(p)  && matches(p, PType::Orbit),
            |p| ext_is_enemy(p) && matches(p, PType::Orbit));

        let extras: [f32; EXTRA_DIM] = [
            state.step as f32,
            now_ss, now_so, now_os, now_oo,
            ext_ss, ext_so, ext_os, ext_oo,
            n_static as f32, n_orbit as f32,
            state.angular_velocity as f32,
        ];
        out[summary_features_v2::DIM..].copy_from_slice(&extras);
        out
    }
}

/// 96-d: summary_v3 plus engineered matchup features used by the XGB
/// retraining pipeline (`train/engineered_features.py`).
pub mod summary_features_v4 {
    use super::*;

    pub const ENGINEERED_DIM: usize = 38;
    pub const DIM: usize = summary_features_v3::DIM + ENGINEERED_DIM;

    #[inline]
    fn share(a: f32, b: f32) -> f32 {
        (a - b) / (a + b).max(1.0)
    }

    #[inline]
    fn safe_min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
        let mut best = f32::INFINITY;
        for v in [a, b, c, d] {
            if v > 0.0 && v < best {
                best = v;
            }
        }
        if best.is_finite() { best } else { 1_000_000.0 }
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let base = summary_features_v3::extract(state, me);
        let mut out = [0f32; DIM];
        out[..summary_features_v3::DIM].copy_from_slice(&base);

        let me_cur_ships = base[0];
        let me_cur_flying = base[1];
        let me_cur_static = base[2];
        let me_cur_orbit = base[3];
        let me_cur_comet = base[4];
        let me_cur_prod_static = base[5];
        let me_cur_prod_orbit = base[6];
        let me_cur_prod_comet = base[7];

        let op_cur_ships = base[10];
        let op_cur_flying = base[11];
        let op_cur_static = base[12];
        let op_cur_orbit = base[13];
        let op_cur_comet = base[14];
        let op_cur_prod_static = base[15];
        let op_cur_prod_orbit = base[16];
        let op_cur_prod_comet = base[17];

        let me_ext_ships = base[20];
        let me_ext_static = base[21];
        let me_ext_orbit = base[22];
        let me_ext_comet = base[23];
        let me_ext_prod_static = base[24];
        let me_ext_prod_orbit = base[25];
        let me_ext_prod_comet = base[26];

        let op_ext_ships = base[29];
        let op_ext_static = base[30];
        let op_ext_orbit = base[31];
        let op_ext_comet = base[32];
        let op_ext_prod_static = base[33];
        let op_ext_prod_orbit = base[34];
        let op_ext_prod_comet = base[35];

        let me_cur_prod = me_cur_prod_static + me_cur_prod_orbit + me_cur_prod_comet;
        let op_cur_prod = op_cur_prod_static + op_cur_prod_orbit + op_cur_prod_comet;
        let me_ext_prod = me_ext_prod_static + me_ext_prod_orbit + me_ext_prod_comet;
        let op_ext_prod = op_ext_prod_static + op_ext_prod_orbit + op_ext_prod_comet;
        let me_cur_planets = me_cur_static + me_cur_orbit + me_cur_comet;
        let op_cur_planets = op_cur_static + op_cur_orbit + op_cur_comet;
        let me_ext_planets = me_ext_static + me_ext_orbit + me_ext_comet;
        let op_ext_planets = op_ext_static + op_ext_orbit + op_ext_comet;
        let cur_ship_diff = (me_cur_ships + me_cur_flying) - (op_cur_ships + op_cur_flying);
        let ext_ship_diff = me_ext_ships - op_ext_ships;
        let cur_prod_diff = me_cur_prod - op_cur_prod;
        let ext_prod_diff = me_ext_prod - op_ext_prod;

        let now_ss = base[47];
        let now_so = base[48];
        let now_os = base[49];
        let now_oo = base[50];
        let ext_ss = base[51];
        let ext_so = base[52];
        let ext_os = base[53];
        let ext_oo = base[54];
        let now_min = safe_min4(now_ss, now_so, now_os, now_oo);
        let ext_min = safe_min4(ext_ss, ext_so, ext_os, ext_oo);
        let tick_frac = (base[46] / 500.0).clamp(0.0, 1.0);

        let engineered: [f32; ENGINEERED_DIM] = [
            me_cur_ships - op_cur_ships,
            me_cur_flying - op_cur_flying,
            cur_ship_diff,
            share(me_cur_ships + me_cur_flying, op_cur_ships + op_cur_flying),
            me_cur_static - op_cur_static,
            me_cur_orbit - op_cur_orbit,
            me_cur_comet - op_cur_comet,
            me_cur_planets - op_cur_planets,
            me_cur_prod_static - op_cur_prod_static,
            me_cur_prod_orbit - op_cur_prod_orbit,
            me_cur_prod_comet - op_cur_prod_comet,
            cur_prod_diff,
            share(me_cur_prod, op_cur_prod),
            me_ext_ships - op_ext_ships,
            me_ext_static - op_ext_static,
            me_ext_orbit - op_ext_orbit,
            me_ext_comet - op_ext_comet,
            me_ext_planets - op_ext_planets,
            me_ext_prod_static - op_ext_prod_static,
            me_ext_prod_orbit - op_ext_prod_orbit,
            me_ext_prod_comet - op_ext_prod_comet,
            ext_prod_diff,
            share(me_ext_prod, op_ext_prod),
            ext_ship_diff - cur_ship_diff,
            (me_ext_static - op_ext_static) - (me_cur_static - op_cur_static),
            (me_ext_orbit - op_ext_orbit) - (me_cur_orbit - op_cur_orbit),
            ext_prod_diff - cur_prod_diff,
            now_min,
            ext_min,
            ext_min - now_min,
            now_so.min(now_os) - now_ss.min(now_oo),
            ext_so.min(ext_os) - ext_ss.min(ext_oo),
            tick_frac,
            1.0 - tick_frac,
            if tick_frac < 1.0 / 6.0 { 1.0 } else { 0.0 },
            if tick_frac >= 1.0 / 6.0 && tick_frac < 1.0 / 3.0 { 1.0 } else { 0.0 },
            if tick_frac >= 1.0 / 3.0 && tick_frac < 2.0 / 3.0 { 1.0 } else { 0.0 },
            if tick_frac >= 2.0 / 3.0 { 1.0 } else { 0.0 },
        ];
        out[summary_features_v3::DIM..].copy_from_slice(&engineered);
        out
    }
}

/// 112-d: summary_v4 plus time-aware forecast and pressure features used by
/// the current XGB retraining pipeline (`train/engineered_features.py`).
pub mod summary_features_v5 {
    use super::*;

    pub const EXTRA_DIM: usize = 16;
    pub const ENGINEERED_DIM: usize = summary_features_v4::ENGINEERED_DIM + EXTRA_DIM;
    pub const DIM: usize = summary_features_v3::DIM + ENGINEERED_DIM;

    #[inline]
    fn share(a: f32, b: f32) -> f32 {
        (a - b) / (a + b).max(1.0)
    }

    #[inline]
    fn commitment(flying: f32, stationed: f32) -> f32 {
        flying / (flying + stationed).max(1.0)
    }

    #[inline]
    fn safe_min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
        let mut best = f32::INFINITY;
        for v in [a, b, c, d] {
            if v > 0.0 && v < best {
                best = v;
            }
        }
        if best.is_finite() { best } else { 1_000_000.0 }
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let v4 = summary_features_v4::extract(state, me);
        let mut out = [0f32; DIM];
        out[..summary_features_v4::DIM].copy_from_slice(&v4);
        let base = &v4[..summary_features_v3::DIM];

        let me_cur_ships = base[0];
        let me_cur_flying = base[1];
        let me_cur_prod = base[5] + base[6] + base[7];
        let op_cur_ships = base[10];
        let op_cur_flying = base[11];
        let op_cur_prod = base[15] + base[16] + base[17];
        let me_ext_ships = base[20];
        let me_ext_prod = base[24] + base[25] + base[26];
        let op_ext_ships = base[29];
        let op_ext_prod = base[33] + base[34] + base[35];

        let cur_ship_diff = (me_cur_ships + me_cur_flying) - (op_cur_ships + op_cur_flying);
        let ext_ship_diff = me_ext_ships - op_ext_ships;
        let cur_prod_diff = me_cur_prod - op_cur_prod;
        let ext_prod_diff = me_ext_prod - op_ext_prod;
        let cur_ship_share = share(me_cur_ships + me_cur_flying, op_cur_ships + op_cur_flying);
        let cur_prod_share = share(me_cur_prod, op_cur_prod);

        let now_min = safe_min4(base[47], base[48], base[49], base[50]);
        let ext_min = safe_min4(base[51], base[52], base[53], base[54]);
        let tick_frac = (base[46] / 500.0).clamp(0.0, 1.0);
        let remaining_frac = 1.0 - tick_frac;
        let remaining_ticks = (500.0 - base[46]).max(0.0);
        let horizon_100 = remaining_ticks.min(100.0);
        let my_commit = commitment(me_cur_flying, me_cur_ships);
        let opp_commit = commitment(op_cur_flying, op_cur_ships);

        let extra: [f32; EXTRA_DIM] = [
            cur_prod_diff * remaining_ticks,
            cur_prod_diff * horizon_100,
            cur_ship_diff + cur_prod_diff * remaining_ticks,
            cur_ship_diff + cur_prod_diff * horizon_100,
            ext_prod_diff * remaining_ticks,
            ext_ship_diff + ext_prod_diff * remaining_ticks,
            cur_prod_share * remaining_frac,
            cur_ship_share * tick_frac,
            me_cur_flying + op_cur_flying,
            my_commit,
            opp_commit,
            my_commit - opp_commit,
            cur_ship_diff / (1.0 + now_min),
            cur_prod_diff / (1.0 + now_min),
            ext_ship_diff / (1.0 + ext_min),
            ext_prod_diff / (1.0 + ext_min),
        ];
        out[summary_features_v4::DIM..].copy_from_slice(&extra);
        out
    }
}

/// 140-d: summary_v5 plus horizon, fleet-speed, comet-spawn, and rotation
/// features used by the current XGB retraining pipeline.
pub mod summary_features_v6 {
    use super::*;

    pub const EXTRA_DIM: usize = 28;
    pub const ENGINEERED_DIM: usize = summary_features_v5::ENGINEERED_DIM + EXTRA_DIM;
    pub const DIM: usize = summary_features_v3::DIM + ENGINEERED_DIM;

    #[inline]
    fn fleet_speed(ships: f32) -> f32 {
        let s = ships.max(1.0);
        let speed = 1.0 + 5.0 * (s.ln() / 1000.0f32.ln()).powf(1.5);
        speed.clamp(1.0, 6.0)
    }

    #[inline]
    fn signed_log1p(x: f32) -> f32 {
        x.signum() * x.abs().ln_1p()
    }

    #[inline]
    fn ticks_to_next_comet(tick: f32) -> f32 {
        let mut best = 500.0;
        for spawn in [50.0, 150.0, 250.0, 350.0, 450.0] {
            let dt = spawn - tick;
            if dt >= 0.0 && dt < best {
                best = dt;
            }
        }
        best
    }

    #[inline]
    fn safe_min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
        let mut best = f32::INFINITY;
        for v in [a, b, c, d] {
            if v > 0.0 && v < best {
                best = v;
            }
        }
        if best.is_finite() { best } else { 1_000_000.0 }
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let v5 = summary_features_v5::extract(state, me);
        let mut out = [0f32; DIM];
        out[..summary_features_v5::DIM].copy_from_slice(&v5);
        let base = &v5[..summary_features_v3::DIM];

        let me_cur_ships = base[0];
        let me_cur_flying = base[1];
        let me_cur_orbit = base[3];
        let me_cur_prod = base[5] + base[6] + base[7];
        let op_cur_ships = base[10];
        let op_cur_flying = base[11];
        let op_cur_orbit = base[13];
        let op_cur_prod = base[15] + base[16] + base[17];
        let me_ext_ships = base[20];
        let me_ext_prod = base[24] + base[25] + base[26];
        let op_ext_ships = base[29];
        let op_ext_prod = base[33] + base[34] + base[35];

        let cur_ship_diff = (me_cur_ships + me_cur_flying) - (op_cur_ships + op_cur_flying);
        let ext_ship_diff = me_ext_ships - op_ext_ships;
        let cur_prod_diff = me_cur_prod - op_cur_prod;
        let ext_prod_diff = me_ext_prod - op_ext_prod;
        let remaining_ticks = (500.0 - base[46]).max(0.0);
        let horizon_25 = remaining_ticks.min(25.0);
        let horizon_50 = remaining_ticks.min(50.0);
        let horizon_100 = remaining_ticks.min(100.0);
        let horizon_150 = remaining_ticks.min(150.0);
        let cur_adv_25 = cur_ship_diff + cur_prod_diff * horizon_25;
        let cur_adv_50 = cur_ship_diff + cur_prod_diff * horizon_50;
        let cur_adv_100 = cur_ship_diff + cur_prod_diff * horizon_100;
        let cur_adv_150 = cur_ship_diff + cur_prod_diff * horizon_150;
        let cur_adv_remaining = cur_ship_diff + cur_prod_diff * remaining_ticks;
        let ext_adv_100 = ext_ship_diff + ext_prod_diff * horizon_100;

        let my_stationed_speed = fleet_speed(me_cur_ships);
        let op_stationed_speed = fleet_speed(op_cur_ships);
        let my_total_speed = fleet_speed(me_cur_ships + me_cur_flying);
        let op_total_speed = fleet_speed(op_cur_ships + op_cur_flying);
        let now_min = safe_min4(base[47], base[48], base[49], base[50]);
        let ext_min = safe_min4(base[51], base[52], base[53], base[54]);
        let ticks_to_comet = ticks_to_next_comet(base[46]);
        let angular_velocity = base[57];

        let extra: [f32; EXTRA_DIM] = [
            cur_prod_diff * horizon_25,
            cur_prod_diff * horizon_50,
            cur_adv_25,
            cur_adv_50,
            cur_adv_150,
            ext_adv_100,
            signed_log1p(cur_adv_100),
            cur_adv_100.abs(),
            cur_adv_100.signum(),
            signed_log1p(cur_adv_remaining),
            signed_log1p(cur_ship_diff),
            signed_log1p(cur_prod_diff),
            my_stationed_speed,
            op_stationed_speed,
            my_stationed_speed - op_stationed_speed,
            my_total_speed,
            op_total_speed,
            my_total_speed - op_total_speed,
            now_min / my_stationed_speed,
            ext_min / my_stationed_speed,
            now_min / op_stationed_speed,
            ticks_to_comet,
            ticks_to_comet / 500.0,
            if ticks_to_comet <= 25.0 { 1.0 } else { 0.0 },
            if ticks_to_comet <= 50.0 { 1.0 } else { 0.0 },
            (me_cur_orbit - op_cur_orbit) * angular_velocity,
            (me_cur_orbit + op_cur_orbit) * angular_velocity,
            angular_velocity * 100.0,
        ];
        out[summary_features_v5::DIM..].copy_from_slice(&extra);
        out
    }
}

/// 156-d: curated v6 without brittle sign/absolute projected-margin columns,
/// plus strategy proxies for payback, area control, and garrison density.
pub mod summary_features_v7 {
    use super::*;

    pub const EXTRA_DIM: usize = 44;
    pub const ENGINEERED_DIM: usize = summary_features_v5::ENGINEERED_DIM + EXTRA_DIM;
    pub const DIM: usize = summary_features_v3::DIM + ENGINEERED_DIM;

    #[inline]
    fn pos(x: f32) -> f32 {
        x.max(0.0)
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let v6 = summary_features_v6::extract(state, me);
        let mut out = [0f32; DIM];
        let keep_prefix = summary_features_v5::DIM + 7;
        out[..keep_prefix].copy_from_slice(&v6[..keep_prefix]);
        let v6_tail_start = summary_features_v5::DIM + 9;
        let v6_tail_len = summary_features_v6::DIM - v6_tail_start;
        out[keep_prefix..keep_prefix + v6_tail_len].copy_from_slice(&v6[v6_tail_start..]);
        let write = keep_prefix + v6_tail_len;

        let base = &v6[..summary_features_v3::DIM];
        let me_cur_ships = base[0];
        let me_cur_flying = base[1];
        let me_cur_static = base[2];
        let me_cur_orbit = base[3];
        let me_cur_comet = base[4];
        let me_cur_prod = base[5] + base[6] + base[7];
        let me_cur_neutrals_closer = base[8];
        let me_cur_enemies_closer = base[9];
        let op_cur_ships = base[10];
        let op_cur_flying = base[11];
        let op_cur_static = base[12];
        let op_cur_orbit = base[13];
        let op_cur_comet = base[14];
        let op_cur_prod = base[15] + base[16] + base[17];
        let op_cur_neutrals_closer = base[18];
        let op_cur_enemies_closer = base[19];

        let my_planets = me_cur_static + me_cur_orbit + me_cur_comet;
        let op_planets = op_cur_static + op_cur_orbit + op_cur_comet;
        let my_planets_safe = my_planets.max(1.0);
        let op_planets_safe = op_planets.max(1.0);
        let cur_ship_diff = (me_cur_ships + me_cur_flying) - (op_cur_ships + op_cur_flying);
        let cur_prod_diff = me_cur_prod - op_cur_prod;
        let cur_ship_share = (cur_ship_diff) / (me_cur_ships + me_cur_flying + op_cur_ships + op_cur_flying).max(1.0);
        let cur_prod_share = cur_prod_diff / (me_cur_prod + op_cur_prod).max(1.0);
        let neutral_closer_diff = me_cur_neutrals_closer - op_cur_neutrals_closer;
        let enemy_reach_diff = me_cur_enemies_closer - op_cur_enemies_closer;
        let prod_payback_turns = if cur_prod_diff > 0.0 {
            (pos(-cur_ship_diff) / cur_prod_diff.max(1.0)).min(500.0)
        } else {
            500.0
        };

        let extra: [f32; 18] = [
            prod_payback_turns,
            pos(-cur_ship_diff) / pos(cur_prod_diff).max(1.0),
            pos(cur_prod_share) * pos(-cur_ship_share),
            pos(cur_ship_share) * pos(-cur_prod_share),
            cur_prod_share * (1.0 - cur_ship_share.abs().min(1.0)),
            neutral_closer_diff,
            enemy_reach_diff,
            neutral_closer_diff + enemy_reach_diff + my_planets - op_planets,
            neutral_closer_diff * cur_prod_share,
            enemy_reach_diff * cur_ship_share,
            (me_cur_ships + me_cur_flying) / my_planets_safe,
            (op_cur_ships + op_cur_flying) / op_planets_safe,
            (me_cur_ships + me_cur_flying) / my_planets_safe - (op_cur_ships + op_cur_flying) / op_planets_safe,
            me_cur_prod / my_planets_safe,
            op_cur_prod / op_planets_safe,
            me_cur_prod / my_planets_safe - op_cur_prod / op_planets_safe,
            me_cur_flying / my_planets_safe,
            op_cur_flying / op_planets_safe,
        ];
        out[write..].copy_from_slice(&extra);
        out
    }
}

/// 146-d: curated v6 without brittle sign/absolute projected-margin columns,
/// plus focused phase interactions aimed at transition-game errors.
pub mod summary_features_v8 {
    use super::*;

    pub const EXTRA_DIM: usize = summary_features_v6::EXTRA_DIM - 2 + 8;
    pub const ENGINEERED_DIM: usize = summary_features_v5::ENGINEERED_DIM + EXTRA_DIM;
    pub const DIM: usize = summary_features_v3::DIM + ENGINEERED_DIM;

    #[inline]
    fn fleet_speed(ships: f32) -> f32 {
        let s = ships.max(1.0);
        let speed = 1.0 + 5.0 * (s.ln() / 1000.0f32.ln()).powf(1.5);
        speed.clamp(1.0, 6.0)
    }

    #[inline]
    fn safe_min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
        let mut best = f32::INFINITY;
        for v in [a, b, c, d] {
            if v > 0.0 && v < best {
                best = v;
            }
        }
        if best.is_finite() { best } else { 1_000_000.0 }
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let v6 = summary_features_v6::extract(state, me);
        let mut out = [0f32; DIM];

        let keep_prefix = summary_features_v5::DIM + 7;
        out[..keep_prefix].copy_from_slice(&v6[..keep_prefix]);
        let v6_tail_start = summary_features_v5::DIM + 9;
        let v6_tail_len = summary_features_v6::DIM - v6_tail_start;
        out[keep_prefix..keep_prefix + v6_tail_len].copy_from_slice(&v6[v6_tail_start..]);
        let write = keep_prefix + v6_tail_len;

        let base = &v6[..summary_features_v3::DIM];
        let me_cur_ships = base[0];
        let me_cur_flying = base[1];
        let me_cur_prod = base[5] + base[6] + base[7];
        let op_cur_ships = base[10];
        let op_cur_flying = base[11];
        let op_cur_prod = base[15] + base[16] + base[17];

        let cur_ship_diff = (me_cur_ships + me_cur_flying) - (op_cur_ships + op_cur_flying);
        let cur_prod_diff = me_cur_prod - op_cur_prod;
        let cur_ship_share =
            cur_ship_diff / (me_cur_ships + me_cur_flying + op_cur_ships + op_cur_flying).max(1.0);
        let cur_prod_share = cur_prod_diff / (me_cur_prod + op_cur_prod).max(1.0);
        let remaining_ticks = (500.0 - base[46]).max(0.0);
        let cur_adv_50 = cur_ship_diff + cur_prod_diff * remaining_ticks.min(50.0);
        let cur_adv_100 = cur_ship_diff + cur_prod_diff * remaining_ticks.min(100.0);
        let tick_frac = (base[46] / 500.0).clamp(0.0, 1.0);
        let early_phase = if tick_frac < 1.0 / 6.0 { 1.0 } else { 0.0 };
        let transition_phase = if tick_frac >= 1.0 / 6.0 && tick_frac < 1.0 / 3.0 {
            1.0
        } else {
            0.0
        };
        let midgame_phase = if tick_frac >= 1.0 / 3.0 && tick_frac < 2.0 / 3.0 {
            1.0
        } else {
            0.0
        };
        let endgame_phase = if tick_frac >= 2.0 / 3.0 { 1.0 } else { 0.0 };
        let total_speed_diff =
            fleet_speed(me_cur_ships + me_cur_flying) - fleet_speed(op_cur_ships + op_cur_flying);
        let now_min = safe_min4(base[47], base[48], base[49], base[50]);

        let extra: [f32; 8] = [
            cur_adv_50 * transition_phase,
            cur_adv_100 * transition_phase,
            (me_cur_flying - op_cur_flying) * transition_phase,
            total_speed_diff * transition_phase,
            now_min * transition_phase,
            cur_prod_share * early_phase,
            cur_ship_share * midgame_phase,
            cur_ship_share * endgame_phase,
        ];
        out[write..].copy_from_slice(&extra);
        out
    }
}

/// 157-d: summary_v8 plus causal 50-turn tempo slopes from real observed
/// history. Call `observe_root_state` once per incoming game observation.
pub mod summary_features_v9 {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    pub const TEMPO_DIM: usize = 11;
    pub const DIM: usize = summary_features_v8::DIM + TEMPO_DIM;
    const WINDOW: f32 = 50.0;
    const BASE: usize = summary_features_v3::DIM;
    const METRIC_DIM: usize = 9;
    const PROD_DIFF: usize = BASE + 11;
    const SHIPS_TOTAL_DIFF: usize = BASE + 2;
    const SHIPS_PLANETS_DIFF: usize = BASE;
    const PLANET_COUNT_DIFF: usize = BASE + 7;
    const STATIC_COUNT_DIFF: usize = BASE + 4;
    const PROD_SHARE: usize = BASE + 12;
    const SHIPS_SHARE: usize = BASE + 3;
    const ADV_100: usize = BASE + 41;
    const FLYING_COMMITMENT_DIFF: usize = BASE + 49;

    #[derive(Clone, Copy)]
    struct TempoSample {
        step: f32,
        metrics: [f32; METRIC_DIM],
    }

    thread_local! {
        static HISTORY: RefCell<HashMap<i32, Vec<TempoSample>>> = RefCell::new(HashMap::new());
    }

    fn metrics(core: &[f32; summary_features_v8::DIM]) -> [f32; METRIC_DIM] {
        [
            core[PROD_DIFF],
            core[SHIPS_TOTAL_DIFF],
            core[SHIPS_PLANETS_DIFF],
            core[PLANET_COUNT_DIFF],
            core[STATIC_COUNT_DIFF],
            core[PROD_SHARE],
            core[SHIPS_SHARE],
            core[ADV_100],
            core[FLYING_COMMITMENT_DIFF],
        ]
    }

    fn tempo_from_samples(samples: &[TempoSample], step: f32, current: [f32; METRIC_DIM]) -> [f32; TEMPO_DIM] {
        let mut points: Vec<TempoSample> = samples
            .iter()
            .copied()
            .filter(|s| s.step <= step && s.step >= step - WINDOW)
            .collect();
        if !points.iter().any(|s| (s.step - step).abs() < 1e-6) {
            points.push(TempoSample { step, metrics: current });
        }
        points.sort_by(|a, b| a.step.partial_cmp(&b.step).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = [0.0f32; TEMPO_DIM];
        if points.len() < 2 {
            return out;
        }
        let n = points.len() as f32;
        let sum_x: f32 = points.iter().map(|p| p.step).sum();
        let sum_x2: f32 = points.iter().map(|p| p.step * p.step).sum();
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() > 1e-6 {
            for j in 0..METRIC_DIM {
                let sum_y: f32 = points.iter().map(|p| p.metrics[j]).sum();
                let sum_xy: f32 = points.iter().map(|p| p.step * p.metrics[j]).sum();
                out[j] = (n * sum_xy - sum_x * sum_y) / denom;
            }
        }
        let min_step = points.first().map(|p| p.step).unwrap_or(step);
        out[9] = ((step - min_step) / WINDOW).clamp(0.0, 1.0);
        out[10] = out[1] * WINDOW + out[0] * (0.5 * WINDOW * WINDOW);
        out
    }

    pub fn observe_root_state(state: &GameState) {
        let me = state.player;
        let step = state.step as f32;
        let core = summary_features_v8::extract(state, me);
        let sample = TempoSample { step, metrics: metrics(&core) };
        HISTORY.with(|cell| {
            let mut history = cell.borrow_mut();
            let samples = history.entry(me).or_default();
            if samples.last().map(|s| step < s.step).unwrap_or(false) {
                samples.clear();
            }
            if samples.last().map(|s| (step - s.step).abs() < 1e-6).unwrap_or(false) {
                if let Some(last) = samples.last_mut() {
                    *last = sample;
                }
            } else {
                samples.push(sample);
            }
            let keep_from = step - WINDOW;
            samples.retain(|s| s.step >= keep_from);
        });
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let core = summary_features_v8::extract(state, me);
        let current = metrics(&core);
        let step = state.step as f32;
        let tempo = HISTORY.with(|cell| {
            let history = cell.borrow();
            let samples = history.get(&me).map(|v| v.as_slice()).unwrap_or(&[]);
            tempo_from_samples(samples, step, current)
        });
        let mut out = [0f32; DIM];
        out[..summary_features_v8::DIM].copy_from_slice(&core);
        out[summary_features_v8::DIM..].copy_from_slice(&tempo);
        out
    }
}
