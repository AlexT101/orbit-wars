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
use std::sync::atomic::{AtomicBool, Ordering};
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
    /// 170-d v10: v9 plus 13 spatial/reachability features.
    SummaryV10,
    /// 236-d 4P-specific layout with global/map features, my block, three
    /// threat-ordered opponent blocks, aggregate enemy features, and pairwise
    /// matchup/rank features.
    FourPV1,
    /// 278-d 4P v2: v1 plus pairwise placement probabilities/margins and
    /// dogpile/exposure features.
    FourPV2,
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
    } else if input_dim == summary_features_v10::DIM {
        Some(InputKind::SummaryV10)
    } else if input_dim == summary_features_4p_v1::DIM {
        Some(InputKind::FourPV1)
    } else if input_dim == summary_features_4p_v1::DIM_V2 {
        Some(InputKind::FourPV2)
    } else {
        eprintln!(
            "[alphaow] unknown input_dim={} (expected {} full / {} summary / {} summary_v2 / {} summary_v3 / {} summary_v4 / {} summary_v5 / {} summary_v6 / {} summary_v7 / {} summary_v8 / {} summary_v9 / {} 4p_v1 / {} 4p_v2)",
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
            summary_features_v9::DIM,
            summary_features_4p_v1::DIM,
            summary_features_4p_v1::DIM_V2
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

struct ModelSet {
    default: Option<Model>,
    two_p: Option<Model>,
    four_p: Option<Model>,
}

fn load_model_from_path(path: &str, label: &str) -> Option<Model> {
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[alphaow] could not read {} weights at {}: {}", label, path, e);
            return None;
        }
    };
    // Auto-detect: JSON XGB dump (leading '{') vs legacy AOWV binary.
    if crate::xgb::looks_like_json(&bytes) {
        let model = match crate::xgb::load(&bytes) {
            Some(m) => m,
            None => {
                eprintln!("[alphaow] failed to parse {} XGB JSON at {}", label, path);
                return None;
            }
        };
        let kind = detect_kind(model.num_feature)?;
        eprintln!(
            "[alphaow] loaded {} XGB value net (kind={:?}, objective={:?}, num_feature={}, base_score={:.4}) from {}",
            label, kind, model.objective, model.num_feature, model.base_score, path
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
                "[alphaow] loaded {} value net (kind={:?}, arch={}) from {}",
                label,
                mw.kind,
                arch.join("->"),
                path
            );
        }
        None => eprintln!("[alphaow] failed to parse {} value net weights at {}", label, path),
    }
    w.map(Model::Mlp)
}

fn load_weights() -> ModelSet {
    let default = std::env::var("ALPHAOW_VALUE_NET_PATH")
        .ok()
        .and_then(|p| load_model_from_path(&p, "default"));
    let two_p = std::env::var("ALPHAOW_VALUE_NET_PATH_2P")
        .ok()
        .and_then(|p| load_model_from_path(&p, "2P"));
    let four_p = std::env::var("ALPHAOW_VALUE_NET_PATH_4P")
        .ok()
        .and_then(|p| load_model_from_path(&p, "4P"));
    if default.is_none() && two_p.is_none() && four_p.is_none() {
        eprintln!(
            "[alphaow] no value net paths set (ALPHAOW_VALUE_NET_PATH, _2P, _4P); using duck heuristic"
        );
    }
    ModelSet { default, two_p, four_p }
}

static WEIGHTS: OnceLock<ModelSet> = OnceLock::new();
static SAW_4P_GAME: AtomicBool = AtomicBool::new(false);

fn weights() -> &'static ModelSet {
    WEIGHTS.get_or_init(load_weights)
}

fn looks_like_4p_state(state: &GameState) -> bool {
    if state.player >= 2 {
        SAW_4P_GAME.store(true, Ordering::Relaxed);
        return true;
    }
    let mut owners = [false; 4];
    for p in &state.planets {
        if (0..4).contains(&p.owner) {
            owners[p.owner as usize] = true;
        }
    }
    for f in &state.fleets {
        if (0..4).contains(&f.owner) {
            owners[f.owner as usize] = true;
        }
    }
    let n = owners.iter().filter(|&&x| x).count();
    if n >= 3 || owners[2] || owners[3] {
        SAW_4P_GAME.store(true, Ordering::Relaxed);
        true
    } else {
        SAW_4P_GAME.load(Ordering::Relaxed)
    }
}

fn routed_model(state: &GameState) -> Option<&'static Model> {
    let w = weights();
    if looks_like_4p_state(state) {
        w.four_p.as_ref().or(w.default.as_ref())
    } else {
        w.two_p.as_ref().or(w.default.as_ref())
    }
}

/// True iff weights are loaded and ready for inference.
pub fn is_ready() -> bool {
    let w = weights();
    w.default.is_some() || w.two_p.is_some() || w.four_p.is_some()
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
    let m = routed_model(state)?;
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
            InputKind::SummaryV10 => {
                let feats = summary_features_v10::extract(state, me);
                forward_raw(w, &feats)
            }
            InputKind::FourPV1 => {
                let feats = summary_features_4p_v1::extract(state, me);
                forward_raw(w, &feats)
            }
            InputKind::FourPV2 => {
                let feats = summary_features_4p_v1::extract_v2(state, me);
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
            InputKind::SummaryV10 => {
                let feats = summary_features_v10::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::FourPV1 => {
                let feats = summary_features_4p_v1::extract(state, me);
                model.predict_value(&feats)
            }
            InputKind::FourPV2 => {
                let feats = summary_features_4p_v1::extract_v2(state, me);
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

/// 13-d spatial / reachability features, mirrored exactly from the Python
/// `train/spatial_features.py` (inspired by the open-source `producer_lite`
/// planner). These add real planet-position geometry — distance-decayed
/// reachable enemy mass, frontline distance, capture vulnerability, in-flight
/// fleet threat, and center control — that the spatially-blind summary
/// aggregates lack. Parity is checked by `train/check_spatial_parity.py`.
///
/// All features are from the perspective of player `me`; positive differences
/// favour `me`. Computed in f64 then cast to f32 to match the Python builder.
pub mod summary_features_spatial {
    use super::*;

    pub const DIM: usize = 13;
    const HORIZON: f64 = 18.0;
    const SUN: (f64, f64) = (50.0, 50.0);

    fn fleet_speed(ships: f64) -> f64 {
        let s = ships.max(1.0);
        let speed = 1.0 + 5.0 * (s.ln() / 1000.0_f64.ln()).powf(1.5);
        speed.clamp(1.0, 6.0)
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let mut out = [0f32; DIM];
        let planets = &state.planets;
        let n = planets.len();
        if n == 0 {
            return out;
        }

        // Per-planet arrays.
        let xy: Vec<(f64, f64)> = planets.iter().map(|p| (p.x, p.y)).collect();
        let ships: Vec<f64> = planets.iter().map(|p| (p.ships as f64).max(0.0)).collect();
        let prod: Vec<f64> = planets.iter().map(|p| p.production as f64).collect();
        let is_me: Vec<bool> = planets.iter().map(|p| p.owner == me).collect();
        let is_opp: Vec<bool> = planets.iter().map(|p| p.owner >= 0 && p.owner != me).collect();
        let n_me = is_me.iter().filter(|&&b| b).count();
        let n_opp = is_opp.iter().filter(|&&b| b).count();
        if n_me == 0 || n_opp == 0 {
            return out;
        }
        let speed: Vec<f64> = ships.iter().map(|&s| fleet_speed(s)).collect();
        let reach: Vec<f64> = speed.iter().map(|&sp| (sp * HORIZON).max(1e-6)).collect();

        let dist = |a: (f64, f64), b: (f64, f64)| -> f64 {
            let dx = a.0 - b.0;
            let dy = a.1 - b.1;
            (dx * dx + dy * dy).max(0.0).sqrt()
        };

        // Reachable-mass aggregates onto each target planet.
        let mut enemy_onto = vec![0.0f64; n]; // enemy mass reaching tgt
        let mut my_onto = vec![0.0f64; n]; // my mass reaching tgt
        for s in 0..n {
            if !(is_me[s] || is_opp[s]) {
                continue;
            }
            for t in 0..n {
                if s == t {
                    continue;
                }
                let decay = (1.0 - dist(xy[s], xy[t]) / reach[s]).max(0.0);
                if decay <= 0.0 {
                    continue;
                }
                let c = ships[s] * decay;
                if is_opp[s] {
                    enemy_onto[t] += c;
                } else if is_me[s] {
                    my_onto[t] += c;
                }
            }
        }

        let mut my_recv = 0.0;
        let mut opp_recv = 0.0;
        let mut max_enemy_pressure = 0.0f64;
        let mut my_vuln = 0.0f64;
        let mut opp_vuln = 0.0f64;
        let mut threatened_prod_me = 0.0f64;
        let mut threatened_prod_opp = 0.0f64;
        let mut frontline_min = f64::INFINITY;
        let mut frontline_nearest_sum = 0.0f64;
        let mut center_me = 0.0f64;
        let mut center_opp = 0.0f64;

        for t in 0..n {
            let dc = dist(xy[t], SUN);
            let w = 1.0 / (1.0 + dc);
            if is_me[t] {
                my_recv += enemy_onto[t];
                if enemy_onto[t] > max_enemy_pressure {
                    max_enemy_pressure = enemy_onto[t];
                }
                if enemy_onto[t] > ships[t] {
                    my_vuln += 1.0;
                    threatened_prod_me += prod[t];
                }
                center_me += ships[t] * w;
                // Nearest enemy distance from this (my) planet.
                let mut nearest = f64::INFINITY;
                for j in 0..n {
                    if is_opp[j] {
                        let d = dist(xy[t], xy[j]);
                        if d < nearest {
                            nearest = d;
                        }
                        if d < frontline_min {
                            frontline_min = d;
                        }
                    }
                }
                frontline_nearest_sum += nearest;
            } else if is_opp[t] {
                opp_recv += my_onto[t];
                if my_onto[t] > ships[t] {
                    opp_vuln += 1.0;
                    threatened_prod_opp += prod[t];
                }
                center_opp += ships[t] * w;
            }
        }

        // In-flight fleet threat.
        let mut incoming = 0.0f64;
        let mut outgoing = 0.0f64;
        for f in &state.fleets {
            if f.owner < 0 {
                continue;
            }
            let fxy = (f.x, f.y);
            let fs = (f.ships as f64).max(0.0);
            let fr = (fleet_speed(fs) * HORIZON).max(1e-6);
            if f.owner != me {
                // enemy fleet -> nearest my planet
                let mut nearest = f64::INFINITY;
                for t in 0..n {
                    if is_me[t] {
                        let d = dist(fxy, xy[t]);
                        if d < nearest {
                            nearest = d;
                        }
                    }
                }
                if nearest.is_finite() {
                    incoming += fs * (1.0 - nearest / fr).max(0.0);
                }
            } else {
                // my fleet -> nearest enemy planet
                let mut nearest = f64::INFINITY;
                for t in 0..n {
                    if is_opp[t] {
                        let d = dist(fxy, xy[t]);
                        if d < nearest {
                            nearest = d;
                        }
                    }
                }
                if nearest.is_finite() {
                    outgoing += fs * (1.0 - nearest / fr).max(0.0);
                }
            }
        }

        out[0] = my_recv as f32;
        out[1] = opp_recv as f32;
        out[2] = (opp_recv - my_recv) as f32;
        out[3] = max_enemy_pressure as f32;
        out[4] = frontline_min as f32;
        out[5] = (frontline_nearest_sum / n_me as f64) as f32;
        out[6] = my_vuln as f32;
        out[7] = opp_vuln as f32;
        out[8] = (opp_vuln - my_vuln) as f32;
        out[9] = incoming as f32;
        out[10] = outgoing as f32;
        out[11] = (threatened_prod_opp - threatened_prod_me) as f32;
        out[12] = (center_me - center_opp) as f32;
        out
    }
}

/// 170-d v10: summary_v9 (157) plus 13 spatial/reachability features.
/// Layout: `[summary_features_v9 (157)][summary_features_spatial (13)]`.
pub mod summary_features_v10 {
    use super::*;

    pub const DIM: usize = summary_features_v9::DIM + summary_features_spatial::DIM;

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let v9 = summary_features_v9::extract(state, me);
        let sp = summary_features_spatial::extract(state, me);
        let mut out = [0f32; DIM];
        out[..summary_features_v9::DIM].copy_from_slice(&v9);
        out[summary_features_v9::DIM..].copy_from_slice(&sp);
        out
    }
}

/// 236-d 4-player-specific evaluator features.
///
/// Layout:
///   global/map features (18)
///   player blocks for me + three threat-ordered opponents (4 * 26)
///   aggregate enemy field (22)
///   per-opponent relational blocks (3 * 24)
///   rank/field summary (20)
pub mod summary_features_4p_v1 {
    use super::*;

    pub const GLOBAL_DIM: usize = 18;
    pub const PLAYER_BLOCK_DIM: usize = 26;
    pub const ENEMY_AGG_DIM: usize = 22;
    pub const OPP_REL_DIM: usize = 24;
    pub const RANK_DIM: usize = 20;
    pub const DIM: usize = GLOBAL_DIM + 4 * PLAYER_BLOCK_DIM + ENEMY_AGG_DIM + 3 * OPP_REL_DIM + RANK_DIM;

    #[derive(Clone, Copy, Default)]
    struct PStats {
        ships_planets: f32,
        ships_flying: f32,
        n_static: f32,
        n_orbit: f32,
        n_comet: f32,
        prod_static: f32,
        prod_orbit: f32,
        prod_comet: f32,
        neutrals_closer: f32,
        enemies_closer: f32,
    }

    impl PStats {
        #[inline]
        fn ships_total(self) -> f32 {
            self.ships_planets + self.ships_flying
        }
        #[inline]
        fn planets(self) -> f32 {
            self.n_static + self.n_orbit + self.n_comet
        }
        #[inline]
        fn prod_total(self) -> f32 {
            self.prod_static + self.prod_orbit + self.prod_comet
        }
    }

    #[derive(Clone, Copy)]
    struct OppInfo {
        owner: i32,
        cur: PStats,
        ext: PStats,
        threat: f32,
        nearest_now: f32,
        nearest_ext: f32,
        my_to_opp_pressure: f32,
        opp_to_my_pressure: f32,
    }

    #[inline]
    fn share(a: f32, b: f32) -> f32 {
        (a - b) / (a + b).max(1.0)
    }

    #[inline]
    fn fleet_speed(ships: f32) -> f32 {
        let s = ships.max(1.0);
        let speed = 1.0 + 5.0 * (s.ln() / 1000.0f32.ln()).powf(1.5);
        speed.clamp(1.0, 6.0)
    }

    fn owners(state: &GameState, me: i32) -> Vec<i32> {
        let mut out = vec![0, 1, 2, 3, me];
        for p in &state.planets {
            if p.owner >= 0 {
                out.push(p.owner);
            }
        }
        for f in &state.fleets {
            if f.owner >= 0 {
                out.push(f.owner);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn owner_of(
        state: &GameState,
        extrap: Option<&std::collections::HashMap<i64, (i32, i64)>>,
        planet: &Planet,
    ) -> i32 {
        extrap
            .and_then(|m| m.get(&planet.id).copied())
            .map(|x| x.0)
            .unwrap_or_else(|| state.planets.iter().find(|p| p.id == planet.id).map(|p| p.owner).unwrap_or(planet.owner))
    }

    fn ships_of(extrap: Option<&std::collections::HashMap<i64, (i32, i64)>>, planet: &Planet) -> i64 {
        extrap
            .and_then(|m| m.get(&planet.id).copied())
            .map(|x| x.1)
            .unwrap_or(planet.ships)
    }

    fn min_dist_to_owner(
        state: &GameState,
        x: f64,
        y: f64,
        owner: i32,
        extrap: Option<&std::collections::HashMap<i64, (i32, i64)>>,
        exclude_id: Option<i64>,
    ) -> f32 {
        let mut best = f32::INFINITY;
        for p in &state.planets {
            if Some(p.id) == exclude_id {
                continue;
            }
            if owner_of(state, extrap, p) != owner {
                continue;
            }
            let dx = (p.x - x) as f32;
            let dy = (p.y - y) as f32;
            let d = (dx * dx + dy * dy).sqrt();
            if d < best {
                best = d;
            }
        }
        best
    }

    fn closer_counts(
        state: &GameState,
        owner: i32,
        player_ids: &[i32],
        extrap: Option<&std::collections::HashMap<i64, (i32, i64)>>,
    ) -> (f32, f32) {
        let mut neutrals = 0.0;
        let mut enemies = 0.0;
        for o in &state.planets {
            let oo = owner_of(state, extrap, o);
            if oo == owner {
                continue;
            }
            let d_me = min_dist_to_owner(state, o.x, o.y, owner, extrap, None);
            if !d_me.is_finite() {
                continue;
            }
            if oo == -1 {
                let mut best_enemy = f32::INFINITY;
                for &pid in player_ids {
                    if pid == owner {
                        continue;
                    }
                    best_enemy = best_enemy.min(min_dist_to_owner(state, o.x, o.y, pid, extrap, None));
                }
                if d_me < best_enemy {
                    neutrals += 1.0;
                }
            } else {
                let mut best_other = f32::INFINITY;
                for &pid in player_ids {
                    if pid == owner {
                        continue;
                    }
                    best_other = best_other.min(min_dist_to_owner(state, o.x, o.y, pid, extrap, Some(o.id)));
                }
                if d_me < best_other {
                    enemies += 1.0;
                }
            }
        }
        (neutrals, enemies)
    }

    fn stats_for(
        state: &GameState,
        owner: i32,
        player_ids: &[i32],
        extrap: Option<&std::collections::HashMap<i64, (i32, i64)>>,
    ) -> PStats {
        let mut s = PStats::default();
        for p in &state.planets {
            if owner_of(state, extrap, p) != owner {
                continue;
            }
            s.ships_planets += ships_of(extrap, p).max(0) as f32;
            let prod = p.production as f32;
            if p.is_comet {
                s.n_comet += 1.0;
                s.prod_comet += prod;
            } else if p.is_orbiting {
                s.n_orbit += 1.0;
                s.prod_orbit += prod;
            } else {
                s.n_static += 1.0;
                s.prod_static += prod;
            }
        }
        if extrap.is_none() {
            for f in &state.fleets {
                if f.owner == owner {
                    s.ships_flying += f.ships as f32;
                }
            }
        }
        let (neutrals, enemies) = closer_counts(state, owner, player_ids, extrap);
        s.neutrals_closer = neutrals;
        s.enemies_closer = enemies;
        s
    }

    fn nearest_owner_dist(
        state: &GameState,
        a_owner: i32,
        b_owner: i32,
        extrap: Option<&std::collections::HashMap<i64, (i32, i64)>>,
    ) -> f32 {
        let mut best = f32::INFINITY;
        for a in &state.planets {
            if owner_of(state, extrap, a) != a_owner {
                continue;
            }
            for b in &state.planets {
                if owner_of(state, extrap, b) != b_owner {
                    continue;
                }
                let dx = (a.x - b.x) as f32;
                let dy = (a.y - b.y) as f32;
                let d = (dx * dx + dy * dy).sqrt();
                if d < best {
                    best = d;
                }
            }
        }
        if best.is_finite() { best } else { 0.0 }
    }

    fn pressure_from_to(
        state: &GameState,
        from_owner: i32,
        to_owner: i32,
        extrap: Option<&std::collections::HashMap<i64, (i32, i64)>>,
    ) -> f32 {
        let mut total = 0.0;
        for src in &state.planets {
            if owner_of(state, extrap, src) != from_owner {
                continue;
            }
            let ships = ships_of(extrap, src).max(0) as f32;
            let d = min_dist_to_owner(state, src.x, src.y, to_owner, extrap, None);
            if d.is_finite() {
                total += ships / (1.0 + d);
            }
        }
        total
    }

    fn push_player_block(out: &mut Vec<f32>, cur: PStats, ext: PStats, remaining: f32) {
        let cur_planets = cur.planets();
        let cur_prod = cur.prod_total();
        let cur_total = cur.ships_total();
        let ext_planets = ext.planets();
        let ext_prod = ext.prod_total();
        let horizon = remaining.min(100.0);
        out.extend_from_slice(&[
            cur.ships_planets,
            cur.ships_flying,
            cur_total,
            cur.n_static,
            cur.n_orbit,
            cur.n_comet,
            cur_planets,
            cur.prod_static,
            cur.prod_orbit,
            cur.prod_comet,
            cur_prod,
            cur.neutrals_closer,
            cur.enemies_closer,
            cur_total / cur_planets.max(1.0),
            cur_prod / cur_planets.max(1.0),
            cur.ships_flying / cur_total.max(1.0),
            fleet_speed(cur_total),
            ext.ships_planets,
            ext.n_static,
            ext.n_orbit,
            ext.n_comet,
            ext_planets,
            ext_prod,
            ext.ships_planets - cur.ships_planets,
            ext_planets - cur_planets,
            cur_total + cur_prod * horizon,
        ]);
    }

    fn rank_desc(values: &[(i32, f32)], owner: i32) -> f32 {
        let mine = values.iter().find(|x| x.0 == owner).map(|x| x.1).unwrap_or(0.0);
        1.0 + values.iter().filter(|x| x.1 > mine).count() as f32
    }

    fn top_value(values: &[(i32, f32)]) -> f32 {
        values.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max).max(0.0)
    }

    fn sorted_values(mut values: Vec<f32>) -> Vec<f32> {
        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        values
    }

    #[inline]
    fn sigmoid_margin(x: f32, scale: f32) -> f32 {
        1.0 / (1.0 + (-x / scale.max(1.0)).exp())
    }

    pub const V2_EXTRA_DIM: usize = 42;
    pub const DIM_V2: usize = DIM + V2_EXTRA_DIM;

    fn extract_vec(state: &GameState, me: i32, include_v2: bool) -> Vec<f32> {
        let player_ids = owners(state, me);
        let extrap = extrapolate_fleets(state);
        let remaining = (500.0 - state.step as f32).max(0.0);
        let h25 = remaining.min(25.0);
        let h50 = remaining.min(50.0);
        let h100 = remaining.min(100.0);

        let me_cur = stats_for(state, me, &player_ids, None);
        let me_ext = stats_for(state, me, &player_ids, Some(&extrap));
        let my_proj_100 = me_cur.ships_total() + me_cur.prod_total() * h100;

        let mut opps: Vec<OppInfo> = player_ids
            .iter()
            .copied()
            .filter(|&p| p != me && p >= 0)
            .map(|owner| {
                let cur = stats_for(state, owner, &player_ids, None);
                let ext = stats_for(state, owner, &player_ids, Some(&extrap));
                let nearest_now = nearest_owner_dist(state, me, owner, None);
                let nearest_ext = nearest_owner_dist(state, me, owner, Some(&extrap));
                let my_to_opp_pressure = pressure_from_to(state, me, owner, None);
                let opp_to_my_pressure = pressure_from_to(state, owner, me, None);
                let threat = cur.ships_total()
                    + cur.prod_total() * h100
                    + opp_to_my_pressure * 200.0
                    - my_to_opp_pressure * 50.0
                    - my_proj_100 * 0.25;
                OppInfo {
                    owner,
                    cur,
                    ext,
                    threat,
                    nearest_now,
                    nearest_ext,
                    my_to_opp_pressure,
                    opp_to_my_pressure,
                }
            })
            .collect();
        opps.sort_by(|a, b| {
            b.threat
                .partial_cmp(&a.threat)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.owner.cmp(&b.owner))
        });
        while opps.len() < 3 {
            opps.push(OppInfo {
                owner: -1,
                cur: PStats::default(),
                ext: PStats::default(),
                threat: 0.0,
                nearest_now: 0.0,
                nearest_ext: 0.0,
                my_to_opp_pressure: 0.0,
                opp_to_my_pressure: 0.0,
            });
        }
        opps.truncate(3);

        let mut out = Vec::with_capacity(DIM);

        let mut n_static = 0.0;
        let mut n_orbit = 0.0;
        let mut n_comet = 0.0;
        let mut n_neutral = 0.0;
        let mut neutral_ships = 0.0;
        let mut neutral_prod = 0.0;
        let mut neutral_comet_time = 0.0;
        let mut total_ships_planets = 0.0;
        let mut total_prod = 0.0;
        let mut total_planets_owned = 0.0;
        for p in &state.planets {
            if p.is_comet {
                n_comet += 1.0;
            } else if p.is_orbiting {
                n_orbit += 1.0;
            } else {
                n_static += 1.0;
            }
            if p.owner == -1 {
                n_neutral += 1.0;
                neutral_ships += p.ships as f32;
                neutral_prod += p.production as f32;
                if p.is_comet {
                    neutral_comet_time += state.comet_remaining(p) as f32;
                }
            } else {
                total_planets_owned += 1.0;
                total_ships_planets += p.ships as f32;
                total_prod += p.production as f32;
            }
        }
        let total_ships_flying: f32 = state.fleets.iter().map(|f| f.ships as f32).sum();
        let enemy_count = opps
            .iter()
            .filter(|o| o.cur.planets() + o.cur.ships_flying + o.ext.planets() > 0.0)
            .count() as f32;

        out.extend_from_slice(&[
            state.step as f32,
            (state.step as f32 / 500.0).clamp(0.0, 1.0),
            (remaining / 500.0).clamp(0.0, 1.0),
            state.angular_velocity as f32,
            state.planets.len() as f32,
            n_static,
            n_orbit,
            n_comet,
            n_neutral,
            neutral_ships,
            neutral_prod,
            neutral_comet_time,
            total_ships_planets,
            total_ships_flying,
            total_prod,
            total_planets_owned,
            enemy_count,
            state.angular_velocity as f32 * n_orbit,
        ]);

        push_player_block(&mut out, me_cur, me_ext, remaining);
        for opp in &opps {
            push_player_block(&mut out, opp.cur, opp.ext, remaining);
        }

        let mut enemy_sum_cur = PStats::default();
        let mut enemy_sum_ext = PStats::default();
        let mut max_cur_ships = 0.0f32;
        let mut max_cur_prod = 0.0f32;
        let mut max_ext_proj = 0.0f32;
        let mut sum_threat = 0.0f32;
        let mut max_threat = 0.0f32;
        let mut sum_pressure_to_me = 0.0f32;
        let mut max_pressure_to_me = 0.0f32;
        for opp in &opps {
            enemy_sum_cur.ships_planets += opp.cur.ships_planets;
            enemy_sum_cur.ships_flying += opp.cur.ships_flying;
            enemy_sum_cur.n_static += opp.cur.n_static;
            enemy_sum_cur.n_orbit += opp.cur.n_orbit;
            enemy_sum_cur.n_comet += opp.cur.n_comet;
            enemy_sum_cur.prod_static += opp.cur.prod_static;
            enemy_sum_cur.prod_orbit += opp.cur.prod_orbit;
            enemy_sum_cur.prod_comet += opp.cur.prod_comet;
            enemy_sum_cur.neutrals_closer += opp.cur.neutrals_closer;
            enemy_sum_cur.enemies_closer += opp.cur.enemies_closer;
            enemy_sum_ext.ships_planets += opp.ext.ships_planets;
            enemy_sum_ext.n_static += opp.ext.n_static;
            enemy_sum_ext.n_orbit += opp.ext.n_orbit;
            enemy_sum_ext.n_comet += opp.ext.n_comet;
            enemy_sum_ext.prod_static += opp.ext.prod_static;
            enemy_sum_ext.prod_orbit += opp.ext.prod_orbit;
            enemy_sum_ext.prod_comet += opp.ext.prod_comet;
            max_cur_ships = max_cur_ships.max(opp.cur.ships_total());
            max_cur_prod = max_cur_prod.max(opp.cur.prod_total());
            max_ext_proj = max_ext_proj.max(opp.cur.ships_total() + opp.cur.prod_total() * h100);
            sum_threat += opp.threat;
            max_threat = max_threat.max(opp.threat);
            sum_pressure_to_me += opp.opp_to_my_pressure;
            max_pressure_to_me = max_pressure_to_me.max(opp.opp_to_my_pressure);
        }
        let enemy_count_safe = enemy_count.max(1.0);
        out.extend_from_slice(&[
            enemy_sum_cur.ships_planets,
            enemy_sum_cur.ships_flying,
            enemy_sum_cur.ships_total(),
            enemy_sum_cur.planets(),
            enemy_sum_cur.prod_total(),
            enemy_sum_cur.n_static,
            enemy_sum_cur.n_orbit,
            enemy_sum_cur.n_comet,
            enemy_sum_cur.neutrals_closer,
            enemy_sum_cur.enemies_closer,
            enemy_sum_ext.ships_planets,
            enemy_sum_ext.planets(),
            enemy_sum_ext.prod_total(),
            max_cur_ships,
            max_cur_prod,
            max_ext_proj,
            enemy_sum_cur.ships_total() / enemy_count_safe,
            enemy_sum_cur.prod_total() / enemy_count_safe,
            max_threat,
            sum_threat,
            max_pressure_to_me,
            sum_pressure_to_me,
        ]);

        for (rank, opp) in opps.iter().enumerate() {
            let opp_total = opp.cur.ships_total();
            let my_total = me_cur.ships_total();
            let my_prod = me_cur.prod_total();
            let opp_prod = opp.cur.prod_total();
            let my_planets = me_cur.planets();
            let opp_planets = opp.cur.planets();
            let cur_ship_diff = my_total - opp_total;
            let cur_prod_diff = my_prod - opp_prod;
            let ext_ship_diff = me_ext.ships_planets - opp.ext.ships_planets;
            let ext_prod_diff = me_ext.prod_total() - opp.ext.prod_total();
            out.extend_from_slice(&[
                if opp.owner >= 0 && (opp.cur.planets() + opp.cur.ships_flying + opp.ext.planets()) > 0.0 { 1.0 } else { 0.0 },
                (rank + 1) as f32,
                opp.threat,
                opp.nearest_now,
                opp.nearest_ext,
                opp.my_to_opp_pressure,
                opp.opp_to_my_pressure,
                cur_ship_diff,
                share(my_total, opp_total),
                cur_prod_diff,
                share(my_prod, opp_prod),
                my_planets - opp_planets,
                me_cur.n_static - opp.cur.n_static,
                me_cur.n_orbit - opp.cur.n_orbit,
                me_cur.n_comet - opp.cur.n_comet,
                me_cur.ships_flying - opp.cur.ships_flying,
                cur_ship_diff + cur_prod_diff * h25,
                cur_ship_diff + cur_prod_diff * h50,
                cur_ship_diff + cur_prod_diff * h100,
                cur_ship_diff + cur_prod_diff * remaining,
                ext_ship_diff,
                ext_prod_diff,
                me_ext.planets() - opp.ext.planets(),
                ext_ship_diff + ext_prod_diff * h100,
            ]);
        }

        let all_cur: Vec<(i32, PStats)> = player_ids
            .iter()
            .copied()
            .filter(|&p| p >= 0)
            .map(|p| (p, stats_for(state, p, &player_ids, None)))
            .collect();
        let ship_vals: Vec<(i32, f32)> = all_cur.iter().map(|(p, s)| (*p, s.ships_total())).collect();
        let prod_vals: Vec<(i32, f32)> = all_cur.iter().map(|(p, s)| (*p, s.prod_total())).collect();
        let planet_vals: Vec<(i32, f32)> = all_cur.iter().map(|(p, s)| (*p, s.planets())).collect();
        let adv50_vals: Vec<(i32, f32)> = all_cur
            .iter()
            .map(|(p, s)| (*p, s.ships_total() + s.prod_total() * h50))
            .collect();
        let my_adv50 = me_cur.ships_total() + me_cur.prod_total() * h50;
        let my_adv100 = me_cur.ships_total() + me_cur.prod_total() * h100;
        let adv_sorted = sorted_values(adv50_vals.iter().map(|x| x.1).collect());
        let leader_adv50 = *adv_sorted.first().unwrap_or(&0.0);
        let second_adv50 = *adv_sorted.get(1).unwrap_or(&leader_adv50);
        let last_adv50 = *adv_sorted.last().unwrap_or(&0.0);
        out.extend_from_slice(&[
            rank_desc(&ship_vals, me),
            rank_desc(&prod_vals, me),
            rank_desc(&planet_vals, me),
            rank_desc(&adv50_vals, me),
            if rank_desc(&ship_vals, me) <= 1.0 { 1.0 } else { 0.0 },
            if rank_desc(&prod_vals, me) <= 1.0 { 1.0 } else { 0.0 },
            if rank_desc(&planet_vals, me) <= 1.0 { 1.0 } else { 0.0 },
            if rank_desc(&adv50_vals, me) <= 1.0 { 1.0 } else { 0.0 },
            me_cur.ships_total() - top_value(&ship_vals),
            me_cur.prod_total() - top_value(&prod_vals),
            me_cur.planets() - top_value(&planet_vals),
            my_adv50 - leader_adv50,
            my_adv50 - second_adv50,
            my_adv50 - last_adv50,
            me_cur.ships_total() - enemy_sum_cur.ships_total(),
            me_cur.prod_total() - enemy_sum_cur.prod_total(),
            me_cur.planets() - enemy_sum_cur.planets(),
            my_adv50 - enemy_sum_cur.ships_total() - enemy_sum_cur.prod_total() * h50,
            my_adv100 - enemy_sum_cur.ships_total() - enemy_sum_cur.prod_total() * h100,
            me_cur.neutrals_closer - opps.iter().map(|o| o.cur.neutrals_closer).fold(0.0, f32::max),
        ]);

        if include_v2 {
            let mut pair_margins_25 = Vec::with_capacity(3);
            let mut pair_margins_50 = Vec::with_capacity(3);
            let mut pair_margins_100 = Vec::with_capacity(3);
            let mut pair_margins_remaining = Vec::with_capacity(3);
            let mut pair_ext_100 = Vec::with_capacity(3);
            let mut pair_probs_50 = Vec::with_capacity(3);
            let mut pair_probs_100 = Vec::with_capacity(3);
            let mut pressure_balances = Vec::with_capacity(3);

            for opp in &opps {
                let my_total = me_cur.ships_total();
                let opp_total = opp.cur.ships_total();
                let my_prod = me_cur.prod_total();
                let opp_prod = opp.cur.prod_total();
                let ship_margin = my_total - opp_total;
                let prod_margin = my_prod - opp_prod;
                let m25 = ship_margin + prod_margin * h25;
                let m50 = ship_margin + prod_margin * h50;
                let m100 = ship_margin + prod_margin * h100;
                let mrem = ship_margin + prod_margin * remaining;
                let ext100 = (me_ext.ships_planets - opp.ext.ships_planets)
                    + (me_ext.prod_total() - opp.ext.prod_total()) * h100;
                let pressure_balance = opp.my_to_opp_pressure - opp.opp_to_my_pressure;
                pair_margins_25.push(m25);
                pair_margins_50.push(m50);
                pair_margins_100.push(m100);
                pair_margins_remaining.push(mrem);
                pair_ext_100.push(ext100);
                pair_probs_50.push(sigmoid_margin(m50, 75.0));
                pair_probs_100.push(sigmoid_margin(m100, 125.0));
                pressure_balances.push(pressure_balance);

                out.extend_from_slice(&[
                    m25,
                    m50,
                    m100,
                    mrem,
                    ext100,
                    sigmoid_margin(m50, 75.0),
                    sigmoid_margin(m100, 125.0),
                    pressure_balance,
                    pressure_balance / (me_cur.ships_total() + opp.cur.ships_total()).max(1.0),
                ]);
            }

            let mean = |xs: &[f32]| -> f32 {
                if xs.is_empty() {
                    0.0
                } else {
                    xs.iter().sum::<f32>() / xs.len() as f32
                }
            };
            let minv = |xs: &[f32]| -> f32 {
                xs.iter().copied().fold(f32::INFINITY, f32::min)
            };
            let maxv = |xs: &[f32]| -> f32 {
                xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            };
            let lead_count = |xs: &[f32]| -> f32 { xs.iter().filter(|&&x| x > 0.0).count() as f32 };
            let incoming_pressure = sum_pressure_to_me;
            let outgoing_pressure: f32 = opps.iter().map(|o| o.my_to_opp_pressure).sum();
            let dogpile_pressure = incoming_pressure / me_cur.ships_total().max(1.0);
            let is_adv50_leader = if rank_desc(&adv50_vals, me) <= 1.0 { 1.0 } else { 0.0 };
            let lead_over_second = my_adv50 - if is_adv50_leader > 0.0 { second_adv50 } else { leader_adv50 };
            out.extend_from_slice(&[
                mean(&pair_probs_50),
                mean(&pair_probs_100),
                lead_count(&pair_margins_50),
                lead_count(&pair_margins_100),
                minv(&pair_margins_50),
                minv(&pair_margins_100),
                maxv(&pair_margins_50),
                mean(&pair_margins_50),
                mean(&pair_ext_100),
                mean(&pressure_balances),
                incoming_pressure,
                outgoing_pressure,
                dogpile_pressure,
                is_adv50_leader * dogpile_pressure,
                lead_over_second,
            ]);
        }

        debug_assert_eq!(out.len(), if include_v2 { DIM_V2 } else { DIM });
        out
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let out = extract_vec(state, me, false);
        let mut arr = [0.0f32; DIM];
        arr.copy_from_slice(&out[..DIM]);
        arr
    }

    pub fn extract_v2(state: &GameState, me: i32) -> [f32; DIM_V2] {
        let out = extract_vec(state, me, true);
        let mut arr = [0.0f32; DIM_V2];
        arr.copy_from_slice(&out[..DIM_V2]);
        arr
    }
}
