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
//! input_dim u32
//! hidden    u32
//! w1        f32[hidden * input_dim]   row-major (hidden first)
//! b1        f32[hidden]
//! w2        f32[hidden]
//! b2        f32
//! ```
//!
//! Version 2 adds one more hidden layer:
//!
//! ```text
//! magic     u32 = 0x564f4157
//! version   u32 = 2
//! input_dim u32
//! hidden1   u32
//! hidden2   u32
//! w1        f32[hidden1 * input_dim]
//! b1        f32[hidden1]
//! w2        f32[hidden2 * hidden1]
//! b2        f32[hidden2]
//! w3        f32[hidden2]
//! b3        f32
//! ```
//!
//! Forward pass: `y = tanh(b2 + w2 · ReLU(w1 · x + b1))`. Output is a
//! scalar in `[-1, 1]` interpreted as MCTS value from MY perspective.
//!
//! If no weights file is found or the file is malformed, `predict`
//! returns `None`. Callers should fall back to the duck heuristic.

use crate::ow2_plan::cached_predict_fleet_collision;
use crate::pathing::{fleet_speed, point_to_segment_distance};
use crate::{GameState, Planet};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const MAX_OBJECTS: usize = 44;
pub const PER_OBJECT: usize = 9;
pub const PER_BLOCK: usize = MAX_OBJECTS * PER_OBJECT;
pub const DIST_BLOCK: usize = MAX_OBJECTS * MAX_OBJECTS;
pub const INPUT_DIM: usize = 2 * PER_BLOCK + DIST_BLOCK;
pub const TRANSFORMER_TOKEN_DIM: usize = 24;
pub const TRANSFORMER_MAX_PLANETS: usize = 44;
pub const TRANSFORMER_MAX_FLEETS: usize = 32;
pub const TRANSFORMER_MAX_TOKENS: usize = 1 + TRANSFORMER_MAX_PLANETS + TRANSFORMER_MAX_FLEETS;

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
    hidden2: usize,
    input_dim: usize,
    kind: InputKind,
    w1: Vec<f32>, // [hidden, input_dim]
    b1: Vec<f32>, // [hidden]
    w2: Vec<f32>, // v1: [hidden] output weights; v2: [hidden2, hidden]
    b2: Vec<f32>, // v2 only: [hidden2]
    w3: Vec<f32>, // v2 only: [hidden2] output weights
    b3: f32,      // v1 output bias or v2 output bias
}

struct TransformerLayerWeights {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    qkv_w: Vec<f32>, // [3*d_model, d_model]
    qkv_b: Vec<f32>, // [3*d_model]
    out_w: Vec<f32>, // [d_model, d_model]
    out_b: Vec<f32>, // [d_model]
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    ff1_w: Vec<f32>, // [ff_dim, d_model]
    ff1_b: Vec<f32>, // [ff_dim]
    ff2_w: Vec<f32>, // [d_model, ff_dim]
    ff2_b: Vec<f32>, // [d_model]
}

struct TransformerWeights {
    token_dim: usize,
    d_model: usize,
    layers: usize,
    heads: usize,
    ff_dim: usize,
    max_tokens: usize,
    cls: Vec<f32>,
    embed_w: Vec<f32>, // [d_model, token_dim]
    embed_b: Vec<f32>, // [d_model]
    blocks: Vec<TransformerLayerWeights>,
    ln_f_w: Vec<f32>,
    ln_f_b: Vec<f32>,
    summary_dim: usize,
    summary_hidden: usize,
    summary_w: Vec<f32>, // [summary_hidden, summary_dim]
    summary_b: Vec<f32>, // [summary_hidden]
    head_w: Vec<f32>, // v3: [d_model], v4: [d_model + summary_hidden]
    head_b: f32,
}

enum LoadedWeights {
    Mlp(MlpWeights),
    Transformer(TransformerWeights),
}

fn read_u32_le(slice: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(slice.get(..4)?.try_into().ok()?))
}

fn read_f32_block(bytes: &[u8], count: usize, cursor: &mut usize) -> Option<Vec<f32>> {
    let end = *cursor + 4 * count;
    let slice = bytes.get(*cursor..end)?;
    let mut out = Vec::with_capacity(count);
    for chunk in slice.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().ok()?));
    }
    *cursor = end;
    Some(out)
}

fn parse_transformer_weights(bytes: &[u8], version: u32) -> Option<TransformerWeights> {
    if bytes.len() < 32 {
        return None;
    }
    let token_dim = read_u32_le(bytes.get(8..12)?)? as usize;
    let d_model = read_u32_le(bytes.get(12..16)?)? as usize;
    let layers = read_u32_le(bytes.get(16..20)?)? as usize;
    let heads = read_u32_le(bytes.get(20..24)?)? as usize;
    let ff_dim = read_u32_le(bytes.get(24..28)?)? as usize;
    let max_tokens = read_u32_le(bytes.get(28..32)?)? as usize;
    let (summary_dim, summary_hidden, header) = if version == 4 {
        if bytes.len() < 40 {
            return None;
        }
        (
            read_u32_le(bytes.get(32..36)?)? as usize,
            read_u32_le(bytes.get(36..40)?)? as usize,
            40usize,
        )
    } else {
        (0usize, 0usize, 32usize)
    };
    if token_dim != TRANSFORMER_TOKEN_DIM
        || max_tokens != TRANSFORMER_MAX_TOKENS
        || (version == 4 && summary_dim != summary_features_v2::DIM && summary_dim != summary_features_v3::DIM)
        || d_model == 0
        || d_model > 256
        || layers == 0
        || layers > 8
        || heads == 0
        || d_model % heads != 0
        || ff_dim == 0
        || ff_dim > 1024
        || summary_hidden > 256
    {
        eprintln!(
            "[alphaow] bad transformer shape token_dim={} d_model={} layers={} heads={} ff_dim={} max_tokens={} summary_dim={} summary_hidden={}",
            token_dim, d_model, layers, heads, ff_dim, max_tokens, summary_dim, summary_hidden
        );
        return None;
    }
    let per_layer = 2 * d_model
        + 3 * d_model * d_model
        + 3 * d_model
        + d_model * d_model
        + d_model
        + 2 * d_model
        + ff_dim * d_model
        + ff_dim
        + d_model * ff_dim
        + d_model;
    let total_floats = d_model
        + d_model * token_dim
        + d_model
        + layers * per_layer
        + 2 * d_model
        + if version == 4 {
            summary_hidden * summary_dim + summary_hidden + d_model + summary_hidden
        } else {
            d_model
        }
        + 1;
    let need = header + 4 * total_floats;
    if bytes.len() < need {
        eprintln!("[alphaow] transformer weights truncated: have={} need={}", bytes.len(), need);
        return None;
    }
    let mut cursor = header;
    let cls = read_f32_block(bytes, d_model, &mut cursor)?;
    let embed_w = read_f32_block(bytes, d_model * token_dim, &mut cursor)?;
    let embed_b = read_f32_block(bytes, d_model, &mut cursor)?;
    let mut blocks = Vec::with_capacity(layers);
    for _ in 0..layers {
        blocks.push(TransformerLayerWeights {
            ln1_w: read_f32_block(bytes, d_model, &mut cursor)?,
            ln1_b: read_f32_block(bytes, d_model, &mut cursor)?,
            qkv_w: read_f32_block(bytes, 3 * d_model * d_model, &mut cursor)?,
            qkv_b: read_f32_block(bytes, 3 * d_model, &mut cursor)?,
            out_w: read_f32_block(bytes, d_model * d_model, &mut cursor)?,
            out_b: read_f32_block(bytes, d_model, &mut cursor)?,
            ln2_w: read_f32_block(bytes, d_model, &mut cursor)?,
            ln2_b: read_f32_block(bytes, d_model, &mut cursor)?,
            ff1_w: read_f32_block(bytes, ff_dim * d_model, &mut cursor)?,
            ff1_b: read_f32_block(bytes, ff_dim, &mut cursor)?,
            ff2_w: read_f32_block(bytes, d_model * ff_dim, &mut cursor)?,
            ff2_b: read_f32_block(bytes, d_model, &mut cursor)?,
        });
    }
    let ln_f_w = read_f32_block(bytes, d_model, &mut cursor)?;
    let ln_f_b = read_f32_block(bytes, d_model, &mut cursor)?;
    let (summary_w, summary_b, head_w) = if version == 4 {
        (
            read_f32_block(bytes, summary_hidden * summary_dim, &mut cursor)?,
            read_f32_block(bytes, summary_hidden, &mut cursor)?,
            read_f32_block(bytes, d_model + summary_hidden, &mut cursor)?,
        )
    } else {
        (
            Vec::new(),
            Vec::new(),
            read_f32_block(bytes, d_model, &mut cursor)?,
        )
    };
    let head_b = f32::from_le_bytes(bytes.get(cursor..cursor + 4)?.try_into().ok()?);
    Some(TransformerWeights {
        token_dim,
        d_model,
        layers,
        heads,
        ff_dim,
        max_tokens,
        cls,
        embed_w,
        embed_b,
        blocks,
        ln_f_w,
        ln_f_b,
        summary_dim,
        summary_hidden,
        summary_w,
        summary_b,
        head_w,
        head_b,
    })
}

fn parse_weights(bytes: &[u8]) -> Option<LoadedWeights> {
    if bytes.len() < 16 {
        return None;
    }
    if read_u32_le(&bytes[0..4])? != WEIGHTS_MAGIC {
        return None;
    }
    let version = read_u32_le(&bytes[4..8])?;
    if version == 3 || version == 4 {
        return parse_transformer_weights(bytes, version).map(LoadedWeights::Transformer);
    }
    if version != 1 && version != 2 {
        return None;
    }
    let input_dim = read_u32_le(&bytes[8..12])? as usize;
    let hidden = read_u32_le(&bytes[12..16])? as usize;
    if hidden == 0 {
        return None;
    }
    let hidden2 = if version == 2 {
        let h2 = read_u32_le(bytes.get(16..20)?)? as usize;
        if h2 == 0 {
            return None;
        }
        h2
    } else {
        0
    };
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
    let header = if version == 2 { 20 } else { 16 };
    let weights = if version == 2 {
        hidden * input_dim + hidden + hidden2 * hidden + hidden2 + hidden2 + 1
    } else {
        hidden * input_dim + hidden + hidden + 1
    };
    let need = header + 4 * weights;
    if bytes.len() < need {
        return None;
    }
    let mut cursor = header;
    let w1 = read_f32_block(bytes, hidden * input_dim, &mut cursor)?;
    let b1 = read_f32_block(bytes, hidden, &mut cursor)?;
    if version == 2 {
        let w2 = read_f32_block(bytes, hidden2 * hidden, &mut cursor)?;
        let b2 = read_f32_block(bytes, hidden2, &mut cursor)?;
        let w3 = read_f32_block(bytes, hidden2, &mut cursor)?;
        let b3 = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().ok()?);
        Some(LoadedWeights::Mlp(MlpWeights { hidden, hidden2, input_dim, kind, w1, b1, w2, b2, w3, b3 }))
    } else {
        let w2 = read_f32_block(bytes, hidden, &mut cursor)?;
        let b3 = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().ok()?);
        Some(LoadedWeights::Mlp(MlpWeights {
            hidden,
            hidden2,
            input_dim,
            kind,
            w1,
            b1,
            w2,
            b2: Vec::new(),
            w3: Vec::new(),
            b3,
        }))
    }
}

fn load_weights() -> Option<LoadedWeights> {
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
        Some(LoadedWeights::Mlp(mw)) => {
            eprintln!(
                "[alphaow] loaded value net (kind={:?}, hidden={}{} input_dim={}) from {}",
                mw.kind,
                mw.hidden,
                if mw.hidden2 > 0 { format!(",{}", mw.hidden2) } else { String::new() },
                mw.input_dim,
                path
            )
        }
        Some(LoadedWeights::Transformer(tw)) => {
            eprintln!(
                "[alphaow] loaded transformer value net (tokens={} token_dim={} d_model={} layers={} heads={} ff_dim={} summary_hidden={}) from {}",
                tw.max_tokens, tw.token_dim, tw.d_model, tw.layers, tw.heads, tw.ff_dim, tw.summary_hidden, path
            )
        }
        None => eprintln!("[alphaow] failed to parse value net weights at {}", path),
    }
    w
}

static WEIGHTS: OnceLock<Option<LoadedWeights>> = OnceLock::new();

fn weights() -> Option<&'static LoadedWeights> {
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
    if w.hidden2 > 0 {
        let mut hidden2 = [0f32; 512];
        debug_assert!(w.hidden2 <= hidden2.len(), "hidden2 > 512 not supported");
        for h in 0..w.hidden2 {
            let row = &w.w2[h * w.hidden..(h + 1) * w.hidden];
            let s = dot_neon(row, &hidden[..w.hidden], w.b2[h], w.hidden);
            hidden2[h] = if s > 0.0 { s } else { 0.0 };
        }
        let mut out = w.b3;
        for h in 0..w.hidden2 {
            out += w.w3[h] * hidden2[h];
        }
        out.tanh()
    } else {
        let mut out = w.b3;
        for h in 0..w.hidden {
            out += w.w2[h] * hidden[h];
        }
        out.tanh()
    }
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

fn rel_owner(owner: i32, me: i32) -> (f32, f32, f32) {
    if owner == -1 {
        (0.0, 0.0, 1.0)
    } else if owner == me {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    }
}

fn token_ships(ships: i64) -> f32 {
    ((ships.max(0) as f32 + 1.0).ln() / 1001.0f32.ln()).clamp(0.0, 2.0)
}

/// Entity-token features for the transformer value net.
///
/// Token 0 is reserved for learned CLS and is represented by the transformer's
/// `cls` vector directly. Tokens 1.. are planets/comets sorted by id, followed
/// by the largest in-flight fleets.
pub fn transformer_tokens(state: &GameState, me: i32) -> ([[f32; TRANSFORMER_TOKEN_DIM]; TRANSFORMER_MAX_TOKENS], [bool; TRANSFORMER_MAX_TOKENS]) {
    let mut tokens = [[0f32; TRANSFORMER_TOKEN_DIM]; TRANSFORMER_MAX_TOKENS];
    let mut mask = [false; TRANSFORMER_MAX_TOKENS];
    mask[0] = true;

    let extrap = extrapolate_fleets(state);
    let mut planets: Vec<&Planet> = state.planets.iter().collect();
    planets.sort_by_key(|p| p.id);
    let n_planets = planets.len().min(TRANSFORMER_MAX_PLANETS);
    for i in 0..n_planets {
        let p = planets[i];
        let slot = 1 + i;
        mask[slot] = true;
        let t = &mut tokens[slot];
        let (om, oo, on) = rel_owner(p.owner, me);
        let (eo, es) = extrap.get(&p.id).copied().unwrap_or((p.owner, p.ships));
        let (em, eopp, en) = rel_owner(eo, me);
        let dx = ((p.x - crate::CENTER_X) / 50.0) as f32;
        let dy = ((p.y - crate::CENTER_Y) / 50.0) as f32;
        t[0] = 1.0; // planet/comet token
        t[2] = om;
        t[3] = oo;
        t[4] = on;
        t[5] = dx;
        t[6] = dy;
        t[7] = (p.radius as f32 / 5.0).clamp(0.0, 2.0);
        t[8] = token_ships(p.ships);
        t[9] = (p.production as f32 / 5.0).clamp(0.0, 1.5);
        t[10] = if !p.is_orbiting && !p.is_comet { 1.0 } else { 0.0 };
        t[11] = if p.is_orbiting { 1.0 } else { 0.0 };
        t[12] = if p.is_comet { 1.0 } else { 0.0 };
        t[13] = (p.orbital_radius as f32 / 50.0).clamp(0.0, 2.0);
        t[14] = p.initial_angle.sin() as f32;
        t[15] = p.initial_angle.cos() as f32;
        t[16] = (state.comet_remaining(p) as f32 / 500.0).clamp(0.0, 1.5);
        t[17] = em;
        t[18] = eopp;
        t[19] = en;
        t[20] = token_ships(es);
    }

    let mut fleets: Vec<&crate::Fleet> = state.fleets.iter().collect();
    fleets.sort_by(|a, b| b.ships.cmp(&a.ships).then_with(|| a.id.cmp(&b.id)));
    let n_fleets = fleets.len().min(TRANSFORMER_MAX_FLEETS);
    for i in 0..n_fleets {
        let f = fleets[i];
        let slot = 1 + TRANSFORMER_MAX_PLANETS + i;
        mask[slot] = true;
        let t = &mut tokens[slot];
        let (om, oo, on) = rel_owner(f.owner, me);
        t[1] = 1.0; // fleet token
        t[2] = om;
        t[3] = oo;
        t[4] = on;
        t[5] = ((f.x - crate::CENTER_X) / 50.0) as f32;
        t[6] = ((f.y - crate::CENTER_Y) / 50.0) as f32;
        t[8] = token_ships(f.ships);
        t[21] = f.angle.sin() as f32;
        t[22] = f.angle.cos() as f32;
        let from_owner = state
            .planets
            .iter()
            .find(|p| p.id == f.from_planet_id)
            .map(|p| p.owner)
            .unwrap_or(f.owner);
        let (fm, _, _) = rel_owner(from_owner, me);
        t[23] = fm;
    }

    (tokens, mask)
}

fn layer_norm_into(input: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) {
    let d = input.len();
    let mean = input.iter().sum::<f32>() / d as f32;
    let mut var = 0.0;
    for &v in input {
        let z = v - mean;
        var += z * z;
    }
    let inv = (var / d as f32 + 1e-5).sqrt().recip();
    for i in 0..d {
        out[i] = (input[i] - mean) * inv * gamma[i] + beta[i];
    }
}

fn linear_row_major(input: &[f32], w: &[f32], b: &[f32], out_dim: usize, in_dim: usize, out: &mut [f32]) {
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        out[o] = dot_neon(row, input, b[o], in_dim);
    }
}

fn transformer_forward(w: &TransformerWeights, state: &GameState, me: i32) -> f32 {
    let (tokens, mask) = transformer_tokens(state, me);
    let n = w.max_tokens;
    let d = w.d_model;
    let h = w.heads;
    let hd = d / h;
    let mut x = vec![0.0f32; n * d];
    x[..d].copy_from_slice(&w.cls);
    for tok in 1..n {
        if !mask[tok] {
            continue;
        }
        let src = &tokens[tok];
        for j in 0..d {
            x[tok * d + j] = dot_neon(
                &w.embed_w[j * w.token_dim..(j + 1) * w.token_dim],
                src,
                w.embed_b[j],
                w.token_dim,
            );
        }
    }

    let mut norm = vec![0.0f32; n * d];
    let mut qkv = vec![0.0f32; n * 3 * d];
    let mut attn_out = vec![0.0f32; n * d];
    let mut proj = vec![0.0f32; n * d];
    let mut ff_hidden = vec![0.0f32; w.ff_dim];
    let mut ff_out = vec![0.0f32; d];
    let mut scores = vec![0.0f32; n];

    for block in &w.blocks {
        for tok in 0..n {
            if mask[tok] {
                layer_norm_into(
                    &x[tok * d..(tok + 1) * d],
                    &block.ln1_w,
                    &block.ln1_b,
                    &mut norm[tok * d..(tok + 1) * d],
                );
                linear_row_major(
                    &norm[tok * d..(tok + 1) * d],
                    &block.qkv_w,
                    &block.qkv_b,
                    3 * d,
                    d,
                    &mut qkv[tok * 3 * d..(tok + 1) * 3 * d],
                );
            }
        }
        attn_out.fill(0.0);
        let scale = (hd as f32).sqrt().recip();
        for head in 0..h {
            for tq in 0..n {
                if !mask[tq] {
                    continue;
                }
                let q_base = tq * 3 * d + head * hd;
                let mut max_s = f32::NEG_INFINITY;
                for tk in 0..n {
                    if !mask[tk] {
                        scores[tk] = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_base = tk * 3 * d + d + head * hd;
                    let mut s = 0.0;
                    for j in 0..hd {
                        s += qkv[q_base + j] * qkv[k_base + j];
                    }
                    s *= scale;
                    scores[tk] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut denom = 0.0;
                for tk in 0..n {
                    if mask[tk] {
                        let e = (scores[tk] - max_s).exp();
                        scores[tk] = e;
                        denom += e;
                    }
                }
                let out_base = tq * d + head * hd;
                for tk in 0..n {
                    if !mask[tk] {
                        continue;
                    }
                    let a = scores[tk] / denom.max(1e-12);
                    let v_base = tk * 3 * d + 2 * d + head * hd;
                    for j in 0..hd {
                        attn_out[out_base + j] += a * qkv[v_base + j];
                    }
                }
            }
        }
        for tok in 0..n {
            if !mask[tok] {
                continue;
            }
            linear_row_major(
                &attn_out[tok * d..(tok + 1) * d],
                &block.out_w,
                &block.out_b,
                d,
                d,
                &mut proj[tok * d..(tok + 1) * d],
            );
            for j in 0..d {
                x[tok * d + j] += proj[tok * d + j];
            }
        }

        for tok in 0..n {
            if !mask[tok] {
                continue;
            }
            layer_norm_into(
                &x[tok * d..(tok + 1) * d],
                &block.ln2_w,
                &block.ln2_b,
                &mut norm[tok * d..(tok + 1) * d],
            );
            linear_row_major(
                &norm[tok * d..(tok + 1) * d],
                &block.ff1_w,
                &block.ff1_b,
                w.ff_dim,
                d,
                &mut ff_hidden,
            );
            for v in &mut ff_hidden {
                *v = v.max(0.0);
            }
            linear_row_major(&ff_hidden, &block.ff2_w, &block.ff2_b, d, w.ff_dim, &mut ff_out);
            for j in 0..d {
                x[tok * d + j] += ff_out[j];
            }
        }
    }

    let mut cls = vec![0.0f32; d];
    layer_norm_into(&x[..d], &w.ln_f_w, &w.ln_f_b, &mut cls);
    if w.summary_hidden == 0 {
        return dot_neon(&w.head_w, &cls, w.head_b, d).tanh();
    }

    let summary_v2;
    let summary_v3;
    let summary: &[f32] = if w.summary_dim == summary_features_v3::DIM {
        summary_v3 = summary_features_v3::extract(state, me);
        &summary_v3
    } else {
        summary_v2 = summary_features_v2::extract(state, me);
        &summary_v2
    };
    let mut summary_hidden = vec![0.0f32; w.summary_hidden];
    linear_row_major(
        &summary,
        &w.summary_w,
        &w.summary_b,
        w.summary_hidden,
        w.summary_dim,
        &mut summary_hidden,
    );
    for v in &mut summary_hidden {
        *v = v.max(0.0);
    }
    let mut out = w.head_b;
    for j in 0..d {
        out += w.head_w[j] * cls[j];
    }
    for j in 0..w.summary_hidden {
        out += w.head_w[d + j] * summary_hidden[j];
    }
    out.tanh()
}

/// Run the value net on `state` from `me`'s perspective. Returns `None`
/// if no weights are loaded (caller should fall back to the heuristic).
/// Output is in `[-1, 1]` — MY perspective.
pub fn predict(state: &GameState, me: i32) -> Option<f64> {
    let y = match weights()? {
        LoadedWeights::Mlp(w) => match w.kind {
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
        },
        LoadedWeights::Transformer(w) => transformer_forward(w, state, me),
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

/// v3 = summary_v2 plus cheap route / relation aggregates.
///
/// Extra 20 features are 4 relation groups × 5 stats:
///   relation groups: ally->neutral, ally->enemy, enemy->ally, enemy->neutral
///   stats: count/100, min_travel_time/100, mean_travel_time/100,
///          direct_sun_clear_fraction, capture_feasible_fraction
///
/// The path features are intentionally cheap. They use direct distance and a
/// sun-intersection check, not the full moving-obstacle pather. That keeps leaf
/// eval inference predictable while still telling the value net when a board is
/// geometrically awkward.
pub mod summary_features_v3 {
    use super::*;

    pub const ROUTE_DIM: usize = 20;
    pub const DIM: usize = summary_features_v2::DIM + ROUTE_DIM;

    const REL_MY_NEUTRAL: usize = 0;
    const REL_MY_ENEMY: usize = 1;
    const REL_ENEMY_MY: usize = 2;
    const REL_ENEMY_NEUTRAL: usize = 3;
    const RELS: usize = 4;
    const STATS: usize = 5;

    fn rel_group(src_owner: i32, dst_owner: i32, me: i32) -> Option<usize> {
        if src_owner == me && dst_owner == -1 {
            Some(REL_MY_NEUTRAL)
        } else if src_owner == me && dst_owner != me && dst_owner != -1 {
            Some(REL_MY_ENEMY)
        } else if src_owner != me && src_owner != -1 && dst_owner == me {
            Some(REL_ENEMY_MY)
        } else if src_owner != me && src_owner != -1 && dst_owner == -1 {
            Some(REL_ENEMY_NEUTRAL)
        } else {
            None
        }
    }

    fn route_features(state: &GameState, me: i32) -> [f32; ROUTE_DIM] {
        let mut count = [0.0f32; RELS];
        let mut min_t = [1000.0f32; RELS];
        let mut sum_t = [0.0f32; RELS];
        let mut clear = [0.0f32; RELS];
        let mut feasible = [0.0f32; RELS];
        for src in &state.planets {
            if src.owner == -1 || src.ships <= 0 {
                continue;
            }
            for dst in &state.planets {
                if src.id == dst.id {
                    continue;
                }
                let Some(g) = rel_group(src.owner, dst.owner, me) else {
                    continue;
                };
                let dx = src.x - dst.x;
                let dy = src.y - dst.y;
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);
                let send = if dst.owner == src.owner {
                    src.ships.max(1)
                } else {
                    (dst.ships + 1).clamp(1, src.ships.max(1))
                };
                let speed = fleet_speed(send, state.max_speed).max(0.01);
                let travel = (dist / speed).min(500.0) as f32;
                let sun_blocked = point_to_segment_distance(
                    (crate::CENTER_X, crate::CENTER_Y),
                    (src.x, src.y),
                    (dst.x, dst.y),
                ) < crate::SUN_RADIUS;
                count[g] += 1.0;
                sum_t[g] += travel;
                if travel < min_t[g] {
                    min_t[g] = travel;
                }
                if !sun_blocked {
                    clear[g] += 1.0;
                }
                if src.owner == dst.owner || src.ships > dst.ships {
                    feasible[g] += 1.0;
                }
            }
        }

        let mut out = [0.0f32; ROUTE_DIM];
        for g in 0..RELS {
            let base = g * STATS;
            let n = count[g].max(1.0);
            out[base] = (count[g] / 100.0).clamp(0.0, 2.0);
            out[base + 1] = if count[g] > 0.0 { (min_t[g] / 100.0).clamp(0.0, 5.0) } else { 5.0 };
            out[base + 2] = if count[g] > 0.0 { (sum_t[g] / n / 100.0).clamp(0.0, 5.0) } else { 5.0 };
            out[base + 3] = clear[g] / n;
            out[base + 4] = feasible[g] / n;
        }
        out
    }

    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        let base = summary_features_v2::extract(state, me);
        let route = route_features(state, me);
        let mut out = [0.0f32; DIM];
        out[..summary_features_v2::DIM].copy_from_slice(&base);
        out[summary_features_v2::DIM..].copy_from_slice(&route);
        out
    }
}
