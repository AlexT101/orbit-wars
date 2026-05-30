//! Pure-Rust XGBoost binary-logistic inference.
//!
//! Parses an XGBoost `bst.save_model('*.json')` dump and walks each tree per
//! inference. Supports `binary:logistic` only (the objective we train value
//! nets with). For each tree, walk from node 0 using `split_indices` /
//! `split_conditions` until reaching a leaf (`left_children[i] == -1`); the
//! leaf's `base_weights[i]` is its logit contribution. Sum across all trees
//! plus the base-score logit, sigmoid to a probability, then map to a value
//! in `[-1, 1]` (matching the MLP's tanh-output convention).
//!
//! Loaded via `ALPHAOW_VALUE_NET_PATH` (auto-detected by leading `{` vs
//! AOWV magic byte). See `value_net.rs::weights()`.

use serde::Deserialize;

#[derive(Deserialize)]
struct XgbDump {
    learner: Learner,
}
#[derive(Deserialize)]
struct Learner {
    gradient_booster: GradientBooster,
    learner_model_param: ModelParam,
    objective: Objective,
}
#[derive(Deserialize)]
struct Objective {
    name: String,
}
#[derive(Deserialize)]
struct ModelParam {
    base_score: String,
    num_feature: String,
}
#[derive(Deserialize)]
struct GradientBooster {
    model: BoosterModel,
}
#[derive(Deserialize)]
struct BoosterModel {
    trees: Vec<TreeJson>,
}
#[derive(Deserialize)]
struct TreeJson {
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_indices: Vec<i32>,
    split_conditions: Vec<f32>,
    #[allow(dead_code)]
    default_left: Vec<u8>,
    base_weights: Vec<f32>,
}

/// A single packed tree. Internal nodes hold (feat_idx, threshold,
/// left_idx, right_idx); leaves hold the logit contribution in `value`.
/// `feat_idx == u32::MAX` flags a leaf so the inner loop is branch-light.
struct CompiledTree {
    feat_idx: Vec<u32>,
    threshold: Vec<f32>,
    left_or_value: Vec<u32>,   // u32 child idx for internal, leaf logit bits for leaf
    right: Vec<u32>,
}

const LEAF_MARKER: u32 = u32::MAX;

impl CompiledTree {
    #[inline(always)]
    fn predict(&self, x: &[f32]) -> f32 {
        let mut i = 0usize;
        loop {
            let f = self.feat_idx[i];
            if f == LEAF_MARKER {
                // The leaf logit was stored as f32 bits in left_or_value
                return f32::from_bits(self.left_or_value[i]);
            }
            // SAFETY/perf: feat_idx was validated < num_feature at load time
            let v = x[f as usize];
            i = if v < self.threshold[i] {
                self.left_or_value[i] as usize
            } else {
                self.right[i] as usize
            };
        }
    }
}

pub struct XgbModel {
    trees: Vec<CompiledTree>,
    pub base_score_logit: f32,
    pub num_feature: usize,
}

impl XgbModel {
    /// Total logit (before sigmoid) for input `x`.
    pub fn predict_logit(&self, x: &[f32]) -> f32 {
        let mut s = self.base_score_logit;
        for t in &self.trees {
            s += t.predict(x);
        }
        s
    }

    /// Output mapped to `[-1, 1]` to match the MLP tanh convention used by
    /// the rest of the bot. `2*sigmoid(z) - 1 = tanh(z/2)`.
    pub fn predict_value(&self, x: &[f32]) -> f32 {
        let z = self.predict_logit(x);
        // tanh(z/2) is numerically nicer than 2/(1+e^-z) - 1 for large |z|.
        (z * 0.5).tanh()
    }
}

/// Quick check: looks like JSON (skips leading whitespace)?
pub fn looks_like_json(bytes: &[u8]) -> bool {
    bytes.iter().copied().find(|b| !b.is_ascii_whitespace()) == Some(b'{')
}

pub fn load(bytes: &[u8]) -> Option<XgbModel> {
    let dump: XgbDump = match serde_json::from_slice(bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[alphaow] xgb JSON parse error: {}", e);
            return None;
        }
    };
    let obj = &dump.learner.objective.name;
    if obj != "binary:logistic" {
        eprintln!("[alphaow] xgb unsupported objective: {} (only binary:logistic)", obj);
        return None;
    }
    // base_score in XGB JSON is stored as a JSON string like "[5E-1]" or "0.5".
    // Strip optional brackets and parse as f32 (probability).
    let bs_str = dump
        .learner
        .learner_model_param
        .base_score
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    let base_prob: f32 = bs_str.parse().ok()?;
    let base_score_logit = if base_prob > 0.0 && base_prob < 1.0 {
        (base_prob / (1.0 - base_prob)).ln()
    } else {
        0.0
    };
    let num_feature: usize = dump
        .learner
        .learner_model_param
        .num_feature
        .parse()
        .ok()?;

    let mut trees = Vec::with_capacity(dump.learner.gradient_booster.model.trees.len());
    for t in &dump.learner.gradient_booster.model.trees {
        let n = t.left_children.len();
        if t.right_children.len() != n
            || t.split_indices.len() != n
            || t.split_conditions.len() != n
            || t.base_weights.len() != n
        {
            eprintln!("[alphaow] xgb tree has inconsistent per-node array lengths");
            return None;
        }
        let mut feat_idx = vec![0u32; n];
        let mut threshold = vec![0f32; n];
        let mut left_or_value = vec![0u32; n];
        let mut right = vec![0u32; n];
        for i in 0..n {
            let lc = t.left_children[i];
            if lc < 0 {
                feat_idx[i] = LEAF_MARKER;
                left_or_value[i] = t.base_weights[i].to_bits();
            } else {
                let fi = t.split_indices[i];
                if fi < 0 || (fi as usize) >= num_feature {
                    eprintln!("[alphaow] xgb tree feat_idx {} out of range", fi);
                    return None;
                }
                feat_idx[i] = fi as u32;
                threshold[i] = t.split_conditions[i];
                left_or_value[i] = lc as u32;
                right[i] = t.right_children[i].max(0) as u32;
            }
        }
        trees.push(CompiledTree { feat_idx, threshold, left_or_value, right });
    }
    Some(XgbModel { trees, base_score_logit, num_feature })
}
