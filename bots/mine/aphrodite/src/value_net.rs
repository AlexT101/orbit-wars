//! Value network for aphrodite leaf evaluation.
//!
//! Two-stream input per leaf state:
//!   1. CURRENT — `(owner, ships, radius, type, production)` for every
//!      object (planets + comets).
//!   2. EXTRAPOLATED — same per-object features but with every in-flight
//!      fleet already resolved into its predicted target with the corrected
//!      extrapolation combat.
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
//! ## Weight format (`APHRODITE_VALUE_NET_PATH`)
//!
//! XGBoost JSON model (`bst.save_model("...json")`), currently trained on
//! SummaryV2 features.
//!
//! Forward pass: `y = tanh(b2 + w2 · ReLU(w1 · x + b1))`. Output is a
//! scalar in `[-1, 1]` interpreted as MCTS value from MY perspective.
//!
//! If no weights file is found or the file is malformed, `predict`
//! returns `None`. Callers should fall back to the duck heuristic.

use crate::apollo::cache::EntityCache;
use crate::ow2_plan::cached_predict_fleet_collision;
use crate::{GameState, Planet};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const MAX_OBJECTS: usize = 44;
pub const PER_OBJECT: usize = 9;
pub const PER_BLOCK: usize = MAX_OBJECTS * PER_OBJECT;
pub const DIST_BLOCK: usize = MAX_OBJECTS * MAX_OBJECTS;
pub const INPUT_DIM: usize = 2 * PER_BLOCK + DIST_BLOCK;

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

fn pack_object(
    buf: &mut [f32; PER_BLOCK],
    slot: usize,
    p: &Planet,
    ships: i64,
    owner: i32,
    me: i32,
) {
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
/// planet. Fleets that die in the sun or fly off the board contribute
/// nothing.
///
/// Combat at each planet processes arrivals in arrival-time order, adds
/// production on owned planets between arrival ticks, groups same-tick
/// arrivals by owner before combat, and handles the tied-attacker rule.
pub fn extrapolate_fleets(state: &GameState) -> HashMap<i64, (i32, i64)> {
    let mut arrivals: HashMap<i64, Vec<(i64, i32, i64)>> = HashMap::new();
    for fleet in &state.fleets {
        if let Some((pid, dt)) = cached_predict_fleet_collision(fleet, state) {
            arrivals
                .entry(pid)
                .or_default()
                .push((dt, fleet.owner, fleet.ships));
        }
    }
    let prod_for: HashMap<i64, i64> = state.planets.iter().map(|p| (p.id, p.production)).collect();
    let mut result: HashMap<i64, (i32, i64)> = state
        .planets
        .iter()
        .map(|p| (p.id, (p.owner, p.ships)))
        .collect();
    for (pid, mut arrs) in arrivals {
        arrs.sort_by_key(|x| x.0);
        let entry = result.entry(pid).or_insert((-1, 0));
        let (mut owner, mut ships) = *entry;
        let prod = *prod_for.get(&pid).unwrap_or(&0);
        let mut cur_t = 0i64;
        let mut i = 0;
        while i < arrs.len() {
            let t = arrs[i].0;
            if owner != -1 && t > cur_t {
                ships += prod * (t - cur_t);
            }
            // Aggregate same-tick arrivals by owner (matches engine
            // Combat step 1).
            let mut by_owner: HashMap<i32, i64> = HashMap::new();
            while i < arrs.len() && arrs[i].0 == t {
                *by_owner.entry(arrs[i].1).or_insert(0) += arrs[i].2;
                i += 1;
            }
            let mut sorted: Vec<(i32, i64)> = by_owner.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let (top_owner, top_ships) = sorted[0];
            let (sv_owner, sv_ships) = if sorted.len() > 1 {
                let second = sorted[1].1;
                if top_ships == second {
                    (-1, 0) // tied attackers all destroyed
                } else {
                    (top_owner, top_ships - second)
                }
            } else {
                (top_owner, top_ships)
            };
            if sv_ships > 0 {
                if sv_owner == owner {
                    ships += sv_ships;
                } else if sv_ships > ships {
                    owner = sv_owner;
                    ships = sv_ships - ships;
                } else {
                    ships -= sv_ships;
                }
            }
            cur_t = t;
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
    Features {
        current,
        extrap,
        dist,
    }
}

/// Input variant for the loaded weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum InputKind {
    /// 2728-d raw two-stream + distance matrix.
    Full,
    /// 46-d v2 summary (per-player + extrap + neutral block;
    /// see `summary_features_v2::extract`).
    SummaryV2,
}

/// Map a model's declared input width to the feature variant that produces it.
fn detect_kind(input_dim: usize) -> Option<InputKind> {
    if input_dim == INPUT_DIM {
        Some(InputKind::Full)
    } else if input_dim == summary_features_v2::DIM {
        Some(InputKind::SummaryV2)
    } else {
        eprintln!(
            "[aphrodite] unknown input_dim={} (expected {} full / {} summary_v2)",
            input_dim,
            INPUT_DIM,
            summary_features_v2::DIM
        );
        None
    }
}

/// Loaded XGBoost gbtree dump (`bst.save_model('*.json')`).
enum Model {
    Xgb {
        model: crate::xgb::XgbModel,
        kind: InputKind,
    },
}

fn load_weights() -> Option<Model> {
    load_weights_from("APHRODITE_VALUE_NET_PATH", false)
}

/// Load an XGB value net from the file named by env var `var`. `optional`
/// suppresses the "not set" warning for nets allowed to be absent (e.g. the
/// 2-players-left net).
fn load_weights_from(var: &str, optional: bool) -> Option<Model> {
    let path = match std::env::var(var) {
        Ok(p) => p,
        Err(_) => {
            if !optional {
                eprintln!("[aphrodite] {} not set; using duck heuristic", var);
            }
            return None;
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[aphrodite] could not read weights at {}: {}", path, e);
            return None;
        }
    };
    if !crate::xgb::looks_like_json(&bytes) {
        eprintln!("[aphrodite] value net must be an XGB JSON model: {}", path);
        return None;
    }
    let model = match crate::xgb::load(&bytes) {
        Some(m) => m,
        None => {
            eprintln!("[aphrodite] failed to parse XGB JSON at {}", path);
            return None;
        }
    };
    let kind = detect_kind(model.num_feature)?;
    eprintln!(
        "[aphrodite] loaded XGB value net (kind={:?}, num_feature={}, base_score_logit={:.4}) from {}",
        kind, model.num_feature, model.base_score_logit, path
    );
    Some(Model::Xgb { model, kind })
}

static WEIGHTS: OnceLock<Option<Model>> = OnceLock::new();

fn weights() -> Option<&'static Model> {
    WEIGHTS.get_or_init(load_weights).as_ref()
}

static WEIGHTS_2P: OnceLock<Option<Model>> = OnceLock::new();

/// Optional secondary net, used when a position has only two players left
/// alive (a 4p game collapsed to a 1v1). Loaded from
/// `APHRODITE_VALUE_NET_PATH_2P`; absent is fine (we fall back to the primary).
fn weights_2p() -> Option<&'static Model> {
    WEIGHTS_2P
        .get_or_init(|| load_weights_from("APHRODITE_VALUE_NET_PATH_2P", true))
        .as_ref()
}

/// True iff weights are loaded and ready for inference.
pub fn is_ready() -> bool {
    weights().is_some()
}

/// Number of players with at least one planet or in-flight fleet. Mirrors
/// `apollo::helpers::count_alive_players`, but over the crate `GameState`
/// types the value net works with.
fn count_alive_players(state: &GameState) -> usize {
    let mut alive = [false; 8];
    for p in &state.planets {
        if p.owner >= 0 && (p.owner as usize) < alive.len() {
            alive[p.owner as usize] = true;
        }
    }
    for f in &state.fleets {
        if f.owner >= 0 && (f.owner as usize) < alive.len() {
            alive[f.owner as usize] = true;
        }
    }
    alive.iter().filter(|&&a| a).count()
}

/// Run the value net on `state` from `me`'s perspective, reusing a prebuilt
/// apollo `EntityCache` (e.g. duct's per-search shared cache) for any aim-based
/// features. Caller must have set the cache's current turn to `state.step`
/// (duct's `with_cache_at` does this). Returns `None` if no weights are loaded.
/// Output is in `[-1, 1]` — MY perspective.
pub fn predict_with_cache(state: &GameState, me: i32, cache: &EntityCache) -> Option<f64> {
    // Once only two players are alive the position is effectively 2-player, so
    // score it with the dedicated 2p net when one is loaded. The count comes
    // from the evaluated state, so a 4p game's late 2-survivor leaves switch to
    // the 2p model automatically. Falls back to the primary net if no 2p net.
    let two_left = count_alive_players(state) == 2;
    let m = if two_left {
        weights_2p().or_else(weights)
    } else {
        weights()
    }?;
    let y = match m {
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
            InputKind::SummaryV2 => {
                let feats = summary_features_v2::extract_with_cache(state, me, cache);
                model.predict_value(&feats)
            }
        },
    };
    Some(y as f64)
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
    use crate::apollo::constants::OFFSET_LOOKAHEAD;
    use crate::apollo::strategy::resolve_shot;

    pub const DIM: usize = 65;

    /// Turn horizon for the relational block (frontier / pressure / support /
    /// inbound-outbound). Matches the "within 10 turns" spec.
    const REL_HORIZON: i64 = 10;

    /// Monotone-decreasing distance weight over `REL_HORIZON`. `t = 0` (lands
    /// now) → 1.0; every turn within the horizon keeps positive weight; beyond
    /// the horizon → 0. Used identically for static-garrison turn-distance and
    /// in-flight fleet ETA so a fleet `t` turns out and a garrison `t` turns
    /// away weigh the same per ship.
    #[inline]
    fn rel_weight(turns: i64) -> f32 {
        if turns < 0 || turns > REL_HORIZON {
            0.0
        } else {
            (REL_HORIZON + 1 - turns) as f32 / (REL_HORIZON + 1) as f32
        }
    }

    #[inline]
    fn mean(sum: f32, count: usize) -> f32 {
        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }

    /// Distance-weighted ship "pressure" `src` can project onto `dst`. Unlike
    /// `reach_turns` this sweeps *every* launch offset (no early exit) and takes
    /// the **max** contribution, because a later launch carries more ships
    /// (production accrued) but lands with lower weight — the worst-case threat
    /// can peak at any offset. Ship count at offset `o` = `garrison + prod*o`
    /// (owner assumed constant over the short sweep); that count also feeds the
    /// shot so faster large fleets are reflected.
    fn pressure_from(cache: &EntityCache, src: &Planet, dst_id: i64) -> f32 {
        let mut best = 0.0f32;
        for off in 0..=OFFSET_LOOKAHEAD {
            let ships = (src.ships + src.production * off).max(1);
            if let Some(r) = resolve_shot(cache, src.id, dst_id, ships, off, None) {
                let contrib = ships as f32 * rel_weight(off + r.1);
                if contrib > best {
                    best = contrib;
                }
            }
        }
        best
    }

    /// 24-d relational block (shared, not per-player). "Enemy" = every non-me,
    /// non-neutral planet (neutrals exert no pressure). Pressure/support come
    /// from apollo's aimer (`resolve_shot`), so sun/planet blocking and orbital
    /// motion are respected; in-flight fleets are folded into pressure/support
    /// at their predicted destination, weighted by ETA, so launching a fleet
    /// immediately raises the target's threat and (via the emptied source
    /// garrison) its source's exposure. Shares/fractions are me-vs-all-enemies
    /// ratios; `step` lets the tree discount the early-game volatility of shares.
    fn relational_block(state: &GameState, me: i32, cache: &EntityCache) -> [f32; 24] {
        let planets = &state.planets;
        let n = planets.len();
        let mine: Vec<usize> = (0..n).filter(|&i| planets[i].owner == me).collect();
        let enemy: Vec<usize> = (0..n)
            .filter(|&i| planets[i].owner != me && planets[i].owner != -1)
            .collect();

        // In-flight fleet arrivals: dest planet id -> (owner, ships, eta).
        let mut arrivals: HashMap<i64, Vec<(i32, i64, i64)>> = HashMap::new();
        for f in &state.fleets {
            if let Some((pid, dt)) = cached_predict_fleet_collision(f, state) {
                arrivals
                    .entry(pid)
                    .or_default()
                    .push((f.owner, f.ships, dt));
            }
        }
        // Distance-weighted inbound fleet ships at planet `pid` for one side.
        let inbound_weight = |pid: i64, want_me: bool| -> f32 {
            let mut s = 0.0f32;
            if let Some(v) = arrivals.get(&pid) {
                for &(o, sh, eta) in v {
                    let is_me = o == me;
                    let is_enemy = o != me && o != -1;
                    if (want_me && is_me) || (!want_me && is_enemy) {
                        s += sh as f32 * rel_weight(eta);
                    }
                }
            }
            s
        };

        // Summed static pressure on planet[d] from a set of source planets.
        let pressure_on = |d: usize, srcs: &[usize]| -> f32 {
            let dst_id = planets[d].id;
            let mut s = 0.0f32;
            for &si in srcs {
                if si == d {
                    continue;
                }
                s += pressure_from(cache, &planets[si], dst_id);
            }
            s
        };

        // Pressure / support / vulnerability over my planets. Also track the
        // worst single-planet threat and the production sitting on vulnerable
        // planets (production-at-risk is sharper than a raw vulnerable count).
        // (prod-only ablation: unweighted avg_enemy_pressure dropped.)
        let mut sum_ally_support = 0.0f32;
        let mut n_my_vuln = 0usize;
        let mut max_enemy_pressure = 0.0f32;
        let mut my_prod_at_risk = 0.0f32;
        let mut pw_enemy_pressure = 0.0f32; // Σ threat·prod over my planets
        for &d in &mine {
            let pid = planets[d].id;
            let threat = pressure_on(d, &enemy) + inbound_weight(pid, false);
            let support = pressure_on(d, &mine) + inbound_weight(pid, true);
            sum_ally_support += support;
            pw_enemy_pressure += threat * planets[d].production as f32;
            if threat > max_enemy_pressure {
                max_enemy_pressure = threat;
            }
            // Defense includes the planet's own garrison, not just neighbor
            // support — a well-garrisoned lone planet should not read as
            // vulnerable. (Deviates from the literal "pressure > support".)
            if threat > support + planets[d].ships as f32 {
                n_my_vuln += 1;
                my_prod_at_risk += planets[d].production as f32;
            }
        }
        // Pressure / support / vulnerability over enemy planets (reverse). The
        // enemy-side production-at-risk is *my* production-at-opportunity.
        let mut sum_enemy_support = 0.0f32;
        let mut n_enemy_vuln = 0usize;
        let mut max_ally_pressure = 0.0f32;
        let mut my_prod_at_opportunity = 0.0f32;
        let mut pw_ally_pressure = 0.0f32; // Σ threat·prod over enemy planets
        for &d in &enemy {
            let pid = planets[d].id;
            let threat = pressure_on(d, &mine) + inbound_weight(pid, true);
            let support = pressure_on(d, &enemy) + inbound_weight(pid, false);
            sum_enemy_support += support;
            pw_ally_pressure += threat * planets[d].production as f32;
            if threat > max_ally_pressure {
                max_ally_pressure = threat;
            }
            if threat > support + planets[d].ships as f32 {
                n_enemy_vuln += 1;
                my_prod_at_opportunity += planets[d].production as f32;
            }
        }

        // Ship / production totals per side (ships count garrisons + in-flight).
        let ally_all_ships: f32 = mine.iter().map(|&i| planets[i].ships as f32).sum();
        let enemy_all_ships: f32 = enemy.iter().map(|&i| planets[i].ships as f32).sum();
        let avg_ally_ships = mean(ally_all_ships, mine.len());
        let avg_enemy_ships = mean(enemy_all_ships, enemy.len());
        let my_prod: f32 = mine.iter().map(|&i| planets[i].production as f32).sum();
        let enemy_prod: f32 = enemy.iter().map(|&i| planets[i].production as f32).sum();
        let mut my_flying = 0.0f32;
        let mut enemy_flying = 0.0f32;
        for f in &state.fleets {
            if f.owner == me {
                my_flying += f.ships as f32;
            } else if f.owner != -1 {
                enemy_flying += f.ships as f32;
            }
        }
        // share = mine / (mine + enemy); 0.5 (even) when neither side has any.
        let share = |a: f32, b: f32| -> f32 {
            let t = a + b;
            if t > 0.0 {
                a / t
            } else {
                0.5
            }
        };
        let ship_share = share(ally_all_ships + my_flying, enemy_all_ships + enemy_flying);
        let production_share = share(my_prod, enemy_prod);
        // fraction of a side's force currently committed to fleets; 0 if no force.
        let committed = |flying: f32, on_planets: f32| -> f32 {
            let t = flying + on_planets;
            if t > 0.0 {
                flying / t
            } else {
                0.0
            }
        };
        let my_fleet_fraction = committed(my_flying, ally_all_ships);
        let enemy_fleet_fraction = committed(enemy_flying, enemy_all_ships);

        // Centroid-to-centroid distance (front-line proximity of the empires).
        let centroid = |idxs: &[usize]| -> Option<(f64, f64)> {
            if idxs.is_empty() {
                return None;
            }
            let mut sx = 0.0;
            let mut sy = 0.0;
            for &i in idxs {
                sx += planets[i].x;
                sy += planets[i].y;
            }
            Some((sx / idxs.len() as f64, sy / idxs.len() as f64))
        };
        let centroid_dist = match (centroid(&mine), centroid(&enemy)) {
            (Some(a), Some(b)) => (((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()) as f32,
            _ => 0.0,
        };

        // (prod-only ablation: unweighted separation dropped in favor of
        // economic dispersion below.)

        // Economic dispersion: RMS distance of production mass from its
        // production-weighted centroid (how spread out a side's *economy* is).
        let dispersion = |idxs: &[usize]| -> f32 {
            let wsum: f64 = idxs.iter().map(|&i| planets[i].production as f64).sum();
            if wsum <= 0.0 {
                return 0.0;
            }
            let mut cx = 0.0;
            let mut cy = 0.0;
            for &i in idxs {
                let w = planets[i].production as f64;
                cx += w * planets[i].x;
                cy += w * planets[i].y;
            }
            cx /= wsum;
            cy /= wsum;
            let mut num = 0.0;
            for &i in idxs {
                let w = planets[i].production as f64;
                num += w * ((planets[i].x - cx).powi(2) + (planets[i].y - cy).powi(2));
            }
            (num / wsum).sqrt() as f32
        };

        // Production-weighted mean threat (threat aimed at valuable planets),
        // comparable in units to avg_*_pressure. Denominator is the side's
        // total production.
        let pw_enemy_pressure = if my_prod > 0.0 {
            pw_enemy_pressure / my_prod
        } else {
            0.0
        };
        let pw_ally_pressure = if enemy_prod > 0.0 {
            pw_ally_pressure / enemy_prod
        } else {
            0.0
        };

        // ── 4p / FFA standing ───────────────────────────────────────────────
        // Per-player strength (ships on planets + in-flight) and aliveness.
        // (Sparse arrivals-derived 4p features and the Euclidean border count
        // were ablated out — near-zero gain on limited 4p data.)
        const NP: usize = 4; // engine MAX_PLAYERS
        let mut strength = [0.0f32; NP];
        let mut alive = [false; NP];
        for p in planets {
            if p.owner >= 0 && (p.owner as usize) < NP {
                strength[p.owner as usize] += p.ships as f32;
                alive[p.owner as usize] = true;
            }
        }
        for f in &state.fleets {
            if f.owner >= 0 && (f.owner as usize) < NP {
                strength[f.owner as usize] += f.ships as f32;
                alive[f.owner as usize] = true;
            }
        }
        let me_u = (me as usize).min(NP - 1);
        let my_strength = strength[me_u];
        let n_alive = (0..NP).filter(|&p| alive[p]).count() as f32;
        // alive opponents (not me, not neutral)
        let opp_players: Vec<usize> = (0..NP).filter(|&p| alive[p] && p != me_u).collect();
        let max_opp = opp_players
            .iter()
            .map(|&p| strength[p])
            .fold(0.0f32, f32::max);
        let min_opp = opp_players
            .iter()
            .map(|&p| strength[p])
            .fold(f32::INFINITY, f32::min);
        let total_opp: f32 = opp_players.iter().map(|&p| strength[p]).sum();
        // rank: how many opponents are strictly stronger than me (0 = leader).
        let my_strength_rank = opp_players
            .iter()
            .filter(|&&p| strength[p] > my_strength)
            .count() as f32;
        // ratio to the strongest opponent (>1 ⇒ I lead; denom clamped ≥1).
        let leader_strength_ratio = my_strength / max_opp.max(1.0);
        // spread among opponents (low ⇒ balanced field likely to fight itself).
        let opponent_strength_spread = if opp_players.len() >= 2 && total_opp > 0.0 {
            (max_opp - min_opp) / total_opp
        } else {
            0.0
        };

        [
            state.step as f32,                    // 0  step
            avg_ally_ships,                       // 1  avg_ally_ships_per_planet
            avg_enemy_ships,                      // 2  avg_enemy_ships_per_planet
            mean(sum_ally_support, mine.len()),   // 3  avg_ally_support
            mean(sum_enemy_support, enemy.len()), // 4  avg_enemy_support
            n_my_vuln as f32,                     // 5  num_my_vulnerable_planets
            n_enemy_vuln as f32,                  // 6  num_enemy_vulnerable_planets
            ship_share,                           // 7  ship_share (me / me+enemy)
            production_share,                     // 8  production_share
            my_prod_at_risk,                      // 9  my_production_at_risk
            my_prod_at_opportunity,               // 10 enemy_production_at_opportunity
            max_enemy_pressure,                   // 11 max_enemy_pressure (on my planets)
            max_ally_pressure,                    // 12 max_ally_pressure (on enemy planets)
            centroid_dist,                        // 13 centroid_to_centroid
            my_fleet_fraction,                    // 14 my_fleet_fraction
            enemy_fleet_fraction,                 // 15 enemy_fleet_fraction
            pw_enemy_pressure,                    // 16 prod_weighted_enemy_pressure
            pw_ally_pressure,                     // 17 prod_weighted_ally_pressure
            dispersion(&mine),                    // 18 ally_economic_dispersion
            dispersion(&enemy),                   // 19 enemy_economic_dispersion
            my_strength_rank,                     // 20 my_strength_rank
            leader_strength_ratio,                // 21 leader_strength_ratio
            opponent_strength_spread,             // 22 opponent_strength_spread
            n_alive,                              // 23 n_alive_players
        ]
    }

    /// Compute min distance from object `o` to any planet whose
    /// classification function returns true. Returns `INF` if no such
    /// planet exists.
    fn min_dist_to<F: Fn(&Planet) -> bool>(planets: &[Planet], o_x: f64, o_y: f64, pred: F) -> f32 {
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

    /// 9-d per-player feature row for the CURRENT state.
    /// (prod_comet omitted: comets always have production 1, so it was an
    /// exact duplicate of n_comet — confirmed near-zero feature importance.)
    fn current_player_block(state: &GameState, p: i32, _cache: &EntityCache) -> [f32; 9] {
        let mut ships_on_planets = 0.0f32;
        let mut ships_flying = 0.0f32;
        let mut n_static = 0.0f32;
        let mut n_orbit = 0.0f32;
        let mut n_comet = 0.0f32;
        let mut prod_static = 0.0f32;
        let mut prod_orbit = 0.0f32;
        for planet in &state.planets {
            if planet.owner != p {
                continue;
            }
            ships_on_planets += planet.ships as f32;
            let prod = planet.production as f32;
            if planet.is_comet {
                n_comet += 1.0;
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
                let d_other = min_dist_to(&state.planets, o.x, o.y, |q| {
                    q.owner != p && q.owner != -1 && q.id != o.id
                });
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
            n_neutrals_closer,
            n_enemies_closer,
        ]
    }

    /// 8-d per-player feature row for the EXTRAPOLATED state
    /// (ships_flying is omitted — by construction, no fleets are still
    /// in flight after extrapolation; prod_comet omitted as in
    /// `current_player_block`).
    fn extrap_player_block(
        state: &GameState,
        p: i32,
        extrap: &std::collections::HashMap<i64, (i32, i64)>,
        _cache: &EntityCache,
    ) -> [f32; 8] {
        let mut ships_on_planets = 0.0f32;
        let mut n_static = 0.0f32;
        let mut n_orbit = 0.0f32;
        let mut n_comet = 0.0f32;
        let mut prod_static = 0.0f32;
        let mut prod_orbit = 0.0f32;
        // Build extrap-owner-by-planet lookup once.
        // (extrap already covers all planet IDs.)
        // For "closer to me" we evaluate using extrap owners, but the
        // planet positions are unchanged (extrapolation is about owner
        // and ship counts only).
        for planet in &state.planets {
            let (eo, es) = extrap
                .get(&planet.id)
                .copied()
                .unwrap_or((planet.owner, planet.ships));
            if eo != p {
                continue;
            }
            ships_on_planets += es as f32;
            let prod = planet.production as f32;
            if planet.is_comet {
                n_comet += 1.0;
            } else if planet.is_orbiting {
                n_orbit += 1.0;
                prod_orbit += prod;
            } else {
                n_static += 1.0;
                prod_static += prod;
            }
        }
        let owner_of = |id: i64| -> i32 {
            extrap.get(&id).map(|x| x.0).unwrap_or_else(|| {
                state
                    .planets
                    .iter()
                    .find(|p| p.id == id)
                    .map(|p| p.owner)
                    .unwrap_or(-1)
            })
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
            n_neutrals_closer,
            n_enemies_closer,
        ]
    }

    fn neutral_block(state: &GameState) -> [f32; 7] {
        let mut ships = 0.0f32;
        let mut n_static = 0.0f32;
        let mut n_orbit = 0.0f32;
        let mut n_comet = 0.0f32;
        let mut prod_static = 0.0f32;
        let mut prod_orbit = 0.0f32;
        let mut comet_time = 0.0f32;
        for planet in &state.planets {
            if planet.owner == -1 {
                ships += planet.ships as f32;
                let prod = planet.production as f32;
                if planet.is_comet {
                    n_comet += 1.0;
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

    /// This is the entry the `extract_v2` training binary uses (one cache build per row, offline).
    pub fn extract(state: &GameState, me: i32) -> [f32; DIM] {
        // Build throwaway cache (search callers should prefer predict_with_cache for efficiency)
        let mut cache = crate::apollo_bridge::rollout_cache(state);
        cache.set_current_turn(state.step);
        extract_with_cache(state, me, &cache)
    }

    /// Build the 65-d feature row, reusing a prebuilt apollo `EntityCache`.
    /// Caller must have set the cache's current turn to `state.step`. The cache
    /// is threaded to the relational block, whose pressure/support features
    /// query apollo's aimer (`resolve_shot`) per leaf.
    pub fn extract_with_cache(state: &GameState, me: i32, cache: &EntityCache) -> [f32; DIM] {
        let opp = dominant_enemy(state, me);
        let extrap = extrapolate_fleets(state);
        let me_cur = current_player_block(state, me, cache);
        let opp_cur = current_player_block(state, opp, cache);
        let me_ext = extrap_player_block(state, me, &extrap, cache);
        let opp_ext = extrap_player_block(state, opp, &extrap, cache);
        let neut = neutral_block(state);
        let rel = relational_block(state, me, cache);
        let mut out = [0f32; DIM];
        out[..9].copy_from_slice(&me_cur);
        out[9..18].copy_from_slice(&opp_cur);
        out[18..26].copy_from_slice(&me_ext);
        out[26..34].copy_from_slice(&opp_ext);
        out[34..41].copy_from_slice(&neut);
        out[41..65].copy_from_slice(&rel);
        out
    }
}

#[cfg(test)]
mod apollo_cache_tests {
    //! Scaffolding check: the value-net extractor can thread duct's apollo
    //! `EntityCache` and reach the aim path (`HellburnerModel::plan_shot`)
    //! in-crate, with no duplication of the apollo modules.
    use super::*;
    use crate::apollo::strategy::resolve_shot;

    fn static_planet(id: i64, owner: i32, x: f64, y: f64, ships: i64) -> Planet {
        let dx = x - crate::CENTER_X;
        let dy = y - crate::CENTER_Y;
        Planet {
            id,
            owner,
            x,
            y,
            radius: 1.5,
            ships,
            production: 1,
            orbital_radius: (dx * dx + dy * dy).sqrt(),
            initial_angle: dy.atan2(dx),
            is_orbiting: false,
            is_comet: false,
        }
    }

    /// Two static planets (corner positions, far from the sun) owned by p0/p1,
    /// with a clear horizontal lane between them at y=10.
    fn two_planet_state() -> GameState {
        GameState {
            player: 0,
            step: 0,
            planets: vec![
                static_planet(0, 0, 90.0, 10.0, 20),
                static_planet(1, 1, 10.0, 10.0, 20),
            ],
            fleets: vec![],
            angular_velocity: 0.03,
            comets: vec![],
            max_speed: 6.0,
            comet_speed: 4.0,
        }
    }

    #[test]
    fn cache_threads_into_extract_and_plan_shot() {
        let state = two_planet_state();

        // 1. Feature extraction with a prebuilt cache runs and is finite.
        let mut cache = crate::apollo_bridge::rollout_cache(&state);
        cache.set_current_turn(state.step);
        let feats = summary_features_v2::extract_with_cache(&state, 0, &cache);
        assert_eq!(feats.len(), summary_features_v2::DIM);
        assert!(feats.iter().all(|f| f.is_finite()));

        // 2. The apollo aim path is reachable from the same cache with no model:
        //    resolve a shot p0 -> p1 directly (L2/L3 cached). Some = aimable,
        //    None = no solution; both valid — only assert it doesn't panic.
        let _shot = resolve_shot(&cache, 0, 1, 10, 0, None);
    }
}
