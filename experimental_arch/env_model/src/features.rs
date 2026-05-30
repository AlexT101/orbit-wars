//! Model input features for Orbit Wars: turns an [`EngineState`] into the
//! tensors a policy/value net consumes. The bot calls [`encode`] on its native
//! state; training calls it through the `encode_obs` pyo3 wrapper — one
//! implementation, no train/serve skew.
//!
//! Frames: `NUM_FRAMES` snapshots at `t`, `t+1`, `t+10`, `t_resolved` (first
//! future turn with no fleets in flight). Each frame has `PLANET_SLOTS = 44`
//! planet tokens. Action space is `(44, 44, ACTIONS_DIM)` = `(source, target,
//! action)`: noop, send `{25%, 50%, 75%, 100%}` of the source's ships, a
//! constant `42`, or `target_resolved + 1` (min ships to take the target after
//! in-flight fleets resolve; invalid if it's ally-held then).
//!
//! Outputs (dict keys / shapes in [`Features::into_py_dict`]):
//!   - `tokens`   `(NUM_FRAMES, 44, TOKEN_DIM)`  per-planet features, all frames
//!   - `globals`  `(GLOBAL_DIM,)`                board-level summary at frame t
//!   - `presence` `(NUM_FRAMES, 44)`             1 where a slot holds a planet
//!   - `turns` / `angles` / `mask` `(44, 44, ACTIONS_DIM)`  decision frame (t) only
//!   - `ship_counts` `(44, 44, ACTIONS_DIM)` integer ships sent by each action
//!   - `reachable_mask` `(44, 44, ACTIONS_DIM)` 1 when the launch reaches target
//!
//! `turns`/`angles`/`mask`/`ship_counts`/`reachable_mask` are frame t only: the
//! policy acts now and the aim solve is the whole cost, so it isn't run for
//! lookahead frames.
//!
//! Aim solver: one geometric solve per `(frame, i, j, count)` — lead the moving
//! target for a launch angle, then project the straight-line fleet turn-by-turn
//! against planets / sun / board using the engine's own collision code, so
//! validity is bit-identical to what the engine accepts. Planet trajectories are
//! forward-simulated once and shared across frames.

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    distance, fleet_speed, point_to_segment_distance, swept_pair_hit, EngineState, MoveAction,
    Planet, BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
};

use std::cell::RefCell;

thread_local! {
    /// Reused per-projection scratch marking which planet ids survive the
    /// broad-phase cull. Avoids an allocation on every `project` call.
    static CULL_KEEP: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
}

pub const PLANET_SLOTS: usize = 44;
pub const ACTIONS_DIM: usize = 7;
pub const NUM_FRAMES: usize = 4;
/// Per-planet token width. See [`fill_token`] for the layout.
pub const TOKEN_DIM: usize = 11;
/// Board-level feature width. See [`compute_globals`] for the layout.
pub const GLOBAL_DIM: usize = 16;

/// Future-frame offsets (turns from `t`). The 4th frame, `t_resolved`, is found
/// dynamically (first turn with no fleets), so it isn't a fixed offset here.
const FRAME_T1: usize = 1;
const FRAME_T10: usize = 10;

/// Max flight turns the aim solver will consider. A fleet that can't reach a
/// target within this many turns is treated as unreachable (action invalid).
/// Bounds the projection cost; raise if very slow long-range fleets matter.
const AIM_HORIZON: usize = 64;
/// Cap on how far ahead we look for the "all fleets resolved" turn.
const RESOLVE_CAP: usize = 96;

const SEND_FRACTIONS: [f64; 4] = [0.25, 0.50, 0.75, 1.00];
/// Per-source noop action. It is valid only at target slot 0, giving each
/// source row one way to do nothing without duplicating noop across targets.
const NOOP_ACTION: usize = 0;
const CONST_SEND: i64 = 42;
const CONST_SEND_ACTION: usize = 5;
/// Index of the `target_resolved + 1` action.
const RESOLVED_ACTION: usize = 6;

// ---- normalization -------------------------------------------------------
/// `log1p(x) / log1p(full)`: 0 at x=0, ~1 at x=full. For non-negative inputs.
#[inline]
fn log_norm(x: f64, full: f64) -> f32 {
    (x.max(0.0).ln_1p() / full.ln_1p()) as f32
}
/// Signed version of [`log_norm`] for quantities that can go either way.
#[inline]
fn signed_log_norm(x: f64, full: f64) -> f32 {
    (x.signum() * (x.abs().ln_1p() / full.ln_1p())) as f32
}
#[inline]
fn norm_dist(d: f64) -> f32 {
    (d / 100.0) as f32
}
#[inline]
fn norm_turns(t: usize) -> f32 {
    (t as f64 / 20.0) as f32
}
#[inline]
fn norm_ships(s: i64) -> f32 {
    log_norm(s as f64, 1000.0)
}
#[inline]
fn norm_prod(p: i64) -> f32 {
    log_norm(p as f64, 10.0)
}
#[inline]
fn norm_count(c: usize) -> f32 {
    c as f32 / PLANET_SLOTS as f32
}

/// Geometry of one planet at one turn.
#[derive(Clone, Copy, Debug)]
struct Geom {
    x: f64,
    y: f64,
    r: f64,
}

/// A planet's swept segment over a single turn (`old -> new`), precomputed so
/// the projection inner loop is allocation- and hashmap-free.
#[derive(Clone, Copy, Debug)]
struct Seg {
    id: i64,
    ox: f64,
    oy: f64,
    nx: f64,
    ny: f64,
    r: f64,
}

/// Planet positions over a horizon, forward-simulated once with empty actions
/// and shared across frames (positions depend only on orbital / comet motion).
struct Trajectory {
    /// `snapshots[turn]` = planets after `turn` empty steps from `t`.
    snapshots: Vec<Vec<Planet>>,
    /// `comet_ids[turn]` = comet planet ids at that turn.
    comet_ids: Vec<FxHashSet<i64>>,
    /// `by_id[id][turn]` = that planet's geometry, or None if absent.
    by_id: FxHashMap<i64, Vec<Option<Geom>>>,
    /// `segments[turn]` = swept `old->new` segments of present planets, for the
    /// projection hot loop (no per-step hashmap lookups).
    segments: Vec<Vec<Seg>>,
    /// Turn offsets of the 4 frames: `[0, 1, 10, resolved]`.
    offsets: [usize; NUM_FRAMES],
    /// Game-start planet positions, for the `is_orbiting` token.
    initial_xy: FxHashMap<i64, (f64, f64)>,
    ship_speed: f64,
    /// Globally-stationary planets `(id, x, y, radius)` — a fleet line staying
    /// farther than `radius` from one can never hit it, so it's culled from the
    /// projection hot loop.
    stationary: Vec<(i64, f64, f64, f64)>,
    /// `max(planet id) + 1` — sizes the cull scratch.
    id_bound: usize,
}

impl Trajectory {
    fn build(state: &EngineState) -> Self {
        let np = state.num_players.max(1);
        let empty: Vec<Vec<MoveAction>> = vec![Vec::new(); np];
        let total = RESOLVE_CAP + AIM_HORIZON;

        let mut sim = state.clone();
        let mut snapshots: Vec<Vec<Planet>> = Vec::with_capacity(total + 1);
        let mut comet_ids: Vec<FxHashSet<i64>> = Vec::with_capacity(total + 1);
        let mut resolved: Option<usize> = if sim.fleets.is_empty() { Some(0) } else { None };

        snapshots.push(sim.planets.clone());
        comet_ids.push(sim.comet_planet_ids.iter().copied().collect());
        for k in 1..=total {
            let _ = sim.step_with_actions(&empty);
            snapshots.push(sim.planets.clone());
            comet_ids.push(sim.comet_planet_ids.iter().copied().collect());
            if resolved.is_none() && sim.fleets.is_empty() {
                resolved = Some(k);
            }
        }
        let resolved = resolved.unwrap_or(RESOLVE_CAP).min(total);
        let len = snapshots.len();

        // Per-id geometry timeline for O(1) target lookup.
        let mut by_id: FxHashMap<i64, Vec<Option<Geom>>> = FxHashMap::default();
        for (turn, planets) in snapshots.iter().enumerate() {
            for p in planets {
                let slot = by_id.entry(p.id).or_insert_with(|| vec![None; len]);
                slot[turn] = Some(Geom {
                    x: p.x,
                    y: p.y,
                    r: p.radius,
                });
            }
        }

        // Per-turn swept segments (old at `t`, new at `t+1`; a planet absent at
        // `t+1` is treated as stationary, matching engine comet expiry).
        let mut segments: Vec<Vec<Seg>> = Vec::with_capacity(len.saturating_sub(1));
        for t in 0..len.saturating_sub(1) {
            let mut segs = Vec::with_capacity(snapshots[t].len());
            for p in &snapshots[t] {
                let (nx, ny) = by_id
                    .get(&p.id)
                    .and_then(|v| v[t + 1])
                    .map(|g| (g.x, g.y))
                    .unwrap_or((p.x, p.y));
                segs.push(Seg {
                    id: p.id,
                    ox: p.x,
                    oy: p.y,
                    nx,
                    ny,
                    r: p.radius,
                });
            }
            segments.push(segs);
        }

        // Classify globally-stationary planets for the broad-phase cull.
        let mut stationary = Vec::new();
        let mut id_bound = 0usize;
        for (&id, tl) in &by_id {
            id_bound = id_bound.max(id as usize + 1);
            let mut iter = tl.iter().flatten();
            if let Some(first) = iter.next() {
                let moves =
                    iter.any(|g| (g.x - first.x).abs() > 1e-12 || (g.y - first.y).abs() > 1e-12);
                if !moves {
                    stationary.push((id, first.x, first.y, first.r));
                }
            }
        }

        let initial_xy = state
            .initial_planets
            .iter()
            .map(|p| (p.id, (p.x, p.y)))
            .collect();

        Self {
            snapshots,
            comet_ids,
            by_id,
            segments,
            offsets: [0, FRAME_T1, FRAME_T10, resolved],
            initial_xy,
            ship_speed: state.configuration.ship_speed,
            stationary,
            id_bound,
        }
    }

    fn frame_planets(&self, f: usize) -> &[Planet] {
        &self.snapshots[self.offsets[f]]
    }

    /// Project a straight-line fleet launched at `frame_off` and return the turn
    /// it cleanly arrives at `dst_id` (`Some`), or `None` if it first hits any
    /// other planet / the sun / the board edge, or never reaches `dst_id` within
    /// the horizon. Mirrors the engine's collision order (planets, bounds, sun).
    fn project(
        &self,
        frame_off: usize,
        launch: (f64, f64),
        speed: f64,
        theta: f64,
        dst_id: i64,
    ) -> Option<usize> {
        let (uhx, uhy) = (theta.cos(), theta.sin());
        let (vx, vy) = (uhx * speed, uhy * speed);

        // Broad-phase cull: a stationary planet farther than its radius from the
        // (infinite) fleet line can never be hit, so skip its per-turn test.
        // Squared distances + reused scratch, so the setup is near-free.
        CULL_KEEP.with(|cell| {
            let mut keep = cell.borrow_mut();
            keep.clear();
            keep.resize(self.id_bound, true);
            for &(id, cx, cy, r) in &self.stationary {
                let dx = cx - launch.0;
                let dy = cy - launch.1;
                let along = dx * uhx + dy * uhy;
                let perp2 = (dx * dx + dy * dy) - along * along;
                let rr = r + 1e-6;
                if perp2 > rr * rr {
                    keep[id as usize] = false;
                }
            }

            let mut pos = launch;
            for k in 0..AIM_HORIZON {
                let t_old = frame_off + k;
                if t_old >= self.segments.len() {
                    break;
                }
                let new = (pos.0 + vx, pos.1 + vy);
                for s in &self.segments[t_old] {
                    if !keep[s.id as usize] {
                        continue;
                    }
                    if swept_pair_hit(pos, new, (s.ox, s.oy), (s.nx, s.ny), s.r) {
                        return if s.id == dst_id { Some(k + 1) } else { None };
                    }
                }
                if !(0.0..=BOARD_SIZE).contains(&new.0) || !(0.0..=BOARD_SIZE).contains(&new.1) {
                    return None;
                }
                if point_to_segment_distance((CENTER, CENTER), pos, new) < SUN_RADIUS {
                    return None;
                }
                pos = new;
            }
            None
        })
    }

    /// Solve for a clean intercept of `dst_id` from `src_id` at `frame_off`
    /// sending `count` ships. Returns `(arrival_turn, launch_angle)` or `None`.
    /// Leads the moving target for a candidate angle, then verifies with `project`.
    fn aim(&self, frame_off: usize, src_id: i64, dst_id: i64, count: i64) -> Option<(usize, f64)> {
        let speed = fleet_speed(count, self.ship_speed);
        let dst_tl = self.by_id.get(&dst_id)?;
        let src = self.by_id.get(&src_id)?.get(frame_off).copied().flatten()?;
        for tau in 1..=AIM_HORIZON {
            // Past the board diagonal (~141) no target can satisfy the gate below.
            if speed * tau as f64 > 150.0 {
                break;
            }
            let Some(dst) = dst_tl.get(frame_off + tau).copied().flatten() else {
                continue;
            };
            let d = distance((src.x, src.y), (dst.x, dst.y));
            // Consistency gate: only attempt arrivals where flown ≈ target dist.
            // `project` is the source of truth.
            if (speed * tau as f64 - d).abs() <= dst.r + speed + 1.0 {
                let theta = (dst.y - src.y).atan2(dst.x - src.x);
                let launch = (
                    src.x + theta.cos() * (src.r + 0.1),
                    src.y + theta.sin() * (src.r + 0.1),
                );
                if let Some(turn) = self.project(frame_off, launch, speed, theta, dst_id) {
                    return Some((turn, theta));
                }
            }
        }
        None
    }
}

/// Ship count for a launch action `a` from a source with `src_ships`, against a target
/// whose post-resolution garrison is `resolved_ships` (owned by ally =
/// `resolved_ally`, absent = `resolved_absent`). `None` if the action is
/// structurally invalid (resolved+1 on an ally/absent target) or the count is
/// not physically sendable (`< 1` or `> src_ships`). The noop action is not a
/// launch and is handled directly by `encode`.
fn action_count(
    a: usize,
    src_ships: i64,
    resolved_ships: i64,
    resolved_ally: bool,
    resolved_absent: bool,
) -> Option<i64> {
    let c = match a {
        NOOP_ACTION => return None,
        1..=4 => (src_ships as f64 * SEND_FRACTIONS[a - 1]).floor() as i64,
        CONST_SEND_ACTION => CONST_SEND,
        RESOLVED_ACTION => {
            if resolved_absent || resolved_ally {
                return None;
            }
            resolved_ships + 1
        }
        _ => return None,
    };
    (c >= 1 && c <= src_ships).then_some(c)
}

/// Encoded features for one observation. Flat row-major buffers for cheap numpy
/// reshaping on the Python side.
#[derive(Clone, Debug)]
pub struct Features {
    /// Planet id per slot (length 44), `-1` for empty slots.
    pub slot_id: Vec<i64>,
    /// Number of real planets at frame `t`.
    pub n: usize,
    /// Turn offsets of the 4 frames `[0, 1, 10, resolved]`.
    pub offsets: [usize; NUM_FRAMES],
    pub tokens: Vec<f32>,        // (NUM_FRAMES, 44, TOKEN_DIM)
    pub globals: Vec<f32>,       // (GLOBAL_DIM,)
    pub presence: Vec<f32>,      // (NUM_FRAMES, 44)
    pub turns: Vec<f32>,         // (44, 44, ACTIONS_DIM), frame t
    pub angles: Vec<f32>,        // (44, 44, ACTIONS_DIM), frame t
    pub mask: Vec<u8>,           // (44, 44, ACTIONS_DIM), frame t
    pub ship_counts: Vec<i64>,   // (44, 44, ACTIONS_DIM), frame t
    pub reachable_mask: Vec<u8>, // (44, 44, ACTIONS_DIM), frame t
    /// Raw per-frame planet state `[id, owner, x, y, ships]`, present planets
    /// only. Not a model input — exposed for validation/debugging against the
    /// reference engine.
    pub frame_planets: Vec<Vec<(i64, i64, f64, f64, i64)>>,
}

impl Features {
    /// Serialize to a Python dict, consuming `self`. Numeric buffers are moved
    /// into numpy zero-copy (`torch.from_numpy`-ready); reshape with the paired
    /// `*_shape` tuple.
    pub fn into_py_dict(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        // Small metadata stays as plain Python objects.
        d.set_item("planet_ids", self.slot_id)?;
        d.set_item("num_planets", self.n)?;
        d.set_item("frame_offsets", self.offsets.to_vec())?;
        d.set_item("frame_planets", self.frame_planets)?;
        // Numeric tensors: zero-copy move into numpy (1-D + shape).
        d.set_item("tokens", self.tokens.into_pyarray(py))?;
        d.set_item("tokens_shape", (NUM_FRAMES, PLANET_SLOTS, TOKEN_DIM))?;
        d.set_item("globals", self.globals.into_pyarray(py))?;
        d.set_item("globals_shape", (GLOBAL_DIM,))?;
        d.set_item("presence", self.presence.into_pyarray(py))?;
        d.set_item("presence_shape", (NUM_FRAMES, PLANET_SLOTS))?;
        d.set_item("turns", self.turns.into_pyarray(py))?;
        d.set_item("turns_shape", (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM))?;
        d.set_item("angles", self.angles.into_pyarray(py))?;
        d.set_item("angles_shape", (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM))?;
        d.set_item("mask", self.mask.into_pyarray(py))?;
        d.set_item("mask_shape", (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM))?;
        d.set_item("ship_counts", self.ship_counts.into_pyarray(py))?;
        d.set_item(
            "ship_counts_shape",
            (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM),
        )?;
        d.set_item("reachable_mask", self.reachable_mask.into_pyarray(py))?;
        d.set_item(
            "reachable_mask_shape",
            (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM),
        )?;
        Ok(d.into_any().unbind())
    }
}

/// Per-planet token: `[is_mine, is_enemy, is_neutral, is_comet, is_orbiting,
/// production, ships, x, y, dist_to_sun, angular_velocity]`. Positions and
/// distance-to-sun are normalized by board size. `angular_velocity` is the
/// board's orbital rate for orbiting planets and 0 for static planets / comets,
/// scaled to ~[0, 1] (board rate is ~0.025–0.05).
fn fill_token(
    tk: &mut [f32],
    p: &Planet,
    player: i64,
    comet: &FxHashSet<i64>,
    initial: &FxHashMap<i64, (f64, f64)>,
    angular_velocity: f64,
) {
    let is_comet = comet.contains(&p.id);
    let is_orbiting = !is_comet
        && initial.get(&p.id).is_some_and(|&(ix, iy)| {
            let orbital_r = ((ix - CENTER).powi(2) + (iy - CENTER).powi(2)).sqrt();
            orbital_r + p.radius < ROTATION_RADIUS_LIMIT
        });
    tk[0] = (p.owner == player) as i32 as f32;
    tk[1] = (p.owner >= 0 && p.owner != player) as i32 as f32;
    tk[2] = (p.owner == -1) as i32 as f32;
    tk[3] = is_comet as i32 as f32;
    tk[4] = is_orbiting as i32 as f32;
    tk[5] = norm_prod(p.production);
    tk[6] = norm_ships(p.ships);
    tk[7] = norm_dist(p.x);
    tk[8] = norm_dist(p.y);
    tk[9] = norm_dist(distance((p.x, p.y), (CENTER, CENTER)));
    tk[10] = if is_orbiting {
        (angular_velocity / 0.05) as f32
    } else {
        0.0
    };
}

/// Board-level features from `player`'s view at the current turn. Layout:
/// `[remaining, own_score, enemy_score, neutral_ships, own_planets,
/// enemy_planets, neutral_planets, own_production, enemy_production,
/// own_fleet_ships, enemy_fleet_ships, score_share, production_share,
/// planet_share, score_diff, production_diff]`. `score = planet_ships +
/// fleet_ships`; enemy aggregates all non-self players; shares are own/(own+enemy)
/// (0.5 when both are 0). Counts are /44; ships/production are log-normalized;
/// diffs are signed-log.
fn compute_globals(state: &EngineState, player: i64) -> Vec<f32> {
    let np = state.num_players;
    let (mut own_planet_ships, mut enemy_planet_ships, mut neutral_ships) = (0i64, 0i64, 0i64);
    let (mut own_planets, mut enemy_planets, mut neutral_planets) = (0usize, 0usize, 0usize);
    let (mut own_production, mut enemy_production) = (0i64, 0i64);
    for p in &state.planets {
        if p.owner == player {
            own_planet_ships += p.ships;
            own_planets += 1;
            own_production += p.production;
        } else if p.owner >= 0 && (p.owner as usize) < np {
            enemy_planet_ships += p.ships;
            enemy_planets += 1;
            enemy_production += p.production;
        } else {
            neutral_ships += p.ships;
            neutral_planets += 1;
        }
    }
    let (mut own_fleet_ships, mut enemy_fleet_ships) = (0i64, 0i64);
    for f in &state.fleets {
        if f.owner == player {
            own_fleet_ships += f.ships;
        } else if f.owner >= 0 && (f.owner as usize) < np {
            enemy_fleet_ships += f.ships;
        }
    }
    let own_score = own_planet_ships + own_fleet_ships;
    let enemy_score = enemy_planet_ships + enemy_fleet_ships;
    let episode = state.configuration.episode_steps.max(1) as f64;
    let remaining = ((episode - state.step as f64).max(0.0)) / episode;

    let share = |a: i64, b: i64| -> f32 {
        if a + b > 0 {
            a as f32 / (a + b) as f32
        } else {
            0.5
        }
    };

    vec![
        remaining as f32,
        norm_ships(own_score),
        norm_ships(enemy_score),
        norm_ships(neutral_ships),
        norm_count(own_planets),
        norm_count(enemy_planets),
        norm_count(neutral_planets),
        log_norm(own_production as f64, 100.0),
        log_norm(enemy_production as f64, 100.0),
        norm_ships(own_fleet_ships),
        norm_ships(enemy_fleet_ships),
        share(own_score, enemy_score),
        share(own_production, enemy_production),
        share(own_planets as i64, enemy_planets as i64),
        signed_log_norm((own_score - enemy_score) as f64, 1000.0),
        signed_log_norm((own_production - enemy_production) as f64, 100.0),
    ]
}

/// Compute the `(44, ACTIONS_DIM)` turns row for one source slot `si` at frame
/// `f` (offset `off`), writing into `t_row`. For frame t (`extra` = Some), also
/// fills that source's angles, legal-action mask, ship-count, and reachability rows. Pure / read-only
/// over the trajectory, so it's safe to run across sources in parallel.
#[allow(clippy::too_many_arguments)]
fn compute_source_row(
    traj: &Trajectory,
    off: usize,
    si: usize,
    slot_id: &[i64],
    by: &FxHashMap<i64, &Planet>,
    resolved: &FxHashMap<i64, (i64, i64)>,
    player: i64,
    t_row: &mut [f32],
    mut extra: Option<(&mut [f32], &mut [u8], &mut [i64], &mut [u8])>,
) {
    let id_i = slot_id[si];
    if id_i < 0 {
        return;
    }
    let Some(pi) = by.get(&id_i) else { return };
    for sj in 0..PLANET_SLOTS {
        if si == sj {
            continue;
        }
        let id_j = slot_id[sj];
        if id_j < 0 || !by.contains_key(&id_j) {
            continue;
        }
        let (rj_owner, rj_ships) = resolved.get(&id_j).copied().unwrap_or((-2, 0));
        // Different actions that send the same count share one aim solve.
        let mut memo: [(i64, Option<(usize, f64)>); ACTIONS_DIM] = [(i64::MIN, None); ACTIONS_DIM];
        let mut memo_len = 0usize;
        for a in 0..ACTIONS_DIM {
            let Some(count) =
                action_count(a, pi.ships, rj_ships, rj_owner == player, rj_owner == -2)
            else {
                continue;
            };
            if let Some((_, _, c_row, _)) = extra.as_mut() {
                c_row[sj * ACTIONS_DIM + a] = count;
            }
            let res = match memo[..memo_len].iter().find(|(c, _)| *c == count) {
                Some(&(_, r)) => r,
                None => {
                    let r = traj.aim(off, id_i, id_j, count);
                    memo[memo_len] = (count, r);
                    memo_len += 1;
                    r
                }
            };
            if let Some((turn, theta)) = res {
                let k = sj * ACTIONS_DIM + a;
                t_row[k] = norm_turns(turn);
                if let Some((a_row, m_row, _, r_row)) = extra.as_mut() {
                    a_row[k] = theta as f32;
                    r_row[k] = 1;
                    if pi.owner == player {
                        m_row[k] = 1;
                    }
                }
            }
        }
    }
}

/// Encode an [`EngineState`] into model features from `player`'s perspective.
pub fn encode(state: &EngineState, player: i64) -> Features {
    let traj = Trajectory::build(state);

    // Slot assignment is fixed by frame t's planet order (first 44).
    let base = traj.frame_planets(0);
    let n = base.len().min(PLANET_SLOTS);
    let mut slot_id = vec![-1i64; PLANET_SLOTS];
    for (s, p) in base.iter().take(PLANET_SLOTS).enumerate() {
        slot_id[s] = p.id;
    }

    // Resolved garrisons (for the resolved+1 action) from the last frame.
    let resolved: FxHashMap<i64, (i64, i64)> = traj
        .frame_planets(NUM_FRAMES - 1)
        .iter()
        .map(|p| (p.id, (p.owner, p.ships)))
        .collect();

    let globals = compute_globals(state, player);

    let mut tokens = vec![0f32; NUM_FRAMES * PLANET_SLOTS * TOKEN_DIM];
    let mut presence = vec![0f32; NUM_FRAMES * PLANET_SLOTS];
    // Decision-frame action tensors; lookahead frames give temporal context
    // through their tokens/presence.
    let mut turns = vec![0f32; PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM];
    let mut angles = vec![0f32; PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM];
    let mut mask = vec![0u8; PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM];
    let mut ship_counts = vec![0i64; PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM];
    let mut reachable_mask = vec![0u8; PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM];
    for si in 0..PLANET_SLOTS {
        mask[(si * PLANET_SLOTS) * ACTIONS_DIM + NOOP_ACTION] = 1;
    }
    let mut frame_planets: Vec<Vec<(i64, i64, f64, f64, i64)>> = Vec::with_capacity(NUM_FRAMES);

    for f in 0..NUM_FRAMES {
        let off = traj.offsets[f];
        let fp = traj.frame_planets(f);
        let by: FxHashMap<i64, &Planet> = fp.iter().map(|p| (p.id, p)).collect();
        let comet = &traj.comet_ids[off];
        frame_planets.push(
            fp.iter()
                .map(|p| (p.id, p.owner, p.x, p.y, p.ships))
                .collect(),
        );

        // Tokens + presence (all frames — this is the temporal context).
        for si in 0..PLANET_SLOTS {
            let id = slot_id[si];
            if id < 0 {
                continue;
            }
            if let Some(p) = by.get(&id) {
                presence[f * PLANET_SLOTS + si] = 1.0;
                let base = (f * PLANET_SLOTS + si) * TOKEN_DIM;
                fill_token(
                    &mut tokens[base..base + TOKEN_DIM],
                    p,
                    player,
                    comet,
                    &traj.initial_xy,
                    state.angular_velocity,
                );
            }
        }

        // Turns + action metadata: frame t only.
        if f != 0 {
            continue;
        }
        const ROW: usize = PLANET_SLOTS * ACTIONS_DIM;
        for si in 0..PLANET_SLOTS {
            let t_row = &mut turns[si * ROW..(si + 1) * ROW];
            let a_row = &mut angles[si * ROW..(si + 1) * ROW];
            let m_row = &mut mask[si * ROW..(si + 1) * ROW];
            let c_row = &mut ship_counts[si * ROW..(si + 1) * ROW];
            let r_row = &mut reachable_mask[si * ROW..(si + 1) * ROW];
            compute_source_row(
                &traj,
                off,
                si,
                &slot_id,
                &by,
                &resolved,
                player,
                t_row,
                Some((a_row, m_row, c_row, r_row)),
            );
        }
    }

    Features {
        slot_id,
        n,
        offsets: traj.offsets,
        tokens,
        globals,
        presence,
        turns,
        angles,
        mask,
        ship_counts,
        reachable_mask,
        frame_planets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Configuration;

    fn planet(id: i64, owner: i64, x: f64, y: f64, ships: i64) -> Planet {
        Planet {
            id,
            owner,
            x,
            y,
            radius: 1.5,
            ships,
            production: 0,
        }
    }

    fn state(planets: Vec<Planet>, fleets: Vec<crate::Fleet>) -> EngineState {
        let initial = planets.clone();
        EngineState::new(
            0,
            0.02,
            planets,
            initial,
            fleets,
            1000,
            Vec::new(),
            Vec::new(),
            2,
            Configuration::default(),
        )
    }

    struct Lcg(u64);
    impl Lcg {
        fn new(s: u64) -> Self {
            Lcg(s)
        }
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / ((1u64 << 53) as f64)
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// A random board well clear of the sun, ships, mixed ownership.
    fn random_state(rng: &mut Lcg) -> EngineState {
        let np = if rng.below(2) == 0 { 2 } else { 4 };
        let n = 3 + rng.below(8);
        let mut planets = Vec::new();
        for id in 0..n as i64 {
            // keep planets away from the central sun (radius 10 at center)
            let (mut x, mut y);
            loop {
                x = rng.range(8.0, 92.0);
                y = rng.range(8.0, 92.0);
                if distance((x, y), (CENTER, CENTER)) > 16.0 {
                    break;
                }
            }
            // Force planets 0 and 1 to players 0 and 1 so >= 2 players are
            // always alive. With no fleets + empty actions, ownership never
            // changes, so the game never hits `done` and planets orbit for the
            // whole horizon (no frozen-trajectory artifact).
            let owner = match id {
                0 => 0,
                1 => 1,
                _ => (rng.below(np + 1) as i64) - 1, // -1..np-1
            };
            planets.push(planet(id, owner, x, y, 1 + rng.below(200) as i64));
        }
        let mut st = state(planets, Vec::new());
        st.num_players = np;
        st
    }

    /// Replay a single launch through the *real* engine and report which planet
    /// it hit and on which turn — the ground truth for the aim solver.
    ///
    /// We inject the fleet directly at the engine's launch position (rather than
    /// via `process_moves`) so we don't mutate ownership or deduct ships, and we
    /// step with empty actions — exactly the trajectory `Trajectory::build`
    /// assumes. Production is zeroed so the only ship-count change comes from
    /// our fleet landing, which makes the hit planet unambiguous. (Assumes no
    /// pre-existing fleets, which holds for `random_state`.)
    fn engine_arrival(
        st: &EngineState,
        src_id: i64,
        angle: f64,
        count: i64,
    ) -> (Option<i64>, usize) {
        let mut s = st.clone();
        for p in &mut s.planets {
            p.production = 0;
        }
        let np = s.num_players.max(1);
        let empty: Vec<Vec<MoveAction>> = vec![Vec::new(); np];
        let src = s.planets.iter().find(|p| p.id == src_id).unwrap();
        let launch = (
            src.x + angle.cos() * (src.radius + 0.1),
            src.y + angle.sin() * (src.radius + 0.1),
        );
        let fleet_id = s.next_fleet_id;
        s.fleets.push(crate::Fleet {
            id: fleet_id,
            owner: 0, // player 0 is always alive in random_state, so no `done` shift
            x: launch.0,
            y: launch.1,
            angle,
            from_planet_id: src_id,
            ships: count,
        });
        s.next_fleet_id += 1;

        for turn in 1..=AIM_HORIZON {
            let before: FxHashMap<i64, (i64, i64)> = s
                .planets
                .iter()
                .map(|p| (p.id, (p.owner, p.ships)))
                .collect();
            let _ = s.step_with_actions(&empty);
            if !s.fleets.iter().any(|fl| fl.id == fleet_id) {
                for p in &s.planets {
                    if let Some(&(ow, sh)) = before.get(&p.id) {
                        if p.owner != ow || p.ships != sh {
                            return (Some(p.id), turn);
                        }
                    }
                }
                return (None, turn); // left board or hit the sun
            }
        }
        (None, 0)
    }

    // ---- the core correctness test ------------------------------------------

    #[test]
    fn aim_matches_engine_exactly() {
        let mut rng = Lcg::new(0xA1A1);
        let mut checked = 0u32;
        for _ in 0..120 {
            let st = random_state(&mut rng);
            let traj = Trajectory::build(&st);
            let ids: Vec<i64> = st.planets.iter().map(|p| p.id).collect();
            for &i in &ids {
                for &j in &ids {
                    if i == j {
                        continue;
                    }
                    let src_ships = st.planets.iter().find(|p| p.id == i).unwrap().ships;
                    for count in [1, 5, 42, (src_ships / 2).max(1), src_ships] {
                        if count < 1 || count > src_ships {
                            continue;
                        }
                        if let Some((turn, theta)) = traj.aim(0, i, j, count) {
                            let (hit, eturn) = engine_arrival(&st, i, theta, count);
                            assert_eq!(
                                hit,
                                Some(j),
                                "aim said {i}->{j} count {count} clean; engine hit {hit:?}"
                            );
                            assert_eq!(
                                eturn, turn,
                                "arrival turn mismatch for {i}->{j} count {count}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(
            checked > 200,
            "too few valid intercepts exercised: {checked}"
        );
    }

    /// Whatever the mask marks valid must (a) be ally-owned at t and (b) replay
    /// cleanly to the intended target at the predicted turn in the real engine.
    #[test]
    fn mask_is_sound() {
        let mut rng = Lcg::new(0x77AA);
        let mut checked = 0u32;
        for _ in 0..60 {
            let st = random_state(&mut rng);
            let f = encode(&st, 0);
            for si in 0..PLANET_SLOTS {
                for sj in 0..PLANET_SLOTS {
                    for a in 0..ACTIONS_DIM {
                        let mi = (si * PLANET_SLOTS + sj) * ACTIONS_DIM + a;
                        if f.mask[mi] == 0 {
                            continue;
                        }
                        if a == NOOP_ACTION {
                            assert_eq!(
                                sj, 0,
                                "noop should only be valid at canonical target slot 0"
                            );
                            continue;
                        }
                        let id_i = f.slot_id[si];
                        let id_j = f.slot_id[sj];
                        let pi = st.planets.iter().find(|p| p.id == id_i).unwrap();
                        assert_eq!(pi.owner, 0, "mask allowed a non-owned source");
                        // recompute count the same way encode did
                        let resolved: FxHashMap<i64, (i64, i64)> = Trajectory::build(&st)
                            .frame_planets(NUM_FRAMES - 1)
                            .iter()
                            .map(|p| (p.id, (p.owner, p.ships)))
                            .collect();
                        let (ro, rs) = resolved.get(&id_j).copied().unwrap_or((-2, 0));
                        let count = action_count(a, pi.ships, rs, ro == 0, ro == -2).unwrap();
                        assert_eq!(
                            f.ship_counts[mi], count,
                            "ship_counts disagreed with action_count"
                        );
                        assert_eq!(
                            f.reachable_mask[mi], 1,
                            "valid launch missing reachable bit"
                        );
                        let angle = f.angles[mi] as f64;
                        let (hit, turn) = engine_arrival(&st, id_i, angle, count);
                        assert_eq!(hit, Some(id_j), "masked-valid action didn't reach target");
                        let expect = (f.turns[(si * PLANET_SLOTS + sj) * ACTIONS_DIM + a] * 20.0)
                            .round() as usize;
                        assert_eq!(turn, expect, "masked-valid action arrival turn mismatch");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 50, "too few masked actions exercised: {checked}");
    }

    #[test]
    fn frame_caches_match_forward_sim() {
        // The trajectory snapshots ARE forward sims, so this checks indexing:
        // frame t+1 / t+10 / resolved planet states equal stepping `state`
        // with empty actions the right number of times.
        let mut rng = Lcg::new(0x3C3C);
        for _ in 0..40 {
            let st = random_state(&mut rng);
            let traj = Trajectory::build(&st);
            let np = st.num_players.max(1);
            let empty: Vec<Vec<MoveAction>> = vec![Vec::new(); np];
            for f in 0..NUM_FRAMES {
                let mut s = st.clone();
                for _ in 0..traj.offsets[f] {
                    let _ = s.step_with_actions(&empty);
                }
                let got = traj.frame_planets(f);
                assert_eq!(got.len(), s.planets.len(), "frame {f} planet count");
                for (a, b) in got.iter().zip(s.planets.iter()) {
                    assert_eq!(a.id, b.id);
                    assert_eq!(a.owner, b.owner);
                    assert_eq!(a.ships, b.ships);
                    assert!((a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9);
                }
            }
            // resolved frame really has no fleets in flight (unless capped).
            if traj.offsets[NUM_FRAMES - 1] < RESOLVE_CAP {
                let mut s = st.clone();
                for _ in 0..traj.offsets[NUM_FRAMES - 1] {
                    let _ = s.step_with_actions(&empty);
                }
                assert!(s.fleets.is_empty(), "resolved frame still has fleets");
            }
        }
    }

    #[test]
    fn shapes_and_finiteness() {
        let mut rng = Lcg::new(0xF1A7);
        for _ in 0..40 {
            let st = random_state(&mut rng);
            let f = encode(&st, 0);
            assert_eq!(f.tokens.len(), NUM_FRAMES * PLANET_SLOTS * TOKEN_DIM);
            assert_eq!(f.globals.len(), GLOBAL_DIM);
            assert_eq!(f.turns.len(), PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM);
            assert_eq!(f.mask.len(), PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM);
            assert_eq!(f.angles.len(), PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM);
            assert_eq!(
                f.ship_counts.len(),
                PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM
            );
            assert_eq!(
                f.reachable_mask.len(),
                PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM
            );
            for v in f
                .tokens
                .iter()
                .chain(f.turns.iter())
                .chain(f.angles.iter())
                .chain(f.globals.iter())
            {
                assert!(v.is_finite(), "non-finite feature");
            }
            // Shares live in [0, 1]; remaining in [0, 1] at the game start.
            for &idx in &[0usize, 11, 12, 13] {
                assert!(
                    (0.0..=1.0).contains(&f.globals[idx]),
                    "global {idx} out of [0,1]: {}",
                    f.globals[idx]
                );
            }
            for &v in &f.turns {
                assert!(v >= 0.0, "negative turns");
            }
            for token in f.tokens.chunks_exact(TOKEN_DIM) {
                if token[0..5].iter().any(|&x| x > 0.0) {
                    assert!(
                        (0.0..=1.0).contains(&token[7]),
                        "x out of [0,1]: {}",
                        token[7]
                    );
                    assert!(
                        (0.0..=1.0).contains(&token[8]),
                        "y out of [0,1]: {}",
                        token[8]
                    );
                }
            }
            // Every source row has one canonical noop; launch masks imply a positive turns entry.
            for si in 0..PLANET_SLOTS {
                assert_eq!(f.mask[(si * PLANET_SLOTS) * ACTIONS_DIM + NOOP_ACTION], 1);
            }
            for si in 0..PLANET_SLOTS {
                for sj in 0..PLANET_SLOTS {
                    for a in 0..ACTIONS_DIM {
                        let mi = (si * PLANET_SLOTS + sj) * ACTIONS_DIM + a;
                        if f.mask[mi] == 1 && a != NOOP_ACTION {
                            let di = ((si * PLANET_SLOTS) + sj) * ACTIONS_DIM + a;
                            assert!(f.turns[di] > 0.0, "masked launch action with zero turns");
                            assert_eq!(
                                f.reachable_mask[di], 1,
                                "masked launch action not reachable"
                            );
                            assert!(
                                f.ship_counts[di] > 0,
                                "masked launch action with zero ship count"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deterministic() {
        let mut rng = Lcg::new(0xD37);
        for _ in 0..20 {
            let st = random_state(&mut rng);
            let a = encode(&st, 0);
            let b = encode(&st, 0);
            assert_eq!(a.turns, b.turns);
            assert_eq!(a.mask, b.mask);
            assert_eq!(a.tokens, b.tokens);
            assert_eq!(a.globals, b.globals);
        }
    }
}
