//! Model input features for Orbit Wars.
//!
//! Single source of truth for turning an [`EngineState`] into the tensors a
//! policy/value network consumes. The bot calls [`encode`] natively on its
//! `env_model` state; training calls it via the `encode_obs` pyo3 wrapper on
//! `env_engine` observations. One implementation, two callers — no skew.
//!
//! # Shape
//!
//! `NUM_FRAMES = 4` snapshots at turns `t`, `t+1`, `t+10`, `t_resolved` (the
//! first future turn with no fleets in flight). Each frame carries
//! `PLANET_SLOTS = 44` planet tokens. The action space is
//! `(44, 44, ACTIONS_DIM)` = `(source, target, action)`, with
//! `ACTIONS_DIM = 6` actions: send `{25%, 50%, 75%, 100%}` of the source's
//! current ships, a constant `42`, or `target_resolved + 1` (min ships to take
//! the target once all in-flight fleets resolve; invalid if the target is
//! ally-held at resolution).
//!
//! Outputs:
//!   - `tokens`  `(NUM_FRAMES, 44, TOKEN_DIM)`     — per-planet features (all frames = temporal context)
//!   - `presence``(NUM_FRAMES, 44)`                — 1 if that slot's planet exists at the frame
//!   - `turns`   `(44, 44, 6)`                     — normalized turns-to-arrive at frame t (0 if invalid)
//!   - `angles`  `(44, 44, 6)`                     — launch angle (radians) at frame t, to issue the move
//!   - `mask`    `(44, 44, 6)`                      — 1 if the action is legal *now* (frame t)
//!
//! `turns`/`angles`/`mask` are the decision frame (t) only: the aim solve is the
//! whole cost and the policy acts now, so it isn't run for the lookahead frames.
//!
//! # Aim solver
//!
//! `turns`/`mask`/`angles` come from one geometric solve per `(frame, i, j,
//! count)`: lead the (moving) target to pick a launch angle, then *project* the
//! straight-line fleet turn-by-turn against every planet's swept path, the sun,
//! and the board, reusing env_model's own collision code so validity is
//! bit-identical to what the engine accepts. Planet trajectories are
//! precomputed once (forward-sim with empty actions) and shared across frames.

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
pub const ACTIONS_DIM: usize = 6;
pub const NUM_FRAMES: usize = 4;
pub const TOKEN_DIM: usize = 8;

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
const CONST_SEND: i64 = 42;
/// Index of the `target_resolved + 1` action.
const RESOLVED_ACTION: usize = 5;

// ---- normalization -------------------------------------------------------
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
    // log1p, scaled so 1000 ships -> ~1.0.
    ((s.max(0) as f64).ln_1p() / 1000.0_f64.ln_1p()) as f32
}
#[inline]
fn norm_prod(p: i64) -> f32 {
    ((p.max(0) as f64).ln_1p() / 10.0_f64.ln_1p()) as f32
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
/// and shared across all frames. Positions depend only on orbital motion /
/// comet paths, so they're independent of which fleets exist or who owns what.
struct Trajectory {
    /// `snapshots[turn]` = planets (full state) after `turn` empty steps from `t`.
    snapshots: Vec<Vec<Planet>>,
    /// `comet_ids[turn]` = set of comet planet ids at that turn.
    comet_ids: Vec<FxHashSet<i64>>,
    /// `by_id[id][turn]` = that planet's geometry, or None if absent.
    by_id: FxHashMap<i64, Vec<Option<Geom>>>,
    /// `segments[turn]` = swept `old->new` segments of every planet present at
    /// `turn`, for the projection hot loop (no per-step hashmap lookups).
    segments: Vec<Vec<Seg>>,
    /// Turn offsets of the 4 frames: `[0, 1, 10, resolved]`.
    offsets: [usize; NUM_FRAMES],
    /// Initial (t=0 game start) planet positions, for the `is_orbiting` token.
    initial_xy: FxHashMap<i64, (f64, f64)>,
    ship_speed: f64,
    /// Planets whose position never changes over the horizon, as
    /// `(id, x, y, radius)`. These admit a cheap once-per-projection broad-phase
    /// cull (a fleet ray that stays farther than `radius` from the planet can
    /// never hit it), so far stationary planets are skipped in the hot loop.
    stationary: Vec<(i64, f64, f64, f64)>,
    /// `max(planet id) + 1` over the whole trajectory — sizes the cull scratch.
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
                slot[turn] = Some(Geom { x: p.x, y: p.y, r: p.radius });
            }
        }

        // Precompute per-turn swept segments (old at `t`, new at `t+1`; a planet
        // absent at `t+1` is treated as stationary, matching engine comet
        // expiry). All hashmap lookups happen here, once — not in the hot loop.
        let mut segments: Vec<Vec<Seg>> = Vec::with_capacity(len.saturating_sub(1));
        for t in 0..len.saturating_sub(1) {
            let mut segs = Vec::with_capacity(snapshots[t].len());
            for p in &snapshots[t] {
                let (nx, ny) = by_id
                    .get(&p.id)
                    .and_then(|v| v[t + 1])
                    .map(|g| (g.x, g.y))
                    .unwrap_or((p.x, p.y));
                segs.push(Seg { id: p.id, ox: p.x, oy: p.y, nx, ny, r: p.radius });
            }
            segments.push(segs);
        }

        // Classify globally-stationary planets (constant position across the
        // horizon) for the broad-phase cull, and find the id bound.
        let mut stationary = Vec::new();
        let mut id_bound = 0usize;
        for (&id, tl) in &by_id {
            id_bound = id_bound.max(id as usize + 1);
            let mut iter = tl.iter().flatten();
            if let Some(first) = iter.next() {
                let moves = iter.any(|g| (g.x - first.x).abs() > 1e-12 || (g.y - first.y).abs() > 1e-12);
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

    /// Project a straight-line fleet (fixed speed, fixed angle) launched at
    /// `frame_off` and return the turn it first collides — but only as `Some` if
    /// the first thing it hits is `dst_id` (a clean arrival). Returns `None` if
    /// it hits any other planet / the sun / the board edge first, or never
    /// reaches `dst_id` within the horizon. Mirrors env_model's per-step
    /// collision order (planets, then bounds, then sun).
    fn project(&self, frame_off: usize, launch: (f64, f64), speed: f64, theta: f64, dst_id: i64) -> Option<usize> {
        let (uhx, uhy) = (theta.cos(), theta.sin());
        let (vx, vy) = (uhx * speed, uhy * speed);

        // Broad-phase cull: a stationary planet whose center is farther than its
        // radius from the (infinite) fleet line can never be hit by any segment
        // on that line, so skip its per-turn swept test. Exact — culled planets
        // provably never collide — so the vector-order first-hit is unchanged.
        // Uses squared distances (no sqrt) and a reused scratch (no per-call
        // alloc), so the cull setup is near-free.
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

    /// Solve for a clean intercept of planet `dst_id` from planet `src_id` at
    /// `frame_off`, sending `count` ships. Returns `(arrival_turn, launch_angle)`
    /// or `None` if no clean trajectory exists. Leads the moving target to pick a
    /// candidate angle, then verifies with `project`.
    fn aim(&self, frame_off: usize, src_id: i64, dst_id: i64, count: i64) -> Option<(usize, f64)> {
        let speed = fleet_speed(count, self.ship_speed);
        // Hoist the timeline lookups out of the per-tau loop.
        let dst_tl = self.by_id.get(&dst_id)?;
        let src = self.by_id.get(&src_id)?.get(frame_off).copied().flatten()?;
        for tau in 1..=AIM_HORIZON {
            // Nothing on a 100x100 board is farther than the diagonal (~141);
            // once the fleet would have flown well past that, no target can
            // satisfy the consistency gate, so stop scanning.
            if speed * tau as f64 > 150.0 {
                break;
            }
            let Some(dst) = dst_tl.get(frame_off + tau).copied().flatten() else {
                continue;
            };
            let d = distance((src.x, src.y), (dst.x, dst.y));
            // Consistency gate: the fleet travels `speed*tau` in `tau` turns, so
            // only attempt arrivals where that roughly equals the (moving)
            // target distance. `project` is the source of truth.
            if (speed * tau as f64 - d).abs() <= dst.r + speed + 1.0 {
                let theta = (dst.y - src.y).atan2(dst.x - src.x);
                let launch = (src.x + theta.cos() * (src.r + 0.1), src.y + theta.sin() * (src.r + 0.1));
                if let Some(turn) = self.project(frame_off, launch, speed, theta, dst_id) {
                    return Some((turn, theta));
                }
            }
        }
        None
    }
}

/// Ship count for action `a` from a source with `src_ships`, against a target
/// whose post-resolution garrison is `resolved_ships` (owned by ally =
/// `resolved_ally`, absent = `resolved_absent`). `None` if the action is
/// structurally invalid (resolved+1 on an ally/absent target) or the count is
/// not physically sendable (`< 1` or `> src_ships`).
fn action_count(
    a: usize,
    src_ships: i64,
    resolved_ships: i64,
    resolved_ally: bool,
    resolved_absent: bool,
) -> Option<i64> {
    let c = match a {
        0..=3 => (src_ships as f64 * SEND_FRACTIONS[a]).floor() as i64,
        4 => CONST_SEND,
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
    pub tokens: Vec<f32>,    // (NUM_FRAMES, 44, TOKEN_DIM)
    pub presence: Vec<f32>,  // (NUM_FRAMES, 44)
    pub turns: Vec<f32>,     // (44, 44, ACTIONS_DIM), frame t
    pub angles: Vec<f32>,    // (44, 44, ACTIONS_DIM), frame t
    pub mask: Vec<u8>,       // (44, 44, ACTIONS_DIM), frame t
    /// Raw per-frame planet state `[id, owner, x, y, ships]`, present planets
    /// only. Not a model input — exposed for validation/debugging against the
    /// reference engine.
    pub frame_planets: Vec<Vec<(i64, i64, f64, f64, i64)>>,
}

impl Features {
    /// Serialize to a Python dict, consuming `self`. The big numeric buffers
    /// (`tokens`/`presence`/`turns`/`angles`/`mask`) are returned as **1-D numpy
    /// arrays via a zero-copy move** — numpy takes ownership of the Rust `Vec`'s
    /// allocation, so there's no copy here and `torch.from_numpy` is a view on
    /// the Python side. Reshape with the accompanying `*_shape` tuple (a numpy
    /// reshape of a contiguous array is also a zero-copy view).
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
        d.set_item("presence", self.presence.into_pyarray(py))?;
        d.set_item("presence_shape", (NUM_FRAMES, PLANET_SLOTS))?;
        d.set_item("turns", self.turns.into_pyarray(py))?;
        d.set_item("turns_shape", (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM))?;
        d.set_item("angles", self.angles.into_pyarray(py))?;
        d.set_item("angles_shape", (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM))?;
        d.set_item("mask", self.mask.into_pyarray(py))?;
        d.set_item("mask_shape", (PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM))?;
        Ok(d.into_any().unbind())
    }
}

fn fill_token(tk: &mut [f32], p: &Planet, player: i64, comet: &FxHashSet<i64>, initial: &FxHashMap<i64, (f64, f64)>) {
    let is_comet = comet.contains(&p.id);
    tk[0] = (p.owner == player) as i32 as f32; // is_mine
    tk[1] = (p.owner >= 0 && p.owner != player) as i32 as f32; // is_enemy
    tk[2] = (p.owner == -1) as i32 as f32; // is_neutral
    tk[3] = is_comet as i32 as f32; // is_comet
    tk[4] = if !is_comet {
        match initial.get(&p.id) {
            Some(&(ix, iy)) => {
                let orbital_r = ((ix - CENTER).powi(2) + (iy - CENTER).powi(2)).sqrt();
                (orbital_r + p.radius < ROTATION_RADIUS_LIMIT) as i32 as f32
            }
            None => 0.0,
        }
    } else {
        0.0
    }; // is_orbiting
    tk[5] = norm_prod(p.production);
    tk[6] = norm_ships(p.ships);
    tk[7] = norm_dist(distance((p.x, p.y), (CENTER, CENTER)));
}

/// Compute the `(44, ACTIONS_DIM)` turns row for one source slot `si` at frame
/// `f` (offset `off`), writing into `t_row`. For frame t (`extra` = Some), also
/// fills that source's angles row and legal-action mask row. Pure / read-only
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
    mut extra: Option<(&mut [f32], &mut [u8])>,
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
            let Some(count) = action_count(a, pi.ships, rj_ships, rj_owner == player, rj_owner == -2)
            else {
                continue;
            };
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
                if let Some((a_row, m_row)) = extra.as_mut() {
                    a_row[k] = theta as f32;
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

    let mut tokens = vec![0f32; NUM_FRAMES * PLANET_SLOTS * TOKEN_DIM];
    let mut presence = vec![0f32; NUM_FRAMES * PLANET_SLOTS];
    // `turns`/`angles`/`mask` are for the *decision* frame (t) only — the policy
    // acts now, and the aim solve is the whole cost, so computing it for the
    // lookahead frames isn't worth ~4x the work. The lookahead frames still
    // provide temporal context through their tokens/presence.
    let mut turns = vec![0f32; PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM];
    let mut angles = vec![0f32; PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM];
    let mut mask = vec![0u8; PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM];
    let mut frame_planets: Vec<Vec<(i64, i64, f64, f64, i64)>> = Vec::with_capacity(NUM_FRAMES);

    for f in 0..NUM_FRAMES {
        let off = traj.offsets[f];
        let fp = traj.frame_planets(f);
        let by: FxHashMap<i64, &Planet> = fp.iter().map(|p| (p.id, p)).collect();
        let comet = &traj.comet_ids[off];
        frame_planets.push(fp.iter().map(|p| (p.id, p.owner, p.x, p.y, p.ships)).collect());

        // Tokens + presence (all frames — this is the temporal context).
        for si in 0..PLANET_SLOTS {
            let id = slot_id[si];
            if id < 0 {
                continue;
            }
            if let Some(p) = by.get(&id) {
                presence[f * PLANET_SLOTS + si] = 1.0;
                let base = (f * PLANET_SLOTS + si) * TOKEN_DIM;
                fill_token(&mut tokens[base..base + TOKEN_DIM], p, player, comet, &traj.initial_xy);
            }
        }

        // Turns + angles + mask: frame t only.
        if f != 0 {
            continue;
        }
        const ROW: usize = PLANET_SLOTS * ACTIONS_DIM;
        for si in 0..PLANET_SLOTS {
            let t_row = &mut turns[si * ROW..(si + 1) * ROW];
            let (a_row, m_row) = (&mut angles[si * ROW..(si + 1) * ROW], &mut mask[si * ROW..(si + 1) * ROW]);
            compute_source_row(&traj, off, si, &slot_id, &by, &resolved, player, t_row, Some((a_row, m_row)));
        }
    }

    Features { slot_id, n, offsets: traj.offsets, tokens, presence, turns, angles, mask, frame_planets }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Configuration;

    fn planet(id: i64, owner: i64, x: f64, y: f64, ships: i64) -> Planet {
        Planet { id, owner, x, y, radius: 1.5, ships, production: 0 }
    }

    fn state(planets: Vec<Planet>, fleets: Vec<crate::Fleet>) -> EngineState {
        let initial = planets.clone();
        EngineState::new(0, 0.02, planets, initial, fleets, 1000, Vec::new(), Vec::new(), 2, Configuration::default())
    }

    struct Lcg(u64);
    impl Lcg {
        fn new(s: u64) -> Self { Lcg(s) }
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / ((1u64 << 53) as f64) }
        fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.unit() }
        fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
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
    fn engine_arrival(st: &EngineState, src_id: i64, angle: f64, count: i64) -> (Option<i64>, usize) {
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
            let before: FxHashMap<i64, (i64, i64)> =
                s.planets.iter().map(|p| (p.id, (p.owner, p.ships))).collect();
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
                            assert_eq!(hit, Some(j), "aim said {i}->{j} count {count} clean; engine hit {hit:?}");
                            assert_eq!(eturn, turn, "arrival turn mismatch for {i}->{j} count {count}");
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 200, "too few valid intercepts exercised: {checked}");
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
                        let angle = f.angles[mi] as f64;
                        let (hit, turn) = engine_arrival(&st, id_i, angle, count);
                        assert_eq!(hit, Some(id_j), "masked-valid action didn't reach target");
                        let expect = (f.turns[(si * PLANET_SLOTS + sj) * ACTIONS_DIM + a] * 20.0).round() as usize;
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
            assert_eq!(f.turns.len(), PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM);
            assert_eq!(f.mask.len(), PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM);
            assert_eq!(f.angles.len(), PLANET_SLOTS * PLANET_SLOTS * ACTIONS_DIM);
            for v in f.tokens.iter().chain(f.turns.iter()).chain(f.angles.iter()) {
                assert!(v.is_finite(), "non-finite feature");
            }
            for &v in &f.turns {
                assert!(v >= 0.0, "negative turns");
            }
            // mask implies a positive turns entry at frame 0
            for si in 0..PLANET_SLOTS {
                for sj in 0..PLANET_SLOTS {
                    for a in 0..ACTIONS_DIM {
                        let mi = (si * PLANET_SLOTS + sj) * ACTIONS_DIM + a;
                        if f.mask[mi] == 1 {
                            let di = ((si * PLANET_SLOTS) + sj) * ACTIONS_DIM + a;
                            assert!(f.turns[di] > 0.0, "masked action with zero turns");
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
        }
    }
}
