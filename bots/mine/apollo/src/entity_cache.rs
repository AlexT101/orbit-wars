//! Pre-computed per-entity tables for the bot
//! Refreshed at start of game and when new comets spawn
//! Lookups are relative to current turn, i.e. `EntityCache::position(id, turns_ahead)`

#![allow(dead_code)]

use std::f64::consts::FRAC_PI_2;
use std::sync::Mutex;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::blockers;
use crate::constants::{CENTER, COMET_RADIUS, COMET_SPAWN_STEPS, EPISODE_STEPS, ROTATION_LIMIT};
use crate::engine::{CometGroup, Planet};

/// Aim solver result tuple: `(angle, turns, target_x, target_y,
/// fractional_flight_time)`. See [`crate::blockers::AimResult`].
pub use crate::blockers::AimResult;

#[derive(Clone, Copy)]
struct CachedAim {
    result: Option<AimResult>,
    /// Absolute game turn at which this entry was stored. Used to decide
    /// whether a [`COMET_SPAWN_STEPS`] event has occurred since (in which
    /// case `Some` results need re-verification — new comets can only block
    /// paths, never enable them, so `None` entries are valid forever).
    stored_at_turn: i64,
}

/// Verdict from [`EntityCache::aim_cache_lookup`]. `Stale` means an entry
/// existed but failed post-comet re-verification and was evicted; the caller
/// must recompute.
pub enum AimCacheVerdict {
    Miss,
    Hit(Option<AimResult>),
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    StaticPlanet,
    OrbitingPlanet,
    Comet,
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: i64,
    pub kind: EntityKind,
    pub radius: f64,
    pub orbital_radius: f64,                // 0.0 for comets since they don't orbit sun
    pub positions: Vec<Option<[f64; 2]>>,   // Pre-computed positions, or None if not on board (comets)
    /// First absolute turn at which the entity is off the board, precomputed at
    /// build time so [`EntityCache::remaining_life`] is O(1). Comet paths are a
    /// single contiguous on-board window, so this is `last_on_board + 1`
    /// (clamped to `EPISODE_STEPS`). Planets never leave: `EPISODE_STEPS`.
    off_board_turn: i64,
}

impl Entity {
    #[inline]
    pub fn is_static(&self) -> bool {
        matches!(self.kind, EntityKind::StaticPlanet)
    }

    #[inline]
    pub fn is_dynamic(&self) -> bool {
        !self.is_static()
    }

    #[inline]
    pub fn is_comet(&self) -> bool {
        matches!(self.kind, EntityKind::Comet)
    }
}

/// Game-wide pre-computed position tables, keyed by planet/comet id.
pub struct EntityCache {
    pub current_turn: i64,
    pub angular_velocity: f64,
    pub entities: HashMap<i64, Entity>,
    /// Per-turn aim cache. `aim_cache[abs_launch_turn]` maps
    /// `(src, target, ships)` to a cached aim result. Indexed by the
    /// **absolute launch turn** (= `current_turn + launch_turn_offset` at
    /// storage time) rather than the storage turn, so that:
    ///   * Delayed-launch entries from the hellburner early-game DFS and
    ///     launch-now entries from later real turns share the slot whose
    ///     `abs_launch` matches, giving cross-turn reuse for free.
    ///   * Rollout forward-sim entries (stored at the rollout's notion of
    ///     `current_turn`) likewise share slots with the bot's real turns.
    aim_cache: Mutex<Vec<HashMap<(i64, i64, i64), CachedAim>>>,
}

impl EntityCache {
    pub fn build(
        initial_planets: &[Planet],
        comets: &[CometGroup],
        comet_planet_ids: &[i64],
        angular_velocity: f64,
        current_step: i64,
    ) -> Self {
        let comet_ids: HashSet<i64> = comet_planet_ids.iter().copied().collect();
        let mut entities =
            HashMap::with_capacity_and_hasher(initial_planets.len(), Default::default());

        for ip in initial_planets {
            if comet_ids.contains(&ip.id) {
                continue;
            }
            entities.insert(ip.id, build_planet_entity(ip, angular_velocity));
        }

        for group in comets {
            for (idx, &pid) in group.planet_ids.iter().enumerate() {
                entities.insert(pid, build_comet_entity(pid, group, idx, current_step));
            }
        }

        let aim_cache = (0..EPISODE_STEPS).map(|_| HashMap::default()).collect();

        Self {
            current_turn: current_step,
            angular_velocity,
            entities,
            aim_cache: Mutex::new(aim_cache),
        }
    }

    /// Drop expired comet entries and add newly-spawned ones.
    pub fn refresh_comets(
        &mut self,
        comets: &[CometGroup],
        comet_planet_ids: &[i64],
        current_step: i64,
    ) {
        let comet_ids: HashSet<i64> = comet_planet_ids.iter().copied().collect();

        self.entities
            .retain(|id, ent| !ent.is_comet() || comet_ids.contains(id));

        for group in comets {
            for (idx, &pid) in group.planet_ids.iter().enumerate() {
                self.entities
                    .entry(pid)
                    .or_insert_with(|| build_comet_entity(pid, group, idx, current_step));
            }
        }
    }

    #[inline]
    pub fn set_current_turn(&mut self, turn: i64) {
        self.current_turn = turn;
    }

    #[inline]
    pub fn get(&self, id: i64) -> Option<&Entity> {
        self.entities.get(&id)
    }

    /// Look up a cached aim result for a shot launching at
    /// `current_turn + launch_turn_offset`.
    ///
    /// `Some` entries stored across a [`COMET_SPAWN_STEPS`] boundary are
    /// re-verified against the obstacle set at the entry's launch turn:
    /// passing entries are returned (with `stored_at_turn` refreshed so the
    /// next lookup is free); failing entries are evicted and reported as
    /// `Stale`. `None` entries never need re-verification — a fresh comet
    /// can only block paths, never enable them.
    pub fn aim_cache_lookup(
        &self,
        src: i64,
        target: i64,
        ships: i64,
        launch_turn_offset: i64,
    ) -> AimCacheVerdict {
        let abs_launch = self.current_turn + launch_turn_offset;
        if abs_launch < 0 || (abs_launch as usize) >= EPISODE_STEPS as usize {
            return AimCacheVerdict::Miss;
        }
        let slot = abs_launch as usize;
        let key = (src, target, ships);

        let entry = {
            let map = self.aim_cache.lock().unwrap();
            match map[slot].get(&key) {
                None => return AimCacheVerdict::Miss,
                Some(e) => *e,
            }
        };

        match entry.result {
            None => AimCacheVerdict::Hit(None),
            Some(result) => {
                if !comet_spawn_crossed(entry.stored_at_turn, self.current_turn) {
                    return AimCacheVerdict::Hit(Some(result));
                }
                let (angle, _turns, _tx, _ty, flight_time) = result;
                if blockers::shot_still_clear(
                    self,
                    src,
                    target,
                    ships,
                    angle,
                    flight_time,
                    launch_turn_offset,
                ) {
                    self.aim_cache.lock().unwrap()[slot].insert(
                        key,
                        CachedAim {
                            result: Some(result),
                            stored_at_turn: self.current_turn,
                        },
                    );
                    AimCacheVerdict::Hit(Some(result))
                } else {
                    self.aim_cache.lock().unwrap()[slot].remove(&key);
                    AimCacheVerdict::Stale
                }
            }
        }
    }

    /// Store an aim result and, for free, populate its three quartet siblings.
    ///
    /// The whole obstacle field (sun at CENTER + every planet + every comet) is
    /// invariant under rotation by 90° about CENTER **at every absolute turn**
    /// — orbital motion and comet motion commute with that rotation, and each
    /// quartet's four members are exactly the four reflections of the same base
    /// point (see `rust_engine/src/lib.rs`). Hence rotating *both* endpoints to
    /// their corresponding quartet members (member `i` → member `j` via the
    /// rotation `(TAG[j] − TAG[i])·90°`) yields a congruent aim problem whose
    /// solution is this one rotated by the same `k` quarter-turns: `angle` gains
    /// `k·π/2`, the target point rotates about CENTER, and `turns`/`flight_time`
    /// are unchanged. `None` (no feasible shot) propagates too. Both endpoints
    /// are rotated by the *same* `k`, so same-quartet `src`/`target` pairs need
    /// no special handling. Siblings share the `abs_launch` slot (the symmetry
    /// holds at that exact turn) and a fresh `stored_at_turn`.
    pub fn aim_cache_store(
        &self,
        src: i64,
        target: i64,
        ships: i64,
        launch_turn_offset: i64,
        result: Option<AimResult>,
    ) {
        let abs_launch = self.current_turn + launch_turn_offset;
        if abs_launch < 0 || (abs_launch as usize) >= EPISODE_STEPS as usize {
            return;
        }
        let slot = abs_launch as usize;
        let mut map = self.aim_cache.lock().unwrap();

        // Original entry (always stored, preserving prior behavior).
        map[slot].insert(
            (src, target, ships),
            CachedAim {
                result,
                stored_at_turn: self.current_turn,
            },
        );

        // Three rotated siblings. Gated on existence so we never populate ids for
        // a comet member that has expired or not yet spawned.
        for k in 1..=3 {
            let sib_src = rot_sibling(src, k);
            let sib_target = rot_sibling(target, k);
            if !self.entities.contains_key(&sib_src)
                || !self.entities.contains_key(&sib_target)
            {
                continue;
            }
            map[slot].insert(
                (sib_src, sib_target, ships),
                CachedAim {
                    result: rotate_aim_result(result, k),
                    stored_at_turn: self.current_turn,
                },
            );
        }
    }

    /// Drop all cached aim results for launches at absolute `turn`. Called
    /// once per bot turn (with `turn = current_turn - 1`) to release slots
    /// whose launch time has passed and that can no longer be queried.
    pub fn clear_aim_cache_slot(&mut self, turn: i64) {
        if turn < 0 || (turn as usize) >= EPISODE_STEPS as usize {
            return;
        }
        if let Some(slot) = self.aim_cache.get_mut().unwrap().get_mut(turn as usize) {
            slot.clear();
        }
    }

    #[inline]
    pub fn position(&self, id: i64, turns_ahead: i64) -> Option<[f64; 2]> {
        let entity = self.entities.get(&id)?;
        let abs = self.current_turn + turns_ahead;
        if abs < 0 || abs >= EPISODE_STEPS {
            return None;
        }
        entity.positions[abs as usize]
    }

    /// Position of `id` at an **absolute** game turn, indexed directly into the
    /// precomputed table (unlike [`position`], which is relative to
    /// `current_turn`). Used by [`crate::engine::Simulator`] during rollout,
    /// where the simulator tracks its own absolute step and `current_turn` reflects
    /// a different (per-bot-turn) notion. Returns `None` when the entity is
    /// unknown, off-board (comets), or the turn is outside `[0, EPISODE_STEPS)`.
    #[inline]
    pub fn position_abs(&self, id: i64, abs_step: i64) -> Option<[f64; 2]> {
        if abs_step < 0 || abs_step >= EPISODE_STEPS {
            return None;
        }
        self.entities.get(&id)?.positions[abs_step as usize]
    }

    /// Turns remaining until `id` leaves the board (for comets) or game end (for
    /// planets), relative to `current_turn`. O(1) via the precomputed
    /// `off_board_turn`; returns 0 for a comet not currently on the board
    /// (already gone, or not yet spawned).
    pub fn remaining_life(&self, id: i64) -> i64 {
        let Some(entity) = self.entities.get(&id) else {
            return 0;
        };
        if !entity.is_comet() {
            return (EPISODE_STEPS - self.current_turn).max(0);
        }
        let cur = self.current_turn;
        if cur < 0 || cur >= EPISODE_STEPS || entity.positions[cur as usize].is_none() {
            return 0;
        }
        (entity.off_board_turn - cur).max(0)
    }
}

/// `true` iff any [`COMET_SPAWN_STEPS`] tick falls in `(stored, current]`.
/// Comets always spawn on these fixed game steps, so this is the exact
/// criterion for "the obstacle set may have grown since the cache write."
#[inline]
fn comet_spawn_crossed(stored: i64, current: i64) -> bool {
    if current <= stored {
        return false;
    }
    COMET_SPAWN_STEPS
        .iter()
        .any(|&s| s > stored && s <= current)
}

/// Member-index ⇄ rotation-tag map for a quartet. The four members of a base
/// point are its four D4 reflections, generated in this index order (engine
/// `generate_planets` / `generate_comet_paths`); ordering them by the
/// rotation taking member 0 → member i gives tags `[0, 1, 3, 2]` (members 2 and
/// 3 are swapped vs. naive index order). `TAG` is its own inverse, so the same
/// table maps both directions. The rotation from member `i` to member `j` is
/// `(TAG[j] − TAG[i]) mod 4` quarter-turns CCW about CENTER.
const TAG: [i64; 4] = [0, 1, 3, 2];

/// Id of the quartet sibling reached by rotating `id`'s position `k`
/// quarter-turns CCW about CENTER. Ids are assigned 4-per-group on a `/4`
/// boundary (planets `0..4G`; each comet group's four ids start at `max_id+1`,
/// itself a multiple of 4), so `id & 3` is the member index and `id - (id & 3)`
/// the quartet base.
#[inline]
pub(crate) fn rot_sibling(id: i64, k: i64) -> i64 {
    let m = (id & 3) as usize;
    let rotated = TAG[((TAG[m] + k) & 3) as usize];
    (id - id % 4) + rotated
}

/// Rotate `(x, y)` `k` quarter-turns CCW about CENTER. Exact (each step is the
/// integer-coefficient map `(x, y) → (-y, x)` about CENTER).
#[inline]
fn rot_point(x: f64, y: f64, k: i64) -> (f64, f64) {
    let mut dx = x - CENTER;
    let mut dy = y - CENTER;
    for _ in 0..(k & 3) {
        let (nx, ny) = (-dy, dx);
        dx = nx;
        dy = ny;
    }
    (CENTER + dx, CENTER + dy)
}

/// Rotate an aim result `k` quarter-turns CCW: bearing gains `k·π/2`, the target
/// point rotates about CENTER, `turns`/`flight_time` are invariant. `None`
/// (no feasible shot) maps to `None`.
#[inline]
fn rotate_aim_result(result: Option<AimResult>, k: i64) -> Option<AimResult> {
    result.map(|(angle, turns, tx, ty, flight_time)| {
        let (rx, ry) = rot_point(tx, ty, k);
        (angle + (k & 3) as f64 * FRAC_PI_2, turns, rx, ry, flight_time)
    })
}

fn build_planet_entity(planet: &Planet, angular_velocity: f64) -> Entity {
    let dx = planet.x - CENTER;
    let dy = planet.y - CENTER;
    let orbital_radius = (dx * dx + dy * dy).sqrt();
    let is_static = orbital_radius + planet.radius >= ROTATION_LIMIT;
    let kind = if is_static {
        EntityKind::StaticPlanet
    } else {
        EntityKind::OrbitingPlanet
    };

    let cap = EPISODE_STEPS as usize;
    let mut positions = Vec::with_capacity(cap);
    if is_static {
        for _ in 0..cap {
            positions.push(Some([planet.x, planet.y]));
        }
    } else {
        let init_angle = dy.atan2(dx);
        for t in 0..(EPISODE_STEPS) {
            let effective = (t - 1).max(0);
            let angle = init_angle + angular_velocity * effective as f64;
            positions.push(Some([
                CENTER + orbital_radius * angle.cos(),
                CENTER + orbital_radius * angle.sin(),
            ]));
        }
    }

    Entity {
        id: planet.id,
        kind,
        radius: planet.radius,
        orbital_radius,
        positions,
        off_board_turn: EPISODE_STEPS, // planets stay on the board all game
    }
}

fn build_comet_entity(
    pid: i64,
    group: &CometGroup,
    idx: usize,
    current_step: i64,
) -> Entity {
    let cap = EPISODE_STEPS as usize;
    let mut positions = vec![None; cap];

    let mut last_on_board: i64 = -1;
    if let Some(path) = group.paths.get(idx) {
        let base = group.path_index - current_step;
        for t in 0..(EPISODE_STEPS as i64) {
            let pi = base + t;
            if pi >= 0 && (pi as usize) < path.len() {
                let p = path[pi as usize];
                positions[t as usize] = Some([p[0], p[1]]);
                last_on_board = t;
            }
        }
    }
    // First off-board turn = one past the last on-board turn (clamped). 0 if the
    // comet is never on board within range.
    let off_board_turn = if last_on_board < 0 {
        0
    } else {
        (last_on_board + 1).min(EPISODE_STEPS)
    };

    Entity {
        id: pid,
        kind: EntityKind::Comet,
        radius: COMET_RADIUS,
        orbital_radius: 0.0,
        positions,
        off_board_turn,
    }
}
