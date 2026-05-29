//! Pre-computed per-entity tables for the bot
//! Refreshed at start of game and when new comets spawn
//! Lookups are relative to current turn, i.e. `EntityCache::position(id, turns_ahead)`

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::blockers::{self, BlockerTable};
use crate::constants::{
    CENTER, COMET_RADIUS, COMET_SPAWN_STEPS, EPISODE_STEPS, MAX_SHIP_SPEED, ROTATION_LIMIT,
};
use crate::engine::{fleet_speed, CometGroup, Planet};

/// Aim solver result tuple: `(angle, turns, target_x, target_y,
/// fractional_flight_time)`. See [`crate::blockers::AimResult`].
pub type AimResult = (f64, i64, f64, f64, f64);

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
    /// Lazily-built [`BlockerTable`]s keyed by
    /// `(shooter_id, absolute_launch_turn, ships)`, built at the exact
    /// `fleet_speed(ships)`. Only consulted on the blocked path of
    /// [`blockers::aim_with_prediction`] for the blocked arc's angular edges
    /// (clear/blocked verdicts use the table-free [`blockers::shot_blocked_exact`]).
    /// Cleared on comet spawn; orbiter geometry is permanent but a fresh comet
    /// may have introduced entries not in the cached table.
    blocker_tables: Mutex<HashMap<(i64, i64, i64), Arc<BlockerTable>>>,
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
            blocker_tables: Mutex::new(HashMap::default()),
        }
    }

    /// Drop expired comet entries and add newly-spawned ones. Also clears the
    /// blocker-table cache: orbiter geometry is permanent but cached tables
    /// may have been built without the freshly-spawned comets.
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

        self.blocker_tables.get_mut().unwrap().clear();
    }

    #[inline]
    pub fn set_current_turn(&mut self, turn: i64) {
        self.current_turn = turn;
    }

    #[inline]
    pub fn get(&self, id: i64) -> Option<&Entity> {
        self.entities.get(&id)
    }

    /// Cached blocker table for `(shooter_id, launch_turn_offset, ships)`, built
    /// at the **exact** fleet speed `fleet_speed(ships)` and keyed internally by
    /// the *absolute* launch turn (so entries are reused across `set_current_turn`
    /// during rollout forward-sim) and by `ships` (so the table matches the engine
    /// trajectory exactly — no speed quantization). This is only consulted on the
    /// blocked path of [`blockers::aim_with_prediction`] to obtain the blocked
    /// arc's angular edges; clear/blocked verdicts themselves come from the
    /// table-free exact swept-pair in [`blockers::shot_blocked_exact`].
    pub fn blocker_table(
        &self,
        shooter_id: i64,
        launch_turn_offset: i64,
        ships: i64,
    ) -> Arc<BlockerTable> {
        let ships = ships.max(1);
        let v = fleet_speed(ships, MAX_SHIP_SPEED);
        let abs_launch = self.current_turn + launch_turn_offset;
        let key = (shooter_id, abs_launch, ships);
        let mut guard = self.blocker_tables.lock().unwrap();
        if let Some(t) = guard.get(&key) {
            return t.clone();
        }
        let table = Arc::new(blockers::build_blocker_table(
            self,
            shooter_id,
            launch_turn_offset,
            v,
        ));
        guard.insert(key, table.clone());
        table
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
        let entry = CachedAim {
            result,
            stored_at_turn: self.current_turn,
        };
        self.aim_cache.lock().unwrap()[abs_launch as usize].insert((src, target, ships), entry);
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
