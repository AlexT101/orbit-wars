//! Pre-computed per-entity tables for the bot
//! Refreshed at start of game and when new comets spawn
//! Lookups are relative to current turn, i.e. `EntityCache::position(id, turns_ahead)`

#![allow(dead_code)]

use std::sync::Mutex;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::constants::{CENTER, COMET_RADIUS, EPISODE_STEPS, ROTATION_LIMIT};
use crate::engine::{CometGroup, Planet};
use crate::helpers::verify_shot_hits;

/// Aim solver result tuple: `(angle, turns, target_x, target_y)`.
pub type AimResult = (f64, i64, f64, f64);

#[derive(Clone, Copy)]
struct CachedAim {
    result: Option<AimResult>,
    /// Value of `last_comet_spawn_step` when this entry was stored. Mismatch
    /// against the cache's current value means a comet group has spawned (or
    /// expired) since, so a `Some` result must be re-verified before reuse.
    /// `None` results never need re-verification — new comets can only block
    /// paths, never enable them.
    comet_state: i64,
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
    /// Most recent absolute turn at which `refresh_comets` ran. Aim entries
    /// stored with a different value of this field may reference a path now
    /// blocked by a fresh comet and must be re-verified on lookup.
    last_comet_spawn_step: i64,
    /// Per-turn aim cache. `aim_cache[abs_turn]` maps `(src, target, ships)`
    /// to a cached aim result. Indexed by absolute turn because aim geometry
    /// is fully determined by the source/target positions at that turn.
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
            last_comet_spawn_step: 0,
            aim_cache: Mutex::new(aim_cache),
        }
    }

    /// Drop expired comet entries and add newly-spawned ones
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

        // A new comet may now block paths cached on prior turns. Bumping
        // `last_comet_spawn_step` invalidates the comet_state stamp on every
        // existing `Some` entry, so they get lazily re-verified on next lookup.
        self.last_comet_spawn_step = current_step;
    }

    #[inline]
    pub fn set_current_turn(&mut self, turn: i64) {
        self.current_turn = turn;
    }

    #[inline]
    pub fn get(&self, id: i64) -> Option<&Entity> {
        self.entities.get(&id)
    }

    /// Look up a cached aim result keyed at the current absolute turn.
    ///
    /// `Some` entries stored before the most recent comet spawn are
    /// re-verified against the current obstacle set: passing entries are
    /// returned (with `turn_stored` refreshed so the next lookup is free);
    /// failing entries are evicted and reported as `Stale` so the caller
    /// recomputes. `None` (no-shot) entries are valid forever — a new comet
    /// can only block paths, never enable them.
    pub fn aim_cache_lookup(&self, src: i64, target: i64, ships: i64) -> AimCacheVerdict {
        let slot = self.current_turn;
        if slot < 0 || (slot as usize) >= EPISODE_STEPS as usize {
            return AimCacheVerdict::Miss;
        }
        let slot = slot as usize;
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
            Some(result) if entry.comet_state == self.last_comet_spawn_step => {
                AimCacheVerdict::Hit(Some(result))
            }
            Some(result) => {
                let (angle, turns, _, _) = result;
                let src_radius = match self.get(src) {
                    Some(ent) => ent.radius,
                    None => {
                        self.aim_cache.lock().unwrap()[slot].remove(&key);
                        return AimCacheVerdict::Stale;
                    }
                };
                let [sx, sy] = match self.position(src, 0) {
                    Some(p) => p,
                    None => {
                        self.aim_cache.lock().unwrap()[slot].remove(&key);
                        return AimCacheVerdict::Stale;
                    }
                };
                if verify_shot_hits(sx, sy, src_radius, angle, turns, ships, target, self) {
                    self.aim_cache.lock().unwrap()[slot].insert(
                        key,
                        CachedAim {
                            result: Some(result),
                            comet_state: self.last_comet_spawn_step,
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

    /// Store an aim result for `(src, target, ships)` at the current turn.
    pub fn aim_cache_store(&self, src: i64, target: i64, ships: i64, result: Option<AimResult>) {
        let slot = self.current_turn;
        if slot < 0 || (slot as usize) >= EPISODE_STEPS as usize {
            return;
        }
        let entry = CachedAim {
            result,
            comet_state: self.last_comet_spawn_step,
        };
        self.aim_cache.lock().unwrap()[slot as usize].insert((src, target, ships), entry);
    }

    /// Drop all cached aim results for `turn`. Called once per bot turn to
    /// keep the cache from accumulating dead entries over a 500-turn match.
    pub fn clear_aim_cache_slot(&mut self, turn: i64) {
        if turn < 0 || (turn as usize) >= EPISODE_STEPS as usize {
            return;
        }
        if let Some(slot) = self.aim_cache.get_mut().unwrap().get_mut(turn as usize) {
            slot.clear();
        }
    }

    /// Position of `id` at `current_turn + turns_ahead`
    #[inline]
    pub fn position(&self, id: i64, turns_ahead: i64) -> Option<[f64; 2]> {
        let entity = self.entities.get(&id)?;
        let abs = self.current_turn + turns_ahead;
        if abs < 0 || abs >= EPISODE_STEPS {
            return None;
        }
        entity.positions[abs as usize]
    }

    /// Turns remaining until `id` leaves the board (for comets) or game end (for planets), relative to `current_turn`.
    pub fn remaining_life(&self, id: i64) -> i64 {
        let Some(entity) = self.entities.get(&id) else {
            return 0;
        };
        if !entity.is_comet() {
            return (EPISODE_STEPS - self.current_turn).max(0);
        }
        let start = self.current_turn.max(0) as usize;
        let end = EPISODE_STEPS as usize;
        for t in start..end {
            if entity.positions[t].is_none() {
                return t as i64 - self.current_turn;
            }
        }
        (EPISODE_STEPS - self.current_turn).max(0)
    }
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

    if let Some(path) = group.paths.get(idx) {
        let base = group.path_index - current_step;
        for t in 0..(EPISODE_STEPS as i64) {
            let pi = base + t;
            if pi >= 0 && (pi as usize) < path.len() {
                let p = path[pi as usize];
                positions[t as usize] = Some([p[0], p[1]]);
            }
        }
    }

    Entity {
        id: pid,
        kind: EntityKind::Comet,
        radius: COMET_RADIUS,
        orbital_radius: 0.0,
        positions,
    }
}
