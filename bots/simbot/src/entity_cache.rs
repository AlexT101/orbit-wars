//! Pre-computed per-entity tables for the bot
//! Refreshed at start of game and when new comets spawn
//! Lookups are relative to current turn, i.e. `EntityCache::position(id, turns_ahead)`

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::constants::{CENTER, COMET_RADIUS, EPISODE_STEPS, ROTATION_LIMIT};
use crate::engine::{CometGroup, Planet};

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
    pub tolerance: i64,                     // Turn tolerance for intercept checks
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
        let mut entities = HashMap::with_capacity(initial_planets.len());

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

        Self {
            current_turn: current_step,
            angular_velocity,
            entities,
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
    }

    #[inline]
    pub fn set_current_turn(&mut self, turn: i64) {
        self.current_turn = turn;
    }

    #[inline]
    pub fn get(&self, id: i64) -> Option<&Entity> {
        self.entities.get(&id)
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

    let tolerance = if is_static {
        1
    } else if orbital_radius * angular_velocity.abs() >= 1.0 {
        2
    } else {
        1
    };

    Entity {
        id: planet.id,
        kind,
        radius: planet.radius,
        orbital_radius,
        tolerance,
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
        tolerance: 2,
        positions,
    }
}
