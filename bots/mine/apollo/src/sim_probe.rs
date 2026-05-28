//! Lookahead simulator for bot decision-making — a fast-path version of
//! `engine::EngineState::step_with_actions` for repeated short-horizon rollouts.
//!
//! - Borrows comet path tables and initial-planet positions from the parent
//!   engine instead of cloning (≈8KB saved per probe).
//! - Reuses per-step scratch buffers across the rollout.
//! - Emits a per-step event log so callers observe the rollout instead of
//!   re-deriving collision math.
//!
//! Two deliberate simplifications:
//!
//! 1. Comets spawned mid-rollout are skipped — the bot can't observe them on
//!    turn 0 anyway.
//! 2. Episode termination and reward computation are skipped; callers cap
//!    the horizon.

#![allow(dead_code)]

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::constants::{BOARD_SIZE, CENTER, EPISODE_STEPS, MAX_PLAYERS, ROTATION_LIMIT, SUN_RADIUS};
use crate::engine::{
    fleet_speed, point_to_segment_distance, swept_pair_hit,
    EngineState, Fleet, MoveAction, Planet, PlanetPath,
};

/// Predicted landing of one in-flight fleet, with `turns` measured from the
/// probe's start step.
#[derive(Debug, Clone, Copy)]
pub struct ArrivalEvent {
    pub turns: i64,
    pub owner: i64,
    pub ships: i64,
}

/// Per-step rollout events. `turn` is 1-indexed turns since the probe was
/// constructed (`turn = 1` means "end of the first stepped turn").
#[derive(Debug, Clone, Copy)]
pub enum SimEvent {
    /// A fleet hit a planet. Emitted *before* combat resolution: the fleet is
    /// consumed and counts toward this turn's combat tally on `planet_id`.
    FleetLanded {
        turn: i64,
        planet_id: i64,
        fleet_owner: i64,
        fleet_ships: i64,
    },
    /// A planet's owner changed post-combat this turn. `new_owner == -1` means
    /// the top two attackers tied and neutralised each other.
    OwnerChanged {
        turn: i64,
        planet_id: i64,
        prev_owner: i64,
        new_owner: i64,
        ships: i64,
    },
}

/// Comet group with paths borrowed from the parent engine. `path_index` and
/// `planet_ids` are owned because they advance/shrink during the rollout.
struct ProbeCometGroup<'a> {
    planet_ids: Vec<i64>,
    paths: &'a [Vec<[f64; 2]>],
    path_index: i64,
}

pub struct SimProbe<'a> {
    // ── Borrowed from parent (immutable for the lifetime of the probe).
    initial_by_id: HashMap<i64, &'a Planet>,
    angular_velocity: f64,
    ship_speed: f64,
    num_players: usize,

    // ── Owned mutable state.
    step: i64,
    initial_step: i64,
    planets: Vec<Planet>,
    fleets: Vec<Fleet>,
    next_fleet_id: i64,
    comet_planet_ids: Vec<i64>,
    comet_groups: Vec<ProbeCometGroup<'a>>,
    planet_index_by_id: HashMap<i64, usize>,

    // ── Scratch buffers reused across step calls (kept allocated for the
    // probe's lifetime; cleared, not re-allocated, on each step).
    planet_paths: Vec<Option<PlanetPath>>,
    fleets_to_remove: Vec<bool>,
    combat_lists: Vec<Vec<(i64, i64)>>,
    comet_id_set: HashSet<i64>,
    expired_postmove: Vec<i64>,

    events: Vec<SimEvent>,
}

impl<'a> SimProbe<'a> {
    /// Construct a probe seeded from `state`. The probe borrows `state`'s
    /// comet path tables and initial-planet table; the engine reference must
    /// outlive the probe.
    pub fn from_engine(state: &'a EngineState) -> Self {
        let planet_count = state.planets.len();
        let fleet_count = state.fleets.len();

        let initial_by_id: HashMap<i64, &'a Planet> = state
            .initial_planets
            .iter()
            .map(|p| (p.id, p))
            .collect();

        let comet_groups: Vec<ProbeCometGroup<'a>> = state
            .comets
            .iter()
            .map(|g| ProbeCometGroup {
                planet_ids: g.planet_ids.clone(),
                paths: g.paths.as_slice(),
                path_index: g.path_index,
            })
            .collect();

        let planet_index_by_id: HashMap<i64, usize> = state
            .planets
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id, i))
            .collect();

        Self {
            initial_by_id,
            angular_velocity: state.angular_velocity,
            ship_speed: state.configuration.ship_speed,
            num_players: state.num_players,
            step: state.step,
            initial_step: state.step,
            planets: state.planets.clone(),
            fleets: state.fleets.clone(),
            next_fleet_id: state.next_fleet_id,
            comet_planet_ids: state.comet_planet_ids.clone(),
            comet_groups,
            planet_index_by_id,

            planet_paths: vec![None; planet_count],
            fleets_to_remove: vec![false; fleet_count],
            combat_lists: (0..planet_count).map(|_| Vec::new()).collect(),
            comet_id_set: HashSet::with_capacity_and_hasher(
                state.comet_planet_ids.len(),
                Default::default(),
            ),
            expired_postmove: Vec::new(),

            events: Vec::with_capacity(64),
        }
    }

    #[inline]
    pub fn planets(&self) -> &[Planet] { &self.planets }
    #[inline]
    pub fn fleets(&self) -> &[Fleet] { &self.fleets }
    #[inline]
    pub fn events(&self) -> &[SimEvent] { &self.events }
    #[inline]
    pub fn angular_velocity(&self) -> f64 { self.angular_velocity }
    #[inline]
    pub fn num_players(&self) -> usize { self.num_players }
    #[inline]
    pub fn comet_planet_ids(&self) -> &[i64] { &self.comet_planet_ids }
    /// Engine step number after the most recent `step()`.
    #[inline]
    pub fn step_count(&self) -> i64 { self.step }
    /// Turns elapsed since `from_engine`.
    #[inline]
    pub fn turns_elapsed(&self) -> i64 { self.step - self.initial_step }

    pub fn clear_events(&mut self) { self.events.clear(); }

    /// Fork a sub-probe that shares the parent's borrowed comet path tables
    /// and initial-planet table, but owns an independent copy of the mutable
    /// rollout state (planets / fleets / comet groups). Used by
    /// `TimelineCache::build` to walk forward `HORIZON` turns from the parent
    /// probe's current step without disturbing the parent.
    ///
    /// `initial_step` is reset to the parent's current step so arrival event
    /// turns are measured from the fork point.
    pub fn fork(&self) -> SimProbe<'a> {
        let planet_count = self.planets.len();
        let fleet_count = self.fleets.len();
        SimProbe {
            initial_by_id: self.initial_by_id.clone(),
            angular_velocity: self.angular_velocity,
            ship_speed: self.ship_speed,
            num_players: self.num_players,
            step: self.step,
            initial_step: self.step,
            planets: self.planets.clone(),
            fleets: self.fleets.clone(),
            next_fleet_id: self.next_fleet_id,
            comet_planet_ids: self.comet_planet_ids.clone(),
            comet_groups: self
                .comet_groups
                .iter()
                .map(|g| ProbeCometGroup {
                    planet_ids: g.planet_ids.clone(),
                    paths: g.paths,
                    path_index: g.path_index,
                })
                .collect(),
            planet_index_by_id: self.planet_index_by_id.clone(),
            planet_paths: vec![None; planet_count],
            fleets_to_remove: vec![false; fleet_count],
            combat_lists: (0..planet_count).map(|_| Vec::new()).collect(),
            comet_id_set: HashSet::with_capacity_and_hasher(
                self.comet_planet_ids.len(),
                Default::default(),
            ),
            expired_postmove: Vec::new(),
            events: Vec::with_capacity(64),
        }
    }

    /// Step one turn with no player actions.
    #[inline]
    pub fn step(&mut self) {
        self.step_with_actions(&[]);
    }

    /// Step `n` turns with no player actions. Caps at `EPISODE_STEPS` so a
    /// probe started near the end of the game won't simulate phantom turns
    /// the real game never reaches.
    pub fn step_n(&mut self, n: i64) {
        for _ in 0..n.max(0) {
            if self.step >= EPISODE_STEPS {
                break;
            }
            self.step_with_actions(&[]);
        }
    }

    /// Step one turn applying `actions[p]` as player `p`'s moves. Players
    /// beyond `actions.len()` take no action. No-op once `self.step` reaches
    /// `EPISODE_STEPS` — the episode is over.
    pub fn step_with_actions(&mut self, actions: &[&[MoveAction]]) {
        if self.step >= EPISODE_STEPS {
            return;
        }
        self.expire_pre_step();

        // (skip) Comet spawning — intentional, see module doc.

        for (player_idx, moves) in actions.iter().enumerate() {
            self.process_moves(player_idx as i64, moves);
        }

        for planet in &mut self.planets {
            if planet.owner != -1 {
                planet.ships += planet.production;
            }
        }

        let turn_step = self.step;
        let event_turn = self.step + 1 - self.initial_step;
        let planet_count = self.planets.len();
        let fleet_count = self.fleets.len();
        self.reset_scratch(planet_count, fleet_count);

        for (idx, planet) in self.planets.iter().enumerate() {
            if self.comet_id_set.contains(&planet.id) {
                continue;
            }
            // Lookup by id (not by index) so this still works after comet
            // expiry has shifted self.planets while initial_by_id stays put.
            let initial_p = self
                .initial_by_id
                .get(&planet.id)
                .expect("non-comet planet missing initial entry");
            let old_pos = (planet.x, planet.y);
            let mut new_pos = old_pos;
            let dx = initial_p.x - CENTER;
            let dy = initial_p.y - CENTER;
            let orbital_r = (dx * dx + dy * dy).sqrt();
            if orbital_r + planet.radius < ROTATION_LIMIT {
                let initial_angle = dy.atan2(dx);
                let current_angle = initial_angle + self.angular_velocity * turn_step as f64;
                new_pos = (
                    CENTER + orbital_r * current_angle.cos(),
                    CENTER + orbital_r * current_angle.sin(),
                );
            }
            self.planet_paths[idx] = Some(PlanetPath {
                old_pos,
                new_pos,
                check_collision: true,
            });
        }

        // Comet movement; record postmove expiries for cleanup at the end.
        self.expired_postmove.clear();
        for group in &mut self.comet_groups {
            group.path_index += 1;
            let idx = group.path_index as usize;
            for (i, pid) in group.planet_ids.iter().enumerate() {
                let Some(&planet_idx) = self.planet_index_by_id.get(pid) else {
                    continue;
                };
                let planet = &self.planets[planet_idx];
                let old_pos = (planet.x, planet.y);
                let p_path = &group.paths[i];
                if idx >= p_path.len() {
                    self.expired_postmove.push(*pid);
                    self.planet_paths[planet_idx] = Some(PlanetPath {
                        old_pos,
                        new_pos: old_pos,
                        check_collision: true,
                    });
                } else {
                    let next = p_path[idx];
                    self.planet_paths[planet_idx] = Some(PlanetPath {
                        old_pos,
                        new_pos: (next[0], next[1]),
                        check_collision: old_pos.0 >= 0.0,
                    });
                }
            }
        }

        for (fleet_idx, fleet) in self.fleets.iter_mut().enumerate() {
            let old_pos = (fleet.x, fleet.y);
            let speed = fleet_speed(fleet.ships, self.ship_speed);
            fleet.x += fleet.angle.cos() * speed;
            fleet.y += fleet.angle.sin() * speed;
            let new_pos = (fleet.x, fleet.y);

            let mut hit_planet = false;
            for (planet_idx, planet) in self.planets.iter().enumerate() {
                let Some(path) = &self.planet_paths[planet_idx] else {
                    continue;
                };
                if !path.check_collision {
                    continue;
                }
                if swept_pair_hit(old_pos, new_pos, path.old_pos, path.new_pos, planet.radius) {
                    self.combat_lists[planet_idx].push((fleet.owner, fleet.ships));
                    self.fleets_to_remove[fleet_idx] = true;
                    self.events.push(SimEvent::FleetLanded {
                        turn: event_turn,
                        planet_id: planet.id,
                        fleet_owner: fleet.owner,
                        fleet_ships: fleet.ships,
                    });
                    hit_planet = true;
                    break;
                }
            }
            if hit_planet {
                continue;
            }

            if !(0.0..=BOARD_SIZE).contains(&fleet.x) || !(0.0..=BOARD_SIZE).contains(&fleet.y) {
                self.fleets_to_remove[fleet_idx] = true;
                continue;
            }
            if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS {
                self.fleets_to_remove[fleet_idx] = true;
                continue;
            }
        }

        for (idx, planet) in self.planets.iter_mut().enumerate() {
            if let Some(path) = &self.planet_paths[idx] {
                planet.x = path.new_pos.0;
                planet.y = path.new_pos.1;
            }
        }

        for (idx, planet) in self.planets.iter_mut().enumerate() {
            let planet_fleets = &self.combat_lists[idx];
            if planet_fleets.is_empty() {
                continue;
            }

            let mut player_ships = [0i64; MAX_PLAYERS];
            for &(owner, ships) in planet_fleets {
                if owner >= 0 && (owner as usize) < MAX_PLAYERS {
                    player_ships[owner as usize] += ships;
                }
            }

            // Top-1/top-2 scan with ascending player_idx tie-break.
            let mut top_player: i64 = -1;
            let mut top_ships: i64 = -1;
            let mut second_ships: i64 = -1;
            let mut entry_count = 0;
            for (pidx, &ships) in player_ships.iter().enumerate() {
                if ships <= 0 {
                    continue;
                }
                entry_count += 1;
                if ships > top_ships {
                    second_ships = top_ships;
                    top_ships = ships;
                    top_player = pidx as i64;
                } else if ships > second_ships {
                    second_ships = ships;
                }
            }
            if entry_count == 0 {
                continue;
            }

            let (survivor_owner, survivor_ships) = if entry_count > 1 {
                let s = if top_ships == second_ships {
                    0
                } else {
                    top_ships - second_ships
                };
                let o = if s > 0 { top_player } else { -1 };
                (o, s)
            } else {
                (top_player, top_ships)
            };

            if survivor_ships > 0 {
                let prev_owner = planet.owner;
                if planet.owner == survivor_owner {
                    planet.ships += survivor_ships;
                } else {
                    planet.ships -= survivor_ships;
                    if planet.ships < 0 {
                        planet.owner = survivor_owner;
                        planet.ships = planet.ships.abs();
                    }
                }
                if planet.owner != prev_owner {
                    self.events.push(SimEvent::OwnerChanged {
                        turn: event_turn,
                        planet_id: planet.id,
                        prev_owner,
                        new_owner: planet.owner,
                        ships: planet.ships,
                    });
                }
            }
        }

        // Apply postmove comet expiry now that combat has been resolved
        // against the pre-removal planet indexing.
        if !self.expired_postmove.is_empty() {
            // Swap-out to release the borrow on self.
            let mut expired = std::mem::take(&mut self.expired_postmove);
            self.remove_planets(&expired);
            expired.clear();
            self.expired_postmove = expired;
        }

        // Remove destroyed fleets in place, indexed by pre-retain position.
        let removal_flags = &self.fleets_to_remove;
        let mut idx = 0usize;
        self.fleets.retain(|_| {
            let keep = !removal_flags[idx];
            idx += 1;
            keep
        });

        // (skip) termination check + rewards — see module doc.

        self.step += 1;
    }

    /// Convenience: step one turn applying `moves` as player `player`'s moves,
    /// with all other players taking no action. Avoids heap allocation for the
    /// per-player action slice.
    pub fn step_with_player_actions(&mut self, player: i64, moves: &[MoveAction]) {
        let empty: &[MoveAction] = &[];
        let mut slots: [&[MoveAction]; MAX_PLAYERS] = [empty; MAX_PLAYERS];
        if player >= 0 && (player as usize) < MAX_PLAYERS {
            slots[player as usize] = moves;
        }
        self.step_with_actions(&slots);
    }

    /// Re-bucket `FleetLanded` events into the per-planet arrival ledger shape
    /// that the bot's strategy layer consumes. `OwnerChanged` events are
    /// ignored — read them separately via `events()`.
    pub fn collect_arrivals(&self) -> HashMap<i64, Vec<ArrivalEvent>> {
        let mut out: HashMap<i64, Vec<ArrivalEvent>> = HashMap::default();
        for ev in &self.events {
            if let SimEvent::FleetLanded {
                turn,
                planet_id,
                fleet_owner,
                fleet_ships,
            } = *ev
            {
                out.entry(planet_id).or_default().push(ArrivalEvent {
                    turns: turn,
                    owner: fleet_owner,
                    ships: fleet_ships,
                });
            }
        }
        out
    }

    // ── Internal ─────────────────────────────────────────────────────────

    fn process_moves(&mut self, player_id: i64, actions: &[MoveAction]) {
        for ma in actions {
            let Some(&idx) = self.planet_index_by_id.get(&ma.from_id) else {
                continue;
            };
            let from = &mut self.planets[idx];
            if from.owner != player_id {
                continue;
            }
            if ma.ships <= 0 || from.ships < ma.ships {
                continue;
            }
            from.ships -= ma.ships;
            // Snapshot geometry so the &mut self.planets borrow ends before
            // we push to self.fleets.
            let radius = from.radius;
            let fx = from.x;
            let fy = from.y;

            let start_x = fx + ma.angle.cos() * (radius + 0.1);
            let start_y = fy + ma.angle.sin() * (radius + 0.1);
            self.fleets.push(Fleet {
                id: self.next_fleet_id,
                owner: player_id,
                x: start_x,
                y: start_y,
                angle: ma.angle,
                from_planet_id: ma.from_id,
                ships: ma.ships,
            });
            self.next_fleet_id += 1;
        }
    }

    /// Drop comets whose path ran out before this turn began.
    fn expire_pre_step(&mut self) {
        if self.comet_groups.is_empty() {
            return;
        }
        let mut expired: Vec<i64> = Vec::new();
        for group in &self.comet_groups {
            let idx = group.path_index;
            for (i, pid) in group.planet_ids.iter().enumerate() {
                if idx >= group.paths[i].len() as i64 {
                    expired.push(*pid);
                }
            }
        }
        if !expired.is_empty() {
            self.remove_planets(&expired);
        }
    }

    fn remove_planets(&mut self, expired_ids: &[i64]) {
        let expired_set: HashSet<i64> = expired_ids.iter().copied().collect();
        self.planets.retain(|p| !expired_set.contains(&p.id));
        self.comet_planet_ids
            .retain(|pid| !expired_set.contains(pid));
        for group in &mut self.comet_groups {
            group.planet_ids.retain(|pid| !expired_set.contains(pid));
        }
        self.comet_groups.retain(|g| !g.planet_ids.is_empty());
        self.rebuild_planet_index();
    }

    fn rebuild_planet_index(&mut self) {
        self.planet_index_by_id.clear();
        self.planet_index_by_id.reserve(self.planets.len());
        for (i, p) in self.planets.iter().enumerate() {
            self.planet_index_by_id.insert(p.id, i);
        }
    }

    /// Size scratch buffers to fit the current step's planet and fleet counts,
    /// then zero/clear them. Buffers only grow.
    fn reset_scratch(&mut self, planet_count: usize, fleet_count: usize) {
        if self.planet_paths.len() < planet_count {
            self.planet_paths.resize(planet_count, None);
        }
        for slot in &mut self.planet_paths[..planet_count] {
            *slot = None;
        }

        if self.combat_lists.len() < planet_count {
            self.combat_lists.resize_with(planet_count, Vec::new);
        }
        for list in &mut self.combat_lists[..planet_count] {
            list.clear();
        }

        if self.fleets_to_remove.len() < fleet_count {
            self.fleets_to_remove.resize(fleet_count, false);
        }
        for slot in &mut self.fleets_to_remove[..fleet_count] {
            *slot = false;
        }

        self.comet_id_set.clear();
        self.comet_id_set.extend(self.comet_planet_ids.iter().copied());
    }
}
