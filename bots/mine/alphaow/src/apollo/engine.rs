//! The bot's in-process engine: shared simulation data types + pure-math
//! helpers, the lightweight [`EngineState`] container, and [`Simulator`] — the
//! forward simulator the bot actually steps during planning/rollout.
//!
//! [`Simulator`] borrows an `EngineState` (built from a live observation via
//! [`EngineState::from_observation_parts`]) and steps it with reused scratch
//! buffers + precomputed positions, emitting a per-step event log. The full
//! seeded/spawning/reward engine — used only as the differential-test oracle —
//! lives in `tests/reference_engine.rs` and shares these types/helpers so the
//! parity tests stay honest (one definition of `Planet`, `Fleet`, the
//! swept-collision test, etc.).
#![allow(dead_code)]

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::apollo::constants::{
    BOARD_SIZE, CENTER, COMET_SPEED, EPISODE_STEPS, MAX_PLAYERS, MAX_SHIP_SPEED, ROTATION_LIMIT,
    SUN_RADIUS,
};
use crate::apollo::entity_cache::EntityCache;

#[derive(Clone, Copy, Debug)]
pub struct Planet {
    pub id: i64,
    pub owner: i64,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i64,
    pub production: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct Fleet {
    pub id: i64,
    pub owner: i64,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i64,
    pub ships: i64,
}

#[derive(Clone, Debug)]
pub struct CometGroup {
    pub planet_ids: Vec<i64>,
    pub paths: Vec<Vec<[f64; 2]>>,
    pub path_index: i64,
}

#[derive(Clone, Debug)]
pub struct Configuration {
    pub episode_steps: i64,
    pub act_timeout: i64,
    pub ship_speed: f64,
    pub sun_radius: f64,
    pub board_size: f64,
    pub comet_speed: f64,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            episode_steps: EPISODE_STEPS,
            act_timeout: 1,
            ship_speed: MAX_SHIP_SPEED,
            sun_radius: SUN_RADIUS,
            board_size: BOARD_SIZE,
            comet_speed: COMET_SPEED,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveAction {
    pub from_id: i64,
    pub angle: f64,
    pub ships: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct PlanetPath {
    pub old_pos: (f64, f64),
    pub new_pos: (f64, f64),
    pub check_collision: bool,
}

/// Immutable per-turn board snapshot the [`Simulator`] borrows from. Built from
/// a live observation via [`Self::from_observation_parts`]; stepping happens in
/// the simulator, not here.
#[derive(Clone, Debug)]
pub struct EngineState {
    pub step: i64,
    pub angular_velocity: f64,
    pub planets: Vec<Planet>,
    pub initial_planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub next_fleet_id: i64,
    pub comet_planet_ids: Vec<i64>,
    pub comets: Vec<CometGroup>,
    pub num_players: usize,
}

impl EngineState {
    /// Construct from raw observation parts. This is the only constructor used
    /// in production — there is no random map generation here (that lives in
    /// the test oracle).
    #[allow(clippy::too_many_arguments)]
    pub fn from_observation_parts(
        step: i64,
        angular_velocity: f64,
        planets: Vec<Planet>,
        initial_planets: Vec<Planet>,
        fleets: Vec<Fleet>,
        next_fleet_id: i64,
        comet_planet_ids: Vec<i64>,
        comets: Vec<CometGroup>,
        num_players: usize,
    ) -> Self {
        Self {
            step,
            angular_velocity,
            planets,
            initial_planets,
            fleets,
            next_fleet_id,
            comet_planet_ids,
            comets,
            num_players,
        }
    }
}

pub fn distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt()
}

pub fn point_to_segment_distance(p: (f64, f64), v: (f64, f64), w: (f64, f64)) -> f64 {
    let l2 = (v.0 - w.0).powi(2) + (v.1 - w.1).powi(2);
    if l2 == 0.0 {
        return distance(p, v);
    }
    let t = (((p.0 - v.0) * (w.0 - v.0) + (p.1 - v.1) * (w.1 - v.1)) / l2).clamp(0.0, 1.0);
    let projection = (v.0 + t * (w.0 - v.0), v.1 + t * (w.1 - v.1));
    distance(p, projection)
}

/// Squared distance from `p` to segment `v→w`. Same projection math as
/// [`point_to_segment_distance`] but without the final `sqrt`, for callers
/// that only compare against a threshold: `dist < R ⟺ dist_sq < R²` for
/// non-negative `R`, so the boolean is identical save for a 1-ULP knife edge
/// at exactly `R`.
#[inline]
pub fn point_to_segment_distance_sq(p: (f64, f64), v: (f64, f64), w: (f64, f64)) -> f64 {
    let l2 = (v.0 - w.0).powi(2) + (v.1 - w.1).powi(2);
    let (qx, qy) = if l2 == 0.0 {
        (v.0, v.1)
    } else {
        let t = (((p.0 - v.0) * (w.0 - v.0) + (p.1 - v.1) * (w.1 - v.1)) / l2).clamp(0.0, 1.0);
        (v.0 + t * (w.0 - v.0), v.1 + t * (w.1 - v.1))
    };
    let dx = p.0 - qx;
    let dy = p.1 - qy;
    dx * dx + dy * dy
}

/// Axis-aligned bounds `(x_min, x_max, y_min, y_max)` of the segment `old→new`.
#[inline]
pub fn swept_bounds(old: (f64, f64), new: (f64, f64)) -> (f64, f64, f64, f64) {
    let (x_min, x_max) = if old.0 <= new.0 { (old.0, new.0) } else { (new.0, old.0) };
    let (y_min, y_max) = if old.1 <= new.1 { (old.1, new.1) } else { (new.1, old.1) };
    (x_min, x_max, y_min, y_max)
}

pub fn swept_pair_hit(
    a: (f64, f64),
    b: (f64, f64),
    p0: (f64, f64),
    p1: (f64, f64),
    radius: f64,
) -> bool {
    let d0x = a.0 - p0.0;
    let d0y = a.1 - p0.1;
    let dvx = (b.0 - a.0) - (p1.0 - p0.0);
    let dvy = (b.1 - a.1) - (p1.1 - p0.1);
    let a_coeff = dvx * dvx + dvy * dvy;
    let b_coeff = 2.0 * (d0x * dvx + d0y * dvy);
    let c_coeff = d0x * d0x + d0y * d0y - radius * radius;
    if a_coeff < 1e-12 {
        return c_coeff <= 0.0;
    }
    let disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
    if disc < 0.0 {
        return false;
    }
    let sq = disc.sqrt();
    let t1 = (-b_coeff - sq) / (2.0 * a_coeff);
    let t2 = (-b_coeff + sq) / (2.0 * a_coeff);
    t2 >= 0.0 && t1 <= 1.0
}

pub fn fleet_speed(ships: i64, max_speed: f64) -> f64 {
    let speed = 1.0 + (max_speed - 1.0) * ((ships as f64).ln() / 1000.0f64.ln()).powf(1.5);
    speed.min(max_speed)
}

/// Single definition of same-turn planet combat, shared by the forward
/// [`Simulator`] and the per-planet timeline projection in `helpers`.
///
/// `incoming[p]` is the total ships player `p` lands this turn; neutral (`-1`)
/// arrivals aren't representable here and are simply absent (real fleets always
/// have an owner `>= 0`). The top two attackers by ship count cancel; the
/// survivor (`top - second`, ascending-player-id tie-break) then fights the
/// pre-combat `garrison`. A top-two tie neutralises both, leaving the planet
/// untouched. Returns the post-combat `(owner, ships)` (ships always `>= 0`).
#[inline]
pub fn resolve_combat(owner: i64, garrison: i64, incoming: &[i64; MAX_PLAYERS]) -> (i64, i64) {
    let mut top_player: i64 = -1;
    let mut top_ships: i64 = -1;
    let mut second_ships: i64 = -1;
    let mut entry_count = 0;
    for (pidx, &ships) in incoming.iter().enumerate() {
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

    let (survivor_owner, survivor_ships) = if entry_count > 1 {
        let s = top_ships - second_ships;
        let o = if s > 0 { top_player } else { -1 };
        (o, s)
    } else if entry_count == 1 {
        (top_player, top_ships)
    } else {
        (-1, 0)
    };

    if survivor_ships <= 0 {
        return (owner, garrison.max(0));
    }
    if owner == survivor_owner {
        return (owner, garrison + survivor_ships);
    }
    let new_garrison = garrison - survivor_ships;
    if new_garrison < 0 {
        (survivor_owner, -new_garrison)
    } else {
        (owner, new_garrison)
    }
}

// ===========================================================================
// Simulator — the bot's in-process forward/lookahead engine.
// ===========================================================================

/// Predicted landing of one in-flight fleet, with `turns` measured from the
/// simulator's start step.
#[derive(Debug, Clone, Copy)]
pub struct ArrivalEvent {
    pub turns: i64,
    pub owner: i64,
    pub ships: i64,
}

/// Per-step rollout events. `turn` is 1-indexed turns since the simulator was
/// constructed (`turn = 1` means "end of the first stepped turn").
#[derive(Debug, Clone, Copy)]
pub enum StepEvent {
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
struct SimCometGroup<'a> {
    planet_ids: Vec<i64>,
    paths: &'a [Vec<[f64; 2]>],
    path_index: i64,
}

pub struct Simulator<'a> {
    // ── Borrowed from parent (immutable for the lifetime of the simulator).
    initial_by_id: HashMap<i64, &'a Planet>,
    angular_velocity: f64,
    num_players: usize,

    // ── Owned mutable state.
    step: i64,
    initial_step: i64,
    planets: Vec<Planet>,
    fleets: Vec<Fleet>,
    next_fleet_id: i64,
    comet_planet_ids: Vec<i64>,
    comet_groups: Vec<SimCometGroup<'a>>,
    planet_index_by_id: HashMap<i64, usize>,

    // ── Scratch buffers reused across step calls (kept allocated for the
    // simulator's lifetime; cleared, not re-allocated, on each step).
    planet_paths: Vec<Option<PlanetPath>>,
    /// Broad-phase reject cache, parallel to `planet_paths`. Each present entry
    /// is the planet's swept-segment AABB expanded by its radius, stored as
    /// `[x_min, x_max, y_min, y_max]`. A fleet whose own swept AABB is strictly
    /// disjoint from this on any axis cannot collide (see the collision loop),
    /// so the full `swept_pair_hit` is skipped. Computed once per planet per
    /// step; only valid where `planet_paths[i]` is `Some`.
    planet_collision_aabb: Vec<[f64; 4]>,
    fleets_to_remove: Vec<bool>,
    combat_lists: Vec<Vec<(i64, i64)>>,
    comet_id_set: HashSet<i64>,
    expired_postmove: Vec<i64>,

    events: Vec<StepEvent>,
    /// When false, `step_with_actions` skips populating `events`. The rollout
    /// scoring sim turns this off since it scores from planets/fleets and never
    /// reads `events()`/`collect_arrivals()`. `fork()` always re-enables it
    /// because the arrival-ledger walk depends on `FleetLanded` events.
    record_events: bool,
}

impl<'a> Simulator<'a> {
    /// Construct a simulator seeded from `state`. The simulator borrows `state`'s
    /// comet path tables and initial-planet table; the engine reference must
    /// outlive the simulator.
    pub fn new(state: &'a EngineState) -> Self {
        let planet_count = state.planets.len();
        let fleet_count = state.fleets.len();

        let initial_by_id: HashMap<i64, &'a Planet> = state
            .initial_planets
            .iter()
            .map(|p| (p.id, p))
            .collect();

        let comet_groups: Vec<SimCometGroup<'a>> = state
            .comets
            .iter()
            .map(|g| SimCometGroup {
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
            planet_collision_aabb: vec![[0.0; 4]; planet_count],
            fleets_to_remove: vec![false; fleet_count],
            combat_lists: (0..planet_count).map(|_| Vec::new()).collect(),
            comet_id_set: HashSet::with_capacity_and_hasher(
                state.comet_planet_ids.len(),
                Default::default(),
            ),
            expired_postmove: Vec::new(),

            events: Vec::with_capacity(64),
            record_events: true,
        }
    }

    #[inline]
    pub fn planets(&self) -> &[Planet] { &self.planets }
    #[inline]
    pub fn fleets(&self) -> &[Fleet] { &self.fleets }
    #[inline]
    pub fn events(&self) -> &[StepEvent] { &self.events }
    #[inline]
    pub fn angular_velocity(&self) -> f64 { self.angular_velocity }
    #[inline]
    pub fn num_players(&self) -> usize { self.num_players }
    #[inline]
    pub fn comet_planet_ids(&self) -> &[i64] { &self.comet_planet_ids }
    /// Engine step number after the most recent `step()`.
    #[inline]
    pub fn step_count(&self) -> i64 { self.step }
    /// Turns elapsed since `new`.
    #[inline]
    pub fn turns_elapsed(&self) -> i64 { self.step - self.initial_step }

    pub fn clear_events(&mut self) { self.events.clear(); }

    /// Fork a sub-simulator that shares the parent's borrowed comet path tables
    /// and initial-planet table, but owns an independent copy of the mutable
    /// rollout state (planets / fleets / comet groups). Used by
    /// `TimelineCache::build` to walk forward `HORIZON` turns from the parent
    /// simulator's current step without disturbing the parent.
    ///
    /// `initial_step` is reset to the parent's current step so arrival event
    /// turns are measured from the fork point.
    pub fn fork(&self) -> Simulator<'a> {
        let planet_count = self.planets.len();
        let fleet_count = self.fleets.len();
        Simulator {
            initial_by_id: self.initial_by_id.clone(),
            angular_velocity: self.angular_velocity,
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
                .map(|g| SimCometGroup {
                    planet_ids: g.planet_ids.clone(),
                    paths: g.paths,
                    path_index: g.path_index,
                })
                .collect(),
            planet_index_by_id: self.planet_index_by_id.clone(),
            planet_paths: vec![None; planet_count],
            planet_collision_aabb: vec![[0.0; 4]; planet_count],
            fleets_to_remove: vec![false; fleet_count],
            combat_lists: (0..planet_count).map(|_| Vec::new()).collect(),
            comet_id_set: HashSet::with_capacity_and_hasher(
                self.comet_planet_ids.len(),
                Default::default(),
            ),
            expired_postmove: Vec::new(),
            events: Vec::with_capacity(64),
            // Forks drive the arrival-ledger walk, which reads `FleetLanded`.
            record_events: true,
        }
    }

    /// Toggle per-step event recording (default on). Turn off for sims that
    /// only need the final planet/fleet state (the rollout scoring sim); leave
    /// on for anything that calls `events()` or `collect_arrivals()`.
    #[inline]
    pub fn set_record_events(&mut self, on: bool) {
        self.record_events = on;
    }

    /// Step one turn with no player actions. `cache` supplies precomputed
    /// planet positions (see [`step_with_actions`]); pass `None` to recompute
    /// them with trig.
    #[inline]
    pub fn step(&mut self, cache: Option<&EntityCache>) {
        self.step_with_actions(&[], cache);
    }

    /// Step `n` turns with no player actions. Caps at `EPISODE_STEPS` so a
    /// simulator started near the end of the game won't simulate phantom turns
    /// the real game never reaches.
    pub fn step_n(&mut self, n: i64, cache: Option<&EntityCache>) {
        for _ in 0..n.max(0) {
            if self.step >= EPISODE_STEPS {
                break;
            }
            self.step_with_actions(&[], cache);
        }
    }

    /// Step one turn applying `actions[p]` as player `p`'s moves. Players
    /// beyond `actions.len()` take no action. No-op once `self.step` reaches
    /// `EPISODE_STEPS` — the episode is over.
    ///
    /// `cache`, when supplied, provides precomputed planet positions so the
    /// orbital loop reads a table entry instead of recomputing
    /// `sqrt`/`atan2`/`cos`/`sin` every step. Planet motion is a pure function
    /// of `(initial position, angular_velocity, step)` — independent of player
    /// actions — so the cached table is bit-identical to the trig fallback.
    /// Pass `None` to always use trig.
    pub fn step_with_actions(&mut self, actions: &[&[MoveAction]], cache: Option<&EntityCache>) {
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

        // When there are no active comets (the common case between spawn
        // boundaries) every planet is a normal planet, so skip the per-planet
        // comet-set membership test.
        let has_comets = !self.comet_planet_ids.is_empty();

        for (idx, planet) in self.planets.iter().enumerate() {
            if has_comets && self.comet_id_set.contains(&planet.id) {
                continue;
            }
            let old_pos = (planet.x, planet.y);

            // Fast path: read the precomputed position for this turn. The move
            // out of `turn_step` uses orbital factor `turn_step`, which the
            // cache stores at index `turn_step + 1` (its `(t - 1).max(0)`
            // indexing). Falls back to trig with no cache, past the table's end
            // (last turn), or for any planet missing from the cache.
            let new_pos = match cache.and_then(|c| c.position_abs(planet.id, turn_step + 1)) {
                Some(p) => (p[0], p[1]),
                None => {
                    // Lookup by id (not by index) so this still works after
                    // comet expiry has shifted self.planets while initial_by_id
                    // stays put.
                    let initial_p = self
                        .initial_by_id
                        .get(&planet.id)
                        .expect("non-comet planet missing initial entry");
                    let dx = initial_p.x - CENTER;
                    let dy = initial_p.y - CENTER;
                    let orbital_r = (dx * dx + dy * dy).sqrt();
                    if orbital_r + planet.radius < ROTATION_LIMIT {
                        let initial_angle = dy.atan2(dx);
                        let current_angle = initial_angle + self.angular_velocity * turn_step as f64;
                        (
                            CENTER + orbital_r * current_angle.cos(),
                            CENTER + orbital_r * current_angle.sin(),
                        )
                    } else {
                        old_pos
                    }
                }
            };
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

        // Precompute each moving planet's radius-expanded swept AABB once for
        // this step. The collision loop below uses it as a broad-phase reject.
        for (idx, slot) in self.planet_paths[..planet_count].iter().enumerate() {
            if let Some(path) = slot {
                let r = self.planets[idx].radius;
                let (x_min, x_max, y_min, y_max) = swept_bounds(path.old_pos, path.new_pos);
                self.planet_collision_aabb[idx] = [x_min - r, x_max + r, y_min - r, y_max + r];
            }
        }

        for (fleet_idx, fleet) in self.fleets.iter_mut().enumerate() {
            let old_pos = (fleet.x, fleet.y);
            let speed = fleet_speed(fleet.ships, MAX_SHIP_SPEED);
            fleet.x += fleet.angle.cos() * speed;
            fleet.y += fleet.angle.sin() * speed;
            let new_pos = (fleet.x, fleet.y);

            // Fleet's own swept-segment AABB (no expansion — the planet AABB
            // already carries the radius margin).
            let (fx_min, fx_max, fy_min, fy_max) = swept_bounds(old_pos, new_pos);

            let mut hit_planet = false;
            for (planet_idx, planet) in self.planets.iter().enumerate() {
                let Some(path) = &self.planet_paths[planet_idx] else {
                    continue;
                };
                if !path.check_collision {
                    continue;
                }
                // Broad phase: if the fleet's swept AABB is strictly disjoint
                // from the planet's radius-expanded swept AABB on any axis, no
                // same-`s` pair of points can be within `radius`, so
                // `swept_pair_hit` cannot fire. Strict `<`/`>` keeps boundary
                // ties falling through to the exact test below.
                let [px_min, px_max, py_min, py_max] = self.planet_collision_aabb[planet_idx];
                if fx_max < px_min || fx_min > px_max || fy_max < py_min || fy_min > py_max {
                    continue;
                }
                if swept_pair_hit(old_pos, new_pos, path.old_pos, path.new_pos, planet.radius) {
                    self.combat_lists[planet_idx].push((fleet.owner, fleet.ships));
                    self.fleets_to_remove[fleet_idx] = true;
                    if self.record_events {
                        self.events.push(StepEvent::FleetLanded {
                            turn: event_turn,
                            planet_id: planet.id,
                            fleet_owner: fleet.owner,
                            fleet_ships: fleet.ships,
                        });
                    }
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
            if point_to_segment_distance_sq((CENTER, CENTER), old_pos, new_pos)
                < SUN_RADIUS * SUN_RADIUS
            {
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

            let prev_owner = planet.owner;
            let (new_owner, new_ships) = resolve_combat(planet.owner, planet.ships, &player_ships);
            planet.owner = new_owner;
            planet.ships = new_ships;
            if self.record_events && planet.owner != prev_owner {
                self.events.push(StepEvent::OwnerChanged {
                    turn: event_turn,
                    planet_id: planet.id,
                    prev_owner,
                    new_owner: planet.owner,
                    ships: planet.ships,
                });
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
    pub fn step_with_player_actions(
        &mut self,
        player: i64,
        moves: &[MoveAction],
        cache: Option<&EntityCache>,
    ) {
        let empty: &[MoveAction] = &[];
        let mut slots: [&[MoveAction]; MAX_PLAYERS] = [empty; MAX_PLAYERS];
        if player >= 0 && (player as usize) < MAX_PLAYERS {
            slots[player as usize] = moves;
        }
        self.step_with_actions(&slots, cache);
    }

    /// Re-bucket `FleetLanded` events into the per-planet arrival ledger shape
    /// that the bot's strategy layer consumes. `OwnerChanged` events are
    /// ignored — read them separately via `events()`.
    pub fn collect_arrivals(&self) -> HashMap<i64, Vec<ArrivalEvent>> {
        let mut out: HashMap<i64, Vec<ArrivalEvent>> = HashMap::default();
        for ev in &self.events {
            if let StepEvent::FleetLanded {
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

        // Parallel to `planet_paths`; entries are (re)written for every present
        // path before the collision loop reads them, so no per-step clear is
        // needed — only grow to fit.
        if self.planet_collision_aabb.len() < planet_count {
            self.planet_collision_aabb.resize(planet_count, [0.0; 4]);
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
