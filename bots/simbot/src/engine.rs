//! In-bot clone of the Orbit Wars native engine (`rust_engine/src/lib.rs`).
//!
//! Vendored so the bot can run forward simulations in-process: build an
//! `EngineState` from the observation, push candidate `MoveAction`s, and call
//! `step_with_actions` to score them — no Python round-trip.
//!
//! Kept as a near-verbatim copy of the engine to stay parity-faithful; resync
//! from `rust_engine/src/lib.rs` when that changes. The engine's `#[pymodule]`
//! is intentionally dropped here (the bot's own `lib.rs` owns the module init),
//! and `dead_code` is allowed because the Python-binding helpers and serializers
//! are not all exercised by the bot yet.
#![allow(dead_code)]

use std::f64::consts::PI;
use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sha2::{Digest, Sha512};

use crate::constants::{
    ANG_VEL_MAX, ANG_VEL_MIN, BOARD_SIZE, CENTER, COMET_PRODUCTION, COMET_RADIUS, COMET_SPAWN_STEPS,
    COMET_SPEED, EPISODE_STEPS, MAX_PLANET_GROUPS, MAX_PLAYERS, MAX_SHIP_SPEED, MIN_PLANET_GROUPS,
    MIN_STATIC_GROUPS, PLANET_CLEARANCE, ROTATION_LIMIT, SUN_RADIUS,
};

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

/// Per-section timing accumulators for `step_with_actions`, compiled in only
/// under `--features profile`. Lets us see where simulation time actually goes
/// (orbital math vs collision vs combat vs allocation/finalize) instead of
/// guessing.
#[cfg(feature = "profile")]
pub mod prof {
    use std::cell::RefCell;
    use std::time::Duration;

    pub const N: usize = 8;
    pub const LABELS: [&str; N] = [
        "expired+remove",
        "spawn_comets",
        "moves+prod",
        "orbital",
        "comet_move",
        "fleet+collision",
        "apply+combat",
        "finalize",
    ];

    thread_local! {
        static ACC: RefCell<[Duration; N]> = const { RefCell::new([Duration::ZERO; N]) };
    }

    pub fn add(idx: usize, d: Duration) {
        ACC.with(|a| a.borrow_mut()[idx] += d);
    }
    pub fn reset() {
        ACC.with(|a| *a.borrow_mut() = [Duration::ZERO; N]);
    }
    pub fn snapshot() -> [Duration; N] {
        ACC.with(|a| *a.borrow())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PyRandom {
    mt: [u32; MT_N],
    index: usize,
}

impl PyRandom {
    pub(crate) fn new_from_u64(seed: u64) -> Self {
        let mut key = Vec::new();
        let mut n = seed;
        loop {
            key.push((n & 0xffff_ffff) as u32);
            n >>= 32;
            if n == 0 {
                break;
            }
        }
        Self::init_by_array(&key)
    }

    fn init_by_array(key: &[u32]) -> Self {
        let mut mt = [0u32; MT_N];
        mt[0] = 19_650_218;
        for i in 1..MT_N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }

        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = MT_N.max(key.len());
        while k > 0 {
            mt[i] = (mt[i]
                ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525u32)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                mt[0] = mt[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }

        k = MT_N - 1;
        while k > 0 {
            mt[i] = (mt[i]
                ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941u32)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                mt[0] = mt[MT_N - 1];
                i = 1;
            }
            k -= 1;
        }

        mt[0] = 0x8000_0000;
        Self { mt, index: MT_N }
    }

    fn new_from_big_endian_bytes(bytes: &[u8]) -> Self {
        let mut key = Vec::new();
        let mut end = bytes.len();
        while end > 0 {
            let start = end.saturating_sub(4);
            let mut word = 0u32;
            for &byte in &bytes[start..end] {
                word = (word << 8) | byte as u32;
            }
            key.push(word);
            end = start;
        }
        if key.is_empty() {
            key.push(0);
        }
        Self::init_by_array(&key)
    }

    pub(crate) fn new_from_py_str_seed(seed: &str) -> Self {
        let seed_bytes = seed.as_bytes();
        let mut data = seed_bytes.to_vec();
        let digest = Sha512::digest(seed_bytes);
        data.extend_from_slice(&digest);
        Self::new_from_big_endian_bytes(&data)
    }

    fn gen_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            for kk in 0..(MT_N - MT_M) {
                let y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
                self.mt[kk] =
                    self.mt[kk + MT_M] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
            }
            for kk in (MT_N - MT_M)..(MT_N - 1) {
                let y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
                self.mt[kk] = self.mt[kk + MT_M - MT_N]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
            }
            let y = (self.mt[MT_N - 1] & MT_UPPER_MASK) | (self.mt[0] & MT_LOWER_MASK);
            self.mt[MT_N - 1] =
                self.mt[MT_M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
            self.index = 0;
        }

        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub(crate) fn random(&mut self) -> f64 {
        let a = (self.gen_u32() >> 5) as u64;
        let b = (self.gen_u32() >> 6) as u64;
        ((a << 26) + b) as f64 / 9_007_199_254_740_992.0
    }

    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        let mut bits_left = k;
        let mut out = 0u64;
        let mut offset = 0u32;
        while bits_left >= 32 {
            out |= (self.gen_u32() as u64) << offset;
            bits_left -= 32;
            offset += 32;
        }
        if bits_left > 0 {
            let val = self.gen_u32() >> (32 - bits_left);
            out |= (val as u64) << offset;
        }
        out
    }

    pub(crate) fn randbelow(&mut self, n: u32) -> u32 {
        assert!(n > 0);
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n as u64 {
                return r as u32;
            }
        }
    }

    pub(crate) fn randint(&mut self, a: i64, b: i64) -> i64 {
        assert!(a <= b);
        let span = (b - a + 1) as u32;
        a + self.randbelow(span) as i64
    }
}

#[derive(Clone, Debug)]
pub struct Planet {
    pub id: i64,
    pub owner: i64,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i64,
    pub production: i64,
}

impl Planet {
    pub fn as_tuple(&self) -> (i64, i64, f64, f64, f64, i64, i64) {
        (
            self.id,
            self.owner,
            self.x,
            self.y,
            self.radius,
            self.ships,
            self.production,
        )
    }
}

#[derive(Clone, Debug)]
pub struct Fleet {
    pub id: i64,
    pub owner: i64,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i64,
    pub ships: i64,
}

impl Fleet {
    pub fn as_tuple(&self) -> (i64, i64, f64, f64, f64, i64, i64) {
        (
            self.id,
            self.owner,
            self.x,
            self.y,
            self.angle,
            self.from_planet_id,
            self.ships,
        )
    }
}

#[derive(Clone, Debug)]
pub struct CometGroup {
    pub planet_ids: Vec<i64>,
    pub paths: Vec<Vec<[f64; 2]>>,
    pub path_index: i64,
}

impl CometGroup {
    pub fn as_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        let paths = self
            .paths
            .iter()
            .map(|path| path.iter().map(|pt| (pt[0], pt[1])).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        dict.set_item("planet_ids", self.planet_ids.clone())?;
        dict.set_item("paths", paths)?;
        dict.set_item("path_index", self.path_index)?;
        Ok(dict.into_any().unbind())
    }
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

#[derive(Clone, Copy, Debug)]
pub struct MoveAction {
    pub from_id: i64,
    pub angle: f64,
    pub ships: i64,
}

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
    pub done: bool,
    pub rewards: Option<Vec<f64>>,
    pub seed: u64,
    pub num_players: usize,
    pub configuration: Configuration,
    // Cached `planet.id -> index-in-planets`. Maintained whenever the planets
    // vec is mutated (new / spawn_comets / remove_comets). NEVER read or
    // written elsewhere — keep mutation centralized so it can't drift.
    planet_index_by_id: HashMap<i64, usize>,
    // Bumped whenever `initial_planets` changes (comet spawn / remove). Lets
    // the PyO3 layer reuse a serialized `initial_planets` Python list across
    // the many steps where it doesn't change, instead of rebuilding it every
    // observation + snapshot.
    initial_planets_version: u64,
}

impl EngineState {
    pub fn new(seed: u64, num_players: usize, configuration: Configuration) -> Self {
        let mut rng = PyRandom::new_from_u64(seed);
        let angular_velocity = rng.uniform(ANG_VEL_MIN, ANG_VEL_MAX);
        let mut planets = generate_planets(&mut rng);
        let initial_planets = planets.clone();
        let fleets = Vec::new();
        let next_fleet_id = 0;
        let comet_planet_ids = Vec::new();
        let comets = Vec::new();

        let num_groups = planets.len() / 4;
        if num_groups > 0 {
            let home_group = rng.randint(0, (num_groups - 1) as i64) as usize;
            let base = home_group * 4;
            if num_players == 2 {
                planets[base].owner = 0;
                planets[base].ships = 10;
                planets[base + 3].owner = 1;
                planets[base + 3].ships = 10;
            } else if num_players == 4 {
                for j in 0..4 {
                    planets[base + j].owner = j as i64;
                    planets[base + j].ships = 10;
                }
            }
        }

        let planet_index_by_id = planets
            .iter()
            .enumerate()
            .map(|(idx, planet)| (planet.id, idx))
            .collect();
        Self {
            step: 0,
            angular_velocity,
            planets,
            initial_planets,
            fleets,
            next_fleet_id,
            comet_planet_ids,
            comets,
            done: false,
            rewards: None,
            seed,
            num_players,
            configuration,
            planet_index_by_id,
            initial_planets_version: 0,
        }
    }

    pub fn rebuild_planet_index(&mut self) {
        self.planet_index_by_id.clear();
        self.planet_index_by_id.reserve(self.planets.len());
        for (idx, planet) in self.planets.iter().enumerate() {
            self.planet_index_by_id.insert(planet.id, idx);
        }
    }

    pub fn planet_index_of(&self, planet_id: i64) -> Option<usize> {
        self.planet_index_by_id.get(&planet_id).copied()
    }

    /// Build the `num_players` observation dicts plus the snapshot dict for the
    /// current state, sharing the heavy entity lists (`planets`, `fleets`,
    /// `comets`, `comet_planet_ids`) across every dict instead of rebuilding
    /// them per observation. `initial_obj` is the pre-built `initial_planets`
    /// list, supplied by the caller so it can be cached across steps.
    ///
    /// Observations differ only by the `player` field; the snapshot adds the
    /// engine-private fields. All values are byte-for-byte identical to what
    /// the old per-observation `observation_py` / `snapshot_py` produced.
    pub fn assemble<'py>(
        &self,
        py: Python<'py>,
        initial_obj: &Bound<'py, PyAny>,
    ) -> PyResult<(Vec<Py<PyAny>>, Py<PyAny>)> {
        let planets_obj = PyList::new(py, self.planets.iter().map(Planet::as_tuple))?.into_any();
        let fleets_obj = PyList::new(py, self.fleets.iter().map(Fleet::as_tuple))?.into_any();
        let comets_obj = py_comets(py, &self.comets)?;
        let comet_ids_obj = PyList::new(py, self.comet_planet_ids.iter().copied())?.into_any();

        let mut observations = Vec::with_capacity(self.num_players);
        for player in 0..self.num_players {
            let dict = PyDict::new(py);
            dict.set_item("player", player)?;
            dict.set_item("step", self.step)?;
            dict.set_item("angular_velocity", self.angular_velocity)?;
            dict.set_item("planets", &planets_obj)?;
            dict.set_item("initial_planets", initial_obj)?;
            dict.set_item("fleets", &fleets_obj)?;
            dict.set_item("comets", &comets_obj)?;
            dict.set_item("comet_planet_ids", &comet_ids_obj)?;
            observations.push(dict.into_any().unbind());
        }

        let snapshot = PyDict::new(py);
        snapshot.set_item("step", self.step)?;
        snapshot.set_item("angular_velocity", self.angular_velocity)?;
        snapshot.set_item("planets", &planets_obj)?;
        snapshot.set_item("initial_planets", initial_obj)?;
        snapshot.set_item("fleets", &fleets_obj)?;
        snapshot.set_item("next_fleet_id", self.next_fleet_id)?;
        snapshot.set_item("comet_planet_ids", &comet_ids_obj)?;
        snapshot.set_item("comets", &comets_obj)?;
        snapshot.set_item("done", self.done)?;
        snapshot.set_item("rewards", self.rewards.clone())?;
        snapshot.set_item("seed", self.seed)?;
        snapshot.set_item("configuration", configuration_to_py(py, &self.configuration)?)?;

        Ok((observations, snapshot.into_any().unbind()))
    }

    pub fn snapshot_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("step", self.step)?;
        dict.set_item("angular_velocity", self.angular_velocity)?;
        dict.set_item(
            "planets",
            self.planets.iter().map(Planet::as_tuple).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "initial_planets",
            self.initial_planets
                .iter()
                .map(Planet::as_tuple)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "fleets",
            self.fleets.iter().map(Fleet::as_tuple).collect::<Vec<_>>(),
        )?;
        dict.set_item("next_fleet_id", self.next_fleet_id)?;
        dict.set_item("comet_planet_ids", self.comet_planet_ids.clone())?;
        dict.set_item("comets", py_comets(py, &self.comets)?)?;
        dict.set_item("done", self.done)?;
        dict.set_item("rewards", self.rewards.clone())?;
        dict.set_item("seed", self.seed)?;
        dict.set_item("configuration", configuration_to_py(py, &self.configuration)?)?;
        Ok(dict.into_any().unbind())
    }

    pub fn step_with_actions(&mut self, actions: &[Vec<MoveAction>]) -> Result<bool, String> {
        if self.done {
            return Ok(true);
        }
        if actions.len() != self.num_players {
            return Err(format!(
                "need {} action lists, got {}",
                self.num_players,
                actions.len()
            ));
        }
        #[cfg(feature = "profile")]
        let _p0 = std::time::Instant::now();

        let expired_prelaunch = self.expired_comet_ids();
        if !expired_prelaunch.is_empty() {
            self.remove_comets(&expired_prelaunch);
        }

        #[cfg(feature = "profile")]
        let _ps0 = std::time::Instant::now();

        if COMET_SPAWN_STEPS.contains(&(self.step + 1)) {
            self.spawn_comets();
        }

        #[cfg(feature = "profile")]
        let _ps1 = std::time::Instant::now();

        self.rewards = None;
        for (player_id, action) in actions.iter().enumerate() {
            self.process_moves(player_id as i64, action);
        }

        for planet in &mut self.planets {
            if planet.owner != -1 {
                planet.ships += planet.production;
            }
        }

        let turn_step = self.step;
        let planet_count = self.planets.len();

        #[cfg(feature = "profile")]
        let _p1 = std::time::Instant::now();

        // Per-planet movement path, indexed by current planet position. We
        // build entries for every planet here; the fleet collision loop reads
        // by index via enumerate, so no hash lookups in the hot loop.
        let mut planet_paths: Vec<Option<PlanetPath>> = vec![None; planet_count];

        // Orbital movement for non-comet planets. planets[i].id == initial_planets[i].id
        // is an invariant maintained by spawn_comets/remove_comets, so we can
        // index initial_planets by the same `i`.
        // Membership set built once instead of an O(planets * comet_ids) scan
        // inside the loop. Comet planets are handled separately below.
        let comet_id_set: HashSet<i64> = self.comet_planet_ids.iter().copied().collect();
        for (idx, planet) in self.planets.iter().enumerate() {
            if comet_id_set.contains(&planet.id) {
                continue;
            }
            let old_pos = (planet.x, planet.y);
            let mut new_pos = old_pos;
            let initial_p = &self.initial_planets[idx];
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
            planet_paths[idx] = Some(PlanetPath {
                old_pos,
                new_pos,
                check_collision: true,
            });
        }

        #[cfg(feature = "profile")]
        let _p2 = std::time::Instant::now();

        // Comet movement. Use planet_index_by_id to convert each comet's
        // planet id into a position in self.planets in O(1).
        let mut expired_postmove: Vec<i64> = Vec::new();
        for group in &mut self.comets {
            group.path_index += 1;
            let idx = group.path_index as usize;
            for (i, pid) in group.planet_ids.iter().enumerate() {
                let Some(planet_idx) = self.planet_index_by_id.get(pid).copied() else {
                    continue;
                };
                let planet = &self.planets[planet_idx];
                let old_pos = (planet.x, planet.y);
                let p_path = &group.paths[i];
                if idx >= p_path.len() {
                    expired_postmove.push(*pid);
                    planet_paths[planet_idx] = Some(PlanetPath {
                        old_pos,
                        new_pos: old_pos,
                        check_collision: true,
                    });
                } else {
                    let next = p_path[idx];
                    planet_paths[planet_idx] = Some(PlanetPath {
                        old_pos,
                        new_pos: (next[0], next[1]),
                        check_collision: old_pos.0 >= 0.0,
                    });
                }
            }
        }

        #[cfg(feature = "profile")]
        let _p3 = std::time::Instant::now();

        // Fleet movement + collision detection. fleets_to_remove is a
        // per-fleet bool flag indexed by current fleet position.
        let fleet_count = self.fleets.len();
        let mut fleets_to_remove = vec![false; fleet_count];
        // Combat only needs (owner, ships) per attacker, so store just those two
        // ints rather than cloning the whole Fleet (id/pos/angle are unused here).
        let mut combat_lists: Vec<Vec<(i64, i64)>> = vec![Vec::new(); planet_count];
        for (fleet_idx, fleet) in self.fleets.iter_mut().enumerate() {
            let old_pos = (fleet.x, fleet.y);
            let speed = fleet_speed(fleet.ships, self.configuration.ship_speed);
            fleet.x += fleet.angle.cos() * speed;
            fleet.y += fleet.angle.sin() * speed;
            let new_pos = (fleet.x, fleet.y);

            let mut hit_planet = false;
            for (planet_idx, planet) in self.planets.iter().enumerate() {
                let Some(path) = &planet_paths[planet_idx] else {
                    continue;
                };
                if !path.check_collision {
                    continue;
                }
                if swept_pair_hit(old_pos, new_pos, path.old_pos, path.new_pos, planet.radius) {
                    combat_lists[planet_idx].push((fleet.owner, fleet.ships));
                    fleets_to_remove[fleet_idx] = true;
                    hit_planet = true;
                    break;
                }
            }
            if hit_planet {
                continue;
            }

            if !(0.0..=BOARD_SIZE).contains(&fleet.x) || !(0.0..=BOARD_SIZE).contains(&fleet.y) {
                fleets_to_remove[fleet_idx] = true;
                continue;
            }

            if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS {
                fleets_to_remove[fleet_idx] = true;
                continue;
            }
        }

        #[cfg(feature = "profile")]
        let _p4 = std::time::Instant::now();

        // Apply movement results to planets and resolve combat before any
        // planet-vec mutation, so combat_lists stays aligned with planet
        // positions. Iterating with enumerate gives us the index directly.
        for (idx, planet) in self.planets.iter_mut().enumerate() {
            if let Some(path) = &planet_paths[idx] {
                planet.x = path.new_pos.0;
                planet.y = path.new_pos.1;
            }
        }

        for (idx, planet) in self.planets.iter_mut().enumerate() {
            let planet_fleets = &combat_lists[idx];
            if planet_fleets.is_empty() {
                continue;
            }

            // Sum attacker ships per player into a fixed-size array. The
            // game caps num_players at 4, so no Vec/HashMap allocations.
            let mut player_ships = [0i64; MAX_PLAYERS];
            for &(owner, ships) in planet_fleets {
                if owner >= 0 && (owner as usize) < MAX_PLAYERS {
                    player_ships[owner as usize] += ships;
                }
            }

            // Find top and second by ship count, scanning by ascending
            // player id. Tie-breaking for the "top" identity is irrelevant
            // for the result: when top_ships == second_ships, survivor is
            // forced to (-1, 0); when top_ships > second_ships, the top
            // entry is unique by definition. Matches the previous
            // sort_by-on-HashMap-iter behavior.
            let mut top_player: i64 = -1;
            let mut top_ships: i64 = -1;
            let mut second_ships: i64 = -1;
            let mut entry_count = 0;
            for (player_idx, &ships) in player_ships.iter().enumerate() {
                if ships <= 0 {
                    continue;
                }
                entry_count += 1;
                if ships > top_ships {
                    second_ships = top_ships;
                    top_ships = ships;
                    top_player = player_idx as i64;
                } else if ships > second_ships {
                    second_ships = ships;
                }
            }

            if entry_count == 0 {
                continue;
            }

            let (survivor_owner, survivor_ships) = if entry_count > 1 {
                let survivor_ships = if top_ships == second_ships {
                    0
                } else {
                    top_ships - second_ships
                };
                let survivor_owner = if survivor_ships > 0 { top_player } else { -1 };
                (survivor_owner, survivor_ships)
            } else {
                (top_player, top_ships)
            };

            if survivor_ships > 0 {
                if planet.owner == survivor_owner {
                    planet.ships += survivor_ships;
                } else {
                    planet.ships -= survivor_ships;
                    if planet.ships < 0 {
                        planet.owner = survivor_owner;
                        planet.ships = planet.ships.abs();
                    }
                }
            }
        }

        #[cfg(feature = "profile")]
        let _p5 = std::time::Instant::now();

        // Now that combat is resolved against the build-time planet indexing,
        // mutate the vecs.
        if !expired_postmove.is_empty() {
            self.remove_comets(&expired_postmove);
        }

        // Remove destroyed fleets in place using the per-fleet bool flag.
        let mut retain_idx = 0usize;
        self.fleets.retain(|_| {
            let keep = !fleets_to_remove[retain_idx];
            retain_idx += 1;
            keep
        });

        let mut terminated = turn_step >= self.configuration.episode_steps - 2;
        let mut alive = [false; MAX_PLAYERS];
        for planet in &self.planets {
            if planet.owner >= 0 && (planet.owner as usize) < MAX_PLAYERS {
                alive[planet.owner as usize] = true;
            }
        }
        for fleet in &self.fleets {
            if fleet.owner >= 0 && (fleet.owner as usize) < MAX_PLAYERS {
                alive[fleet.owner as usize] = true;
            }
        }
        let alive_count: usize = alive.iter().filter(|&&b| b).count();
        if alive_count <= 1 {
            terminated = true;
        }

        if terminated {
            let rewards = self.compute_rewards();
            self.rewards = Some(rewards);
            self.done = true;
            self.step = 0;
        } else {
            self.done = false;
            self.step += 1;
        }

        #[cfg(feature = "profile")]
        {
            let _p6 = std::time::Instant::now();
            prof::add(0, _ps0 - _p0); // expired + remove
            prof::add(1, _ps1 - _ps0); // spawn_comets
            prof::add(2, _p1 - _ps1); // moves + production
            prof::add(3, _p2 - _p1); // orbital
            prof::add(4, _p3 - _p2); // comet_move
            prof::add(5, _p4 - _p3); // fleet + collision
            prof::add(6, _p5 - _p4); // apply + combat
            prof::add(7, _p6 - _p5); // finalize
        }

        Ok(self.done)
    }

    pub fn expired_comet_ids(&self) -> Vec<i64> {
        if self.comets.is_empty() {
            return Vec::new();
        }
        let mut expired = Vec::new();
        for group in &self.comets {
            let idx = group.path_index;
            for (i, pid) in group.planet_ids.iter().enumerate() {
                if idx >= group.paths[i].len() as i64 {
                    expired.push(*pid);
                }
            }
        }
        expired
    }

    pub fn remove_comets(&mut self, expired_ids: &[i64]) {
        let expired_set: HashSet<i64> = expired_ids.iter().copied().collect();
        self.planets.retain(|planet| !expired_set.contains(&planet.id));
        self.initial_planets
            .retain(|planet| !expired_set.contains(&planet.id));
        self.comet_planet_ids
            .retain(|pid| !expired_set.contains(pid));
        for group in &mut self.comets {
            group.planet_ids.retain(|pid| !expired_set.contains(pid));
        }
        self.comets.retain(|group| !group.planet_ids.is_empty());
        self.rebuild_planet_index();
        self.initial_planets_version = self.initial_planets_version.wrapping_add(1);
    }

    pub fn spawn_comets(&mut self) {
        let seed = format!("orbit_wars-comet-{}-{}", self.seed, self.step + 1);
        let mut comet_rng = PyRandom::new_from_py_str_seed(&seed);
        let Some(comet_paths) = generate_comet_paths(
            &self.initial_planets,
            self.angular_velocity,
            self.step + 1,
            &self.comet_planet_ids,
            self.configuration.comet_speed,
            &mut comet_rng,
        ) else {
            return;
        };

        let next_id = self
            .planets
            .iter()
            .map(|planet| planet.id)
            .max()
            .unwrap_or(-1)
            + 1;
        let comet_ships = (0..4)
            .map(|_| comet_rng.randint(1, 99))
            .min()
            .unwrap_or(1);
        let mut group = CometGroup {
            planet_ids: Vec::new(),
            paths: comet_paths,
            path_index: -1,
        };
        for i in 0..group.paths.len() {
            let pid = next_id + i as i64;
            group.planet_ids.push(pid);
            self.comet_planet_ids.push(pid);
            let planet = Planet {
                id: pid,
                owner: -1,
                x: -99.0,
                y: -99.0,
                radius: COMET_RADIUS,
                ships: comet_ships,
                production: COMET_PRODUCTION,
            };
            self.planet_index_by_id.insert(pid, self.planets.len());
            self.planets.push(planet.clone());
            self.initial_planets.push(planet);
        }
        self.comets.push(group);
        self.initial_planets_version = self.initial_planets_version.wrapping_add(1);
    }

    pub fn process_moves(&mut self, player_id: i64, action: &[MoveAction]) {
        for move_action in action {
            let Some(from_planet_idx) = self.planet_index_of(move_action.from_id) else {
                continue;
            };
            let from_planet = &mut self.planets[from_planet_idx];
            if from_planet.owner != player_id {
                continue;
            }
            if move_action.ships <= 0 || from_planet.ships < move_action.ships {
                continue;
            }

            from_planet.ships -= move_action.ships;
            let start_x =
                from_planet.x + move_action.angle.cos() * (from_planet.radius + 0.1);
            let start_y =
                from_planet.y + move_action.angle.sin() * (from_planet.radius + 0.1);
            self.fleets.push(Fleet {
                id: self.next_fleet_id,
                owner: player_id,
                x: start_x,
                y: start_y,
                angle: move_action.angle,
                from_planet_id: move_action.from_id,
                ships: move_action.ships,
            });
            self.next_fleet_id += 1;
        }
    }

    pub fn compute_rewards(&self) -> Vec<f64> {
        let mut scores = vec![0i64; self.num_players];
        for planet in &self.planets {
            if planet.owner != -1 {
                scores[planet.owner as usize] += planet.ships;
            }
        }
        for fleet in &self.fleets {
            scores[fleet.owner as usize] += fleet.ships;
        }

        let max_score = *scores.iter().max().unwrap_or(&0);
        scores
            .into_iter()
            .map(|score| if score == max_score && max_score > 0 { 1.0 } else { -1.0 })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlanetPath {
    pub old_pos: (f64, f64),
    pub new_pos: (f64, f64),
    pub check_collision: bool,
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

fn py_comets<'py>(py: Python<'py>, comets: &[CometGroup]) -> PyResult<Bound<'py, PyAny>> {
    let items = comets
        .iter()
        .map(|comet| comet.as_py(py))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

fn configuration_to_py(py: Python<'_>, configuration: &Configuration) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("episodeSteps", configuration.episode_steps)?;
    dict.set_item("actTimeout", configuration.act_timeout)?;
    dict.set_item("shipSpeed", configuration.ship_speed)?;
    dict.set_item("sunRadius", configuration.sun_radius)?;
    dict.set_item("boardSize", configuration.board_size)?;
    dict.set_item("cometSpeed", configuration.comet_speed)?;
    Ok(dict.into_any().unbind())
}

fn configuration_from_py(configuration: Option<&Bound<'_, PyAny>>) -> PyResult<Configuration> {
    let mut parsed = Configuration::default();
    let Some(configuration) = configuration else {
        return Ok(parsed);
    };
    if configuration.is_none() {
        return Ok(parsed);
    }
    let dict = configuration.downcast::<PyDict>()?;
    if let Some(value) = dict.get_item("episodeSteps")? {
        parsed.episode_steps = value.extract()?;
    }
    if let Some(value) = dict.get_item("actTimeout")? {
        parsed.act_timeout = value.extract()?;
    }
    if let Some(value) = dict.get_item("shipSpeed")? {
        parsed.ship_speed = value.extract()?;
    }
    if let Some(value) = dict.get_item("sunRadius")? {
        parsed.sun_radius = value.extract()?;
    }
    if let Some(value) = dict.get_item("boardSize")? {
        parsed.board_size = value.extract()?;
    }
    if let Some(value) = dict.get_item("cometSpeed")? {
        parsed.comet_speed = value.extract()?;
    }
    Ok(parsed)
}

fn py_any_to_f64(value: &Bound<'_, PyAny>) -> Option<f64> {
    value
        .extract::<f64>()
        .ok()
        .or_else(|| value.extract::<i64>().ok().map(|v| v as f64))
}

fn py_any_to_i64(value: &Bound<'_, PyAny>) -> Option<i64> {
    value
        .extract::<i64>()
        .ok()
        .or_else(|| value.extract::<u64>().ok().and_then(|v| i64::try_from(v).ok()))
        .or_else(|| value.extract::<f64>().ok().map(|v| v as i64))
}

fn parse_py_actions(
    actions: &Bound<'_, PyAny>,
    num_players: usize,
) -> PyResult<Vec<Vec<MoveAction>>> {
    let actions_list = actions.downcast::<PyList>()?;
    if actions_list.len() != num_players {
        return Err(PyRuntimeError::new_err(format!(
            "need {num_players} action lists, got {}",
            actions_list.len()
        )));
    }

    let mut parsed = Vec::with_capacity(num_players);
    for player_actions in actions_list.iter() {
        let Ok(moves) = player_actions.downcast::<PyList>() else {
            parsed.push(Vec::new());
            continue;
        };
        let mut parsed_moves = Vec::with_capacity(moves.len());
        for move_value in moves.iter() {
            let Ok(parts) = move_value.downcast::<PyList>() else {
                continue;
            };
            if parts.len() != 3 {
                continue;
            }
            let Some(from_id) = py_any_to_i64(&parts.get_item(0)?) else {
                continue;
            };
            let Some(angle) = py_any_to_f64(&parts.get_item(1)?) else {
                continue;
            };
            let Some(ships) = py_any_to_i64(&parts.get_item(2)?) else {
                continue;
            };
            parsed_moves.push(MoveAction {
                from_id,
                angle,
                ships,
            });
        }
        parsed.push(parsed_moves);
    }

    Ok(parsed)
}

fn generate_planets(rng: &mut PyRandom) -> Vec<Planet> {
    let mut planets: Vec<Planet> = Vec::new();
    let num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS);
    let mut id_counter = 0i64;

    let mut static_groups = 0i64;
    for _ in 0..5000 {
        if static_groups >= MIN_STATIC_GROUPS {
            break;
        }
        let prod = rng.randint(1, 5);
        let r = 1.0 + (prod as f64).ln();
        let angle = rng.uniform(0.0, PI / 2.0);
        let min_orbital = ROTATION_LIMIT - r;
        let max_orbital = (BOARD_SIZE - CENTER - r) / angle.cos().max(angle.sin());
        if min_orbital > max_orbital {
            continue;
        }
        let orbital_r = rng.uniform(min_orbital, max_orbital);
        let x = CENTER + orbital_r * angle.cos();
        let y = CENTER + orbital_r * angle.sin();

        if x + r > BOARD_SIZE || x - r < 0.0 || y + r > BOARD_SIZE || y - r < 0.0 {
            continue;
        }
        if (BOARD_SIZE - x) - r < 0.0 || (BOARD_SIZE - y) - r < 0.0 {
            continue;
        }
        if (x - CENTER) < r + 5.0 || (y - CENTER) < r + 5.0 {
            continue;
        }

        let ships = rng.randint(5, 99).min(rng.randint(5, 99));
        let temp_planets = vec![
            Planet {
                id: id_counter,
                owner: -1,
                x: y,
                y: x,
                radius: r,
                ships,
                production: prod,
            },
            Planet {
                id: id_counter + 1,
                owner: -1,
                x: BOARD_SIZE - x,
                y,
                radius: r,
                ships,
                production: prod,
            },
            Planet {
                id: id_counter + 2,
                owner: -1,
                x,
                y: BOARD_SIZE - y,
                radius: r,
                ships,
                production: prod,
            },
            Planet {
                id: id_counter + 3,
                owner: -1,
                x: BOARD_SIZE - y,
                y: BOARD_SIZE - x,
                radius: r,
                ships,
                production: prod,
            },
        ];

        let mut valid = true;
        for tp in &temp_planets {
            for p in &planets {
                if distance((p.x, p.y), (tp.x, tp.y)) < p.radius + tp.radius + PLANET_CLEARANCE {
                    valid = false;
                    break;
                }
            }
            if !valid {
                break;
            }
        }

        if valid {
            planets.extend(temp_planets);
            id_counter += 4;
            static_groups += 1;
        }
    }

    let mut attempts = 0i64;
    let max_attempts = 5000i64;
    let mut has_orbiting = false;

    while planets.len() < (num_q1 * 4) as usize || (!has_orbiting && attempts < max_attempts) {
        attempts += 1;
        if attempts >= max_attempts {
            break;
        }
        let prod = rng.randint(1, 5);
        let r = 1.0 + (prod as f64).ln();
        let x = rng.uniform(CENTER + 15.0, BOARD_SIZE - r - 5.0);
        let y = rng.uniform(CENTER + 15.0, BOARD_SIZE - r - 5.0);

        let orbital_radius = distance((x, y), (CENTER, CENTER));
        if orbital_radius < SUN_RADIUS + r + 10.0 {
            continue;
        }

        if orbital_radius + r >= ROTATION_LIMIT
            && (x + r > BOARD_SIZE || x - r < 0.0 || y + r > BOARD_SIZE || y - r < 0.0)
        {
            continue;
        }

        let ships = rng.randint(5, 30);
        let temp_planets = vec![
            Planet {
                id: id_counter,
                owner: -1,
                x: y,
                y: x,
                radius: r,
                ships,
                production: prod,
            },
            Planet {
                id: id_counter + 1,
                owner: -1,
                x: BOARD_SIZE - x,
                y,
                radius: r,
                ships,
                production: prod,
            },
            Planet {
                id: id_counter + 2,
                owner: -1,
                x,
                y: BOARD_SIZE - y,
                radius: r,
                ships,
                production: prod,
            },
            Planet {
                id: id_counter + 3,
                owner: -1,
                x: BOARD_SIZE - y,
                y: BOARD_SIZE - x,
                radius: r,
                ships,
                production: prod,
            },
        ];

        let mut valid = true;
        for tp in &temp_planets {
            let tp_orbital = distance((tp.x, tp.y), (CENTER, CENTER));
            let tp_is_rotating = tp_orbital + tp.radius < ROTATION_LIMIT;

            for p in &planets {
                let p_orbital = distance((p.x, p.y), (CENTER, CENTER));
                let p_is_rotating = p_orbital + p.radius < ROTATION_LIMIT;

                if distance((p.x, p.y), (tp.x, tp.y)) < p.radius + tp.radius + PLANET_CLEARANCE {
                    valid = false;
                    break;
                }

                if tp_is_rotating != p_is_rotating
                    && (tp_orbital - p_orbital).abs() < tp.radius + p.radius + PLANET_CLEARANCE
                {
                    valid = false;
                    break;
                }
            }

            if !valid {
                break;
            }
        }

        if valid {
            if orbital_radius + r < ROTATION_LIMIT {
                has_orbiting = true;
            }
            planets.extend(temp_planets);
            id_counter += 4;
        }
    }

    planets
}

fn generate_comet_paths(
    initial_planets: &[Planet],
    angular_velocity: f64,
    spawn_step: i64,
    comet_planet_ids: &[i64],
    comet_speed: f64,
    rng: &mut PyRandom,
) -> Option<Vec<Vec<[f64; 2]>>> {
    let comet_pid_set: HashSet<i64> = comet_planet_ids.iter().copied().collect();
    for _ in 0..300 {
        let eccentricity = rng.uniform(0.75, 0.93);
        let semi_major = rng.uniform(60.0, 150.0);
        let perihelion = semi_major * (1.0 - eccentricity);
        if perihelion < SUN_RADIUS + COMET_RADIUS {
            continue;
        }

        let semi_minor = semi_major * (1.0 - eccentricity * eccentricity).sqrt();
        let focus_c = semi_major * eccentricity;
        let phi = rng.uniform(PI / 6.0, PI / 3.0);
        // phi is constant across the dense loop; computing cos/sin once is
        // bit-identical to recomputing them per point (same pure function).
        let phi_cos = phi.cos();
        let phi_sin = phi.sin();

        let mut dense = Vec::with_capacity(5000);
        let num = 5000usize;
        for i in 0..num {
            let t = 0.3 * PI + 1.4 * PI * i as f64 / (num - 1) as f64;
            let ex = focus_c + semi_major * t.cos();
            let ey = semi_minor * t.sin();
            let x = CENTER + ex * phi_cos - ey * phi_sin;
            let y = CENTER + ex * phi_sin + ey * phi_cos;
            dense.push((x, y));
        }

        let mut path = vec![dense[0]];
        let mut cum = 0.0;
        let mut target = comet_speed;
        for i in 1..dense.len() {
            cum += distance(dense[i], dense[i - 1]);
            if cum >= target {
                path.push(dense[i]);
                target += comet_speed;
            }
        }

        let mut board_start = None;
        let mut board_end = None;
        for (i, &(x, y)) in path.iter().enumerate() {
            if (0.0..=BOARD_SIZE).contains(&x) && (0.0..=BOARD_SIZE).contains(&y) {
                if board_start.is_none() {
                    board_start = Some(i);
                }
                board_end = Some(i);
            }
        }

        let Some(start_idx) = board_start else {
            continue;
        };
        let end_idx = board_end.unwrap();
        let visible = path[start_idx..=end_idx].to_vec();
        if !(5..=40).contains(&visible.len()) {
            continue;
        }

        let paths = vec![
            visible
                .iter()
                .map(|&(x, y)| [y, x])
                .collect::<Vec<_>>(),
            visible
                .iter()
                .map(|&(x, y)| [BOARD_SIZE - x, y])
                .collect::<Vec<_>>(),
            visible
                .iter()
                .map(|&(x, y)| [x, BOARD_SIZE - y])
                .collect::<Vec<_>>(),
            visible
                .iter()
                .map(|&(x, y)| [BOARD_SIZE - y, BOARD_SIZE - x])
                .collect::<Vec<_>>(),
        ];

        let mut static_planets = Vec::new();
        // For orbiting planets, orb_r and init_angle depend only on the planet's
        // (constant) initial position, not on the comet point. Precompute them
        // once here instead of recomputing per comet point below. Computed
        // exactly as the inner loop did (dx*dx form, not `distance`'s powi), so
        // the resulting positions are bit-identical.
        let mut orbiting_planets: Vec<(f64, f64, f64)> = Vec::new(); // (orb_r, init_angle, radius)
        for planet in initial_planets {
            if comet_pid_set.contains(&planet.id) {
                continue;
            }
            let pr = distance((planet.x, planet.y), (CENTER, CENTER));
            if pr + planet.radius < ROTATION_LIMIT {
                let dx = planet.x - CENTER;
                let dy = planet.y - CENTER;
                let orb_r = (dx * dx + dy * dy).sqrt();
                let init_angle = dy.atan2(dx);
                orbiting_planets.push((orb_r, init_angle, planet.radius));
            } else {
                static_planets.push(planet);
            }
        }

        let mut valid = true;
        let buf = COMET_RADIUS + 0.5;
        for (k, &(cx, cy)) in visible.iter().enumerate() {
            if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS {
                valid = false;
                break;
            }

            let sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ];

            for planet in &static_planets {
                for sp in sym_pts {
                    if distance(sp, (planet.x, planet.y)) < planet.radius + buf {
                        valid = false;
                        break;
                    }
                }
                if !valid {
                    break;
                }
            }
            if !valid {
                break;
            }

            let game_step = spawn_step - 1 + k as i64;
            for &(orb_r, init_angle, radius) in &orbiting_planets {
                let cur_angle = init_angle + angular_velocity * game_step as f64;
                let px = CENTER + orb_r * cur_angle.cos();
                let py = CENTER + orb_r * cur_angle.sin();
                for sp in sym_pts {
                    if distance(sp, (px, py)) < radius + COMET_RADIUS {
                        valid = false;
                        break;
                    }
                }
                if !valid {
                    break;
                }
            }
            if !valid {
                break;
            }
        }

        if valid {
            return Some(paths);
        }
    }
    None
}

#[pyclass]
struct RustEngineCore {
    state: Option<EngineState>,
    // Serialized `initial_planets` list cached across steps, tagged with the
    // `initial_planets_version` it was built from. Reused on every step where
    // the version is unchanged (i.e. no comet spawn/remove), which is the
    // common case — only ~5 spawns occur per 500-step episode.
    initial_planets_cache: Option<(u64, Py<PyAny>)>,
}

impl RustEngineCore {
    /// Return the serialized `initial_planets` list, rebuilding it only when the
    /// engine's `initial_planets_version` has advanced since the cached copy.
    fn initial_planets_obj<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let version = self
            .state
            .as_ref()
            .expect("state present")
            .initial_planets_version;
        let stale = match &self.initial_planets_cache {
            Some((cached_version, _)) => *cached_version != version,
            None => true,
        };
        if stale {
            let state = self.state.as_ref().expect("state present");
            let obj = PyList::new(py, state.initial_planets.iter().map(Planet::as_tuple))?
                .into_any();
            self.initial_planets_cache = Some((version, obj.unbind()));
        }
        Ok(self
            .initial_planets_cache
            .as_ref()
            .expect("cache populated")
            .1
            .bind(py)
            .clone())
    }

    /// Build the observation list + snapshot for the current state, reusing the
    /// cached `initial_planets` list.
    fn build_payload(&mut self, py: Python<'_>) -> PyResult<(Vec<Py<PyAny>>, Py<PyAny>)> {
        let initial_obj = self.initial_planets_obj(py)?;
        let state = self.state.as_ref().expect("state present");
        state.assemble(py, &initial_obj)
    }
}

#[pymethods]
impl RustEngineCore {
    #[new]
    fn new() -> Self {
        Self {
            state: None,
            initial_planets_cache: None,
        }
    }

    #[pyo3(signature = (seed, num_players, configuration=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        seed: u64,
        num_players: usize,
        configuration: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if num_players != 2 && num_players != 4 {
            return Err(PyRuntimeError::new_err(format!(
                "orbit_wars supports 2 or 4 players, got {num_players}"
            )));
        }
        let configuration = configuration_from_py(configuration)?;
        let state = EngineState::new(seed, num_players, configuration);
        self.state = Some(state);
        // New game: a stale cache from a prior game could collide on version 0.
        self.initial_planets_cache = None;
        let (observations, snapshot) = self.build_payload(py)?;

        let dict = PyDict::new(py);
        dict.set_item("observations", observations)?;
        dict.set_item("snapshot", snapshot)?;
        Ok(dict.into_any().unbind())
    }

    fn step(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let num_players = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call reset before step"))?
            .num_players;
        let parsed_actions = parse_py_actions(actions, num_players)?;
        let done = {
            let state = self
                .state
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("call reset before step"))?;
            state
                .step_with_actions(&parsed_actions)
                .map_err(PyRuntimeError::new_err)?
        };
        let (observations, snapshot) = self.build_payload(py)?;

        let dict = PyDict::new(py);
        dict.set_item("observations", observations)?;
        dict.set_item("snapshot", snapshot)?;
        dict.set_item("done", done)?;
        Ok(dict.into_any().unbind())
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call reset before snapshot"))?;
        state.snapshot_py(py)
    }
}

// NOTE: the engine's `#[pymodule] fn orbit_wars_rust` is intentionally omitted.
// This is an in-crate module, not its own extension module — the bot's
// `lib.rs` defines the single `#[pymodule]` for this cdylib. `RustEngineCore`
// above is retained as a usable wrapper but is not registered here.
