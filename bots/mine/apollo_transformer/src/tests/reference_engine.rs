//! Reference Orbit Wars engine — the **test-only ground truth** that
//! `Simulator` is validated against.
//!
//! A near-verbatim copy of the canonical engine (`rust_engine/src/lib.rs`),
//! kept deliberately *independent* of the production simulator so the parity
//! tests do real differential testing: a bug in `Simulator` surfaces as a
//! mismatch here, and a bug shared by both can't hide. Resync from
//! `rust_engine` when the engine physics change.
//!
//! Unlike the production path it keeps full-game behavior — seeded RNG, planet
//! generation, comet spawning, rewards, and termination — none of which the bot
//! itself needs. The shared data types and pure-math helpers are reused from
//! [`crate::engine`]; those carry no simulation logic, so sharing them doesn't
//! weaken the differential test. Use [`RefEngine::snapshot`] to seed a
//! [`crate::engine::Simulator`] from a reference board.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use sha2::{Digest, Sha512};

use crate::constants::{
    ANG_VEL_MAX, ANG_VEL_MIN, BOARD_SIZE, CENTER, COMET_PRODUCTION, COMET_RADIUS,
    COMET_SPAWN_STEPS, MAX_PLANET_GROUPS, MAX_PLAYERS, MIN_PLANET_GROUPS, MIN_STATIC_GROUPS,
    PLANET_CLEARANCE, ROTATION_LIMIT, SUN_RADIUS,
};
use crate::engine::{
    distance, fleet_speed, swept_pair_hit, CometGroup, Configuration, EngineState, Fleet,
    MoveAction, Planet, PlanetPath,
};

/// Euclidean distance from point `p` to segment `v→w`. The production engine
/// compares squared distances (`point_to_segment_distance_sq`); this reference
/// re-derivation keeps the explicit `sqrt` form so its sun-collision check is an
/// independent cross-check rather than a call into the code under test.
fn point_to_segment_distance(p: (f64, f64), v: (f64, f64), w: (f64, f64)) -> f64 {
    let l2 = (v.0 - w.0).powi(2) + (v.1 - w.1).powi(2);
    if l2 == 0.0 {
        return distance(p, v);
    }
    let t = (((p.0 - v.0) * (w.0 - v.0) + (p.1 - v.1) * (w.1 - v.1)) / l2).clamp(0.0, 1.0);
    let projection = (v.0 + t * (w.0 - v.0), v.1 + t * (w.1 - v.1));
    distance(p, projection)
}

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

/// Per-section timing accumulators for `step_with_actions`, gated on the
/// `profile` feature. Reveals where simulation time actually goes (orbital
/// vs collision vs combat vs finalize).
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
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525u32)))
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
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941u32)))
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
                self.mt[kk] =
                    self.mt[kk + MT_M - MT_N] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
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
pub struct RefEngine {
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
    // Cached `planet.id -> index-in-planets`. Mutated only on new /
    // spawn_comets / remove_comets — keep centralized so it can't drift.
    planet_index_by_id: HashMap<i64, usize>,
}

impl RefEngine {
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
        }
    }

    /// Snapshot this reference board into a production [`EngineState`] so a
    /// [`crate::engine::Simulator`] can be seeded from it for parity checks.
    pub fn snapshot(&self) -> EngineState {
        EngineState::from_observation_parts(
            self.step,
            self.angular_velocity,
            self.planets.clone(),
            self.initial_planets.clone(),
            self.fleets.clone(),
            self.next_fleet_id,
            self.comet_planet_ids.clone(),
            self.comets.clone(),
            self.num_players,
        )
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

        // Per-planet movement path indexed by planet position so the fleet
        // collision loop can read by enumerate index — no hash lookups in the
        // hot loop.
        let mut planet_paths: Vec<Option<PlanetPath>> = vec![None; planet_count];

        // Invariant: planets[i].id == initial_planets[i].id (maintained by
        // spawn_comets/remove_comets), so initial_planets is indexed by `i`.
        // Comet planets are handled separately below.
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

        // Apply movement and resolve combat before any planet-vec mutation,
        // so combat_lists stays aligned with planet indices.
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

            // Find top and second by ship count. Tie-breaking for "top" is
            // irrelevant: when top == second, survivor is forced to (-1, 0);
            // when top > second, the top entry is unique by definition.
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

        if !expired_postmove.is_empty() {
            self.remove_comets(&expired_postmove);
        }

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
        self.planets
            .retain(|planet| !expired_set.contains(&planet.id));
        self.initial_planets
            .retain(|planet| !expired_set.contains(&planet.id));
        self.comet_planet_ids
            .retain(|pid| !expired_set.contains(pid));
        for group in &mut self.comets {
            group.planet_ids.retain(|pid| !expired_set.contains(pid));
        }
        self.comets.retain(|group| !group.planet_ids.is_empty());
        self.rebuild_planet_index();
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
        let comet_ships = (0..4).map(|_| comet_rng.randint(1, 99)).min().unwrap_or(1);
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
            let start_x = from_planet.x + move_action.angle.cos() * (from_planet.radius + 0.1);
            let start_y = from_planet.y + move_action.angle.sin() * (from_planet.radius + 0.1);
            self.fleets.push(Fleet {
                id: self.next_fleet_id,
                owner: player_id,
                x: start_x,
                y: start_y,
                angle: move_action.angle,
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
            .map(|score| {
                if score == max_score && max_score > 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }
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
        // phi is loop-invariant; hoisting cos/sin is bit-identical.
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
            visible.iter().map(|&(x, y)| [y, x]).collect::<Vec<_>>(),
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
        // orb_r / init_angle depend only on the planet's initial position, so
        // hoist them out of the per-comet-point loop. Uses dx*dx (not
        // `distance`'s powi) to stay bit-identical to the inline form.
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
