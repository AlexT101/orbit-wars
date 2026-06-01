//! Orbit Wars training engine — self-contained simulator with comet/planet
//! generation and configurable reward shaping. Physics ported from
//! `../../rust_engine`; differences:
//!
//! 1. PyO3 class renamed to `OrbitWarsEngine`, module to `orbit_wars_engine`.
//! 2. Reward weights are configurable at construction; `step` returns a
//!    per-player shaped reward plus a breakdown of components for logging.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sha2::{Digest, Sha512};

const BOARD_SIZE: f64 = 100.0;
const CENTER: f64 = BOARD_SIZE / 2.0;
const SUN_RADIUS: f64 = 10.0;
// const SUN_RADIUS: f64 = 50.0; // test validation
const ROTATION_RADIUS_LIMIT: f64 = 50.0;
const COMET_RADIUS: f64 = 1.0;
const COMET_PRODUCTION: i64 = 1;
const PLANET_CLEARANCE: f64 = 7.0;
const MIN_PLANET_GROUPS: i64 = 5;
const MAX_PLANET_GROUPS: i64 = 10;
const MIN_STATIC_GROUPS: i64 = 3;
const COMET_SPAWN_STEPS: [i64; 5] = [50, 150, 250, 350, 450];
const MAX_PLAYERS: usize = 4;

// Python-equivalent Mersenne Twister so seed-derived state matches kaggle's
// reference env bit-for-bit (planet layout + comet spawn).
const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Clone, Debug)]
struct PyRandom {
    mt: [u32; MT_N],
    index: usize,
}

impl PyRandom {
    fn new_from_u64(seed: u64) -> Self {
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

    fn new_from_py_str_seed(seed: &str) -> Self {
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

    fn random(&mut self) -> f64 {
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

    fn randbelow(&mut self, n: u32) -> u32 {
        assert!(n > 0);
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n as u64 {
                return r as u32;
            }
        }
    }

    fn randint(&mut self, a: i64, b: i64) -> i64 {
        assert!(a <= b);
        let span = (b - a + 1) as u32;
        a + self.randbelow(span) as i64
    }
}

#[derive(Clone, Debug)]
struct Planet {
    id: i64,
    owner: i64,
    x: f64,
    y: f64,
    radius: f64,
    ships: i64,
    production: i64,
}

impl Planet {
    fn as_tuple(&self) -> (i64, i64, f64, f64, f64, i64, i64) {
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
struct Fleet {
    id: i64,
    owner: i64,
    x: f64,
    y: f64,
    angle: f64,
    from_planet_id: i64,
    ships: i64,
}

impl Fleet {
    fn as_tuple(&self) -> (i64, i64, f64, f64, f64, i64, i64) {
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
struct CometGroup {
    planet_ids: Vec<i64>,
    paths: Vec<Vec<[f64; 2]>>,
    path_index: i64,
}

#[derive(Clone, Debug)]
struct Configuration {
    episode_steps: i64,
    ship_speed: f64,
    comet_speed: f64,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            episode_steps: 500,
            ship_speed: 6.0,
            comet_speed: 4.0,
        }
    }
}

/// Weights applied per step. Set any to 0.0 to disable that term.
#[derive(Clone, Debug)]
struct RewardWeights {
    /// Terminal centered ships-share: own_ships / all_players_ships - 1/N.
    /// This keeps self-play rewards zero-sum while still rewarding ending with
    /// a larger share of the board's ships.
    terminal: f64,
    /// Terminal time bonus: zero-sum outcome × remaining_turns/episode_steps.
    /// For two players this is `+` for the winner and `-` for the loser; ties
    /// are zero. Rewards fast wins / harshens fast losses.
    terminal_time: f64,
    /// Per-step shaping: weight × centered absolute production, where centered
    /// production = own_production - mean_player_production. This is a small
    /// dense zero-sum nudge toward holding productive planets.
    production_income: f64,
    /// Per-successful-launch shaping. This is intentionally non-zero-sum:
    /// each launched fleet costs the launching player a tiny amount.
    launch_penalty: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            terminal: 1.0,
            terminal_time: 1.0,
            production_income: 0.0002,
            launch_penalty: -0.00004,
            // launch_penalty: -9.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MoveAction {
    from_id: i64,
    angle: f64,
    ships: i64,
}

#[derive(Clone, Debug)]
struct EngineState {
    step: i64,
    angular_velocity: f64,
    planets: Vec<Planet>,
    initial_planets: Vec<Planet>,
    fleets: Vec<Fleet>,
    next_fleet_id: i64,
    comet_planet_ids: Vec<i64>,
    comets: Vec<CometGroup>,
    done: bool,
    seed: u64,
    num_players: usize,
    configuration: Configuration,
    reward_weights: RewardWeights,
    planet_index_by_id: HashMap<i64, usize>,
    /// Last step's per-player reward & component breakdown, exposed to
    /// Python after step.
    last_reward: Vec<f64>,
    last_components: RewardComponents,
    /// Scratch buffer reused across steps to hold the current per-player
    /// production count (used for the production_income term). Pre-allocated to
    /// skip the per-step `Vec::with_capacity`.
    scratch_production: Vec<i64>,
}

#[derive(Clone, Debug, Default)]
struct RewardComponents {
    terminal: Vec<f64>,
    terminal_time: Vec<f64>,
    production_income: Vec<f64>,
    launch_penalty: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct PlanetPath {
    old_pos: (f64, f64),
    new_pos: (f64, f64),
    check_collision: bool,
}

impl EngineState {
    fn new(
        seed: u64,
        num_players: usize,
        configuration: Configuration,
        reward_weights: RewardWeights,
    ) -> Self {
        let mut rng = PyRandom::new_from_u64(seed);
        let angular_velocity = rng.uniform(0.025, 0.05);
        let mut planets = generate_planets(&mut rng);
        let initial_planets = planets.clone();

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
            .map(|(idx, p)| (p.id, idx))
            .collect();

        let s = Self {
            step: 0,
            angular_velocity,
            planets,
            initial_planets,
            fleets: Vec::new(),
            next_fleet_id: 0,
            comet_planet_ids: Vec::new(),
            comets: Vec::new(),
            done: false,
            seed,
            num_players,
            configuration,
            reward_weights,
            planet_index_by_id,
            last_reward: vec![0.0; num_players],
            last_components: RewardComponents {
                terminal: vec![0.0; num_players],
                terminal_time: vec![0.0; num_players],
                production_income: vec![0.0; num_players],
                launch_penalty: vec![0.0; num_players],
            },
            scratch_production: vec![0; num_players],
        };
        s
    }

    fn rebuild_planet_index(&mut self) {
        self.planet_index_by_id.clear();
        self.planet_index_by_id.reserve(self.planets.len());
        for (idx, p) in self.planets.iter().enumerate() {
            self.planet_index_by_id.insert(p.id, idx);
        }
    }

    fn step_with_actions(&mut self, actions: &[Vec<MoveAction>]) -> Result<bool, String> {
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

        let expired_prelaunch = self.expired_comet_ids();
        if !expired_prelaunch.is_empty() {
            self.remove_comets(&expired_prelaunch);
        }

        if COMET_SPAWN_STEPS.contains(&(self.step + 1)) {
            self.spawn_comets();
        }

        let mut launches_by_player = vec![0usize; self.num_players];
        for (player_id, action) in actions.iter().enumerate() {
            launches_by_player[player_id] = self.process_moves(player_id as i64, action);
        }

        for planet in &mut self.planets {
            if planet.owner != -1 {
                planet.ships += planet.production;
            }
        }

        let turn_step = self.step;
        let planet_count = self.planets.len();
        let mut planet_paths: Vec<Option<PlanetPath>> = vec![None; planet_count];

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
            if orbital_r + planet.radius < ROTATION_RADIUS_LIMIT {
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

        let fleet_count = self.fleets.len();
        let mut fleets_to_remove = vec![false; fleet_count];
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
            let mut player_ships = [0i64; MAX_PLAYERS];
            for &(owner, ships) in planet_fleets {
                if owner >= 0 && (owner as usize) < MAX_PLAYERS {
                    player_ships[owner as usize] += ships;
                }
            }
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

        // ---- reward computation ---------------------------------------
        // Recompute production per player into the scratch buffer (no alloc).
        for p in self.scratch_production.iter_mut() {
            *p = 0;
        }
        for planet in &self.planets {
            if planet.owner >= 0 && (planet.owner as usize) < self.num_players {
                self.scratch_production[planet.owner as usize] += planet.production;
            }
        }

        // Total player production this step, for the production_income term.
        let total_production: i64 = self.scratch_production[..self.num_players].iter().sum();

        // Compute terminal shares / outcomes (if any) before mutably borrowing
        // last_components.
        let terminal_info: Option<(Vec<f64>, Vec<f64>, f64)> = if terminated {
            let (shares, outcomes) = self.terminal_shares_and_outcomes();
            // Turns left when the game ends; full at turn 0, ~0 at the cap.
            let remaining = (self.configuration.episode_steps - turn_step).max(0) as f64;
            let remaining_frac =
                (remaining / self.configuration.episode_steps.max(1) as f64).clamp(0.0, 1.0);
            Some((shares, outcomes, remaining_frac))
        } else {
            None
        };

        let n = self.num_players;
        let baseline = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        let mean_production = if n > 0 {
            total_production as f64 / n as f64
        } else {
            0.0
        };
        let production_w = self.reward_weights.production_income;
        let launch_w = self.reward_weights.launch_penalty;
        let term_w = self.reward_weights.terminal;
        let term_t_w = self.reward_weights.terminal_time;
        let c = &mut self.last_components;
        for i in 0..n {
            let centered_production = self.scratch_production[i] as f64 - mean_production;
            c.production_income[i] = production_w * centered_production;
            c.launch_penalty[i] = launch_w * launches_by_player[i] as f64;
            c.terminal[i] = 0.0;
            c.terminal_time[i] = 0.0;
        }
        if let Some((shares, outcomes, remaining_frac)) = terminal_info {
            for i in 0..n {
                c.terminal[i] = term_w * (shares[i] - baseline);
                c.terminal_time[i] = term_t_w * outcomes[i] * remaining_frac;
            }
        }
        for i in 0..n {
            self.last_reward[i] =
                c.terminal[i] + c.terminal_time[i] + c.production_income[i] + c.launch_penalty[i];
        }

        self.done = terminated;
        self.step += 1;
        Ok(self.done)
    }

    /// Terminal reward inputs per player: `(ships_share, outcome)`.
    /// `score` = total ships a player controls (planets + in-flight fleets).
    /// `ships_share` = own_score / Σ all players' scores (0 if no player ships).
    /// `outcome` is zero-sum: tied games are all zero; otherwise the winner set
    /// sums to +1 and the loser set sums to -1.
    fn terminal_shares_and_outcomes(&self) -> (Vec<f64>, Vec<f64>) {
        let mut scores = vec![0i64; self.num_players];
        for planet in &self.planets {
            if planet.owner >= 0 && (planet.owner as usize) < self.num_players {
                scores[planet.owner as usize] += planet.ships;
            }
        }
        for fleet in &self.fleets {
            if fleet.owner >= 0 && (fleet.owner as usize) < self.num_players {
                scores[fleet.owner as usize] += fleet.ships;
            }
        }
        let total: i64 = scores.iter().sum();
        let max_score = *scores.iter().max().unwrap_or(&0);
        let baseline = if self.num_players > 0 {
            1.0 / self.num_players as f64
        } else {
            0.0
        };
        let shares = scores
            .iter()
            .map(|&s| {
                if total > 0 {
                    s as f64 / total as f64
                } else {
                    baseline
                }
            })
            .collect();
        let mut outcomes = vec![0.0; self.num_players];
        if max_score > 0 {
            let winners: Vec<usize> = scores
                .iter()
                .enumerate()
                .filter_map(|(i, &s)| if s == max_score { Some(i) } else { None })
                .collect();
            let loser_count = self.num_players.saturating_sub(winners.len());
            if !winners.is_empty() && loser_count > 0 {
                let winner_value = 1.0 / winners.len() as f64;
                let loser_value = -1.0 / loser_count as f64;
                for &i in &winners {
                    outcomes[i] = winner_value;
                }
                for (i, outcome) in outcomes.iter_mut().enumerate() {
                    if !winners.contains(&i) {
                        *outcome = loser_value;
                    }
                }
            }
        }
        (shares, outcomes)
    }

    fn expired_comet_ids(&self) -> Vec<i64> {
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

    fn remove_comets(&mut self, expired_ids: &[i64]) {
        let expired_set: HashSet<i64> = expired_ids.iter().copied().collect();
        self.planets.retain(|p| !expired_set.contains(&p.id));
        self.initial_planets
            .retain(|p| !expired_set.contains(&p.id));
        self.comet_planet_ids
            .retain(|pid| !expired_set.contains(pid));
        for group in &mut self.comets {
            group.planet_ids.retain(|pid| !expired_set.contains(pid));
        }
        self.comets.retain(|g| !g.planet_ids.is_empty());
        self.rebuild_planet_index();
    }

    fn spawn_comets(&mut self) {
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
        let next_id = self.planets.iter().map(|p| p.id).max().unwrap_or(-1) + 1;
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

    fn process_moves(&mut self, player_id: i64, action: &[MoveAction]) -> usize {
        let mut launches = 0usize;
        for mv in action {
            let Some(idx) = self.planet_index_by_id.get(&mv.from_id).copied() else {
                continue;
            };
            let p = &mut self.planets[idx];
            if p.owner != player_id {
                continue;
            }
            if mv.ships <= 0 || p.ships < mv.ships {
                continue;
            }
            p.ships -= mv.ships;
            let start_x = p.x + mv.angle.cos() * (p.radius + 0.1);
            let start_y = p.y + mv.angle.sin() * (p.radius + 0.1);
            self.fleets.push(Fleet {
                id: self.next_fleet_id,
                owner: player_id,
                x: start_x,
                y: start_y,
                angle: mv.angle,
                from_planet_id: mv.from_id,
                ships: mv.ships,
            });
            self.next_fleet_id += 1;
            launches += 1;
        }
        launches
    }
}

fn distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt()
}

fn point_to_segment_distance(p: (f64, f64), v: (f64, f64), w: (f64, f64)) -> f64 {
    let l2 = (v.0 - w.0).powi(2) + (v.1 - w.1).powi(2);
    if l2 == 0.0 {
        return distance(p, v);
    }
    let t = (((p.0 - v.0) * (w.0 - v.0) + (p.1 - v.1) * (w.1 - v.1)) / l2).clamp(0.0, 1.0);
    let projection = (v.0 + t * (w.0 - v.0), v.1 + t * (w.1 - v.1));
    distance(p, projection)
}

fn swept_pair_hit(
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

fn fleet_speed(ships: i64, max_speed: f64) -> f64 {
    let speed = 1.0 + (max_speed - 1.0) * ((ships as f64).ln() / 1000.0f64.ln()).powf(1.5);
    speed.min(max_speed)
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
        let min_orbital = ROTATION_RADIUS_LIMIT - r;
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
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT
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
            let tp_is_rotating = tp_orbital + tp.radius < ROTATION_RADIUS_LIMIT;
            for p in &planets {
                let p_orbital = distance((p.x, p.y), (CENTER, CENTER));
                let p_is_rotating = p_orbital + p.radius < ROTATION_RADIUS_LIMIT;
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
            if orbital_radius + r < ROTATION_RADIUS_LIMIT {
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
        let mut orbiting_planets: Vec<(f64, f64, f64)> = Vec::new();
        for planet in initial_planets {
            if comet_pid_set.contains(&planet.id) {
                continue;
            }
            let pr = distance((planet.x, planet.y), (CENTER, CENTER));
            if pr + planet.radius < ROTATION_RADIUS_LIMIT {
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

// ---- PyO3 parsing helpers -------------------------------------------------

fn py_any_to_f64(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value
        .extract::<f64>()
        .or_else(|_| value.extract::<i64>().map(|v| v as f64))
}

fn py_any_to_i64(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    value
        .extract::<i64>()
        .or_else(|_| value.extract::<f64>().map(|v| v as i64))
}

fn parse_configuration(value: Option<&Bound<'_, PyAny>>) -> PyResult<Configuration> {
    let mut cfg = Configuration::default();
    let Some(v) = value else { return Ok(cfg) };
    if v.is_none() {
        return Ok(cfg);
    }
    let d = v.downcast::<PyDict>()?;
    if let Some(x) = d.get_item("episodeSteps")? {
        cfg.episode_steps = x.extract()?;
    }
    if let Some(x) = d.get_item("shipSpeed")? {
        cfg.ship_speed = x.extract()?;
    }
    if let Some(x) = d.get_item("cometSpeed")? {
        cfg.comet_speed = x.extract()?;
    }
    Ok(cfg)
}

fn parse_reward_weights(value: Option<&Bound<'_, PyAny>>) -> PyResult<RewardWeights> {
    let mut w = RewardWeights::default();
    let Some(v) = value else { return Ok(w) };
    if v.is_none() {
        return Ok(w);
    }
    let d = v.downcast::<PyDict>()?;
    if let Some(x) = d.get_item("terminal")? {
        w.terminal = x.extract()?;
    }
    if let Some(x) = d.get_item("terminal_time")? {
        w.terminal_time = x.extract()?;
    }
    if d.get_item("production_share")?.is_some() {
        return Err(PyRuntimeError::new_err(
            "reward weight 'production_share' was removed; use 'production_income'",
        ));
    }
    if let Some(x) = d.get_item("production_income")? {
        w.production_income = x.extract()?;
    }
    if let Some(x) = d.get_item("launch_penalty")? {
        w.launch_penalty = x.extract()?;
    }
    Ok(w)
}

fn parse_actions(actions: &Bound<'_, PyAny>, num_players: usize) -> PyResult<Vec<Vec<MoveAction>>> {
    let actions_list = actions.downcast::<PyList>()?;
    if actions_list.len() != num_players {
        return Err(PyRuntimeError::new_err(format!(
            "need {num_players} action lists, got {}",
            actions_list.len()
        )));
    }
    let mut out = Vec::with_capacity(num_players);
    for player_actions in actions_list.iter() {
        let Ok(moves) = player_actions.downcast::<PyList>() else {
            out.push(Vec::new());
            continue;
        };
        let mut parsed = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            let Ok(parts) = mv.downcast::<PyList>() else {
                continue;
            };
            if parts.len() != 3 {
                continue;
            }
            let from_id = py_any_to_i64(&parts.get_item(0)?)?;
            let angle = py_any_to_f64(&parts.get_item(1)?)?;
            let ships = py_any_to_i64(&parts.get_item(2)?)?;
            parsed.push(MoveAction {
                from_id,
                angle,
                ships,
            });
        }
        out.push(parsed);
    }
    Ok(out)
}

// ---- PyO3 serialization helpers ------------------------------------------

fn py_comets<'py>(py: Python<'py>, comets: &[CometGroup]) -> PyResult<Bound<'py, PyAny>> {
    let items: PyResult<Vec<Py<PyAny>>> = comets
        .iter()
        .map(|c| {
            let d = PyDict::new(py);
            d.set_item("planet_ids", c.planet_ids.clone())?;
            let paths: Vec<Vec<(f64, f64)>> = c
                .paths
                .iter()
                .map(|p| p.iter().map(|pt| (pt[0], pt[1])).collect())
                .collect();
            d.set_item("paths", paths)?;
            d.set_item("path_index", c.path_index)?;
            Ok::<Py<PyAny>, PyErr>(d.into_any().unbind())
        })
        .collect();
    Ok(PyList::new(py, items?)?.into_any())
}

fn build_observation<'py>(
    py: Python<'py>,
    state: &EngineState,
    player: usize,
) -> PyResult<Py<PyAny>> {
    let planets_obj = PyList::new(py, state.planets.iter().map(Planet::as_tuple))?.into_any();
    let initial_obj =
        PyList::new(py, state.initial_planets.iter().map(Planet::as_tuple))?.into_any();
    let fleets_obj = PyList::new(py, state.fleets.iter().map(Fleet::as_tuple))?.into_any();
    let comets_obj = py_comets(py, &state.comets)?;
    let comet_ids_obj = PyList::new(py, state.comet_planet_ids.iter().copied())?.into_any();
    let d = PyDict::new(py);
    d.set_item("player", player)?;
    d.set_item("step", state.step)?;
    d.set_item("angular_velocity", state.angular_velocity)?;
    d.set_item("planets", planets_obj)?;
    d.set_item("initial_planets", initial_obj)?;
    d.set_item("fleets", fleets_obj)?;
    d.set_item("comets", comets_obj)?;
    d.set_item("comet_planet_ids", comet_ids_obj)?;
    d.set_item("next_fleet_id", state.next_fleet_id)?;
    Ok(d.into_any().unbind())
}

fn reward_components_to_py<'py>(py: Python<'py>, c: &RewardComponents) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("terminal", c.terminal.clone())?;
    d.set_item("terminal_time", c.terminal_time.clone())?;
    d.set_item("production_income", c.production_income.clone())?;
    d.set_item("launch_penalty", c.launch_penalty.clone())?;
    Ok(d.into_any().unbind())
}

// ---- Public Python class -------------------------------------------------

#[pyclass]
struct OrbitWarsEngine {
    state: Option<EngineState>,
    default_num_players: usize,
    default_config: Configuration,
    default_weights: RewardWeights,
}

#[pymethods]
impl OrbitWarsEngine {
    #[new]
    #[pyo3(signature = (num_players=2, configuration=None, reward_weights=None))]
    fn new(
        num_players: usize,
        configuration: Option<&Bound<'_, PyAny>>,
        reward_weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if num_players != 2 && num_players != 4 {
            return Err(PyRuntimeError::new_err(format!(
                "num_players must be 2 or 4, got {num_players}"
            )));
        }
        Ok(Self {
            state: None,
            default_num_players: num_players,
            default_config: parse_configuration(configuration)?,
            default_weights: parse_reward_weights(reward_weights)?,
        })
    }

    /// Reset to a fresh game. `seed` controls planet/comet RNG (kaggle-equivalent).
    /// Returns `{observations: [...]}`.
    #[pyo3(signature = (seed))]
    fn reset(&mut self, py: Python<'_>, seed: u64) -> PyResult<Py<PyAny>> {
        let state = EngineState::new(
            seed,
            self.default_num_players,
            self.default_config.clone(),
            self.default_weights.clone(),
        );
        self.state = Some(state);
        let state = self.state.as_ref().expect("state present");
        let mut observations = Vec::with_capacity(state.num_players);
        for p in 0..state.num_players {
            observations.push(build_observation(py, state, p)?);
        }
        let d = PyDict::new(py);
        d.set_item("observations", observations)?;
        Ok(d.into_any().unbind())
    }

    /// Step the engine. Returns:
    ///   {
    ///     observations: [obs_p0, obs_p1, ...],
    ///     done: bool,
    ///     reward: [f64; num_players],          # weighted scalar per player
    ///     reward_components: {                 # for logging / debugging
    ///         terminal:           [f64; n],
    ///         terminal_time:      [f64; n],
    ///         production_income:  [f64; n],
    ///         launch_penalty:     [f64; n],
    ///     },
    ///   }
    fn step(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let np = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call reset before step"))?
            .num_players;
        let parsed = parse_actions(actions, np)?;
        let done = {
            let s = self.state.as_mut().expect("state present");
            s.step_with_actions(&parsed)
                .map_err(PyRuntimeError::new_err)?
        };
        let state = self.state.as_ref().expect("state present");
        let mut observations = Vec::with_capacity(np);
        for p in 0..np {
            observations.push(build_observation(py, state, p)?);
        }
        let d = PyDict::new(py);
        d.set_item("observations", observations)?;
        d.set_item("done", done)?;
        d.set_item("reward", state.last_reward.clone())?;
        d.set_item(
            "reward_components",
            reward_components_to_py(py, &state.last_components)?,
        )?;
        Ok(d.into_any().unbind())
    }

    /// Step without returning anything (no dict allocation). Used for
    /// timing the engine work in isolation; reward is still computed and
    /// available via `last_reward` / `last_components` properties below.
    fn step_silent(&mut self, actions: &Bound<'_, PyAny>) -> PyResult<bool> {
        let np = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call reset before step"))?
            .num_players;
        let parsed = parse_actions(actions, np)?;
        let s = self.state.as_mut().expect("state present");
        s.step_with_actions(&parsed)
            .map_err(PyRuntimeError::new_err)
    }

    #[getter]
    fn last_reward(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .state
            .as_ref()
            .map(|s| s.last_reward.clone())
            .unwrap_or_default())
    }

    /// Faster variant that skips per-player observation dicts.
    fn step_fast(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let np = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call reset before step"))?
            .num_players;
        let parsed = parse_actions(actions, np)?;
        let done = {
            let s = self.state.as_mut().expect("state present");
            s.step_with_actions(&parsed)
                .map_err(PyRuntimeError::new_err)?
        };
        let state = self.state.as_ref().expect("state present");
        let d = PyDict::new(py);
        d.set_item("done", done)?;
        d.set_item("reward", state.last_reward.clone())?;
        Ok(d.into_any().unbind())
    }

    /// Snapshot of the current state (no rewards — those are returned by step).
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let s = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call reset first"))?;
        let d = PyDict::new(py);
        d.set_item("step", s.step)?;
        d.set_item("angular_velocity", s.angular_velocity)?;
        d.set_item(
            "planets",
            s.planets.iter().map(Planet::as_tuple).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "initial_planets",
            s.initial_planets
                .iter()
                .map(Planet::as_tuple)
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "fleets",
            s.fleets.iter().map(Fleet::as_tuple).collect::<Vec<_>>(),
        )?;
        d.set_item("next_fleet_id", s.next_fleet_id)?;
        d.set_item("comet_planet_ids", s.comet_planet_ids.clone())?;
        d.set_item("comets", py_comets(py, &s.comets)?)?;
        d.set_item("done", s.done)?;
        d.set_item("seed", s.seed)?;
        Ok(d.into_any().unbind())
    }

    /// Replace the reward weights without restarting the game.
    fn set_reward_weights(&mut self, reward_weights: &Bound<'_, PyAny>) -> PyResult<()> {
        let w = parse_reward_weights(Some(reward_weights))?;
        if let Some(s) = self.state.as_mut() {
            s.reward_weights = w.clone();
        }
        self.default_weights = w;
        Ok(())
    }

    #[getter]
    fn done(&self) -> PyResult<bool> {
        Ok(self.state.as_ref().map(|s| s.done).unwrap_or(false))
    }

    #[getter]
    fn step_count(&self) -> PyResult<i64> {
        Ok(self.state.as_ref().map(|s| s.step).unwrap_or(0))
    }
}

#[pymodule]
fn orbit_wars_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OrbitWarsEngine>()?;
    Ok(())
}
