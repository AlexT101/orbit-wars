//! Game-state simulator. Mirrors the orbit_wars engine's per-turn flow:
//! launches → production → fleet movement (swept-pair collisions) → planet
//! /comet movement → combat resolution → comet expiration.
//!
//! Comet *spawning* (at steps 50/150/.../450) is RNG-driven in the real engine
//! and its location is unobservable from a game state, so forward simulation
//! deliberately does not attempt it.

use crate::pathing::{fleet_speed, point_to_segment_distance, swept_pair_hit};
use crate::{Fleet, GameState, BOARD_SIZE, CENTER_X, CENTER_Y, SUN_RADIUS};
use std::collections::HashMap;

/// (source_planet_id, angle, ships, owner).
pub type Action = (i64, f64, i64, i32);

pub fn apply_launches(state: &mut GameState, actions: &[Action]) {
    let mut next_id = state.fleets.iter().map(|f| f.id).max().unwrap_or(-1) + 1;
    for &(from_id, angle, ships, owner) in actions {
        if ships <= 0 {
            continue;
        }
        let p_idx = match state.planets.iter().position(|p| p.id == from_id) {
            Some(i) => i,
            None => continue,
        };
        let p = &mut state.planets[p_idx];
        if p.owner != owner || p.ships < ships {
            continue;
        }
        p.ships -= ships;
        let r = p.radius + 0.1;
        let start_x = p.x + angle.cos() * r;
        let start_y = p.y + angle.sin() * r;
        state.fleets.push(Fleet {
            id: next_id,
            owner,
            x: start_x,
            y: start_y,
            angle,
            from_planet_id: from_id,
            ships,
        });
        next_id += 1;
    }
}

/// Advance state by one engine turn (after launches have already been applied).
///
/// Comet-free and fully deterministic: the real engine spawns new comets at
/// fixed steps but at RNG-determined, unobservable locations, so forward
/// simulation never invents them (see module docs). Comets already on the board
/// still move and expire.
pub fn tick(state: &mut GameState) {
    // 1. Production
    for p in state.planets.iter_mut() {
        if p.owner != -1 {
            p.ships += p.production;
        }
    }

    // 2. Pre-compute planet/comet motion segments for swept-pair collisions.
    let max_speed = state.max_speed;
    let mut planet_paths: HashMap<i64, ((f64, f64), (f64, f64))> = HashMap::new();
    // Static + orbiting planets (use planet_pos_at for dt=1, where dt=1 gives end-of-this-turn position)
    for p in &state.planets {
        if p.is_comet {
            continue;
        }
        let old_pos = (p.x, p.y);
        let new_pos = state.planet_pos_at(p, 1).unwrap_or(old_pos);
        planet_paths.insert(p.id, (old_pos, new_pos));
    }
    // Comets: increment path_index and move
    let mut comet_expired: Vec<i64> = Vec::new();
    for group in state.comets.iter_mut() {
        group.path_index += 1;
        let idx = group.path_index;
        for (i, pid) in group.planet_ids.iter().enumerate() {
            let planet = match state.planets.iter().find(|p| p.id == *pid) {
                Some(p) => p,
                None => continue,
            };
            let old_pos = (planet.x, planet.y);
            if idx < 0 || idx as usize >= group.paths[i].len() {
                comet_expired.push(*pid);
                planet_paths.insert(*pid, (old_pos, old_pos));
            } else {
                let np = group.paths[i][idx as usize];
                planet_paths.insert(*pid, (old_pos, np));
            }
        }
    }

    // 3. Fleet movement with swept-pair collision detection.
    let mut combat: HashMap<i64, Vec<Fleet>> = HashMap::new();
    let mut surviving_fleets: Vec<Fleet> = Vec::with_capacity(state.fleets.len());
    for fleet in state.fleets.drain(..) {
        let speed = fleet_speed(fleet.ships, max_speed);
        let old = (fleet.x, fleet.y);
        let new = (
            fleet.x + speed * fleet.angle.cos(),
            fleet.y + speed * fleet.angle.sin(),
        );

        let mut hit_pid: Option<i64> = None;
        for planet in &state.planets {
            if let Some(&(p_old, p_new)) = planet_paths.get(&planet.id) {
                if p_old.0 < -50.0 && p_new.0 < -50.0 {
                    continue;
                }
                if swept_pair_hit(old, new, p_old, p_new, planet.radius) {
                    hit_pid = Some(planet.id);
                    break;
                }
            }
        }
        if let Some(pid) = hit_pid {
            combat.entry(pid).or_default().push(fleet);
            continue;
        }
        // Out of bounds
        if new.0 < 0.0 || new.0 > BOARD_SIZE || new.1 < 0.0 || new.1 > BOARD_SIZE {
            continue;
        }
        // Sun hit
        if point_to_segment_distance((CENTER_X, CENTER_Y), old, new) < SUN_RADIUS {
            continue;
        }
        let mut f = fleet;
        f.x = new.0;
        f.y = new.1;
        surviving_fleets.push(f);
    }
    state.fleets = surviving_fleets;

    // 4. Apply planet positions (move them to their new spots).
    for p in state.planets.iter_mut() {
        if let Some(&(_, new_pos)) = planet_paths.get(&p.id) {
            p.x = new_pos.0;
            p.y = new_pos.1;
        }
    }

    // 5. Combat resolution.
    for (pid, fleets) in combat {
        let planet = match state.planets.iter_mut().find(|p| p.id == pid) {
            Some(p) => p,
            None => continue,
        };
        let mut by_owner: HashMap<i32, i64> = HashMap::new();
        for f in &fleets {
            *by_owner.entry(f.owner).or_insert(0) += f.ships;
        }
        let mut sorted: Vec<(i32, i64)> = by_owner.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let (top_owner, top_ships) = sorted[0];
        let (sv_owner, sv_ships) = if sorted.len() > 1 {
            let sec = sorted[1].1;
            if top_ships == sec {
                (-1, 0)
            } else {
                (top_owner, top_ships - sec)
            }
        } else {
            (top_owner, top_ships)
        };
        if sv_ships > 0 {
            if planet.owner == sv_owner {
                planet.ships += sv_ships;
            } else {
                planet.ships -= sv_ships;
                if planet.ships < 0 {
                    planet.owner = sv_owner;
                    planet.ships = -planet.ships;
                }
            }
        }
    }

    // 6. Remove expired comets.
    if !comet_expired.is_empty() {
        let dead: std::collections::HashSet<i64> = comet_expired.into_iter().collect();
        state.planets.retain(|p| !dead.contains(&p.id));
        for g in state.comets.iter_mut() {
            g.planet_ids.retain(|id| !dead.contains(id));
        }
        state.comets.retain(|g| !g.planet_ids.is_empty());
    }

    // 7. Bookkeeping: advance step.
    state.step += 1;
}

/// Total ships across planets + fleets owned by `player`.
pub fn player_score(state: &GameState, player: i32) -> i64 {
    let p: i64 = state
        .planets
        .iter()
        .filter(|p| p.owner == player)
        .map(|p| p.ships)
        .sum();
    let f: i64 = state.fleets.iter().filter(|f| f.owner == player).map(|f| f.ships).sum();
    p + f
}

/// Number of distinct active player IDs (owners of planets or fleets).
/// Hot path — uses a small stack-allocated bitset instead of HashSet.
pub fn alive_players(state: &GameState) -> usize {
    // Player IDs in 2-4 player games are 0..4 (or -1 for neutral).
    let mut seen: u32 = 0;
    for p in &state.planets {
        if p.owner >= 0 && p.owner < 32 {
            seen |= 1 << p.owner;
        }
    }
    for f in &state.fleets {
        if f.owner >= 0 && f.owner < 32 {
            seen |= 1 << f.owner;
        }
    }
    seen.count_ones() as usize
}
