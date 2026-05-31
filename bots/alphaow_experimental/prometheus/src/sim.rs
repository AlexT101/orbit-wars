//! Game-state simulator. Mirrors the orbit_wars engine's per-turn flow:
//! launches → production → fleet movement (swept-pair collisions) → planet
//! /comet movement → combat resolution → comet expiration.
//!
//! Comet *spawning* (at steps 50/150/.../450) is omitted because we don't
//! have the RNG; this drifts from reality after the first spawn but is
//! acceptable for MCTS rollouts per the user.

use crate::pathing::{fleet_speed, point_to_segment_distance, swept_pair_hit};
use crate::policy::XorRng;
use crate::{
    CometGroup, Fleet, GameState, Planet, BOARD_SIZE, CENTER_X, CENTER_Y, COMET_PRODUCTION,
    COMET_RADIUS, SUN_RADIUS,
};
use std::collections::HashMap;
use std::sync::OnceLock;

const SPAWN_STEPS: &[i64] = &[50, 150, 250, 350, 450];
const HARVEST_JSON: &[u8] = include_bytes!("../comet_harvest.json");

#[derive(Clone)]
struct HarvestedComet {
    path: Vec<(f64, f64)>,
    ships: i64,
}

fn harvest() -> &'static HashMap<i64, Vec<Vec<HarvestedComet>>> {
    static M: OnceLock<HashMap<i64, Vec<Vec<HarvestedComet>>>> = OnceLock::new();
    M.get_or_init(|| {
        let v: serde_json::Value = serde_json::from_slice(HARVEST_JSON).unwrap_or(serde_json::Value::Null);
        let mut out: HashMap<i64, Vec<Vec<HarvestedComet>>> = HashMap::new();
        if let serde_json::Value::Object(map) = v {
            for (k, groups) in map {
                let step: i64 = match k.parse() { Ok(n) => n, Err(_) => continue };
                if let serde_json::Value::Array(arr) = groups {
                    let mut list: Vec<Vec<HarvestedComet>> = Vec::new();
                    for g in arr {
                        let comets = match g.as_array() { Some(a) => a, None => continue };
                        let mut hg: Vec<HarvestedComet> = Vec::with_capacity(4);
                        for c in comets {
                            let path = match c.get("path").and_then(|p| p.as_array()) {
                                Some(a) => a.iter().filter_map(|pt| {
                                    let p = pt.as_array()?;
                                    Some((p.get(0)?.as_f64()?, p.get(1)?.as_f64()?))
                                }).collect(),
                                None => continue,
                            };
                            let ships = c.get("ships").and_then(|s| s.as_i64()).unwrap_or(1);
                            hg.push(HarvestedComet { path, ships });
                        }
                        if !hg.is_empty() {
                            list.push(hg);
                        }
                    }
                    out.insert(step, list);
                }
            }
        }
        out
    })
}

pub fn maybe_spawn_comets(state: &mut GameState, rng: &mut XorRng) {
    let next_step = state.step + 1;
    if !SPAWN_STEPS.contains(&next_step) {
        return;
    }
    let h = harvest();
    let groups = match h.get(&next_step) {
        Some(g) if !g.is_empty() => g,
        _ => return,
    };
    let idx = (rng.next_u64() % groups.len() as u64) as usize;
    let chosen = &groups[idx];
    let mut next_id = state.planets.iter().map(|p| p.id).max().unwrap_or(-1) + 1;
    let mut group_ids: Vec<i64> = Vec::with_capacity(chosen.len());
    let mut paths: Vec<Vec<(f64, f64)>> = Vec::with_capacity(chosen.len());
    for c in chosen {
        let pid = next_id;
        next_id += 1;
        group_ids.push(pid);
        paths.push(c.path.clone());
        state.planets.push(Planet {
            id: pid,
            owner: -1,
            x: -99.0,
            y: -99.0,
            radius: COMET_RADIUS,
            ships: c.ships,
            production: COMET_PRODUCTION,
            orbital_radius: 0.0,
            initial_angle: 0.0,
            is_orbiting: false,
            is_comet: true,
        });
    }
    state.comets.push(CometGroup {
        planet_ids: group_ids,
        paths,
        path_index: -1,
    });
}

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
/// `rng` is used for sampling future comet spawns from harvest data.
pub fn tick(state: &mut GameState, rng: &mut XorRng) {
    tick_inner(state, rng, true);
}

/// Like `tick` but never spawns comets (used by the sim validator).
pub fn tick_no_spawn(state: &mut GameState, rng: &mut XorRng) {
    tick_inner(state, rng, false);
}

fn tick_inner(state: &mut GameState, rng: &mut XorRng, do_spawn: bool) {
    // 0. Comet spawning (sampled from harvest). Engine does this BEFORE
    // launches in real games; we accept the order swap since agents can't
    // launch at not-yet-spawned comets anyway.
    if do_spawn {
        maybe_spawn_comets(state, rng);
    }
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
