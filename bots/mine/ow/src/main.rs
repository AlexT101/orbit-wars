//! Orbit Wars bot — daemon mode.
//!
//! Reads one JSON observation per line on stdin, writes one JSON moves array
//! per line on stdout. The Python wrapper (`ow/main.py`) spawns us once and
//! pipes observations each turn.

mod pathing;

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

// ---- Game constants ----
pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER_X: f64 = 50.0;
pub const CENTER_Y: f64 = 50.0;
pub const SUN_RADIUS: f64 = 10.0;
pub const ROTATION_RADIUS_LIMIT: f64 = 50.0;
pub const COMET_RADIUS: f64 = 1.0;
pub const COMET_PRODUCTION: i64 = 1;
pub const DEFAULT_MAX_FLEET_SPEED: f64 = 6.0;
pub const DEFAULT_COMET_SPEED: f64 = 4.0;

// ---- Types ----
#[derive(Clone, Debug)]
pub struct Planet {
    pub id: i64,
    pub owner: i32, // -1 neutral
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i64,
    pub production: i64,
    // derived
    pub orbital_radius: f64,
    pub initial_angle: f64,
    pub is_orbiting: bool,
    pub is_comet: bool,
}

#[derive(Clone, Debug)]
pub struct Fleet {
    pub id: i64,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i64,
    pub ships: i64,
}

#[derive(Clone, Debug)]
pub struct CometGroup {
    pub planet_ids: Vec<i64>,
    pub paths: Vec<Vec<(f64, f64)>>,
    pub path_index: i64,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub player: i32,
    pub step: i64,
    pub planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub angular_velocity: f64,
    pub comets: Vec<CometGroup>,
    pub max_speed: f64,
    pub comet_speed: f64,
}

impl GameState {
    pub fn comet_group_for(&self, comet_id: i64) -> Option<(&CometGroup, usize)> {
        for g in &self.comets {
            if let Some(i) = g.planet_ids.iter().position(|&id| id == comet_id) {
                return Some((g, i));
            }
        }
        None
    }

    /// Number of remaining production turns for this comet (counting now).
    /// Returns 0 if it's not a live comet.
    pub fn comet_remaining(&self, planet: &Planet) -> i64 {
        if !planet.is_comet {
            return 0;
        }
        if let Some((g, i)) = self.comet_group_for(planet.id) {
            return (g.paths[i].len() as i64 - g.path_index).max(0);
        }
        0
    }

    /// Position of `planet` at relative future turn `dt` (dt=0 = right now).
    /// Returns None if the comet has expired by then.
    pub fn planet_pos_at(&self, planet: &Planet, dt: i64) -> Option<(f64, f64)> {
        if planet.is_comet {
            let (g, i) = self.comet_group_for(planet.id)?;
            let idx = g.path_index + dt;
            if idx < 0 || idx as usize >= g.paths[i].len() {
                return None;
            }
            return Some(g.paths[i][idx as usize]);
        }
        if planet.is_orbiting {
            // Engine produces env.steps[K] by calling the interpreter with
            // obs.step=K-1 (Kaggle increments AFTER), so the rotation
            // applied is `omega * (K-1)`. The planet shown at obs.step=K
            // is therefore at init + omega*(K-1), not init + omega*K.
            let abs_step = (self.step + dt - 1).max(0);
            let a = planet.initial_angle + self.angular_velocity * abs_step as f64;
            Some((
                CENTER_X + planet.orbital_radius * a.cos(),
                CENTER_Y + planet.orbital_radius * a.sin(),
            ))
        } else {
            Some((planet.x, planet.y))
        }
    }
}

// ---- Parsing ----
fn as_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| v.as_i64().unwrap_or(0) as f64)
}

fn parse_state(v: &Value) -> GameState {
    let player = v["player"].as_i64().unwrap_or(0) as i32;
    let step = v["step"].as_i64().unwrap_or(0);
    let angular_velocity = v["angular_velocity"].as_f64().unwrap_or(0.0);

    let comet_ids: std::collections::HashSet<i64> = v["comet_planet_ids"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default();

    let initial_pos: HashMap<i64, (f64, f64)> = v["initial_planets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let arr = p.as_array()?;
                    let id = arr.get(0)?.as_i64()?;
                    let x = as_f64(arr.get(2)?);
                    let y = as_f64(arr.get(3)?);
                    Some((id, (x, y)))
                })
                .collect()
        })
        .unwrap_or_default();

    let planets: Vec<Planet> = v["planets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let arr = p.as_array()?;
                    let id = arr.get(0)?.as_i64()?;
                    let owner = arr.get(1)?.as_i64()? as i32;
                    let x = as_f64(arr.get(2)?);
                    let y = as_f64(arr.get(3)?);
                    let radius = as_f64(arr.get(4)?);
                    let ships = arr.get(5)?.as_i64().unwrap_or(0);
                    let production = arr.get(6)?.as_i64().unwrap_or(0);
                    let is_comet = comet_ids.contains(&id);
                    let (ix, iy) = *initial_pos.get(&id).unwrap_or(&(x, y));
                    let dx = ix - CENTER_X;
                    let dy = iy - CENTER_Y;
                    let orbital_radius = (dx * dx + dy * dy).sqrt();
                    let initial_angle = dy.atan2(dx);
                    let is_orbiting =
                        !is_comet && orbital_radius + radius < ROTATION_RADIUS_LIMIT;
                    Some(Planet {
                        id,
                        owner,
                        x,
                        y,
                        radius,
                        ships,
                        production,
                        orbital_radius,
                        initial_angle,
                        is_orbiting,
                        is_comet,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let fleets: Vec<Fleet> = v["fleets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|f| {
                    let arr = f.as_array()?;
                    Some(Fleet {
                        id: arr.get(0)?.as_i64()?,
                        owner: arr.get(1)?.as_i64()? as i32,
                        x: as_f64(arr.get(2)?),
                        y: as_f64(arr.get(3)?),
                        angle: as_f64(arr.get(4)?),
                        from_planet_id: arr.get(5)?.as_i64()?,
                        ships: arr.get(6)?.as_i64().unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let comets: Vec<CometGroup> = v["comets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|g| {
                    let pids = g["planet_ids"]
                        .as_array()?
                        .iter()
                        .filter_map(|x| x.as_i64())
                        .collect();
                    let paths = g["paths"]
                        .as_array()?
                        .iter()
                        .filter_map(|p| {
                            Some(
                                p.as_array()?
                                    .iter()
                                    .filter_map(|pt| {
                                        let arr = pt.as_array()?;
                                        Some((as_f64(arr.get(0)?), as_f64(arr.get(1)?)))
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    let path_index = g["path_index"].as_i64()?;
                    Some(CometGroup {
                        planet_ids: pids,
                        paths,
                        path_index,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let cfg = v.get("config");
    let max_speed = cfg
        .and_then(|c| c.get("shipSpeed"))
        .and_then(|v| v.as_f64())
        .unwrap_or(DEFAULT_MAX_FLEET_SPEED);
    let comet_speed = cfg
        .and_then(|c| c.get("cometSpeed"))
        .and_then(|v| v.as_f64())
        .unwrap_or(DEFAULT_COMET_SPEED);

    GameState {
        player,
        step,
        planets,
        fleets,
        angular_velocity,
        comets,
        max_speed,
        comet_speed,
    }
}

// ---- Combat simulation helpers ----

/// Resolve one round of combat: arriving ships per-owner, plus current
/// (owner, ships) garrison. Returns (new_owner, new_ships).
fn resolve_combat(
    mut owner: i32,
    mut ships: i64,
    arrivals_by_owner: &HashMap<i32, i64>,
) -> (i32, i64) {
    if arrivals_by_owner.is_empty() {
        return (owner, ships);
    }
    let mut srt: Vec<(i32, i64)> = arrivals_by_owner.iter().map(|(k, v)| (*k, *v)).collect();
    srt.sort_by(|a, b| b.1.cmp(&a.1));
    let (top_o, top_s) = srt[0];
    let (sv_o, sv_s) = if srt.len() > 1 {
        let sec = srt[1].1;
        if top_s == sec {
            (-1, 0)
        } else {
            (top_o, top_s - sec)
        }
    } else {
        (top_o, top_s)
    };
    if sv_s > 0 {
        if owner == sv_o {
            ships += sv_s;
        } else {
            ships -= sv_s;
            if ships < 0 {
                owner = sv_o;
                ships = -ships;
            }
        }
    }
    (owner, ships)
}

/// Simulate the timeline of a planet's (owner, ships) under a given set of
/// arrivals. Each arrival is `(dt, owner, ships)`. Returns the (owner, ships)
/// at relative turn `target_dt`. Production accrues turn-by-turn for the
/// current owner. Comets that expire before `target_dt` return (-1, 0).
fn simulate_at(
    planet: &Planet,
    target_dt: i64,
    arrivals: &[(i64, i32, i64)],
    state: &GameState,
) -> (i32, i64) {
    let mut sorted: Vec<(i64, i32, i64)> = arrivals.to_vec();
    sorted.sort_by_key(|x| x.0);
    let mut owner = planet.owner;
    let mut ships = planet.ships;
    let mut cur_dt = 0i64;
    let mut i = 0;
    while i < sorted.len() && sorted[i].0 <= target_dt {
        let dt = sorted[i].0;
        // Comet expiration check
        if planet.is_comet && state.planet_pos_at(planet, dt).is_none() {
            return (-1, 0);
        }
        if owner != -1 {
            ships += planet.production * (dt - cur_dt);
        }
        let mut by_owner: HashMap<i32, i64> = HashMap::new();
        while i < sorted.len() && sorted[i].0 == dt {
            *by_owner.entry(sorted[i].1).or_insert(0) += sorted[i].2;
            i += 1;
        }
        let (no, ns) = resolve_combat(owner, ships, &by_owner);
        owner = no;
        ships = ns;
        cur_dt = dt;
    }
    if owner != -1 && target_dt > cur_dt {
        ships += planet.production * (target_dt - cur_dt);
    }
    (owner, ships)
}

/// True iff, from the first event in which `target` becomes mine, ownership
/// never flips away (and the planet is mine at the last event). Captures the
/// user's "never falls into their control" rule for both currently-owned
/// planets and ones we'd capture via in-flight fleets — even a *temporary*
/// loss returns false (target should be reinforced).
fn stays_mine_throughout(
    target: &Planet,
    arrivals: &[(i64, i32, i64)],
    me: i32,
    state: &GameState,
) -> bool {
    let mut sorted: Vec<(i64, i32, i64)> = arrivals.to_vec();
    sorted.sort_by_key(|x| x.0);
    let mut owner = target.owner;
    let mut ships = target.ships;
    let mut cur_dt = 0i64;
    let mut became_mine = owner == me;
    let mut ever_lost = false;
    let mut i = 0;
    while i < sorted.len() {
        let dt = sorted[i].0;
        if target.is_comet && state.planet_pos_at(target, dt).is_none() {
            break;
        }
        if owner != -1 {
            ships += target.production * (dt - cur_dt);
        }
        let mut by_owner: HashMap<i32, i64> = HashMap::new();
        while i < sorted.len() && sorted[i].0 == dt {
            *by_owner.entry(sorted[i].1).or_insert(0) += sorted[i].2;
            i += 1;
        }
        let (no, ns) = resolve_combat(owner, ships, &by_owner);
        owner = no;
        ships = ns;
        if owner == me {
            became_mine = true;
        } else if became_mine {
            ever_lost = true;
        }
        cur_dt = dt;
    }
    !ever_lost && owner == me
}

/// True iff sending `send_amount` ships from `planet` (mine) keeps it mine
/// throughout the projected arrivals timeline.
fn simulates_safe(
    planet: &Planet,
    send_amount: i64,
    arrivals: &[(i64, i32, i64)],
    me: i32,
    state: &GameState,
) -> bool {
    if planet.owner != me {
        return send_amount == 0;
    }
    if send_amount < 0 || send_amount > planet.ships {
        return false;
    }
    let mut sorted: Vec<(i64, i32, i64)> = arrivals.to_vec();
    sorted.sort_by_key(|x| x.0);
    let mut owner = me;
    let mut ships = planet.ships - send_amount;
    let mut cur_dt = 0i64;
    let mut i = 0;
    while i < sorted.len() {
        let dt = sorted[i].0;
        if planet.is_comet && state.planet_pos_at(planet, dt).is_none() {
            break;
        }
        if owner != -1 {
            ships += planet.production * (dt - cur_dt);
        }
        let mut by_owner: HashMap<i32, i64> = HashMap::new();
        while i < sorted.len() && sorted[i].0 == dt {
            *by_owner.entry(sorted[i].1).or_insert(0) += sorted[i].2;
            i += 1;
        }
        let (no, ns) = resolve_combat(owner, ships, &by_owner);
        owner = no;
        ships = ns;
        if owner != me {
            return false;
        }
        cur_dt = dt;
    }
    true
}

// ---- Strategy ----

#[derive(Clone, Copy)]
struct PlanEntry {
    from_id: i64,
    ships: i64,
    angle: f64,
}

/// For a given target and capture time `t`, find a feasible dispatch plan.
/// Returns the dispatches (a list of (from_planet_id, ships, angle)) and the
/// total ships sent, or None if `t` is infeasible.
fn plan_for_time(
    target: &Planet,
    t: i64,
    my_planets: &[Planet],
    available: &HashMap<i64, i64>,
    state: &GameState,
    arrivals: &HashMap<i64, Vec<(i64, i32, i64)>>,
) -> Option<(Vec<PlanEntry>, i64)> {
    let empty = Vec::new();
    let arr = arrivals.get(&target.id).unwrap_or(&empty);
    let (owner_t, ships_t) = simulate_at(target, t, arr, state);
    let me = state.player;
    let required: i64 = if owner_t == me {
        0
    } else {
        ships_t + 1
    };
    if required <= 0 {
        return None; // nothing to do — caller skips this target
    }

    // For each owned planet, find min_s = smallest ship count that arrives
    // by t, and the arrival time at min_s. Sum-of-min_s across all feasible
    // planets is the "total capacity" used for feasibility (some of those
    // planets won't dispatch this turn — they'll wait until they're the
    // bottleneck — but the existence of that capacity is what makes T
    // achievable across multiple turns of greedy planning).
    let mut feas: Vec<PlanetFeas> = Vec::new();
    for mp in my_planets {
        if mp.id == target.id {
            continue;
        }
        let avail = *available.get(&mp.id).unwrap_or(&0);
        if avail <= 0 {
            continue;
        }
        let top_path = pathing::dir_to_hit(mp, target, avail, state, 0);
        if !top_path.as_ref().map(|r| r.time <= t).unwrap_or(false) {
            continue;
        }
        // Binary search min_s.
        let mut lo = 1i64;
        let mut hi = avail;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let ok = pathing::dir_to_hit(mp, target, mid, state, 0)
                .map(|r| r.time <= t)
                .unwrap_or(false);
            if ok {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let min_s = lo;
        let r_min = match pathing::dir_to_hit(mp, target, min_s, state, 0) {
            Some(r) => r,
            None => continue,
        };
        // Largest ship count from this planet that still arrives at t
        // (not strictly earlier) — so a tied planet can send more than min
        // while preserving same-T arrival. Arrival time decreases as ships
        // increase, so we binary-search the boundary.
        let mut lo2 = min_s;
        let mut hi2 = avail;
        while lo2 < hi2 {
            let mid = (lo2 + hi2 + 1) / 2;
            let arr_mid = pathing::dir_to_hit(mp, target, mid, state, 0)
                .map(|r| r.time)
                .unwrap_or(i64::MAX);
            if arr_mid >= t {
                lo2 = mid;
            } else {
                hi2 = mid - 1;
            }
        }
        let max_s = lo2;
        feas.push(PlanetFeas {
            from_id: mp.id,
            min_s,
            max_s,
            arr_at_min: r_min.time,
            angle_at_min: r_min.angle,
        });
    }

    // Feasibility: sum of min_s across ALL feasible sources must cover
    // required. (Faster planets contribute via future-turn dispatches.)
    let cap_min: i64 = feas.iter().map(|x| x.min_s).sum();
    if cap_min < required {
        return None;
    }

    // Dispatch only from planets tied for the slowest arrival at t (i.e.,
    // arr_at_min == t). Faster planets wait — they'll become bottlenecks on
    // later turns. Each tied planet sends in [min_s, max_s_for_t], chosen
    // greedily to fill `required` while preserving same-t arrival.
    let mut tied: Vec<&PlanetFeas> = feas.iter().filter(|f| f.arr_at_min == t).collect();
    if tied.is_empty() {
        return None;
    }
    // Send min_s from each first (base). Then distribute remainder.
    let base: i64 = tied.iter().map(|f| f.min_s).sum();
    let mut remaining_extra = (required - base).max(0);
    // Allocate extra to whichever tied planet has the most headroom (max_s - min_s).
    tied.sort_by(|a, b| (b.max_s - b.min_s).cmp(&(a.max_s - a.min_s)));
    let mut sends: HashMap<i64, i64> = HashMap::new();
    for f in &tied {
        sends.insert(f.from_id, f.min_s);
    }
    for f in &tied {
        if remaining_extra <= 0 {
            break;
        }
        let cap = (f.max_s - f.min_s).max(0);
        let add = remaining_extra.min(cap);
        if add > 0 {
            *sends.get_mut(&f.from_id).unwrap() += add;
            remaining_extra -= add;
        }
    }
    let total: i64 = sends.values().sum();
    if total < required {
        // Even at max-for-t across tied planets we don't reach required.
        // Increasing send any further breaks same-t arrival. Caller will
        // try a larger t.
        return None;
    }
    let mut dispatches: Vec<PlanEntry> = Vec::new();
    for f in &tied {
        let send = *sends.get(&f.from_id).unwrap_or(&0);
        if send <= 0 {
            continue;
        }
        // Refresh angle for actual send (same-t arrival, so r.time should == t).
        let mp = match my_planets.iter().find(|p| p.id == f.from_id) {
            Some(p) => p,
            None => continue,
        };
        let angle = match pathing::dir_to_hit(mp, target, send, state, 0) {
            Some(r) => r.angle,
            None => f.angle_at_min,
        };
        dispatches.push(PlanEntry {
            from_id: f.from_id,
            ships: send,
            angle,
        });
    }
    Some((dispatches, total))
}

struct PlanetFeas {
    from_id: i64,
    min_s: i64,
    max_s: i64,
    arr_at_min: i64,
    angle_at_min: f64,
}

fn plan(state: &GameState) -> Vec<(i64, f64, i64)> {
    let me = state.player;
    let mut moves: Vec<(i64, f64, i64)> = Vec::new();

    // 1. Predict each in-flight fleet's collision.
    let mut arrivals: HashMap<i64, Vec<(i64, i32, i64)>> = HashMap::new();
    for fleet in &state.fleets {
        if let Some((pid, dt)) = pathing::predict_fleet_collision(fleet, state) {
            arrivals
                .entry(pid)
                .or_default()
                .push((dt, fleet.owner, fleet.ships));
        }
    }
    for v in arrivals.values_mut() {
        v.sort_by_key(|x| x.0);
    }

    // 2. Compute safe-to-send count per owned planet.
    let mut safe: HashMap<i64, i64> = HashMap::new();
    for p in &state.planets {
        if p.owner != me {
            continue;
        }
        let empty = Vec::new();
        let arr = arrivals.get(&p.id).unwrap_or(&empty);
        let mut lo = 0;
        let mut hi = p.ships;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if simulates_safe(p, mid, arr, me, state) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        safe.insert(p.id, lo);
    }

    // 3. Snapshot lists once.
    let my_planets: Vec<Planet> = state.planets.iter().filter(|p| p.owner == me).cloned().collect();
    let enemy_planets: Vec<Planet> = state
        .planets
        .iter()
        .filter(|p| p.owner != me && p.owner != -1)
        .cloned()
        .collect();

    // 4. Race filter: instead of "10 ships from each closest planet", use
    //    the *full sendable count* (= ships - defense reservation) for every
    //    source on each side. For each target, my fastest possible arrival
    //    (across all my planets sending their sendable count) is compared
    //    against the opponent's fastest possible arrival (same rule for
    //    them). If mine is faster-or-equal, the race is winnable.
    //
    //    For enemy-owned targets, exclude the target itself from the
    //    opponent's planet list (it's trivially "closest to itself").

    // Per-enemy-planet safe count (how many they can dispatch and still
    // hold the planet, mirroring our own safe_excess calc).
    let mut enemy_safe: HashMap<i64, i64> = HashMap::new();
    for p in &state.planets {
        if p.owner == me || p.owner == -1 {
            continue;
        }
        let empty = Vec::new();
        let arr = arrivals.get(&p.id).unwrap_or(&empty);
        let mut lo = 0;
        let mut hi = p.ships;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if simulates_safe(p, mid, arr, p.owner, state) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        enemy_safe.insert(p.id, lo);
    }

    let mut race_ok: HashMap<i64, bool> = HashMap::new();
    for target in &state.planets {
        if target.owner == me {
            continue;
        }
        let my_t = my_planets
            .iter()
            .filter_map(|mp| {
                let ships = *safe.get(&mp.id).unwrap_or(&0);
                if ships <= 0 {
                    return None;
                }
                pathing::dir_to_hit(mp, target, ships, state, 0).map(|r| r.time)
            })
            .min()
            .unwrap_or(i64::MAX);
        let their_t = enemy_planets
            .iter()
            .filter(|ep| ep.id != target.id)
            .filter_map(|ep| {
                let ships = *enemy_safe.get(&ep.id).unwrap_or(&0);
                if ships <= 0 {
                    return None;
                }
                pathing::dir_to_hit(ep, target, ships, state, 0).map(|r| r.time)
            })
            .min()
            .unwrap_or(i64::MAX);
        race_ok.insert(target.id, my_t <= their_t);
    }

    // 5. Greedy target selection. Each iteration picks the best target, plans
    //    the dispatch, deducts ships from `available`, repeats.
    let mut available = safe.clone();
    let mut guard = 0usize;
    let debug = std::env::var("OW_DEBUG_PLAN").is_ok();
    let mut skip_counts: HashMap<&str, i64> = HashMap::new();
    loop {
        guard += 1;
        if guard > 200 {
            break;
        }
        let mut best: Option<(f64, i64, i64, Vec<PlanEntry>)> = None;
        for target in &state.planets {
            let empty = Vec::new();
            let arr = arrivals.get(&target.id).unwrap_or(&empty);
            // Skip iff the planet ends up mine AND ownership never flips
            // away after we first acquire it. A *temporary* loss between
            // capture and the final event must trigger a defense pass —
            // checking only `final_owner == me` misses that case.
            if stays_mine_throughout(target, arr, me, state) {
                *skip_counts.entry("becomes_mine").or_insert(0) += 1;
                continue;
            }
            // Race filter applies to neutral and enemy targets — the
            // user's anti-trade-loss rule. For own threatened planets
            // (target.owner == me) we always try to defend, so skip
            // the filter for those.
            if target.owner != me && !*race_ok.get(&target.id).unwrap_or(&false) {
                *skip_counts.entry("race_fail").or_insert(0) += 1;
                continue;
            }
            // Find the MIN feasible T for this target (per the user's spec:
            // "binary search over capture time"). Score = production /
            // ships_required (higher = better); smaller T is the tiebreak.
            let hi = pathing::MAX_TIME.min(60);
            let mut found: Option<(i64, Vec<PlanEntry>, i64)> = None;
            let mut t = 1i64;
            while t <= hi {
                if let Some((plan, total)) = plan_for_time(target, t, &my_planets, &available, state, &arrivals) {
                    found = Some((t, plan, total));
                    break;
                }
                t += 1;
            }
            if let Some((t_best, plan_best, total_best)) = found {
                // Comet pay-off: only worth capturing a comet if the
                // production we'd collect from arrival to expiration covers
                // the ships we lose in the capture combat.
                if target.is_comet && target.owner != me {
                    let remaining = state.comet_remaining(target);
                    let productions_after = (remaining - t_best).max(0);
                    let ships_lost = (total_best - 1).max(0);
                    if ships_lost >= productions_after * target.production {
                        *skip_counts.entry("comet_no_payoff").or_insert(0) += 1;
                        continue;
                    }
                }
                let prod = target.production.max(1) as f64;
                let score = prod / (total_best.max(1) as f64);
                // Tiny T tiebreak — same-score plans prefer faster captures.
                let s = score - 1e-6 * t_best as f64;
                if best.as_ref().map(|b| s > b.0).unwrap_or(true) {
                    best = Some((s, target.id, t_best, plan_best));
                }
            } else {
                *skip_counts.entry("plan_infeasible").or_insert(0) += 1;
            }
        }
        if let Some((_, target_id, t_best, dispatches)) = best {
            if dispatches.is_empty() {
                break;
            }
            // Add the dispatches into the target's projected arrivals so the
            // next greedy iteration sees them and doesn't double-commit.
            let entry = arrivals.entry(target_id).or_default();
            for d in &dispatches {
                entry.push((t_best, me, d.ships));
            }
            entry.sort_by_key(|x| x.0);
            for d in dispatches {
                moves.push((d.from_id, d.angle, d.ships));
                if let Some(a) = available.get_mut(&d.from_id) {
                    *a -= d.ships;
                    if *a < 0 {
                        *a = 0;
                    }
                }
            }
        } else {
            break;
        }
    }

    // Departing-comet evacuation: any comet that expires next turn loses
    // its garrison to the void. Dump all remaining ships at the reachable
    // target with the smallest arrival time. (Score-agnostic: any landing
    // beats losing them.)
    for p in &state.planets {
        if p.owner != me || !p.is_comet {
            continue;
        }
        if state.comet_remaining(p) > 1 {
            continue;
        }
        let avail = available.get(&p.id).copied().unwrap_or(0).min(p.ships);
        if avail <= 0 {
            continue;
        }
        let mut best: Option<(i64, f64)> = None;
        for tgt in &state.planets {
            if tgt.id == p.id {
                continue;
            }
            if let Some(r) = pathing::dir_to_hit(p, tgt, avail, state, 0) {
                if best.as_ref().map(|b| r.time < b.0).unwrap_or(true) {
                    best = Some((r.time, r.angle));
                }
            }
        }
        if let Some((_, angle)) = best {
            moves.push((p.id, angle, avail));
            if let Some(a) = available.get_mut(&p.id) {
                *a = 0;
            }
        }
    }

    if debug && moves.is_empty() {
        let neutral = state.planets.iter().filter(|p| p.owner == -1).count();
        let safe_total: i64 = safe.values().sum();
        let ships_total: i64 = state.planets.iter().filter(|p| p.owner == me).map(|p| p.ships).sum();
        eprintln!(
            "[ow plan] step={} moves=0 neutrals={} ships={} safe={} skips={:?} arrivals_keys={}",
            state.step,
            neutral,
            ships_total,
            safe_total,
            skip_counts,
            arrivals.len(),
        );
    }
    moves
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let debug = std::env::var("OW_DEBUG").is_ok();
    let mut err = io::stderr();
    let mut buf = String::new();
    let mut handle = stdin.lock();
    loop {
        buf.clear();
        let n = handle.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        let line = buf.trim_end();
        if line.is_empty() {
            writeln!(out, "[]")?;
            out.flush()?;
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                writeln!(out, "[]")?;
                out.flush()?;
                continue;
            }
        };
        let state = parse_state(&v);
        let mv = plan(&state);
        if debug {
            let me = state.player;
            let my_count = state.planets.iter().filter(|p| p.owner == me).count();
            let my_ships: i64 = state.planets.iter().filter(|p| p.owner == me).map(|p| p.ships).sum();
            let neutral = state.planets.iter().filter(|p| p.owner == -1).count();
            let enemy = state.planets.iter().filter(|p| p.owner != me && p.owner != -1).count();
            writeln!(
                err,
                "[ow p{}] step={} planets={}(m)/{}(n)/{}(e) ships={} fleets={} moves={}",
                me, state.step, my_count, neutral, enemy, my_ships, state.fleets.len(), mv.len()
            ).ok();
        }
        let arr: Vec<Value> = mv
            .into_iter()
            .map(|(fid, ang, ships)| json!([fid, ang, ships]))
            .collect();
        writeln!(out, "{}", Value::Array(arr))?;
        out.flush()?;
    }
    Ok(())
}
