//! Cheap Apollo-flavored candidate generator for Prometheus DUCT.
//!
//! Full Apollo strategy is strong but expensive because every candidate call
//! builds a rich world, arrival ledger, and target timelines. This module keeps
//! the useful fast pieces: cached Apollo aiming, production/ship/travel target
//! scoring, and a handful of distinct tactical styles for DUCT to evaluate.

use crate::apollo::cache::{AimCacheVerdict, EntityCache, InvariantVerdict};
use crate::apollo::helpers::{aim_ignoring_comets, aim_with_prediction};
use crate::pathing::fleet_speed;
use crate::sim::Action;
use crate::{GameState, Planet};
use std::collections::HashSet;

const ROOT_TARGETS: usize = 12;
const NODE_TARGETS: usize = 8;
const ROOT_SOURCES: usize = 7;
const NODE_SOURCES: usize = 5;

#[derive(Clone, Copy)]
struct Shot {
    src_id: i64,
    angle: f64,
    ships: i64,
    arrival: i64,
}

#[derive(Clone, Copy)]
struct SourceInfo<'a> {
    planet: &'a Planet,
    available: i64,
    dist: f64,
}

#[derive(Clone, Copy)]
struct TargetInfo<'a> {
    planet: &'a Planet,
    score: f64,
    nearest_dist: f64,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn pos_now(state: &GameState, p: &Planet) -> (f64, f64) {
    state.planet_pos_at(p, 0).unwrap_or((p.x, p.y))
}

fn dominant_enemy(state: &GameState, me: i32) -> Option<i32> {
    let mut best: Option<(i32, i64)> = None;
    for p in &state.planets {
        if p.owner == me || p.owner == -1 {
            continue;
        }
        let score = crate::sim::player_score(state, p.owner);
        if best.map(|(_, s)| score > s).unwrap_or(true) {
            best = Some((p.owner, score));
        }
    }
    for f in &state.fleets {
        if f.owner == me || f.owner == -1 {
            continue;
        }
        let score = crate::sim::player_score(state, f.owner);
        if best.map(|(_, s)| score > s).unwrap_or(true) {
            best = Some((f.owner, score));
        }
    }
    best.map(|(p, _)| p)
}

fn cached_aim(cache: &EntityCache, src_id: i64, target_id: i64, ships: i64) -> Option<crate::apollo::aim::AimResult> {
    let ships = ships.max(1);
    match cache.aim_cache_lookup(src_id, target_id, ships, 0) {
        AimCacheVerdict::Hit(r) => return r,
        AimCacheVerdict::Miss | AimCacheVerdict::Stale => {}
    }
    let result = match cache.invariant_aim_lookup(src_id, target_id, ships, 0) {
        InvariantVerdict::Use(r) => Some(r),
        InvariantVerdict::SingleSolve => {
            let r = aim_with_prediction(cache, src_id, target_id, ships, 0);
            cache.aim_cache_store(src_id, target_id, ships, 0, r);
            r
        }
        InvariantVerdict::DualSolve => {
            let base = aim_ignoring_comets(cache, src_id, target_id, ships, 0);
            cache.invariant_aim_store(src_id, target_id, ships, 0, base);
            match cache.invariant_aim_lookup(src_id, target_id, ships, 0) {
                InvariantVerdict::Use(r) => Some(r),
                _ => {
                    let r = aim_with_prediction(cache, src_id, target_id, ships, 0);
                    cache.aim_cache_store(src_id, target_id, ships, 0, r);
                    r
                }
            }
        }
    };
    result
}

fn target_need(target: &Planet, arrival: i64) -> i64 {
    let growth = if target.owner == -1 {
        0
    } else {
        target.production * arrival.max(0)
    };
    (target.ships + growth + 1).max(1)
}

fn reserve_for(src: &Planet, style: f64) -> i64 {
    ((src.ships as f64) * style).round() as i64
}

fn source_infos<'a>(state: &'a GameState, me: i32, target: &Planet, reserve_style: f64) -> Vec<SourceInfo<'a>> {
    let tpos = pos_now(state, target);
    let mut out: Vec<_> = state
        .planets
        .iter()
        .filter(|p| p.owner == me && p.ships > 1)
        .map(|p| {
            let reserve = reserve_for(p, reserve_style).clamp(1, p.ships);
            SourceInfo {
                planet: p,
                available: (p.ships - reserve).max(0),
                dist: dist(pos_now(state, p), tpos),
            }
        })
        .filter(|s| s.available > 0)
        .collect();
    out.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.available.cmp(&a.available))
    });
    out
}

fn target_infos<'a>(state: &'a GameState, me: i32, root: bool) -> Vec<TargetInfo<'a>> {
    let enemy = dominant_enemy(state, me);
    let sources: Vec<&Planet> = state
        .planets
        .iter()
        .filter(|p| p.owner == me && p.ships > 1)
        .collect();
    let mut out: Vec<_> = state
        .planets
        .iter()
        .filter(|p| p.owner != me)
        .filter_map(|p| {
            let ppos = pos_now(state, p);
            let nearest = sources
                .iter()
                .map(|s| dist(pos_now(state, s), ppos))
                .fold(f64::INFINITY, f64::min);
            if !nearest.is_finite() {
                return None;
            }
            let owner_bonus = if Some(p.owner) == enemy {
                90.0
            } else if p.owner == -1 {
                35.0
            } else {
                55.0
            };
            let comet_penalty = if p.is_comet {
                let rem = state.comet_remaining(p);
                if rem <= 0 { 80.0 } else { (20 - rem).max(0) as f64 * 2.0 }
            } else {
                0.0
            };
            let type_bonus = if p.is_orbiting { 14.0 } else { 0.0 };
            let score = owner_bonus
                + p.production as f64 * 95.0
                + p.radius * 7.0
                + type_bonus
                - p.ships as f64 * 0.48
                - nearest * if p.owner == -1 { 1.35 } else { 0.85 }
                - comet_penalty;
            Some(TargetInfo {
                planet: p,
                score,
                nearest_dist: nearest,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.nearest_dist
                    .partial_cmp(&b.nearest_dist)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let cap = env_usize(
        if root { "OW_MOVEGEN_ROOT_TARGETS" } else { "OW_MOVEGEN_NODE_TARGETS" },
        if root { ROOT_TARGETS } else { NODE_TARGETS },
    );
    out.truncate(cap);
    out
}

fn push_unique(out: &mut Vec<Vec<Action>>, seen: &mut HashSet<Vec<(i64, i64, i64, i32)>>, mut plan: Vec<Action>) {
    plan.retain(|a| a.2 > 0);
    plan.sort_by_key(|a| (a.0, (a.1 * 10_000.0).round() as i64, a.2, a.3));
    let key: Vec<_> = plan
        .iter()
        .map(|a| (a.0, (a.1 * 10_000.0).round() as i64, a.2, a.3))
        .collect();
    if seen.insert(key) {
        out.push(plan);
    }
}

fn shot_from(cache: &EntityCache, src: &Planet, target: &Planet, ships: i64) -> Option<Shot> {
    let ships = ships.clamp(1, src.ships);
    let (angle, turns, _, _, _) = cached_aim(cache, src.id, target.id, ships)?;
    Some(Shot {
        src_id: src.id,
        angle,
        ships,
        arrival: turns.max(1),
    })
}

fn focused_capture(
    state: &GameState,
    player: i32,
    cache: &EntityCache,
    target: &Planet,
    reserve_style: f64,
    overkill: f64,
    source_cap: usize,
) -> Vec<Action> {
    let mut plan = Vec::new();
    let mut committed = 0i64;
    let mut sources = source_infos(state, player, target, reserve_style);
    sources.truncate(source_cap);
    for source in sources {
        let speed = fleet_speed(source.available.max(1), state.max_speed);
        let eta_guess = (source.dist / speed).ceil() as i64;
        let need = ((target_need(target, eta_guess) as f64) * overkill).ceil() as i64;
        if committed >= need {
            break;
        }
        let want = (need - committed).min(source.available);
        if want <= 0 {
            continue;
        }
        let Some(shot) = shot_from(cache, source.planet, target, want) else {
            continue;
        };
        let need_after_aim = ((target_need(target, shot.arrival) as f64) * overkill).ceil() as i64;
        let ships = if committed + shot.ships < need_after_aim {
            (need_after_aim - committed).min(source.available)
        } else {
            shot.ships
        };
        let Some(final_shot) = shot_from(cache, source.planet, target, ships) else {
            continue;
        };
        committed += final_shot.ships;
        plan.push((final_shot.src_id, final_shot.angle, final_shot.ships, player));
        if committed >= need_after_aim {
            break;
        }
    }
    plan
}

fn pressure_attack(
    state: &GameState,
    player: i32,
    cache: &EntityCache,
    target: &Planet,
    fraction: f64,
) -> Vec<Action> {
    let mut sources = source_infos(state, player, target, 0.18);
    sources.truncate(3);
    let mut plan = Vec::new();
    for source in sources {
        let ships = ((source.available as f64) * fraction).round() as i64;
        if ships <= 0 {
            continue;
        }
        if let Some(shot) = shot_from(cache, source.planet, target, ships) {
            plan.push((shot.src_id, shot.angle, shot.ships, player));
        }
    }
    plan
}

fn greedy_multi(state: &GameState, player: i32, cache: &EntityCache, targets: &[TargetInfo<'_>]) -> Vec<Action> {
    let mut remaining: Vec<(i64, i64)> = state
        .planets
        .iter()
        .filter(|p| p.owner == player)
        .map(|p| (p.id, (p.ships - reserve_for(p, 0.22).max(1)).max(0)))
        .collect();
    let mut plan = Vec::new();
    for target in targets.iter().take(4) {
        let t = target.planet;
        let mut sources: Vec<_> = state
            .planets
            .iter()
            .filter(|p| p.owner == player && p.ships > 1)
            .filter_map(|p| {
                let rem = remaining.iter().find(|(id, _)| *id == p.id)?.1;
                if rem <= 0 {
                    None
                } else {
                    Some((p, dist(pos_now(state, p), pos_now(state, t)), rem))
                }
            })
            .collect();
        sources.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut committed = 0i64;
        for (src, d, rem) in sources.into_iter().take(4) {
            let eta = (d / fleet_speed(rem.max(1), state.max_speed)).ceil() as i64;
            let need = ((target_need(t, eta) as f64) * 1.08).ceil() as i64;
            if committed >= need {
                break;
            }
            let ships = (need - committed).min(rem);
            let Some(shot) = shot_from(cache, src, t, ships) else {
                continue;
            };
            if let Some((_, r)) = remaining.iter_mut().find(|(id, _)| *id == src.id) {
                *r -= shot.ships;
            }
            committed += shot.ships;
            plan.push((shot.src_id, shot.angle, shot.ships, player));
            if committed >= need {
                break;
            }
        }
    }
    plan
}

fn reinforcement_plan(state: &GameState, player: i32, cache: &EntityCache) -> Vec<Action> {
    let my: Vec<&Planet> = state.planets.iter().filter(|p| p.owner == player).collect();
    if my.len() < 2 {
        return Vec::new();
    }
    let enemyish: Vec<&Planet> = state.planets.iter().filter(|p| p.owner != player && p.owner != -1).collect();
    let mut frontline: Vec<&Planet> = my
        .iter()
        .copied()
        .filter(|p| {
            enemyish
                .iter()
                .map(|e| dist(pos_now(state, p), pos_now(state, e)))
                .fold(f64::INFINITY, f64::min)
                < 34.0
        })
        .collect();
    if frontline.is_empty() {
        frontline = my.clone();
    }
    frontline.sort_by_key(|p| p.ships);
    let target = frontline[0];
    let mut plan = Vec::new();
    for src in my {
        if src.id == target.id || src.ships < 8 {
            continue;
        }
        let enemy_dist = enemyish
            .iter()
            .map(|e| dist(pos_now(state, src), pos_now(state, e)))
            .fold(f64::INFINITY, f64::min);
        let target_enemy_dist = enemyish
            .iter()
            .map(|e| dist(pos_now(state, target), pos_now(state, e)))
            .fold(f64::INFINITY, f64::min);
        if enemy_dist <= target_enemy_dist + 8.0 {
            continue;
        }
        let ships = ((src.ships as f64) * 0.45).round() as i64;
        if let Some(shot) = shot_from(cache, src, target, ships) {
            plan.push((shot.src_id, shot.angle, shot.ships, player));
        }
    }
    plan
}

pub fn candidates(
    state: &GameState,
    player: i32,
    cache: &EntityCache,
    max_plans: usize,
    root: bool,
) -> Vec<Vec<Action>> {
    let targets = target_infos(state, player, root);
    let source_cap = env_usize(
        if root { "OW_MOVEGEN_ROOT_SOURCES" } else { "OW_MOVEGEN_NODE_SOURCES" },
        if root { ROOT_SOURCES } else { NODE_SOURCES },
    );
    let mut out = Vec::with_capacity(max_plans + 2);
    let mut seen: HashSet<Vec<(i64, i64, i64, i32)>> = HashSet::new();

    for target in targets.iter().take(max_plans.saturating_mul(2).max(3)) {
        let p = target.planet;
        push_unique(
            &mut out,
            &mut seen,
            focused_capture(state, player, cache, p, 0.22, 1.04, source_cap),
        );
        if out.len() >= max_plans {
            break;
        }
        push_unique(
            &mut out,
            &mut seen,
            focused_capture(state, player, cache, p, 0.05, 1.25, source_cap),
        );
        if out.len() >= max_plans {
            break;
        }
        if p.owner != -1 {
            push_unique(&mut out, &mut seen, pressure_attack(state, player, cache, p, 0.55));
            if out.len() >= max_plans {
                break;
            }
        }
    }

    if out.len() < max_plans {
        push_unique(&mut out, &mut seen, greedy_multi(state, player, cache, &targets));
    }
    if out.len() < max_plans {
        push_unique(&mut out, &mut seen, reinforcement_plan(state, player, cache));
    }
    if out.len() > max_plans {
        out.truncate(max_plans);
    }
    let has_noop = out.iter().any(|p| p.is_empty());
    if !has_noop && max_plans > 0 {
        if out.len() >= max_plans {
            out.pop();
        }
        push_unique(&mut out, &mut seen, Vec::new());
    }
    out
}
