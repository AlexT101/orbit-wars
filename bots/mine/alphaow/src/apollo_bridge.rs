//! Bridge between alphaow's `GameState` and the vendored apollo engine, so
//! alphaow's MCTS can use apollo's `hellburner::search_candidates` as its child
//! candidate generator.
//!
//! Conversions:
//!   - alphaow `Planet`/`Fleet`/`CometGroup` -> apollo `engine` equivalents
//!     (apollo uses `owner: i64`, comet paths as `[f64; 2]`).
//!   - apollo `initial_planets` is reconstructed from each planet's orbital
//!     params (`CENTER + r*(cos,sin)(initial_angle)`); for static planets that
//!     equals the current position. Comet planets are included but apollo's
//!     `EntityCache` skips them (it builds comet entities from the comet paths).
//!   - apollo `FleetOrder = (from_id, angle, ships)` -> alphaow
//!     `Action = (from_id, angle, ships, owner)` with `owner = player`.

use crate::apollo::engine::{CometGroup as ACometGroup, Fleet as AFleet, Planet as APlanet};
use crate::apollo::entity_cache::EntityCache;
use crate::apollo::hellburner;
use crate::apollo::world::WorldState;
use crate::sim::Action;
use crate::{GameState, Planet, CENTER_X, CENTER_Y};

fn to_apollo_planet_current(p: &Planet) -> APlanet {
    APlanet {
        id: p.id,
        owner: p.owner as i64,
        x: p.x,
        y: p.y,
        radius: p.radius,
        ships: p.ships,
        production: p.production,
    }
}

fn to_apollo_planet_initial(p: &Planet) -> APlanet {
    // Step-0 position from orbital params. For non-orbiting (static) planets the
    // reconstruction reproduces the fixed position; comet planets are skipped by
    // EntityCache so their value here is irrelevant.
    let (ix, iy) = if p.is_comet {
        (p.x, p.y)
    } else {
        (
            CENTER_X + p.orbital_radius * p.initial_angle.cos(),
            CENTER_Y + p.orbital_radius * p.initial_angle.sin(),
        )
    };
    APlanet {
        id: p.id,
        owner: p.owner as i64,
        x: ix,
        y: iy,
        radius: p.radius,
        ships: p.ships,
        production: p.production,
    }
}

fn to_apollo_fleet(f: &crate::Fleet) -> AFleet {
    AFleet {
        id: f.id,
        owner: f.owner as i64,
        x: f.x,
        y: f.y,
        angle: f.angle,
        from_planet_id: f.from_planet_id,
        ships: f.ships,
    }
}

fn to_apollo_comets(state: &GameState) -> (Vec<ACometGroup>, Vec<i64>) {
    let mut comet_planet_ids: Vec<i64> = Vec::new();
    let comets: Vec<ACometGroup> = state
        .comets
        .iter()
        .map(|g| {
            comet_planet_ids.extend(g.planet_ids.iter().copied());
            ACometGroup {
                planet_ids: g.planet_ids.clone(),
                paths: g
                    .paths
                    .iter()
                    .map(|path| path.iter().map(|&(x, y)| [x, y]).collect())
                    .collect(),
                path_index: g.path_index,
            }
        })
        .collect();
    (comets, comet_planet_ids)
}

/// Build an apollo `EntityCache` from the leaf state. Orbiter geometry is fixed
/// for the whole game (orbital params are set at parse and survive `tick`), so a
/// single cache can be reused across every tick of a rollout — call
/// [`refresh_cache_comets`] when the comet set changes and `set_current_turn`
/// before each plan.
pub fn rollout_cache(state: &GameState) -> EntityCache {
    let initial_planets: Vec<APlanet> =
        state.planets.iter().map(to_apollo_planet_initial).collect();
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    EntityCache::build(
        &initial_planets,
        &comets,
        &comet_planet_ids,
        state.angular_velocity,
        state.step,
    )
}

/// Sync the cache's comet entities with the current state (adds spawned comets,
/// drops expired ones, clears the blocker-table cache). Only call when the comet
/// id set actually changed — it discards cached blocker tables.
pub fn refresh_cache_comets(cache: &mut EntityCache, state: &GameState) {
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    cache.refresh_comets(&comets, &comet_planet_ids, state.step);
}

/// apollo's greedy hellburner plan for `player` as an alphaow launch list,
/// reusing a prebuilt `cache`. Caller must `cache.set_current_turn(state.step)`
/// (and refresh comets if needed) beforehand.
pub fn apollo_plan(state: &GameState, player: i32, cache: &EntityCache) -> Vec<Action> {
    let planets: Vec<APlanet> = state.planets.iter().map(to_apollo_planet_current).collect();
    let initial_planets: Vec<APlanet> =
        state.planets.iter().map(to_apollo_planet_initial).collect();
    let fleets: Vec<AFleet> = state.fleets.iter().map(to_apollo_fleet).collect();
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    let world = WorldState::build(
        player as i64,
        state.step,
        planets,
        fleets,
        initial_planets,
        comets,
        comet_planet_ids,
        state.angular_velocity,
        cache,
    );
    hellburner::plan(&world)
        .into_iter()
        .map(|(from_id, angle, ships)| (from_id, angle, ships, player))
        .collect()
}

/// Generate apollo's hellburner child candidates for `player`, each converted to
/// an alphaow launch list. Returns one `Vec<Action>` per candidate strategy.
pub fn apollo_candidates(state: &GameState, player: i32) -> Vec<Vec<Action>> {
    let planets: Vec<APlanet> = state.planets.iter().map(to_apollo_planet_current).collect();
    let initial_planets: Vec<APlanet> =
        state.planets.iter().map(to_apollo_planet_initial).collect();
    let fleets: Vec<AFleet> = state.fleets.iter().map(to_apollo_fleet).collect();
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    let av = state.angular_velocity;
    let step = state.step;

    let cache = EntityCache::build(&initial_planets, &comets, &comet_planet_ids, av, step);
    let world = WorldState::build(
        player as i64,
        step,
        planets,
        fleets,
        initial_planets,
        comets,
        comet_planet_ids,
        av,
        &cache,
    );

    hellburner::search_candidates(&world)
        .into_iter()
        .map(|orders| {
            orders
                .into_iter()
                .map(|(from_id, angle, ships)| (from_id, angle, ships, player))
                .collect::<Vec<Action>>()
        })
        .collect()
}

// ── Focused single-target candidate generator ────────────────────────────
//
// One candidate plan per "target" (= planet I won't own at end of fleet
// extrapolation). Each plan = (a) apollo's best capture orders for that
// target, plus (b) "healing" fleets from rear-line "support" planets to
// frontline planets that are sending ships this turn.
//
// Algorithm:
//   1. Run `value_net::extrapolate_fleets` -> per-planet (owner, ships)
//      after all in-flight fleets resolve.
//   2. Targets = planets whose extrapolated owner is not me.
//   3. For each target, call `hellburner::evaluate_one_target` -> a list of
//      PlannedOrders. Keep only `effective_offset == 0` (= physically
//      launching this turn) -> these become the attack orders for this
//      candidate plan. Frontline = the set of source planet ids in these.
//   4. Support pool = my planets NOT in frontline AND NOT in the target
//      set, sorted by distance to nearest enemy planet DESC (furthest
//      from enemy first).
//   5. For each support (in order), compute its `safe_drain` (max ships
//      it can send without ceasing to be mine at the extrapolation
//      horizon — found by binary search). Distribute up to `safe_drain`
//      ships among frontline planets in order of distance ASC (closest
//      first), capped per-frontline at the number of ships it sent out
//      this turn (so we "heal" each frontline exactly back to its
//      pre-turn count, no more).
//   6. Concatenate attack orders + healing orders -> one candidate plan.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

const SAFE_DRAIN_RESERVE: i64 = 1; // always leave ≥1 ship after drain check
const MAX_TARGETS: usize = 32;     // hard cap so we never explode candidate count

fn dist2(a: &Planet, b: &Planet) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

fn min_enemy_dist2(p: &Planet, enemy: &[&Planet]) -> f64 {
    enemy
        .iter()
        .map(|e| dist2(p, e))
        .fold(f64::INFINITY, f64::min)
}

/// Predict whether `planet_id` is still owned by `player` at the end of the
/// extrapolation horizon when we drain `drain` ships from it this turn.
/// Simple model: we look at the planet's extrapolated (owner, ships); if
/// already not ours -> always false. If ours: we keep it iff
/// `extrap_ships >= drain + SAFE_DRAIN_RESERVE`.
///
/// This deliberately ignores second-order effects (the drained ships going
/// elsewhere -> different combat outcomes). Good enough as a per-source
/// upper bound; we err on the conservative side.
fn stays_mine_after_drain(
    extrap: &HashMap<i64, (i32, i64)>,
    planet_id: i64,
    drain: i64,
    player: i32,
) -> bool {
    if let Some((owner, ships)) = extrap.get(&planet_id) {
        *owner == player && *ships - drain >= SAFE_DRAIN_RESERVE
    } else {
        // not in extrap -> no in-flight changes; treat current state as final
        false
    }
}

fn safe_drain(extrap: &HashMap<i64, (i32, i64)>, planet: &Planet, player: i32) -> i64 {
    // current upper bound: planet.ships - reserve
    let cap = (planet.ships - SAFE_DRAIN_RESERVE).max(0);
    if cap == 0 {
        return 0;
    }
    // largest X in [0, cap] such that draining X keeps the planet mine
    if stays_mine_after_drain(extrap, planet.id, cap, player) {
        return cap;
    }
    // binary search down
    let mut lo = 0i64;
    let mut hi = cap;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if stays_mine_after_drain(extrap, planet.id, mid, player) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// One candidate plan per target. See module-level algorithm comment above.
pub fn focused_candidates(state: &GameState, player: i32) -> Vec<Vec<Action>> {
    use crate::value_net::extrapolate_fleets;

    // Step 1-2: targets = planets I won't own at end of extrapolation.
    let extrap: HashMap<i64, (i32, i64)> = extrapolate_fleets(state)
        .into_iter()
        .collect();
    let target_ids: Vec<i64> = {
        let mut ids: Vec<(i64, f64)> = state
            .planets
            .iter()
            .filter_map(|p| {
                let (owner, _) = extrap.get(&p.id).copied().unwrap_or((p.owner, p.ships));
                if owner == player {
                    None
                } else {
                    // closer-to-me-now targets first (loose prioritisation; DUCT
                    // will rerank via PUCT anyway)
                    let nearest_mine = state
                        .planets
                        .iter()
                        .filter(|q| q.owner == player)
                        .map(|q| dist2(p, q))
                        .fold(f64::INFINITY, f64::min);
                    Some((p.id, nearest_mine))
                }
            })
            .collect();
        ids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ids.into_iter().take(MAX_TARGETS).map(|(id, _)| id).collect()
    };

    if target_ids.is_empty() {
        return vec![Vec::new()];
    }

    // Build apollo WorldState (one-time per call).
    let planets: Vec<APlanet> = state.planets.iter().map(to_apollo_planet_current).collect();
    let initial_planets: Vec<APlanet> = state.planets.iter().map(to_apollo_planet_initial).collect();
    let fleets: Vec<AFleet> = state.fleets.iter().map(to_apollo_fleet).collect();
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    let av = state.angular_velocity;
    let step = state.step;
    let cache = EntityCache::build(&initial_planets, &comets, &comet_planet_ids, av, step);
    let world = WorldState::build(
        player as i64, step, planets, fleets, initial_planets, comets, comet_planet_ids, av,
        &cache,
    );

    let target_set: HashSet<i64> = target_ids.iter().copied().collect();

    let mut plans: Vec<Vec<Action>> = Vec::with_capacity(target_ids.len());
    for &tgt in &target_ids {
        // Step 3: attack orders via apollo's single-target evaluator.
        let (raw_orders, _max_arrival) = match hellburner::evaluate_one_target(&world, tgt) {
            Some(x) => x,
            None => continue, // target not capturable from this state — skip
        };
        let attack_orders: Vec<Action> = raw_orders
            .iter()
            .filter(|(_, _, _, off)| *off == 0) // only "physically launching this turn"
            .map(|&(src, ang, ships, _)| (src, ang, ships, player))
            .collect();
        if attack_orders.is_empty() {
            // Apollo wants to start an attack on a later turn — skip; no
            // immediate action to commit for this target.
            continue;
        }

        // Frontline = { (src_id, ships_sent_out_this_turn) }, summed across
        // multiple orders from the same source.
        let mut frontline: HashMap<i64, i64> = HashMap::default();
        for (src, _, ships, _) in &attack_orders {
            *frontline.entry(*src).or_insert(0) += ships;
        }

        // Step 4: support pool — my planets not in frontline and not a
        // target, sorted by distance to nearest enemy DESC.
        let enemy_planets: Vec<&Planet> = state
            .planets
            .iter()
            .filter(|p| p.owner != player && p.owner != -1)
            .collect();
        let mut supports: Vec<(&Planet, f64)> = state
            .planets
            .iter()
            .filter(|p| {
                p.owner == player
                    && !frontline.contains_key(&p.id)
                    && !target_set.contains(&p.id)
            })
            .map(|p| (p, min_enemy_dist2(p, &enemy_planets)))
            .collect();
        // furthest from enemy first
        supports.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Step 5: build healing orders.
        let mut healing_orders: Vec<Action> = Vec::new();
        let mut healed_received: HashMap<i64, i64> = HashMap::default();
        for (support, _) in supports {
            let mut budget = safe_drain(&extrap, support, player);
            if budget == 0 {
                continue;
            }
            // frontline sorted by distance to this support ASC (closest first)
            let mut fl: Vec<(i64, i64)> = frontline.iter().map(|(k, v)| (*k, *v)).collect();
            fl.sort_by(|a, b| {
                let pa = state.planets.iter().find(|p| p.id == a.0).unwrap();
                let pb = state.planets.iter().find(|p| p.id == b.0).unwrap();
                dist2(support, pa)
                    .partial_cmp(&dist2(support, pb))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (fl_id, fl_sent_out) in fl {
                if budget == 0 {
                    break;
                }
                let already = *healed_received.get(&fl_id).unwrap_or(&0);
                let needed = fl_sent_out - already;
                if needed <= 0 {
                    continue;
                }
                let send = budget.min(needed);
                // angle from support to frontline planet
                let target_planet = state.planets.iter().find(|p| p.id == fl_id).unwrap();
                let angle = (target_planet.y - support.y).atan2(target_planet.x - support.x);
                healing_orders.push((support.id, angle, send, player));
                *healed_received.entry(fl_id).or_insert(0) += send;
                budget -= send;
            }
        }

        // Step 6: combine.
        let mut plan = attack_orders;
        plan.extend(healing_orders);
        plans.push(plan);
    }

    if plans.is_empty() {
        return vec![Vec::new()];
    }
    plans
}
