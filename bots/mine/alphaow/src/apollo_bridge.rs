//! Bridge between alphaow's `GameState` and the vendored apollo engine, so
//! alphaow's MCTS can use apollo's `strategy::search_candidates` as its child
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

use crate::apollo::constants::HORIZON;
use crate::apollo::engine::{
    CometGroup as ACometGroup, EngineState, Fleet as AFleet, Planet as APlanet, Simulator,
};
use crate::apollo::cache::EntityCache;
use crate::apollo::strategy;
use crate::apollo::helpers::{count_players, ArrivalLedger};
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
/// drops expired ones). Only updates `entities`; the aim cache is left intact
/// (it is keyed by absolute launch turn, and new comets are handled by
/// `aim_cache_lookup`'s post-spawn re-verification). Call only when the comet id
/// set actually changed (i.e. on a `COMET_SPAWN_STEPS` boundary).
pub fn refresh_cache_comets(cache: &mut EntityCache, state: &GameState) {
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    cache.refresh_comets(&comets, &comet_planet_ids, state.step);
}

/// Convert the alphaow `GameState` into apollo's **player-agnostic**
/// `EngineState` (planet/fleet/comet conversion + player count + next fleet id).
/// This plus the `Simulator` and `ArrivalLedger` derived from it are identical
/// for every player, so building them once and deriving each player's
/// `WorldState` via [`WorldState::from_simulator_with_ledger`] avoids repeating
/// the expensive `HORIZON`-turn ledger walk per player — mirroring apollo's
/// ledger sharing in `rollout::rollout_score` / `pick_plan_by_rollout`.
fn build_engine(state: &GameState) -> EngineState {
    let planets: Vec<APlanet> = state.planets.iter().map(to_apollo_planet_current).collect();
    let initial_planets: Vec<APlanet> =
        state.planets.iter().map(to_apollo_planet_initial).collect();
    let fleets: Vec<AFleet> = state.fleets.iter().map(to_apollo_fleet).collect();
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    let num_players = count_players(&planets, &fleets);
    let next_fleet_id = fleets.iter().map(|f| f.id).max().map(|m| m + 1).unwrap_or(0);
    EngineState::from_observation_parts(
        state.step,
        state.angular_velocity,
        planets,
        initial_planets,
        fleets,
        next_fleet_id,
        comet_planet_ids,
        comets,
        num_players,
    )
}

#[inline]
fn plan_from_ledger(
    sim: &Simulator,
    ledger: &ArrivalLedger,
    player: i32,
    cache: &EntityCache,
) -> Vec<Action> {
    let world = WorldState::from_simulator_with_ledger(player as i64, sim, ledger, cache);
    strategy::plan(&world)
        .into_iter()
        .map(|m| (m.from_id, m.angle, m.ships, player))
        .collect()
}

#[inline]
fn candidates_from_ledger(
    sim: &Simulator,
    ledger: &ArrivalLedger,
    player: i32,
    cache: &EntityCache,
) -> Vec<Vec<Action>> {
    let world = WorldState::from_simulator_with_ledger(player as i64, sim, ledger, cache);
    strategy::search_candidates(&world)
        .into_iter()
        .map(|orders| {
            orders
                .into_iter()
                .map(|m| (m.from_id, m.angle, m.ships, player))
                .collect::<Vec<Action>>()
        })
        .collect()
}

/// Greedy hellburner plans for `me` and (optionally) `opp` from a single shared
/// `Simulator` + `ArrivalLedger`. Use this instead of two [`apollo_plan`] calls
/// when planning both players at the same state (e.g. a rollout reactive tick) —
/// it pays the `HORIZON`-turn ledger walk once. Caller must
/// `cache.set_current_turn(state.step)` first.
pub fn apollo_plan_pair(
    state: &GameState,
    me: i32,
    opp: Option<i32>,
    cache: &EntityCache,
) -> (Vec<Action>, Vec<Action>) {
    let engine = build_engine(state);
    let sim = Simulator::new(&engine);
    let ledger = ArrivalLedger::build(&sim, HORIZON, cache);
    let my = plan_from_ledger(&sim, &ledger, me, cache);
    let op = opp
        .map(|o| plan_from_ledger(&sim, &ledger, o, cache))
        .unwrap_or_default();
    (my, op)
}

/// Hellburner child candidates for `me` and `opp` from a single shared
/// `Simulator` + `ArrivalLedger` (one `HORIZON`-turn walk for both players).
/// Caller must `cache.set_current_turn(state.step)` first.
pub fn apollo_candidates_pair(
    state: &GameState,
    me: i32,
    opp: i32,
    cache: &EntityCache,
) -> (Vec<Vec<Action>>, Vec<Vec<Action>>) {
    let engine = build_engine(state);
    let sim = Simulator::new(&engine);
    let ledger = ArrivalLedger::build(&sim, HORIZON, cache);
    (
        candidates_from_ledger(&sim, &ledger, me, cache),
        candidates_from_ledger(&sim, &ledger, opp, cache),
    )
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
    strategy::plan(&world)
        .into_iter()
        .map(|m| (m.from_id, m.angle, m.ships, player))
        .collect()
}

/// Generate apollo's hellburner child candidates for `player`, each converted to
/// an alphaow launch list. Returns one `Vec<Action>` per candidate strategy.
///
/// Reuses a prebuilt, shared `cache` (the obstacle/aim geometry is owner-agnostic
/// and game-static, so one cache serves every node of every turn). Caller must
/// `cache.set_current_turn(state.step)` (and refresh comets if needed) first,
/// exactly like [`apollo_plan`].
pub fn apollo_candidates(state: &GameState, player: i32, cache: &EntityCache) -> Vec<Vec<Action>> {
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

    strategy::search_candidates(&world)
        .into_iter()
        .map(|orders| {
            orders
                .into_iter()
                .map(|m| (m.from_id, m.angle, m.ships, player))
                .collect::<Vec<Action>>()
        })
        .collect()
}

// (focused single-target candidate generator moved to src/focused_plan.rs)

