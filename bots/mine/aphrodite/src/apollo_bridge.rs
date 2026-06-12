//! Bridge between aphrodite's `GameState` and the vendored apollo engine, so
//! aphrodite's MCTS can use apollo's `strategy::search_candidates` as its child
//! candidate generator.
//!
//! Conversions:
//!   - aphrodite `Planet`/`Fleet`/`CometGroup` -> apollo `engine` equivalents
//!     (apollo uses `owner: i64`, comet paths as `[f64; 2]`).
//!   - apollo `initial_planets` is reconstructed from each planet's orbital
//!     params (`CENTER + r*(cos,sin)(initial_angle)`); for static planets that
//!     equals the current position. Comet planets are included but apollo's
//!     `EntityCache` skips them (it builds comet entities from the comet paths).
//!   - apollo `FleetOrder = (from_id, angle, ships)` -> aphrodite
//!     `Action = (from_id, angle, ships, owner)` with `owner = player`.

use crate::apollo::cache::EntityCache;
use crate::apollo::constants::Config;
use crate::apollo::engine::{
    CometGroup as ACometGroup, EngineState, Fleet as AFleet, MoveAction, Planet as APlanet,
    Simulator,
};
use crate::apollo::helpers::{count_alive_players, count_players, ArrivalLedger};
use crate::apollo::strategy;
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

/// Convert the aphrodite `GameState` into apollo's **player-agnostic**
/// `EngineState` (planet/fleet/comet conversion + player count + next fleet id).
/// This plus the `Simulator` and `ArrivalLedger` derived from it are identical
/// for every player, so building them once and deriving each player's
/// `WorldState` via [`WorldState::from_simulator_with_ledger`] avoids repeating
/// the expensive `HORIZON`-turn ledger walk per player.
fn build_engine(state: &GameState) -> EngineState {
    let planets: Vec<APlanet> = state.planets.iter().map(to_apollo_planet_current).collect();
    let initial_planets: Vec<APlanet> =
        state.planets.iter().map(to_apollo_planet_initial).collect();
    let fleets: Vec<AFleet> = state.fleets.iter().map(to_apollo_fleet).collect();
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    let num_players = count_players(&planets, &fleets);
    let next_fleet_id = fleets
        .iter()
        .map(|f| f.id)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
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
fn candidates_from_ledger(
    sim: &Simulator,
    ledger: &ArrivalLedger,
    player: i32,
    cache: &EntityCache,
    rollout_internal: bool,
) -> Vec<Vec<Action>> {
    let mut world = WorldState::from_simulator_with_ledger(player as i64, sim, ledger, cache);
    world.rollout_internal = rollout_internal;
    strategy::search_candidates_subsets(&world)
        .into_iter()
        .map(|orders| {
            orders
                .into_iter()
                .map(|m| (m.from_id, m.angle, m.ships, player))
                .collect::<Vec<Action>>()
        })
        .collect()
}

/// Hellburner child candidates for `me` and `opp` from a single shared
/// `Simulator` + `ArrivalLedger` (one `HORIZON`-turn walk for both players).
/// Caller must `cache.set_current_turn(state.step)` first. `rollout_internal`
/// is forwarded onto both `WorldState`s so the early-game opening DFS stands
/// down at non-root nodes (see `early_game::plan_opening`).
pub fn apollo_candidates_pair(
    state: &GameState,
    me: i32,
    opp: i32,
    cache: &EntityCache,
    rollout_internal: bool,
) -> (Vec<Vec<Action>>, Vec<Vec<Action>>) {
    let engine = build_engine(state);
    let sim = Simulator::new(&engine);
    let horizon = Config::for_alive(count_alive_players(sim.planets(), sim.fleets())).horizon;
    let ledger = ArrivalLedger::build(&sim, horizon, cache);
    (
        candidates_from_ledger(&sim, &ledger, me, cache, rollout_internal),
        candidates_from_ledger(&sim, &ledger, opp, cache, rollout_internal),
    )
}

/// Generate apollo's hellburner child candidates for `player`, each converted to
/// an aphrodite launch list. Returns one `Vec<Action>` per candidate strategy.
///
/// Reuses a prebuilt, shared `cache` (the obstacle/aim geometry is owner-agnostic
/// and game-static, so one cache serves every node of every turn). Caller must
/// `cache.set_current_turn(state.step)` (and refresh comets if needed) first.
pub fn apollo_candidates(
    state: &GameState,
    player: i32,
    cache: &EntityCache,
    rollout_internal: bool,
) -> Vec<Vec<Action>> {
    let planets: Vec<APlanet> = state.planets.iter().map(to_apollo_planet_current).collect();
    let initial_planets: Vec<APlanet> =
        state.planets.iter().map(to_apollo_planet_initial).collect();
    let fleets: Vec<AFleet> = state.fleets.iter().map(to_apollo_fleet).collect();
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    let mut world = WorldState::build(
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
    world.rollout_internal = rollout_internal;

    strategy::search_candidates_subsets(&world)
        .into_iter()
        .map(|orders| {
            orders
                .into_iter()
                .map(|m| (m.from_id, m.angle, m.ships, player))
                .collect::<Vec<Action>>()
        })
        .collect()
}

/// Apollo's single greedy `ScorePerShip` plan for `player` (apollo's
/// `STRATEGIES[0]`, via [`strategy::plan`]), converted to aphrodite launches.
/// This is the cheap "assumed reply" used for the non-branched minor players in
/// 4p DUCT expansion: every player commits privately from the same observed node
/// state, so a minor player's launches are a pure function of `state` and can be
/// computed once per node. Reuses the shared owner-agnostic `cache`; caller must
/// `cache.set_current_turn(state.step)` first.
///
/// This is apollo's cheap in-rollout reply policy, so it always runs with
/// `rollout_internal = true`: the early-game opening DFS stands down (a
/// per-minor-player DFS at every node would blow the turn budget).
pub fn apollo_greedy(state: &GameState, player: i32, cache: &EntityCache) -> Vec<Action> {
    let planets: Vec<APlanet> = state.planets.iter().map(to_apollo_planet_current).collect();
    let initial_planets: Vec<APlanet> =
        state.planets.iter().map(to_apollo_planet_initial).collect();
    let fleets: Vec<AFleet> = state.fleets.iter().map(to_apollo_fleet).collect();
    let (comets, comet_planet_ids) = to_apollo_comets(state);
    let mut world = WorldState::build(
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
    world.rollout_internal = true;
    strategy::plan(&world)
        .into_iter()
        .map(|m| (m.from_id, m.angle, m.ships, player))
        .collect()
}

/// Recover the destination planet a launched fleet was aimed at by matching its
/// launch `angle` against `plan_shot(from, c, ships)` for every non-comet planet
/// `c`. aphrodite's `Action` tuple carries only `(from_id, angle, ships, owner)` —
/// the `target` that apollo threads through `MoveAction` is dropped at the tuple
/// boundary — so we re-derive it here (somewhat inefficiently but we only have a few fleets and planets).
/// when nothing matches, which makes `redirect_moves` leave the fleet untouched.
fn recover_target(model: &strategy::HellburnerModel, from_id: i64, angle: f64, ships: i64) -> i64 {
    let mut best: Option<(f64, i64)> = None;
    for &c in &model.non_comet_ids {
        if c == from_id {
            continue;
        }
        if let Some((a, _, _, _, _)) = model.plan_shot(from_id, c, ships, 0) {
            let d = (a - angle).abs();
            if best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, c));
            }
        }
    }
    match best {
        Some((d, c)) if d < 1e-6 => c,
        _ => -1,
    }
}

/// Final no-loss reroute pass over the move set the planner has already chosen,
/// mirroring apollo's `redirect_moves` call at the tail of `Bot::get_action`.
/// This runs after DUCT has fully committed to a plan — it never
/// influences the search, it only rewrites the moves we are about to emit. For
/// each launch `A → B`, if routing through an intermediate ally `C` reaches `B`
/// no later (`A → C → B`), the fleet is retargeted to `C`.
pub fn redirect_actions(state: &GameState, player: i32, actions: Vec<Action>) -> Vec<Action> {
    if actions.is_empty() {
        return actions;
    }
    let mut cache = rollout_cache(state);
    cache.set_current_turn(state.step);

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
        &cache,
    );
    let model = strategy::HellburnerModel::build(&world);

    // Reconstruct apollo MoveActions for our own launches; pass any non-`player`
    // actions through untouched (best_move returns only `player`'s today).
    let mut moves: Vec<MoveAction> = Vec::with_capacity(actions.len());
    let mut passthrough: Vec<Action> = Vec::new();
    for a in &actions {
        let (from_id, angle, ships, owner) = *a;
        if owner != player {
            passthrough.push(*a);
            continue;
        }
        let target = recover_target(&model, from_id, angle, ships);
        moves.push(MoveAction {
            from_id,
            angle,
            ships,
            target,
        });
    }

    let mut out: Vec<Action> = strategy::redirect_moves(&world, moves)
        .into_iter()
        .map(|m| (m.from_id, m.angle, m.ships, player))
        .collect();
    out.extend(passthrough);
    out
}
