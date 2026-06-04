from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1.0
MAX_PLAYERS = 4
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)


@dataclass(frozen=True)
class Limits:
    """Static shape limits for the compiled JAX environment."""

    max_planets: int = 44
    max_fleets: int = 256
    max_actions: int = 44
    max_comet_path: int = 40


@dataclass(frozen=True)
class Configuration:
    episode_steps: int = 500
    ship_speed: float = 6.0
    comet_speed: float = 4.0


@dataclass(frozen=True)
class RewardWeights:
    terminal: float = 1.0
    terminal_time: float = 0.0
    production_income: float = 0.0
    launch_penalty: float = 0.0


class State(NamedTuple):
    step: jax.Array
    angular_velocity: jax.Array
    planets: jax.Array
    initial_planets: jax.Array
    planet_count: jax.Array
    fleets: jax.Array
    fleet_count: jax.Array
    next_fleet_id: jax.Array
    comet_group_active: jax.Array
    comet_path_index: jax.Array
    comet_planet_ids: jax.Array
    comet_paths: jax.Array
    comet_path_lengths: jax.Array
    comet_ships: jax.Array
    done: jax.Array
    rewards: jax.Array
    reward_components: jax.Array
    seed: jax.Array
    num_players: jax.Array
    episode_steps: jax.Array
    ship_speed: jax.Array
    reward_weights: jax.Array


@lru_cache(maxsize=1)
def _official_module() -> ModuleType:
    official_path = Path("/home/ec2-user/official_env/orbit_wars.py")
    if official_path.exists():
        spec = importlib.util.spec_from_file_location("_official_orbit_wars", official_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {official_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from kaggle_environments.envs.orbit_wars import orbit_wars

    return orbit_wars


def _pad_rows(rows: list[list[Any]], limit: int, width: int = 7) -> np.ndarray:
    if len(rows) > limit:
        raise ValueError(f"{len(rows)} rows exceed static limit {limit}")
    out = np.zeros((limit, width), dtype=np.float64)
    if rows:
        out[: len(rows), :] = np.asarray(rows, dtype=np.float64)
    return out


def _empty_state_arrays(limits: Limits) -> dict[str, np.ndarray]:
    return {
        "planets": np.zeros((limits.max_planets, 7), dtype=np.float64),
        "initial_planets": np.zeros((limits.max_planets, 7), dtype=np.float64),
        "fleets": np.zeros((limits.max_fleets, 7), dtype=np.float64),
        "comet_planet_ids": np.full((5, 4), -1.0, dtype=np.float64),
        "comet_paths": np.zeros((5, 4, limits.max_comet_path, 2), dtype=np.float64),
        "comet_path_lengths": np.zeros((5,), dtype=np.int32),
    }


def _precompute_comets(
    official: ModuleType,
    seed: int,
    initial_planets: list[list[Any]],
    angular_velocity: float,
    configuration: Configuration,
    limits: Limits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    comet_planet_ids = np.full((5, 4), -1.0, dtype=np.float64)
    comet_paths = np.zeros((5, 4, limits.max_comet_path, 2), dtype=np.float64)
    comet_path_lengths = np.zeros((5,), dtype=np.int32)
    comet_ships = np.zeros((5,), dtype=np.float64)

    base_planets = [list(p) for p in initial_planets]
    current_initial = [list(p) for p in base_planets]
    current_comet_ids: list[int] = []
    max_base_id = int(max(p[0] for p in base_planets)) if base_planets else -1

    for spawn_slot, spawn_step in enumerate(COMET_SPAWN_STEPS):
        # Comet paths are at most 40 turns and spawn intervals are 100 turns,
        # so every previous group has expired before the next scheduled spawn.
        current_initial = [list(p) for p in base_planets]
        current_comet_ids = []
        rng = __import__("random").Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        paths = official.generate_comet_paths(
            current_initial,
            angular_velocity,
            spawn_step,
            current_comet_ids,
            configuration.comet_speed,
            rng=rng,
        )
        if not paths:
            continue
        path_len = len(paths[0])
        if path_len > limits.max_comet_path:
            raise ValueError(f"comet path length {path_len} exceeds limit {limits.max_comet_path}")
        next_id = max_base_id + 1
        # Consume ship-count RNG draws so future parity remains exact if this
        # helper is extended. The ships value is deterministic but not part of
        # the path arrays.
        ship_count = min(rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99))
        for i, path in enumerate(paths):
            comet_planet_ids[spawn_slot, i] = next_id + i
            comet_paths[spawn_slot, i, :path_len, :] = np.asarray(path, dtype=np.float64)
        comet_path_lengths[spawn_slot] = path_len
        comet_ships[spawn_slot] = ship_count
    return comet_planet_ids, comet_paths, comet_path_lengths, comet_ships


def reset(
    seed: int,
    num_players: int = 2,
    configuration: Configuration | None = None,
    reward_weights: RewardWeights | None = None,
    limits: Limits | None = None,
) -> State:
    if num_players not in (2, 4):
        raise ValueError(f"orbit_wars supports 2 or 4 players, got {num_players}")
    configuration = configuration or Configuration()
    reward_weights = reward_weights or RewardWeights()
    limits = limits or Limits()
    official = _official_module()

    rng = __import__("random").Random(int(seed))
    angular_velocity = rng.uniform(0.025, 0.05)
    planets = official.generate_planets(rng)
    initial_planets = [list(p) for p in planets]

    num_groups = len(planets) // 4
    if num_groups > 0:
        home_group = rng.randint(0, num_groups - 1)
        base = home_group * 4
        if num_players == 2:
            planets[base][1] = 0
            planets[base][5] = 10
            planets[base + 3][1] = 1
            planets[base + 3][5] = 10
        else:
            for j in range(4):
                planets[base + j][1] = j
                planets[base + j][5] = 10

    if len(planets) > limits.max_planets:
        raise ValueError(f"{len(planets)} planets exceed static limit {limits.max_planets}")
    arrays = _empty_state_arrays(limits)
    arrays["planets"] = _pad_rows(planets, limits.max_planets)
    arrays["initial_planets"] = _pad_rows(initial_planets, limits.max_planets)
    (
        arrays["comet_planet_ids"],
        arrays["comet_paths"],
        arrays["comet_path_lengths"],
        comet_ships,
    ) = _precompute_comets(official, int(seed), initial_planets, angular_velocity, configuration, limits)

    rw = np.asarray(
        [
            reward_weights.terminal,
            reward_weights.terminal_time,
            reward_weights.production_income,
            reward_weights.launch_penalty,
        ],
        dtype=np.float64,
    )
    return State(
        step=jnp.asarray(0, dtype=jnp.int32),
        angular_velocity=jnp.asarray(angular_velocity, dtype=jnp.float64),
        planets=jnp.asarray(arrays["planets"]),
        initial_planets=jnp.asarray(arrays["initial_planets"]),
        planet_count=jnp.asarray(len(planets), dtype=jnp.int32),
        fleets=jnp.asarray(arrays["fleets"]),
        fleet_count=jnp.asarray(0, dtype=jnp.int32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        comet_group_active=jnp.zeros((5,), dtype=jnp.bool_),
        comet_path_index=jnp.full((5,), -2, dtype=jnp.int32),
        comet_planet_ids=jnp.asarray(arrays["comet_planet_ids"]),
        comet_paths=jnp.asarray(arrays["comet_paths"]),
        comet_path_lengths=jnp.asarray(arrays["comet_path_lengths"]),
        comet_ships=jnp.asarray(comet_ships),
        done=jnp.asarray(False),
        rewards=jnp.zeros((MAX_PLAYERS,), dtype=jnp.float64),
        reward_components=jnp.zeros((4, MAX_PLAYERS), dtype=jnp.float64),
        seed=jnp.asarray(int(seed), dtype=jnp.int64),
        num_players=jnp.asarray(num_players, dtype=jnp.int32),
        episode_steps=jnp.asarray(configuration.episode_steps, dtype=jnp.int32),
        ship_speed=jnp.asarray(configuration.ship_speed, dtype=jnp.float64),
        reward_weights=jnp.asarray(rw),
    )


def batch_reset(
    seeds: list[int] | np.ndarray,
    num_players: int = 2,
    configuration: Configuration | None = None,
    reward_weights: RewardWeights | None = None,
    limits: Limits | None = None,
) -> State:
    states = [reset(int(seed), num_players, configuration, reward_weights, limits) for seed in seeds]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def actions_to_jax(
    actions: list[list[list[Any]]],
    limits: Limits | None = None,
) -> tuple[jax.Array, jax.Array]:
    limits = limits or Limits()
    arr = np.zeros((MAX_PLAYERS, limits.max_actions, 3), dtype=np.float64)
    mask = np.zeros((MAX_PLAYERS, limits.max_actions), dtype=np.bool_)
    for player, moves in enumerate(actions[:MAX_PLAYERS]):
        if not isinstance(moves, list):
            continue
        for i, move in enumerate(moves[: limits.max_actions]):
            if not isinstance(move, (list, tuple)) or len(move) != 3:
                continue
            arr[player, i, :] = [float(move[0]), float(move[1]), float(int(move[2]))]
            mask[player, i] = True
    return jnp.asarray(arr), jnp.asarray(mask)


def _fleet_speed(ships: jax.Array, max_speed: jax.Array) -> jax.Array:
    safe_ships = jnp.maximum(ships, 1.0)
    speed = 1.0 + (max_speed - 1.0) * (jnp.log(safe_ships) / jnp.log(1000.0)) ** 1.5
    return jnp.minimum(speed, max_speed)


def _point_to_segment_distance(px, py, vx, vy, wx, wy):
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    raw_t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / jnp.where(l2 == 0.0, 1.0, l2)
    t = jnp.clip(raw_t, 0.0, 1.0)
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    dist = jnp.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    point_dist = jnp.sqrt((px - vx) ** 2 + (py - vy) ** 2)
    return jnp.where(l2 == 0.0, point_dist, dist)


def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, radius):
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    disc = b * b - 4.0 * a * c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    denom = 2.0 * jnp.where(a < 1e-12, 1.0, a)
    t1 = (-b - sq) / denom
    t2 = (-b + sq) / denom
    quadratic_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(a < 1e-12, c <= 0.0, quadratic_hit)


def _compact_prefix(rows: jax.Array, keep: jax.Array) -> tuple[jax.Array, jax.Array]:
    idx = jnp.arange(rows.shape[0])
    order = jnp.argsort(jnp.where(keep, idx, idx + rows.shape[0]))
    compact = rows[order]
    count = jnp.sum(keep).astype(jnp.int32)
    active = idx < count
    compact = jnp.where(active[:, None], compact, jnp.zeros_like(compact))
    return compact, count


def _spawn_comets(state: State) -> State:
    spawn_steps = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.int32)
    spawn_match = spawn_steps == (state.step + 1)
    spawn_slot = jnp.argmax(spawn_match).astype(jnp.int32)
    should_spawn = jnp.any(spawn_match) & (state.comet_path_lengths[spawn_slot] > 0)

    def do_spawn(s: State) -> State:
        pids = s.comet_planet_ids[spawn_slot]
        base_idx = s.planet_count
        offsets = jnp.arange(4, dtype=jnp.int32)
        slots = base_idx + offsets
        comet_rows = jnp.stack(
            [
                pids,
                jnp.full((4,), -1.0),
                jnp.full((4,), -99.0),
                jnp.full((4,), -99.0),
                jnp.full((4,), COMET_RADIUS),
                jnp.full((4,), s.comet_ships[spawn_slot]),
                jnp.full((4,), COMET_PRODUCTION),
            ],
            axis=1,
        )
        planets = s.planets.at[slots].set(comet_rows)
        initial = s.initial_planets.at[slots].set(comet_rows)
        return s._replace(
            planets=planets,
            initial_planets=initial,
            planet_count=s.planet_count + 4,
            comet_group_active=s.comet_group_active.at[spawn_slot].set(True),
            comet_path_index=s.comet_path_index.at[spawn_slot].set(-1),
        )

    return jax.lax.cond(should_spawn, do_spawn, lambda s: s, state)


def _process_launches(state: State, actions: jax.Array, action_mask: jax.Array) -> tuple[State, jax.Array]:
    flat_actions = actions.reshape((-1, 3))
    flat_mask = action_mask.reshape((-1,))
    action_players = jnp.repeat(jnp.arange(MAX_PLAYERS, dtype=jnp.int32), actions.shape[1])
    launches_by_player = jnp.zeros((MAX_PLAYERS,), dtype=jnp.float64)

    def body(carry, x):
        s, launches = carry
        mv, valid, player = x
        from_id = mv[0]
        angle = mv[1]
        ships = jnp.floor(mv[2])
        planet_idxes = jnp.arange(s.planets.shape[0])
        planet_active = planet_idxes < s.planet_count
        id_match = planet_active & (s.planets[:, 0] == from_id)
        pidx = jnp.argmax(id_match).astype(jnp.int32)
        found = jnp.any(id_match)
        p = s.planets[pidx]
        can_launch = (
            valid
            & (player < s.num_players)
            & found
            & (p[1] == player.astype(jnp.float64))
            & (ships > 0.0)
            & (p[5] >= ships)
            & (s.fleet_count < s.fleets.shape[0])
        )
        start_x = p[2] + jnp.cos(angle) * (p[4] + 0.1)
        start_y = p[3] + jnp.sin(angle) * (p[4] + 0.1)
        new_fleet = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        new_fleet = new_fleet.at[:].set(
            jnp.stack(
                [
                    s.next_fleet_id.astype(jnp.float64),
                    player.astype(jnp.float64),
                    start_x,
                    start_y,
                    angle,
                    from_id,
                    ships,
                ]
            )
        )
        planets = s.planets.at[pidx, 5].add(jnp.where(can_launch, -ships, 0.0))
        fleets = s.fleets.at[s.fleet_count].set(jnp.where(can_launch, new_fleet, s.fleets[s.fleet_count]))
        s = s._replace(
            planets=planets,
            fleets=fleets,
            fleet_count=s.fleet_count + can_launch.astype(jnp.int32),
            next_fleet_id=s.next_fleet_id + can_launch.astype(jnp.int32),
        )
        launches = launches.at[player].add(can_launch.astype(jnp.float64))
        return (s, launches), None

    (state, launches_by_player), _ = jax.lax.scan(
        body,
        (state, launches_by_player),
        (flat_actions, flat_mask, action_players),
    )
    return state, launches_by_player


def step(state: State, actions: jax.Array, action_mask: jax.Array | None = None) -> State:
    if action_mask is None:
        action_mask = jnp.ones(actions.shape[:2], dtype=jnp.bool_)

    def done_noop(s: State) -> State:
        return s

    def active_step(s: State) -> State:
        s = _spawn_comets(s)
        s, launches_by_player = _process_launches(s, actions, action_mask)

        pidxes = jnp.arange(s.planets.shape[0])
        fidxes = jnp.arange(s.fleets.shape[0])
        planet_active = pidxes < s.planet_count
        fleet_active = fidxes < s.fleet_count

        owned = planet_active & (s.planets[:, 1] != -1.0)
        planets = s.planets.at[:, 5].add(jnp.where(owned, s.planets[:, 6], 0.0))
        s = s._replace(planets=planets)

        old_px = s.planets[:, 2]
        old_py = s.planets[:, 3]
        init_dx = s.initial_planets[:, 2] - CENTER
        init_dy = s.initial_planets[:, 3] - CENTER
        orbital_r = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
        init_angle = jnp.arctan2(init_dy, init_dx)
        orbiting = orbital_r + s.planets[:, 4] < ROTATION_RADIUS_LIMIT
        comet_flat_ids = s.comet_planet_ids.reshape((-1,))
        is_comet = jnp.any(s.planets[:, 0:1] == comet_flat_ids[None, :], axis=1)
        current_angle = init_angle + s.angular_velocity * s.step.astype(jnp.float64)
        path_new_x = jnp.where(
            planet_active & orbiting & (~is_comet),
            CENTER + orbital_r * jnp.cos(current_angle),
            old_px,
        )
        path_new_y = jnp.where(
            planet_active & orbiting & (~is_comet),
            CENTER + orbital_r * jnp.sin(current_angle),
            old_py,
        )
        path_check = planet_active & (~is_comet)

        new_path_index = jnp.where(s.comet_group_active, s.comet_path_index + 1, s.comet_path_index)

        def comet_body(carry, gi):
            px_new, py_new, check = carry
            active = s.comet_group_active[gi]
            path_i = new_path_index[gi]
            valid_i = jnp.clip(path_i, 0, s.comet_paths.shape[2] - 1)
            pids = s.comet_planet_ids[gi]

            def one_comet(inner, ci):
                xnew, ynew, chk = inner
                pid = pids[ci]
                match = planet_active & (s.planets[:, 0] == pid)
                pi = jnp.argmax(match).astype(jnp.int32)
                exists = active & jnp.any(match)
                in_path = path_i < s.comet_path_lengths[gi]
                pos = s.comet_paths[gi, ci, valid_i]
                xnew = xnew.at[pi].set(jnp.where(exists & in_path, pos[0], xnew[pi]))
                ynew = ynew.at[pi].set(jnp.where(exists & in_path, pos[1], ynew[pi]))
                chk = chk.at[pi].set(jnp.where(exists, old_px[pi] >= 0.0, chk[pi]))
                return (xnew, ynew, chk), None

            (px_new, py_new, check), _ = jax.lax.scan(one_comet, (px_new, py_new, check), jnp.arange(4))
            return (px_new, py_new, check), None

        (path_new_x, path_new_y, path_check), _ = jax.lax.scan(
            comet_body, (path_new_x, path_new_y, path_check), jnp.arange(5)
        )
        s = s._replace(comet_path_index=new_path_index)

        old_fx = s.fleets[:, 2]
        old_fy = s.fleets[:, 3]
        speed = _fleet_speed(s.fleets[:, 6], s.ship_speed)
        new_fx = old_fx + jnp.cos(s.fleets[:, 4]) * speed
        new_fy = old_fy + jnp.sin(s.fleets[:, 4]) * speed

        hit = _swept_pair_hit(
            old_fx[:, None],
            old_fy[:, None],
            new_fx[:, None],
            new_fy[:, None],
            old_px[None, :],
            old_py[None, :],
            path_new_x[None, :],
            path_new_y[None, :],
            s.planets[None, :, 4],
        )
        hit = hit & fleet_active[:, None] & path_check[None, :]
        hit_any = jnp.any(hit, axis=1)
        hit_idx = jnp.argmax(hit, axis=1).astype(jnp.int32)

        fleets = s.fleets.at[:, 2].set(jnp.where(fleet_active, new_fx, s.fleets[:, 2]))
        fleets = fleets.at[:, 3].set(jnp.where(fleet_active, new_fy, s.fleets[:, 3]))
        s = s._replace(fleets=fleets)

        owners = jnp.clip(s.fleets[:, 1].astype(jnp.int32), 0, MAX_PLAYERS - 1)
        combat = jnp.zeros((s.planets.shape[0], MAX_PLAYERS), dtype=jnp.float64)
        combat = combat.at[hit_idx, owners].add(jnp.where(hit_any & fleet_active, s.fleets[:, 6], 0.0))

        sun_dist = _point_to_segment_distance(CENTER, CENTER, old_fx, old_fy, new_fx, new_fy)
        out_of_bounds = (new_fx < 0.0) | (new_fx > BOARD_SIZE) | (new_fy < 0.0) | (new_fy > BOARD_SIZE)
        remove_fleet = fleet_active & (hit_any | out_of_bounds | (sun_dist < SUN_RADIUS))

        planets = s.planets.at[:, 2].set(jnp.where(planet_active, path_new_x, s.planets[:, 2]))
        planets = planets.at[:, 3].set(jnp.where(planet_active, path_new_y, s.planets[:, 3]))

        top_player = jnp.argmax(combat, axis=1).astype(jnp.int32)
        top_ships = jnp.max(combat, axis=1)
        second_ships = jnp.max(jnp.where(jnp.arange(MAX_PLAYERS)[None, :] == top_player[:, None], -1.0, combat), axis=1)
        entry_count = jnp.sum(combat > 0.0, axis=1)
        survivor_ships = jnp.where(entry_count > 1, jnp.where(top_ships == second_ships, 0.0, top_ships - second_ships), top_ships)
        survivor_owner = jnp.where(survivor_ships > 0.0, top_player.astype(jnp.float64), -1.0)
        has_combat = planet_active & (entry_count > 0) & (survivor_ships > 0.0)
        same_owner = planets[:, 1] == survivor_owner
        new_ships = jnp.where(same_owner, planets[:, 5] + survivor_ships, planets[:, 5] - survivor_ships)
        captured = has_combat & (~same_owner) & (new_ships < 0.0)
        planets = planets.at[:, 5].set(jnp.where(has_combat, jnp.where(captured, -new_ships, new_ships), planets[:, 5]))
        planets = planets.at[:, 1].set(jnp.where(captured, survivor_owner, planets[:, 1]))
        s = s._replace(planets=planets)

        expired_groups = s.comet_group_active & (s.comet_path_index >= s.comet_path_lengths)
        expired_ids = jnp.where(expired_groups[:, None], s.comet_planet_ids, -999999.0).reshape((-1,))
        keep_planets = (jnp.arange(s.planets.shape[0]) < s.planet_count) & (~jnp.any(s.planets[:, 0:1] == expired_ids[None, :], axis=1))
        planets, planet_count = _compact_prefix(s.planets, keep_planets)
        initial_planets, _ = _compact_prefix(s.initial_planets, keep_planets)
        s = s._replace(
            planets=planets,
            initial_planets=initial_planets,
            planet_count=planet_count,
            comet_group_active=s.comet_group_active & (~expired_groups),
        )

        keep_fleets = fleet_active & (~remove_fleet)
        fleets, fleet_count = _compact_prefix(s.fleets, keep_fleets)
        s = s._replace(fleets=fleets, fleet_count=fleet_count)

        pidxes = jnp.arange(s.planets.shape[0])
        fidxes = jnp.arange(s.fleets.shape[0])
        planet_active = pidxes < s.planet_count
        fleet_active = fidxes < s.fleet_count
        planet_owner = s.planets[:, 1].astype(jnp.int32)
        fleet_owner = s.fleets[:, 1].astype(jnp.int32)
        alive_from_planets = jnp.any(
            planet_active[:, None] & (s.planets[:, 1:2] == jnp.arange(MAX_PLAYERS, dtype=jnp.float64)[None, :]),
            axis=0,
        )
        alive_from_fleets = jnp.any(
            fleet_active[:, None] & (s.fleets[:, 1:2] == jnp.arange(MAX_PLAYERS, dtype=jnp.float64)[None, :]),
            axis=0,
        )
        alive = (alive_from_planets | alive_from_fleets) & (jnp.arange(MAX_PLAYERS) < s.num_players)
        terminated = (s.step >= s.episode_steps - 2) | (jnp.sum(alive) <= 1)

        planet_scores = jnp.zeros((MAX_PLAYERS,), dtype=jnp.float64).at[jnp.clip(planet_owner, 0, MAX_PLAYERS - 1)].add(
            jnp.where(planet_active & (s.planets[:, 1] >= 0.0), s.planets[:, 5], 0.0)
        )
        fleet_scores = jnp.zeros((MAX_PLAYERS,), dtype=jnp.float64).at[jnp.clip(fleet_owner, 0, MAX_PLAYERS - 1)].add(
            jnp.where(fleet_active & (s.fleets[:, 1] >= 0.0), s.fleets[:, 6], 0.0)
        )
        scores = (planet_scores + fleet_scores) * (jnp.arange(MAX_PLAYERS) < s.num_players)
        max_score = jnp.max(scores)
        official_rewards = jnp.where((scores == max_score) & (max_score > 0.0) & (jnp.arange(MAX_PLAYERS) < s.num_players), 1.0, -1.0)

        production = jnp.zeros((MAX_PLAYERS,), dtype=jnp.float64).at[jnp.clip(planet_owner, 0, MAX_PLAYERS - 1)].add(
            jnp.where(planet_active & (s.planets[:, 1] >= 0.0), s.planets[:, 6], 0.0)
        )
        n = s.num_players.astype(jnp.float64)
        mean_prod = jnp.sum(production) / n
        terminal_component = s.reward_weights[0] * official_rewards
        remaining_frac = jnp.clip((s.episode_steps - s.step).astype(jnp.float64) / s.episode_steps.astype(jnp.float64), 0.0, 1.0)
        centered_outcome = jnp.where(official_rewards > 0.0, 1.0, -1.0)
        terminal_time_component = s.reward_weights[1] * centered_outcome * remaining_frac
        production_component = s.reward_weights[2] * (production - mean_prod)
        launch_component = s.reward_weights[3] * launches_by_player
        components = jnp.stack(
            [
                jnp.where(terminated, terminal_component, 0.0),
                jnp.where(terminated, terminal_time_component, 0.0),
                production_component,
                launch_component,
            ],
            axis=0,
        )
        shaped_rewards = jnp.sum(components, axis=0)

        return s._replace(
            step=jnp.where(terminated, 0, s.step + 1),
            done=terminated,
            rewards=jnp.where(terminated, official_rewards, shaped_rewards),
            reward_components=components,
        )

    return jax.lax.cond(state.done, done_noop, active_step, state)


jit_step = jax.jit(step)


def _rows_from_prefix(rows: np.ndarray, count: int) -> list[list[Any]]:
    out: list[list[Any]] = []
    for row in rows[:count]:
        vals = row.tolist()
        out.append([int(vals[0]), int(vals[1]), vals[2], vals[3], vals[4], int(round(vals[5])), int(round(vals[6]))])
    return out


def snapshot_from_state(state: State) -> dict[str, Any]:
    host = jax.tree.map(lambda x: np.asarray(x), state)
    planet_count = int(host.planet_count)
    fleet_count = int(host.fleet_count)
    planets = _rows_from_prefix(host.planets, planet_count)
    initial_planets = _rows_from_prefix(host.initial_planets, planet_count)
    fleets = _rows_from_prefix(host.fleets, fleet_count)

    comets = []
    comet_ids = []
    for gi, active in enumerate(host.comet_group_active.tolist()):
        if not active:
            continue
        pids = [int(x) for x in host.comet_planet_ids[gi].tolist()]
        path_len = int(host.comet_path_lengths[gi])
        comet_ids.extend(pids)
        comets.append(
            {
                "planet_ids": pids,
                "path_index": int(host.comet_path_index[gi]),
                "paths": [
                    host.comet_paths[gi, ci, :path_len, :].tolist()
                    for ci in range(4)
                ],
            }
        )
    return {
        "step": int(host.step),
        "angular_velocity": float(host.angular_velocity),
        "planets": planets,
        "initial_planets": initial_planets,
        "fleets": fleets,
        "next_fleet_id": int(host.next_fleet_id),
        "comet_planet_ids": comet_ids,
        "comets": comets,
        "done": bool(host.done),
        "rewards": host.rewards[: int(host.num_players)].tolist() if bool(host.done) else None,
        "seed": int(host.seed),
    }


def observations_from_state(state: State) -> list[dict[str, Any]]:
    snap = snapshot_from_state(state)
    return [
        {
            "player": player,
            "step": snap["step"],
            "angular_velocity": snap["angular_velocity"],
            "planets": snap["planets"],
            "initial_planets": snap["initial_planets"],
            "fleets": snap["fleets"],
            "comets": snap["comets"],
            "comet_planet_ids": snap["comet_planet_ids"],
        }
        for player in range(int(np.asarray(state.num_players)))
    ]


class JaxOrbitWarsEngine:
    """Parity-harness compatible wrapper around the JAX engine."""

    def __init__(
        self,
        configuration: Configuration | None = None,
        reward_weights: RewardWeights | None = None,
        limits: Limits | None = None,
        use_jit: bool = True,
    ) -> None:
        self.configuration = configuration or Configuration()
        self.reward_weights = reward_weights or RewardWeights()
        self.limits = limits or Limits()
        self.use_jit = use_jit
        self.state: State | None = None

    def reset(self, seed: int, num_players: int, configuration: dict | None = None):
        cfg = self.configuration
        if configuration:
            cfg = Configuration(
                episode_steps=int(configuration.get("episodeSteps", cfg.episode_steps)),
                ship_speed=float(configuration.get("shipSpeed", cfg.ship_speed)),
                comet_speed=float(configuration.get("cometSpeed", cfg.comet_speed)),
            )
        self.state = reset(seed, num_players, cfg, self.reward_weights, self.limits)
        from engine_parity_checker.engine import PlayerObs

        return [PlayerObs(**obs) for obs in observations_from_state(self.state)]

    def step(self, actions):
        assert self.state is not None, "call reset() first"
        arr, mask = actions_to_jax(actions, self.limits)
        step_fn = jit_step if self.use_jit else step
        self.state = step_fn(self.state, arr, mask)
        self.state.done.block_until_ready()
        from engine_parity_checker.engine import PlayerObs

        return [PlayerObs(**obs) for obs in observations_from_state(self.state)], bool(np.asarray(self.state.done))

    def snapshot(self):
        assert self.state is not None, "call reset() first"
        from engine_parity_checker.engine import Snapshot

        raw = snapshot_from_state(self.state)
        return Snapshot(
            step=raw["step"],
            angular_velocity=raw["angular_velocity"],
            planets=raw["planets"],
            initial_planets=raw["initial_planets"],
            fleets=raw["fleets"],
            next_fleet_id=raw["next_fleet_id"],
            comet_planet_ids=raw["comet_planet_ids"],
            comets=raw["comets"],
            done=raw["done"],
            rewards=raw["rewards"],
            seed=raw["seed"],
            info={"engine": "jax"},
        )
