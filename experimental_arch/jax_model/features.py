from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .engine import (
    BOARD_SIZE,
    CENTER,
    MAX_PLAYERS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    Limits,
    State,
    _fleet_speed,
    _point_to_segment_distance,
    _swept_pair_hit,
    step,
)


PLANET_SLOTS = 44
ACTIONS_DIM = 2
NUM_FRAMES = 4
TOKEN_DIM = 11
GLOBAL_DIM = 16

FRAME_T1 = 1
FRAME_T10 = 10
RESOLVE_CAP = 96

NOOP_ACTION = 0
SEND_ALL_ACTION = 1


@dataclass(frozen=True)
class FeatureConfig:
    """Static feature-shape settings."""

    resolve_cap: int = RESOLVE_CAP
    limits: Limits = Limits()


class Features(NamedTuple):
    planet_ids: jax.Array
    num_planets: jax.Array
    frame_offsets: jax.Array
    tokens: jax.Array
    globals: jax.Array
    presence: jax.Array
    turns: jax.Array
    angles: jax.Array
    mask: jax.Array
    ship_counts: jax.Array
    reachable_mask: jax.Array
    frame_planets: jax.Array


def _log_norm(x: jax.Array, full: float) -> jax.Array:
    return jnp.log1p(jnp.maximum(x, 0.0)) / jnp.log1p(full)


def _signed_log_norm(x: jax.Array, full: float) -> jax.Array:
    return jnp.sign(x) * (jnp.log1p(jnp.abs(x)) / jnp.log1p(full))


def _norm_dist(x: jax.Array) -> jax.Array:
    return x / BOARD_SIZE


def _norm_turns(x: jax.Array) -> jax.Array:
    return x / 20.0


def _norm_ships(x: jax.Array) -> jax.Array:
    return _log_norm(x, 1000.0)


def _norm_prod(x: jax.Array) -> jax.Array:
    return _log_norm(x, 10.0)


def _norm_count(x: jax.Array) -> jax.Array:
    return x / PLANET_SLOTS


def _empty_actions(limits: Limits) -> tuple[jax.Array, jax.Array]:
    actions = jnp.zeros((MAX_PLAYERS, limits.max_actions, 3), dtype=jnp.float64)
    mask = jnp.zeros((MAX_PLAYERS, limits.max_actions), dtype=jnp.bool_)
    return actions, mask


def _slow_trajectory(state: State, config: FeatureConfig) -> tuple[State, jax.Array]:
    actions, action_mask = _empty_actions(config.limits)

    def body(s: State, _):
        ns = step(s, actions, action_mask)
        return ns, ns

    states = jax.lax.scan(body, state, jnp.arange(config.resolve_cap))[1]
    stacked = jax.tree.map(lambda a, b: jnp.concatenate([a[None, ...], b], axis=0), state, states)
    fleet_counts = stacked.fleet_count
    resolved_hits = fleet_counts == 0
    hit_any = jnp.any(resolved_hits)
    resolved = jnp.where(hit_any, jnp.argmax(resolved_hits), config.resolve_cap).astype(jnp.int32)
    offsets = jnp.asarray([0, FRAME_T1, FRAME_T10, resolved], dtype=jnp.int32)
    offsets = jnp.minimum(offsets, config.resolve_cap)
    frames = jax.tree.map(lambda x: x[offsets], stacked)
    return frames, offsets


def _is_comet_id(state: State, ids: jax.Array) -> jax.Array:
    comet_ids = state.comet_planet_ids.reshape((-1,))
    return jnp.any(ids[..., None] == comet_ids[None, :], axis=-1)


def _base_planet_positions(state: State, offset: jax.Array) -> tuple[jax.Array, jax.Array]:
    rows = state.planets
    active = jnp.arange(rows.shape[0]) < state.planet_count
    init_dx = state.initial_planets[:, 2] - CENTER
    init_dy = state.initial_planets[:, 3] - CENTER
    orbital_r = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    init_angle = jnp.arctan2(init_dy, init_dx)
    orbiting = orbital_r + rows[:, 4] < ROTATION_RADIUS_LIMIT
    is_comet = _is_comet_id(state, rows[:, 0])
    angle_step = state.step.astype(jnp.float64) + offset.astype(jnp.float64) - 1.0
    orbit_x = CENTER + orbital_r * jnp.cos(init_angle + state.angular_velocity * angle_step)
    orbit_y = CENTER + orbital_r * jnp.sin(init_angle + state.angular_velocity * angle_step)
    moved = (offset > 0) & active & orbiting & (~is_comet)
    rows = rows.at[:, 2].set(jnp.where(moved, orbit_x, rows[:, 2]))
    rows = rows.at[:, 3].set(jnp.where(moved, orbit_y, rows[:, 3]))

    def comet_group(carry, gi):
        out, act = carry
        group_active = state.comet_group_active[gi]
        path_i = state.comet_path_index[gi] + offset
        valid = group_active & (path_i >= 0) & (path_i < state.comet_path_lengths[gi])
        clipped = jnp.clip(path_i, 0, state.comet_paths.shape[2] - 1)
        pids = state.comet_planet_ids[gi]

        def one_comet(inner, ci):
            cur, cur_active = inner
            pid = pids[ci]
            match = active & (cur[:, 0] == pid)
            idx = jnp.argmax(match).astype(jnp.int32)
            exists = jnp.any(match) & valid
            pos = state.comet_paths[gi, ci, clipped]
            cur = cur.at[idx, 2].set(jnp.where(exists, pos[0], cur[idx, 2]))
            cur = cur.at[idx, 3].set(jnp.where(exists, pos[1], cur[idx, 3]))
            cur_active = cur_active.at[idx].set(jnp.where(group_active & jnp.any(match), exists, cur_active[idx]))
            return (cur, cur_active), None

        (out, act), _ = jax.lax.scan(one_comet, (out, act), jnp.arange(4))
        return (out, act), None

    rows, active = jax.lax.scan(comet_group, (rows, active), jnp.arange(5))[0]
    return rows, active


def _future_comet_rows(state: State, offset: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    spawn_steps = jnp.asarray([50, 150, 250, 350, 450], dtype=jnp.int32)
    deltas = spawn_steps - state.step
    valid_spawn = (deltas >= 1) & (deltas <= offset)
    spawn_slot = jnp.argmax(valid_spawn).astype(jnp.int32)
    has_spawn = jnp.any(valid_spawn)
    path_i = offset - deltas[spawn_slot]
    active = has_spawn & (path_i >= 0) & (path_i < state.comet_path_lengths[spawn_slot])
    clipped = jnp.clip(path_i, 0, state.comet_paths.shape[2] - 1)
    pids = state.comet_planet_ids[spawn_slot]
    pos = state.comet_paths[spawn_slot, :, clipped]
    rows = jnp.stack(
        [
            pids,
            jnp.full((4,), -1.0),
            pos[:, 0],
            pos[:, 1],
            jnp.full((4,), 1.0),
            jnp.full((4,), state.comet_ships[spawn_slot]),
            jnp.full((4,), 1.0),
        ],
        axis=1,
    )
    initial = rows.at[:, 2].set(-99.0).at[:, 3].set(-99.0)
    active_mask = jnp.full((4,), active)
    return rows, initial, active_mask


def _position_rows_at_offset(state: State, offset: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    rows, active = _base_planet_positions(state, offset)
    initial = state.initial_planets
    future_rows, future_initial, future_active = _future_comet_rows(state, offset)
    slots = state.planet_count + jnp.arange(4, dtype=jnp.int32)
    in_bounds = slots < rows.shape[0]
    rows = rows.at[slots].set(jnp.where(in_bounds[:, None], future_rows, rows[slots]))
    initial = initial.at[slots].set(jnp.where(in_bounds[:, None], future_initial, initial[slots]))
    active = active.at[slots].set(jnp.where(in_bounds, future_active, active[slots]))
    rows = jnp.where(active[:, None], rows, jnp.zeros_like(rows))
    initial = jnp.where(active[:, None], initial, jnp.zeros_like(initial))
    return rows, initial, active


def _step_end_rows_for_collision(state: State, step_start_offset: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    start, _, active = _position_rows_at_offset(state, step_start_offset)
    end, _, _ = _position_rows_at_offset(state, step_start_offset + 1)
    is_comet = _is_comet_id(state, start[:, 0])

    def comet_group(carry, gi):
        out = carry
        group_active = state.comet_group_active[gi]
        start_i = state.comet_path_index[gi] + step_start_offset
        end_i = start_i + 1
        expires = group_active & (start_i >= 0) & (start_i < state.comet_path_lengths[gi]) & (end_i >= state.comet_path_lengths[gi])
        pids = state.comet_planet_ids[gi]

        def one_comet(cur, ci):
            match = active & (cur[:, 0] == pids[ci])
            idx = jnp.argmax(match).astype(jnp.int32)
            return cur.at[idx, 2:4].set(jnp.where(expires & jnp.any(match), start[idx, 2:4], cur[idx, 2:4])), None

        out, _ = jax.lax.scan(one_comet, out, jnp.arange(4))
        return out, None

    end, _ = jax.lax.scan(comet_group, end, jnp.arange(5))
    check = active & ((~is_comet) | (start[:, 2] >= 0.0))
    return start, end, check


def _fleet_events(state: State, config: FeatureConfig) -> tuple[jax.Array, jax.Array, jax.Array]:
    cap = config.resolve_cap
    fleet_active = jnp.arange(state.fleets.shape[0]) < state.fleet_count

    def compute_events(_: None):
        ks = jnp.arange(cap, dtype=jnp.int32)
        starts, ends, checks = jax.vmap(lambda k: _step_end_rows_for_collision(state, k))(ks)
        turns = ks + 1
        f = state.fleets
        speed = _fleet_speed(f[:, 6], state.ship_speed)
        ux = jnp.cos(f[:, 4])
        uy = jnp.sin(f[:, 4])
        dist0 = ks.astype(jnp.float64)[None, :] * speed[:, None]
        old_fx = f[:, 2:3] + ux[:, None] * dist0
        old_fy = f[:, 3:4] + uy[:, None] * dist0
        new_fx = old_fx + ux[:, None] * speed[:, None]
        new_fy = old_fy + uy[:, None] * speed[:, None]
        hit = _swept_pair_hit(
            old_fx[:, :, None],
            old_fy[:, :, None],
            new_fx[:, :, None],
            new_fy[:, :, None],
            starts[None, :, :, 2],
            starts[None, :, :, 3],
            ends[None, :, :, 2],
            ends[None, :, :, 3],
            starts[None, :, :, 4],
        )
        hit = hit & checks[None, :, :] & fleet_active[:, None, None]
        hit_any = jnp.any(hit, axis=2)
        hit_idx_by_turn = jnp.argmax(hit, axis=2).astype(jnp.int32)
        sun_dist = _point_to_segment_distance(CENTER, CENTER, old_fx, old_fy, new_fx, new_fy)
        out = (new_fx < 0.0) | (new_fx > BOARD_SIZE) | (new_fy < 0.0) | (new_fy > BOARD_SIZE)
        remove = fleet_active[:, None] & (hit_any | out | (sun_dist < SUN_RADIUS))
        any_remove = jnp.any(remove, axis=1)
        first_turn_idx = jnp.argmax(remove, axis=1).astype(jnp.int32)
        remove_turn = jnp.where(any_remove, first_turn_idx + 1, cap + 1).astype(jnp.int32)
        hit_idx = jnp.take_along_axis(hit_idx_by_turn, first_turn_idx[:, None], axis=1)[:, 0]
        hit_at_remove = jnp.take_along_axis(hit_any, first_turn_idx[:, None], axis=1)[:, 0]
        hit_idx = jnp.where(any_remove & hit_at_remove, hit_idx, -1).astype(jnp.int32)
        resolved = jnp.where(jnp.any(fleet_active), jnp.max(jnp.where(fleet_active, remove_turn, 0)), 0)
        resolved = jnp.minimum(resolved, cap).astype(jnp.int32)
        return remove_turn, hit_idx, resolved

    empty_turns = jnp.full((state.fleets.shape[0],), cap + 1, dtype=jnp.int32)
    empty_hits = jnp.full((state.fleets.shape[0],), -1, dtype=jnp.int32)
    return jax.lax.cond(
        state.fleet_count > 0,
        compute_events,
        lambda _: (empty_turns, empty_hits, jnp.asarray(0, dtype=jnp.int32)),
        operand=None,
    )


def _owner_ship_timeline(
    state: State,
    config: FeatureConfig,
    remove_turn: jax.Array,
    hit_idx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    cap = config.resolve_cap
    active0 = jnp.arange(state.planets.shape[0]) < state.planet_count
    owner0 = jnp.where(active0, state.planets[:, 1], -1.0)
    ships0 = jnp.where(active0, state.planets[:, 5], 0.0)
    prod0 = jnp.where(active0, state.planets[:, 6], 0.0)
    fleet_owner = jnp.clip(state.fleets[:, 1].astype(jnp.int32), 0, MAX_PLAYERS - 1)
    fleet_ships = state.fleets[:, 6]
    fleet_active = jnp.arange(state.fleets.shape[0]) < state.fleet_count

    def body(carry, turn):
        owner, ships = carry
        rows, _, active = _position_rows_at_offset(state, turn + 1)
        spawned = active & (prod0 == 0.0) & (rows[:, 6] > 0.0) & (ships == 0.0) & _is_comet_id(state, rows[:, 0])
        owner = jnp.where(spawned, rows[:, 1], owner)
        ships = jnp.where(spawned, rows[:, 5], ships)
        prod = jnp.where(active, rows[:, 6], 0.0)
        ships = ships + jnp.where(active & (owner != -1.0), prod, 0.0)

        arrives = fleet_active & (remove_turn == (turn + 1)) & (hit_idx >= 0)
        combat = jnp.zeros((state.planets.shape[0], MAX_PLAYERS), dtype=jnp.float64)
        combat = combat.at[hit_idx, fleet_owner].add(jnp.where(arrives, fleet_ships, 0.0))
        top_player = jnp.argmax(combat, axis=1).astype(jnp.int32)
        top_ships = jnp.max(combat, axis=1)
        second_ships = jnp.max(
            jnp.where(jnp.arange(MAX_PLAYERS)[None, :] == top_player[:, None], -1.0, combat),
            axis=1,
        )
        entry_count = jnp.sum(combat > 0.0, axis=1)
        survivor_ships = jnp.where(
            entry_count > 1,
            jnp.where(top_ships == second_ships, 0.0, top_ships - second_ships),
            top_ships,
        )
        survivor_owner = jnp.where(survivor_ships > 0.0, top_player.astype(jnp.float64), -1.0)
        has_combat = active & (entry_count > 0) & (survivor_ships > 0.0)
        same_owner = owner == survivor_owner
        new_ships = jnp.where(same_owner, ships + survivor_ships, ships - survivor_ships)
        captured = has_combat & (~same_owner) & (new_ships < 0.0)
        ships = jnp.where(has_combat, jnp.where(captured, -new_ships, new_ships), ships)
        owner = jnp.where(captured, survivor_owner, owner)
        owner = jnp.where(active, owner, -1.0)
        ships = jnp.where(active, ships, 0.0)
        return (owner, ships), (owner, ships)

    (owner_last, ships_last), hist = jax.lax.scan(body, (owner0, ships0), jnp.arange(cap, dtype=jnp.int32))
    owners = jnp.concatenate([owner0[None, :], hist[0]], axis=0)
    ships = jnp.concatenate([ships0[None, :], hist[1]], axis=0)
    return owners, ships


def _fast_trajectory(state: State, config: FeatureConfig) -> tuple[State, jax.Array]:
    remove_turn, hit_idx, resolved = _fleet_events(state, config)
    offsets = jnp.asarray([0, FRAME_T1, FRAME_T10, resolved], dtype=jnp.int32)
    offsets = jnp.minimum(offsets, config.resolve_cap)
    owners, ships = _owner_ship_timeline(state, config, remove_turn, hit_idx)

    rows, initials, actives = jax.vmap(lambda off: _position_rows_at_offset(state, off))(offsets)
    frame_owners = owners[offsets]
    frame_ships = ships[offsets]
    rows = rows.at[:, :, 1].set(frame_owners)
    rows = rows.at[:, :, 5].set(frame_ships)
    rows = jnp.where(actives[:, :, None], rows, jnp.zeros_like(rows))
    initials = jnp.where(actives[:, :, None], initials, jnp.zeros_like(initials))
    counts = jnp.sum(actives, axis=1).astype(jnp.int32)

    frames = jax.tree.map(lambda x: jnp.broadcast_to(x, (NUM_FRAMES,) + x.shape), state)
    frames = frames._replace(planets=rows, initial_planets=initials, planet_count=counts, fleet_count=jnp.zeros((NUM_FRAMES,), dtype=jnp.int32))
    return frames, offsets


def _slot_ids(state: State) -> jax.Array:
    idx = jnp.arange(PLANET_SLOTS)
    ids = jnp.where(idx < state.planet_count, state.planets[:PLANET_SLOTS, 0], -1.0)
    return ids.astype(jnp.int32)


def _match_planets(frame: State, slot_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    active = jnp.arange(frame.planets.shape[0]) < frame.planet_count
    match = (frame.planets[:, 0][None, :] == slot_ids[:, None].astype(jnp.float64)) & active[None, :]
    present = jnp.any(match, axis=1) & (slot_ids >= 0)
    idx = jnp.argmax(match, axis=1).astype(jnp.int32)
    rows = frame.planets[idx]
    rows = jnp.where(present[:, None], rows, jnp.zeros_like(rows))
    return rows, present


def _frame_tokens(frame: State, slot_ids: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    planets, present = _match_planets(frame, slot_ids)
    _, initial_present = _match_planets(
        frame._replace(planets=frame.initial_planets, planet_count=frame.planet_count), slot_ids
    )
    initial, _ = _match_planets(
        frame._replace(planets=frame.initial_planets, planet_count=frame.planet_count), slot_ids
    )

    comet_ids = frame.comet_planet_ids.reshape((-1,)).astype(jnp.int32)
    is_comet = jnp.any(slot_ids[:, None] == comet_ids[None, :], axis=1)
    dx = initial[:, 2] - CENTER
    dy = initial[:, 3] - CENTER
    orbital_r = jnp.sqrt(dx * dx + dy * dy)
    is_orbiting = (~is_comet) & initial_present & (orbital_r + planets[:, 4] < ROTATION_RADIUS_LIMIT)
    dist_sun = jnp.sqrt((planets[:, 2] - CENTER) ** 2 + (planets[:, 3] - CENTER) ** 2)

    player = jnp.asarray(0.0, dtype=jnp.float64)
    tokens = jnp.stack(
        [
            planets[:, 1] == player,
            (planets[:, 1] >= 0.0) & (planets[:, 1] != player),
            planets[:, 1] == -1.0,
            is_comet,
            is_orbiting,
            _norm_prod(planets[:, 6]),
            _norm_ships(planets[:, 5]),
            _norm_dist(planets[:, 2]),
            _norm_dist(planets[:, 3]),
            _norm_dist(dist_sun),
            jnp.where(is_orbiting, frame.angular_velocity / 0.05, 0.0),
        ],
        axis=1,
    ).astype(jnp.float32)
    tokens = jnp.where(present[:, None], tokens, jnp.zeros_like(tokens))
    frame_planets = jnp.stack([planets[:, 0], planets[:, 1], planets[:, 2], planets[:, 3], planets[:, 5]], axis=1)
    frame_planets = jnp.where(present[:, None], frame_planets, jnp.zeros_like(frame_planets))
    return tokens, present.astype(jnp.float32), frame_planets


def compute_globals(state: State, player: int = 0) -> jax.Array:
    pidx = jnp.arange(state.planets.shape[0])
    fidx = jnp.arange(state.fleets.shape[0])
    planet_active = pidx < state.planet_count
    fleet_active = fidx < state.fleet_count
    p_owner = state.planets[:, 1].astype(jnp.int32)
    f_owner = state.fleets[:, 1].astype(jnp.int32)
    pships = state.planets[:, 5]
    prod = state.planets[:, 6]
    fships = state.fleets[:, 6]
    player_i = jnp.asarray(player, dtype=jnp.int32)

    own_p = planet_active & (p_owner == player_i)
    enemy_p = planet_active & (p_owner >= 0) & (p_owner != player_i) & (p_owner < state.num_players)
    neutral_p = planet_active & (p_owner == -1)
    own_f = fleet_active & (f_owner == player_i)
    enemy_f = fleet_active & (f_owner >= 0) & (f_owner != player_i) & (f_owner < state.num_players)

    own_planet_ships = jnp.sum(jnp.where(own_p, pships, 0.0))
    enemy_planet_ships = jnp.sum(jnp.where(enemy_p, pships, 0.0))
    neutral_ships = jnp.sum(jnp.where(neutral_p, pships, 0.0))
    own_planets = jnp.sum(own_p)
    enemy_planets = jnp.sum(enemy_p)
    neutral_planets = jnp.sum(neutral_p)
    own_production = jnp.sum(jnp.where(own_p, prod, 0.0))
    enemy_production = jnp.sum(jnp.where(enemy_p, prod, 0.0))
    own_fleet_ships = jnp.sum(jnp.where(own_f, fships, 0.0))
    enemy_fleet_ships = jnp.sum(jnp.where(enemy_f, fships, 0.0))
    own_score = own_planet_ships + own_fleet_ships
    enemy_score = enemy_planet_ships + enemy_fleet_ships
    remaining = jnp.maximum(state.episode_steps.astype(jnp.float64) - state.step.astype(jnp.float64), 0.0)
    remaining = remaining / jnp.maximum(state.episode_steps.astype(jnp.float64), 1.0)

    def share(a, b):
        return jnp.where(a + b > 0.0, a / (a + b), 0.5)

    return jnp.asarray(
        [
            remaining,
            _norm_ships(own_score),
            _norm_ships(enemy_score),
            _norm_ships(neutral_ships),
            _norm_count(own_planets),
            _norm_count(enemy_planets),
            _norm_count(neutral_planets),
            _log_norm(own_production, 100.0),
            _log_norm(enemy_production, 100.0),
            _norm_ships(own_fleet_ships),
            _norm_ships(enemy_fleet_ships),
            share(own_score, enemy_score),
            share(own_production, enemy_production),
            share(own_planets.astype(jnp.float64), enemy_planets.astype(jnp.float64)),
            _signed_log_norm(own_score - enemy_score, 1000.0),
            _signed_log_norm(own_production - enemy_production, 100.0),
        ],
        dtype=jnp.float32,
    )


def _direct_action_tensors(frames: State, slot_ids: jax.Array, presence: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    planets0, _ = _match_planets(jax.tree.map(lambda x: x[0], frames), slot_ids)
    frame_planets, _ = jax.vmap(_match_planets, in_axes=(0, None))(frames, slot_ids)

    src = frame_planets[:, :, None, :]
    dst = frame_planets[:, None, :, :]
    dx = dst[..., 2] - src[..., 2]
    dy = dst[..., 3] - src[..., 3]
    distance = jnp.sqrt(dx * dx + dy * dy)
    src_ships = src[..., 5]
    speed = _fleet_speed(jnp.maximum(src_ships, 1.0), frames.ship_speed[:, None, None])
    direct_turns = _norm_turns(jnp.ceil(distance / jnp.maximum(speed, 1e-6)))
    direct_angles = jnp.arctan2(planets0[None, :, 3] - planets0[:, None, 3], planets0[None, :, 2] - planets0[:, None, 2])

    turns = jnp.zeros((NUM_FRAMES, PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM), dtype=jnp.float32)
    turns = turns.at[..., SEND_ALL_ACTION].set(direct_turns.astype(jnp.float32))
    angles = jnp.zeros((PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM), dtype=jnp.float32)
    angles = angles.at[..., SEND_ALL_ACTION].set(direct_angles.astype(jnp.float32))

    reachable_mask = jnp.zeros((NUM_FRAMES, PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM), dtype=jnp.uint8)

    policy_mask = jnp.zeros((PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM), dtype=jnp.uint8)
    policy_mask = policy_mask.at[:, 0, NOOP_ACTION].set(1)

    ship_counts = jnp.zeros((PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM), dtype=jnp.int32)
    ship_counts = ship_counts.at[..., SEND_ALL_ACTION].set(jnp.floor(planets0[:, None, 5]).astype(jnp.int32))
    ship_counts = jnp.where(policy_mask.astype(jnp.bool_), ship_counts, jnp.zeros_like(ship_counts))
    return turns, angles, policy_mask, ship_counts, reachable_mask


def encode(state: State, player: int = 0, config: FeatureConfig | None = None) -> Features:
    if player != 0:
        raise NotImplementedError("jax_model feature encoding currently supports player 0 only")
    config = config or FeatureConfig()
    frames, offsets = _fast_trajectory(state, config)
    slot_ids = _slot_ids(state)
    tokens, presence, frame_planets = jax.vmap(_frame_tokens, in_axes=(0, None))(frames, slot_ids)
    turns, angles, mask, ship_counts, reachable_mask = _direct_action_tensors(frames, slot_ids, presence)
    return Features(
        planet_ids=slot_ids,
        num_planets=jnp.minimum(state.planet_count, PLANET_SLOTS),
        frame_offsets=offsets,
        tokens=tokens,
        globals=compute_globals(state, player),
        presence=presence,
        turns=turns,
        angles=angles,
        mask=mask,
        ship_counts=ship_counts,
        reachable_mask=reachable_mask,
        frame_planets=frame_planets,
    )


def step_and_encode(
    state: State,
    actions: jax.Array,
    action_mask: jax.Array | None = None,
    player: int = 0,
    config: FeatureConfig | None = None,
) -> tuple[State, Features]:
    next_state = step(state, actions, action_mask)
    return next_state, encode(next_state, player=player, config=config)


jit_encode = jax.jit(encode, static_argnames=("player", "config"))
jit_step_and_encode = jax.jit(step_and_encode, static_argnames=("player", "config"))
