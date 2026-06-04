"""Pure-JAX, vmap-able, JIT-compilable step function.

Mirrors `step_numpy.py` operation-by-operation, but built out of `jnp` and
`lax` primitives so it can be traced and run on GPU. The single-game step
is written for a single state; `step_batch` applies `vmap` + `jit` for
batched execution.

Parity expectation: with `jax_enable_x64`, the JAX path matches Kaggle
bit-exactly on the parity harness (same atol=0). Without x64 the engine
defaults to float32 and small last-bit divergences are expected.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .action import ActionBatch
from .init import (
    BOARD_SIZE,
    CENTER,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from .state import (
    BatchState,
    F_ANGLE,
    F_FROM,
    F_ID,
    F_OWNER,
    F_SHIPS,
    F_X,
    F_Y,
    MAX_COMET_GROUPS,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_PLAYERS_PAD,
    P_ID,
    P_OWNER,
    P_PROD,
    P_R,
    P_SHIPS,
    P_X,
    P_Y,
)


# ---------------------------------------------------------------------------
# Math helpers (single-game ops).
# ---------------------------------------------------------------------------


def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """Vectorizable swept-pair: returns bool. All inputs scalar or broadcastable."""
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r

    disc = b * b - 4.0 * a * c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    # Avoid div-by-zero: if a < 1e-12, branch returns (c <= 0).
    a_safe = jnp.where(a < 1e-12, 1.0, a)
    t1 = (-b - sq) / (2.0 * a_safe)
    t2 = (-b + sq) / (2.0 * a_safe)
    quadratic_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    degenerate_hit = c <= 0.0
    return jnp.where(a < 1e-12, degenerate_hit, quadratic_hit)


def _point_seg_dist(px, py, vx, vy, wx, wy):
    """Distance from point (px,py) to segment (vx,vy)-(wx,wy)."""
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    # t along segment, clipped to [0, 1]. When l2 == 0, t doesn't matter
    # because v == w and we just return |p - v|.
    l2_safe = jnp.where(l2 == 0.0, 1.0, l2)
    t_raw = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2_safe
    t = jnp.clip(t_raw, 0.0, 1.0)
    qx = vx + t * (wx - vx)
    qy = vy + t * (wy - vy)
    d_seg = jnp.sqrt((px - qx) ** 2 + (py - qy) ** 2)
    d_v = jnp.sqrt((px - vx) ** 2 + (py - vy) ** 2)
    return jnp.where(l2 == 0.0, d_v, d_seg)


def _fleet_speed(ships, ship_speed):
    """Fleet speed matches Kaggle: capped log-curve over ship count."""
    speed = 1.0 + (ship_speed - 1.0) * (jnp.log(ships) / jnp.log(1000.0)) ** 1.5
    return jnp.minimum(speed, ship_speed)


# ---------------------------------------------------------------------------
# Single-game step. Operates on a `BatchState` slice (no batch dim).
# ---------------------------------------------------------------------------


def _step_single(state: BatchState, action_moves: jnp.ndarray,
                 action_mask: jnp.ndarray) -> BatchState:
    """One-tick advance for a single game. Trace-time fixed shapes only.

    `action_moves`: (NUM_PLAYERS_PAD, MAX_ACTIONS_PER_PLAYER, 3)
    `action_mask` : (NUM_PLAYERS_PAD, MAX_ACTIONS_PER_PLAYER)
    """

    # Quick-exit if game already done: return state unchanged (no step++).
    # We implement via `lax.cond` at the boundary to avoid wasted work.
    is_done = state.done
    do_nothing = lambda _: state
    do_work = lambda _: _step_active(state, action_moves, action_mask)
    return lax.cond(is_done, do_nothing, do_work, operand=None)


def _step_active(state: BatchState, action_moves: jnp.ndarray,
                 action_mask: jnp.ndarray) -> BatchState:
    """Active-game step body (called only when not done)."""

    step = state.step
    n_orig = state.n_initial
    angular_velocity = state.angular_velocity
    ship_speed = state.ship_speed
    episode_steps = state.episode_steps
    num_players = state.num_players

    planets = state.planets
    planet_mask = state.planet_mask
    initial_planets = state.initial_planets
    initial_mask = state.initial_mask
    fleets = state.fleets
    fleet_mask = state.fleet_mask
    next_fleet_id = state.next_fleet_id
    comet_path_index = state.comet_path_index
    comet_group_active = state.comet_group_active
    comet_path_lens = state.comet_path_lens
    comet_planet_ids = state.comet_planet_ids
    comet_ships_init = state.comet_ships_init
    comet_spawn_step = state.comet_spawn_step
    comet_group_valid = state.comet_group_valid
    comet_paths = state.comet_paths

    comet_slots = n_orig + jnp.arange(4, dtype=jnp.int32)  # (4,)

    # ------------------------------------------------------------------
    # 0a. Expired-comet removal (BEFORE fleet launch).
    # ------------------------------------------------------------------
    # A group is expired iff active AND path_index >= min(path_lens over qi).
    min_path_lens = jnp.min(comet_path_lens, axis=1)  # (MAX_COMET_GROUPS,)
    pre_expired = comet_group_active & (comet_path_index >= min_path_lens)

    # Which planet slots belong to a pre-expired group?
    # slot_is_pre_expired[s] = any(pre_expired[gi] and s in comet_slots)
    # Since comet slots are fixed at n_orig..n_orig+3 and only ONE group
    # can be active at a time (by lifetime argument), pre_expired across
    # groups OR-reduces to a single bool per quadrant slot.
    any_pre_expired = jnp.any(pre_expired)
    slot_idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    is_comet_slot = (slot_idx >= n_orig) & (slot_idx < n_orig + 4)
    pre_expire_slot = is_comet_slot & any_pre_expired

    planet_mask = jnp.where(pre_expire_slot, False, planet_mask)
    initial_mask = jnp.where(pre_expire_slot, False, initial_mask)
    planets = jnp.where(pre_expire_slot[:, None], 0.0, planets)
    initial_planets = jnp.where(pre_expire_slot[:, None], 0.0, initial_planets)
    comet_group_active = jnp.where(pre_expired, False, comet_group_active)

    # ------------------------------------------------------------------
    # 0b. Comet spawn at (step + 1) in COMET_SPAWN_STEPS.
    # ------------------------------------------------------------------
    target = step + 1
    is_spawn_step = jnp.any(
        target == jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.int32)
    )
    # Group to activate: index gi where comet_spawn_step[gi] == target and valid
    # and currently inactive. Since spawn_steps are unique per group, at most one.
    spawn_mask = (
        comet_group_valid
        & (comet_spawn_step == target)
        & ~comet_group_active
    )
    activate = is_spawn_step & spawn_mask  # (MAX_COMET_GROUPS,)
    any_activate = jnp.any(activate)
    # Which group index activates? argmax of activate (assumes at most one).
    activated_gi = jnp.argmax(activate.astype(jnp.int32))

    # Update active flag.
    comet_group_active = jnp.where(activate, True, comet_group_active)
    # Reset path_index to -1 for the activated group.
    comet_path_index = jnp.where(activate, -1, comet_path_index)

    # Materialize the 4 comet-planet rows for the activated group.
    def _build_comet_row(qi):
        pid = comet_planet_ids[activated_gi, qi]
        ships = comet_ships_init[activated_gi]
        row = jnp.array(
            [pid, -1, -99.0, -99.0, COMET_RADIUS, ships, 1], dtype=jnp.float64
        )
        return row

    new_comet_rows = jax.vmap(_build_comet_row)(jnp.arange(4))  # (4, 7)

    def _maybe_spawn(p_arr, ip_arr, pm_arr, im_arr):
        # Write rows into slots [n_orig..n_orig+3] iff any_activate.
        def _write(arrays):
            p, ip, pm, im = arrays
            n0 = n_orig.astype(jnp.int32)
            p2 = lax.dynamic_update_slice(p, new_comet_rows,
                                          (n0, jnp.int32(0)))
            ip2 = lax.dynamic_update_slice(ip, new_comet_rows,
                                           (n0, jnp.int32(0)))
            mask_update = jnp.ones((4,), dtype=jnp.bool_)
            pm2 = lax.dynamic_update_slice(pm, mask_update, (n0,))
            im2 = lax.dynamic_update_slice(im, mask_update, (n0,))
            return (p2, ip2, pm2, im2)

        def _passthrough(arrays):
            return arrays

        return lax.cond(any_activate, _write, _passthrough,
                        operand=(p_arr, ip_arr, pm_arr, im_arr))

    planets, initial_planets, planet_mask, initial_mask = _maybe_spawn(
        planets, initial_planets, planet_mask, initial_mask
    )

    # ------------------------------------------------------------------
    # 1. Fleet launch (process_moves). Sequential per (player, slot).
    # ------------------------------------------------------------------
    n_actions_total = NUM_PLAYERS_PAD * action_moves.shape[1]
    flat_moves = action_moves.reshape(n_actions_total, 3)
    flat_mask = action_mask.reshape(n_actions_total)
    flat_player = (
        jnp.repeat(jnp.arange(NUM_PLAYERS_PAD, dtype=jnp.int32),
                   action_moves.shape[1])
    )

    def _launch_step(carry, idx):
        planets_c, fleets_c, fleet_mask_c, next_id_c = carry
        valid = flat_mask[idx]
        from_id = flat_moves[idx, 0].astype(jnp.int32)
        angle = flat_moves[idx, 1]
        ships = flat_moves[idx, 2].astype(jnp.int32)
        player = flat_player[idx]
        valid = valid & (player < num_players) & (ships > 0)

        # Find planet slot whose id == from_id and is in use.
        match = planet_mask & (planets_c[:, P_ID].astype(jnp.int32) == from_id)
        any_match = jnp.any(match)
        # First matching slot — only one is possible since ids unique.
        slot_idx = jnp.argmax(match.astype(jnp.int32))

        owner_ok = planets_c[slot_idx, P_OWNER].astype(jnp.int32) == player
        ships_avail = planets_c[slot_idx, P_SHIPS].astype(jnp.int32)
        ships_ok = ships_avail >= ships

        do_launch = valid & any_match & owner_ok & ships_ok

        # Find first free fleet slot. argmax over ~fleet_mask picks the
        # lowest free index; if all full, argmax returns 0 (wrong), so we
        # guard with `any_free` below.
        free_mask = ~fleet_mask_c
        any_free = jnp.any(free_mask)
        f_slot = jnp.argmax(free_mask.astype(jnp.int32))

        do_launch = do_launch & any_free

        # Build the fleet row.
        radius = planets_c[slot_idx, P_R]
        start_x = planets_c[slot_idx, P_X] + jnp.cos(angle) * (radius + 0.1)
        start_y = planets_c[slot_idx, P_Y] + jnp.sin(angle) * (radius + 0.1)
        fleet_row = jnp.stack([
            next_id_c.astype(jnp.float64),
            player.astype(jnp.float64),
            start_x,
            start_y,
            angle,
            from_id.astype(jnp.float64),
            ships.astype(jnp.float64),
        ])

        # Apply launch.
        ships_after = jnp.where(do_launch, ships_avail - ships, ships_avail)
        planets_n = planets_c.at[slot_idx, P_SHIPS].set(ships_after.astype(jnp.float64))
        fleets_n = jnp.where(
            do_launch,
            fleets_c.at[f_slot].set(fleet_row),
            fleets_c,
        )
        fleet_mask_n = jnp.where(
            do_launch,
            fleet_mask_c.at[f_slot].set(True),
            fleet_mask_c,
        )
        next_id_n = jnp.where(do_launch, next_id_c + 1, next_id_c)
        return (planets_n, fleets_n, fleet_mask_n, next_id_n), None

    (planets, fleets, fleet_mask, next_fleet_id), _ = lax.scan(
        _launch_step,
        (planets, fleets, fleet_mask, next_fleet_id),
        jnp.arange(n_actions_total, dtype=jnp.int32),
    )

    # ------------------------------------------------------------------
    # 2. Production.
    # ------------------------------------------------------------------
    owned = planet_mask & (planets[:, P_OWNER].astype(jnp.int32) != -1)
    planets = planets.at[:, P_SHIPS].set(
        jnp.where(owned, planets[:, P_SHIPS] + planets[:, P_PROD], planets[:, P_SHIPS])
    )

    # ------------------------------------------------------------------
    # 3a. Planet motion (vectorized).
    # ------------------------------------------------------------------
    planet_old_x = planets[:, P_X]
    planet_old_y = planets[:, P_Y]

    init_dx = initial_planets[:, P_X] - CENTER
    init_dy = initial_planets[:, P_Y] - CENTER
    r_orbit = jnp.sqrt(init_dx ** 2 + init_dy ** 2)
    # Static condition matches Kaggle exactly.
    is_orbiting = (r_orbit + planets[:, P_R]) < ROTATION_RADIUS_LIMIT
    init_angle = jnp.arctan2(init_dy, init_dx)
    cur_angle = init_angle + angular_velocity * step.astype(jnp.float64)
    new_x_orbit = CENTER + r_orbit * jnp.cos(cur_angle)
    new_y_orbit = CENTER + r_orbit * jnp.sin(cur_angle)

    # Build planet_new ignoring comets first.
    is_comet_slot_now = (jnp.arange(MAX_PLANETS) >= n_orig) & (
        jnp.arange(MAX_PLANETS) < n_orig + 4
    )
    # Regular (non-comet) motion uses orbit-or-static.
    planet_new_x_reg = jnp.where(is_orbiting & ~is_comet_slot_now,
                                 new_x_orbit, planet_old_x)
    planet_new_y_reg = jnp.where(is_orbiting & ~is_comet_slot_now,
                                 new_y_orbit, planet_old_y)
    # Collision check: default to planet_mask (real planets only).
    planet_check_collision_reg = planet_mask & ~is_comet_slot_now

    # ------------------------------------------------------------------
    # 3b. Comet motion: advance index, place at path[idx]; expired -> stay.
    # ------------------------------------------------------------------
    new_path_index = jnp.where(comet_group_active,
                               comet_path_index + 1,
                               comet_path_index)

    # Identify the (possibly newly) active group's idx, lens, paths.
    any_active = jnp.any(comet_group_active)
    active_gi = jnp.argmax(comet_group_active.astype(jnp.int32))
    active_idx = new_path_index[active_gi]
    # Per-quadrant length for active group.
    active_lens = comet_path_lens[active_gi]  # (4,)
    # Expired (in this tick) per quadrant: idx >= len.
    expired_q = active_idx >= active_lens  # (4,) bool
    any_q_expired = jnp.any(expired_q) & any_active

    # Read path[idx] safely (clamp idx within bounds to avoid OOB).
    safe_idx = jnp.clip(active_idx, 0, comet_paths.shape[2] - 1)
    paths_at_idx = comet_paths[active_gi, :, safe_idx, :]  # (4, 2)

    # For each comet slot (0..3 corresponding to quadrant qi):
    # - If not active: leave as-is (zero entries handled by mask).
    # - If active and expired this tick: stay at old_pos, collision check ON.
    # - If active and not expired: move to path[idx], collision check on iff
    #     old_pos[0] >= 0 (first placement off-board uses old=(-99,-99)).
    planet_new_x_cmt = planet_new_x_reg
    planet_new_y_cmt = planet_new_y_reg
    planet_check_collision_cmt = planet_check_collision_reg

    # Build per-quadrant comet updates in a small Python loop (qi = 0..3
    # — unrolled at trace time, no Python dependence on values).
    for qi in range(4):
        slot = n_orig + qi
        old_x = jnp.take(planet_old_x, slot)
        old_y = jnp.take(planet_old_y, slot)
        cx = paths_at_idx[qi, 0]
        cy = paths_at_idx[qi, 1]

        do_stay = any_active & expired_q[qi]
        # Move only if active and NOT expired this tick.
        do_move = any_active & ~expired_q[qi]
        new_x_slot = jnp.where(do_move, cx, old_x)
        new_y_slot = jnp.where(do_move, cy, old_y)
        # Collision check: stay -> True; move -> (old_x >= 0); otherwise reg.
        check_move = old_x >= 0.0
        new_check = jnp.where(do_stay, True,
                              jnp.where(do_move, check_move,
                                        planet_check_collision_reg[slot]))

        planet_new_x_cmt = planet_new_x_cmt.at[slot].set(new_x_slot)
        planet_new_y_cmt = planet_new_y_cmt.at[slot].set(new_y_slot)
        planet_check_collision_cmt = planet_check_collision_cmt.at[slot].set(new_check)

    planet_new_x = planet_new_x_cmt
    planet_new_y = planet_new_y_cmt
    planet_check_collision = planet_check_collision_cmt

    # Track which groups are newly-expired-this-tick (will be removed
    # AFTER combat).
    newly_expired = jnp.zeros(MAX_COMET_GROUPS, dtype=jnp.bool_)
    newly_expired = newly_expired.at[active_gi].set(any_q_expired)

    # Set comet_path_index back into the array (only for active groups).
    comet_path_index = jnp.where(comet_group_active,
                                 new_path_index,
                                 comet_path_index)

    # ------------------------------------------------------------------
    # 4. Fleet movement + per-fleet first-hit collision.
    # ------------------------------------------------------------------
    ships_per_fleet = fleets[:, F_SHIPS]
    speeds = _fleet_speed(jnp.maximum(ships_per_fleet, 1.0), ship_speed)
    angles = fleets[:, F_ANGLE]
    old_fx = fleets[:, F_X]
    old_fy = fleets[:, F_Y]
    new_fx = old_fx + jnp.cos(angles) * speeds
    new_fy = old_fy + jnp.sin(angles) * speeds

    # Persist new positions on the fleet rows (for snapshot ordering even
    # when fleet later dies this tick — matches Kaggle which mutates in
    # place before the dead-fleet filter).
    fleets = fleets.at[:, F_X].set(new_fx)
    fleets = fleets.at[:, F_Y].set(new_fy)

    # Per-fleet hit mask vs all planets: (MAX_FLEETS, MAX_PLANETS).
    # We compute swept-pair hit for every (fleet, planet) and combine with
    # `planet_check_collision & planet_mask`.
    radii = planets[:, P_R]
    hit_grid = _swept_pair_hit(
        old_fx[:, None], old_fy[:, None],
        new_fx[:, None], new_fy[:, None],
        planet_old_x[None, :], planet_old_y[None, :],
        planet_new_x[None, :], planet_new_y[None, :],
        radii[None, :],
    )  # (MAX_FLEETS, MAX_PLANETS)
    eligible = planet_check_collision[None, :] & planet_mask[None, :]
    hit_grid = hit_grid & eligible & fleet_mask[:, None]

    # First-hit per fleet: lowest planet slot index where hit_grid is True.
    BIG = jnp.int32(MAX_PLANETS)
    p_slots = jnp.arange(MAX_PLANETS, dtype=jnp.int32)[None, :]  # (1, MP)
    masked_slots = jnp.where(hit_grid, p_slots, BIG)
    first_hit = jnp.min(masked_slots, axis=1)  # (MAX_FLEETS,)
    hits_planet = first_hit < BIG

    # Out-of-bounds.
    oob = ~((0.0 <= new_fx) & (new_fx <= BOARD_SIZE)
            & (0.0 <= new_fy) & (new_fy <= BOARD_SIZE))
    # Sun crossing.
    sun_d = _point_seg_dist(
        jnp.float64(CENTER), jnp.float64(CENTER),
        old_fx, old_fy, new_fx, new_fy,
    )
    crossed_sun = sun_d < SUN_RADIUS

    fleets_alive = fleet_mask & ~hits_planet & ~oob & ~crossed_sun

    # ------------------------------------------------------------------
    # 5. Combat resolution per planet (vectorized over planets).
    # ------------------------------------------------------------------
    # For each (fleet, planet), if first_hit[fleet] == planet, the fleet
    # contributes its ships to that planet's combat.
    fleet_hits_planet = hits_planet[:, None] & (
        first_hit[:, None] == p_slots
    )  # (MAX_FLEETS, MAX_PLANETS)
    # Per-player ship sums: (NUM_PLAYERS_PAD, MAX_PLANETS).
    owners = fleets[:, F_OWNER].astype(jnp.int32)
    player_idx = jnp.arange(NUM_PLAYERS_PAD, dtype=jnp.int32)
    owner_match = owners[:, None] == player_idx[None, :]  # (MF, NP)
    ships_int = fleets[:, F_SHIPS].astype(jnp.int32)
    # contribution[f, p, q] = fleet_hits_planet[f, p] * owner_match[f, q] * ships
    # Sum over f gives (MAX_PLANETS, NUM_PLAYERS_PAD).
    contributions = (
        fleet_hits_planet[:, :, None]
        & owner_match[:, None, :]
    ).astype(jnp.int32) * ships_int[:, None, None]
    ships_per_planet_player = jnp.sum(contributions, axis=0)  # (MP, NP)

    # First-appearance order: track for each (planet, player) the earliest
    # fleet index that contributed. Players with no contribution stay at BIG.
    F_BIG = jnp.int32(MAX_FLEETS)
    f_indices = jnp.arange(MAX_FLEETS, dtype=jnp.int32)
    contributes = fleet_hits_planet[:, :, None] & owner_match[:, None, :]  # (MF, MP, NP)
    contrib_idx = jnp.where(contributes, f_indices[:, None, None], F_BIG)
    first_seen = jnp.min(contrib_idx, axis=0)  # (MP, NP)

    # For each planet, find top1 and top2 owners. Custom rank: among the
    # NUM_PLAYERS_PAD slots, pick the one with the LARGEST ship count, ties
    # broken by SMALLEST first_seen (Kaggle's dict-insertion order).
    ships_arr = ships_per_planet_player  # (MP, NP)
    # Rank key: maximize ships, minimize first_seen. Combine into a single
    # sortable key with ships dominating.
    # key = ships * (1 + MF) - first_seen   (larger key = better)
    rank_key = ships_arr * (MAX_FLEETS + 1) - first_seen
    # Disallow players with zero ships (not actually present).
    NEG_INF = jnp.int32(-2 ** 31 + 1)
    rank_key = jnp.where(ships_arr > 0, rank_key, NEG_INF)

    top1_owner = jnp.argmax(rank_key, axis=1)  # (MP,)
    top1_ships = jnp.take_along_axis(ships_arr, top1_owner[:, None], axis=1)[:, 0]

    # Mask out top1 to find top2.
    mp_idx = jnp.arange(NUM_PLAYERS_PAD, dtype=jnp.int32)[None, :]
    top2_key = jnp.where(mp_idx == top1_owner[:, None], NEG_INF, rank_key)
    top2_owner = jnp.argmax(top2_key, axis=1)
    top2_ships = jnp.take_along_axis(ships_arr, top2_owner[:, None], axis=1)[:, 0]
    has_top2 = jnp.take_along_axis(top2_key, top2_owner[:, None], axis=1)[:, 0] > NEG_INF

    # Survivor compute.
    surv_ships_single = top1_ships
    surv_owner_single = top1_owner
    surv_ships_multi_raw = top1_ships - top2_ships
    tied = top1_ships == top2_ships
    surv_ships_multi = jnp.where(tied, 0, surv_ships_multi_raw)
    surv_owner_multi = jnp.where(surv_ships_multi > 0, top1_owner, -1)

    surv_ships = jnp.where(has_top2, surv_ships_multi, surv_ships_single)
    surv_owner = jnp.where(has_top2, surv_owner_multi, surv_owner_single)

    has_combat = jnp.any(fleet_hits_planet, axis=0) & planet_mask  # (MP,)
    apply_combat = has_combat & (surv_ships > 0)

    cur_owner = planets[:, P_OWNER].astype(jnp.int32)
    cur_ships = planets[:, P_SHIPS].astype(jnp.int32)
    same_owner = cur_owner == surv_owner
    delta = jnp.where(same_owner, surv_ships, -surv_ships)
    new_ships_raw = cur_ships + delta
    # If new_ships_raw < 0, flip owner and take abs.
    flip = new_ships_raw < 0
    final_owner = jnp.where(apply_combat & flip, surv_owner, cur_owner)
    final_ships = jnp.where(apply_combat,
                            jnp.where(flip, -new_ships_raw, new_ships_raw),
                            cur_ships)

    planets = planets.at[:, P_OWNER].set(final_owner.astype(jnp.float64))
    planets = planets.at[:, P_SHIPS].set(final_ships.astype(jnp.float64))

    # Apply planet motion now (after combat resolves on old/new sweep).
    planets = planets.at[:, P_X].set(planet_new_x)
    planets = planets.at[:, P_Y].set(planet_new_y)

    # ------------------------------------------------------------------
    # Post-combat: drop dead fleets, remove newly-expired comets.
    # ------------------------------------------------------------------
    # Zero out dead fleets and update mask.
    keep_fleet = fleets_alive
    fleets = jnp.where(keep_fleet[:, None], fleets, 0.0)
    fleet_mask = keep_fleet

    # Remove newly-expired comet planets.
    any_expired_now = jnp.any(newly_expired)
    remove_slot = is_comet_slot_now & any_expired_now
    planet_mask = jnp.where(remove_slot, False, planet_mask)
    initial_mask = jnp.where(remove_slot, False, initial_mask)
    planets = jnp.where(remove_slot[:, None], 0.0, planets)
    initial_planets = jnp.where(remove_slot[:, None], 0.0, initial_planets)
    comet_group_active = jnp.where(newly_expired, False, comet_group_active)

    # ------------------------------------------------------------------
    # 6. Termination + rewards.
    # ------------------------------------------------------------------
    step_new = step + 1
    terminated_by_step = step_new - 1 >= episode_steps - 2  # uses pre-increment step
    # Re-evaluate alive players (after combat / comet removal).
    owner_int = planets[:, P_OWNER].astype(jnp.int32)
    owner_valid = planet_mask & (owner_int != -1)
    fowner_int = fleets[:, F_OWNER].astype(jnp.int32)
    fleet_valid = fleet_mask

    def _player_alive(p):
        return jnp.any(owner_valid & (owner_int == p)) | jnp.any(
            fleet_valid & (fowner_int == p)
        )

    alive = jax.vmap(_player_alive)(jnp.arange(NUM_PLAYERS_PAD, dtype=jnp.int32))
    # Only count for actual players (< num_players).
    real_player = jnp.arange(NUM_PLAYERS_PAD, dtype=jnp.int32) < num_players
    alive_real = alive & real_player
    n_alive = jnp.sum(alive_real.astype(jnp.int32))
    terminated_by_alive = n_alive <= 1
    terminated = terminated_by_step | terminated_by_alive

    # Scores: per-player total ships across owned planets + fleets.
    planet_ship_sums = jnp.zeros(NUM_PLAYERS_PAD, dtype=jnp.int32)
    fleet_ship_sums = jnp.zeros(NUM_PLAYERS_PAD, dtype=jnp.int32)

    planet_ships_int = planets[:, P_SHIPS].astype(jnp.int32)
    fleet_ships_int = fleets[:, F_SHIPS].astype(jnp.int32)

    def _per_player_score(p):
        ps = jnp.sum(jnp.where(owner_valid & (owner_int == p), planet_ships_int, 0))
        fs = jnp.sum(jnp.where(fleet_valid & (fowner_int == p), fleet_ships_int, 0))
        return ps + fs

    scores = jax.vmap(_per_player_score)(jnp.arange(NUM_PLAYERS_PAD, dtype=jnp.int32))
    # Restrict to real players for max.
    max_score = jnp.max(jnp.where(real_player, scores, jnp.int32(-2 ** 31)))
    is_winner = (scores == max_score) & (max_score > 0) & real_player
    new_rewards = jnp.where(real_player,
                            jnp.where(is_winner, 1, -1),
                            0).astype(jnp.int32)
    rewards = jnp.where(terminated, new_rewards, state.rewards)

    return state._replace(
        planets=planets,
        planet_mask=planet_mask,
        initial_planets=initial_planets,
        initial_mask=initial_mask,
        fleets=fleets,
        fleet_mask=fleet_mask,
        next_fleet_id=next_fleet_id,
        comet_path_index=comet_path_index,
        comet_group_active=comet_group_active,
        step=step_new,
        done=terminated,
        rewards=rewards,
    )


# ---------------------------------------------------------------------------
# Batched/JIT entry points.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=())
def step_batch(state: BatchState, actions: ActionBatch) -> BatchState:
    """JIT'd batched step. Pure function; safe to use with `vmap` already
    baked in via leading batch dim."""
    return jax.vmap(_step_single)(state, actions.moves, actions.mask)


@partial(jax.jit, static_argnums=())
def step_single(state: BatchState, action_moves: jnp.ndarray,
                action_mask: jnp.ndarray) -> BatchState:
    """Single-game JIT (no leading batch dim) — useful for parity tests."""
    return _step_single(state, action_moves, action_mask)
