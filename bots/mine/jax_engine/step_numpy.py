"""Numpy reference step function (single game, no batch dim).

Used as the parity oracle during the JAX port. Logic is intentionally
written in the same order as Kaggle's `interpreter`, with named helpers
where they make a diff easier to localize. Once this passes parity, the
JAX version in step.py mirrors it under jnp + vmap.

State here is a plain dict with the same keys as `BatchState` but one
game's slice (no leading batch dim).
"""

from __future__ import annotations

import math

import numpy as np

from .init import (
    BOARD_SIZE,
    CENTER,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from .state import (
    F_ANGLE,
    F_FROM,
    F_ID,
    F_OWNER,
    F_SHIPS,
    F_X,
    F_Y,
    MAX_COMET_GROUPS,
    NUM_PLAYERS_PAD,
    P_ID,
    P_OWNER,
    P_PROD,
    P_R,
    P_SHIPS,
    P_X,
    P_Y,
)


def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """Replica of Kaggle's swept_pair_hit. Returns bool."""
    d0x, d0y = ax - p0x, ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def _point_seg_dist(px, py, vx, vy, wx, wy):
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    if l2 == 0.0:
        return math.sqrt((px - vx) ** 2 + (py - vy) ** 2)
    t = max(0.0, min(1.0, ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2))
    qx = vx + t * (wx - vx)
    qy = vy + t * (wy - vy)
    return math.sqrt((px - qx) ** 2 + (py - qy) ** 2)


def step_numpy(state: dict, actions: list[list[list]]) -> dict:
    """Advance one tick. `state` is a single-game dict (no batch dim).
    `actions` is a list of per-player action lists in Python form
    (matches engine_parity_checker's JointAction)."""

    if bool(state["done"]):
        return state  # frozen

    num_players = int(state["num_players"])

    # ------------------------------------------------------------------
    # 0a. Expired-comet removal (BEFORE fleet launch). A group expires
    #     when its path_index >= path length for any quadrant. In our
    #     padded layout, all 4 quadrants share path_lens up to small
    #     differences, and `path_index >= min(path_lens) -> True`. The
    #     Kaggle code checks per-quadrant; in practice all 4 paths have
    #     the same visible length because they're symmetric copies.
    # ------------------------------------------------------------------

    # Work on copies of mutable arrays.
    planets = state["planets"].copy()
    planet_mask = state["planet_mask"].copy()
    initial_planets = state["initial_planets"].copy()
    initial_mask = state["initial_mask"].copy()
    fleets = state["fleets"].copy()
    fleet_mask = state["fleet_mask"].copy()
    next_fleet_id = int(state["next_fleet_id"])
    comet_path_index = state["comet_path_index"].copy()
    comet_group_active = state["comet_group_active"].copy()
    comet_path_lens = state["comet_path_lens"]
    comet_planet_ids = state["comet_planet_ids"]
    comet_ships_init = state["comet_ships_init"]
    comet_spawn_step = state["comet_spawn_step"]
    comet_group_valid = state["comet_group_valid"]
    comet_paths = state["comet_paths"]
    step = int(state["step"])
    angular_velocity = float(state["angular_velocity"])
    ship_speed = float(state["ship_speed"])
    episode_steps = int(state["episode_steps"])

    n_orig = int(state["n_initial"])

    # Expired-comet removal: clear planets/initial_planets slots for any
    # currently-active group whose path is exhausted.
    for gi in range(MAX_COMET_GROUPS):
        if not comet_group_active[gi]:
            continue
        idx = int(comet_path_index[gi])
        any_expired = False
        for qi in range(4):
            if idx >= int(comet_path_lens[gi, qi]):
                any_expired = True
                break
        if any_expired:
            # Remove all 4 comet planets for this group.
            for qi in range(4):
                slot = n_orig + qi
                planet_mask[slot] = False
                initial_mask[slot] = False
                planets[slot] = 0.0
                initial_planets[slot] = 0.0
            comet_group_active[gi] = False

    # ------------------------------------------------------------------
    # 0b. Comet spawn at (step + 1) in COMET_SPAWN_STEPS.
    # ------------------------------------------------------------------
    spawn_now = (step + 1) in COMET_SPAWN_STEPS
    if spawn_now:
        # Find the group whose spawn_step matches step+1.
        target = step + 1
        for gi in range(MAX_COMET_GROUPS):
            if (
                comet_group_valid[gi]
                and int(comet_spawn_step[gi]) == target
                and not comet_group_active[gi]
            ):
                # Activate group gi.
                comet_group_active[gi] = True
                comet_path_index[gi] = -1
                ships = int(comet_ships_init[gi])
                for qi in range(4):
                    slot = n_orig + qi
                    pid = int(comet_planet_ids[gi, qi])
                    # Initial position off-board.
                    planets[slot, P_ID] = pid
                    planets[slot, P_OWNER] = -1
                    planets[slot, P_X] = -99.0
                    planets[slot, P_Y] = -99.0
                    planets[slot, P_R] = COMET_RADIUS
                    planets[slot, P_SHIPS] = ships
                    planets[slot, P_PROD] = 1
                    planet_mask[slot] = True
                    initial_planets[slot] = planets[slot]
                    initial_mask[slot] = True
                break

    # ------------------------------------------------------------------
    # 1. Fleet launch (process_moves). Per-player, in order.
    # ------------------------------------------------------------------
    # Build a slot lookup from planet id -> slot for O(1) action validation.
    id_to_slot = {}
    for slot in range(planets.shape[0]):
        if planet_mask[slot]:
            id_to_slot[int(planets[slot, P_ID])] = slot

    for pid_player in range(num_players):
        action = actions[pid_player] if actions else []
        if not isinstance(action, list):
            continue
        for move in action:
            if not isinstance(move, list) or len(move) != 3:
                continue
            from_id, angle, ships = move
            try:
                ships = int(ships)
            except (TypeError, ValueError):
                continue
            if ships <= 0:
                continue
            slot = id_to_slot.get(int(from_id))
            if slot is None:
                continue
            if int(planets[slot, P_OWNER]) != pid_player:
                continue
            if int(planets[slot, P_SHIPS]) < ships:
                continue
            planets[slot, P_SHIPS] -= ships
            angle_f = float(angle)
            radius = float(planets[slot, P_R])
            start_x = float(planets[slot, P_X]) + math.cos(angle_f) * (radius + 0.1)
            start_y = float(planets[slot, P_Y]) + math.sin(angle_f) * (radius + 0.1)
            # Insert into first free fleet slot.
            free = np.flatnonzero(~fleet_mask)
            assert free.size > 0, "ran out of fleet slots; raise MAX_FLEETS"
            f_slot = int(free[0])
            fleets[f_slot, F_ID] = next_fleet_id
            fleets[f_slot, F_OWNER] = pid_player
            fleets[f_slot, F_X] = start_x
            fleets[f_slot, F_Y] = start_y
            fleets[f_slot, F_ANGLE] = angle_f
            fleets[f_slot, F_FROM] = int(from_id)
            fleets[f_slot, F_SHIPS] = ships
            fleet_mask[f_slot] = True
            next_fleet_id += 1

    # ------------------------------------------------------------------
    # 2. Production: owned planets gain prod ships.
    # ------------------------------------------------------------------
    owned = planet_mask & (planets[:, P_OWNER] != -1)
    planets[owned, P_SHIPS] += planets[owned, P_PROD]

    # ------------------------------------------------------------------
    # 3a. Planet motion: old_pos -> new_pos using INITIAL angle + av*step.
    # ------------------------------------------------------------------
    # The "step" Kaggle reads here defaults to 1 when missing; here it's
    # always set, so just use the local `step`. Comets are handled below.
    planet_old = planets[:, [P_X, P_Y]].copy()
    planet_new = planet_old.copy()
    planet_check_collision = planet_mask.copy()  # True where fleet can hit

    is_comet_slot = np.zeros(planets.shape[0], dtype=bool)
    for gi in range(MAX_COMET_GROUPS):
        if comet_group_active[gi]:
            for qi in range(4):
                is_comet_slot[n_orig + qi] = True

    for slot in range(planets.shape[0]):
        if not planet_mask[slot] or is_comet_slot[slot]:
            continue
        if not initial_mask[slot]:
            continue
        dx = float(initial_planets[slot, P_X]) - CENTER
        dy = float(initial_planets[slot, P_Y]) - CENTER
        r = math.sqrt(dx * dx + dy * dy)
        if r + float(planets[slot, P_R]) < ROTATION_RADIUS_LIMIT:
            init_angle = math.atan2(dy, dx)
            cur_angle = init_angle + angular_velocity * step
            planet_new[slot, 0] = CENTER + r * math.cos(cur_angle)
            planet_new[slot, 1] = CENTER + r * math.sin(cur_angle)
        # else: static -> new == old (already set)

    # ------------------------------------------------------------------
    # 3b. Comet movement: advance path_index, set new pos from path table.
    #     First placement (path_index transitions -1 -> 0) uses old_pos =
    #     off-board (-99,-99); we suppress collision for that tick.
    # ------------------------------------------------------------------
    newly_expired_groups = []
    for gi in range(MAX_COMET_GROUPS):
        if not comet_group_active[gi]:
            continue
        comet_path_index[gi] += 1
        idx = int(comet_path_index[gi])
        # Even after expiration check above, a group could have idx == path_len
        # on the very edge case where a quadrant has shorter length. Handle.
        any_q_expired = False
        for qi in range(4):
            if idx >= int(comet_path_lens[gi, qi]):
                any_q_expired = True
                break
        if any_q_expired:
            # Comet stays put (old_pos -> old_pos), then removed after combat.
            newly_expired_groups.append(gi)
            for qi in range(4):
                slot = n_orig + qi
                planet_check_collision[slot] = True  # stays at old_pos
                planet_new[slot] = planet_old[slot]
            continue
        # Place at path[idx] for each quadrant.
        for qi in range(4):
            slot = n_orig + qi
            cx = float(comet_paths[gi, qi, idx, 0])
            cy = float(comet_paths[gi, qi, idx, 1])
            planet_new[slot, 0] = cx
            planet_new[slot, 1] = cy
            # check_collision = (old_pos[0] >= 0). First placement: old is
            # the off-board placeholder (-99,-99) -> check=False.
            planet_check_collision[slot] = bool(planet_old[slot, 0] >= 0)

    # ------------------------------------------------------------------
    # 4. Fleet movement + continuous collision against planet swept-pair.
    # ------------------------------------------------------------------
    fleets_alive = fleet_mask.copy()
    # combat_hits: per-planet-slot, list of fleet slots that hit.
    combat_hits: list[list[int]] = [[] for _ in range(planets.shape[0])]

    # Iterate fleets in slot order (matches Kaggle's append order).
    for f_slot in range(fleets.shape[0]):
        if not fleets_alive[f_slot]:
            continue
        angle = float(fleets[f_slot, F_ANGLE])
        ships = float(fleets[f_slot, F_SHIPS])
        speed = 1.0 + (ship_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
        speed = min(speed, ship_speed)
        old_x = float(fleets[f_slot, F_X])
        old_y = float(fleets[f_slot, F_Y])
        new_x = old_x + math.cos(angle) * speed
        new_y = old_y + math.sin(angle) * speed
        fleets[f_slot, F_X] = new_x
        fleets[f_slot, F_Y] = new_y

        # Check planets in slot order; first hit wins.
        hit = -1
        for p_slot in range(planets.shape[0]):
            if not planet_mask[p_slot] or not planet_check_collision[p_slot]:
                continue
            p_old_x = planet_old[p_slot, 0]
            p_old_y = planet_old[p_slot, 1]
            p_new_x = planet_new[p_slot, 0]
            p_new_y = planet_new[p_slot, 1]
            r = float(planets[p_slot, P_R])
            if _swept_pair_hit(old_x, old_y, new_x, new_y,
                               p_old_x, p_old_y, p_new_x, p_new_y, r):
                hit = p_slot
                break
        if hit >= 0:
            combat_hits[hit].append(f_slot)
            fleets_alive[f_slot] = False
            continue

        # Out of bounds.
        if not (0.0 <= new_x <= BOARD_SIZE and 0.0 <= new_y <= BOARD_SIZE):
            fleets_alive[f_slot] = False
            continue

        # Sun crossing.
        if _point_seg_dist(CENTER, CENTER, old_x, old_y, new_x, new_y) < SUN_RADIUS:
            fleets_alive[f_slot] = False
            continue

    # Apply planet motion.
    planets[:, P_X] = planet_new[:, 0]
    planets[:, P_Y] = planet_new[:, 1]

    # Remove expired comets (post-combat).
    for gi in newly_expired_groups:
        comet_group_active[gi] = False
        for qi in range(4):
            slot = n_orig + qi
            planet_mask[slot] = False
            initial_mask[slot] = False
            planets[slot] = 0.0
            initial_planets[slot] = 0.0

    # ------------------------------------------------------------------
    # 5. Combat resolution per planet. Runs BEFORE we zero dead fleets so
    #    we can still read each fleet's owner/ships.
    # ------------------------------------------------------------------
    for p_slot in range(planets.shape[0]):
        hits = combat_hits[p_slot]
        if not hits or not planet_mask[p_slot]:
            continue
        # Sum ships per player.
        ships_per_player = [0] * NUM_PLAYERS_PAD
        for fs in hits:
            owner = int(fleets[fs, F_OWNER])
            ships_per_player[owner] += int(fleets[fs, F_SHIPS])
        # Kaggle sorts by ships desc; dict iteration preserves insertion
        # order so equal counts break by first-appearance. For p<=4 the
        # difference is rare, but we replicate: take owners in first-
        # appearance order, then stable-sort by -ships.
        first_order = []
        seen = set()
        for fs in hits:
            owner = int(fleets[fs, F_OWNER])
            if owner not in seen:
                seen.add(owner)
                first_order.append(owner)
        sorted_players = sorted(first_order, key=lambda o: -ships_per_player[o])
        top_owner = sorted_players[0]
        top_ships = ships_per_player[top_owner]
        if len(sorted_players) > 1:
            second_ships = ships_per_player[sorted_players[1]]
            survivor_ships = top_ships - second_ships
            if top_ships == second_ships:
                survivor_ships = 0
            survivor_owner = top_owner if survivor_ships > 0 else -1
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships
        if survivor_ships > 0:
            planet_owner = int(planets[p_slot, P_OWNER])
            if planet_owner == survivor_owner:
                planets[p_slot, P_SHIPS] += survivor_ships
            else:
                planets[p_slot, P_SHIPS] -= survivor_ships
                if planets[p_slot, P_SHIPS] < 0:
                    planets[p_slot, P_OWNER] = survivor_owner
                    planets[p_slot, P_SHIPS] = -planets[p_slot, P_SHIPS]

    # Drop dead fleets now that combat has read their owner/ships.
    for f_slot in range(fleets.shape[0]):
        if not fleets_alive[f_slot] and fleet_mask[f_slot]:
            fleet_mask[f_slot] = False
            fleets[f_slot] = 0.0
        else:
            fleet_mask[f_slot] = fleets_alive[f_slot]

    # ------------------------------------------------------------------
    # 6. Termination.
    # ------------------------------------------------------------------
    terminated = False
    if step >= episode_steps - 2:
        terminated = True
    alive_players = set()
    for p_slot in range(planets.shape[0]):
        if planet_mask[p_slot] and int(planets[p_slot, P_OWNER]) != -1:
            alive_players.add(int(planets[p_slot, P_OWNER]))
    for f_slot in range(fleets.shape[0]):
        if fleet_mask[f_slot]:
            alive_players.add(int(fleets[f_slot, F_OWNER]))
    if len(alive_players) <= 1:
        terminated = True

    rewards = state["rewards"].copy()
    if terminated:
        scores = [0] * NUM_PLAYERS_PAD
        for p_slot in range(planets.shape[0]):
            if planet_mask[p_slot]:
                owner = int(planets[p_slot, P_OWNER])
                if owner != -1:
                    scores[owner] += int(planets[p_slot, P_SHIPS])
        for f_slot in range(fleets.shape[0]):
            if fleet_mask[f_slot]:
                scores[int(fleets[f_slot, F_OWNER])] += int(fleets[f_slot, F_SHIPS])
        max_score = max(scores[:num_players])
        for i in range(num_players):
            if scores[i] == max_score and max_score > 0:
                rewards[i] = 1
            else:
                rewards[i] = -1

    new_state = dict(state)
    new_state["planets"] = planets
    new_state["planet_mask"] = planet_mask
    new_state["initial_planets"] = initial_planets
    new_state["initial_mask"] = initial_mask
    new_state["fleets"] = fleets
    new_state["fleet_mask"] = fleet_mask
    new_state["next_fleet_id"] = np.int32(next_fleet_id)
    new_state["comet_path_index"] = comet_path_index
    new_state["comet_group_active"] = comet_group_active
    new_state["step"] = np.int32(step + 1)
    new_state["done"] = np.bool_(terminated)
    new_state["rewards"] = rewards
    return new_state
