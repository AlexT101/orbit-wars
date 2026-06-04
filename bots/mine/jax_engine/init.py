"""CPU-side initialization that bit-exactly matches Kaggle's seeded init.

We reuse Python's `random.Random` so the planet layout, home assignment,
and all five comet trajectories match Kaggle exactly for the same seed.
The output is a `BatchState` of padded numpy arrays ready to upload to
the device.

Pre-computing all five comet groups at reset works because:

  * comet path generation uses `initial_planets` minus the current
    `comet_planet_ids`. Each comet's visible path length is at most 40,
    so comet k always expires by step (50 + 100*k) + 40 — well before
    comet k+1 spawns. By the next spawn the prior comet's planets have
    been removed from both `initial_planets` and `comet_planet_ids`,
    so each spawn sees the same `initial_planets` as if no prior comet
    had ever spawned.
  * Comet ids are assigned from `max(planet_id) + 1`. Since prior
    comet planets are always gone by the next spawn, every comet group
    gets the same id range `[N, N+1, N+2, N+3]` where N = original
    planet count.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .state import (
    BatchState,
    FLEET_COLS,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_PLAYERS_PAD,
    PLANET_COLS,
)


# Constants mirrored from kaggle_environments.envs.orbit_wars.orbit_wars.
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

DEFAULT_EPISODE_STEPS = 500
DEFAULT_SHIP_SPEED = 6.0
DEFAULT_COMET_SPEED = 4.0


# ---------------------------------------------------------------------------
# Kaggle helpers (kept structurally identical for bit-exact float ops).
# ---------------------------------------------------------------------------


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _generate_planets(rng):
    planets = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0
    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)
        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue
        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        temp = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]
        valid = True
        for tp in temp:
            for p in planets:
                if _dist((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            planets.extend(temp)
            id_counter += 4
            static_groups += 1

    attempts = 0
    max_attempts = 5000
    has_orbiting = False
    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        orbital_radius = _dist((x, y), (CENTER, CENTER))
        if orbital_radius < SUN_RADIUS + r + 10:
            continue
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue
        valid = True
        ships = rng.randint(5, 30)
        temp = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]
        for tp in temp:
            tp_orb = _dist((tp[2], tp[3]), (CENTER, CENTER))
            tp_rot = tp_orb + tp[4] < ROTATION_RADIUS_LIMIT
            for p in planets:
                p_orb = _dist((p[2], p[3]), (CENTER, CENTER))
                p_rot = p_orb + p[4] < ROTATION_RADIUS_LIMIT
                if _dist((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
                if tp_rot != p_rot:
                    if abs(tp_orb - p_orb) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp)
            id_counter += 4
    return planets


def _generate_comet_paths(initial_planets, angular_velocity, spawn_step,
                          comet_planet_ids, comet_speed, rng):
    cp_ids = set(comet_planet_ids or [])
    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue
        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)
        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))
        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _dist(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed
        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i
        if board_start is None:
            continue
        visible = path[board_start: board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue
        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]
        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in cp_ids:
                continue
            pr = _dist((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)
        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if _dist((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break
            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if _dist(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx ** 2 + dy ** 2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _dist(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return paths
    return None


# ---------------------------------------------------------------------------
# Per-seed init (returns numpy buffers shaped for the BatchState pytree).
# ---------------------------------------------------------------------------


def _empty_planet_row():
    return [0.0] * PLANET_COLS


def init_single(seed: int, num_players: int,
                episode_steps: int = DEFAULT_EPISODE_STEPS,
                ship_speed: float = DEFAULT_SHIP_SPEED,
                comet_speed: float = DEFAULT_COMET_SPEED) -> dict:
    """Initialize one game. Returns a dict of numpy arrays (one game's
    worth, no batch dim). `init_batch` stacks these."""

    assert num_players in (2, 4)

    rng = random.Random(seed)
    angular_velocity = rng.uniform(0.025, 0.05)
    planets = _generate_planets(rng)

    # Match Kaggle: snapshot initial_planets BEFORE home assignment so
    # the home planet's initial ownership/ships stay at -1/initial-ships.
    initial_planets = [p[:] for p in planets]

    # Home assignment (mutates obs0.planets only).
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

    n_planets = len(planets)
    assert n_planets <= MAX_PLANETS - 4, (
        f"too many initial planets ({n_planets}); raise MAX_PLANETS"
    )

    # Pad planets / initial_planets to MAX_PLANETS.
    planet_arr = np.zeros((MAX_PLANETS, PLANET_COLS), dtype=np.float64)
    planet_mask = np.zeros(MAX_PLANETS, dtype=bool)
    for i, p in enumerate(planets):
        planet_arr[i] = p
        planet_mask[i] = True
    init_arr = np.zeros((MAX_PLANETS, PLANET_COLS), dtype=np.float64)
    init_mask = np.zeros(MAX_PLANETS, dtype=bool)
    for i, p in enumerate(initial_planets):
        init_arr[i] = p
        init_mask[i] = True

    # Pre-generate all 5 comet groups. ID range is always [n_planets..n_planets+3]
    # (see module docstring for the lifetime argument).
    comet_paths = np.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH_LEN, 2),
                           dtype=np.float64)
    comet_path_lens = np.zeros((MAX_COMET_GROUPS, 4), dtype=np.int32)
    comet_planet_ids = np.zeros((MAX_COMET_GROUPS, 4), dtype=np.int32)
    comet_ships_init = np.zeros(MAX_COMET_GROUPS, dtype=np.int32)
    comet_spawn_step = np.zeros(MAX_COMET_GROUPS, dtype=np.int32)
    comet_group_valid = np.zeros(MAX_COMET_GROUPS, dtype=bool)

    for gi, spawn_step in enumerate(COMET_SPAWN_STEPS):
        crng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        paths = _generate_comet_paths(
            initial_planets,
            angular_velocity,
            spawn_step,
            None,
            comet_speed,
            crng,
        )
        if paths is None:
            # Failure: comet_ships_init draws must still happen on the
            # Kaggle path AFTER a successful path gen. So we mirror Kaggle:
            # only call the 4 randint draws after success.
            continue
        ships = min(
            crng.randint(1, 99),
            crng.randint(1, 99),
            crng.randint(1, 99),
            crng.randint(1, 99),
        )
        for qi, qpath in enumerate(paths):
            L = len(qpath)
            assert L <= MAX_COMET_PATH_LEN, (
                f"comet path too long ({L}); raise MAX_COMET_PATH_LEN"
            )
            # Store the raw visible path at slots 0..L-1; path_index ticks
            # -1 -> 0 -> ... -> L-1, expires when idx >= L (matches Kaggle).
            for ki, (cx, cy) in enumerate(qpath):
                comet_paths[gi, qi, ki, 0] = cx
                comet_paths[gi, qi, ki, 1] = cy
            comet_path_lens[gi, qi] = L
            comet_planet_ids[gi, qi] = n_planets + qi
        comet_ships_init[gi] = ships
        comet_spawn_step[gi] = spawn_step
        comet_group_valid[gi] = True

    return {
        "planets": planet_arr,
        "planet_mask": planet_mask,
        "initial_planets": init_arr,
        "initial_mask": init_mask,
        "fleets": np.zeros((MAX_FLEETS, FLEET_COLS), dtype=np.float64),
        "fleet_mask": np.zeros(MAX_FLEETS, dtype=bool),
        "next_fleet_id": np.int32(0),
        "comet_paths": comet_paths,
        "comet_path_lens": comet_path_lens,
        "comet_path_index": np.full(MAX_COMET_GROUPS, -1, dtype=np.int32),
        "comet_planet_ids": comet_planet_ids,
        "comet_ships_init": comet_ships_init,
        "comet_spawn_step": comet_spawn_step,
        "comet_group_valid": comet_group_valid,
        "comet_group_active": np.zeros(MAX_COMET_GROUPS, dtype=bool),
        "step": np.int32(0),
        "angular_velocity": np.float64(angular_velocity),
        "done": np.bool_(False),
        "rewards": np.zeros(NUM_PLAYERS_PAD, dtype=np.int32),
        "num_players": np.int32(num_players),
        "episode_steps": np.int32(episode_steps),
        "ship_speed": np.float64(ship_speed),
        "seed": np.int32(seed),
        "n_initial": np.int32(n_planets),
    }


def init_batch(seeds: list[int], num_players: int,
               episode_steps: int = DEFAULT_EPISODE_STEPS,
               ship_speed: float = DEFAULT_SHIP_SPEED,
               comet_speed: float = DEFAULT_COMET_SPEED) -> BatchState:
    """Initialize a batch of games (different seeds, same num_players).
    Returns a `BatchState` whose leaves are numpy arrays (caller can
    jax.device_put if desired)."""
    singles = [
        init_single(s, num_players, episode_steps, ship_speed, comet_speed)
        for s in seeds
    ]
    stacked = {k: np.stack([s[k] for s in singles], axis=0) for k in singles[0]}
    return BatchState(**stacked)
