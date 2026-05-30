"""Observation → fixed-size tensors.

Per-planet table (MAX_PLANETS × PLANET_FEATURES) + global summary vector +
validity mask. Anything beyond MAX_PLANETS planets is truncated.

Features intentionally crude — see the design doc for the apollo-grounded
version we'll grow into."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import GLOBAL_FEATURES, MAX_PLANETS, PLANET_FEATURES


# Owner index layout per planet (one-hot, after rotation so me=0):
#   [is_me, is_enemy, is_neutral]
# Plus: log(1+ships), ships/100 (clipped), production/5,
#   (x-50)/50, (y-50)/50, dist_to_sun/50, radius/3, is_my_home placeholder,
#   step/500.
# That's 12 features — matches PLANET_FEATURES.


@dataclass
class ObsView:
    """Parsed observation as numpy tensors plus the raw planet list."""

    planet_table: np.ndarray  # (MAX_PLANETS, PLANET_FEATURES)
    planet_mask: np.ndarray  # (MAX_PLANETS,) — 1 where slot is real
    mine_mask: np.ndarray  # (MAX_PLANETS,) — 1 where planet belongs to me
    globals_: np.ndarray  # (GLOBAL_FEATURES,)
    planets_raw: list  # raw [id, owner, x, y, radius, ships, production]
    id_to_slot: dict  # planet id → slot index in the table
    player: int
    step: int


def _planet_features(planet, player: int, step: int) -> np.ndarray:
    pid, owner, x, y, radius, ships, production = planet
    is_me = 1.0 if owner == player else 0.0
    is_enemy = 1.0 if (owner != player and owner != -1) else 0.0
    is_neutral = 1.0 if owner == -1 else 0.0
    log_ships = math.log1p(max(ships, 0))
    ships_norm = min(ships, 1000) / 100.0
    prod_norm = production / 5.0
    dx = (x - 50.0) / 50.0
    dy = (y - 50.0) / 50.0
    dist_sun = math.hypot(x - 50.0, y - 50.0) / 50.0
    radius_norm = radius / 3.0
    # Placeholder for is_my_home — unused for now, kept to anchor feature
    # count at PLANET_FEATURES.
    is_home_placeholder = 0.0
    step_norm = step / 500.0
    return np.array(
        [
            is_me,
            is_enemy,
            is_neutral,
            log_ships,
            ships_norm,
            prod_norm,
            dx,
            dy,
            dist_sun,
            radius_norm,
            is_home_placeholder,
            step_norm,
        ],
        dtype=np.float32,
    )


def parse_obs(obs) -> ObsView:
    """Build an ObsView from a Kaggle observation (dict or namespace)."""

    def get(key, default):
        if isinstance(obs, dict):
            value = obs.get(key, default)
        else:
            value = getattr(obs, key, default)
        return default if value is None else value

    player = int(get("player", 0))
    step = int(get("step", 0))
    planets = list(get("planets", []))
    fleets = list(get("fleets", []))

    table = np.zeros((MAX_PLANETS, PLANET_FEATURES), dtype=np.float32)
    mask = np.zeros((MAX_PLANETS,), dtype=np.float32)
    mine_mask = np.zeros((MAX_PLANETS,), dtype=np.float32)
    id_to_slot: dict = {}

    n = min(len(planets), MAX_PLANETS)
    for i in range(n):
        table[i] = _planet_features(planets[i], player, step)
        mask[i] = 1.0
        id_to_slot[int(planets[i][0])] = i
        if int(planets[i][1]) == player:
            mine_mask[i] = 1.0

    # Globals: step/500, my-share of ships, my-share of production,
    # ship totals (log), fleet count log, planet share, opp ship share,
    # neutral planet share.
    my_ships = 0
    enemy_ships = 0
    neutral_planets_count = 0
    my_prod = 0
    enemy_prod = 0
    my_count = 0
    enemy_count = 0
    for planet in planets:
        _, owner, _x, _y, _r, ships, prod = planet
        if owner == player:
            my_ships += ships
            my_prod += prod
            my_count += 1
        elif owner == -1:
            neutral_planets_count += 1
        else:
            enemy_ships += ships
            enemy_prod += prod
            enemy_count += 1
    for fleet in fleets:
        _, owner, _x, _y, _angle, _from, ships = fleet
        if owner == player:
            my_ships += ships
        elif owner != -1:
            enemy_ships += ships
    total_ships = max(my_ships + enemy_ships, 1)
    total_prod = max(my_prod + enemy_prod, 1)
    total_planets = max(my_count + enemy_count + neutral_planets_count, 1)
    g = np.array(
        [
            step / 500.0,
            my_ships / total_ships,
            my_prod / total_prod,
            math.log1p(my_ships) / 10.0,
            math.log1p(enemy_ships) / 10.0,
            my_count / total_planets,
            enemy_count / total_planets,
            neutral_planets_count / total_planets,
        ],
        dtype=np.float32,
    )
    assert g.shape[0] == GLOBAL_FEATURES

    return ObsView(
        planet_table=table,
        planet_mask=mask,
        mine_mask=mine_mask,
        globals_=g,
        planets_raw=planets,
        id_to_slot=id_to_slot,
        player=player,
        step=step,
    )
