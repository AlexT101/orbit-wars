"""Derived evaluator features shared by the training pipeline.

The raw 46+12 layout intentionally stays close to the extractors. These
engineered columns add common matchup transforms that tree models otherwise
have to rediscover from paired "me" and "opponent" columns.
"""

from __future__ import annotations

import numpy as np


ENGINEERED_NAMES = [
    "eng.cur.ships_planets_diff",
    "eng.cur.ships_flying_diff",
    "eng.cur.ships_total_diff",
    "eng.cur.ships_total_share",
    "eng.cur.static_count_diff",
    "eng.cur.orbit_count_diff",
    "eng.cur.comet_count_diff",
    "eng.cur.planet_count_diff",
    "eng.cur.prod_static_diff",
    "eng.cur.prod_orbit_diff",
    "eng.cur.prod_comet_diff",
    "eng.cur.prod_total_diff",
    "eng.cur.prod_total_share",
    "eng.ext.ships_planets_diff",
    "eng.ext.static_count_diff",
    "eng.ext.orbit_count_diff",
    "eng.ext.comet_count_diff",
    "eng.ext.planet_count_diff",
    "eng.ext.prod_static_diff",
    "eng.ext.prod_orbit_diff",
    "eng.ext.prod_comet_diff",
    "eng.ext.prod_total_diff",
    "eng.ext.prod_total_share",
    "eng.pending.ships_swing",
    "eng.pending.static_count_swing",
    "eng.pending.orbit_count_swing",
    "eng.pending.prod_total_swing",
    "eng.geo.now_min_dist",
    "eng.geo.ext_min_dist",
    "eng.geo.dist_change",
    "eng.geo.now_mixed_gap",
    "eng.geo.ext_mixed_gap",
    "eng.time.tick_frac",
    "eng.time.remaining_frac",
    "eng.time.early_phase",
    "eng.time.transition_phase",
    "eng.time.midgame_phase",
    "eng.time.endgame_phase",
    "eng.forecast.cur_prod_remaining",
    "eng.forecast.cur_prod_100",
    "eng.forecast.cur_adv_remaining",
    "eng.forecast.cur_adv_100",
    "eng.forecast.ext_prod_remaining",
    "eng.forecast.ext_adv_remaining",
    "eng.time.cur_prod_remaining_frac",
    "eng.time.cur_ships_elapsed_frac",
    "eng.cur.flying_total",
    "eng.cur.my_flying_commitment",
    "eng.cur.opp_flying_commitment",
    "eng.cur.flying_commitment_diff",
    "eng.geo.cur_ship_pressure",
    "eng.geo.cur_prod_pressure",
    "eng.geo.ext_ship_pressure",
    "eng.geo.ext_prod_pressure",
    "eng.forecast.cur_prod_25",
    "eng.forecast.cur_prod_50",
    "eng.forecast.cur_adv_25",
    "eng.forecast.cur_adv_50",
    "eng.forecast.cur_adv_150",
    "eng.forecast.ext_adv_100",
    "eng.margin.cur_adv_100_log",
    "eng.margin.cur_adv_remaining_log",
    "eng.margin.ships_total_diff_log",
    "eng.margin.prod_total_diff_log",
    "eng.speed.my_stationed_speed",
    "eng.speed.opp_stationed_speed",
    "eng.speed.stationed_speed_diff",
    "eng.speed.my_total_speed",
    "eng.speed.opp_total_speed",
    "eng.speed.total_speed_diff",
    "eng.speed.now_travel_turns_my",
    "eng.speed.ext_travel_turns_my",
    "eng.speed.now_travel_turns_opp",
    "eng.comet.ticks_to_next_spawn",
    "eng.comet.next_spawn_frac",
    "eng.comet.spawn_soon_25",
    "eng.comet.spawn_soon_50",
    "eng.map.orbit_rotation_pressure",
    "eng.map.orbit_count_rotation",
    "eng.map.angular_velocity_scaled",
    "eng.phase.transition_adv_50",
    "eng.phase.transition_adv_100",
    "eng.phase.transition_flying_diff",
    "eng.phase.transition_speed_diff",
    "eng.phase.transition_distance",
    "eng.phase.early_prod_share",
    "eng.phase.midgame_ship_share",
    "eng.phase.endgame_ship_share",
]


ENGINEERED_DIM = len(ENGINEERED_NAMES)
BASE_DIM = 58


def _share(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(1.0, a + b)


def _safe_min4(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    stacked = np.stack([a, b, c, d], axis=1)
    return np.min(np.where(stacked > 0.0, stacked, np.inf), axis=1).clip(max=1e6)


def _commitment(flying: np.ndarray, stationed: np.ndarray) -> np.ndarray:
    return flying / np.maximum(1.0, flying + stationed)


def _fleet_speed(ships: np.ndarray) -> np.ndarray:
    s = np.maximum(1.0, ships.astype(np.float32, copy=False))
    speed = 1.0 + 5.0 * np.power(np.log(s) / np.log(1000.0), 1.5)
    return np.clip(speed, 1.0, 6.0)


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _ticks_to_next_comet(tick: np.ndarray) -> np.ndarray:
    spawns = np.array([50.0, 150.0, 250.0, 350.0, 450.0], dtype=np.float32)
    dt = spawns[None, :] - tick[:, None]
    future = np.where(dt >= 0.0, dt, np.inf)
    return np.min(future, axis=1).clip(max=500.0)


def _pos(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def engineered_from_base(base: np.ndarray) -> np.ndarray:
    """Return engineered columns for a [N,58] summary_v2+extras_v4 matrix."""
    if base.ndim != 2 or base.shape[1] != BASE_DIM:
        raise ValueError(f"expected [N,{BASE_DIM}] base features, got {base.shape}")
    x = base.astype(np.float32, copy=False)
    out = np.empty((x.shape[0], ENGINEERED_DIM), dtype=np.float32)

    me_cur_ships = x[:, 0]
    me_cur_flying = x[:, 1]
    me_cur_static = x[:, 2]
    me_cur_orbit = x[:, 3]
    me_cur_comet = x[:, 4]
    me_cur_prod_static = x[:, 5]
    me_cur_prod_orbit = x[:, 6]
    me_cur_prod_comet = x[:, 7]
    me_cur_neutrals_closer = x[:, 8]
    me_cur_enemies_closer = x[:, 9]

    op_cur_ships = x[:, 10]
    op_cur_flying = x[:, 11]
    op_cur_static = x[:, 12]
    op_cur_orbit = x[:, 13]
    op_cur_comet = x[:, 14]
    op_cur_prod_static = x[:, 15]
    op_cur_prod_orbit = x[:, 16]
    op_cur_prod_comet = x[:, 17]
    op_cur_neutrals_closer = x[:, 18]
    op_cur_enemies_closer = x[:, 19]

    me_ext_ships = x[:, 20]
    me_ext_static = x[:, 21]
    me_ext_orbit = x[:, 22]
    me_ext_comet = x[:, 23]
    me_ext_prod_static = x[:, 24]
    me_ext_prod_orbit = x[:, 25]
    me_ext_prod_comet = x[:, 26]

    op_ext_ships = x[:, 29]
    op_ext_static = x[:, 30]
    op_ext_orbit = x[:, 31]
    op_ext_comet = x[:, 32]
    op_ext_prod_static = x[:, 33]
    op_ext_prod_orbit = x[:, 34]
    op_ext_prod_comet = x[:, 35]

    me_cur_prod = me_cur_prod_static + me_cur_prod_orbit + me_cur_prod_comet
    op_cur_prod = op_cur_prod_static + op_cur_prod_orbit + op_cur_prod_comet
    me_ext_prod = me_ext_prod_static + me_ext_prod_orbit + me_ext_prod_comet
    op_ext_prod = op_ext_prod_static + op_ext_prod_orbit + op_ext_prod_comet
    me_cur_planets = me_cur_static + me_cur_orbit + me_cur_comet
    op_cur_planets = op_cur_static + op_cur_orbit + op_cur_comet
    me_ext_planets = me_ext_static + me_ext_orbit + me_ext_comet
    op_ext_planets = op_ext_static + op_ext_orbit + op_ext_comet
    cur_ship_diff = (me_cur_ships + me_cur_flying) - (op_cur_ships + op_cur_flying)
    ext_ship_diff = me_ext_ships - op_ext_ships
    cur_prod_diff = me_cur_prod - op_cur_prod
    ext_prod_diff = me_ext_prod - op_ext_prod

    now_ss, now_so, now_os, now_oo = x[:, 47], x[:, 48], x[:, 49], x[:, 50]
    ext_ss, ext_so, ext_os, ext_oo = x[:, 51], x[:, 52], x[:, 53], x[:, 54]
    now_min = _safe_min4(now_ss, now_so, now_os, now_oo)
    ext_min = _safe_min4(ext_ss, ext_so, ext_os, ext_oo)
    tick_frac = np.clip(x[:, 46] / 500.0, 0.0, 1.0)
    remaining_frac = 1.0 - tick_frac
    remaining_ticks = np.maximum(0.0, 500.0 - x[:, 46])
    horizon_100 = np.minimum(100.0, remaining_ticks)
    cur_ship_share = _share(me_cur_ships + me_cur_flying, op_cur_ships + op_cur_flying)
    cur_prod_share = _share(me_cur_prod, op_cur_prod)
    ext_prod_share = _share(me_ext_prod, op_ext_prod)
    me_flying_commit = _commitment(me_cur_flying, me_cur_ships)
    op_flying_commit = _commitment(op_cur_flying, op_cur_ships)
    early_phase = (tick_frac < 1.0 / 6.0).astype(np.float32)
    transition_phase = ((tick_frac >= 1.0 / 6.0) & (tick_frac < 1.0 / 3.0)).astype(np.float32)
    midgame_phase = ((tick_frac >= 1.0 / 3.0) & (tick_frac < 2.0 / 3.0)).astype(np.float32)
    endgame_phase = (tick_frac >= 2.0 / 3.0).astype(np.float32)
    cur_adv_25 = cur_ship_diff + cur_prod_diff * np.minimum(25.0, remaining_ticks)
    cur_adv_50 = cur_ship_diff + cur_prod_diff * np.minimum(50.0, remaining_ticks)
    cur_adv_100 = cur_ship_diff + cur_prod_diff * horizon_100
    cur_adv_150 = cur_ship_diff + cur_prod_diff * np.minimum(150.0, remaining_ticks)
    cur_adv_remaining = cur_ship_diff + cur_prod_diff * remaining_ticks
    ext_adv_100 = ext_ship_diff + ext_prod_diff * horizon_100
    my_stationed_speed = _fleet_speed(me_cur_ships)
    op_stationed_speed = _fleet_speed(op_cur_ships)
    my_total_speed = _fleet_speed(me_cur_ships + me_cur_flying)
    op_total_speed = _fleet_speed(op_cur_ships + op_cur_flying)
    total_speed_diff = my_total_speed - op_total_speed
    ticks_to_comet = _ticks_to_next_comet(x[:, 46])

    cols = [
        me_cur_ships - op_cur_ships,
        me_cur_flying - op_cur_flying,
        cur_ship_diff,
        cur_ship_share,
        me_cur_static - op_cur_static,
        me_cur_orbit - op_cur_orbit,
        me_cur_comet - op_cur_comet,
        me_cur_planets - op_cur_planets,
        me_cur_prod_static - op_cur_prod_static,
        me_cur_prod_orbit - op_cur_prod_orbit,
        me_cur_prod_comet - op_cur_prod_comet,
        cur_prod_diff,
        cur_prod_share,
        me_ext_ships - op_ext_ships,
        me_ext_static - op_ext_static,
        me_ext_orbit - op_ext_orbit,
        me_ext_comet - op_ext_comet,
        me_ext_planets - op_ext_planets,
        me_ext_prod_static - op_ext_prod_static,
        me_ext_prod_orbit - op_ext_prod_orbit,
        me_ext_prod_comet - op_ext_prod_comet,
        ext_prod_diff,
        ext_prod_share,
        ext_ship_diff - cur_ship_diff,
        (me_ext_static - op_ext_static) - (me_cur_static - op_cur_static),
        (me_ext_orbit - op_ext_orbit) - (me_cur_orbit - op_cur_orbit),
        ext_prod_diff - cur_prod_diff,
        now_min,
        ext_min,
        ext_min - now_min,
        np.minimum(now_so, now_os) - np.minimum(now_ss, now_oo),
        np.minimum(ext_so, ext_os) - np.minimum(ext_ss, ext_oo),
        tick_frac,
        remaining_frac,
        early_phase,
        transition_phase,
        midgame_phase,
        endgame_phase,
        cur_prod_diff * remaining_ticks,
        cur_prod_diff * horizon_100,
        cur_adv_remaining,
        cur_adv_100,
        ext_prod_diff * remaining_ticks,
        ext_ship_diff + ext_prod_diff * remaining_ticks,
        cur_prod_share * remaining_frac,
        cur_ship_share * tick_frac,
        me_cur_flying + op_cur_flying,
        me_flying_commit,
        op_flying_commit,
        me_flying_commit - op_flying_commit,
        cur_ship_diff / (1.0 + now_min),
        cur_prod_diff / (1.0 + now_min),
        ext_ship_diff / (1.0 + ext_min),
        ext_prod_diff / (1.0 + ext_min),
        cur_prod_diff * np.minimum(25.0, remaining_ticks),
        cur_prod_diff * np.minimum(50.0, remaining_ticks),
        cur_adv_25,
        cur_adv_50,
        cur_adv_150,
        ext_adv_100,
        _signed_log1p(cur_adv_100),
        _signed_log1p(cur_adv_remaining),
        _signed_log1p(cur_ship_diff),
        _signed_log1p(cur_prod_diff),
        my_stationed_speed,
        op_stationed_speed,
        my_stationed_speed - op_stationed_speed,
        my_total_speed,
        op_total_speed,
        total_speed_diff,
        now_min / my_stationed_speed,
        ext_min / my_stationed_speed,
        now_min / op_stationed_speed,
        ticks_to_comet,
        ticks_to_comet / 500.0,
        (ticks_to_comet <= 25.0).astype(np.float32),
        (ticks_to_comet <= 50.0).astype(np.float32),
        (me_cur_orbit - op_cur_orbit) * x[:, 57],
        (me_cur_orbit + op_cur_orbit) * x[:, 57],
        x[:, 57] * 100.0,
        cur_adv_50 * transition_phase,
        cur_adv_100 * transition_phase,
        (me_cur_flying - op_cur_flying) * transition_phase,
        total_speed_diff * transition_phase,
        now_min * transition_phase,
        cur_prod_share * early_phase,
        cur_ship_share * midgame_phase,
        cur_ship_share * endgame_phase,
    ]
    for i, col in enumerate(cols):
        out[:, i] = col
    return out


def append_engineered_features(base: np.ndarray) -> np.ndarray:
    """Append engineered features unless they are already present."""
    if base.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got {base.shape}")
    if base.shape[1] == BASE_DIM + ENGINEERED_DIM:
        return base.astype(np.float32, copy=False)
    if base.shape[1] < BASE_DIM:
        raise ValueError(f"expected at least {BASE_DIM} features, got {base.shape[1]}")
    base58 = base[:, :BASE_DIM].astype(np.float32, copy=False)
    return np.concatenate([base58, engineered_from_base(base58)], axis=1).astype(np.float32)
