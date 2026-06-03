"""Spatial / reachability value-function features computed directly from a raw
observation, inspired by the open-source `producer_lite` planner bot.

The shipped summary_v2 features are mostly spatially-blind aggregates. These 13
columns add the planner's actual threat geometry and gave +1.9% held-out
accuracy (94.4% -> 96.3%) over the 157-d base:

  - `cheap_enemy_pressure`-style distance-decayed REACHABLE enemy mass per
    planet (fleet-speed-scaled horizon), aggregated per side,
  - real frontline distance from planet positions,
  - capture-vulnerability counts (garrison vs reachable enemy mass),
  - in-flight fleet threat aimed at each side's planets,
  - center-control advantage.

All features are from the perspective of player `me`. Sign convention: a more
positive value is better for `me` wherever a difference is taken.

These are mirrored exactly in Rust (`value_net::summary_features_spatial`) so the
deployed bot computes the identical vector; parity is checked by
`train/check_spatial_parity.py`.

Planet layout:  [id, owner, x, y, radius, ships, production]
Fleet  layout:  [id, owner, x, y, angle, from_planet_id, ships]
"""

from __future__ import annotations

import numpy as np

HORIZON = 18.0  # producer_lite 2P default
SUN = np.array([50.0, 50.0], dtype=np.float64)

SPATIAL_NAMES = [
    "sp.my_pressure_received",      # reachable enemy mass onto my planets
    "sp.opp_pressure_received",     # reachable my mass onto enemy planets
    "sp.pressure_adv",              # opp_received - my_received (positive good)
    "sp.max_enemy_pressure",        # worst single-planet enemy pressure on me
    "sp.frontline_min_dist",        # closest my-planet/enemy-planet pair
    "sp.frontline_mean_nearest",    # mean over my planets of nearest enemy dist
    "sp.my_vulnerable_count",       # my planets w/ reachable enemy mass > garrison
    "sp.opp_vulnerable_count",
    "sp.vulnerable_adv",            # opp_vuln - my_vuln (positive good)
    "sp.incoming_fleet_threat",     # enemy in-flight ships reachable onto my planets
    "sp.outgoing_fleet_pressure",   # my in-flight ships reachable onto enemy planets
    "sp.threatened_prod_adv",       # opp threatened prod - my threatened prod
    "sp.center_control_adv",        # my center-weighted ships - opp (positive good)
]
SPATIAL_DIM = len(SPATIAL_NAMES)


def _fleet_speed(ships: np.ndarray) -> np.ndarray:
    s = np.maximum(1.0, np.asarray(ships, dtype=np.float64))
    speed = 1.0 + 5.0 * np.power(np.log(s) / np.log(1000.0), 1.5)
    return np.clip(speed, 1.0, 6.0)


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between point sets a [m,2] and b [n,2]."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.maximum(0.0, (diff * diff).sum(axis=2)))


def compute(obs: dict, me: int) -> np.ndarray:
    out = np.zeros(SPATIAL_DIM, dtype=np.float32)
    planets = obs.get("planets") or []
    if not planets:
        return out
    P = np.asarray(planets, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 7:
        return out
    owner = P[:, 1].astype(np.int64)
    xy = P[:, 2:4]
    ships = np.maximum(0.0, P[:, 5])
    prod = P[:, 6]

    me_mask = owner == me
    opp_mask = (owner >= 0) & (owner != me)
    n_me = int(me_mask.sum())
    n_opp = int(opp_mask.sum())
    if n_me == 0 or n_opp == 0:
        return out

    D = _pairwise_dist(xy, xy)  # [P,P]
    speed = _fleet_speed(ships)  # [P]
    reach = np.maximum(1e-6, speed * HORIZON)  # [P] as source
    decay = np.clip(1.0 - D / reach[:, None], 0.0, None)  # [src, tgt]
    contrib = ships[:, None] * decay  # reachable mass from src onto tgt

    enemy_onto = contrib[opp_mask].sum(axis=0)  # [P] enemy mass reaching each tgt
    my_onto = contrib[me_mask].sum(axis=0)      # [P] my mass reaching each tgt

    my_recv = float(enemy_onto[me_mask].sum())
    opp_recv = float(my_onto[opp_mask].sum())
    out[0] = my_recv
    out[1] = opp_recv
    out[2] = opp_recv - my_recv
    out[3] = float(enemy_onto[me_mask].max())

    # Frontline geometry (real positions).
    D_me_opp = D[np.ix_(me_mask, opp_mask)]  # [n_me, n_opp]
    out[4] = float(D_me_opp.min())
    out[5] = float(D_me_opp.min(axis=1).mean())

    # Capture vulnerability: garrison vs reachable enemy mass at the planet.
    my_ships = ships[me_mask]
    opp_ships = ships[opp_mask]
    my_vuln = enemy_onto[me_mask] > my_ships
    opp_vuln = my_onto[opp_mask] > opp_ships
    out[6] = float(my_vuln.sum())
    out[7] = float(opp_vuln.sum())
    out[8] = out[7] - out[6]

    # In-flight fleet threat.
    fleets = obs.get("fleets") or []
    if fleets:
        F = np.asarray(fleets, dtype=np.float64)
        if F.ndim == 2 and F.shape[1] >= 7:
            f_owner = F[:, 1].astype(np.int64)
            f_xy = F[:, 2:4]
            f_ships = np.maximum(0.0, F[:, 6])
            f_speed = _fleet_speed(f_ships)
            f_reach = np.maximum(1e-6, f_speed * HORIZON)
            enemy_f = (f_owner >= 0) & (f_owner != me)
            my_f = f_owner == me
            my_xy = xy[me_mask]
            opp_xy = xy[opp_mask]
            if enemy_f.any():
                d_ef = _pairwise_dist(f_xy[enemy_f], my_xy).min(axis=1)  # nearest my planet
                thr = f_ships[enemy_f] * np.clip(1.0 - d_ef / f_reach[enemy_f], 0.0, None)
                out[9] = float(thr.sum())
            if my_f.any():
                d_mf = _pairwise_dist(f_xy[my_f], opp_xy).min(axis=1)
                pres = f_ships[my_f] * np.clip(1.0 - d_mf / f_reach[my_f], 0.0, None)
                out[10] = float(pres.sum())

    # Threatened production (losing high-prod planets hurts more).
    out[11] = float(prod[opp_mask][opp_vuln].sum() - prod[me_mask][my_vuln].sum())

    # Center control: ships weighted by proximity to the sun.
    dist_center = np.sqrt(((xy - SUN[None, :]) ** 2).sum(axis=1))
    w = 1.0 / (1.0 + dist_center)
    out[12] = float((ships[me_mask] * w[me_mask]).sum() - (ships[opp_mask] * w[opp_mask]).sum())

    return out
