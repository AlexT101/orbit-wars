"""Build the per-(turn, planet) target-prediction dataset from Kaggle replays.

For each 2p game JSON:
  for each step t in 0 .. T-2:
    for each player p in {0, 1}:
      features  = per-planet feature matrix at state(t) from p's perspective
      globals   = turn-level scalars from p's perspective
      labels[p] = 1 iff player p dispatched ≥1 fleet at step t whose ballistic
                  collision target is planet p (computed by diffing fleets
                  between step t and step t+1 and re-running predict_fleet_collision
                  on each new fleet)

Output NPZ:
  planet_feats: f32 (N, N_max, F_planet)
  globals:      f32 (N, F_global)
  mask:         bool (N, N_max)
  labels:       f32 (N, N_max)
  meta:         i32 (N, 4)  -- (game_int_id, step, player, n_planets)
  planet_ids:   i32 (N, N_max)  -- 0-padded; mask says which slots are real

Mirrors kaggle_rebuild_v2.py's pure-Python physics so we don't need the Rust
extractor. Slow (~minutes/day on one core); use --workers for multiprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time
import zipfile
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Engine constants (mirrors kaggle_rebuild_v2.py / src/lib.rs / src/pathing.rs)
# ---------------------------------------------------------------------------

CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
BOARD = 100.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0
MAX_TIME = 100  # max turns to simulate fleet trajectories looking for impact

N_MAX = 50      # planet pad slot count: games start with up to 40 planets and
                # 4 comets spawn at step ~50, pushing peak to ~44.
PAIR_MAX = 20   # max source->target launches recorded per (turn, player).
                # 99th percentile in this dataset is <10; 20 is comfortable.
F_PLANET = 50   # 2026-06-04: dropped 8 features per user review:
                # orbit_radius (=d_center for non-comets), min_eta_from_me,
                # surplus_at_min_eta_src, my_ships_arrivable_25/50,
                # rank_min_eta_from_me, rank_dist_nearest_my, projected_is_neutral.
F_GLOBAL = 33

KNN_K = 8       # k for nearest-neighbor owner counts

F32 = np.float32


# ---------------------------------------------------------------------------
# Replay JSON parsing (lifted verbatim from kaggle_rebuild_v2.py)
# ---------------------------------------------------------------------------


def parse_state(o):
    step = int(o.get("step", 0))
    av = float(o.get("angular_velocity", 0.0))
    comet_ids = set(int(x) for x in (o.get("comet_planet_ids") or []))
    init_pos = {}
    for p in (o.get("initial_planets") or []):
        init_pos[int(p[0])] = (float(p[2]), float(p[3]))
    planets = []
    for p in (o.get("planets") or []):
        pid = int(p[0])
        owner = int(p[1])
        x = float(p[2]); y = float(p[3]); radius = float(p[4])
        ships = int(p[5]); prod = int(p[6])
        is_comet = pid in comet_ids
        ix, iy = init_pos.get(pid, (x, y))
        dx = ix - CENTER[0]; dy = iy - CENTER[1]
        orb_r = math.sqrt(dx * dx + dy * dy)
        init_angle = math.atan2(dy, dx)
        is_orbiting = (not is_comet) and (orb_r + radius < ROT_LIMIT)
        planets.append(dict(
            id=pid, owner=owner, x=x, y=y, radius=radius, ships=ships, prod=prod,
            orb_r=orb_r, init_angle=init_angle, is_orbiting=is_orbiting, is_comet=is_comet,
        ))
    fleets = []
    for f in (o.get("fleets") or []):
        fleets.append(dict(
            id=int(f[0]), owner=int(f[1]), x=float(f[2]), y=float(f[3]),
            angle=float(f[4]), ships=int(f[6]),
        ))
    comets = []
    for g in (o.get("comets") or []):
        pids = [int(x) for x in g["planet_ids"]]
        paths = [[(float(pt[0]), float(pt[1])) for pt in path] for path in g["paths"]]
        comets.append(dict(planet_ids=pids, paths=paths, path_index=int(g["path_index"])))
    return dict(step=step, av=av, planets=planets, fleets=fleets, comets=comets)


def comet_group_for(state, cid):
    for g in state["comets"]:
        if cid in g["planet_ids"]:
            return g, g["planet_ids"].index(cid)
    return None


def comet_remaining(state, planet):
    """Turns left in the comet's predetermined path; 0 for non-comets."""
    if not planet["is_comet"]:
        return 0
    gi = comet_group_for(state, planet["id"])
    if gi is None:
        return 0
    g, i = gi
    return max(len(g["paths"][i]) - g["path_index"], 0)


def planet_pos_at(state, planet, dt):
    if planet["is_comet"]:
        gi = comet_group_for(state, planet["id"])
        if gi is None:
            return None
        g, i = gi
        idx = g["path_index"] + dt
        if idx < 0 or idx >= len(g["paths"][i]):
            return None
        return g["paths"][i][idx]
    if planet["is_orbiting"]:
        abs_step = max(state["step"] + dt - 1, 0)
        a = planet["init_angle"] + state["av"] * abs_step
        return (CENTER[0] + planet["orb_r"] * math.cos(a),
                CENTER[1] + planet["orb_r"] * math.sin(a))
    return (planet["x"], planet["y"])


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(max(s, 1.0), MAX_SPEED)


def on_board(p):
    return 0.0 <= p[0] <= BOARD and 0.0 <= p[1] <= BOARD


def pt_seg_dist(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 < 1e-12:
        return math.dist(p, v)
    t = max(0.0, min(1.0, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    proj = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return math.dist(p, proj)


def swept_pair_hit(a, b, p0, p1, r):
    d0x = a[0] - p0[0]; d0y = a[1] - p0[1]
    dvx = (b[0] - a[0]) - (p1[0] - p0[0])
    dvy = (b[1] - a[1]) - (p1[1] - p0[1])
    aq = dvx * dvx + dvy * dvy
    bq = 2.0 * (d0x * dvx + d0y * dvy)
    cq = d0x * d0x + d0y * d0y - r * r
    if aq < 1e-12:
        return cq <= 0.0
    disc = bq * bq - 4.0 * aq * cq
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-bq - sq) / (2.0 * aq)
    t2 = (-bq + sq) / (2.0 * aq)
    return t2 >= 0.0 and t1 <= 1.0


def predict_fleet_collision_slow(state, fleet):
    """Pure-Python predictor (kept for reference / fallback)."""
    speed = fleet_speed(fleet["ships"])
    dx = speed * math.cos(fleet["angle"])
    dy = speed * math.sin(fleet["angle"])
    pos = (fleet["x"], fleet["y"])
    for dt in range(1, MAX_TIME + 1):
        new_pos = (pos[0] + dx, pos[1] + dy)
        for planet in state["planets"]:
            p_old = planet_pos_at(state, planet, dt - 1)
            if p_old is None:
                continue
            p_new = planet_pos_at(state, planet, dt)
            if p_new is None:
                continue
            if not on_board(p_old) and not on_board(p_new):
                continue
            if swept_pair_hit(pos, new_pos, p_old, p_new, planet["radius"]):
                return (planet["id"], dt)
        if not on_board(new_pos):
            return None
        if pt_seg_dist(CENTER, pos, new_pos) < SUN_RADIUS:
            return None
        pos = new_pos
    return None


# Lazy fast path: numba-jit'd collision sweep with per-state geometry cache.
# alphaduck/fastsim.py owns the JIT body; here we wrap with a state-level cache
# so each per-row flatten only happens once even if predict is called many times.
_FASTSIM = None


def _get_fastsim():
    global _FASTSIM
    if _FASTSIM is None:
        from pathlib import Path as _P
        _here = _P(__file__).resolve().parents[2] / "alphaduck"
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        import fastsim as _fs
        _FASTSIM = _fs
        _fs.warmup()
    return _FASTSIM


def predict_fleet_collision(state, fleet):
    """Numba-accelerated predictor. Lazily flattens planet geometry per state."""
    try:
        fs = _get_fastsim()
    except Exception:
        return predict_fleet_collision_slow(state, fleet)
    flat = state.get("_flat_cache")
    if flat is None:
        flat = fs.flatten_state(state)
        state["_flat_cache"] = flat
    speed = max(fleet_speed(fleet["ships"]), 1.0)
    pid, eta = fs.predict_one_fleet_fast(flat, fleet["x"], fleet["y"], fleet["angle"], speed)
    if pid is None:
        return None
    return (pid, eta)


# ---------------------------------------------------------------------------
# Feature extraction (per-planet + global)
# ---------------------------------------------------------------------------


PLANET_FEAT_NAMES = [
    # owner one-hot (3)
    "is_mine", "is_neutral", "is_enemy",
    # stock (3)
    "ships", "ships_log1p", "production",
    # orbit/geom (5)
    "cos_theta", "sin_theta", "planet_radius",   # orbit_radius dropped (= d_center for non-comets); omega moved to global
    # position (6)
    "x_now", "y_now", "x_p10", "y_p10", "x_p25", "y_p25",
    # inbound my fleets bucketed by arrival window (5/10/any × count+ships) → 6
    "in_mine_count_5", "in_mine_ships_5",
    "in_mine_count_10", "in_mine_ships_10",
    "in_mine_count_any", "in_mine_ships_any",
    "in_mine_min_eta", "in_mine_mean_eta",
    # inbound enemy fleets bucketed by arrival window → 6 + 2 stats
    "in_enemy_count_5", "in_enemy_ships_5",
    "in_enemy_count_10", "in_enemy_ships_10",
    "in_enemy_count_any", "in_enemy_ships_any",
    "in_enemy_min_eta", "in_enemy_mean_eta",
    # frontier/geom (4)  -- reachability features dropped (derivable from pair feats)
    "dist_nearest_my", "dist_nearest_enemy", "is_closest_neutral_to_me", "is_closest_enemy_to_me",
    # k-NN owner counts at k=8 (3): replaces KDE
    "knn_my", "knn_enemy", "knn_neutral",
    # listwise rank (2)  -- rank_min_eta/rank_dist dropped (derivable)
    "rank_production", "rank_ships",
    # tempo (1)
    "turns_since_owner_change",
    # comet identity (1)  — comet_remaining moved to global
    "is_comet",
    # opening-bot-inspired geometry (3)  — removed kde_unit/kde_prod, removed is_orbiting (we have orb_r)
    "is_orbiting", "dist_to_edge", "d_center", "turns_to_cap_proxy",
    # ships per production (1)
    "ships_per_prod",
    # projected ownership + garrison after all current fleets land (3) -- projected_is_neutral dropped
    "projected_is_mine", "projected_is_enemy", "projected_ships_log1p",
]
assert len(PLANET_FEAT_NAMES) == F_PLANET, f"len={len(PLANET_FEAT_NAMES)} vs F_PLANET={F_PLANET}"


_TYPE_BREAKDOWN_NAMES = [
    f"{metric}_{owner}_{kind}"
    for owner in ("my", "enemy", "neutral")
    for kind in ("stat", "orb", "comet")
    for metric in ("count", "prod")
]
assert len(_TYPE_BREAKDOWN_NAMES) == 18

GLOBAL_FEAT_NAMES = [
    "turn_num", "turn_norm",
    "my_ships_total", "enemy_ships_total",
    "my_prod_total", "enemy_prod_total",
    "my_planet_count", "enemy_planet_count",
    "economy_diff",
    "phase_opening", "phase_mid", "phase_end",
    "omega",                  # moved from per-planet
    "comet_remaining_max",    # max remaining across all live comets (0 if none)
    "time_to_next_comet",     # spawns at 50, 150, 250, … (every 100 turns from 50)
] + _TYPE_BREAKDOWN_NAMES
assert len(GLOBAL_FEAT_NAMES) == F_GLOBAL, f"len={len(GLOBAL_FEAT_NAMES)} vs F_GLOBAL={F_GLOBAL}"


def _angle_of(p):
    """Current orbit angle θ for a planet."""
    if p["orb_r"] < 1e-6:
        return 0.0
    dx = p["x"] - CENTER[0]
    dy = p["y"] - CENTER[1]
    return math.atan2(dy, dx)


def _planet_pos_at_dt(state, p, dt):
    pos = planet_pos_at(state, p, dt)
    if pos is None:
        # comet has expired its path; freeze at last known pos
        pos = (p["x"], p["y"])
    return pos


def _ranks_01(values):
    """Return rank ∈ [0,1] where 1.0 = largest. Ties get average rank."""
    n = len(values)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    for rank, i in enumerate(order):
        out[i] = rank / (n - 1)
    return out


def _time_to_next_comet(step: int) -> int:
    """Comets spawn at 50, 150, 250, … (every 100 turns starting at step 50)."""
    if step < 50:
        return 50 - step
    return 100 - ((step - 50) % 100)


def _project_to_arrivals_for_features(state, pid_to_idx):
    """Event-driven projection mirroring alphaduck's _project_to_arrivals.
    Returns per-planet (final_owner: int, final_ships: float) arrays."""
    planets = state["planets"]
    n = len(planets)
    cur_owner = [p["owner"] for p in planets]
    cur_ships = [float(p["ships"]) for p in planets]
    prod = [p["prod"] for p in planets]
    is_comet_arr = [p["is_comet"] for p in planets]

    events = []
    for f in state["fleets"]:
        pred = predict_fleet_collision(state, f)
        if pred is None:
            continue
        events.append((pred[1], pred[0], f["owner"], f["ships"]))
    events.sort(key=lambda e: e[0])

    last_t = 0
    for eta, dest_pid, f_owner, f_ships in events:
        elapsed = eta - last_t
        if elapsed > 0:
            for i in range(n):
                if cur_owner[i] >= 0 and not is_comet_arr[i]:
                    cur_ships[i] += prod[i] * elapsed
        last_t = eta
        idx = pid_to_idx.get(dest_pid)
        if idx is None:
            continue
        if cur_owner[idx] == f_owner:
            cur_ships[idx] += f_ships
        else:
            if f_ships > cur_ships[idx]:
                cur_owner[idx] = f_owner
                cur_ships[idx] = f_ships - cur_ships[idx]
            else:
                cur_ships[idx] -= f_ships
    return cur_owner, cur_ships


def extract_per_player(state, player, owner_change_turn):
    """Return (planet_feats[N,F_planet], globals[F_global], planet_ids[N]).

    `owner_change_turn[pid]` = last turn the planet's owner changed (or -1).
    """
    planets = state["planets"]
    n = len(planets)
    cur_turn = state["step"]

    # ----- bulk per-planet primitives -----
    is_mine = [int(p["owner"] == player) for p in planets]
    is_neut = [int(p["owner"] == -1) for p in planets]
    is_enem = [int(p["owner"] != player and p["owner"] != -1) for p in planets]

    # pre-compute positions at +10, +25
    pos_now = [(p["x"], p["y"]) for p in planets]
    pos_p10 = [_planet_pos_at_dt(state, p, 10) for p in planets]
    pos_p25 = [_planet_pos_at_dt(state, p, 25) for p in planets]

    # ----- inbound fleet aggregation by destination + ETA bucket -----
    # buckets: 5 (arrive in ≤5), 10 (≤10), any (≤MAX_TIME).
    in_mine_count_5 = [0] * n; in_mine_ships_5 = [0] * n
    in_mine_count_10 = [0] * n; in_mine_ships_10 = [0] * n
    in_mine_count_any = [0] * n; in_mine_ships_any = [0] * n
    in_mine_etas: list[list[float]] = [[] for _ in range(n)]
    in_enem_count_5 = [0] * n; in_enem_ships_5 = [0] * n
    in_enem_count_10 = [0] * n; in_enem_ships_10 = [0] * n
    in_enem_count_any = [0] * n; in_enem_ships_any = [0] * n
    in_enem_etas: list[list[float]] = [[] for _ in range(n)]
    pid_to_idx = {p["id"]: i for i, p in enumerate(planets)}

    for f in state["fleets"]:
        pred = predict_fleet_collision(state, f)
        if pred is None:
            continue
        dest_pid, eta = pred
        idx = pid_to_idx.get(dest_pid)
        if idx is None:
            continue
        if f["owner"] == player:
            in_mine_count_any[idx] += 1
            in_mine_ships_any[idx] += f["ships"]
            in_mine_etas[idx].append(eta)
            if eta <= 5:
                in_mine_count_5[idx] += 1; in_mine_ships_5[idx] += f["ships"]
            if eta <= 10:
                in_mine_count_10[idx] += 1; in_mine_ships_10[idx] += f["ships"]
        else:
            in_enem_count_any[idx] += 1
            in_enem_ships_any[idx] += f["ships"]
            in_enem_etas[idx].append(eta)
            if eta <= 5:
                in_enem_count_5[idx] += 1; in_enem_ships_5[idx] += f["ships"]
            if eta <= 10:
                in_enem_count_10[idx] += 1; in_enem_ships_10[idx] += f["ships"]

    # reachability features were removed (min_eta_from_me, surplus, arrivable_25/50,
    # rank_min_eta, rank_dist) — derivable from pair features in-model.

    # ----- frontier/geom -----
    dist_nearest_my = [math.inf] * n
    dist_nearest_enemy = [math.inf] * n
    for i, p in enumerate(planets):
        for q in planets:
            if q["id"] == p["id"]:
                continue
            d = math.hypot(p["x"] - q["x"], p["y"] - q["y"])
            if q["owner"] == player and d < dist_nearest_my[i]:
                dist_nearest_my[i] = d
            elif q["owner"] != player and q["owner"] != -1 and d < dist_nearest_enemy[i]:
                dist_nearest_enemy[i] = d
        if math.isinf(dist_nearest_my[i]):
            dist_nearest_my[i] = 200.0
        if math.isinf(dist_nearest_enemy[i]):
            dist_nearest_enemy[i] = 200.0

    # is_closest_neutral_to_me / is_closest_enemy_to_me: per planet, is its
    # closest "my planet" closer than its closest "other-owned" planet?
    is_closest_neutral = [0] * n
    is_closest_enemy = [0] * n
    for i, p in enumerate(planets):
        if p["owner"] == -1:
            is_closest_neutral[i] = int(dist_nearest_my[i] < dist_nearest_enemy[i])
        elif p["owner"] != player:
            is_closest_enemy[i] = int(dist_nearest_my[i] < dist_nearest_enemy[i])

    # ----- k-NN owner counts (replaces KDE) -----
    # For each planet, look at its KNN_K nearest real planets and count owners.
    knn_my = [0] * n
    knn_enemy = [0] * n
    knn_neutral = [0] * n
    if n > 1:
        for i, p in enumerate(planets):
            dists = []
            for j, q in enumerate(planets):
                if j == i: continue
                dists.append((math.hypot(p["x"] - q["x"], p["y"] - q["y"]), j))
            dists.sort(key=lambda t: t[0])
            k_eff = min(KNN_K, len(dists))
            for _d, j in dists[:k_eff]:
                o = planets[j]["owner"]
                if o == player: knn_my[i] += 1
                elif o == -1: knn_neutral[i] += 1
                else: knn_enemy[i] += 1

    # ----- projected ownership & ship count after all in-flight fleets land -----
    proj_owner, proj_ships = _project_to_arrivals_for_features(state, pid_to_idx)

    # turns_to_cap_proxy: how long would my total production take to grow ships+1?
    my_total_prod = sum(p["prod"] for p in planets if p["owner"] == player)
    inv_prod = 1.0 / max(my_total_prod, 1)

    # ----- listwise rank within turn (1.0 = largest) -----
    rank_prod = _ranks_01([p["prod"] for p in planets])
    rank_ships = _ranks_01([p["ships"] for p in planets])
    # rank_eta and rank_dist dropped per user review (derivable)

    # ----- pack per-planet features -----
    feats = np.zeros((n, F_PLANET), dtype=F32)
    for i, p in enumerate(planets):
        theta = _angle_of(p)
        ships = p["ships"]
        in_min_mine = min(in_mine_etas[i]) if in_mine_etas[i] else 100.0
        in_mean_mine = (sum(in_mine_etas[i]) / len(in_mine_etas[i])) if in_mine_etas[i] else 100.0
        in_min_enem = min(in_enem_etas[i]) if in_enem_etas[i] else 100.0
        in_mean_enem = (sum(in_enem_etas[i]) / len(in_enem_etas[i])) if in_enem_etas[i] else 100.0
        oct = owner_change_turn.get(p["id"], -1)
        tsoc = (cur_turn - oct) if oct >= 0 else 200
        po_mine = int(proj_owner[i] == player)
        po_enem = int(proj_owner[i] >= 0 and proj_owner[i] != player)
        ships_per_prod = (ships + 1) / max(p["prod"], 1)
        feats[i] = [
            is_mine[i], is_neut[i], is_enem[i],
            ships, math.log1p(max(ships, 0)), p["prod"],
            math.cos(theta), math.sin(theta), p["radius"],   # orbit_radius + omega dropped
            pos_now[i][0], pos_now[i][1], pos_p10[i][0], pos_p10[i][1], pos_p25[i][0], pos_p25[i][1],
            in_mine_count_5[i], in_mine_ships_5[i],
            in_mine_count_10[i], in_mine_ships_10[i],
            in_mine_count_any[i], in_mine_ships_any[i],
            in_min_mine, in_mean_mine,
            in_enem_count_5[i], in_enem_ships_5[i],
            in_enem_count_10[i], in_enem_ships_10[i],
            in_enem_count_any[i], in_enem_ships_any[i],
            in_min_enem, in_mean_enem,
            dist_nearest_my[i], dist_nearest_enemy[i], is_closest_neutral[i], is_closest_enemy[i],
            knn_my[i], knn_enemy[i], knn_neutral[i],
            rank_prod[i], rank_ships[i],
            min(tsoc, 200),
            int(p["is_comet"]),
            int(p["is_orbiting"]),
            min(p["x"], p["y"], BOARD - p["x"], BOARD - p["y"]),
            math.hypot(p["x"] - CENTER[0], p["y"] - CENTER[1]),
            (ships + 1) * inv_prod,
            ships_per_prod,
            po_mine, po_enem, math.log1p(max(proj_ships[i], 0.0)),
        ]

    # ----- globals -----
    my_ships = sum(p["ships"] for p in planets if p["owner"] == player) + \
               sum(f["ships"] for f in state["fleets"] if f["owner"] == player)
    enemy_ships = sum(p["ships"] for p in planets if p["owner"] != player and p["owner"] != -1) + \
                  sum(f["ships"] for f in state["fleets"] if f["owner"] != player)
    my_prod = sum(p["prod"] for p in planets if p["owner"] == player)
    enemy_prod = sum(p["prod"] for p in planets if p["owner"] != player and p["owner"] != -1)
    my_pc = sum(1 for p in planets if p["owner"] == player)
    enemy_pc = sum(1 for p in planets if p["owner"] != player and p["owner"] != -1)
    econ_diff = (my_prod - enemy_prod) / max(my_prod + enemy_prod, 1)
    phase_open = 1.0 if cur_turn < 30 else 0.0
    phase_mid = 1.0 if 30 <= cur_turn < 120 else 0.0
    phase_end = 1.0 if cur_turn >= 120 else 0.0
    # comet_remaining_max: max over live comets (0 if none)
    comet_remaining_max = 0
    for p in planets:
        if p["is_comet"]:
            comet_remaining_max = max(comet_remaining_max, comet_remaining(state, p))
    # Type-breakdown globals: {count, prod} × {my, enemy, neutral} × {stat, orb, comet}
    # Order MUST match _TYPE_BREAKDOWN_NAMES at module top.
    breakdown = []
    for owner_kind in ("my", "enemy", "neutral"):
        for kind in ("stat", "orb", "comet"):
            count = 0; prod_sum = 0
            for pl in planets:
                if owner_kind == "my" and pl["owner"] != player: continue
                if owner_kind == "enemy" and (pl["owner"] == player or pl["owner"] == -1): continue
                if owner_kind == "neutral" and pl["owner"] != -1: continue
                if kind == "stat" and (pl["is_comet"] or pl["is_orbiting"]): continue
                if kind == "orb" and not pl["is_orbiting"]: continue
                if kind == "comet" and not pl["is_comet"]: continue
                count += 1; prod_sum += pl["prod"]
            breakdown.append(count); breakdown.append(prod_sum)
    globals_ = np.array([
        cur_turn, cur_turn / 200.0,
        my_ships, enemy_ships, my_prod, enemy_prod, my_pc, enemy_pc,
        econ_diff, phase_open, phase_mid, phase_end,
        state["av"],                # omega moved from per-planet
        comet_remaining_max,
        _time_to_next_comet(cur_turn),
    ] + breakdown, dtype=F32)

    planet_ids = np.array([p["id"] for p in planets], dtype=np.int32)
    return feats, globals_, planet_ids


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


def labels_for_step(state_t, state_tp1, player, planet_ids):
    """1 iff player dispatched ≥1 fleet at step t whose ballistic-predicted
    destination matches a planet in `planet_ids`.

    A "dispatched fleet" = a fleet present in state_tp1 with an id not in state_t.
    """
    old_ids = {f["id"] for f in state_t["fleets"]}
    new_fleets = [f for f in state_tp1["fleets"] if f["id"] not in old_ids and f["owner"] == player]
    pid_to_idx = {pid: i for i, pid in enumerate(planet_ids)}
    labels = np.zeros(len(planet_ids), dtype=F32)
    for f in new_fleets:
        pred = predict_fleet_collision(state_tp1, f)
        if pred is None:
            continue
        dest_pid, _eta = pred
        idx = pid_to_idx.get(dest_pid)
        if idx is None:
            continue
        labels[idx] = 1.0
    return labels


def all_labels_for_step(state_t, state_tp1, player, planet_ids,
                         future_states, t_now, future_fleet_dests,
                         player_action_at_tp1):
    """Compute four parallel per-planet label vectors for one (turn, player):
      labels:           1 iff player dispatched ≥1 fleet at this decision
                        whose collision-target is this planet (any launch).
      offensive:        labels minus support launches.
      support_target:   1 iff the planet is the *destination* of a support launch.
      support_source:   1 iff the planet is the *source* of a support launch.

    A *support* launch is one where:
      (a) the target planet is currently owned by `player` (per state_t), AND
      (b) no enemy fleet targets the same planet during the window [t, t+eta]
          (either already in flight in state(t+k), or arriving in that window).

    `player_action_at_tp1` = the recorded action list for `player` at step t+1
    (each entry = [src_pid, angle, ships]). Used to match new fleets to source
    planets (matched by (angle, ships) which is unique per launch).
    """
    old_ids = {f["id"] for f in state_t["fleets"]}
    new_fleets = [f for f in state_tp1["fleets"] if f["id"] not in old_ids and f["owner"] == player]
    pid_to_idx = {pid: i for i, pid in enumerate(planet_ids)}
    pid_to_owner = {p["id"]: p["owner"] for p in state_t["planets"]}

    # action_lookup: (rounded_angle, ships) -> src_pid
    action_lookup = {}
    for act in (player_action_at_tp1 or []):
        try:
            src_pid = int(act[0])
            angle = float(act[1])
            ships = int(act[2])
        except Exception:
            continue
        action_lookup[(round(angle, 6), ships)] = src_pid

    labels = np.zeros(len(planet_ids), dtype=F32)
    offensive = np.zeros(len(planet_ids), dtype=F32)
    offensive_source = np.zeros(len(planet_ids), dtype=F32)
    support_target = np.zeros(len(planet_ids), dtype=F32)
    support_source = np.zeros(len(planet_ids), dtype=F32)
    # source->target launches as parallel index lists (truncated to PAIR_MAX)
    pair_src_idx: list[int] = []
    pair_tgt_idx: list[int] = []

    n_future = len(future_states)
    for f in new_fleets:
        pred = predict_fleet_collision(state_tp1, f)
        if pred is None:
            continue
        dest_pid, eta = pred
        idx = pid_to_idx.get(dest_pid)
        if idx is None:
            continue
        labels[idx] = 1.0

        # source planet from action (may be None if matching fails)
        src_pid = action_lookup.get((round(float(f["angle"]), 6), int(f["ships"])))
        src_idx = pid_to_idx.get(src_pid) if src_pid is not None else None

        # record the (source, target) pair
        if src_idx is not None:
            pair_src_idx.append(src_idx)
            pair_tgt_idx.append(idx)

        target_owner = pid_to_owner.get(dest_pid, -1)
        if target_owner != player:
            offensive[idx] = 1.0
            if src_idx is not None:
                offensive_source[src_idx] = 1.0
            continue

        enemy_threatens = False
        for k in range(eta + 1):
            j = t_now + k
            if j >= n_future or future_states[j] is None:
                continue
            dests_k = future_fleet_dests[j] if j < len(future_fleet_dests) else None
            if not dests_k:
                continue
            for fid, dest in dests_k.items():
                if dest != dest_pid:
                    continue
                state_k = future_states[j]
                for fl in state_k["fleets"]:
                    if fl["id"] == fid and fl["owner"] != player:
                        enemy_threatens = True
                        break
                if enemy_threatens:
                    break
            if enemy_threatens:
                break

        if enemy_threatens:
            offensive[idx] = 1.0
            if src_idx is not None:
                offensive_source[src_idx] = 1.0
        else:
            # genuine support launch
            support_target[idx] = 1.0
            if src_idx is not None:
                support_source[src_idx] = 1.0
    return labels, offensive, offensive_source, support_target, support_source, pair_src_idx, pair_tgt_idx


# ---------------------------------------------------------------------------
# Owner-change history tracking
# ---------------------------------------------------------------------------


def update_owner_history(state, last_owner, owner_change_turn, cur_turn):
    """Track only player↔player ownership transitions (neutral-involving
    transitions like capture-from-neutral do not reset the timer)."""
    for p in state["planets"]:
        pid = p["id"]
        if pid not in last_owner:
            last_owner[pid] = p["owner"]
            owner_change_turn[pid] = cur_turn
        elif last_owner[pid] != p["owner"]:
            # Only count when both sides of the transition are players (not -1=neutral).
            if last_owner[pid] != -1 and p["owner"] != -1:
                owner_change_turn[pid] = cur_turn
            last_owner[pid] = p["owner"]


# ---------------------------------------------------------------------------
# Game-level driver
# ---------------------------------------------------------------------------


def process_game(game_id_int, game_json):
    """Return list of (planet_feats, globals, mask, labels, planet_ids, meta) rows."""
    rewards = game_json.get("rewards") or []
    if len(rewards) != 2:
        return []
    # value label per player POV: +1 win, -1 loss, 0 draw or unknown.
    # Some replays have None rewards; treat as 0.
    r0 = float(rewards[0]) if rewards[0] is not None else 0.0
    r1 = float(rewards[1]) if rewards[1] is not None else 0.0
    value_per_player = {0: float(np.sign(r0 - r1)), 1: float(np.sign(r1 - r0))}
    steps = game_json.get("steps") or []
    if len(steps) < 2:
        return []

    rows = []
    last_owner: dict[int, int] = {}
    owner_change_turn: dict[int, int] = {}

    # Pre-parse all observations once (each step has the same obs for both players
    # except the `player` index, which we ignore here).
    parsed: list[dict | None] = [None] * len(steps)
    actions_present = False
    for t, step in enumerate(steps):
        if not step:
            continue
        obs = step[0].get("observation")
        if obs is None:
            continue
        parsed[t] = parse_state(obs)
        for ps in step:
            if ps.get("action"):
                actions_present = True
    if not actions_present:
        return []

    # Pre-compute fleet destinations per turn (for support-launch detection).
    # fleet_dests[t][fleet_id] = dest_pid. Skipped when no fleets.
    fleet_dests: list[dict | None] = [None] * len(steps)
    for t, st in enumerate(parsed):
        if st is None or not st["fleets"]:
            fleet_dests[t] = {}
            continue
        dests = {}
        for f in st["fleets"]:
            pred = predict_fleet_collision(st, f)
            if pred is not None:
                dests[f["id"]] = pred[0]
        fleet_dests[t] = dests

    for t in range(len(steps) - 1):
        state_t = parsed[t]
        state_tp1 = parsed[t + 1]
        if state_t is None or state_tp1 is None:
            continue

        update_owner_history(state_t, last_owner, owner_change_turn, t)

        for player in (0, 1):
            try:
                feats, globals_, planet_ids = extract_per_player(state_t, player, owner_change_turn)
            except Exception:
                continue
            n_real = feats.shape[0]
            if n_real == 0 or n_real > N_MAX:
                continue
            # recover this player's action recorded at step t+1 (drove state_tp1)
            player_action_at_tp1 = None
            if t + 1 < len(steps) and steps[t + 1] and len(steps[t + 1]) > player:
                player_action_at_tp1 = steps[t + 1][player].get("action")
            labels, offensive, off_s, sup_t, sup_s, pair_s, pair_t = all_labels_for_step(
                state_t, state_tp1, player, planet_ids,
                future_states=parsed, t_now=t, future_fleet_dests=fleet_dests,
                player_action_at_tp1=player_action_at_tp1,
            )

            # pad to N_MAX
            pad_feats = np.zeros((N_MAX, F_PLANET), dtype=F32)
            pad_feats[:n_real] = feats
            pad_labels = np.zeros(N_MAX, dtype=F32)
            pad_labels[:n_real] = labels
            pad_offensive = np.zeros(N_MAX, dtype=F32)
            pad_offensive[:n_real] = offensive
            pad_off_source = np.zeros(N_MAX, dtype=F32)
            pad_off_source[:n_real] = off_s
            pad_sup_target = np.zeros(N_MAX, dtype=F32)
            pad_sup_target[:n_real] = sup_t
            pad_sup_source = np.zeros(N_MAX, dtype=F32)
            pad_sup_source[:n_real] = sup_s
            pad_ids = np.zeros(N_MAX, dtype=np.int32)
            pad_ids[:n_real] = planet_ids
            mask = np.zeros(N_MAX, dtype=bool)
            mask[:n_real] = True

            # sparse pair labels: source/target indices into the planet slots,
            # padded with -1 to PAIR_MAX. Truncate overflow (rare).
            n_pairs = min(len(pair_s), PAIR_MAX)
            pad_pair_src = np.full(PAIR_MAX, -1, dtype=np.int16)
            pad_pair_tgt = np.full(PAIR_MAX, -1, dtype=np.int16)
            if n_pairs:
                pad_pair_src[:n_pairs] = pair_s[:n_pairs]
                pad_pair_tgt[:n_pairs] = pair_t[:n_pairs]

            # raw pair-feature inputs: positions at t/+1/+2/+5/+10/+20/+30 (7 horizons)
            # so the model can compute pair distance/ETA for both the near-term
            # decision window (n=src.ships at h=0/1/2) and the longer geometric
            # reach (n=20 at h=5/10/20/30).
            raw_xy = np.zeros((N_MAX, 7, 2), dtype=F32)
            raw_ships = np.zeros(N_MAX, dtype=F32)
            raw_prod = np.zeros(N_MAX, dtype=F32)
            for i, p in enumerate(state_t["planets"]):
                for j, h in enumerate((0, 1, 2, 5, 10, 20, 30)):
                    pos = planet_pos_at(state_t, p, h)
                    raw_xy[i, j] = pos if pos is not None else (p["x"], p["y"])
                raw_ships[i] = p["ships"]
                raw_prod[i] = p["prod"]

            # noop label per source: 1 iff this planet is mine AND we made no
            # launch from it this turn (loss masks to my-planets only).
            pad_noop = np.zeros(N_MAX, dtype=F32)
            for i, p in enumerate(state_t["planets"]):
                if p["owner"] == player and off_s[i] == 0 and sup_s[i] == 0:
                    pad_noop[i] = 1.0

            meta = np.array([game_id_int, t, player, n_real], dtype=np.int32)
            value_label = np.float32(value_per_player[player])
            rows.append((pad_feats, globals_, mask, pad_labels, pad_offensive,
                         pad_off_source, pad_sup_target, pad_sup_source, pad_ids, meta,
                         pad_pair_src, pad_pair_tgt, value_label,
                         raw_xy, raw_ships, raw_prod, pad_noop))
    return rows


# ---------------------------------------------------------------------------
# Zip iteration / multiprocessing
# ---------------------------------------------------------------------------


def iter_game_jsons(zip_paths, max_games=None):
    n = 0
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith(".json"):
                    continue
                with zf.open(name) as f:
                    try:
                        g = json.load(f)
                    except json.JSONDecodeError:
                        continue
                # game id from filename (Kaggle episode id)
                base = pathlib.Path(name).stem
                try:
                    gid = int(base)
                except ValueError:
                    gid = abs(hash(base)) % (2 ** 31 - 1)
                yield gid, g
                n += 1
                if max_games is not None and n >= max_games:
                    return


def _worker(args):
    gid, game_json = args
    try:
        return process_game(gid, game_json)
    except Exception:
        return []


def build(zip_paths, out_path, max_games=None, workers=1, progress_every=200):
    t0 = time.time()
    rows_buf = []
    n_games_kept = 0
    n_games_total = 0

    def _flush_print():
        elapsed = time.time() - t0
        gps = n_games_total / max(elapsed, 1e-6)
        print(f"  [{elapsed:.1f}s] games={n_games_total} kept={n_games_kept} "
              f"rows={len(rows_buf)} ({gps:.1f} g/s)", flush=True)

    if workers <= 1:
        for gid, gj in iter_game_jsons(zip_paths, max_games):
            n_games_total += 1
            rs = process_game(gid, gj)
            if rs:
                n_games_kept += 1
                rows_buf.extend(rs)
            if n_games_total % progress_every == 0:
                _flush_print()
    else:
        # use imap_unordered, chunk by yielding (gid, json) pairs
        with mp.Pool(workers) as pool:
            for rs in pool.imap_unordered(_worker, iter_game_jsons(zip_paths, max_games), chunksize=4):
                n_games_total += 1
                if rs:
                    n_games_kept += 1
                    rows_buf.extend(rs)
                if n_games_total % progress_every == 0:
                    _flush_print()
    _flush_print()

    if not rows_buf:
        print("no rows extracted; aborting", file=sys.stderr)
        return 1

    print(f"packing {len(rows_buf)} rows into NPZ ...", flush=True)
    planet_feats = np.stack([r[0] for r in rows_buf])
    globals_ = np.stack([r[1] for r in rows_buf])
    mask = np.stack([r[2] for r in rows_buf])
    labels = np.stack([r[3] for r in rows_buf])
    offensive_labels = np.stack([r[4] for r in rows_buf])
    offensive_source_labels = np.stack([r[5] for r in rows_buf])
    support_target_labels = np.stack([r[6] for r in rows_buf])
    support_source_labels = np.stack([r[7] for r in rows_buf])
    planet_ids = np.stack([r[8] for r in rows_buf])
    meta = np.stack([r[9] for r in rows_buf])
    pair_source_idx = np.stack([r[10] for r in rows_buf])
    pair_target_idx = np.stack([r[11] for r in rows_buf])
    value_labels = np.array([r[12] for r in rows_buf], dtype=F32)
    raw_xy = np.stack([r[13] for r in rows_buf])         # (N, N_MAX, 3, 2)
    raw_ships = np.stack([r[14] for r in rows_buf])      # (N, N_MAX)
    raw_prod = np.stack([r[15] for r in rows_buf])       # (N, N_MAX)
    noop_labels = np.stack([r[16] for r in rows_buf])    # (N, N_MAX)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        planet_feats=planet_feats,
        globals=globals_,
        mask=mask,
        labels=labels,
        offensive_labels=offensive_labels,
        offensive_source_labels=offensive_source_labels,
        support_target_labels=support_target_labels,
        support_source_labels=support_source_labels,
        planet_ids=planet_ids,
        meta=meta,
        pair_source_idx=pair_source_idx,
        pair_target_idx=pair_target_idx,
        value_labels=value_labels,
        raw_xy=raw_xy,
        raw_ships=raw_ships,
        raw_prod=raw_prod,
        noop_labels=noop_labels,
        feat_names=np.array(PLANET_FEAT_NAMES, dtype=object),
        global_names=np.array(GLOBAL_FEAT_NAMES, dtype=object),
    )
    rates = {k: (v[mask].mean() if mask.any() else 0.0) * 100 for k, v in [
        ("labels", labels),
        ("offensive", offensive_labels),
        ("offensive_source", offensive_source_labels),
        ("support_target", support_target_labels),
        ("support_source", support_source_labels),
    ]}
    print(f"wrote {out_path}  rows={len(rows_buf)}  positive %s: " +
          "  ".join(f"{k}={v:.2f}" for k, v in rates.items()))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="*", default=None,
                    help="explicit zip paths. defaults to /tmp/orbit_days/*.zip")
    ap.add_argument("--zip-dir", type=pathlib.Path, default=pathlib.Path("/tmp/orbit_days"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent / "data" / "targets.npz")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = ap.parse_args()

    zip_paths = args.zips or sorted(str(p) for p in args.zip_dir.glob("*.zip"))
    if not zip_paths:
        print(f"no zips found in {args.zip_dir}", file=sys.stderr)
        return 1
    print(f"input: {len(zip_paths)} zips, workers={args.workers}, max_games={args.max_games}")
    return build(zip_paths, args.out, args.max_games, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
