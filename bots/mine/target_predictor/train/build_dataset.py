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
F_PLANET = 46
F_GLOBAL = 12

KDE_SIGMA = 30.0  # one-bandwidth KDE; opening bot uses 4 but added cost > added signal here.

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


def predict_fleet_collision(state, fleet):
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


# ---------------------------------------------------------------------------
# Feature extraction (per-planet + global)
# ---------------------------------------------------------------------------


PLANET_FEAT_NAMES = [
    # owner one-hot (3)
    "is_mine", "is_neutral", "is_enemy",
    # stock (3)
    "ships", "ships_log1p", "production",
    # orbit/geom (5)
    "orbit_radius", "omega", "cos_theta", "sin_theta", "planet_radius",
    # position (6)
    "x_now", "y_now", "x_p10", "y_p10", "x_p25", "y_p25",
    # inbound fleets per side (8): mine + enemy × {count, total_ships, min_eta, mean_eta}
    "in_mine_count", "in_mine_ships", "in_mine_min_eta", "in_mine_mean_eta",
    "in_enemy_count", "in_enemy_ships", "in_enemy_min_eta", "in_enemy_mean_eta",
    # reachability from my planets (4)
    "min_eta_from_me", "surplus_at_min_eta_src", "my_ships_arrivable_25", "my_ships_arrivable_50",
    # frontier/geom (4)
    "dist_nearest_my", "dist_nearest_enemy", "is_closest_neutral_to_me", "is_closest_enemy_to_me",
    # listwise rank (4): rank in [0,1] within turn (1.0 = best)
    "rank_production", "rank_ships", "rank_min_eta_from_me", "rank_dist_nearest_my",
    # tempo (1)
    "turns_since_owner_change",
    # comet (2)
    "is_comet", "comet_remaining",
    # opening-bot-inspired geometry + density (6)
    "is_orbiting", "dist_to_edge", "d_center",
    "kde_unit", "kde_prod",
    "turns_to_cap_proxy",
]
assert len(PLANET_FEAT_NAMES) == F_PLANET


GLOBAL_FEAT_NAMES = [
    "turn_num", "turn_norm",
    "my_ships_total", "enemy_ships_total",
    "my_prod_total", "enemy_prod_total",
    "my_planet_count", "enemy_planet_count",
    "economy_diff",
    "phase_opening", "phase_mid", "phase_end",
]
assert len(GLOBAL_FEAT_NAMES) == F_GLOBAL


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

    # ----- inbound fleet aggregation by destination (predicted) -----
    in_mine_count = [0] * n
    in_mine_ships = [0] * n
    in_mine_etas: list[list[float]] = [[] for _ in range(n)]
    in_enem_count = [0] * n
    in_enem_ships = [0] * n
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
            in_mine_count[idx] += 1
            in_mine_ships[idx] += f["ships"]
            in_mine_etas[idx].append(eta)
        else:
            in_enem_count[idx] += 1
            in_enem_ships[idx] += f["ships"]
            in_enem_etas[idx].append(eta)

    # ----- reachability from MY planets (proxy ETA = straight-line / fleet_speed(surplus)) -----
    my_planets = [p for p in planets if p["owner"] == player]
    # surplus = ships at planet (no reservation model yet)
    # ETA from src to dest: dist(src, dest_now) / fleet_speed(min(src.ships, dest_now_threat))
    # MVP: use src.ships for speed; if src.ships < 2 → speed 1.
    min_eta_from_me = [math.inf] * n
    surplus_at_src = [0.0] * n
    arrivable_25 = [0.0] * n
    arrivable_50 = [0.0] * n
    for i, dest in enumerate(planets):
        dx_now, dy_now = dest["x"], dest["y"]
        for src in my_planets:
            if src["id"] == dest["id"]:
                continue
            dx = dx_now - src["x"]; dy = dy_now - src["y"]
            d = math.sqrt(dx * dx + dy * dy)
            speed = fleet_speed(max(src["ships"], 1))
            eta = d / max(speed, 1e-6)
            if eta < min_eta_from_me[i]:
                min_eta_from_me[i] = eta
                surplus_at_src[i] = src["ships"]
            if eta <= 25:
                arrivable_25[i] += src["ships"]
            if eta <= 50:
                arrivable_50[i] += src["ships"]
        if math.isinf(min_eta_from_me[i]):
            min_eta_from_me[i] = 200.0  # sentinel: no reach
            surplus_at_src[i] = 0.0

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

    # ----- opening-bot-inspired: KDE (Gaussian density of neighbors) -----
    # For each planet, sum_{q != p} exp(-d²/(2σ²)) at sigma=30.
    # Two variants: unit weight, production weight. Captures cluster density at scale σ.
    two_sig2 = 2.0 * KDE_SIGMA * KDE_SIGMA
    kde_unit = [0.0] * n
    kde_prod = [0.0] * n
    for i, p in enumerate(planets):
        su = 0.0; sp = 0.0
        for q in planets:
            if q["id"] == p["id"]:
                continue
            dx = p["x"] - q["x"]; dy = p["y"] - q["y"]
            w = math.exp(-(dx * dx + dy * dy) / two_sig2)
            su += w
            sp += w * q["prod"]
        kde_unit[i] = su
        kde_prod[i] = sp

    # turns_to_cap_proxy: how long would my total production take to grow ships+1?
    my_total_prod = sum(p["prod"] for p in planets if p["owner"] == player)
    inv_prod = 1.0 / max(my_total_prod, 1)

    # ----- listwise rank within turn (1.0 = largest) -----
    # for ETA we invert (smallest ETA = best = 1.0); same for dist_nearest_my.
    rank_prod = _ranks_01([p["prod"] for p in planets])
    rank_ships = _ranks_01([p["ships"] for p in planets])
    rank_eta = _ranks_01([-min_eta_from_me[i] for i in range(n)])
    rank_dist = _ranks_01([-dist_nearest_my[i] for i in range(n)])

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
        feats[i] = [
            is_mine[i], is_neut[i], is_enem[i],
            ships, math.log1p(max(ships, 0)), p["prod"],
            p["orb_r"], state["av"], math.cos(theta), math.sin(theta), p["radius"],
            pos_now[i][0], pos_now[i][1], pos_p10[i][0], pos_p10[i][1], pos_p25[i][0], pos_p25[i][1],
            in_mine_count[i], in_mine_ships[i], in_min_mine, in_mean_mine,
            in_enem_count[i], in_enem_ships[i], in_min_enem, in_mean_enem,
            min_eta_from_me[i], surplus_at_src[i], arrivable_25[i], arrivable_50[i],
            dist_nearest_my[i], dist_nearest_enemy[i], is_closest_neutral[i], is_closest_enemy[i],
            rank_prod[i], rank_ships[i], rank_eta[i], rank_dist[i],
            min(tsoc, 200),
            int(p["is_comet"]), comet_remaining(state, p),
            int(p["is_orbiting"]),
            min(p["x"], p["y"], BOARD - p["x"], BOARD - p["y"]),
            math.hypot(p["x"] - CENTER[0], p["y"] - CENTER[1]),
            kde_unit[i], kde_prod[i],
            (ships + 1) * inv_prod,
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
    globals_ = np.array([
        cur_turn, cur_turn / 200.0,
        my_ships, enemy_ships, my_prod, enemy_prod, my_pc, enemy_pc,
        econ_diff, phase_open, phase_mid, phase_end,
    ], dtype=F32)

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


# ---------------------------------------------------------------------------
# Owner-change history tracking
# ---------------------------------------------------------------------------


def update_owner_history(state, last_owner, owner_change_turn, cur_turn):
    for p in state["planets"]:
        pid = p["id"]
        if pid not in last_owner:
            last_owner[pid] = p["owner"]
            owner_change_turn[pid] = cur_turn
        elif last_owner[pid] != p["owner"]:
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
            labels = labels_for_step(state_t, state_tp1, player, planet_ids)

            # pad to N_MAX
            pad_feats = np.zeros((N_MAX, F_PLANET), dtype=F32)
            pad_feats[:n_real] = feats
            pad_labels = np.zeros(N_MAX, dtype=F32)
            pad_labels[:n_real] = labels
            pad_ids = np.zeros(N_MAX, dtype=np.int32)
            pad_ids[:n_real] = planet_ids
            mask = np.zeros(N_MAX, dtype=bool)
            mask[:n_real] = True

            meta = np.array([game_id_int, t, player, n_real], dtype=np.int32)
            rows.append((pad_feats, globals_, mask, pad_labels, pad_ids, meta))
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
    planet_ids = np.stack([r[4] for r in rows_buf])
    meta = np.stack([r[5] for r in rows_buf])

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        planet_feats=planet_feats,
        globals=globals_,
        mask=mask,
        labels=labels,
        planet_ids=planet_ids,
        meta=meta,
        feat_names=np.array(PLANET_FEAT_NAMES, dtype=object),
        global_names=np.array(GLOBAL_FEAT_NAMES, dtype=object),
    )
    pos_rate = labels[mask].mean() if mask.any() else 0.0
    print(f"wrote {out_path}  rows={len(rows_buf)} "
          f"positives={pos_rate * 100:.2f}% of masked planet-slots")
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
