from __future__ import annotations

import argparse
import base64
from pathlib import Path

import torch


TEMPLATE = '''from __future__ import annotations

import base64
import io
import math

import numpy as np
import torch
from torch import nn

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_SPEED_MAX = 6.0
PATH_MAX_TIME = 100
MAX_STEPS = 500
MAX_PLANETS = 64
SEND_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
PLANET_FEATURES = 26
GLOBAL_FEATURES = 18
ACTION_DIM = 1 + MAX_PLANETS * MAX_PLANETS * len(SEND_FRACTIONS)
ACTION_TARGET_LIMIT_PER_SOURCE = 16
MAX_LAUNCHES_PER_TURN = __MAX_LAUNCHES_PER_TURN__
MULTI_LAUNCH_LOGIT_MARGIN = __MULTI_LAUNCH_LOGIT_MARGIN__


def _get(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _hits_sun(src, tgt):
    return _seg_dist(CENTER, CENTER, src[2], src[3], tgt[2], tgt[3]) <= SUN_RADIUS + 0.75


def _fleet_speed(ships):
    ships = float(ships)
    if ships <= 1:
        return 1.0
    return max(1.0, min(SHIP_SPEED_MAX, 1.0 + (SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5))


def _norm_angle(a):
    a %= math.tau
    return a + math.tau if a < 0.0 else a


def _angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= math.tau
    while d <= -math.pi:
        d += math.tau
    return d


class _AngleSet:
    def __init__(self):
        self.ivs = []

    def add_arc(self, center, half):
        if half >= math.pi:
            self.ivs = [(0.0, math.tau)]
            return
        if half <= 0.0:
            return
        lo = _norm_angle(center - half)
        hi = lo + 2.0 * half
        parts = [(lo, hi)] if hi <= math.tau else [(lo, math.tau), (0.0, hi - math.tau)]
        out = []
        for a, b in sorted(self.ivs + parts, key=lambda iv: iv[0]):
            if out and a <= out[-1][1] + 1e-9:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        self.ivs = out

    def sub_arc(self, center, half):
        if half <= 0.0:
            return
        if half >= math.pi:
            self.ivs = []
            return
        lo = _norm_angle(center - half)
        hi = lo + 2.0 * half
        parts = [(lo, hi)] if hi <= math.tau else [(lo, math.tau), (0.0, hi - math.tau)]
        for s, e in parts:
            out = []
            for a, b in self.ivs:
                if b <= s + 1e-12 or a >= e - 1e-12:
                    out.append((a, b))
                else:
                    if a < s:
                        out.append((a, s))
                    if b > e:
                        out.append((e, b))
            self.ivs = [(a, b) for a, b in out if b - a > 1e-9]

    def closest_to(self, target):
        target = _norm_angle(target)
        best = None
        best_d = float("inf")
        for a, b in self.ivs:
            cand = target if a <= target <= b else (a if target < a else b)
            d = abs(_angle_diff(cand, target))
            if d < best_d:
                best = cand
                best_d = d
        return best


def _arc_half_angle(d_target, r_target, d_fleet):
    if d_fleet < 1e-9 or d_target < 1e-9:
        return None
    if d_target > d_fleet + r_target + 1e-9:
        return None
    if d_target + r_target < d_fleet - 1e-9:
        return None
    if r_target >= d_target + d_fleet:
        return math.pi
    cos_half = (d_fleet * d_fleet + d_target * d_target - r_target * r_target) / (2.0 * d_fleet * d_target)
    return max(1e-4, math.acos(max(-1.0, min(1.0, cos_half))))


def _swept_hit(a, b, p0, p1, radius):
    d0x = a[0] - p0[0]
    d0y = a[1] - p0[1]
    dvx = (b[0] - a[0]) - (p1[0] - p0[0])
    dvy = (b[1] - a[1]) - (p1[1] - p0[1])
    aq = dvx * dvx + dvy * dvy
    bq = 2.0 * (d0x * dvx + d0y * dvy)
    cq = d0x * d0x + d0y * d0y - radius * radius
    if aq < 1e-12:
        return cq <= 0.0
    disc = bq * bq - 4.0 * aq * cq
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    return (-bq + sq) / (2.0 * aq) >= 0.0 and (-bq - sq) / (2.0 * aq) <= 1.0


def _on_board(p):
    return 0.0 <= p[0] <= BOARD_SIZE and 0.0 <= p[1] <= BOARD_SIZE


def _planet_pos_at(obs, planet, dt):
    pid = int(planet[0])
    comet_ids = set(int(x) for x in (_get(obs, "comet_planet_ids", []) or []))
    if pid in comet_ids:
        for group in _get(obs, "comets", []) or []:
            ids = [int(x) for x in group.get("planet_ids", [])]
            if pid not in ids:
                continue
            idx = ids.index(pid)
            paths = group.get("paths", [])
            path_index = int(group.get("path_index", 0)) + dt
            if idx >= len(paths) or path_index < 0 or path_index >= len(paths[idx]):
                return None
            return float(paths[idx][path_index][0]), float(paths[idx][path_index][1])
        return float(planet[2]), float(planet[3])
    radius = float(planet[4])
    if math.hypot(float(planet[2]) - CENTER, float(planet[3]) - CENTER) + radius >= ROTATION_RADIUS_LIMIT:
        return float(planet[2]), float(planet[3])
    initial = None
    for p0 in _get(obs, "initial_planets", []) or []:
        if int(p0[0]) == pid:
            initial = p0
            break
    if initial is None:
        return float(planet[2]), float(planet[3])
    av = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    step = int(_get(obs, "step", 0) or 0)
    dx0 = float(initial[2]) - CENTER
    dy0 = float(initial[3]) - CENTER
    r = math.hypot(dx0, dy0)
    angle = math.atan2(dy0, dx0) + av * (step + dt)
    return CENTER + r * math.cos(angle), CENTER + r * math.sin(angle)


def _dir_to_hit(obs, source, target, ships):
    speed = _fleet_speed(ships)
    src_pos = (float(source[2]), float(source[3]))
    spawn_offset = float(source[4]) + 0.1
    cand = _AngleSet()
    max_target_t = 0
    for t in range(1, PATH_MAX_TIME + 1):
        target_pos = _planet_pos_at(obs, target, t)
        if target_pos is None or not _on_board(target_pos):
            continue
        d_target = math.hypot(target_pos[0] - src_pos[0], target_pos[1] - src_pos[1])
        if d_target < 1e-6:
            continue
        half = _arc_half_angle(d_target, float(target[4]), spawn_offset + speed * t)
        if half is not None:
            cand.add_arc(math.atan2(target_pos[1] - src_pos[1], target_pos[0] - src_pos[0]), half)
            max_target_t = max(max_target_t, t)
    if not cand.ivs:
        return None
    d_sun = math.hypot(CENTER - src_pos[0], CENTER - src_pos[1])
    if d_sun <= SUN_RADIUS:
        return None
    cand.sub_arc(math.atan2(CENTER - src_pos[1], CENTER - src_pos[0]), math.asin(min(1.0, (SUN_RADIUS + 0.05) / d_sun)))
    raw_planets = list(_get(obs, "planets", []) or [])
    comet_ids = set(int(x) for x in (_get(obs, "comet_planet_ids", []) or []))
    moving = []
    for other in raw_planets:
        if int(other[0]) in (int(source[0]), int(target[0])):
            continue
        r = math.hypot(float(other[2]) - CENTER, float(other[3]) - CENTER)
        is_orbiting = r + float(other[4]) < ROTATION_RADIUS_LIMIT
        is_comet = int(other[0]) in comet_ids
        if is_orbiting or is_comet:
            moving.append(other)
            continue
        d_obs = math.hypot(float(other[2]) - src_pos[0], float(other[3]) - src_pos[1])
        if d_obs > 1e-6:
            cand.sub_arc(math.atan2(float(other[3]) - src_pos[1], float(other[2]) - src_pos[0]), math.asin(min(1.0, (float(other[4]) + 0.1) / d_obs)))
    if not cand.ivs:
        return None
    for k in range(1, max_target_t + 1):
        fleet_d = spawn_offset + speed * k
        for other in moving:
            obs_pos = _planet_pos_at(obs, other, k)
            if obs_pos is None or not _on_board(obs_pos):
                continue
            d_obs = math.hypot(obs_pos[0] - src_pos[0], obs_pos[1] - src_pos[1])
            if d_obs < 1e-6:
                continue
            buf = float(other[4]) + 0.25
            if abs(fleet_d - d_obs) > buf + speed * 0.5 + 0.5:
                continue
            cand.sub_arc(math.atan2(obs_pos[1] - src_pos[1], obs_pos[0] - src_pos[0]), math.asin(min(1.0, buf / d_obs)))
            if not cand.ivs:
                return None
    target_now = _planet_pos_at(obs, target, 0) or (float(target[2]), float(target[3]))
    angle = cand.closest_to(math.atan2(target_now[1] - src_pos[1], target_now[0] - src_pos[0]))
    if angle is None:
        return None
    pos = (src_pos[0] + spawn_offset * math.cos(angle), src_pos[1] + spawn_offset * math.sin(angle))
    dx = speed * math.cos(angle)
    dy = speed * math.sin(angle)
    for t in range(1, PATH_MAX_TIME + 1):
        new_pos = (pos[0] + dx, pos[1] + dy)
        t_old = _planet_pos_at(obs, target, t - 1)
        t_new = _planet_pos_at(obs, target, t)
        if t_old is None or t_new is None:
            break
        if _swept_hit(pos, new_pos, t_old, t_new, float(target[4])):
            return angle, t
        if not _on_board(new_pos):
            break
        pos = new_pos
    return None


def _predict_fleet_target(obs, fleet, planets):
    speed = _fleet_speed(float(fleet[6]))
    angle = float(fleet[4])
    dx = speed * math.cos(angle)
    dy = speed * math.sin(angle)
    pos = (float(fleet[2]), float(fleet[3]))
    for dt in range(1, PATH_MAX_TIME + 1):
        new_pos = (pos[0] + dx, pos[1] + dy)
        for planet in planets:
            p_old = _planet_pos_at(obs, planet, dt - 1)
            p_new = _planet_pos_at(obs, planet, dt)
            if p_old is None or p_new is None:
                continue
            if _swept_hit(pos, new_pos, p_old, p_new, float(planet[4])):
                return int(planet[0])
        if not _on_board(new_pos):
            return None
        if _seg_dist(CENTER, CENTER, pos[0], pos[1], new_pos[0], new_pos[1]) < SUN_RADIUS:
            return None
        pos = new_pos
    return None


def _nearest_distance(x, y, positions, fallback=BOARD_SIZE):
    if not positions:
        return fallback
    return min(math.hypot(x - px, y - py) for px, py in positions)


def _action_index(s, t, b):
    return 1 + s * MAX_PLANETS * len(SEND_FRACTIONS) + t * len(SEND_FRACTIONS) + b


def _decode(index):
    if index <= 0:
        return None
    raw = index - 1
    b = raw % len(SEND_FRACTIONS)
    raw //= len(SEND_FRACTIONS)
    t = raw % MAX_PLANETS
    s = raw // MAX_PLANETS
    return s, t, b


def _candidate_target_slots(raw_planets, source_slot, player, incoming=None, limit=ACTION_TARGET_LIMIT_PER_SOURCE):
    n = min(len(raw_planets), MAX_PLANETS)
    if n <= 1:
        return []
    if n - 1 <= limit:
        return [i for i in range(n) if i != source_slot]
    src = raw_planets[source_slot]
    sx, sy = float(src[2]), float(src[3])
    scored = []
    for i, tgt in enumerate(raw_planets[:n]):
        if i == source_slot:
            continue
        owner = int(tgt[1])
        distance = math.hypot(float(tgt[2]) - sx, float(tgt[3]) - sy)
        ships = float(tgt[5])
        production = float(tgt[6])
        if owner == player:
            inbound_my, inbound_enemy = (incoming or {}).get(int(tgt[0]), [0.0, 0.0])
            owner_bias = 0.6 if inbound_enemy > inbound_my else 2.0
        elif owner == -1:
            owner_bias = 0.0
        else:
            owner_bias = -0.2
        scored.append((owner_bias + distance / 100.0 + ships / 900.0 - production * 0.04, i))
    scored.sort(key=lambda item: item[0])
    return [i for _score, i in scored[:limit]]


def _incoming_by_planet(obs, raw_planets, player):
    incoming = {int(p[0]): [0.0, 0.0] for p in raw_planets[:MAX_PLANETS]}
    for f in list(_get(obs, "fleets", []) or [])[:128]:
        target_id = _predict_fleet_target(obs, f, raw_planets[:MAX_PLANETS])
        if target_id is None or target_id not in incoming:
            continue
        if int(f[1]) == player:
            incoming[target_id][0] += float(f[6])
        else:
            incoming[target_id][1] += float(f[6])
    return incoming


def _remaining(obs):
    return [max(0, int(float(p[5]))) for p in list(_get(obs, "planets", []) or [])[:MAX_PLANETS]]


def _action_mask_for_remaining(obs, remaining):
    player = int(_get(obs, "player", 0))
    raw_planets = list(_get(obs, "planets", []) or [])
    incoming = _incoming_by_planet(obs, raw_planets, player)
    action_mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    action_mask[0] = True
    n = min(len(raw_planets), MAX_PLANETS)
    for s in range(n):
        src = raw_planets[s]
        available = int(remaining[s]) if s < len(remaining) else 0
        if int(src[1]) != player or available <= 1:
            continue
        for t in _candidate_target_slots(raw_planets, s, player, incoming):
            tgt = raw_planets[t]
            for b in range(len(SEND_FRACTIONS)):
                ships = min(available, max(1, int(available * SEND_FRACTIONS[b])))
                if ships > 0 and _dir_to_hit(obs, src, tgt, ships) is not None:
                    action_mask[_action_index(s, t, b)] = True
    return action_mask


def _encode(obs):
    player = int(_get(obs, "player", 0))
    step = int(_get(obs, "step", 0) or 0)
    raw_planets = list(_get(obs, "planets", []) or [])
    raw_fleets = list(_get(obs, "fleets", []) or [])
    comet_ids = set(int(x) for x in (_get(obs, "comet_planet_ids", []) or []))
    av = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    planets = np.zeros((MAX_PLANETS, PLANET_FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_PLANETS, dtype=np.float32)
    action_mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    action_mask[0] = True
    my_ships = enemy_ships = neutral_ships = 0.0
    my_planets = enemy_planets = neutral_planets = 0
    my_prod = enemy_prod = 0.0
    n = min(len(raw_planets), MAX_PLANETS)
    my_positions = []
    enemy_positions = []
    neutral_positions = []
    orbiting_count = 0
    comet_count = 0
    incoming = {}
    for p in raw_planets[:MAX_PLANETS]:
        pid, owner, x, y, radius, _ships, _prod = p
        owner = int(owner)
        x, y, radius = float(x), float(y), float(radius)
        r = math.hypot(x - CENTER, y - CENTER)
        if int(pid) in comet_ids:
            comet_count += 1
        if r + radius < ROTATION_RADIUS_LIMIT:
            orbiting_count += 1
        if owner == player:
            my_positions.append((x, y))
        elif owner == -1:
            neutral_positions.append((x, y))
        else:
            enemy_positions.append((x, y))
        incoming[int(pid)] = [0.0, 0.0]
    for f in raw_fleets[:128]:
        target_id = _predict_fleet_target(obs, f, raw_planets[:MAX_PLANETS])
        if target_id is None or target_id not in incoming:
            continue
        if int(f[1]) == player:
            incoming[target_id][0] += float(f[6])
        else:
            incoming[target_id][1] += float(f[6])

    for i, p in enumerate(raw_planets[:MAX_PLANETS]):
        pid, owner, x, y, radius, ships, prod = p
        owner = int(owner)
        x, y, radius, ships, prod = float(x), float(y), float(radius), float(ships), float(prod)
        dx, dy = x - CENTER, y - CENTER
        r = math.hypot(dx, dy)
        mine = 1.0 if owner == player else 0.0
        neutral = 1.0 if owner == -1 else 0.0
        enemy = 1.0 if owner not in (-1, player) else 0.0
        future_x, future_y = _planet_pos_at(obs, p, 10) or (x, y)
        inbound_my, inbound_enemy = incoming.get(int(pid), [0.0, 0.0])
        if mine:
            my_ships += ships; my_planets += 1; my_prod += prod
        elif enemy:
            enemy_ships += ships; enemy_planets += 1; enemy_prod += prod
        else:
            neutral_ships += ships; neutral_planets += 1
        angle = math.atan2(dy, dx)
        planets[i] = np.array([
            dx / CENTER, dy / CENTER, r / 70.7107, math.sin(angle), math.cos(angle),
            radius / 4.0, math.log1p(ships) / math.log1p(1000.0), min(1.0, ships / 500.0),
            prod / 5.0, mine, enemy, neutral, 1.0 if int(pid) in comet_ids else 0.0,
            1.0 if r + radius < ROTATION_RADIUS_LIMIT else 0.0,
            (future_x - CENTER) / CENTER, (future_y - CENTER) / CENTER,
            max(-1.0, min(1.0, (future_x - x) / 20.0)),
            max(-1.0, min(1.0, (future_y - y) / 20.0)),
            math.log1p(inbound_my) / math.log1p(1000.0),
            math.log1p(inbound_enemy) / math.log1p(1000.0),
            max(-1.0, min(1.0, (inbound_my - inbound_enemy) / 500.0)),
            max(-1.0, min(1.0, (ships + inbound_my - inbound_enemy) / 500.0)),
            _nearest_distance(x, y, my_positions) / 100.0,
            _nearest_distance(x, y, enemy_positions) / 100.0,
            _nearest_distance(x, y, neutral_positions) / 100.0,
            1.0 if mine and ships > 1 else 0.0,
        ], dtype=np.float32)
        mask[i] = 1.0
    fleet_my = fleet_enemy = 0.0
    for f in raw_fleets:
        if int(f[1]) == player:
            fleet_my += float(f[6])
        else:
            fleet_enemy += float(f[6])
    own_total = my_ships + fleet_my
    enemy_total = enemy_ships + fleet_enemy
    total = max(1.0, own_total + enemy_total + neutral_ships)
    prod_total = max(1.0, my_prod + enemy_prod)
    non_neutral_planets = max(1, my_planets + enemy_planets)
    globals_ = np.array([
        step / MAX_STEPS, av / 0.05, my_planets / MAX_PLANETS, enemy_planets / MAX_PLANETS,
        neutral_planets / MAX_PLANETS, own_total / total, enemy_total / total, neutral_ships / total,
        my_prod / 50.0, enemy_prod / 50.0, my_prod / prod_total, enemy_prod / prod_total,
        (my_planets - enemy_planets) / non_neutral_planets, (own_total - enemy_total) / total,
        fleet_my / total, fleet_enemy / total, comet_count / MAX_PLANETS, orbiting_count / MAX_PLANETS
    ], dtype=np.float32)
    for s in range(n):
        src = raw_planets[s]
        if int(src[1]) != player or int(src[5]) <= 1:
            continue
        for t in _candidate_target_slots(raw_planets, s, player, incoming):
            tgt = raw_planets[t]
            for b in range(len(SEND_FRACTIONS)):
                ships = min(int(src[5]), max(1, int(float(src[5]) * SEND_FRACTIONS[b])))
                if ships > 0 and _dir_to_hit(obs, src, tgt, ships) is not None:
                    action_mask[_action_index(s, t, b)] = True
    return planets, mask, globals_, action_mask


def _move(obs, index, remaining=None):
    decoded = _decode(int(index))
    if decoded is None:
        return []
    s, t, b = decoded
    raw_planets = list(_get(obs, "planets", []) or [])
    if s >= len(raw_planets) or t >= len(raw_planets):
        return []
    player = int(_get(obs, "player", 0))
    src, tgt = raw_planets[s], raw_planets[t]
    available = int(remaining[s]) if remaining is not None and s < len(remaining) else int(float(src[5]))
    if int(src[1]) != player or s == t or available <= 1:
        return []
    ships = min(available, max(1, int(available * SEND_FRACTIONS[b])))
    path = _dir_to_hit(obs, src, tgt, ships)
    if path is None:
        return []
    angle = path[0]
    return [[int(src[0]), angle, ships]]


def _turn_moves(obs, logits, first_mask):
    remaining = _remaining(obs)
    moves = []
    raw_logits = logits[0]
    threshold = float(raw_logits[0].item()) + float(MULTI_LAUNCH_LOGIT_MARGIN)
    valid = torch.as_tensor(first_mask, dtype=torch.bool)
    candidate_mask = valid & (raw_logits >= threshold)
    candidate_mask[0] = False
    candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist()
    ranked = sorted((int(i) for i in candidates), key=lambda i: float(raw_logits[i].item()), reverse=True)
    for action in ranked:
        if len(moves) >= max(1, int(MAX_LAUNCHES_PER_TURN)):
            break
        move = _move(obs, action, remaining)
        if not move:
            continue
        moves.extend(move)
        decoded = _decode(action)
        if decoded is None:
            break
        s, _t, _b = decoded
        remaining[s] = max(0, remaining[s] - int(move[0][2]))
    return moves


class OrbitPolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.planet_encoder = nn.Sequential(nn.Linear(PLANET_FEATURES, hidden), nn.LayerNorm(hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pair_head = nn.Sequential(nn.Linear(hidden * 4 + 4 + GLOBAL_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, len(SEND_FRACTIONS)))
        self.noop_head = nn.Sequential(nn.Linear(hidden + GLOBAL_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden + GLOBAL_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, planets, planet_mask, globals_, action_mask=None):
        batch = planets.shape[0]
        enc = self.planet_encoder(planets)
        m = planet_mask.unsqueeze(-1)
        pooled = (enc * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        src = enc.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        tgt = enc.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)
        xy = planets[..., :2]
        d = xy.unsqueeze(1) - xy.unsqueeze(2)
        dist = torch.linalg.norm(d, dim=-1, keepdim=True)
        geom = torch.cat([d, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair_logits = self.pair_head(torch.cat([pair, geom, g], dim=-1)).reshape(batch, -1)
        logits = torch.cat([self.noop_head(torch.cat([pooled, globals_], dim=-1)), pair_logits], dim=-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


class EntityTransformerPolicy(nn.Module):
    def __init__(self, hidden=128, layers=3, heads=4):
        super().__init__()
        self.planet_encoder = nn.Sequential(nn.Linear(PLANET_FEATURES, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_FEATURES, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(hidden)
        self.pair_head = nn.Sequential(nn.Linear(hidden * 4 + 4 + GLOBAL_FEATURES, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(SEND_FRACTIONS)))
        self.noop_head = nn.Sequential(nn.Linear(hidden + GLOBAL_FEATURES, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, planets, planet_mask, globals_, action_mask=None):
        batch = planets.shape[0]
        planet_tokens = self.planet_encoder(planets)
        global_token = self.global_encoder(globals_).unsqueeze(1)
        tokens = torch.cat([global_token, planet_tokens], dim=1)
        valid = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=planet_mask.device), planet_mask.bool()], dim=1)
        enc = self.final_norm(self.transformer(tokens, src_key_padding_mask=~valid))
        global_enc = enc[:, 0]
        planet_enc = enc[:, 1:]
        src = planet_enc.unsqueeze(2).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        tgt = planet_enc.unsqueeze(1).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair = torch.cat([src, tgt, src - tgt, src * tgt], dim=-1)
        xy = planets[..., :2]
        d = xy.unsqueeze(1) - xy.unsqueeze(2)
        dist = torch.linalg.norm(d, dim=-1, keepdim=True)
        geom = torch.cat([d, dist, dist.clamp_min(1e-4).reciprocal().clamp_max(20.0)], dim=-1)
        g = globals_.view(batch, 1, 1, GLOBAL_FEATURES).expand(batch, MAX_PLANETS, MAX_PLANETS, -1)
        pair_logits = self.pair_head(torch.cat([pair, geom, g], dim=-1)).reshape(batch, -1)
        logits = torch.cat([self.noop_head(torch.cat([global_enc, globals_], dim=-1)), pair_logits], dim=-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


def _build_policy(config):
    model_type = config.get("model", "mlp")
    hidden = int(config.get("hidden", 128))
    if model_type == "entity_transformer":
        return EntityTransformerPolicy(hidden, int(config.get("transformer_layers", 3)), int(config.get("transformer_heads", 4)))
    return OrbitPolicy(hidden)


_MODEL = None
_STATE_BYTES = base64.b64decode("__STATE_B64__")


def _model():
    global _MODEL
    if _MODEL is None:
        ckpt = torch.load(io.BytesIO(_STATE_BYTES), map_location="cpu")
        m = _build_policy(ckpt.get("config", {}))
        m.load_state_dict(ckpt["model"], strict=False)
        m.eval()
        _MODEL = m
    return _MODEL


def agent(obs):
    planets, mask, globals_, action_mask = _encode(obs)
    with torch.no_grad():
        logits = _model()(
            torch.as_tensor(planets, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(globals_, dtype=torch.float32).unsqueeze(0),
            None,
        )
    return _turn_moves(obs, logits, action_mask)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a checkpoint as a single-file Kaggle agent.")
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="bots/mine/rl_ppo/main.py")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    payload = {"model": ckpt["model"], "config": ckpt.get("config", {})}
    tmp = Path("/tmp/orbit_wars_rl_export.pt")
    torch.save(payload, tmp)
    data = tmp.read_bytes()
    max_launches = int(ckpt.get("config", {}).get("max_launches_per_turn", 4))
    logit_margin = float(ckpt.get("config", {}).get("multi_launch_logit_margin", 0.0))
    text = (
        TEMPLATE
        .replace("__STATE_B64__", base64.b64encode(data).decode("ascii"))
        .replace("__MAX_LAUNCHES_PER_TURN__", str(max(1, max_launches)))
        .replace("__MULTI_LAUNCH_LOGIT_MARGIN__", repr(logit_margin))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
