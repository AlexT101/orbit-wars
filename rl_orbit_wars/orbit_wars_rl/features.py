from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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
DEFAULT_MAX_LAUNCHES_PER_TURN = 4
DEFAULT_MULTI_LAUNCH_LOGIT_MARGIN = 0.0


@dataclass(frozen=True)
class EncodedObs:
    planets: np.ndarray
    planet_mask: np.ndarray
    globals: np.ndarray
    action_mask: np.ndarray


@dataclass(frozen=True)
class GameStats:
    step: int
    remaining: int
    own_score: float
    enemy_score: float
    neutral_ships: float
    own_planets: int
    enemy_planets: int
    neutral_planets: int
    own_production: float
    enemy_production: float
    own_fleet_ships: float
    enemy_fleet_ships: float
    score_share: float
    production_share: float
    planet_share: float
    score_margin: float
    production_margin: float
    economy_value: float


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def path_hits_sun(src: list[float], tgt: list[float], safety: float = 0.75) -> bool:
    return _segment_distance(CENTER, CENTER, src[2], src[3], tgt[2], tgt[3]) <= SUN_RADIUS + safety


def fleet_speed(ships: int | float, max_speed: float = SHIP_SPEED_MAX) -> float:
    ships = float(ships)
    if ships <= 1:
        return 1.0
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return max(1.0, min(max_speed, speed))


def _norm_angle(angle: float) -> float:
    angle %= math.tau
    return angle + math.tau if angle < 0.0 else angle


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= math.tau
    while d <= -math.pi:
        d += math.tau
    return d


class _AngleSet:
    def __init__(self) -> None:
        self.ivs: list[tuple[float, float]] = []

    def is_empty(self) -> bool:
        return not self.ivs

    def add_arc(self, center: float, half: float) -> None:
        if half >= math.pi:
            self.ivs = [(0.0, math.tau)]
            return
        if half <= 0.0:
            return
        lo = _norm_angle(center - half)
        hi = lo + 2.0 * half
        parts = [(lo, hi)] if hi <= math.tau else [(lo, math.tau), (0.0, hi - math.tau)]
        all_ivs = sorted(self.ivs + parts, key=lambda iv: iv[0])
        out: list[tuple[float, float]] = []
        for a, b in all_ivs:
            if out and a <= out[-1][1] + 1e-9:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        self.ivs = out

    def sub_arc(self, center: float, half: float) -> None:
        if half <= 0.0:
            return
        if half >= math.pi:
            self.ivs = []
            return
        lo = _norm_angle(center - half)
        hi = lo + 2.0 * half
        parts = [(lo, hi)] if hi <= math.tau else [(lo, math.tau), (0.0, hi - math.tau)]
        for s, e in parts:
            out: list[tuple[float, float]] = []
            for a, b in self.ivs:
                if b <= s + 1e-12 or a >= e - 1e-12:
                    out.append((a, b))
                else:
                    if a < s:
                        out.append((a, s))
                    if b > e:
                        out.append((e, b))
            self.ivs = [(a, b) for a, b in out if b - a > 1e-9]

    def closest_to(self, target: float) -> float | None:
        if not self.ivs:
            return None
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


def _arc_half_angle(d_target: float, r_target: float, d_fleet: float) -> float | None:
    if d_fleet < 1e-9 or d_target < 1e-9:
        return None
    if d_target > d_fleet + r_target + 1e-9:
        return None
    if d_target + r_target < d_fleet - 1e-9:
        return None
    if r_target >= d_target + d_fleet:
        return math.pi
    cos_half = (d_fleet * d_fleet + d_target * d_target - r_target * r_target) / (
        2.0 * d_fleet * d_target
    )
    return max(1e-4, math.acos(max(-1.0, min(1.0, cos_half))))


def _swept_pair_hit(
    a: tuple[float, float],
    b: tuple[float, float],
    p0: tuple[float, float],
    p1: tuple[float, float],
    radius: float,
) -> bool:
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
    t1 = (-bq - sq) / (2.0 * aq)
    t2 = (-bq + sq) / (2.0 * aq)
    return t2 >= 0.0 and t1 <= 1.0


def _on_board(pos: tuple[float, float]) -> bool:
    return 0.0 <= pos[0] <= BOARD_SIZE and 0.0 <= pos[1] <= BOARD_SIZE


def _planet_pos_at(obs, planet: list[float], dt: int) -> tuple[float, float] | None:
    pid = int(planet[0])
    comet_ids = set(int(x) for x in (_obs_get(obs, "comet_planet_ids", []) or []))
    if pid in comet_ids:
        for group in _obs_get(obs, "comets", []) or []:
            ids = [int(x) for x in group.get("planet_ids", [])]
            if pid not in ids:
                continue
            idx = ids.index(pid)
            paths = group.get("paths", [])
            if idx >= len(paths):
                return None
            path_index = int(group.get("path_index", 0)) + dt
            path = paths[idx]
            if path_index < 0 or path_index >= len(path):
                return None
            return float(path[path_index][0]), float(path[path_index][1])
        return float(planet[2]), float(planet[3])

    radius = float(planet[4])
    if math.hypot(float(planet[2]) - CENTER, float(planet[3]) - CENTER) + radius >= ROTATION_RADIUS_LIMIT:
        return float(planet[2]), float(planet[3])

    initial = None
    for p0 in _obs_get(obs, "initial_planets", []) or []:
        if int(p0[0]) == pid:
            initial = p0
            break
    if initial is None:
        return float(planet[2]), float(planet[3])
    av = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)
    step = int(_obs_get(obs, "step", 0) or 0)
    dx0 = float(initial[2]) - CENTER
    dy0 = float(initial[3]) - CENTER
    orbital_radius = math.hypot(dx0, dy0)
    angle0 = math.atan2(dy0, dx0)
    angle = angle0 + av * (step + dt)
    return CENTER + orbital_radius * math.cos(angle), CENTER + orbital_radius * math.sin(angle)


def dir_to_hit(
    obs,
    source: list[float],
    target: list[float],
    num_ships: int,
    turns_in_future: int = 0,
) -> tuple[float, int] | None:
    speed = fleet_speed(num_ships)
    src_pos = (float(source[2]), float(source[3]))
    spawn_offset = float(source[4]) + 0.1

    cand = _AngleSet()
    max_target_t = 0
    for t in range(1, PATH_MAX_TIME + 1):
        target_pos = _planet_pos_at(obs, target, turns_in_future + t)
        if target_pos is None or not _on_board(target_pos):
            continue
        d_target = math.hypot(target_pos[0] - src_pos[0], target_pos[1] - src_pos[1])
        if d_target < 1e-6:
            continue
        angle_t = math.atan2(target_pos[1] - src_pos[1], target_pos[0] - src_pos[0])
        half = _arc_half_angle(d_target, float(target[4]), spawn_offset + speed * t)
        if half is not None:
            cand.add_arc(angle_t, half)
            max_target_t = max(max_target_t, t)
    if cand.is_empty():
        return None

    d_sun = math.hypot(CENTER - src_pos[0], CENTER - src_pos[1])
    if d_sun <= SUN_RADIUS:
        return None
    cand.sub_arc(math.atan2(CENTER - src_pos[1], CENTER - src_pos[0]), math.asin(min(1.0, (SUN_RADIUS + 0.05) / d_sun)))

    raw_planets = list(_obs_get(obs, "planets", []) or [])
    comet_ids = set(int(x) for x in (_obs_get(obs, "comet_planet_ids", []) or []))
    for other in raw_planets:
        if int(other[0]) in (int(source[0]), int(target[0])):
            continue
        r = math.hypot(float(other[2]) - CENTER, float(other[3]) - CENTER)
        is_orbiting = r + float(other[4]) < ROTATION_RADIUS_LIMIT
        if is_orbiting or int(other[0]) in comet_ids:
            continue
        d_obs = math.hypot(float(other[2]) - src_pos[0], float(other[3]) - src_pos[1])
        if d_obs < 1e-6:
            continue
        angle_obs = math.atan2(float(other[3]) - src_pos[1], float(other[2]) - src_pos[0])
        cand.sub_arc(angle_obs, math.asin(min(1.0, (float(other[4]) + 0.1) / d_obs)))
    if cand.is_empty():
        return None

    moving = []
    for other in raw_planets:
        if int(other[0]) in (int(source[0]), int(target[0])):
            continue
        r = math.hypot(float(other[2]) - CENTER, float(other[3]) - CENTER)
        if r + float(other[4]) < ROTATION_RADIUS_LIMIT or int(other[0]) in comet_ids:
            moving.append(other)

    for k in range(1, max_target_t + 1):
        fleet_d = spawn_offset + speed * k
        for other in moving:
            obs_pos = _planet_pos_at(obs, other, turns_in_future + k)
            if obs_pos is None or not _on_board(obs_pos):
                continue
            d_obs = math.hypot(obs_pos[0] - src_pos[0], obs_pos[1] - src_pos[1])
            if d_obs < 1e-6:
                continue
            buf = float(other[4]) + 0.25
            if abs(fleet_d - d_obs) > buf + speed * 0.5 + 0.5:
                continue
            cand.sub_arc(
                math.atan2(obs_pos[1] - src_pos[1], obs_pos[0] - src_pos[0]),
                math.asin(min(1.0, buf / d_obs)),
            )
            if cand.is_empty():
                return None

    target_now = _planet_pos_at(obs, target, turns_in_future) or (float(target[2]), float(target[3]))
    direct = math.atan2(target_now[1] - src_pos[1], target_now[0] - src_pos[0])
    angle = cand.closest_to(direct)
    if angle is None:
        return None

    pos = (src_pos[0] + spawn_offset * math.cos(angle), src_pos[1] + spawn_offset * math.sin(angle))
    dx = speed * math.cos(angle)
    dy = speed * math.sin(angle)
    for t in range(1, PATH_MAX_TIME + 1):
        new_pos = (pos[0] + dx, pos[1] + dy)
        t_old = _planet_pos_at(obs, target, turns_in_future + t - 1)
        t_new = _planet_pos_at(obs, target, turns_in_future + t)
        if t_old is None or t_new is None:
            break
        if _swept_pair_hit(pos, new_pos, t_old, t_new, float(target[4])):
            return angle, t
        if not _on_board(new_pos):
            break
        pos = new_pos
    return None


def _predict_fleet_target(obs, fleet: list[float], planets: list[list[float]]) -> int | None:
    speed = fleet_speed(float(fleet[6]))
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
            if _swept_pair_hit(pos, new_pos, p_old, p_new, float(planet[4])):
                return int(planet[0])
        if not _on_board(new_pos):
            return None
        if _segment_distance(CENTER, CENTER, pos[0], pos[1], new_pos[0], new_pos[1]) < SUN_RADIUS:
            return None
        pos = new_pos
    return None


def _nearest_distance(
    x: float,
    y: float,
    positions: list[tuple[float, float]],
    fallback: float = BOARD_SIZE,
) -> float:
    if not positions:
        return fallback
    return min(math.hypot(x - px, y - py) for px, py in positions)


def action_index(source_slot: int, target_slot: int, send_bin: int) -> int:
    per_source = MAX_PLANETS * len(SEND_FRACTIONS)
    return 1 + source_slot * per_source + target_slot * len(SEND_FRACTIONS) + send_bin


def decode_action_index(index: int) -> tuple[int, int, int] | None:
    if index <= 0:
        return None
    raw = index - 1
    send_bin = raw % len(SEND_FRACTIONS)
    raw //= len(SEND_FRACTIONS)
    target_slot = raw % MAX_PLANETS
    source_slot = raw // MAX_PLANETS
    return source_slot, target_slot, send_bin


def _candidate_target_slots(
    raw_planets: list[list[float]],
    source_slot: int,
    player: int,
    incoming: dict[int, list[float]] | None = None,
    limit: int = ACTION_TARGET_LIMIT_PER_SOURCE,
) -> list[int]:
    n = min(len(raw_planets), MAX_PLANETS)
    if n <= 1:
        return []
    if n - 1 <= limit:
        return [i for i in range(n) if i != source_slot]

    src = raw_planets[source_slot]
    sx = float(src[2])
    sy = float(src[3])
    scored: list[tuple[float, int]] = []
    for i, tgt in enumerate(raw_planets[:n]):
        if i == source_slot:
            continue
        owner = int(tgt[1])
        tx = float(tgt[2])
        ty = float(tgt[3])
        ships = float(tgt[5])
        production = float(tgt[6])
        distance = math.hypot(tx - sx, ty - sy)
        if owner == player:
            inbound_my, inbound_enemy = (incoming or {}).get(int(tgt[0]), [0.0, 0.0])
            owner_bias = 0.6 if inbound_enemy > inbound_my else 2.0
        elif owner == -1:
            owner_bias = 0.0
        else:
            owner_bias = -0.2
        score = owner_bias + distance / 100.0 + ships / 900.0 - production * 0.04
        scored.append((score, i))
    scored.sort(key=lambda item: item[0])
    return [i for _score, i in scored[:limit]]


def _incoming_by_planet(obs, raw_planets: list[list[float]], player: int) -> dict[int, list[float]]:
    incoming: dict[int, list[float]] = {int(p[0]): [0.0, 0.0] for p in raw_planets[:MAX_PLANETS]}
    for fleet in list(_obs_get(obs, "fleets", []) or [])[:128]:
        target_id = _predict_fleet_target(obs, fleet, raw_planets[:MAX_PLANETS])
        if target_id is None or target_id not in incoming:
            continue
        if int(fleet[1]) == player:
            incoming[target_id][0] += float(fleet[6])
        else:
            incoming[target_id][1] += float(fleet[6])
    return incoming


def remaining_ships_by_slot(obs) -> list[int]:
    return [max(0, int(float(p[5]))) for p in list(_obs_get(obs, "planets", []) or [])[:MAX_PLANETS]]


def _build_action_mask(
    obs,
    raw_planets: list[list[float]],
    player: int,
    incoming: dict[int, list[float]],
    remaining_ships: list[int] | None = None,
) -> np.ndarray:
    action_mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    action_mask[0] = True
    n = min(len(raw_planets), MAX_PLANETS)
    if remaining_ships is None:
        remaining_ships = remaining_ships_by_slot(obs)

    for si in range(n):
        src = raw_planets[si]
        available = int(remaining_ships[si]) if si < len(remaining_ships) else 0
        if int(src[1]) != player or available <= 1:
            continue
        for ti in _candidate_target_slots(raw_planets, si, player, incoming):
            tgt = raw_planets[ti]
            for bi, frac in enumerate(SEND_FRACTIONS):
                ships = max(1, int(available * frac))
                ships = min(ships, available)
                if ships > 0 and dir_to_hit(obs, src, tgt, ships) is not None:
                    action_mask[action_index(si, ti, bi)] = True
    return action_mask


def action_mask_for_remaining(obs, remaining_ships: list[int] | None = None) -> np.ndarray:
    player = int(_obs_get(obs, "player", 0))
    raw_planets = list(_obs_get(obs, "planets", []) or [])
    incoming = _incoming_by_planet(obs, raw_planets, player)
    return _build_action_mask(obs, raw_planets, player, incoming, remaining_ships)


def encode_obs(obs) -> EncodedObs:
    player = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    raw_planets = list(_obs_get(obs, "planets", []) or [])
    raw_fleets = list(_obs_get(obs, "fleets", []) or [])
    comet_ids = set(int(pid) for pid in (_obs_get(obs, "comet_planet_ids", []) or []))
    angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)

    planets = np.zeros((MAX_PLANETS, PLANET_FEATURES), dtype=np.float32)
    planet_mask = np.zeros(MAX_PLANETS, dtype=np.float32)

    my_ship_total = 0.0
    enemy_ship_total = 0.0
    neutral_ship_total = 0.0
    my_planets = 0
    enemy_planets = 0
    neutral_planets = 0
    my_production = 0.0
    enemy_production = 0.0

    n = min(len(raw_planets), MAX_PLANETS)
    my_positions: list[tuple[float, float]] = []
    enemy_positions: list[tuple[float, float]] = []
    neutral_positions: list[tuple[float, float]] = []
    orbiting_count = 0
    comet_count = 0
    for p in raw_planets[:MAX_PLANETS]:
        pid, owner, x, y, radius, _ships, _production = p
        owner = int(owner)
        x = float(x)
        y = float(y)
        radius = float(radius)
        orbital_radius = math.hypot(x - CENTER, y - CENTER)
        if int(pid) in comet_ids:
            comet_count += 1
        if orbital_radius + radius < ROTATION_RADIUS_LIMIT:
            orbiting_count += 1
        if owner == player:
            my_positions.append((x, y))
        elif owner == -1:
            neutral_positions.append((x, y))
        else:
            enemy_positions.append((x, y))
    incoming = _incoming_by_planet(obs, raw_planets, player)

    for i, p in enumerate(raw_planets[:MAX_PLANETS]):
        pid, owner, x, y, radius, ships, production = p
        owner = int(owner)
        x = float(x)
        y = float(y)
        radius = float(radius)
        ships = float(ships)
        production = float(production)
        dx = x - CENTER
        dy = y - CENTER
        orbital_radius = math.hypot(dx, dy)
        is_mine = 1.0 if owner == player else 0.0
        is_neutral = 1.0 if owner == -1 else 0.0
        is_enemy = 1.0 if owner not in (-1, player) else 0.0
        is_comet = 1.0 if int(pid) in comet_ids else 0.0
        is_orbiting = 1.0 if orbital_radius + radius < ROTATION_RADIUS_LIMIT else 0.0
        future_x, future_y = _planet_pos_at(obs, p, 10) or (x, y)
        inbound_my, inbound_enemy = incoming.get(int(pid), [0.0, 0.0])

        if is_mine:
            my_ship_total += ships
            my_planets += 1
            my_production += production
        elif is_enemy:
            enemy_ship_total += ships
            enemy_planets += 1
            enemy_production += production
        else:
            neutral_ship_total += ships
            neutral_planets += 1

        planets[i] = np.array(
            [
                dx / CENTER,
                dy / CENTER,
                orbital_radius / 70.7107,
                math.sin(math.atan2(dy, dx)),
                math.cos(math.atan2(dy, dx)),
                radius / 4.0,
                math.log1p(ships) / math.log1p(1000.0),
                min(1.0, ships / 500.0),
                production / 5.0,
                is_mine,
                is_enemy,
                is_neutral,
                is_comet,
                is_orbiting,
                (future_x - CENTER) / CENTER,
                (future_y - CENTER) / CENTER,
                max(-1.0, min(1.0, (future_x - x) / 20.0)),
                max(-1.0, min(1.0, (future_y - y) / 20.0)),
                math.log1p(inbound_my) / math.log1p(1000.0),
                math.log1p(inbound_enemy) / math.log1p(1000.0),
                max(-1.0, min(1.0, (inbound_my - inbound_enemy) / 500.0)),
                max(-1.0, min(1.0, (ships + inbound_my - inbound_enemy) / 500.0)),
                _nearest_distance(x, y, my_positions) / 100.0,
                _nearest_distance(x, y, enemy_positions) / 100.0,
                _nearest_distance(x, y, neutral_positions) / 100.0,
                1.0 if is_mine and ships > 1 else 0.0,
            ],
            dtype=np.float32,
        )
        planet_mask[i] = 1.0

    fleet_my = 0.0
    fleet_enemy = 0.0
    for f in raw_fleets:
        owner = int(f[1])
        ships = float(f[6])
        if owner == player:
            fleet_my += ships
        else:
            fleet_enemy += ships

    own_total = my_ship_total + fleet_my
    enemy_total = enemy_ship_total + fleet_enemy
    total_known = max(1.0, own_total + enemy_total + neutral_ship_total)
    production_total = max(1.0, my_production + enemy_production)
    non_neutral_planets = max(1, my_planets + enemy_planets)
    globals_ = np.array(
        [
            step / MAX_STEPS,
            angular_velocity / 0.05,
            my_planets / MAX_PLANETS,
            enemy_planets / MAX_PLANETS,
            neutral_planets / MAX_PLANETS,
            own_total / total_known,
            enemy_total / total_known,
            neutral_ship_total / total_known,
            my_production / 50.0,
            enemy_production / 50.0,
            my_production / production_total,
            enemy_production / production_total,
            (my_planets - enemy_planets) / non_neutral_planets,
            (own_total - enemy_total) / total_known,
            fleet_my / total_known,
            fleet_enemy / total_known,
            comet_count / MAX_PLANETS,
            orbiting_count / MAX_PLANETS,
        ],
        dtype=np.float32,
    )

    action_mask = _build_action_mask(obs, raw_planets, player, incoming)

    return EncodedObs(planets=planets, planet_mask=planet_mask, globals=globals_, action_mask=action_mask)


def decode_move(obs, index: int, remaining_ships: list[int] | None = None) -> list[list[float]]:
    decoded = decode_action_index(int(index))
    if decoded is None:
        return []
    source_slot, target_slot, send_bin = decoded
    raw_planets = list(_obs_get(obs, "planets", []) or [])
    if source_slot >= len(raw_planets) or target_slot >= len(raw_planets):
        return []

    player = int(_obs_get(obs, "player", 0))
    src = raw_planets[source_slot]
    tgt = raw_planets[target_slot]
    available = (
        int(remaining_ships[source_slot])
        if remaining_ships is not None and source_slot < len(remaining_ships)
        else int(float(src[5]))
    )
    if int(src[1]) != player or source_slot == target_slot or available <= 1:
        return []

    frac = SEND_FRACTIONS[send_bin]
    ships = max(1, int(available * frac))
    ships = min(ships, available)
    if ships <= 0:
        return []
    path = dir_to_hit(obs, src, tgt, ships)
    if path is None:
        return []
    angle = path[0]
    return [[int(src[0]), angle, int(ships)]]


def decode_moves(obs, action_indices: list[int]) -> list[list[float]]:
    moves: list[list[float]] = []
    remaining = remaining_ships_by_slot(obs)
    for action_index_value in action_indices:
        move = decode_move(obs, action_index_value, remaining)
        if not move:
            if int(action_index_value) <= 0:
                break
            continue
        moves.extend(move)
        source_slot = decode_action_index(int(action_index_value))[0]  # type: ignore[index]
        remaining[source_slot] = max(0, remaining[source_slot] - int(move[0][2]))
    return moves


def encode_move_as_source_target_slots(obs, move: list[float]) -> tuple[int, int] | None:
    raw_planets = list(_obs_get(obs, "planets", []) or [])
    if not move or len(move) < 3:
        return None
    try:
        from_pid = int(move[0])
        angle = float(move[1])
    except (TypeError, ValueError):
        return None

    source_slot = None
    src = None
    for i, planet in enumerate(raw_planets[:MAX_PLANETS]):
        if int(planet[0]) == from_pid:
            source_slot = i
            src = planet
            break
    if source_slot is None or src is None or int(src[5]) <= 1:
        return None

    src_x = float(src[2])
    src_y = float(src[3])
    best_target_slot = None
    best_score = float("inf")
    for ti, tgt in enumerate(raw_planets[:MAX_PLANETS]):
        if ti == source_slot:
            continue
        dx = float(tgt[2]) - src_x
        dy = float(tgt[3]) - src_y
        target_angle = math.atan2(dy, dx)
        angle_diff = abs(math.atan2(math.sin(angle - target_angle), math.cos(angle - target_angle)))
        distance = max(1.0, math.hypot(dx, dy))
        # Apollo and search bots may aim off center for moving planets. This
        # still maps them to the most plausible discrete target in our smaller
        # action space.
        score = angle_diff + 0.003 * distance
        if score < best_score:
            best_score = score
            best_target_slot = ti

    if best_target_slot is None:
        return None

    return source_slot, best_target_slot


def encode_move_as_action_index(obs, move: list[float]) -> int:
    slots = encode_move_as_source_target_slots(obs, move)
    if slots is None:
        return 0
    source_slot, best_target_slot = slots
    raw_planets = list(_obs_get(obs, "planets", []) or [])
    try:
        ships = max(1, int(move[2]))
    except (TypeError, ValueError):
        return 0

    src = raw_planets[source_slot]
    send_fraction = ships / max(1.0, float(src[5]))
    send_bin = min(range(len(SEND_FRACTIONS)), key=lambda i: abs(SEND_FRACTIONS[i] - send_fraction))
    return action_index(source_slot, best_target_slot, send_bin)


def encode_teacher_moves_as_action_index(obs, moves: list[list[float]]) -> int:
    if not moves:
        return 0
    candidates = []
    for move in moves:
        idx = encode_move_as_action_index(obs, move)
        if idx > 0:
            try:
                ships = int(move[2])
            except (TypeError, ValueError):
                ships = 0
            candidates.append((ships, idx))
    if not candidates:
        return 0
    return max(candidates, key=lambda item: item[0])[1]


def game_stats(obs, player: int | None = None) -> GameStats:
    if player is None:
        player = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0) or 0)
    planets = _obs_get(obs, "planets", []) or []
    fleets = _obs_get(obs, "fleets", []) or []

    own_planet_ships = 0.0
    enemy_planet_ships = 0.0
    neutral_ships = 0.0
    own_planets = 0
    enemy_planets = 0
    neutral_planets = 0
    own_production = 0.0
    enemy_production = 0.0

    for p in planets:
        owner = int(p[1])
        ships = float(p[5])
        production = float(p[6])
        if owner == player:
            own_planet_ships += ships
            own_planets += 1
            own_production += production
        elif owner == -1:
            neutral_ships += ships
            neutral_planets += 1
        else:
            enemy_planet_ships += ships
            enemy_planets += 1
            enemy_production += production

    own_fleet_ships = 0.0
    enemy_fleet_ships = 0.0
    for f in fleets:
        if int(f[1]) == player:
            own_fleet_ships += float(f[6])
        else:
            enemy_fleet_ships += float(f[6])

    own_score = own_planet_ships + own_fleet_ships
    enemy_score = enemy_planet_ships + enemy_fleet_ships
    score_total = max(1.0, own_score + enemy_score)
    production_total = max(1.0, own_production + enemy_production)
    planet_total = max(1, own_planets + enemy_planets)
    remaining = max(0, MAX_STEPS - step)

    # A light economic potential helps the critic value captures before their
    # production has fully paid off. It is deliberately relative to enemy eco.
    economy_value = (
        (own_score - enemy_score)
        + 5.0 * (own_planets - enemy_planets)
        + 0.20 * remaining * (own_production - enemy_production)
    )

    return GameStats(
        step=step,
        remaining=remaining,
        own_score=own_score,
        enemy_score=enemy_score,
        neutral_ships=neutral_ships,
        own_planets=own_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        own_production=own_production,
        enemy_production=enemy_production,
        own_fleet_ships=own_fleet_ships,
        enemy_fleet_ships=enemy_fleet_ships,
        score_share=own_score / score_total,
        production_share=own_production / production_total,
        planet_share=own_planets / planet_total,
        score_margin=own_score - enemy_score,
        production_margin=own_production - enemy_production,
        economy_value=economy_value,
    )


def shaped_score(obs, player: int | None = None) -> float:
    stats = game_stats(obs, player)
    return stats.own_score + 8.0 * stats.own_production + 4.0 * stats.own_planets
