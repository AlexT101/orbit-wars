from __future__ import annotations

import copy
import math
from collections import defaultdict
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

# Send options: 4 discrete fractions plus a "resolved + 1" mode that sends
# exactly (target.ships_resolved + 1), i.e. the minimum to capture the target
# after all currently-in-flight fleets have resolved against it. The "resolved"
# bin is index RESOLVED_BIN; the rest are simple fractions of source ships.
SEND_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
RESOLVED_BIN = len(SEND_FRACTIONS)
NUM_SEND_OPTIONS = len(SEND_FRACTIONS) + 1

# Per-planet arrival bins. For each planet we record how many of MY ships and
# how many ENEMY ships are scheduled to arrive within each delta-turn bucket
# (delta = arrival_turn - current_turn). Right-inclusive bins.
ARRIVAL_BIN_EDGES = (2, 5, 10, 20, 50)
NUM_ARRIVAL_BINS = len(ARRIVAL_BIN_EDGES)

# Per-planet feature schema:
#   19 base features  (geometry + ownership + comet + orbiting + ships_resolved)
# + 2 * NUM_ARRIVAL_BINS bins  (mine vs enemy, see ARRIVAL_BIN_EDGES)
PLANET_BASE_FEATURES = 19
PLANET_FEATURES = PLANET_BASE_FEATURES + 2 * NUM_ARRIVAL_BINS
GLOBAL_FEATURES = 9
ACTION_DIM = 1 + MAX_PLANETS * MAX_PLANETS * NUM_SEND_OPTIONS


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
    # Kaggle's interpreter reads obs.step inside the call, then computes
    # new_pos = initial + av * step. obs.step is incremented AFTER the call,
    # so at obs.step=N (visible to us) the planet's recorded position reflects
    # motion av * (N-1). For dt turns into the future, the (N+dt-1)'th call
    # will be the last to overwrite the position, giving av * (N+dt-1).
    angle = angle0 + av * max(0, step + dt - 1)
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


def action_index(source_slot: int, target_slot: int, send_bin: int) -> int:
    per_source = MAX_PLANETS * NUM_SEND_OPTIONS
    return 1 + source_slot * per_source + target_slot * NUM_SEND_OPTIONS + send_bin


def decode_action_index(index: int) -> tuple[int, int, int] | None:
    if index <= 0:
        return None
    raw = index - 1
    send_bin = raw % NUM_SEND_OPTIONS
    raw //= NUM_SEND_OPTIONS
    target_slot = raw % MAX_PLANETS
    source_slot = raw // MAX_PLANETS
    return source_slot, target_slot, send_bin


# ---------------------------------------------------------------------------
# Forward simulation: predict each in-flight fleet's target planet, then
# resolve each planet's state turn-by-turn until all currently-flying fleets
# have arrived. Returns a dict planet_id -> (final_owner, final_ships, turns).
#
# Approximations: trajectory prediction uses a simple straight-line projection
# from each fleet's current (x, y) at current speed (recomputed from ship count
# per orbit_wars_rules.md:46). The first planet whose continuous-collision
# check passes wins. Orbiting planets use _planet_pos_at to step their position
# forward. Combat resolution follows orbit_wars_rules.md:99-108.
# ---------------------------------------------------------------------------


def _predict_fleet_target(
    obs,
    fleet,
    raw_planets,
    max_horizon: int = PATH_MAX_TIME,
) -> tuple[int, int] | None:
    """Return (target_planet_id, arrival_turn) or None if the fleet never hits."""
    x = float(fleet[2])
    y = float(fleet[3])
    angle = float(fleet[4])
    ships = int(fleet[6])
    speed = fleet_speed(ships)
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)

    pos = (x, y)
    for t in range(1, max_horizon + 1):
        new_pos = (pos[0] + vx, pos[1] + vy)
        # Mirror the kaggle interpreter's order exactly:
        # 1. Planet collision (first hit in obs.planets order wins).
        # 2. Out of bounds.
        # 3. Sun (< SUN_RADIUS, strict).
        for p in raw_planets:
            p_old = _planet_pos_at(obs, p, t - 1) or (float(p[2]), float(p[3]))
            p_new = _planet_pos_at(obs, p, t) or (float(p[2]), float(p[3]))
            radius = float(p[4])
            if _swept_pair_hit(pos, new_pos, p_old, p_new, radius):
                return int(p[0]), t
        if not _on_board(new_pos):
            return None
        if _segment_distance(CENTER, CENTER, pos[0], pos[1], new_pos[0], new_pos[1]) < SUN_RADIUS:
            return None
        pos = new_pos
    return None


def _resolve_combat(garrison: float, garrison_owner: int, attacker_groups: dict[int, float]) -> tuple[float, int]:
    """Apply orbit_wars_rules.md:99-108. attacker_groups: owner -> ships summed."""
    if not attacker_groups:
        return garrison, garrison_owner
    sorted_atk = sorted(attacker_groups.items(), key=lambda kv: -kv[1])
    if len(sorted_atk) >= 2 and sorted_atk[0][1] == sorted_atk[1][1]:
        # Tied top attackers — all attacking ships destroyed.
        return garrison, garrison_owner
    top_owner, top_ships = sorted_atk[0]
    survivors = top_ships - (sorted_atk[1][1] if len(sorted_atk) >= 2 else 0.0)
    if survivors <= 0.0:
        return garrison, garrison_owner
    if top_owner == garrison_owner:
        return garrison + survivors, garrison_owner
    # Attacker fights garrison.
    if survivors > garrison:
        return survivors - garrison, top_owner
    if survivors < garrison:
        return garrison - survivors, garrison_owner
    # Exactly equal — mutual annihilation, planet stays with current owner at 0.
    return 0.0, garrison_owner


def predict_arrivals(obs) -> dict[int, list[tuple[int, int, float]]]:
    """For each planet, predict in-flight-fleet arrivals.

    Returns: planet_id -> [(turn_delta, owner, ships), ...]
    Pure Python; uses straight-line projection + swept-pair collision against
    current planet positions (with orbital motion modeled by _planet_pos_at).
    """
    raw_planets = list(_obs_get(obs, "planets", []) or [])
    raw_fleets = list(_obs_get(obs, "fleets", []) or [])
    arrivals: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for f in raw_fleets:
        pred = _predict_fleet_target(obs, f, raw_planets)
        if pred is None:
            continue
        target_pid, t = pred
        arrivals[target_pid].append((t, int(f[1]), float(f[6])))
    return arrivals


def resolve_via_env_rollout(env, max_steps: int = 250) -> dict[int, tuple[int, float, int]]:
    """Ground-truth fleet resolution by deep-copying the kaggle env and
    stepping with empty actions until no fleets remain.

    Returns: planet_id -> (final_owner, final_ships, turns_to_resolve).
    Correct by construction (it IS the env); ~4-5× slower than the Python sim.
    """
    snap = copy.deepcopy(env)
    steps = 0
    while steps < max_steps:
        if getattr(snap, "done", False):
            break
        obs = snap.state[0].observation
        fleets = obs.get("fleets", []) or []
        if not fleets:
            break
        snap.step([[], []])
        steps += 1
    obs = snap.state[0].observation
    planets = obs.get("planets", []) or []
    return {int(p[0]): (int(p[1]), float(p[5]), steps) for p in planets}


def _resolve_via_interpreter(obs, max_steps: int = 250) -> dict[int, tuple[int, float, int]]:
    """Obs-only resolver. Builds a minimal kaggle state from the obs and calls
    `kaggle_environments.envs.orbit_wars.orbit_wars.interpreter` directly,
    forward-stepping with empty actions until no fleets remain.

    This uses the SAME physics code as the live env, so it's bit-exact
    against env-rollout EXCEPT for one quirk: comet spawns at step ∈
    {50, 150, 250, 350, 450} pull from `env.info["seed"]`; we don't have
    the original seed here, so newly-spawned comets during the rollout
    use seed=0 and won't match reality. New comets only affect resolution
    if they happen to block an in-flight fleet's trajectory — extremely
    rare in practice. For everything else this is exact.
    """
    from types import SimpleNamespace
    import copy

    from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter

    obs0 = SimpleNamespace(
        step=int(_obs_get(obs, "step", 0) or 0),
        planets=copy.deepcopy(_obs_get(obs, "planets", []) or []),
        initial_planets=copy.deepcopy(_obs_get(obs, "initial_planets", []) or []),
        fleets=copy.deepcopy(_obs_get(obs, "fleets", []) or []),
        next_fleet_id=int(_obs_get(obs, "next_fleet_id", 0) or 0),
        comets=copy.deepcopy(_obs_get(obs, "comets", []) or []),
        comet_planet_ids=list(_obs_get(obs, "comet_planet_ids", []) or []),
        angular_velocity=float(_obs_get(obs, "angular_velocity", 0.0) or 0.0),
        player=0,
    )
    state = [
        SimpleNamespace(observation=obs0, action=[], status="ACTIVE", reward=0),
        SimpleNamespace(observation=SimpleNamespace(player=1),
                        action=[], status="ACTIVE", reward=0),
    ]
    env_ns = SimpleNamespace(
        configuration=SimpleNamespace(
            episodeSteps=MAX_STEPS,
            actTimeout=1,
            shipSpeed=SHIP_SPEED_MAX,
            sunRadius=SUN_RADIUS,
            boardSize=BOARD_SIZE,
            cometSpeed=4.0,
            seed=None,
        ),
        done=False,
        info={"seed": 0},  # comet-spawn RNG; won't match reality but rarely matters
    )

    steps = 0
    while steps < max_steps and obs0.fleets:
        interpreter(state, env_ns)
        # Match kaggle wrapper: bump obs.step AFTER each interpreter call.
        obs0.step += 1
        steps += 1
    return {int(p[0]): (int(p[1]), float(p[5]), steps) for p in obs0.planets}


def resolve_all_planets(obs, env=None) -> dict[int, tuple[int, float, int]]:
    """Resolve all planets to their post-in-flight state.

    Resolution priority (all use kaggle's own physics — no reimplementation):
    1. `obs["_resolved"]` cache: pre-computed by OrbitWarsDuelEnv once per
       step and attached to both players' obs. Means opponent and us see
       identical resolved values (symmetric). Zero cost per call.
    2. `env` handle: deep-copy + env.step() rollout. Exact.
    3. Obs-only fallback: build a minimal state and call kaggle's interpreter
       directly. Exact EXCEPT for comet-spawn RNG (the real seed is scrubbed
       from obs by kaggle, so newly-spawned comets during the rollout won't
       match). Only used by the exported Kaggle bot.
    """
    cached = obs.get("_resolved") if isinstance(obs, dict) else None
    if cached is not None:
        return cached
    if env is not None:
        return resolve_via_env_rollout(env)
    return _resolve_via_interpreter(obs)


def arrival_bins_for_planet(
    schedule: list[tuple[int, int, float]],
    me: int,
) -> tuple[list[float], list[float]]:
    """Bucket arrivals at one planet into MY and ENEMY ship counts per bin.

    Bin edges are ARRIVAL_BIN_EDGES (right-inclusive). Arrivals beyond the
    last edge (delta > 50) are discarded — they're rare and low-signal.
    Returns: (mine_by_bin, enemy_by_bin), each length NUM_ARRIVAL_BINS.
    """
    mine = [0.0] * NUM_ARRIVAL_BINS
    enemy = [0.0] * NUM_ARRIVAL_BINS
    for t, owner, ships in schedule:
        # Find first bin edge >= t.
        bin_idx = -1
        for i, edge in enumerate(ARRIVAL_BIN_EDGES):
            if t <= edge:
                bin_idx = i
                break
        if bin_idx < 0:
            continue
        if owner == me:
            mine[bin_idx] += ships
        elif owner != -1:
            enemy[bin_idx] += ships
    return mine, enemy


def encode_obs(obs, env=None) -> EncodedObs:
    player = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    raw_planets = list(_obs_get(obs, "planets", []) or [])
    raw_fleets = list(_obs_get(obs, "fleets", []) or [])
    comet_ids = set(int(pid) for pid in (_obs_get(obs, "comet_planet_ids", []) or []))
    angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)

    planets = np.zeros((MAX_PLANETS, PLANET_FEATURES), dtype=np.float32)
    planet_mask = np.zeros(MAX_PLANETS, dtype=np.float32)

    # Forward-sim resolved state per planet. Used as a feature (slot 17) and
    # for the "send resolved+1" action bin. If `env` is provided we use the
    # ground-truth env-rollout; otherwise we use the Python sim fallback.
    resolved = resolve_all_planets(obs, env=env)
    # Bin features use the Python arrival predictor; cheap and the same
    # predictor that drove the Python resolver, so it stays consistent.
    arrivals = predict_arrivals(obs)

    my_ship_total = 0.0
    enemy_ship_total = 0.0
    neutral_ship_total = 0.0
    my_planets = 0
    enemy_planets = 0
    neutral_planets = 0
    my_production = 0.0
    enemy_production = 0.0

    n = min(len(raw_planets), MAX_PLANETS)
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

        # ships_resolved: signed log-normalized post-resolution garrison.
        # Positive if the resolved owner is me, negative if enemy, 0 if neutral
        # or no in-flight fleets target this planet (then equals current state).
        resolved_owner, resolved_ships, _ = resolved.get(int(pid), (owner, ships, 0))
        resolved_norm = math.log1p(max(0.0, resolved_ships)) / math.log1p(1000.0)
        if resolved_owner == player:
            resolved_signed = resolved_norm
        elif resolved_owner == -1:
            resolved_signed = 0.0
        else:
            resolved_signed = -resolved_norm

        # Arrival bins for this planet. Two channels (mine, enemy) × 5 bins.
        # Each value is log-normalized ship count so big arrivals don't dominate.
        sched = arrivals.get(int(pid), [])
        mine_by_bin, enemy_by_bin = arrival_bins_for_planet(sched, me=player)
        bin_norm = math.log1p(1000.0)
        mine_bins_norm = [math.log1p(v) / bin_norm for v in mine_by_bin]
        enemy_bins_norm = [math.log1p(v) / bin_norm for v in enemy_by_bin]

        base = [
            x / BOARD_SIZE,
            y / BOARD_SIZE,
            dx / BOARD_SIZE,
            dy / BOARD_SIZE,
            orbital_radius / 70.7107,
            math.sin(math.atan2(dy, dx)),
            math.cos(math.atan2(dy, dx)),
            radius / 4.0,
            math.log1p(ships) / math.log1p(1000.0),
            production / 5.0,
            is_mine,
            is_enemy,
            is_neutral,
            is_comet,
            is_orbiting,
            1.0 if x < CENTER else 0.0,
            1.0 if y < CENTER else 0.0,
            resolved_signed,
            1.0,
        ]
        planets[i] = np.array(base + mine_bins_norm + enemy_bins_norm, dtype=np.float32)
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

    total_known = max(1.0, my_ship_total + enemy_ship_total + neutral_ship_total + fleet_my + fleet_enemy)
    globals_ = np.array(
        [
            step / MAX_STEPS,
            angular_velocity / 0.05,
            my_planets / MAX_PLANETS,
            enemy_planets / MAX_PLANETS,
            neutral_planets / MAX_PLANETS,
            (my_ship_total + fleet_my) / total_known,
            (enemy_ship_total + fleet_enemy) / total_known,
            my_production / 50.0,
            enemy_production / 50.0,
        ],
        dtype=np.float32,
    )

    action_mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    action_mask[0] = True
    for si in range(n):
        src = raw_planets[si]
        if int(src[1]) != player or int(src[5]) <= 1:
            continue
        for ti in range(n):
            if ti == si:
                continue
            tgt = raw_planets[ti]
            for bi, frac in enumerate(SEND_FRACTIONS):
                ships = max(1, int(float(src[5]) * frac))
                if ships < int(src[5]):
                    action_mask[action_index(si, ti, bi)] = True
                elif int(src[5]) >= 2:
                    action_mask[action_index(si, ti, bi)] = True
            # "Resolved + 1" bin: legal only if source has at least
            # target.ships_resolved + 1 ships AND that amount is < src.ships
            # (must leave at least one behind, matching the fraction logic).
            _, resolved_ships, _ = resolved.get(int(tgt[0]), (int(tgt[1]), float(tgt[5]), 0))
            needed = max(1, int(resolved_ships) + 1)
            if needed < int(src[5]):
                action_mask[action_index(si, ti, RESOLVED_BIN)] = True

    return EncodedObs(planets=planets, planet_mask=planet_mask, globals=globals_, action_mask=action_mask)


def decode_move(obs, index: int, env=None) -> list[list[float]]:
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
    if int(src[1]) != player or source_slot == target_slot or int(src[5]) <= 1:
        return []

    if send_bin == RESOLVED_BIN:
        # Send exactly (target.ships_resolved + 1).
        resolved = resolve_all_planets(obs, env=env)
        _, resolved_ships, _ = resolved.get(int(tgt[0]), (int(tgt[1]), float(tgt[5]), 0))
        ships = max(1, int(resolved_ships) + 1)
    else:
        frac = SEND_FRACTIONS[send_bin]
        ships = max(1, int(float(src[5]) * frac))
    ships = min(ships, int(src[5]))
    if ships <= 0:
        return []
    path = dir_to_hit(obs, src, tgt, ships)
    if path is None and path_hits_sun(src, tgt):
        return []
    angle = path[0] if path is not None else math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
    return [[int(src[0]), angle, int(ships)]]


def encode_move_as_action_index(obs, move: list[float]) -> int:
    raw_planets = list(_obs_get(obs, "planets", []) or [])
    if not move or len(move) < 3:
        return 0
    try:
        from_pid = int(move[0])
        angle = float(move[1])
        ships = max(1, int(move[2]))
    except (TypeError, ValueError):
        return 0

    source_slot = None
    src = None
    for i, planet in enumerate(raw_planets[:MAX_PLANETS]):
        if int(planet[0]) == from_pid:
            source_slot = i
            src = planet
            break
    if source_slot is None or src is None or int(src[5]) <= 1:
        return 0

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
        return 0

    # BC encoder: only map to the discrete fraction bins (never to the
    # resolved+1 bin) — teachers don't know about that option, so emitting it
    # as a label would corrupt training.
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
