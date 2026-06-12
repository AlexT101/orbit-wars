"""Angle-range aim solver — prototype.

For each candidate flight time t in 1..MAX_T, compute the set of fleet launch
angles that:
  (a) land within target_radius of target's position at turn t
  (b) have NOT been intercepted by any blocker (sun, planet, comet) at any
      earlier turn t' < t along the same straight-line trajectory.

The set is represented as a sorted union of disjoint (start, end) intervals
on (-π, π] (with wraparound at ±π handled by splitting).

Eta semantics match the engine: if fleet launches at turn N and lands during
the move phase of turn N+t, eta = t, and the target has produced ships t times
since launch (matches "(target generates ships)" happening at the start of
every subsequent turn).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mine" / "target_predictor" / "train"))
import build_dataset as bd

CENTER = bd.CENTER
SUN_RADIUS = bd.SUN_RADIUS
BOARD = bd.BOARD
TWO_PI = 2.0 * math.pi
LAUNCH_CLEARANCE = 0.1   # ENGINE constant (bots/alphaduck/aim_native/src/constants.rs)
                         # Fleet spawns at src + (radius + LAUNCH_CLEARANCE)*u, then
                         # moves +v*u per tick.
N_ANGLE_PROBES = 1024    # angular resolution for swept scans

# ---------------------------------------------------------------------------
# Angle range arithmetic
# ---------------------------------------------------------------------------

def wrap(a: float) -> float:
    """Wrap angle into (-π, π]."""
    a = math.fmod(a + math.pi, TWO_PI)
    if a <= 0.0:
        a += TWO_PI
    return a - math.pi


def _interval_split(a: float, b: float) -> list[tuple[float, float]]:
    """Normalize a single arc [a, b] (with possible wraparound) to a list of
    one or two non-wrapping intervals on (-π, π]."""
    a = wrap(a); b = wrap(b)
    if a <= b:
        return [(a, b)]
    # wraps around: [a, π] ∪ [-π, b]
    return [(a, math.pi), (-math.pi, b)]


def union(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals. Input is non-wrapping, in (-π, π]."""
    if not ranges:
        return []
    rs = sorted(ranges)
    out = [rs[0]]
    for a, b in rs[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def subtract(target: list[tuple[float, float]],
             forbidden: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """target - forbidden, both lists of non-overlapping non-wrapping arcs."""
    if not forbidden:
        return list(target)
    out = []
    for ta, tb in target:
        cur = [(ta, tb)]
        for fa, fb in forbidden:
            nxt = []
            for ca, cb in cur:
                # Disjoint?
                if fb < ca or fa > cb:
                    nxt.append((ca, cb)); continue
                # forbidden covers cur
                if fa <= ca and fb >= cb:
                    continue
                # split
                if fa > ca:
                    nxt.append((ca, fa))
                if fb < cb:
                    nxt.append((fb, cb))
            cur = nxt
            if not cur:
                break
        out.extend(cur)
    return out


# ---------------------------------------------------------------------------
# Angular cone math (no sweep, end-of-turn endpoint check first)
# ---------------------------------------------------------------------------

def cone_for_intercept(launch_pos: tuple[float, float],
                       v: float, t: int,
                       tgt_pos: tuple[float, float],
                       tgt_radius: float) -> list[tuple[float, float]]:
    """Angles θ s.t. (launch + t*v*(cos θ, sin θ)) is within tgt_radius of tgt_pos."""
    dx = tgt_pos[0] - launch_pos[0]
    dy = tgt_pos[1] - launch_pos[1]
    D = math.hypot(dx, dy)
    v_t = v * t
    # If fleet at distance v_t can reach within tgt_radius of tgt at D:
    # need |D - v_t| <= tgt_radius (collinear); for off-axis it's law-of-cosines.
    # cos(off) >= (v_t² + D² - r²)/(2·v_t·D)
    if v_t <= 0:
        return []
    if D <= tgt_radius:
        return [(-math.pi, math.pi)]  # we'd already be inside (rare)
    num = v_t * v_t + D * D - tgt_radius * tgt_radius
    den = 2.0 * v_t * D
    cos_off = num / den if den > 0 else 2.0
    if cos_off > 1.0:
        return []
    if cos_off < -1.0:
        return [(-math.pi, math.pi)]
    half = math.acos(cos_off)
    bearing = math.atan2(dy, dx)
    return _interval_split(bearing - half, bearing + half)


def shadow_static(launch_pos, obs_pos, obs_radius) -> list[tuple[float, float]]:
    """Angles whose ray from launch_pos comes within obs_radius of obs_pos at
    any point — i.e. the obstacle blocks the ray forever (it's static)."""
    dx = obs_pos[0] - launch_pos[0]
    dy = obs_pos[1] - launch_pos[1]
    D = math.hypot(dx, dy)
    if D <= obs_radius:
        return [(-math.pi, math.pi)]
    half = math.asin(min(obs_radius / D, 1.0))
    bearing = math.atan2(dy, dx)
    return _interval_split(bearing - half, bearing + half)


def shadow_at_turn(launch_pos, v, t, obs_pos_at_t, obs_radius) -> list[tuple[float, float]]:
    """Angles where fleet's end-of-turn-t position lies inside obs (at its t-position).
    Equivalent geometry to cone_for_intercept; same maths."""
    return cone_for_intercept(launch_pos, v, t, obs_pos_at_t, obs_radius)


# ---------------------------------------------------------------------------
# Swept hit (matches engine.swept_pair_hit): both fleet and target move during
# the same turn t. For a given launch angle θ, the fleet sweeps from
# launch+(t-1)v·u to launch+tv·u while the obstacle sweeps from p0 to p1.
# ---------------------------------------------------------------------------

def _swept_hit_one(A: tuple[float, float], B: tuple[float, float],
                   p0: tuple[float, float], p1: tuple[float, float],
                   r: float) -> bool:
    """Direct port of engine swept_pair_hit: A,B = fleet at start/end of turn;
    p0,p1 = obstacle at start/end of turn; r = collision radius."""
    d0x = A[0] - p0[0]; d0y = A[1] - p0[1]
    dvx = (B[0] - A[0]) - (p1[0] - p0[0])
    dvy = (B[1] - A[1]) - (p1[1] - p0[1])
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


def _arcs_from_hit_array(hits: list[bool]) -> list[tuple[float, float]]:
    """Convert a boolean array indexed by sampled angles in [-π, π) into a
    list of (start, end) arcs in (-π, π], handling wraparound."""
    n = len(hits)
    if not any(hits):
        return []
    if all(hits):
        return [(-math.pi, math.pi)]
    step = TWO_PI / n
    arcs = []
    i = 0
    # Pick a starting index that's False so we don't split a true-arc at the seam.
    start_idx = next(i for i in range(n) if not hits[i])
    in_arc = False
    arc_start = 0.0
    for k in range(n + 1):
        idx = (start_idx + k) % n
        theta = -math.pi + idx * step
        if hits[idx] and not in_arc:
            arc_start = theta; in_arc = True
        elif not hits[idx] and in_arc:
            arc_end = theta
            # Handle wrap: if arc_end < arc_start, split or wrap
            if arc_end < arc_start:
                arcs.append((arc_start, math.pi))
                arcs.append((-math.pi, arc_end))
            else:
                arcs.append((arc_start, arc_end))
            in_arc = False
    return arcs


def swept_arc_for_target(src_center, src_radius, v, t, p0_tgt, p1_tgt, r,
                         n_probes: int = N_ANGLE_PROBES) -> list[tuple[float, float]]:
    """Numerically sweep angles, return arcs where fleet's swept path during
    relative turn t intersects target's swept path.

    Engine convention: fleet spawns at src + (radius + v)*u, so at the START
    of relative-turn t it's at src + (radius + t*v)*u, and at the END at
    src + (radius + (t+1)*v)*u. Target moves from p0 to p1 during the same
    interval.

    `t = 1` is the first move after launch (one ship-generation before).
    """
    hits = [False] * n_probes
    step = TWO_PI / n_probes
    # Engine spawn: src + (radius + LAUNCH_CLEARANCE)*u  (this IS the position
    # at end of the launch turn — engine spawn is BEFORE the move). So after
    # one move the fleet is at src + (radius + LAUNCH_CLEARANCE + v)*u.
    # For relative-turn t (counting moves AFTER spawn, 1 = first move):
    #   start-of-turn-t position = src + (radius + LAUNCH_CLEARANCE + (t-1)*v)*u
    #   end-of-turn-t   position = src + (radius + LAUNCH_CLEARANCE + t*v)*u
    base = src_radius + LAUNCH_CLEARANCE
    for i in range(n_probes):
        theta = -math.pi + i * step
        u = (math.cos(theta), math.sin(theta))
        a_off = base + (t - 1) * v
        b_off = base + t * v
        A = (src_center[0] + a_off * u[0], src_center[1] + a_off * u[1])
        B = (src_center[0] + b_off * u[0], src_center[1] + b_off * u[1])
        hits[i] = _swept_hit_one(A, B, p0_tgt, p1_tgt, r)
    return _arcs_from_hit_array(hits)


# ---------------------------------------------------------------------------
# Main aim
# ---------------------------------------------------------------------------

MAX_T_AIM = 150


def aim_eta_range(state: dict, src_pid: int, tgt_pid: int, ships: int,
                  max_t: int = MAX_T_AIM) -> tuple[int, float] | None:
    """Return (eta, angle) for the fastest reachable launch from src to tgt,
    or None if unreachable within max_t turns. Doesn't run apollo's cone scan
    — pure analytic angle-range subtraction."""
    pid_to_planet = {p["id"]: p for p in state["planets"]}
    src = pid_to_planet.get(src_pid); tgt = pid_to_planet.get(tgt_pid)
    if src is None or tgt is None:
        return None
    v = max(bd.fleet_speed(ships), 1.0)
    launch_off = src["radius"] + LAUNCH_CLEARANCE
    # Static blockers: sun, stationary planets (non-orbital, non-comet)
    static_obstacles = []
    static_obstacles.append((CENTER, SUN_RADIUS))
    for p in state["planets"]:
        if p["id"] == src_pid or p["id"] == tgt_pid:
            continue
        if p.get("is_orbiting") or p.get("is_comet"):
            continue
        static_obstacles.append(((p["x"], p["y"]), p["radius"]))
    moving_planets = []
    for p in state["planets"]:
        if p["id"] == src_pid or p["id"] == tgt_pid:
            continue
        if p.get("is_orbiting") or p.get("is_comet"):
            moving_planets.append(p)
    static_shadows: list[tuple[float, float]] = []
    # Need launch_pos to be src's center + LAUNCH_CLEARANCE in direction of angle —
    # but our cone math assumes a single launch_pos. Approximate by using src center
    # (the geometry only shifts by ~launch_off which is small vs distances).
    launch_pos = (src["x"], src["y"])
    for obs_pos, obs_r in static_obstacles:
        static_shadows.extend(shadow_static(launch_pos, obs_pos, obs_r))
    blocked = union(static_shadows)

    src_center = (src["x"], src["y"])
    src_r = src["radius"]
    for t in range(1, max_t + 1):
        # 1) target SWEPT arcs for turn t (both fleet & target move during turn)
        tgt_p0 = bd.planet_pos_at(state, tgt, t - 1)
        tgt_p1 = bd.planet_pos_at(state, tgt, t)
        if tgt_p0 is None or tgt_p1 is None:
            continue
        target_arcs = swept_arc_for_target(src_center, src_r, v, t, tgt_p0, tgt_p1, tgt["radius"])
        # 2) accumulate moving-obstacle SWEPT shadows for this turn
        for p in moving_planets:
            p0 = bd.planet_pos_at(state, p, t - 1)
            p1 = bd.planet_pos_at(state, p, t)
            if p0 is None or p1 is None:
                continue
            blocked = union(blocked + swept_arc_for_target(src_center, src_r, v, t, p0, p1, p["radius"]))
        # 3) test
        valid = subtract(target_arcs, blocked)
        if valid:
            a, b = valid[0]
            return (t, 0.5 * (a + b))
        # 4) carry forward: at next turn we'd have already hit target if we used this t's arc
        blocked = union(blocked + target_arcs)
    return None


# ---------------------------------------------------------------------------
# Sanity / parity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, zipfile, glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", type=int, default=77828262)
    ap.add_argument("--turn", type=int, default=74)
    ap.add_argument("--src", type=int, default=19)
    ap.add_argument("--tgt", type=int, default=30)
    ap.add_argument("--ships", type=int, default=70)
    args = ap.parse_args()

    zps = sorted(glob.glob("/tmp/orbit_days/*.zip"))
    for zp in zps:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"): continue
                if int(name.split(".")[0]) != args.game_id: continue
                g = json.loads(zf.read(name))
                obs = g["steps"][args.turn][0]["observation"]
                state = bd.parse_state(obs)
                res = aim_eta_range(state, args.src, args.tgt, args.ships)
                print(f"range_aim says: {res}")
                # Compare with engine sim
                src = next(p for p in state["planets"] if p["id"] == args.src)
                speed = max(bd.fleet_speed(args.ships), 1.0)
                # Use the angle range_aim suggests
                if res is not None:
                    _, ang = res
                    lx = src["x"] + (src["radius"] + speed) * math.cos(ang)
                    ly = src["y"] + (src["radius"] + speed) * math.sin(ang)
                    fleet = {"id": -1, "owner": 0, "x": lx, "y": ly, "angle": ang, "ships": args.ships}
                    pred = bd.predict_fleet_collision(state, fleet)
                    print(f"  engine sim with that angle: hits {pred[0] if pred else None}, eta={pred[1] if pred else None}")
                break
            break
