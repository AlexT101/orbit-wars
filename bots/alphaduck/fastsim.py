"""Numba-jit fast collision predictor for alphaduck's hot path.

`flatten_state(state)` pre-computes planet positions at every dt in [0, MAX_TIME]
once per turn (planets don't change geometrically across MCTS iterations — only
ship counts and the fleet list do). Then `predict_one_fleet_jit` returns the
(planet_idx, eta) pair a synthetic fleet would hit, using only flat numpy
arrays so it can be njit-compiled.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mine" / "target_predictor" / "train"))
import build_dataset as bd  # for engine constants

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(f=None, **kwargs):
        if callable(f): return f
        return lambda g: g


MAX_TIME = bd.MAX_TIME
SUN_RADIUS = bd.SUN_RADIUS
CENTER_X, CENTER_Y = bd.CENTER
BOARD = bd.BOARD


def flatten_state(state):
    """Returns a dict of numpy arrays describing planet geometry for every dt.

      radius:    (N,) float64
      pos_x:     (MAX_TIME+1, N) float64   -- planet center at offset dt
      pos_y:     (MAX_TIME+1, N) float64
      active:    (MAX_TIME+1, N) bool      -- comet may expire mid-window
      planet_ids:(N,) int32                -- map planet idx -> game pid
    """
    n = len(state["planets"])
    radius = np.zeros(n, dtype=np.float64)
    pos_x = np.zeros((MAX_TIME + 1, n), dtype=np.float64)
    pos_y = np.zeros((MAX_TIME + 1, n), dtype=np.float64)
    active = np.zeros((MAX_TIME + 1, n), dtype=np.bool_)
    pids = np.zeros(n, dtype=np.int32)
    for i, p in enumerate(state["planets"]):
        radius[i] = p["radius"]
        pids[i] = p["id"]
        for dt in range(MAX_TIME + 1):
            pos = bd.planet_pos_at(state, p, dt)
            if pos is None:
                active[dt, i] = False
            else:
                pos_x[dt, i] = pos[0]
                pos_y[dt, i] = pos[1]
                active[dt, i] = True
    return dict(radius=radius, pos_x=pos_x, pos_y=pos_y, active=active, planet_ids=pids)


@njit(cache=True)
def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    d0x = ax - p0x; d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
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


@njit(cache=True)
def _pt_seg_dist(px, py, vx, vy, wx, wy):
    l2 = (vx - wx) * (vx - wx) + (vy - wy) * (vy - wy)
    if l2 < 1e-12:
        dx = px - vx; dy = py - vy
        return math.sqrt(dx * dx + dy * dy)
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    projx = vx + t * (wx - vx)
    projy = vy + t * (wy - vy)
    dx = px - projx; dy = py - projy
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def _predict_one_fleet_jit(radius, pos_x, pos_y, active,
                            fleet_x, fleet_y, fleet_angle, fleet_speed,
                            max_time, board, sun_radius, cx, cy):
    dx = fleet_speed * math.cos(fleet_angle)
    dy = fleet_speed * math.sin(fleet_angle)
    px = fleet_x; py = fleet_y
    n_planets = radius.shape[0]
    for dt in range(1, max_time + 1):
        new_x = px + dx; new_y = py + dy
        for i in range(n_planets):
            if not active[dt - 1, i] or not active[dt, i]:
                continue
            p_old_x = pos_x[dt - 1, i]; p_old_y = pos_y[dt - 1, i]
            p_new_x = pos_x[dt, i]; p_new_y = pos_y[dt, i]
            on_old = (0.0 <= p_old_x <= board) and (0.0 <= p_old_y <= board)
            on_new = (0.0 <= p_new_x <= board) and (0.0 <= p_new_y <= board)
            if not on_old and not on_new:
                continue
            if _swept_pair_hit(px, py, new_x, new_y,
                               p_old_x, p_old_y, p_new_x, p_new_y, radius[i]):
                return i, dt
        # off-board
        if not ((0.0 <= new_x <= board) and (0.0 <= new_y <= board)):
            return -1, 0
        # sun hit
        if _pt_seg_dist(cx, cy, px, py, new_x, new_y) < sun_radius:
            return -1, 0
        px = new_x; py = new_y
    return -1, 0


# Warmup so the first user call doesn't pay the JIT compile cost.
def warmup():
    if not HAVE_NUMBA:
        return
    radius = np.array([3.0], dtype=np.float64)
    pos_x = np.zeros((MAX_TIME + 1, 1), dtype=np.float64)
    pos_y = np.zeros((MAX_TIME + 1, 1), dtype=np.float64)
    active = np.ones((MAX_TIME + 1, 1), dtype=np.bool_)
    _predict_one_fleet_jit(radius, pos_x, pos_y, active,
                            0.0, 0.0, 0.0, 2.0, MAX_TIME, BOARD, SUN_RADIUS, CENTER_X, CENTER_Y)


def predict_one_fleet_fast(flat, fleet_x, fleet_y, fleet_angle, fleet_speed):
    """High-level wrapper that returns (planet_pid_or_None, eta)."""
    pi, eta = _predict_one_fleet_jit(
        flat["radius"], flat["pos_x"], flat["pos_y"], flat["active"],
        fleet_x, fleet_y, fleet_angle, fleet_speed,
        MAX_TIME, BOARD, SUN_RADIUS, CENTER_X, CENTER_Y,
    )
    if pi < 0:
        return None, 0
    return int(flat["planet_ids"][pi]), int(eta)
