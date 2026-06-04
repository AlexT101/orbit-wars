"""Fixed-shape padded state pytree for the JAX engine.

Every game in the batch has the same fixed-size arrays. Masks distinguish
real entries from padding. The buffers are sized to comfortably hold the
worst case observed in `generate_planets` (10 groups x 4 = 40 planets,
plus 5 comet spawns x 4 = 20 comet-planets => 60 < 64) and a generous
fleet cap.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


# Hard caps on padded dimensions. These are global so callers (e.g. action
# encoders) can build matching tensors without importing the state pytree.
MAX_PLANETS = 64
MAX_FLEETS = 1024
MAX_COMET_GROUPS = 5  # COMET_SPAWN_STEPS = [50,150,250,350,450]
MAX_COMET_PATH_LEN = 48  # visible is 5..40; +1 off-board placeholder
MAX_ACTIONS_PER_PLAYER = 64
NUM_PLAYERS_PAD = 4  # always padded to 4; mask out slots 2,3 for 2p


# Planet columns: [id, owner, x, y, radius, ships, prod]
PLANET_COLS = 7
# Fleet columns: [id, owner, x, y, angle, from_planet_id, ships]
FLEET_COLS = 7


class BatchState(NamedTuple):
    """Batched, padded engine state. All leading dims are batch B."""

    # -- planets ---------------------------------------------------------
    planets: jnp.ndarray  # (B, MAX_PLANETS, 7) float64
    planet_mask: jnp.ndarray  # (B, MAX_PLANETS) bool — slot in use
    initial_planets: jnp.ndarray  # (B, MAX_PLANETS, 7) float64
    initial_mask: jnp.ndarray  # (B, MAX_PLANETS) bool

    # -- fleets ----------------------------------------------------------
    fleets: jnp.ndarray  # (B, MAX_FLEETS, 7) float64
    fleet_mask: jnp.ndarray  # (B, MAX_FLEETS) bool — slot in use
    next_fleet_id: jnp.ndarray  # (B,) int32

    # -- comets ----------------------------------------------------------
    # All 5 spawn groups are pre-computed at reset; activate at spawn step.
    comet_paths: jnp.ndarray  # (B, MAX_COMET_GROUPS, 4, MAX_COMET_PATH_LEN, 2) float64
    comet_path_lens: jnp.ndarray  # (B, MAX_COMET_GROUPS, 4) int32 — per-quadrant length
    comet_path_index: jnp.ndarray  # (B, MAX_COMET_GROUPS) int32 — -1 until activated
    comet_planet_ids: jnp.ndarray  # (B, MAX_COMET_GROUPS, 4) int32 — assigned ids
    comet_ships_init: jnp.ndarray  # (B, MAX_COMET_GROUPS) int32 — ships at spawn
    comet_spawn_step: jnp.ndarray  # (B, MAX_COMET_GROUPS) int32 — step it appears
    comet_group_valid: jnp.ndarray  # (B, MAX_COMET_GROUPS) bool — pre-gen succeeded
    comet_group_active: jnp.ndarray  # (B, MAX_COMET_GROUPS) bool — currently in play

    # -- per-game scalars ------------------------------------------------
    step: jnp.ndarray  # (B,) int32
    angular_velocity: jnp.ndarray  # (B,) float64
    done: jnp.ndarray  # (B,) bool
    rewards: jnp.ndarray  # (B, NUM_PLAYERS_PAD) int32 — set on terminate
    num_players: jnp.ndarray  # (B,) int32 — 2 or 4
    episode_steps: jnp.ndarray  # (B,) int32 — config.episodeSteps
    ship_speed: jnp.ndarray  # (B,) float64
    seed: jnp.ndarray  # (B,) int32 — for debugging only
    n_initial: jnp.ndarray  # (B,) int32 — # of original planets; comet slots = [n_initial..n_initial+3]


# Column indices (so callers can pick rows without magic numbers).
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)
