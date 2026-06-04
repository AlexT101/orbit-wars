"""Fixed-shape action encoding for the JAX step.

Each game's per-step action is a padded tensor `(NUM_PLAYERS_PAD,
MAX_ACTIONS_PER_PLAYER, 3)` of `[from_planet_id, angle, ships]` plus a
matching mask. Empty/invalid slots are masked off and contribute no
fleet launch.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .state import MAX_ACTIONS_PER_PLAYER, NUM_PLAYERS_PAD


class ActionBatch(NamedTuple):
    """Batched padded actions.

    moves: (B, NUM_PLAYERS_PAD, MAX_ACTIONS_PER_PLAYER, 3) float64
           columns: [from_planet_id, angle_radians, ships]
    mask : (B, NUM_PLAYERS_PAD, MAX_ACTIONS_PER_PLAYER) bool
    """

    moves: jnp.ndarray
    mask: jnp.ndarray


def encode_actions(joint_actions: list[list[list]],
                   num_players: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert one game's joint action list to (moves, mask) numpy arrays.

    `joint_actions` is a list of `num_players` per-player action lists, as
    used by `engine_parity_checker.engine.JointAction`. Missing player
    slots (2p game) are padded with empty action lists.
    """
    moves = np.zeros((NUM_PLAYERS_PAD, MAX_ACTIONS_PER_PLAYER, 3), dtype=np.float64)
    mask = np.zeros((NUM_PLAYERS_PAD, MAX_ACTIONS_PER_PLAYER), dtype=bool)
    for p in range(num_players):
        action = joint_actions[p] if p < len(joint_actions) else []
        if not isinstance(action, list):
            continue
        slot = 0
        for move in action:
            if slot >= MAX_ACTIONS_PER_PLAYER:
                break
            if not isinstance(move, list) or len(move) != 3:
                continue
            from_id, angle, ships = move
            try:
                ships_i = int(ships)
            except (TypeError, ValueError):
                continue
            if ships_i <= 0:
                continue
            moves[p, slot, 0] = float(from_id)
            moves[p, slot, 1] = float(angle)
            moves[p, slot, 2] = float(ships_i)
            mask[p, slot] = True
            slot += 1
    return moves, mask


def encode_action_batch(per_game_actions: list[list[list[list]]],
                        num_players: int) -> ActionBatch:
    """Stack per-game joint actions into a batched ActionBatch."""
    moves_list = []
    mask_list = []
    for ja in per_game_actions:
        m, k = encode_actions(ja, num_players)
        moves_list.append(m)
        mask_list.append(k)
    return ActionBatch(
        moves=jnp.asarray(np.stack(moves_list, 0)),
        mask=jnp.asarray(np.stack(mask_list, 0)),
    )
