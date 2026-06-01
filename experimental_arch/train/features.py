from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from orbit_wars_model import encode_obs

from constants import (
    ACTION_CHOICES_PER_SOURCE,
    ACTION_DIMS,
    ACTION_TENSOR_SHAPE,
    ACTIONS_DIM,
    GLOBAL_DIM,
    NOOP_CHOICE,
    PLANET_SLOTS,
    POLICY_MASK_SHAPE,
    PRESENCE_SHAPE,
    SEND_ALL_ACTION,
    TARGET_CHOICES,
    TOKEN_DIM,
    TOKEN_SHAPE,
    TURN_SHAPE,
)


def observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "globals": spaces.Box(-np.inf, np.inf, shape=(GLOBAL_DIM,), dtype=np.float32),
            "tokens": spaces.Box(-np.inf, np.inf, shape=TOKEN_SHAPE, dtype=np.float32),
            "presence": spaces.Box(0.0, 1.0, shape=PRESENCE_SHAPE, dtype=np.float32),
            "turns": spaces.Box(0.0, np.inf, shape=TURN_SHAPE, dtype=np.float32),
            "reachable_mask": spaces.Box(0, 1, shape=TURN_SHAPE, dtype=np.uint8),
            "valid_actions_mask": spaces.Box(0, 1, shape=POLICY_MASK_SHAPE, dtype=np.uint8),
        }
    )


def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(np.array(ACTION_DIMS, dtype=np.int64))


def policy_action_mask(feat: dict[str, Any]) -> np.ndarray:
    raw = feat["mask"].reshape(ACTION_TENSOR_SHAPE).astype(np.uint8)
    out = np.zeros(POLICY_MASK_SHAPE, dtype=np.uint8)
    target_mask = raw[:, :, SEND_ALL_ACTION]
    has_launch = target_mask.any(axis=1)
    out[:, 0] = raw[:, 0, NOOP_CHOICE]
    out[:, 1] = has_launch.astype(np.uint8)
    out[:, 2:] = target_mask

    # MultiDiscrete needs each categorical to have at least one valid choice.
    # If a source cannot launch, the target choice is ignored by decode_action,
    # so slot 0 is a harmless placeholder.
    out[~has_launch, 2] = 1
    return out


def encode_features(obs: dict[str, Any], player: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feat = encode_obs(obs, player)
    model_obs = {
        "globals": feat["globals"].astype(np.float32),
        "tokens": feat["tokens"].reshape(TOKEN_SHAPE).astype(np.float32),
        "presence": feat["presence"].reshape(PRESENCE_SHAPE).astype(np.float32),
        "turns": feat["turns"].reshape(TURN_SHAPE).astype(np.float32),
        "reachable_mask": feat["reachable_mask"].reshape(TURN_SHAPE).astype(np.uint8),
        "valid_actions_mask": policy_action_mask(feat),
    }
    return model_obs, feat


def flat_action_mask(feat: dict[str, Any]) -> np.ndarray:
    return policy_action_mask(feat).astype(bool).ravel()


def decode_action(feat: dict[str, Any], action: np.ndarray) -> list[list[float]]:
    moves: list[list[float]] = []
    planet_ids = feat["planet_ids"]
    angles = feat["angles"]
    ship_counts = feat["ship_counts"]
    mask = feat["mask"]
    for source_slot, pair in enumerate(np.asarray(action, dtype=np.int64).reshape(PLANET_SLOTS, 2)):
        launch = int(pair[0])
        if launch == NOOP_CHOICE:
            continue
        target_slot = int(pair[1])
        if target_slot < 0 or target_slot >= TARGET_CHOICES:
            continue
        idx = (source_slot * PLANET_SLOTS + target_slot) * ACTIONS_DIM + SEND_ALL_ACTION
        if idx < 0 or idx >= mask.shape[0] or not mask[idx]:
            continue
        source_id = int(planet_ids[source_slot])
        ships = int(ship_counts[idx])
        if source_id < 0 or ships <= 0:
            continue
        moves.append([source_id, float(angles[idx]), ships])
    return moves
