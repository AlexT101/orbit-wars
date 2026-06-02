from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from orbit_wars_model import encode_obs as _rust_encode_obs

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

MAX_PLANETS = PLANET_SLOTS
PLANET_FEATURES = TOKEN_DIM
GLOBAL_FEATURES = GLOBAL_DIM
DISCRETE_ACTION_DIM = 1 + PLANET_SLOTS * PLANET_SLOTS
ACTION_DIM = DISCRETE_ACTION_DIM
SEND_FRACTIONS = (1.00,)


@dataclass(frozen=True)
class EncodedObs:
    planets: np.ndarray
    planet_mask: np.ndarray
    globals: np.ndarray
    action_mask: np.ndarray


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
    feat = _rust_encode_obs(obs, player)
    model_obs = {
        "globals": feat["globals"].astype(np.float32),
        "tokens": feat["tokens"].reshape(TOKEN_SHAPE).astype(np.float32),
        "presence": feat["presence"].reshape(PRESENCE_SHAPE).astype(np.float32),
        "turns": feat["turns"].reshape(TURN_SHAPE).astype(np.float32),
        "reachable_mask": feat["reachable_mask"].reshape(TURN_SHAPE).astype(np.uint8),
        "valid_actions_mask": policy_action_mask(feat),
    }
    return model_obs, feat


def encode_obs(obs: dict[str, Any], player: int | None = None) -> EncodedObs:
    if player is None:
        player = int(obs.get("player", 0))
    model_obs, feat = encode_features(obs, player=player)
    return EncodedObs(
        planets=model_obs["tokens"][0],
        planet_mask=model_obs["presence"][0],
        globals=model_obs["globals"],
        action_mask=discrete_action_mask(feat),
    )


def flat_action_mask(feat: dict[str, Any]) -> np.ndarray:
    return policy_action_mask(feat).astype(bool).ravel()


def discrete_action_index(source_slot: int, target_slot: int) -> int:
    return 1 + source_slot * PLANET_SLOTS + target_slot


def decode_action_index(index: int) -> tuple[int, int, int] | None:
    if index <= 0:
        return None
    raw = int(index) - 1
    return raw // PLANET_SLOTS, raw % PLANET_SLOTS, 0


def discrete_action_mask(feat: dict[str, Any]) -> np.ndarray:
    raw = feat["mask"].reshape(ACTION_TENSOR_SHAPE).astype(bool)
    out = np.zeros(DISCRETE_ACTION_DIM, dtype=np.bool_)
    out[0] = True
    out[1:] = raw[:, :, SEND_ALL_ACTION].reshape(-1)
    return out


def decode_move(obs: dict[str, Any], index: int) -> list[list[float]]:
    feat = _rust_encode_obs(obs, int(obs.get("player", 0)))
    decoded = decode_action_index(index)
    if decoded is None:
        return []
    source_slot, target_slot, _send_bin = decoded
    raw_idx = (source_slot * PLANET_SLOTS + target_slot) * ACTIONS_DIM + SEND_ALL_ACTION
    if raw_idx < 0 or raw_idx >= feat["mask"].shape[0] or not feat["mask"][raw_idx]:
        return []
    source_id = int(feat["planet_ids"][source_slot])
    ships = int(feat["ship_counts"][raw_idx])
    if source_id < 0 or ships <= 0:
        return []
    return [[source_id, float(feat["angles"][raw_idx]), ships]]


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
