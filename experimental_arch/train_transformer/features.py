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
    LAUNCH_GATE_CHOICES,
    NOOP_CHOICE,
    PAIR_OUTCOME_SHAPE,
    PLANET_SLOTS,
    PLANET_TIMELINE_DIM,
    PLANET_TIMELINE_SHAPE,
    POLICY_MASK_SHAPE,
    PRESENCE_SHAPE,
    SEND_HALF_ACTION,
    SEND_ALL_ACTION,
    TARGET_CHOICES,
    TOKEN_DIM,
    TOKEN_SHAPE,
    TURN_SHAPE,
)

MAX_PLANETS = PLANET_SLOTS
PLANET_FEATURES = TOKEN_DIM + PLANET_TIMELINE_DIM
GLOBAL_FEATURES = GLOBAL_DIM
SEND_ACTIONS = (SEND_HALF_ACTION, SEND_ALL_ACTION)
SEND_FRACTIONS = (0.50, 1.00)
DISCRETE_ACTION_DIM = 1 + PLANET_SLOTS * PLANET_SLOTS * len(SEND_FRACTIONS)
ACTION_DIM = DISCRETE_ACTION_DIM

LEGACY_TOKEN_DIM = 11


@dataclass(frozen=True)
class EncodedObs:
    planets: np.ndarray
    planet_mask: np.ndarray
    tokens: np.ndarray
    presence: np.ndarray
    globals: np.ndarray
    action_mask: np.ndarray
    pair_turns: np.ndarray
    pair_reachable_mask: np.ndarray
    pair_outcome_features: np.ndarray
    planet_timeline_features: np.ndarray
    owner_ids: np.ndarray
    player_id: int
    alive_players: int


def observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "globals": spaces.Box(-np.inf, np.inf, shape=(GLOBAL_DIM,), dtype=np.float32),
            "tokens": spaces.Box(-np.inf, np.inf, shape=TOKEN_SHAPE, dtype=np.float32),
            "presence": spaces.Box(0.0, 1.0, shape=PRESENCE_SHAPE, dtype=np.float32),
            "turns": spaces.Box(0.0, np.inf, shape=TURN_SHAPE, dtype=np.float32),
            "reachable_mask": spaces.Box(0, 1, shape=TURN_SHAPE, dtype=np.uint8),
            "pair_outcome_features": spaces.Box(-np.inf, np.inf, shape=PAIR_OUTCOME_SHAPE, dtype=np.float32),
            "planet_timeline_features": spaces.Box(
                -np.inf, np.inf, shape=PLANET_TIMELINE_SHAPE, dtype=np.float32
            ),
            "valid_actions_mask": spaces.Box(0, 1, shape=POLICY_MASK_SHAPE, dtype=np.uint8),
        }
    )


def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(np.array(ACTION_DIMS, dtype=np.int64))


def policy_action_mask(feat: dict[str, Any]) -> np.ndarray:
    raw = feat["mask"].reshape(ACTION_TENSOR_SHAPE).astype(np.uint8)
    out = np.zeros(POLICY_MASK_SHAPE, dtype=np.uint8)
    launch_mask = raw[:, :, SEND_ACTIONS]
    target_mask = launch_mask.any(axis=2)
    has_launch = launch_mask.any(axis=1)
    out[:, 0] = raw[:, 0, NOOP_CHOICE]
    out[:, 1 : 1 + len(SEND_ACTIONS)] = has_launch.astype(np.uint8)
    out[:, LAUNCH_GATE_CHOICES:] = target_mask.astype(np.uint8)

    # MultiDiscrete needs each categorical to have at least one valid choice.
    # If a source cannot launch, the target choice is ignored by decode_action,
    # so slot 0 is a harmless placeholder.
    out[~target_mask.any(axis=1), LAUNCH_GATE_CHOICES] = 1
    return out


def _shape_from_feat(feat: dict[str, Any], key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = feat.get(f"{key}_shape")
    if raw is None:
        return default
    return tuple(int(x) for x in raw)


def _production_onehot(obs: dict[str, Any], feat: dict[str, Any], frames: int) -> np.ndarray:
    by_id: dict[int, int] = {}
    for planet in obs.get("planets", []) or []:
        if len(planet) >= 7:
            by_id[int(planet[0])] = int(planet[6])
    planet_ids = [int(x) for x in feat.get("planet_ids", [])]
    out = np.zeros((frames, PLANET_SLOTS, 5), dtype=np.float32)
    for slot, planet_id in enumerate(planet_ids[:PLANET_SLOTS]):
        prod = by_id.get(planet_id)
        if prod is None:
            continue
        idx = max(0, min(4, prod - 1))
        out[:, slot, idx] = 1.0
    return out


def _tokens_from_feat(obs: dict[str, Any], feat: dict[str, Any]) -> np.ndarray:
    raw_shape = _shape_from_feat(feat, "tokens", TOKEN_SHAPE)
    tokens = feat["tokens"].reshape(raw_shape).astype(np.float32)
    if raw_shape == TOKEN_SHAPE:
        return tokens
    if raw_shape == (TOKEN_SHAPE[0], TOKEN_SHAPE[1], LEGACY_TOKEN_DIM):
        upgraded = np.zeros(TOKEN_SHAPE, dtype=np.float32)
        upgraded[..., :5] = tokens[..., :5]
        upgraded[..., 5:10] = _production_onehot(obs, feat, raw_shape[0])
        upgraded[..., 10:] = tokens[..., 5:10]
        return upgraded
    raise ValueError(f"unexpected token shape {raw_shape}, expected {TOKEN_SHAPE}")


def _array_from_feat(feat: dict[str, Any], key: str, shape: tuple[int, ...], dtype) -> np.ndarray:
    if key not in feat or feat[key] is None:
        return np.zeros(shape, dtype=dtype)
    raw_shape = _shape_from_feat(feat, key, shape)
    value = feat[key].reshape(raw_shape).astype(dtype)
    if raw_shape == shape:
        return value
    raise ValueError(f"unexpected {key} shape {raw_shape}, expected {shape}")


def owner_ids_from_obs(obs: dict[str, Any], feat: dict[str, Any]) -> np.ndarray:
    """Absolute owner category per planet slot: 0=neutral/missing, 1..4=player id + 1."""
    owner_by_id: dict[int, int] = {}
    for planet in obs.get("planets", []) or []:
        if len(planet) >= 2:
            owner_by_id[int(planet[0])] = int(planet[1])
    planet_ids = [int(x) for x in feat.get("planet_ids", [])]
    out = np.zeros((PLANET_SLOTS,), dtype=np.int64)
    for slot, planet_id in enumerate(planet_ids[:PLANET_SLOTS]):
        owner = owner_by_id.get(planet_id, -1)
        if 0 <= owner <= 3:
            out[slot] = owner + 1
    return out


def alive_players_from_obs(obs: dict[str, Any]) -> int:
    owners: set[int] = set()
    for planet in obs.get("planets", []) or []:
        if len(planet) >= 6:
            owner = int(planet[1])
            if owner >= 0 and float(planet[5]) > 0:
                owners.add(owner)
    for fleet in obs.get("fleets", []) or []:
        if len(fleet) >= 7:
            owner = int(fleet[1])
            if owner >= 0 and float(fleet[6]) > 0:
                owners.add(owner)
    return len(owners)


def encode_features(obs: dict[str, Any], player: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feat = _rust_encode_obs(obs, player)
    tokens = _tokens_from_feat(obs, feat)
    model_obs = {
        "globals": feat["globals"].astype(np.float32),
        "tokens": tokens,
        "presence": _array_from_feat(feat, "presence", PRESENCE_SHAPE, np.float32),
        "turns": _array_from_feat(feat, "turns", TURN_SHAPE, np.float32),
        "reachable_mask": _array_from_feat(feat, "reachable_mask", TURN_SHAPE, np.uint8),
        "pair_outcome_features": _array_from_feat(
            feat, "pair_outcome_features", PAIR_OUTCOME_SHAPE, np.float32
        ),
        "planet_timeline_features": _array_from_feat(
            feat, "planet_timeline_features", PLANET_TIMELINE_SHAPE, np.float32
        ),
        "valid_actions_mask": policy_action_mask(feat),
        "owner_ids": owner_ids_from_obs(obs, feat),
        "player_ids": np.asarray(int(player), dtype=np.int64),
        "alive_players": np.asarray(alive_players_from_obs(obs), dtype=np.int64),
    }
    return model_obs, feat


def encoded_from_feat(feat: dict[str, Any], obs: dict[str, Any] | None = None) -> EncodedObs:
    if obs is None:
        tokens = feat["tokens"].reshape(TOKEN_SHAPE).astype(np.float32)
        owner_ids = np.zeros((PLANET_SLOTS,), dtype=np.int64)
        player_id = 0
        alive_players = 0
    else:
        tokens = _tokens_from_feat(obs, feat)
        owner_ids = owner_ids_from_obs(obs, feat)
        player_id = int(obs.get("player", 0))
        alive_players = alive_players_from_obs(obs)
    presence = _array_from_feat(feat, "presence", PRESENCE_SHAPE, np.float32)
    turns = _array_from_feat(feat, "turns", TURN_SHAPE, np.float32)
    reachable = _array_from_feat(feat, "reachable_mask", TURN_SHAPE, np.uint8)
    pair_outcome = _array_from_feat(feat, "pair_outcome_features", PAIR_OUTCOME_SHAPE, np.float32)
    timeline = _array_from_feat(feat, "planet_timeline_features", PLANET_TIMELINE_SHAPE, np.float32)
    return EncodedObs(
        planets=tokens[0],
        planet_mask=presence[0],
        tokens=tokens,
        presence=presence,
        globals=feat["globals"].astype(np.float32),
        action_mask=discrete_action_mask(feat),
        pair_turns=turns[0],
        pair_reachable_mask=reachable[0],
        pair_outcome_features=pair_outcome,
        planet_timeline_features=timeline,
        owner_ids=owner_ids,
        player_id=player_id,
        alive_players=alive_players,
    )


def encode_obs(obs: dict[str, Any], player: int | None = None) -> EncodedObs:
    if player is None:
        player = int(obs.get("player", 0))
    feat = _rust_encode_obs(obs, player)
    return encoded_from_feat(feat, obs=obs)


def encode_obs_and_feat(
    obs: dict[str, Any], player: int | None = None
) -> tuple[EncodedObs, dict[str, Any]]:
    """Encode once, returning both the model-ready `EncodedObs` and the raw
    feature dict. Lets a caller run the policy forward and then decode several
    action indices (via `decode_index_from_feat`) off a single encode, instead
    of paying the Rust trajectory build once per decoded move."""
    if player is None:
        player = int(obs.get("player", 0))
    feat = _rust_encode_obs(obs, player)
    return encoded_from_feat(feat, obs=obs), feat


def flat_action_mask(feat: dict[str, Any]) -> np.ndarray:
    return policy_action_mask(feat).astype(bool).ravel()


def discrete_action_index(source_slot: int, target_slot: int, send_bin: int) -> int:
    return 1 + ((source_slot * PLANET_SLOTS + target_slot) * len(SEND_FRACTIONS) + send_bin)


def decode_action_index(index: int) -> tuple[int, int, int] | None:
    if index <= 0:
        return None
    raw = int(index) - 1
    pair, send_bin = divmod(raw, len(SEND_FRACTIONS))
    return pair // PLANET_SLOTS, pair % PLANET_SLOTS, send_bin


def discrete_action_mask(feat: dict[str, Any]) -> np.ndarray:
    raw = feat["mask"].reshape(ACTION_TENSOR_SHAPE).astype(bool)
    out = np.zeros(DISCRETE_ACTION_DIM, dtype=np.bool_)
    out[0] = True
    out[1:] = raw[:, :, SEND_ACTIONS].reshape(-1)
    return out


def decode_index_from_feat(feat: dict[str, Any], index: int) -> list[list[float]]:
    """Decode a flat policy index into a launch using an already-computed raw
    feature dict (from `orbit_wars_model.encode_obs`).

    Pure — does no encoding. A caller that has already encoded the observation
    (e.g. for the policy forward) can decode many indices without re-running the
    costly Rust trajectory build once per index. `decode_move` is the
    encode-then-decode convenience wrapper."""
    decoded = decode_action_index(index)
    if decoded is None:
        return []
    source_slot, target_slot, send_bin = decoded
    if send_bin < 0 or send_bin >= len(SEND_ACTIONS):
        return []
    raw_action = SEND_ACTIONS[send_bin]
    raw_idx = (source_slot * PLANET_SLOTS + target_slot) * ACTIONS_DIM + raw_action
    if raw_idx < 0 or raw_idx >= feat["mask"].shape[0] or not feat["mask"][raw_idx]:
        return []
    source_id = int(feat["planet_ids"][source_slot])
    ships = int(feat["ship_counts"][raw_idx])
    if source_id < 0 or ships <= 0:
        return []
    return [[source_id, float(feat["angles"][raw_idx]), ships]]


def decode_move(obs: dict[str, Any], index: int) -> list[list[float]]:
    feat = _rust_encode_obs(obs, int(obs.get("player", 0)))
    return decode_index_from_feat(feat, index)


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
        if launch not in SEND_ACTIONS:
            continue
        target_slot = int(pair[1])
        if target_slot < 0 or target_slot >= TARGET_CHOICES:
            continue
        idx = (source_slot * PLANET_SLOTS + target_slot) * ACTIONS_DIM + launch
        if idx < 0 or idx >= mask.shape[0] or not mask[idx]:
            continue
        source_id = int(planet_ids[source_slot])
        ships = int(ship_counts[idx])
        if source_id < 0 or ships <= 0:
            continue
        moves.append([source_id, float(angles[idx]), ships])
    return moves
