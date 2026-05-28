"""Closest neutral/enemy all-in heuristic.

For every planet owned by the current player:
  1. Consider only currently neutral or enemy planets.
  2. Keep targets where encoder action 1 is legal. Action 1 sends 100% of the
     source planet's ships.
  3. Send all source ships to the closest legal target.
  4. If there is no legal neutral/enemy target for a source, pass for that
     source by emitting no move.

This is intentionally simple and uses the encoder mask/ship_counts as the
source of truth for legality and the exact ship count.
"""

from __future__ import annotations

import math
from typing import Any

from orbit_wars_model import encode_obs

PLANET_SLOTS = 44
ACTIONS_DIM = 2
SEND_100_ACTION = 1


def _planet_map(obs: dict[str, Any]) -> dict[int, tuple]:
    return {int(p[0]): tuple(p) for p in obs.get("planets", [])}


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _fail(message: str) -> None:
    raise RuntimeError(f"[closest_resolved_plus_one heuristic FAIL] {message}")


def agent(obs: dict[str, Any]) -> list[list[float]]:
    player = int(obs.get("player", 0))
    planets = list(_planet_map(obs).values())
    if not planets:
        return []

    feat = encode_obs(obs, player)
    planet_ids = [int(x) for x in feat["planet_ids"]]
    slot_by_id = {pid: slot for slot, pid in enumerate(planet_ids) if pid >= 0}
    mask = feat["mask"]
    angles = feat["angles"]
    ship_counts = feat["ship_counts"]

    moves: list[list[float]] = []
    for source in planets:
        source_id = int(source[0])
        if int(source[1]) != player:
            continue

        if source_id not in slot_by_id:
            _fail(f"owned source id={source_id} is missing from planet_ids slots")
        si = slot_by_id[source_id]
        source_ships = int(source[5])
        if source_ships <= 0:
            continue

        legal_targets = []
        for target in planets:
            target_id = int(target[0])
            if target_id == source_id or int(target[1]) == player:
                continue
            if target_id not in slot_by_id:
                _fail(f"candidate target id={target_id} is missing from planet_ids slots")
            sj = slot_by_id[target_id]
            flat_idx = (si * PLANET_SLOTS + sj) * ACTIONS_DIM + SEND_100_ACTION
            if not bool(mask[flat_idx]):
                continue
            send_ships = int(ship_counts[flat_idx])
            if send_ships <= 0:
                _fail(
                    "100% action has non-positive ship count: "
                    f"source_id={source_id} target_id={target_id} ships={send_ships}"
                )
            if send_ships != source_ships:
                _fail(
                    "100% action ship count mismatch: "
                    f"source_id={source_id} target_id={target_id} "
                    f"source_ships={source_ships} send_ships={send_ships}"
                )
            legal_targets.append((_dist(source, target), target, flat_idx))

        if not legal_targets:
            continue

        _distance, _target, flat_idx = min(legal_targets, key=lambda item: item[0])
        send_ships = int(ship_counts[flat_idx])
        moves.append([source_id, float(angles[flat_idx]), send_ships])

    return moves
