"""Closest-planet all-in heuristic.

For every planet owned by the current player:
  1. Consider all other planets. Ownership is ignored, so the target may be
     ally, neutral, or enemy.
  2. Keep only planets where the encoder says the 100% action is legal.
  3. Send all ships to the closest legal target.
  4. Skip empty owned planets. Fail loudly if a non-empty owned planet has no
     legal 100% target.

This is deliberately blunt: it is a feature/action-contract sanity checker, not
a good Orbit Wars bot.
"""

from __future__ import annotations

import math
from typing import Any

from orbit_wars_model import encode_obs

PLANET_SLOTS = 44
ACTIONS_DIM = 7
SEND_100_ACTION = 4


def _planet_map(obs: dict[str, Any]) -> dict[int, tuple]:
    return {int(p[0]): tuple(p) for p in obs.get("planets", [])}


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _fail(message: str) -> None:
    raise RuntimeError(f"[closest_all heuristic FAIL] {message}")


def agent(obs: dict[str, Any]) -> list[list[float]]:
    player = int(obs.get("player", 0))
    planets = list(_planet_map(obs).values())
    if not planets:
        return []

    feat = encode_obs(obs, player)
    planet_ids = [int(x) for x in feat["planet_ids"]]
    slot_by_id = {pid: slot for slot, pid in enumerate(planet_ids) if pid >= 0}
    mask = feat["mask"]
    reachable = feat["reachable_mask"].reshape(4, PLANET_SLOTS, PLANET_SLOTS, ACTIONS_DIM)
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
            if target_id == source_id:
                continue
            if target_id not in slot_by_id:
                _fail(f"candidate target id={target_id} is missing from planet_ids slots")
            sj = slot_by_id[target_id]
            flat_idx = (si * PLANET_SLOTS + sj) * ACTIONS_DIM + SEND_100_ACTION
            if not bool(mask[flat_idx]):
                continue
            send_ships = int(ship_counts[flat_idx])
            if send_ships != source_ships:
                _fail(
                    "100% action ship count mismatch: "
                    f"source_id={source_id} target_id={target_id} expected={source_ships} got={send_ships}"
                )
            if send_ships <= 0:
                _fail(f"100% action has non-positive ship count for source_id={source_id}: {send_ships}")
            legal_targets.append((_dist(source, target), target, flat_idx))

        if not legal_targets:
            _fail(
                "no legal 100% target for owned planet: "
                f"player={player} source_id={source_id} source_slot={si} "
                f"source_ships={source_ships} "
                f"reachable_100_targets={int(reachable[0, si, :, SEND_100_ACTION].sum())}"
            )

        _distance, _target, flat_idx = min(legal_targets, key=lambda item: item[0])
        send_ships = int(ship_counts[flat_idx])
        moves.append([source_id, float(angles[flat_idx]), send_ships])

    return moves
