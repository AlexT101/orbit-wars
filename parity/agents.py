"""Scripted, deterministic agents for parity testing.

Each agent is a callable `(obs_dict, rng) -> moves`. We pass an explicit
`random.Random` so the harness can feed the *same* random stream to the
agent regardless of which engine produced the observation — this keeps
action sequences identical across both engines under test.
"""

from __future__ import annotations

import math
import random
from typing import Callable


AgentFn = Callable[[dict, random.Random], list[list]]


def noop_agent(obs: dict, rng: random.Random) -> list[list]:
    return []


def random_agent(obs: dict, rng: random.Random) -> list[list]:
    """Random launches from each owned planet that has ≥20 ships."""
    moves = []
    player = obs["player"]
    for p in obs["planets"]:
        pid, owner, _, _, _, ships, _ = p
        if owner != player or ships <= 0:
            continue
        send = ships // 2
        if send < 20:
            continue
        angle = rng.uniform(0.0, 2.0 * math.pi)
        moves.append([pid, angle, send])
    return moves


def aggressive_agent(obs: dict, rng: random.Random) -> list[list]:
    """Send half of garrison toward nearest non-owned planet. Deterministic
    given obs alone, so divergence between engines is purely engine-caused."""
    moves = []
    player = obs["player"]
    planets = obs["planets"]
    targets = [p for p in planets if p[1] != player]
    if not targets:
        return moves
    for p in planets:
        pid, owner, x, y, _, ships, _ = p
        if owner != player or ships < 10:
            continue
        nearest = min(targets, key=lambda t: (t[2] - x) ** 2 + (t[3] - y) ** 2)
        angle = math.atan2(nearest[3] - y, nearest[2] - x)
        send = ships // 2
        if send >= 5:
            moves.append([pid, angle, send])
    return moves


def nearest_sniper_agent(obs: dict, rng: random.Random) -> list[list]:
    """Capture the nearest non-owned planet when we hold enough ships to
    guarantee the takeover.

    Ported from bots/_open_source/nearest-sniper/main.py so parity tests
    don't depend on that file. The original parses tuples into a `Planet`
    namedtuple from kaggle_environments; here we unpack the raw 7-field
    planet tuple directly to stay self-contained. Deterministic given obs.

    For each owned planet, find the closest planet we don't own. If we have
    more ships than the target's garrison, send exactly garrison + 1 (the
    minimum that guarantees capture); otherwise wait and accumulate.
    """
    moves = []
    player = obs["player"]
    planets = obs["planets"]
    targets = [p for p in planets if p[1] != player]
    if not targets:
        return moves

    for p in planets:
        pid, owner, x, y, _, ships, _ = p
        if owner != player:
            continue

        # Nearest non-owned planet by Euclidean distance.
        nearest = min(targets, key=lambda t: (t[2] - x) ** 2 + (t[3] - y) ** 2)

        # garrison + 1 is the minimum send that guarantees the takeover.
        ships_needed = nearest[5] + 1
        if ships >= ships_needed:
            angle = math.atan2(nearest[3] - y, nearest[2] - x)
            moves.append([pid, angle, ships_needed])

    return moves


AGENTS: dict[str, AgentFn] = {
    "noop": noop_agent,
    "random": random_agent,
    "aggressive": aggressive_agent,
    "nearest_sniper": nearest_sniper_agent,
}
