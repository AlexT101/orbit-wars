"""Baseline opponents used as training partners.

Both expose a `make_agent()` function returning a callable that accepts the
Kaggle obs dict and returns a list of moves."""

from __future__ import annotations

import math
import random


def make_random_agent():
    def agent(obs):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        for p in planets:
            pid, owner, _x, _y, _r, ships, _prod = p
            if owner != player or ships <= 1:
                continue
            angle = random.uniform(0, 2 * math.pi)
            send = ships // 2
            if send >= 20:
                moves.append([int(pid), float(angle), int(send)])
        return moves

    return agent


def make_nearest_sniper_agent():
    """Mirrors bots/baselines/nearest-sniper. Re-implemented here to avoid the
    sys.path dance — we just want a known-strength training opponent."""

    def agent(obs):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

        my_planets = [p for p in planets if p[1] == player]
        targets = [p for p in planets if p[1] != player]
        if not targets:
            return moves

        for mine in my_planets:
            _, _, mx, my_, _, mships, _ = mine
            best = None
            best_d = float("inf")
            for t in targets:
                _, _, tx, ty, _, _, _ = t
                d = math.hypot(mx - tx, my_ - ty)
                if d < best_d:
                    best_d = d
                    best = t
            if best is None:
                continue
            ships_needed = int(best[5]) + 1
            if int(mships) >= ships_needed:
                angle = math.atan2(best[3] - my_, best[2] - mx)
                moves.append([int(mine[0]), float(angle), int(ships_needed)])
        return moves

    return agent


REGISTRY = {
    "random": make_random_agent,
    "nearest-sniper": make_nearest_sniper_agent,
}


def make_opponent(name: str):
    if name not in REGISTRY:
        raise ValueError(f"unknown opponent '{name}' (options: {list(REGISTRY)})")
    return REGISTRY[name]()
