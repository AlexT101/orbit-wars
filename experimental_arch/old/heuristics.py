from __future__ import annotations

import math

from .features import SEND_FRACTIONS, action_index, path_hits_sun


def nearest_capture_action_index(obs) -> int:
    player = int(obs.get("player", 0))
    planets = list(obs.get("planets", []) or [])
    best = None
    best_dist = float("inf")
    for si, src in enumerate(planets):
        if int(src[1]) != player or int(src[5]) <= 1:
            continue
        for ti, tgt in enumerate(planets):
            if si == ti or int(tgt[1]) == player or path_hits_sun(src, tgt):
                continue
            dx = float(tgt[2]) - float(src[2])
            dy = float(tgt[3]) - float(src[3])
            dist = math.hypot(dx, dy)
            needed = int(tgt[5]) + 1
            if int(src[5]) < needed:
                continue
            frac_needed = needed / max(1.0, float(src[5]))
            send_bin = min(
                range(len(SEND_FRACTIONS)),
                key=lambda i: abs(SEND_FRACTIONS[i] - frac_needed),
            )
            if dist < best_dist:
                best_dist = dist
                best = action_index(si, ti, send_bin)
    return int(best) if best is not None else 0

