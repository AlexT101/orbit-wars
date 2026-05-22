"""
Orbit Wars - simbot

A thin Python wrapper that delegates move selection to a native Rust extension
(`simbot_native`). The Rust agent currently reproduces the open-source
`nearest-sniper` baseline: for each owned planet, target the nearest planet we
don't own and send `garrison + 1` ships when we can afford the takeover. The
Rust side also vendors a clone of the engine (`src/engine.rs`) so the strategy
can evolve to score candidate moves with in-bot simulation.

The Python side only marshals data: it pulls `player` and `planets` out of the
observation, hands them to Rust, and returns whatever Rust decides.

Build the native extension first:

    cd bots/simbot
    maturin develop --release
"""

import os
import sys

# Ensure the bundled native module (simbot_native.*.so) sitting next to this
# file is importable, regardless of the harness's cwd or sys.path. This is what
# lets the Kaggle submission tarball work: main.py + the .so live side by side.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simbot_native


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    # Forward raw planet tuples (id, owner, x, y, radius, ships, production)
    # straight to Rust, which does the decision-making.
    planets = [tuple(p) for p in raw_planets]
    moves = simbot_native.compute_moves(player, planets)

    # Match the baseline's [from_planet_id, angle, num_ships] list-of-lists shape.
    return [list(move) for move in moves]
