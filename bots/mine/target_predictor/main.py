"""Orbit-wars bot that uses the pair-prediction model as its launch policy.

Per turn, for each planet I own with surplus ships:
  1. Run the pair-net forward pass to get P(launch from src->tgt) for all pairs.
  2. Sort potential targets by descending prob.
  3. For each target in order: check the launch is legal (collision predictor
     confirms first-hit is the target), then sample `random() < prob`.
  4. On accept, dispatch ALL ships at the planet toward that target.

If the model assigns no probability above the floor (or all top targets are
illegal), the planet sits this turn.

Usage as a Kaggle main.py: imported by run_match*.py automatically.
Direct invocation for smoke test:
  python3 main.py --self-test  /tmp/orbit_days/orbit-wars-episodes-2026-05-27.zip:77828182.json
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Kaggle loads bots via exec() which strips __file__. Recover via the frame.
def _here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        pass
    frame = inspect.currentframe()
    if frame is not None and frame.f_code.co_filename and frame.f_code.co_filename != "<string>":
        return Path(frame.f_code.co_filename).resolve().parent
    env = os.environ.get("PAIR_NET_DIR")
    if env:
        return Path(env)
    # final fallback: working tree default
    return Path.cwd() / "bots" / "mine" / "target_predictor"


HERE = _here()
TRAIN = HERE / "train"
sys.path.insert(0, str(TRAIN))

from set_net import apply_norm
import build_dataset as bd
from pair_net import PlanetTransformerPair

DEFAULT_CKPT = os.environ.get("PAIR_NET_CKPT", str(TRAIN / "weights" / "transformer_pair.pt"))
SEED = int(os.environ.get("PAIR_NET_SEED", "0")) or None
LAUNCH_FLOOR = float(os.environ.get("PAIR_NET_FLOOR", "0.05"))    # don't sample below this prob
MIN_SHIPS = int(os.environ.get("PAIR_NET_MIN_SHIPS", "2"))         # don't launch tiny fleets


_CKPT = None
_MODEL = None
_DEVICE = torch.device(os.environ.get("PAIR_NET_DEVICE", "cpu"))
_RNG = random.Random(SEED) if SEED is not None else random.Random()


def _ensure_loaded():
    global _CKPT, _MODEL
    if _MODEL is not None:
        return
    ck = torch.load(DEFAULT_CKPT, map_location=_DEVICE, weights_only=False)
    model = PlanetTransformerPair(
        ck["f_planet"], ck["f_global"],
        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
        n_layers=ck.get("n_layers", 2), ff=ck.get("ff", 128), dropout=0.0,
    ).to(_DEVICE).eval()
    model.load_state_dict(ck["state_dict"])
    _CKPT = ck
    _MODEL = model


def _features_for(state, player):
    """Run the build_dataset.extract_per_player on a single state. Returns
    (feats[N, F], globals[F_g], planet_ids[N], owner_change_turn dict)."""
    # owner_change_turn is normally tracked across the game; here we synthesize
    # neutral history (everyone "just changed at turn 0") since the bot may not
    # see the full prior trajectory. The feature is low-importance.
    fake_change = {p["id"]: 0 for p in state["planets"]}
    feats, globals_, pids = bd.extract_per_player(state, player, fake_change)
    return feats, globals_, pids


def _planet_pos_at(state, planet, dt):
    pos = bd.planet_pos_at(state, planet, dt)
    if pos is None:
        return (planet["x"], planet["y"])
    return pos


def _lead_target_angle(src, tgt, ships, state, iters=4):
    """Iteratively solve for the launch angle whose ETA matches the target
    planet's predicted position at that ETA. Returns launch angle (radians)
    and estimated eta (turns)."""
    speed = max(bd.fleet_speed(ships), 1.0)
    # initial guess: angle from src center to current target position
    angle = math.atan2(tgt["y"] - src["y"], tgt["x"] - src["x"])
    eta = max(1, int(math.hypot(tgt["x"] - src["x"], tgt["y"] - src["y"]) / speed))
    for _ in range(iters):
        launch_x = src["x"] + src["radius"] * math.cos(angle)
        launch_y = src["y"] + src["radius"] * math.sin(angle)
        future = _planet_pos_at(state, tgt, eta)
        dx = future[0] - launch_x; dy = future[1] - launch_y
        d = math.hypot(dx, dy)
        eta = max(1, int(round(d / speed)))
        angle = math.atan2(dy, dx)
    return angle, eta


def _launch_hits_target(src, tgt, ships, state):
    """Return True iff a fleet launched from src toward tgt (lead-targeted)
    lands first on `tgt` per the engine's collision rules.

    The Kaggle engine effectively advances the fleet one step before the next
    observation, so we start the swept-collision check one step into the
    trajectory (otherwise the fleet's launch point on src's surface trips a
    false collision with src itself, which has also moved one orbital step).
    """
    angle, _eta = _lead_target_angle(src, tgt, ships, state)
    speed = max(bd.fleet_speed(ships), 1.0)
    launch_x = src["x"] + (src["radius"] + speed) * math.cos(angle)
    launch_y = src["y"] + (src["radius"] + speed) * math.sin(angle)
    fleet = {"x": launch_x, "y": launch_y, "angle": angle, "ships": ships, "owner": -2}
    pred = bd.predict_fleet_collision(state, fleet)
    if pred is None:
        return False, angle
    return pred[0] == tgt["id"], angle


def _decide_launches(obs, rng=None) -> list[list]:
    """Returns a list of [src_pid, angle, ships] actions for this turn."""
    rng = rng or _RNG
    state = bd.parse_state(obs)
    player = int(obs.get("player", 0))
    _ensure_loaded()

    feats, globals_, pids = _features_for(state, player)
    n_real = feats.shape[0]
    pf = np.zeros((1, bd.N_MAX, _CKPT["f_planet"]), dtype=np.float32)
    pf[0, :n_real] = feats
    gl = globals_.reshape(1, -1).astype(np.float32)
    mk = np.zeros((1, bd.N_MAX), dtype=bool); mk[0, :n_real] = True
    pf_n, gl_n = apply_norm(pf, gl, _CKPT["p_mean"], _CKPT["p_std"],
                             _CKPT["g_mean"], _CKPT["g_std"])

    with torch.no_grad():
        logits = _MODEL(
            torch.from_numpy(pf_n).to(_DEVICE),
            torch.from_numpy(gl_n).to(_DEVICE),
            torch.from_numpy(mk).to(_DEVICE),
        ).cpu().numpy()[0]               # (N_MAX, N_MAX)
    probs = 1.0 / (1.0 + np.exp(-logits[:n_real, :n_real]))

    pid_to_planet = {p["id"]: p for p in state["planets"]}
    actions = []

    for i, src_pid in enumerate(pids):
        src = pid_to_planet[int(src_pid)]
        if src["owner"] != player or src["ships"] < MIN_SHIPS:
            continue
        order = np.argsort(-probs[i])
        ships_to_send = int(src["ships"])
        for j in order:
            if i == j:
                continue
            p = probs[i, j]
            if p < LAUNCH_FLOOR:
                break  # all subsequent are also below floor
            tgt_pid = int(pids[j])
            tgt = pid_to_planet[tgt_pid]
            # legality check
            hits, angle = _launch_hits_target(src, tgt, ships_to_send, state)
            if not hits:
                continue
            if rng.random() < p:
                actions.append([src["id"], float(angle), ships_to_send])
                break  # one launch per source per turn (we sent all ships)
    return actions


# ---------------------------------------------------------------------------
# Kaggle-environments entrypoint
# ---------------------------------------------------------------------------


def agent(obs, config=None):
    try:
        return _decide_launches(obs)
    except Exception as exc:
        # On any error, do nothing this turn rather than crashing the match.
        sys.stderr.write(f"pair-bot error: {exc!r}\n")
        return []


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    def _self_test(spec: str):
        import io, zipfile
        zp_s, name = spec.split(":", 1)
        with zipfile.ZipFile(zp_s) as zf:
            with zf.open(name) as f:
                g = json.load(io.BytesIO(f.read()))
        rng = random.Random(0)
        for t in [10, 30, 60, 90, 120]:
            if t >= len(g.get("steps", [])):
                break
            obs = g["steps"][t][0]["observation"]
            acts = _decide_launches(obs, rng=rng)
            print(f"step {t} (player 0): {len(acts)} launches")
            for a in acts:
                print(f"  src={a[0]:3d}  angle={a[1]:+.3f}  ships={a[2]}")

    if "--self-test" in sys.argv:
        i = sys.argv.index("--self-test")
        _self_test(sys.argv[i + 1])
    else:
        print("usage: main.py --self-test <zip>:<json>", file=sys.stderr)
