"""alphaduck: pair-policy + DUCT MCTS bot.

Architecture (one turn):
  1. Run the target_predictor pair model on the current state from each side's
     POV. Pair model gives P(launch s -> t) for every (source, target) pair.
  2. Build per-planet action sets: noop + top-K targets per source (per side).
  3. Run DUCT search: each planet's choice is an independent UCB tree; each
     iteration samples a joint action (one per planet on each side), applies
     it to the state (event-driven: extrapolate all in-flight fleets to their
     arrival, then heuristic-evaluate the resulting state).
  4. Return the joint action whose Q values are highest after the budget.

Env vars:
  ALPHADUCK_BUDGET_MS    per-turn search budget in ms (default 400)
  ALPHADUCK_C_PUCT       exploration constant (default 1.4)
  ALPHADUCK_MAX_ITERS    hard cap on iterations (default 600)
  ALPHADUCK_FLOOR        skip target candidates with P < floor (default 0.01)
  ALPHADUCK_COMET_FORCE_LAUNCH  force-launch when an owned comet has ≤ N
                         turns remaining (default 2) so its ships aren't lost
  PAIR_NET_CKPT          path to pair model checkpoint
"""

from __future__ import annotations

import inspect
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        pass
    frame = inspect.currentframe()
    if frame is not None and frame.f_code.co_filename and frame.f_code.co_filename != "<string>":
        return Path(frame.f_code.co_filename).resolve().parent
    return Path.cwd() / "bots" / "alphaduck"


HERE = _here()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "bots" / "mine" / "target_predictor" / "train"))
sys.path.insert(0, str(ROOT / "bots" / "alphaduck" / "train"))
sys.path.insert(0, str(HERE))

import build_dataset as bd
import build_dataset_v17 as bd17
from pair_net_v17 import PairNetV17
import fastsim

# Apollo native module — used to compute the eta_apollo pair feature once per state.
# Module-level path resolution mirrors what build_dataset_v17 does.
sys.path.insert(0, str(ROOT / "bots" / "mine" / "apollo"))
# aim_native (forked aim solver + bit-exact engine_step) lives at
# bots/alphaduck/aim_native. Built via `maturin build --release`.
import aim_native

# Standard alias for backward compat in helper code below.
_BD = bd
# `bd` defaults to the current (50-feature) extractor. _load_model() picks the
# legacy 46-feature extractor (build_dataset_v0) when the ckpt has f_planet=46
# (i.e. transformer_pair.pt or older). Switch via the module-global `_BD` so
# the rest of the bot calls a single namespace.
_BD = bd

# tuning knobs
BUDGET_MS = float(os.environ.get("ALPHADUCK_BUDGET_MS", "600"))
C_PUCT = float(os.environ.get("ALPHADUCK_C_PUCT", "1.4"))
MAX_ITERS = int(os.environ.get("ALPHADUCK_MAX_ITERS", "100000"))
FLOOR = float(os.environ.get("ALPHADUCK_FLOOR", "0.01"))
# Candidate filter (move-gen): keep moves while cum-prior < CAND_CUM AND
# prior >= max_prior / CAND_REL_DEN. Stops both flat-policy fan-out and
# near-zero-prior tail. No hard cap — CAND_CUM is the only count-limiter.
CAND_CUM = float(os.environ.get("ALPHADUCK_CAND_CUM", "0.8"))
CAND_REL_DEN = float(os.environ.get("ALPHADUCK_CAND_REL_DEN", "10.0"))
# Force a launch from a comet we own when its remaining lifetime drops below
# this. Otherwise we lose all ships on the comet when it despawns.
COMET_FORCE_LAUNCH_TURNS = int(os.environ.get("ALPHADUCK_COMET_FORCE_LAUNCH", "2"))
# Per-turn debug logging to stderr. 1 = log step + iters + chosen actions.
DEBUG = int(os.environ.get("ALPHADUCK_DEBUG", "0"))
# 1 = use model.value() at each leaf (slower, learned eval);
# 0 = use hand-coded heuristic (~20× faster iters).
USE_VALUE = int(os.environ.get("ALPHADUCK_USE_VALUE", "1"))
# 1 = project all in-flight fleets to their arrivals before evaluating the leaf
#     (ballistic-style; model sees future state, not current).
# 0 = evaluate the leaf at the state immediately after applying the joint action.
#     Default 0 per the design principle "MLP should eval CURRENT state, not
#     extrapolated"; flip on if the heuristic eval is in use (it benefits from
#     resolving fleets first).
PROJECT_TO_ARRIVALS = int(os.environ.get("ALPHADUCK_PROJECT_ARRIVALS", "0" if USE_VALUE else "1"))
# v18: enable sequential intra-side tree (vs flat DUCT per-planet stats). Sorts
# planets by confidence (max_prior / second_max_prior) and walks a tree where
# each level = one planet's decision, conditional on prior planets in the same
# side. No info leak between sides — each side has its own independent tree.
# Eval still only at joint-complete state (no partial-joint OOD evaluations).
V18_INTRA_SIDE_TREE = int(os.environ.get("ALPHADUCK_V18", "0"))
# Greedy mode: skip MCTS entirely. Per planet, take argmax of the policy head.
# Used for A/B testing whether search adds value over the policy prior alone.
GREEDY = int(os.environ.get("ALPHADUCK_GREEDY", "0"))
GREEDY_LAUNCH_THRESHOLD = float(os.environ.get("ALPHADUCK_GREEDY_THRESHOLD", "0.5"))
# Run one full engine tick after _apply_joint so the leaf state matches the
# training-distribution state(T+1) (after production, fleet movement, planet
# rotation, combat). Without this, the eval sees a "hybrid" state with launches
# applied but no engine tick — a small train/serve mismatch.
STEP_ONE = int(os.environ.get("ALPHADUCK_STEP_ONE", "0"))
# Crop candidate launches that are dominated by "wait one turn": if launching
# next turn arrives at the same turn or earlier (due to planet rotation /
# higher ship count → faster fleet), there's no harm in waiting. Default ON.
CROP_DOMINATED = int(os.environ.get("ALPHADUCK_CROP_DOMINATED", "1"))

DEFAULT_CKPT = os.environ.get(
    "PAIR_NET_CKPT",
    str(ROOT / "bots" / "alphaduck" / "train" / "weights" / "transformer_pair_v17_jun.pt"),
)
# Optional fast leaf-eval model. If set to an XGB .json or MLP .pt path,
# alphaduck uses it instead of the transformer's value head at each MCTS
# leaf. ~3× faster but loses spatial differentiation — value can't tell
# "launched to A" from "launched to B" in pooled feature space.
# Default empty = use transformer value head.
VALUE_NET_PATH = os.environ.get("ALPHADUCK_VALUE_NET", "")

_CKPT = None
_MODEL = None
_DEVICE = torch.device(os.environ.get("PAIR_NET_DEVICE", "cpu"))


# Lazy-loaded fast value net (XGB Booster or MLP). When set, used at MCTS leaves
# instead of the transformer's value head. ~50× faster per leaf.
_VAL_NET = None
_VAL_NET_KIND = None   # "xgb" | "mlp" | None
_VAL_NET_NORM = None   # dict with p_mean/p_std/g_mean/g_std


# Fast value net (XGB) path was removed for v17 — pooled features can't see fleet
# tokens or per-pair attention bias, so they're a poor leaf eval for this model.
_VAL_NET = None  # always None; kept as a sentinel for legacy checks below.


def _load_model():
    """Load v17 transformer ckpt + PairNetV17 model."""
    global _CKPT, _MODEL
    if _MODEL is not None:
        return
    ck = torch.load(DEFAULT_CKPT, map_location=_DEVICE, weights_only=False)
    assert ck.get("version") == "v17", f"v17-only alphaduck, got ckpt version={ck.get('version')}"
    m = PairNetV17(
        f_planet=ck["f_planet"], f_fleet=ck["f_fleet"],
        f_global=ck["f_global"], f_pair=ck["f_pair"],
        n_planet_max=ck["n_planet_max"], n_fleet_max=ck["n_fleet_max"],
        d_model=ck["d_model"], n_heads=ck["n_heads"],
        n_layers=ck["n_layers"], ff=ck["ff"], dropout=0.0,
    ).to(_DEVICE).eval()
    m.load_state_dict(ck["state_dict"])
    _CKPT = ck
    _MODEL = m
    fastsim.warmup()


# Persistent owner-history tracking across turns within a game. Resets when
# we see step go backwards (new game). Matches build_dataset's tempo logic
# (player↔player transitions only).
_LAST_OWNER: dict[int, int] = {}
_OWNER_CHANGE_TURN: dict[int, int] = {}
_LAST_GAME_STEP: int = -1


def _reset_if_new_game(state):
    global _LAST_GAME_STEP
    if state["step"] < _LAST_GAME_STEP:
        _LAST_OWNER.clear(); _OWNER_CHANGE_TURN.clear()
    # Per-turn cache reset: states from previous turns won't recur (step is part
    # of the key, but in-flight fleet positions also drift) so clearing keeps
    # cache hot for the *current* turn's MCTS leaves only.
    _VALUE_CACHE.clear()
    _AIM_CACHE.clear()
    _LAST_GAME_STEP = state["step"]
    _BD.update_owner_history(state, _LAST_OWNER, _OWNER_CHANGE_TURN, state["step"])


def _build_v17_inputs(state, player, obs_for_apollo=None):
    """Compute the v17 input tensors for one state from `player`'s POV.
    Returns dict of numpy arrays at batch dim 1 (model expects (B, ...))."""
    aim = _aim_for_state(state)
    feats = bd17.extract_state_v17(
        state, player, _OWNER_CHANGE_TURN, obs_for_apollo=obs_for_apollo,
        aim_etas=aim,
    )
    pf = feats["planet_feats"][None]                    # (1, N_MAX, F_planet)
    pm = feats["planet_mask"][None]                     # (1, N_MAX)
    ff = feats["fleet_feats"][None]                     # (1, F_MAX, F_fleet)
    fm = feats["fleet_mask"][None]                      # (1, F_MAX)
    fti = feats["fleet_tgt_idx"][None]                  # (1, F_MAX)
    gl = feats["globals"][None]                         # (1, F_global)
    pa = feats["pair_feats"][None]                      # (1, N_MAX, N_MAX, F_pair)
    n = feats["n_real"]
    pids = [int(x) for x in feats["planet_ids"][:n]]
    # Apply normalization stored on the ckpt.
    pf_n = (pf - _CKPT["p_mean"]) / np.clip(_CKPT["p_std"], 1e-6, None)
    gl_n = (gl - _CKPT["g_mean"]) / np.clip(_CKPT["g_std"], 1e-6, None)
    ff_n = (ff - _CKPT["f_mean"]) / np.clip(_CKPT["f_std"], 1e-6, None)
    pa_n = (pa - _CKPT["pa_mean"]) / np.clip(_CKPT["pa_std"], 1e-6, None)
    return dict(
        pf=pf_n.astype(np.float32), pm=pm,
        ff=ff_n.astype(np.float32), fm=fm,
        fti=fti, gl=gl_n.astype(np.float32),
        pa=pa_n.astype(np.float32),
        n_real=n, pids=pids,
    )


_AIM_CACHE: dict = {}
_AIM_CACHE_MAX = 5000


def _aim_for_state(state):
    """Return dict {(src_pid, tgt_pid, src_ships): (eta, angle) | None} for all planet pairs.
    Cached by (turn, planets-tuple) so repeated visits to the same state are free.
    Apollo aim is player-independent geometry, so the same cache serves both POVs.
    NOTE: returning (eta, angle) tuples — callers that only need eta read [0];
    using apollo's angle (not a separately-solved lead-angle) ensures the
    launched fleet actually arrives at the target apollo says it can hit."""
    planets = state["planets"]
    key = (state["step"], tuple((int(p["id"]), int(p["ships"])) for p in planets))
    cached = _AIM_CACHE.get(key)
    if cached is not None:
        return cached
    clean_obs = _state_obs_dict(state)
    triples = []
    for src in planets:
        ships = max(int(src["ships"]), 1)
        s_id = int(src["id"])
        for tgt in planets:
            t_id = int(tgt["id"])
            if s_id == t_id:
                continue
            triples.append((s_id, t_id, ships))
    out = aim_native.aim_eta_angle_batch(clean_obs, triples)
    aim = {t: ea for t, ea in zip(triples, out)}
    if len(_AIM_CACHE) >= _AIM_CACHE_MAX:
        _AIM_CACHE.pop(next(iter(_AIM_CACHE)))
    _AIM_CACHE[key] = aim
    return aim


def _state_obs_dict(state):
    """Reconstruct a kaggle-style obs dict for apollo from a parsed state.
    Apollo parses planets as a sequence of [id, owner, x, y, radius, ships, prod]
    tuples — not dicts. Comets stay as dicts with planet_ids/paths/path_index."""
    planets = [
        [int(p["id"]), int(p["owner"]), float(p["x"]), float(p["y"]),
         float(p.get("radius", 0.0)), int(p["ships"]), int(p["prod"])]
        for p in state["planets"]
    ]
    # Apollo's parse_comets expects each comet dict to have planet_ids, paths, path_index.
    comets = state.get("comets", [])
    comet_planet_ids = [int(p["id"]) for p in state["planets"] if p.get("is_comet")]
    return {
        "planets": planets,
        "angular_velocity": float(state.get("av", 0.0)),
        "comets": comets,
        "comet_planet_ids": comet_planet_ids,
    }


def _root_forward(state, player):
    """Returns (policy_probs[n, n+1], value scalar, planet_ids[n]).

    policy_probs[i, 0]     = P(noop from source i)
    policy_probs[i, 1+j]   = P(launch from source i to planet j)
    Single softmax over N+1 actions per source (noop + N launch targets).
    """
    _reset_if_new_game(state)
    obs_for_apollo = _state_obs_dict(state)
    inp = _build_v17_inputs(state, player, obs_for_apollo=obs_for_apollo)
    n = inp["n_real"]; pids = inp["pids"]
    with torch.no_grad():
        policy_logits, value = _MODEL(
            torch.from_numpy(inp["pf"]).to(_DEVICE),
            torch.from_numpy(inp["pm"]).to(_DEVICE),
            torch.from_numpy(inp["ff"]).to(_DEVICE),
            torch.from_numpy(inp["fm"]).to(_DEVICE),
            torch.from_numpy(inp["fti"].astype(np.int64)).to(_DEVICE),
            torch.from_numpy(inp["gl"]).to(_DEVICE),
            torch.from_numpy(inp["pa"]).to(_DEVICE),
        )
    # policy_logits: (1, N_MAX, N_MAX+1). Slice to real planets.
    logits = policy_logits.cpu().numpy()[0][:n, :n + 1]
    value = float(value.cpu().numpy()[0])
    # Mask launch-to-self (diagonal in launch columns) — training also masks this.
    for i in range(n):
        logits[i, 1 + i] = -1e9
    # Stable softmax over (n+1) actions per source.
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    policy_probs = ex / ex.sum(axis=1, keepdims=True)
    return policy_probs, value, list(pids)


def _pair_probs(state, player):
    """Returns (policy_probs[N, N+1], planet_ids)."""
    policy, _v, pids = _root_forward(state, player)
    return policy, pids


# ---------------------------------------------------------------------------
# State manipulation
# ---------------------------------------------------------------------------


def _clone_state(state):
    """Shallow clone enough to mutate planets/fleets/comets without affecting
    the original. Inner comet path tuples are immutable so a deep clone of
    them isn't needed."""
    return {
        "step": state["step"],
        "av": state["av"],
        "planets": [dict(p) for p in state["planets"]],
        "fleets": [dict(f) for f in state["fleets"]],
        "comets": [{"planet_ids": list(g["planet_ids"]),
                    "paths": g["paths"],
                    "path_index": g["path_index"]} for g in state["comets"]],
    }


def _lead_angle(src, tgt, ships, state, iters=3):
    speed = max(_BD.fleet_speed(ships), 1.0)
    angle = math.atan2(tgt["y"] - src["y"], tgt["x"] - src["x"])
    eta = max(1, int(math.hypot(tgt["x"] - src["x"], tgt["y"] - src["y"]) / speed))
    for _ in range(iters):
        lx = src["x"] + src["radius"] * math.cos(angle)
        ly = src["y"] + src["radius"] * math.sin(angle)
        fp = _BD.planet_pos_at(state, tgt, eta)
        if fp is None:
            fp = (tgt["x"], tgt["y"])
        dx = fp[0] - lx; dy = fp[1] - ly
        eta = max(1, int(round(math.hypot(dx, dy) / speed)))
        angle = math.atan2(dy, dx)
    return angle, eta


# Engine convention (bots/alphaduck/aim_native/src/engine.rs::process_moves):
# fleet spawns at src.x + (radius + LAUNCH_CLEARANCE) * cos(angle), then moves
# by +speed*cos(angle) per turn. Earlier code here used radius+speed which was
# wrong (off by one fleet length on the starting position).
LAUNCH_CLEARANCE = 0.1


def _legal(src, tgt, ships, state):
    angle, eta = _lead_angle(src, tgt, ships, state)
    lx = src["x"] + (src["radius"] + LAUNCH_CLEARANCE) * math.cos(angle)
    ly = src["y"] + (src["radius"] + LAUNCH_CLEARANCE) * math.sin(angle)
    fleet = {"x": lx, "y": ly, "angle": angle, "ships": ships, "owner": -2}
    pred = _BD.predict_fleet_collision(state, fleet)
    if pred is None:
        return False, angle, eta
    return pred[0] == tgt["id"], angle, eta


def _apply_joint(state, joint):
    """joint = list of (player, src_pid, tgt_pid, ships, angle, eta).
    Mutates a clone and returns it. Adds new fleets, deducts source ships."""
    s = _clone_state(state)
    pid_to_planet = {p["id"]: p for p in s["planets"]}
    next_fid = (max((f["id"] for f in s["fleets"]), default=-1)) + 1
    for player, src_pid, _tgt_pid, ships, angle, _eta in joint:
        src = pid_to_planet.get(src_pid)
        if src is None or src["owner"] != player or src["ships"] < ships or ships < 1:
            continue
        src["ships"] -= ships
        lx = src["x"] + (src["radius"] + LAUNCH_CLEARANCE) * math.cos(angle)
        ly = src["y"] + (src["radius"] + LAUNCH_CLEARANCE) * math.sin(angle)
        s["fleets"].append({"id": next_fid, "owner": player,
                            "x": lx, "y": ly, "angle": angle, "ships": ships})
        next_fid += 1
    return s


def _step_one(state):
    """Run one engine tick on a post-_apply_joint state to produce state(T+1).

    Bit-exact wrapper around `aim_native.engine_step` (apollo's vendored
    Simulator). Used for MCTS leaf-eval state advance when ALPHADUCK_STEP_ONE=1
    so the value head sees a state in the training distribution.

    The pure-Python implementation below is retained as a fallback in case
    aim_native is unavailable; expect 35-40% parity (combat queueing, swept
    collisions, etc. are approximated). For correctness, prefer aim_native.
    """
    out = aim_native.engine_step(state)
    # Reshape to match the alphaduck state schema used elsewhere (parse_state
    # output): re-add per-planet derived fields the rest of main.py reads.
    pid_to_in = {p["id"]: p for p in state["planets"]}
    for p in out["planets"]:
        pin = pid_to_in.get(p["id"])
        if pin is not None:
            for k in ("orb_r", "init_angle", "is_orbiting", "is_comet"):
                if k in pin:
                    p[k] = pin[k]
    # Carry comet groups forward (engine_step advances path_index internally,
    # but we re-attach the alphaduck-side group metadata so subsequent calls
    # still have the path tables to advance).
    out_groups = []
    for g in state.get("comets", []):
        new_idx = g["path_index"] + 1
        path_len = len(g["paths"][0]) if g["paths"] else 0
        if new_idx >= path_len:
            continue
        out_groups.append({"planet_ids": list(g["planet_ids"]),
                           "paths": g["paths"],
                           "path_index": new_idx})
    out["comets"] = out_groups
    return out


def _step_one_pure_python(state):
    """Legacy pure-Python single-tick approximation (35-40% parity vs engine).
    Kept for reference / fallback path; not used when aim_native is installed.
    """
    s = _clone_state(state)
    s["step"] += 1
    # 4. Production
    for p in s["planets"]:
        if p["owner"] >= 0 and not p.get("is_comet", False):
            p["ships"] += p["prod"]
    # 6. Planet rotation: use planet_pos_at at dt=0 of the new step.
    for p in s["planets"]:
        if p.get("is_orbiting") or p.get("is_comet"):
            pos = _BD.planet_pos_at(s, p, 0)
            if pos is not None:
                p["x"], p["y"] = pos
    # Advance comet path_index and drop despawned comets.
    despawned_pids: set[int] = set()
    live_groups = []
    for g in s.get("comets", []):
        new_idx = g["path_index"] + 1
        path_len = len(g["paths"][0]) if g["paths"] else 0
        if new_idx >= path_len:
            despawned_pids.update(int(pid) for pid in g["planet_ids"])
            continue
        g["path_index"] = new_idx
        live_groups.append(g)
    if "comets" in s:
        s["comets"] = live_groups
    if despawned_pids:
        s["planets"] = [p for p in s["planets"] if int(p["id"]) not in despawned_pids]
    # 5 + 7. Fleet movement, arrivals, combat.
    surviving = []
    for f in s["fleets"]:
        speed = max(_BD.fleet_speed(f["ships"]), 1.0)
        nx = f["x"] + speed * math.cos(f["angle"])
        ny = f["y"] + speed * math.sin(f["angle"])
        if nx < 0 or nx > _BD.BOARD or ny < 0 or ny > _BD.BOARD:
            continue
        cdx = nx - _BD.CENTER[0]; cdy = ny - _BD.CENTER[1]
        if cdx * cdx + cdy * cdy < _BD.SUN_RADIUS * _BD.SUN_RADIUS:
            continue
        hit = None
        for p in s["planets"]:
            pdx = nx - p["x"]; pdy = ny - p["y"]
            r = p["radius"]
            if pdx * pdx + pdy * pdy <= r * r:
                hit = p
                break
        if hit is not None:
            if hit["owner"] == f["owner"]:
                hit["ships"] += f["ships"]
            else:
                if f["ships"] > hit["ships"]:
                    hit["owner"] = f["owner"]
                    hit["ships"] = f["ships"] - hit["ships"]
                else:
                    hit["ships"] -= f["ships"]
        else:
            f["x"], f["y"] = nx, ny
            surviving.append(f)
    s["fleets"] = surviving
    return s


def _project_to_arrivals(state, flat=None, prebuilt_events=None, new_fleets=None):
    """Event-driven: resolve all in-flight fleets at their arrivals.

    Fast path: caller supplies `flat` (pre-flattened planet geometry from
    fastsim.flatten_state), `prebuilt_events` for pre-existing fleets (already
    predicted once at the search root), and only the *new* fleets need a fresh
    JIT collision prediction. This avoids re-running predict_fleet_collision
    on every pre-existing fleet every iteration.
    """
    s = _clone_state(state)
    pid_to_planet = {p["id"]: p for p in s["planets"]}
    events = []  # (eta, dest_pid, fleet_owner, fleet_ships)
    if prebuilt_events is not None:
        events.extend(prebuilt_events)
    if new_fleets is not None and flat is not None:
        for f in new_fleets:
            speed = max(_BD.fleet_speed(f["ships"]), 1.0)
            dest_pid, eta = fastsim.predict_one_fleet_fast(
                flat, f["x"], f["y"], f["angle"], speed,
            )
            if dest_pid is None:
                continue
            events.append((eta, dest_pid, f["owner"], f["ships"]))
    if prebuilt_events is None and new_fleets is None:
        for f in s["fleets"]:
            pred = _BD.predict_fleet_collision(s, f)
            if pred is None:
                continue
            events.append((pred[1], pred[0], f["owner"], f["ships"]))
    events.sort(key=lambda e: e[0])
    # accrue production at the moment each event resolves
    last_t = 0
    for eta, dest_pid, f_owner, f_ships in events:
        elapsed = eta - last_t
        for p in pid_to_planet.values():
            if p["owner"] >= 0 and not p["is_comet"]:
                p["ships"] += p["prod"] * elapsed
        last_t = eta
        tgt = pid_to_planet.get(dest_pid)
        if tgt is None:
            continue
        if tgt["owner"] == f_owner:
            tgt["ships"] += f_ships
        else:
            if f_ships > tgt["ships"]:
                tgt["owner"] = f_owner
                tgt["ships"] = f_ships - tgt["ships"]
            else:
                tgt["ships"] -= f_ships
    s["fleets"] = []
    # Advance state clock + comet path_index by the total elapsed time so apollo
    # (and any other consumer reading from this state) sees positions at the
    # projected step. If a comet's path_index walks past its path length, the
    # comet has despawned — drop it from the planet list and the comet group
    # list to match the engine's semantics.
    if last_t > 0:
        s["step"] += last_t
        despawned_pids: set[int] = set()
        live_groups = []
        for g in s["comets"]:
            new_idx = g["path_index"] + last_t
            path_len = len(g["paths"][0]) if g["paths"] else 0
            if new_idx >= path_len:
                despawned_pids.update(int(p) for p in g["planet_ids"])
                continue  # drop entire group
            g["path_index"] = new_idx
            live_groups.append(g)
        s["comets"] = live_groups
        if despawned_pids:
            s["planets"] = [p for p in s["planets"]
                            if int(p["id"]) not in despawned_pids]
    return s


def _evaluate(state, player):
    """Heuristic value of `state` from `player`'s POV. Used when ALPHADUCK_USE_VALUE=0."""
    my_p = my_pr = my_s = 0; en_p = en_pr = en_s = 0
    for p in state["planets"]:
        if p["owner"] == player:
            my_p += 1; my_pr += p["prod"]; my_s += p["ships"]
        elif p["owner"] >= 0:
            en_p += 1; en_pr += p["prod"]; en_s += p["ships"]
    return (my_p - en_p) * 5.0 + (my_pr - en_pr) * 8.0 + (my_s - en_s) * 0.05


_VALUE_CACHE: dict = {}
_VALUE_CACHE_MAX = int(os.environ.get("ALPHADUCK_VALUE_CACHE_SIZE", "20000"))


def _state_key(state, player):
    """Hash for value-cache. Captures everything the value head can see.
    Floats use full precision: aggressive rounding caused collisions where
    semantically-different leaves shared a key and returned wrong cached values.
    Cache hits are still common because identical joint actions from the same
    root produce bit-identical states (Python addition is deterministic)."""
    planets = tuple((p["id"], p["owner"], int(p["ships"]), int(p["prod"]))
                    for p in state["planets"])
    fleets = tuple((f["owner"], int(f["ships"]), f["x"], f["y"], f["angle"])
                   for f in state["fleets"])
    return (state["step"], player, planets, fleets)


def _value_from_model(state, player):
    """Single-state value eval. Returns a scalar in [-1, 1]. Cache-aware."""
    key = _state_key(state, player)
    cached = _VALUE_CACHE.get(key)
    if cached is not None:
        return cached
    obs = _state_obs_dict(state)
    inp = _build_v17_inputs(state, player, obs_for_apollo=obs)
    with torch.no_grad():
        _, value = _MODEL(
            torch.from_numpy(inp["pf"]).to(_DEVICE),
            torch.from_numpy(inp["pm"]).to(_DEVICE),
            torch.from_numpy(inp["ff"]).to(_DEVICE),
            torch.from_numpy(inp["fm"]).to(_DEVICE),
            torch.from_numpy(inp["fti"].astype(np.int64)).to(_DEVICE),
            torch.from_numpy(inp["gl"]).to(_DEVICE),
            torch.from_numpy(inp["pa"]).to(_DEVICE),
        )
    v = float(value.cpu().numpy()[0])
    if len(_VALUE_CACHE) < _VALUE_CACHE_MAX:
        _VALUE_CACHE[key] = v
    return v


# ---------------------------------------------------------------------------
# DUCT search (depth-1)
# ---------------------------------------------------------------------------


def _build_candidates(state, policy, pids, player):
    """For each `player`-owned planet, returns:
       (src_pid, ships, [ (kind, tgt_pid_or_None, prob, angle, eta) ... ])

    kind == "noop" or "launch". noop is treated as just another option and
    competes in the same prior-based filter as launches (cum < CAND_CUM AND
    prior >= max_prior / CAND_REL_DEN). If noop's prior is below threshold it
    is dropped — that source is then forced to launch.

    `policy[i]` is a length (N+1) probability distribution: index 0 = noop,
    index 1+j = launch to planet j. Sums to 1.

    All-or-nothing launching: when a source acts, it sends ALL its ships.
    """
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    pid_to_planet = {p["id"]: p for p in state["planets"]}
    # Cached apollo aim (eta, angle) per (src, tgt, ships) triple, shared across
    # POVs and MCTS iters via _AIM_CACHE.
    aim_now = _aim_for_state(state)
    candidates = []
    for src_pid in pids:
        src = pid_to_planet[src_pid]
        if src["owner"] != player or src["ships"] <= 0:
            continue
        i = pid_to_idx[src_pid]
        ships = int(src["ships"])
        # Build all options (noop + each legal launch), then sort by prior desc.
        noop_prior = float(policy[i, 0])
        all_opts = [("noop", None, noop_prior, None, None)]
        pair_pri = policy[i, 1:]
        for j in np.argsort(-pair_pri):
            if i == j: continue
            p = float(pair_pri[j])
            if p <= 0: continue
            tgt_pid = int(pids[j])
            # Apollo says whether this (src, tgt, ships) is reachable, and gives
            # both the ETA and the angle. Use apollo's angle for the launch —
            # the lead-angle Python solver is NOT bit-equivalent to apollo, so
            # using its angle with apollo's eta-reachability would miss.
            ea = aim_now.get((src_pid, tgt_pid, ships))
            if ea is None:
                continue
            eta, angle = ea
            all_opts.append(("launch", tgt_pid, p, angle, eta))
        all_opts.sort(key=lambda o: -o[2])

        # Filter: keep while cum < CAND_CUM AND prior >= max / CAND_REL_DEN.
        max_prior = all_opts[0][2] if all_opts else 0.0
        rel_thresh = max_prior / CAND_REL_DEN
        kept = []
        cum = 0.0
        for opt in all_opts:
            if opt[2] < rel_thresh: break
            if cum >= CAND_CUM: break
            kept.append(opt)
            cum += opt[2]
        # Comet despawn rule: if this source is a comet we own and it's about
        # to disappear within COMET_FORCE_LAUNCH_TURNS turns, force a launch by
        # dropping noop and keeping the best legal launch. Without this we lose
        # the ships on the comet when it despawns.
        if src.get("is_comet"):
            remaining = bd.comet_remaining(state, src)
            if 0 < remaining <= COMET_FORCE_LAUNCH_TURNS:
                launches = [o for o in kept if o[0] == "launch"]
                if not launches:
                    for opt in all_opts:
                        if opt[0] == "launch":
                            launches.append(opt); break
                if launches:
                    kept = launches  # drop noop
        if kept:
            candidates.append((src_pid, ships, kept))
    return candidates


def _ucb_select(stats, c_puct, total_visits):
    """stats = list of (Q_mean, N, prior). Returns index with highest PUCT.
    Priors are assumed to be a proper distribution (sum to 1, set up in _init)."""
    best_i = 0; best_v = -float("inf")
    sqrt_total = math.sqrt(max(total_visits, 1))
    for i, (q, n, prior) in enumerate(stats):
        u = c_puct * prior * sqrt_total / (1 + n)
        v = q + u
        if v > best_v:
            best_v = v; best_i = i
    return best_i


def _duct_search(state, my_player, deadline):
    _load_model()
    my_policy, my_pids = _pair_probs(state, my_player)
    opp_player = 1 - my_player
    opp_policy, opp_pids = _pair_probs(state, opp_player)
    my_cands = _build_candidates(state, my_policy, my_pids, my_player)
    opp_cands = _build_candidates(state, opp_policy, opp_pids, opp_player)

    # ---- per-turn cache: flatten planet geometry once and pre-predict
    # destinations for all fleets already in flight. These don't change across
    # iterations; only fleets spawned by joint actions vary.
    flat = fastsim.flatten_state(state)
    prebuilt_events = []
    for f in state["fleets"]:
        speed = max(_BD.fleet_speed(f["ships"]), 1.0)
        dest_pid, eta = fastsim.predict_one_fleet_fast(
            flat, f["x"], f["y"], f["angle"], speed,
        )
        if dest_pid is None:
            continue
        prebuilt_events.append((eta, dest_pid, f["owner"], f["ships"]))

    # For each source: stats[src][k] = [Q_sum, N, prior] indexed parallel to
    # the candidate options (noop and launches mixed, sorted-then-filtered).
    # Priors come from the trained policy softmax; renormalize because the
    # filter drops mass.
    def _init(cands):
        out = {}
        for src_pid, ships, opts in cands:
            raw = [max(o[2], 1e-6) for o in opts]
            tot = sum(raw)
            priors = [x / tot for x in raw]
            stats = [[0.0, 0, priors[k]] for k in range(len(raw))]
            out[src_pid] = (ships, opts, stats)
        return out

    my_stats = _init(my_cands)
    opp_stats = _init(opp_cands)
    total_iters = 0

    # Pick order: planets with most ships first. Independent UCB per planet,
    # but the dictionary iteration order is fixed for predictability.
    my_order = sorted(my_stats.keys(), key=lambda pid: -my_stats[pid][0])
    opp_order = sorted(opp_stats.keys(), key=lambda pid: -opp_stats[pid][0])

    def _select_side(stats_dict, player_id, order, joint_out, picks_out):
        for src_pid in order:
            ships, opts, stats = stats_dict[src_pid]
            total_n = sum(s[1] for s in stats) + 1
            k = _ucb_select([(s[0] / max(s[1], 1), s[1], s[2]) for s in stats], C_PUCT, total_n)
            picks_out[src_pid] = k
            kind, tgt_pid, _p, angle, _eta = opts[k]
            if kind == "noop":
                continue
            joint_out.append((player_id, src_pid, tgt_pid, ships, angle, _eta))

    while total_iters < MAX_ITERS and time.perf_counter() < deadline:
        joint = []
        my_picks: dict[int, int] = {}; opp_picks: dict[int, int] = {}
        _select_side(my_stats, my_player, my_order, joint, my_picks)
        _select_side(opp_stats, opp_player, opp_order, joint, opp_picks)
        s_eval = _apply_joint(state, joint)
        if USE_VALUE:
            val = _value_from_model(s_eval, my_player)
        else:
            val = _evaluate(s_eval, my_player)
        total_iters += 1
        for src_pid, k in my_picks.items():
            stats = my_stats[src_pid][2]
            stats[k][1] += 1
            stats[k][0] += float(val)
        for src_pid, k in opp_picks.items():
            stats = opp_stats[src_pid][2]
            stats[k][1] += 1
            stats[k][0] -= float(val)  # opp wants to minimize my val

    # Final pick: per source, the most-visited option (DUCT convention).
    # Tiebreak: higher avg Q wins, then non-noop wins (so flat-Q plateaus don't
    # default to noop just from list order).
    actions = []
    for src_pid, (ships, opts, stats) in my_stats.items():
        def _key(i):
            n = stats[i][1]
            q = stats[i][0] / max(n, 1)
            return (n, q, 0 if opts[i][0] == "noop" else 1)
        best_k = max(range(len(stats)), key=_key)
        kind, tgt_pid, _p, angle, _eta = opts[best_k]
        if kind == "noop": continue
        actions.append([int(src_pid), float(angle), int(ships)])
    return actions, total_iters


# ---------------------------------------------------------------------------
# Recursive DUCT tree (multi-ply lookahead)
# Each node = state; per side per source [Q_sum, N, prior].
# Walk down via UCB until depth==MAX_DEPTH or a fresh node; expand+eval there;
# backprop. Same per-source factorization as flat DUCT, but stats are
# *conditioned on the path* so deeper Q estimates inform shallower picks.
# ---------------------------------------------------------------------------


MCTS_DEPTH = int(os.environ.get("ALPHADUCK_MCTS_DEPTH", "4"))


class _MctsNode:
    __slots__ = ("state", "my_cands", "opp_cands", "my_stats", "opp_stats",
                 "my_order", "opp_order", "children", "expanded")

    def __init__(self, state):
        self.state = state
        self.expanded = False
        self.children = {}

    def expand(self, my_player):
        my_policy, my_pids = _pair_probs(self.state, my_player)
        opp_policy, opp_pids = _pair_probs(self.state, 1 - my_player)
        self.my_cands = _build_candidates(self.state, my_policy, my_pids, my_player)
        self.opp_cands = _build_candidates(self.state, opp_policy, opp_pids, 1 - my_player)

        def _init(cands):
            out = {}
            for src_pid, ships, opts in cands:
                raw = [max(o[2], 1e-6) for o in opts]
                tot = sum(raw)
                pri = [x / tot for x in raw]
                out[src_pid] = (ships, opts, [[0.0, 0, pri[k]] for k in range(len(opts))])
            return out

        self.my_stats = _init(self.my_cands)
        self.opp_stats = _init(self.opp_cands)
        self.my_order = sorted(self.my_stats.keys(), key=lambda pid: -self.my_stats[pid][0])
        self.opp_order = sorted(self.opp_stats.keys(), key=lambda pid: -self.opp_stats[pid][0])
        self.expanded = True


def _mcts_pick_joint(node, my_player):
    """UCB-select per source for both sides at this node. Returns (joint_action_list,
    my_picks_dict, opp_picks_dict)."""
    joint = []
    my_picks = {}; opp_picks = {}
    for src_pid in node.my_order:
        ships, opts, stats = node.my_stats[src_pid]
        total_n = sum(s[1] for s in stats) + 1
        k = _ucb_select([(s[0] / max(s[1], 1), s[1], s[2]) for s in stats], C_PUCT, total_n)
        my_picks[src_pid] = k
        kind, tgt_pid, _p, angle, _eta = opts[k]
        if kind != "noop":
            joint.append((my_player, src_pid, tgt_pid, ships, angle, _eta))
    for src_pid in node.opp_order:
        ships, opts, stats = node.opp_stats[src_pid]
        total_n = sum(s[1] for s in stats) + 1
        k = _ucb_select([(s[0] / max(s[1], 1), s[1], s[2]) for s in stats], C_PUCT, total_n)
        opp_picks[src_pid] = k
        kind, tgt_pid, _p, angle, _eta = opts[k]
        if kind != "noop":
            joint.append((1 - my_player, src_pid, tgt_pid, ships, angle, _eta))
    return joint, my_picks, opp_picks


def _mcts_iter(node, depth, max_depth, my_player, deadline):
    """One PUCT walk + leaf eval + backprop. Returns the value at the leaf
    (from my_player's POV)."""
    if not node.expanded:
        node.expand(my_player)
        return _value_from_model(node.state, my_player)
    if depth >= max_depth or time.perf_counter() >= deadline:
        return _value_from_model(node.state, my_player)
    joint, my_picks, opp_picks = _mcts_pick_joint(node, my_player)
    s_after = _apply_joint(node.state, joint)
    s_next = _step_one(s_after)  # advance one tick so the next state differs
    key = (tuple(sorted(my_picks.items())), tuple(sorted(opp_picks.items())))
    child = node.children.get(key)
    if child is None:
        child = _MctsNode(s_next)
        node.children[key] = child
    val = _mcts_iter(child, depth + 1, max_depth, my_player, deadline)
    for src_pid, k in my_picks.items():
        st = node.my_stats[src_pid][2]
        st[k][1] += 1
        st[k][0] += float(val)
    for src_pid, k in opp_picks.items():
        st = node.opp_stats[src_pid][2]
        st[k][1] += 1
        st[k][0] -= float(val)
    return val


def _mcts_search(state, my_player, deadline):
    _load_model()
    root = _MctsNode(state)
    root.expand(my_player)
    total_iters = 0
    while time.perf_counter() < deadline:
        _mcts_iter(root, 0, MCTS_DEPTH, my_player, deadline)
        total_iters += 1
    actions = []
    for src_pid in root.my_order:
        ships, opts, stats = root.my_stats[src_pid]
        def _key(i):
            n = stats[i][1]
            q = stats[i][0] / max(n, 1)
            return (n, q, 0 if opts[i][0] == "noop" else 1)
        best_k = max(range(len(stats)), key=_key)
        kind, tgt_pid, _p, angle, _eta = opts[best_k]
        if kind == "noop":
            continue
        actions.append([int(src_pid), float(angle), int(ships)])
    return actions, total_iters


# ---------------------------------------------------------------------------
# v18: Sequential intra-side tree (DUCT between sides, sequential within each)
# ---------------------------------------------------------------------------


def _confidence(opts):
    """confidence(planet) = max_prior / (second_max_prior + eps).
    High = "model knows what to do". Used to order planets in the intra-side
    tree, most-confident first (their decisions are stable across iters)."""
    if len(opts) <= 1:
        return float("inf")
    priors = sorted((float(o[2]) for o in opts), reverse=True)
    return priors[0] / (priors[1] + 1e-6)


def _v18_search(state, my_player, deadline):
    """Sequential intra-side tree MCTS.

    Per side, planets are ordered by confidence; the tree has one level per
    planet, with stats conditional on prior-planet picks via the *path prefix*.
    PUCT walks down each iter, opp's tree walked independently in parallel
    (no info leak between sides), and the value-net evaluates only the
    joint-complete leaf state."""
    _load_model()
    my_policy, my_pids = _pair_probs(state, my_player)
    opp_player = 1 - my_player
    opp_policy, opp_pids = _pair_probs(state, opp_player)
    my_cands = _build_candidates(state, my_policy, my_pids, my_player)
    opp_cands = _build_candidates(state, opp_policy, opp_pids, opp_player)

    # Build per-planet info: pid -> (ships, options list)
    my_info = {pid: (ships, opts) for pid, ships, opts in my_cands}
    opp_info = {pid: (ships, opts) for pid, ships, opts in opp_cands}

    # Confidence-ordered planet sequences. If a side has 0 active planets, its
    # tree is trivial (empty path → always evaluate same joint).
    my_order = sorted(my_info.keys(), key=lambda p: -_confidence(my_info[p][1]))
    opp_order = sorted(opp_info.keys(), key=lambda p: -_confidence(opp_info[p][1]))

    # Tree: dict path-tuple -> list of [Q_sum, N, prior] for the NEXT planet's
    # options. path-tuple is the sequence of action indices picked so far.
    # path () => stats for first planet in order. (k0,) => stats for second
    # planet, conditional on first picked k0. Etc.
    my_tree: dict[tuple, list[list[float]]] = {}
    opp_tree: dict[tuple, list[list[float]]] = {}

    def _init_stats(opts):
        # Convert option priors to a normalized distribution and seed stats.
        raw = [max(o[2], 1e-6) for o in opts]
        tot = sum(raw)
        priors = [x / tot for x in raw]
        return [[0.0, 0, priors[k]] for k in range(len(raw))]

    def _walk_side(order, info, tree):
        """Walk one side's tree, picking an option per planet in confidence order.
        Returns (joint_actions_list, path_visits). joint_actions_list is the list
        of (player_unset, src_pid, tgt_pid, ships, angle, eta) tuples for this
        side's launches (noops omitted). path_visits is a list of
        (path_tuple_before_pick, action_index_picked) used for backup."""
        path: tuple = ()
        path_visits: list[tuple[tuple, int]] = []
        joint = []
        for src_pid in order:
            ships, opts = info[src_pid]
            stats = tree.get(path)
            if stats is None:
                stats = _init_stats(opts)
                tree[path] = stats
            total_n = sum(s[1] for s in stats) + 1
            k = _ucb_select([(s[0] / max(s[1], 1), s[1], s[2]) for s in stats],
                            C_PUCT, total_n)
            path_visits.append((path, k))
            kind, tgt_pid, _p, angle, _eta = opts[k]
            if kind == "launch":
                joint.append((src_pid, tgt_pid, ships, angle, _eta))
            path = path + (k,)
        return joint, path_visits

    # Per-turn pre-built fleet-events cache (matches _duct_search).
    flat = fastsim.flatten_state(state)
    prebuilt_events = []
    for f in state["fleets"]:
        speed = max(_BD.fleet_speed(f["ships"]), 1.0)
        dest_pid, eta = fastsim.predict_one_fleet_fast(
            flat, f["x"], f["y"], f["angle"], speed,
        )
        if dest_pid is None:
            continue
        prebuilt_events.append((eta, dest_pid, f["owner"], f["ships"]))

    total_iters = 0
    while total_iters < MAX_ITERS and time.perf_counter() < deadline:
        total_iters += 1
        my_joint_partial, my_path = _walk_side(my_order, my_info, my_tree)
        opp_joint_partial, opp_path = _walk_side(opp_order, opp_info, opp_tree)
        # Combine into the engine's joint format: (player, src, tgt, ships, angle, eta)
        joint = [(my_player, sp, tp, sh, an, et) for (sp, tp, sh, an, et) in my_joint_partial]
        joint += [(opp_player, sp, tp, sh, an, et) for (sp, tp, sh, an, et) in opp_joint_partial]

        s_next = _apply_joint(state, joint)
        if PROJECT_TO_ARRIVALS:
            new_fleet_count = len(s_next["fleets"]) - len(state["fleets"])
            new_fleets = s_next["fleets"][-new_fleet_count:] if new_fleet_count > 0 else []
            s_eval = _project_to_arrivals(
                s_next, flat=flat, prebuilt_events=prebuilt_events, new_fleets=new_fleets,
            )
        elif STEP_ONE:
            s_eval = _step_one(s_next)
        else:
            s_eval = s_next
        val = _value_from_model(s_eval, my_player) if USE_VALUE else _evaluate(s_eval, my_player)

        # Backup: +V along my_path, -V along opp_path.
        for (p, k) in my_path:
            stats = my_tree[p]
            stats[k][1] += 1
            stats[k][0] += float(val)
        for (p, k) in opp_path:
            stats = opp_tree[p]
            stats[k][1] += 1
            stats[k][0] -= float(val)

    # Final pick: walk my tree most-visited at each level → joint action.
    actions = []
    path: tuple = ()
    for src_pid in my_order:
        ships, opts = my_info[src_pid]
        stats = my_tree.get(path)
        if stats is None:
            # planet was never reached in any iter (rare). Fall back to noop.
            continue
        def _key(i):
            n = stats[i][1]
            q = stats[i][0] / max(n, 1)
            return (n, q, 0 if opts[i][0] == "noop" else 1)
        best_k = max(range(len(stats)), key=_key)
        kind, tgt_pid, _p, angle, _eta = opts[best_k]
        if kind == "launch":
            actions.append([int(src_pid), float(angle), int(ships)])
        path = path + (best_k,)
    return actions, total_iters


# ---------------------------------------------------------------------------
# Kaggle entrypoint
# ---------------------------------------------------------------------------


def _greedy_act(state, my_player):
    """No search: per planet, launch all ships to policy argmax if 1 - P(noop) >= threshold."""
    _load_model()
    policy, _value, pids = _root_forward(state, my_player)
    pid_to_planet = {p["id"]: p for p in state["planets"]}
    actions = []
    for i, pid in enumerate(pids):
        src = pid_to_planet[pid]
        if src["owner"] != my_player or src["ships"] <= 0:
            continue
        noop_p = float(policy[i, 0])
        if (1.0 - noop_p) < GREEDY_LAUNCH_THRESHOLD:
            continue
        order = sorted(range(len(pids)),
                       key=lambda j: float(policy[i, 1 + j]) if j != i else -1.0,
                       reverse=True)
        for j in order:
            if j == i:
                continue
            tgt = pid_to_planet.get(int(pids[j]))
            if tgt is None:
                continue
            ok, angle, _eta = _legal(src, tgt, int(src["ships"]), state)
            if ok:
                actions.append([int(pid), float(angle), int(src["ships"])])
                break
    return actions


def agent(obs, config=None):
    try:
        deadline = time.perf_counter() + BUDGET_MS / 1000.0
        state = _BD.parse_state(obs)
        my_player = int(obs.get("player", 0))
        if GREEDY:
            actions = _greedy_act(state, my_player)
            if DEBUG:
                sys.stderr.write(f"alphaduck t={state['step']} p={my_player} GREEDY acts={len(actions)}\n")
            return actions
        if MCTS_DEPTH > 1:
            search_fn = _mcts_search
        elif V18_INTRA_SIDE_TREE:
            search_fn = _v18_search
        else:
            search_fn = _duct_search
        actions, n_iters = search_fn(state, my_player, deadline)
        # Post-MCTS CROP_DOMINATED filter: drop any chosen launch where waiting
        # one turn gives an arrival no later than the current launch (with more
        # ships, since src accrues production). We DO NOT prune at candidate
        # generation — MCTS should see the launch, value it, and decide. This
        # filter only fires at submission time, so the rejected launches still
        # informed MCTS's Q-stats.
        if CROP_DOMINATED and actions:
            state_next = _step_one(state)
            pid_to_planet_next = {p["id"]: p for p in state_next["planets"]}
            aim_next = _aim_for_state(state_next)
            kept_actions = []
            for src_pid, angle_now, ships in actions:
                # Find this src's eta_now from the cached aim for `state`.
                aim_now = _aim_for_state(state)
                # We need the matching tgt; re-derive from the action's angle
                # is hard, but predict_fleet_collision gives the tgt + eta the
                # fleet actually hits. Cheap (~5µs).
                src = {p["id"]: p for p in state["planets"]}.get(src_pid)
                if src is None:
                    kept_actions.append([src_pid, angle_now, ships]); continue
                lx = src["x"] + (src["radius"] + LAUNCH_CLEARANCE) * math.cos(angle_now)
                ly = src["y"] + (src["radius"] + LAUNCH_CLEARANCE) * math.sin(angle_now)
                pred = _BD.predict_fleet_collision(state, {"x": lx, "y": ly, "angle": angle_now, "ships": ships, "owner": my_player})
                if pred is None:
                    kept_actions.append([src_pid, angle_now, ships]); continue
                tgt_pid, eta_now = pred
                src_next = pid_to_planet_next.get(src_pid)
                if src_next is None or src_next["owner"] != my_player or src_next["ships"] <= 0:
                    kept_actions.append([src_pid, angle_now, ships]); continue
                ships_next = int(src_next["ships"])
                ea_next = aim_next.get((src_pid, int(tgt_pid), ships_next))
                if ea_next is None:
                    kept_actions.append([src_pid, angle_now, ships]); continue
                eta_wait = ea_next[0]
                arr_now  = state["step"] + eta_now
                arr_wait = state_next["step"] + eta_wait
                if arr_wait <= arr_now:
                    # wait dominates → drop this launch (effectively noop)
                    continue
                kept_actions.append([src_pid, angle_now, ships])
            actions = kept_actions
        if DEBUG:
            sys.stderr.write(f"alphaduck t={state['step']} p={my_player} iters={n_iters} acts={len(actions)}\n")
        return actions
    except Exception as exc:
        sys.stderr.write(f"alphaduck error: {exc!r}\n")
        import traceback; traceback.print_exc(file=sys.stderr)
        return []


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        import io, json, zipfile
        i = sys.argv.index("--self-test")
        spec = sys.argv[i + 1]
        zp_s, name = spec.split(":", 1)
        with zipfile.ZipFile(zp_s) as zf:
            with zf.open(name) as f:
                g = json.load(io.BytesIO(f.read()))
        for t in [10, 30, 60, 90, 120]:
            if t >= len(g.get("steps", [])): break
            obs = g["steps"][t][0]["observation"]
            t0 = time.perf_counter()
            acts = agent(obs)
            dt = (time.perf_counter() - t0) * 1000
            print(f"step {t}: {len(acts)} launches in {dt:.0f}ms")
            for a in acts:
                print(f"  src={a[0]:3d}  angle={a[1]:+.3f}  ships={a[2]}")
