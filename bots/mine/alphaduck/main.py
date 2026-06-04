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
  ALPHADUCK_TOPK         hard cap on candidates per source (default 10)
  ALPHADUCK_C_PUCT       exploration constant (default 1.4)
  ALPHADUCK_MAX_ITERS    hard cap on iterations (default 600)
  ALPHADUCK_FLOOR        skip target candidates with P < floor (default 0.01)
  ALPHADUCK_MIN_SHIPS    don't launch sources with fewer ships (default 2)
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
    return Path.cwd() / "bots" / "mine" / "alphaduck"


HERE = _here()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "bots" / "mine" / "target_predictor" / "train"))
sys.path.insert(0, str(HERE))

import build_dataset as bd
import build_dataset_v0 as bd_v0
from set_net import apply_norm
from pair_net import PlanetTransformerPair
import fastsim

# `bd` defaults to the current (50-feature) extractor. _load_model() picks the
# legacy 46-feature extractor (build_dataset_v0) when the ckpt has f_planet=46
# (i.e. transformer_pair.pt or older). Switch via the module-global `_BD` so
# the rest of the bot calls a single namespace.
_BD = bd

# tuning knobs
BUDGET_MS = float(os.environ.get("ALPHADUCK_BUDGET_MS", "600"))
TOPK = int(os.environ.get("ALPHADUCK_TOPK", "10"))
C_PUCT = float(os.environ.get("ALPHADUCK_C_PUCT", "1.4"))
MAX_ITERS = int(os.environ.get("ALPHADUCK_MAX_ITERS", "600"))
FLOOR = float(os.environ.get("ALPHADUCK_FLOOR", "0.01"))
MIN_SHIPS = int(os.environ.get("ALPHADUCK_MIN_SHIPS", "10"))
# Temperature on the policy prior. Higher = more uniform (more exploration of
# non-favored moves). Applied on logits before softmax. Default 1.0 = trust
# the model. Setting >1 helps if the model is over-confident on noop.
POLICY_TEMP = float(os.environ.get("ALPHADUCK_POLICY_TEMP", "1.0"))
# Optional override for the noop component of the policy distribution. If set
# to a positive value, replaces the model's noop probability with this value
# (per-source), and renormalizes the pair probs to fill (1 - NOOP_OVERRIDE).
# Useful for forcing more exploration of launch actions when the model is
# correctly calibrated but the marginal launch rate (~5%) is too pessimistic
# as an MCTS prior. Set to 0 (default) to disable.
NOOP_OVERRIDE = float(os.environ.get("ALPHADUCK_NOOP_OVERRIDE", "0.3"))
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

DEFAULT_CKPT = os.environ.get(
    "PAIR_NET_CKPT",
    str(ROOT / "bots" / "mine" / "target_predictor" / "train" / "weights" / "transformer_pair_v11_cond.pt"),
)

_CKPT = None
_MODEL = None
_DEVICE = torch.device(os.environ.get("PAIR_NET_DEVICE", "cpu"))


def _load_model():
    global _CKPT, _MODEL, _BD
    if _MODEL is not None:
        return
    ck = torch.load(DEFAULT_CKPT, map_location=_DEVICE, weights_only=False)
    # Dispatch feature-extractor based on the ckpt's feature count: the
    # transformer_pair.pt (46-d) ckpt was trained on the legacy build_dataset
    # before raw_xy / projected_* / in_mine_count_5/10 etc. were added.
    if ck["f_planet"] == 46:
        _BD = bd_v0
        sys.stderr.write(f"alphaduck: loading legacy 46-feature ckpt ({DEFAULT_CKPT})\n")
    else:
        _BD = bd
    m = PlanetTransformerPair(
        ck["f_planet"], ck["f_global"],
        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
        n_layers=ck.get("n_layers", 2), ff=ck.get("ff", 128), dropout=0.0,
    ).to(_DEVICE).eval()
    # Old ckpts (pre-v9) lack noop/value/pair_feat heads in state_dict; load
    # what matches and leave the new heads at their (zero) init.
    sd = ck["state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        sys.stderr.write(f"alphaduck: ckpt missing keys (will use defaults): {missing[:6]}{'…' if len(missing) > 6 else ''}\n")
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
    _LAST_GAME_STEP = state["step"]
    _BD.update_owner_history(state, _LAST_OWNER, _OWNER_CHANGE_TURN, state["step"])


def _root_forward(state, player, override_noop=True):
    """Returns (policy_probs[N, N+1], value scalar, planet_ids[N]).

    policy_probs[i, 0] = P(noop from source i)
    policy_probs[i, 1+j] = P(launch from source i to planet j)
    Per row sums to 1 (softmax over [noop, pair_logits[i,:]]).

    For older v9 checkpoints without the policy head trained, we still emit a
    valid distribution from the pair+noop sigmoids (renormalized).

    override_noop: if True (default), apply NOOP_OVERRIDE/POLICY_TEMP knobs.
    Useful to disable for predicting opponent moves where calibration is what
    we want (we want their REAL likely action, not exploration-boosted).
    """
    _reset_if_new_game(state)
    feats, globals_, pids = _BD.extract_per_player(state, player, _OWNER_CHANGE_TURN)
    n = feats.shape[0]
    pf = np.zeros((1, _BD.N_MAX, _CKPT["f_planet"]), dtype=np.float32); pf[0, :n] = feats
    gl = globals_.reshape(1, -1).astype(np.float32)
    mk = np.zeros((1, _BD.N_MAX), dtype=bool); mk[0, :n] = True

    # raw pair inputs at 7 horizons (0,1,2,5,10,20,30)
    raw_xy = np.zeros((1, _BD.N_MAX, 7, 2), dtype=np.float32)
    raw_ships = np.zeros((1, _BD.N_MAX), dtype=np.float32)
    raw_prod = np.zeros((1, _BD.N_MAX), dtype=np.float32)
    for i, p in enumerate(state["planets"]):
        for j, h in enumerate((0, 1, 2, 5, 10, 20, 30)):
            pos = _BD.planet_pos_at(state, p, h)
            raw_xy[0, i, j] = pos if pos is not None else (p["x"], p["y"])
        raw_ships[0, i] = p["ships"]
        raw_prod[0, i] = p["prod"]

    pf_n, gl_n = apply_norm(pf, gl, _CKPT["p_mean"], _CKPT["p_std"], _CKPT["g_mean"], _CKPT["g_std"])
    # If ckpt was trained with drop_turn_features, zero those globals at inference too.
    if _CKPT.get("drop_turn_features") and _CKPT.get("global_names"):
        names = list(_CKPT["global_names"])
        for j, name in enumerate(names):
            if name.startswith("turn_") or name.startswith("phase_"):
                gl_n[:, j] = 0.0
    with torch.no_grad():
        out = _MODEL(
            torch.from_numpy(pf_n).to(_DEVICE),
            torch.from_numpy(gl_n).to(_DEVICE),
            torch.from_numpy(mk).to(_DEVICE),
            raw_xy=torch.from_numpy(raw_xy).to(_DEVICE),
            raw_ships=torch.from_numpy(raw_ships).to(_DEVICE),
            raw_prod=torch.from_numpy(raw_prod).to(_DEVICE),
            return_value=True, return_noop=True,
        )
        pair_logits, value, noop_logits = out
        pair_logits_np = pair_logits.cpu().numpy()[0][:n, :n]
        value = float(value.cpu().numpy()[0])
        noop_logits_np = noop_logits.cpu().numpy()[0][:n]
    # Build per-source policy.
    has_policy_head = bool(_CKPT.get("policy_loss_weight", 0.0) > 0)
    is_conditional = bool(_CKPT.get("policy_conditional", False))
    pair_logits_masked = pair_logits_np.copy()
    np.fill_diagonal(pair_logits_masked, -1e9)
    if has_policy_head and is_conditional:
        # Conditional: P(target | launch) = softmax(pair_logits) over targets;
        # P(noop) from the noop sigmoid head separately.
        flat = pair_logits_masked / POLICY_TEMP - (pair_logits_masked / POLICY_TEMP).max(axis=1, keepdims=True)
        ex = np.exp(flat)
        cond_pair = ex / ex.sum(axis=1, keepdims=True)
        noop_probs = 1.0 / (1.0 + np.exp(-noop_logits_np))
        pair_probs = (1.0 - noop_probs)[:, None] * cond_pair
        policy_probs = np.concatenate([noop_probs[:, None], pair_probs], axis=1)
    elif has_policy_head:
        # Unified softmax over [noop, t0, t1, ...]
        full = np.concatenate([noop_logits_np[:, None], pair_logits_masked], axis=1) / POLICY_TEMP
        full = full - full.max(axis=1, keepdims=True)
        ex = np.exp(full)
        policy_probs = ex / ex.sum(axis=1, keepdims=True)
    else:
        # Fallback for older ckpts (v9 etc.) — renormalize sigmoid outputs as a distribution.
        pair_probs = 1.0 / (1.0 + np.exp(-pair_logits_np))
        np.fill_diagonal(pair_probs, 0.0)
        noop_probs = 1.0 / (1.0 + np.exp(-noop_logits_np))
        raw = np.concatenate([noop_probs[:, None], pair_probs], axis=1)
        raw = np.clip(raw, 1e-6, None)
        policy_probs = raw / raw.sum(axis=1, keepdims=True)
    # Optional: override noop probability with a fixed value to encourage more
    # action exploration. Calibrated noop ~0.95 is data-correct but pessimistic
    # as an MCTS prior. Pair priors stretch to (1 - NOOP_OVERRIDE).
    if NOOP_OVERRIDE > 0 and override_noop:
        pair_part = policy_probs[:, 1:]
        pair_sum = pair_part.sum(axis=1, keepdims=True).clip(min=1e-6)
        new_pair = pair_part / pair_sum * (1.0 - NOOP_OVERRIDE)
        policy_probs = np.concatenate(
            [np.full((policy_probs.shape[0], 1), NOOP_OVERRIDE, dtype=policy_probs.dtype), new_pair],
            axis=1,
        )
    return policy_probs, value, list(pids)


def _pair_probs(state, player, override_noop=True):
    """Returns (policy_probs[N, N+1], planet_ids)."""
    policy, _v, pids = _root_forward(state, player, override_noop=override_noop)
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


def _legal(src, tgt, ships, state):
    angle, eta = _lead_angle(src, tgt, ships, state)
    speed = max(_BD.fleet_speed(ships), 1.0)
    lx = src["x"] + (src["radius"] + speed) * math.cos(angle)
    ly = src["y"] + (src["radius"] + speed) * math.sin(angle)
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
        speed = max(_BD.fleet_speed(ships), 1.0)
        lx = src["x"] + (src["radius"] + speed) * math.cos(angle)
        ly = src["y"] + (src["radius"] + speed) * math.sin(angle)
        s["fleets"].append({"id": next_fid, "owner": player,
                            "x": lx, "y": ly, "angle": angle, "ships": ships})
        next_fid += 1
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


def _value_from_model(state, player):
    """Run the model's value head on this state. ~5 ms per call on CPU.
    Returns a scalar in [-1, 1] (positive = `player` is winning).
    """
    feats, globals_, pids = _BD.extract_per_player(state, player, _OWNER_CHANGE_TURN)
    n = feats.shape[0]
    pf = np.zeros((1, _BD.N_MAX, _CKPT["f_planet"]), dtype=np.float32); pf[0, :n] = feats
    gl = globals_.reshape(1, -1).astype(np.float32)
    mk = np.zeros((1, _BD.N_MAX), dtype=bool); mk[0, :n] = True
    raw_xy = np.zeros((1, _BD.N_MAX, 7, 2), dtype=np.float32)
    raw_ships = np.zeros((1, _BD.N_MAX), dtype=np.float32)
    raw_prod = np.zeros((1, _BD.N_MAX), dtype=np.float32)
    for i, p in enumerate(state["planets"]):
        for j, h in enumerate((0, 1, 2, 5, 10, 20, 30)):
            pos = _BD.planet_pos_at(state, p, h)
            raw_xy[0, i, j] = pos if pos is not None else (p["x"], p["y"])
        raw_ships[0, i] = p["ships"]
        raw_prod[0, i] = p["prod"]
    pf_n, gl_n = apply_norm(pf, gl, _CKPT["p_mean"], _CKPT["p_std"], _CKPT["g_mean"], _CKPT["g_std"])
    if _CKPT.get("drop_turn_features") and _CKPT.get("global_names"):
        names = list(_CKPT["global_names"])
        for j, name in enumerate(names):
            if name.startswith("turn_") or name.startswith("phase_"):
                gl_n[:, j] = 0.0
    with torch.no_grad():
        _, value, _ = _MODEL(
            torch.from_numpy(pf_n).to(_DEVICE),
            torch.from_numpy(gl_n).to(_DEVICE),
            torch.from_numpy(mk).to(_DEVICE),
            raw_xy=torch.from_numpy(raw_xy).to(_DEVICE),
            raw_ships=torch.from_numpy(raw_ships).to(_DEVICE),
            raw_prod=torch.from_numpy(raw_prod).to(_DEVICE),
            return_value=True, return_noop=True,
        )
    return float(value.cpu().numpy()[0])


# ---------------------------------------------------------------------------
# DUCT search (depth-1)
# ---------------------------------------------------------------------------


def _build_candidates(state, policy, pids, player):
    """For each `player`-owned planet, returns:
       (src_pid, ships, noop_prior, [ (tgt_pid, prob, angle, eta) ...sorted desc, top-K legal ])

    `policy[i]` is a length (N+1) probability distribution: index 0 = noop,
    index 1+j = launch to planet j. Sums to 1 — no renormalization needed.

    All-or-nothing launching: when a source acts, it sends ALL its ships. The
    decision of whether the launch is "valid" is delegated to the model
    policy + value head. Filtering candidates by ship-need is a leaky heuristic
    that gives up the model's ability to learn launch-size implications.
    """
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    pid_to_planet = {p["id"]: p for p in state["planets"]}
    candidates = []
    for src_pid in pids:
        src = pid_to_planet[src_pid]
        if src["owner"] != player or src["ships"] < MIN_SHIPS:
            continue
        i = pid_to_idx[src_pid]
        ships = int(src["ships"])
        # policy[i] is shape (N+1,); pair priors are policy[i, 1:]
        pair_pri = policy[i, 1:]
        order = np.argsort(-pair_pri)
        opts = []
        for j in order:
            if i == j: continue
            p = float(pair_pri[j])
            if p < FLOOR: break
            tgt = pid_to_planet[int(pids[j])]
            ok, angle, eta = _legal(src, tgt, ships, state)
            if not ok: continue
            opts.append((int(pids[j]), p, angle, eta))
            if len(opts) >= TOPK: break
        if opts:
            candidates.append((src_pid, ships, float(policy[i, 0]), opts))
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
    my_policy, my_pids = _pair_probs(state, my_player, override_noop=True)
    opp_player = 1 - my_player
    # For opp prediction, use the model's calibrated policy without exploration overrides.
    opp_policy, opp_pids = _pair_probs(state, opp_player, override_noop=False)
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

    # For each source: candidate actions are [noop, opt0, opt1, ...]
    # stats[src][k] = [Q_sum, N, prior].
    # Priors come from the trained policy softmax which is already a proper
    # distribution over (noop, all targets). We do a final mini-renormalize
    # because we truncate to top-K legal targets (drops some mass).
    def _init(cands):
        out = {}
        for src_pid, ships, noop_prior, opts in cands:
            raw = [max(noop_prior, 1e-6)] + [max(o[1], 1e-6) for o in opts]
            tot = sum(raw)
            priors = [x / tot for x in raw]
            stats = [[0.0, 0, priors[k]] for k in range(len(raw))]
            out[src_pid] = (ships, opts, stats)
        return out

    my_stats = _init(my_cands)
    opp_stats = _init(opp_cands)
    total_iters = 0

    def _select_side(stats_dict, player_id, joint_out, picks_out):
        for src_pid, (ships, opts, stats) in stats_dict.items():
            total_n = sum(s[1] for s in stats) + 1
            k = _ucb_select([(s[0] / max(s[1], 1), s[1], s[2]) for s in stats], C_PUCT, total_n)
            picks_out[src_pid] = k
            if k == 0:
                continue
            tgt_pid, _p, angle, _eta = opts[k - 1]
            joint_out.append((player_id, src_pid, tgt_pid, ships, angle, _eta))

    while total_iters < MAX_ITERS and time.perf_counter() < deadline:
        total_iters += 1
        joint = []
        my_picks: dict[int, int] = {}; opp_picks: dict[int, int] = {}
        _select_side(my_stats, my_player, joint, my_picks)
        _select_side(opp_stats, opp_player, joint, opp_picks)

        s_next = _apply_joint(state, joint)
        if PROJECT_TO_ARRIVALS:
            new_fleet_count = len(s_next["fleets"]) - len(state["fleets"])
            new_fleets = s_next["fleets"][-new_fleet_count:] if new_fleet_count > 0 else []
            s_eval = _project_to_arrivals(
                s_next, flat=flat, prebuilt_events=prebuilt_events, new_fleets=new_fleets,
            )
        else:
            s_eval = s_next
        val = _value_from_model(s_eval, my_player) if USE_VALUE else _evaluate(s_eval, my_player)

        for src_pid, k in my_picks.items():
            stats = my_stats[src_pid][2]
            stats[k][1] += 1
            stats[k][0] += val
        for src_pid, k in opp_picks.items():
            stats = opp_stats[src_pid][2]
            stats[k][1] += 1
            stats[k][0] -= val  # opp wants to minimize my val

    # final pick: per source, choose the most-visited child (DUCT convention).
    # Tiebreaker: higher avg Q wins, then non-noop wins (k>0). Without this
    # the value head's "all-leaves-look-similar" plateaus end up picking noop
    # because k=0 is first in argmax order.
    actions = []
    for src_pid, (ships, opts, stats) in my_stats.items():
        def _key(i):
            n = stats[i][1]
            q = stats[i][0] / max(n, 1)
            return (n, q, 1 if i > 0 else 0)
        best_k = max(range(len(stats)), key=_key)
        if best_k == 0: continue
        tgt_pid, prob, angle, eta = opts[best_k - 1]
        actions.append([int(src_pid), float(angle), int(ships)])
    return actions, total_iters


# ---------------------------------------------------------------------------
# Kaggle entrypoint
# ---------------------------------------------------------------------------


def agent(obs, config=None):
    try:
        deadline = time.perf_counter() + BUDGET_MS / 1000.0
        state = _BD.parse_state(obs)
        my_player = int(obs.get("player", 0))
        actions, n_iters = _duct_search(state, my_player, deadline)
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
