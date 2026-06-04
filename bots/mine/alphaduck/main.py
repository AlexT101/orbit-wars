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
  ALPHADUCK_FLOOR        skip target candidates with P < floor (default 0.2)
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
from set_net import apply_norm
from pair_net import PlanetTransformerPair
import fastsim

# tuning knobs
BUDGET_MS = float(os.environ.get("ALPHADUCK_BUDGET_MS", "400"))
TOPK = int(os.environ.get("ALPHADUCK_TOPK", "10"))
C_PUCT = float(os.environ.get("ALPHADUCK_C_PUCT", "1.4"))
MAX_ITERS = int(os.environ.get("ALPHADUCK_MAX_ITERS", "600"))
FLOOR = float(os.environ.get("ALPHADUCK_FLOOR", "0.2"))
MIN_SHIPS = int(os.environ.get("ALPHADUCK_MIN_SHIPS", "2"))

DEFAULT_CKPT = os.environ.get(
    "PAIR_NET_CKPT",
    str(ROOT / "bots" / "mine" / "target_predictor" / "train" / "weights" / "transformer_pair.pt"),
)

_CKPT = None
_MODEL = None
_DEVICE = torch.device(os.environ.get("PAIR_NET_DEVICE", "cpu"))


def _load_model():
    global _CKPT, _MODEL
    if _MODEL is not None:
        return
    ck = torch.load(DEFAULT_CKPT, map_location=_DEVICE, weights_only=False)
    m = PlanetTransformerPair(
        ck["f_planet"], ck["f_global"],
        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
        n_layers=ck.get("n_layers", 2), ff=ck.get("ff", 128), dropout=0.0,
    ).to(_DEVICE).eval()
    m.load_state_dict(ck["state_dict"])
    _CKPT = ck
    _MODEL = m
    fastsim.warmup()


def _pair_probs(state, player) -> tuple[np.ndarray, list[int]]:
    """Returns (P[N,N] from this player's POV, planet_ids in row order)."""
    fake_change = {p["id"]: 0 for p in state["planets"]}
    feats, globals_, pids = bd.extract_per_player(state, player, fake_change)
    n = feats.shape[0]
    pf = np.zeros((1, bd.N_MAX, _CKPT["f_planet"]), dtype=np.float32); pf[0, :n] = feats
    gl = globals_.reshape(1, -1).astype(np.float32)
    mk = np.zeros((1, bd.N_MAX), dtype=bool); mk[0, :n] = True
    pf_n, gl_n = apply_norm(pf, gl, _CKPT["p_mean"], _CKPT["p_std"], _CKPT["g_mean"], _CKPT["g_std"])
    with torch.no_grad():
        logits = _MODEL(
            torch.from_numpy(pf_n).to(_DEVICE),
            torch.from_numpy(gl_n).to(_DEVICE),
            torch.from_numpy(mk).to(_DEVICE),
        ).cpu().numpy()[0]
    return 1.0 / (1.0 + np.exp(-logits[:n, :n])), list(pids)


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
    speed = max(bd.fleet_speed(ships), 1.0)
    angle = math.atan2(tgt["y"] - src["y"], tgt["x"] - src["x"])
    eta = max(1, int(math.hypot(tgt["x"] - src["x"], tgt["y"] - src["y"]) / speed))
    for _ in range(iters):
        lx = src["x"] + src["radius"] * math.cos(angle)
        ly = src["y"] + src["radius"] * math.sin(angle)
        fp = bd.planet_pos_at(state, tgt, eta)
        if fp is None:
            fp = (tgt["x"], tgt["y"])
        dx = fp[0] - lx; dy = fp[1] - ly
        eta = max(1, int(round(math.hypot(dx, dy) / speed)))
        angle = math.atan2(dy, dx)
    return angle, eta


def _legal(src, tgt, ships, state):
    angle, eta = _lead_angle(src, tgt, ships, state)
    speed = max(bd.fleet_speed(ships), 1.0)
    lx = src["x"] + (src["radius"] + speed) * math.cos(angle)
    ly = src["y"] + (src["radius"] + speed) * math.sin(angle)
    fleet = {"x": lx, "y": ly, "angle": angle, "ships": ships, "owner": -2}
    pred = bd.predict_fleet_collision(state, fleet)
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
        speed = max(bd.fleet_speed(ships), 1.0)
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
            speed = max(bd.fleet_speed(f["ships"]), 1.0)
            dest_pid, eta = fastsim.predict_one_fleet_fast(
                flat, f["x"], f["y"], f["angle"], speed,
            )
            if dest_pid is None:
                continue
            events.append((eta, dest_pid, f["owner"], f["ships"]))
    if prebuilt_events is None and new_fleets is None:
        for f in s["fleets"]:
            pred = bd.predict_fleet_collision(s, f)
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
    """Heuristic value of `state` from `player`'s POV."""
    my_p = my_pr = my_s = 0; en_p = en_pr = en_s = 0
    for p in state["planets"]:
        if p["owner"] == player:
            my_p += 1; my_pr += p["prod"]; my_s += p["ships"]
        elif p["owner"] >= 0:
            en_p += 1; en_pr += p["prod"]; en_s += p["ships"]
    return (my_p - en_p) * 5.0 + (my_pr - en_pr) * 8.0 + (my_s - en_s) * 0.05


# ---------------------------------------------------------------------------
# DUCT search (depth-1)
# ---------------------------------------------------------------------------


def _build_candidates(state, probs, pids, player):
    """For each `player`-owned planet, returns:
       (src_pid, ships, [ (tgt_pid, prob, angle, eta) ...sorted desc, top-K legal ])
    Plus a per-source `noop` option implicit (index -1).
    Skips planets with < MIN_SHIPS."""
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    pid_to_planet = {p["id"]: p for p in state["planets"]}
    candidates = []
    for src_pid in pids:
        src = pid_to_planet[src_pid]
        if src["owner"] != player or src["ships"] < MIN_SHIPS:
            continue
        i = pid_to_idx[src_pid]
        ships = int(src["ships"])
        order = np.argsort(-probs[i])
        opts = []
        for j in order:
            if i == j: continue
            p = float(probs[i, j])
            if p < FLOOR: break
            tgt = pid_to_planet[int(pids[j])]
            ok, angle, eta = _legal(src, tgt, ships, state)
            if not ok: continue
            opts.append((int(pids[j]), p, angle, eta))
            if len(opts) >= TOPK: break
        if opts:
            candidates.append((src_pid, ships, opts))
    return candidates


def _ucb_select(stats, c_puct, total_visits):
    """stats = list of (Q_mean, N, prior). Returns index with highest PUCT."""
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
    my_probs, my_pids = _pair_probs(state, my_player)
    opp_player = 1 - my_player
    opp_probs, opp_pids = _pair_probs(state, opp_player)
    my_cands = _build_candidates(state, my_probs, my_pids, my_player)
    opp_cands = _build_candidates(state, opp_probs, opp_pids, opp_player)

    # ---- per-turn cache: flatten planet geometry once and pre-predict
    # destinations for all fleets already in flight. These don't change across
    # iterations; only fleets spawned by joint actions vary.
    flat = fastsim.flatten_state(state)
    prebuilt_events = []
    for f in state["fleets"]:
        speed = max(bd.fleet_speed(f["ships"]), 1.0)
        dest_pid, eta = fastsim.predict_one_fleet_fast(
            flat, f["x"], f["y"], f["angle"], speed,
        )
        if dest_pid is None:
            continue
        prebuilt_events.append((eta, dest_pid, f["owner"], f["ships"]))

    # For each source: candidate actions are [noop, opt0, opt1, ...]
    # stats[src][k] = [Q_sum, N, prior]
    def _init(cands):
        out = {}
        for src_pid, ships, opts in cands:
            prior_sum = sum(o[1] for o in opts)
            noop_prior = max(1.0 - prior_sum, 1e-3)
            stats = [[0.0, 0, noop_prior]] + [[0.0, 0, o[1]] for o in opts]
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
        new_fleet_count = len(s_next["fleets"]) - len(state["fleets"])
        new_fleets = s_next["fleets"][-new_fleet_count:] if new_fleet_count > 0 else []
        s_final = _project_to_arrivals(
            s_next, flat=flat, prebuilt_events=prebuilt_events, new_fleets=new_fleets,
        )
        val = _evaluate(s_final, my_player)

        for src_pid, k in my_picks.items():
            stats = my_stats[src_pid][2]
            stats[k][1] += 1
            stats[k][0] += val
        for src_pid, k in opp_picks.items():
            stats = opp_stats[src_pid][2]
            stats[k][1] += 1
            stats[k][0] -= val  # opp wants to minimize my val

    # final pick: per source, choose the most-visited child (DUCT convention)
    actions = []
    for src_pid, (ships, opts, stats) in my_stats.items():
        best_k = max(range(len(stats)), key=lambda i: stats[i][1])
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
        state = bd.parse_state(obs)
        my_player = int(obs.get("player", 0))
        actions, n_iters = _duct_search(state, my_player, deadline)
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
