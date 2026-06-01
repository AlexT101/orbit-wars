"""INDEPENDENT cross-check of the summary_v2 extractor.

The user asked: "double check you extracted it right — take some data, run the
extrapolator yourself, then compare with the data results."

So this does NOT call the Rust feature code. It re-implements, from scratch in
pure Python, the ENTIRE pipeline that produced the NPZ:
  - parse_state (planet/fleet/comet decode, orbit-radius derivation)
  - predict_fleet_collision (per-fleet swept orbital collision, the expensive
    "find each ship's target" step)
  - extrapolate_fleets (land every fleet, resolve combat per planet)
  - summary_features_v2 (current + extrapolated + neutral blocks, 46-d)

Then it feeds the SAME normalized observations to the actual `extract_v2` Rust
binary (the exact program that built replays_strong.npz) and compares all 46
features per observation. Every feature is an integer-valued sum (ships / counts
/ production / comet-time), so a correct extractor must match EXACTLY. Any
mismatch is reported with the offending block + a few examples.

Note: like from_replays_fast.py we strip `config`, so max_speed defaults to 6.0
(this is what actually built the NPZ).
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BIN = REPO / "bots" / "mine" / "alphaow" / "target" / "release" / "extract_v2"
REPLAYS = REPO / "replays"

# ---- engine constants (src/lib.rs, src/pathing.rs) ----
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
BOARD = 100.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0          # config stripped -> default
MAX_TIME = 100
F32 = np.float32


# ----------------------------- parse -----------------------------
def parse_state(o: dict):
    step = int(o.get("step", 0))
    av = float(o.get("angular_velocity", 0.0))
    comet_ids = set(int(x) for x in (o.get("comet_planet_ids") or []))
    init_pos = {}
    for p in (o.get("initial_planets") or []):
        init_pos[int(p[0])] = (float(p[2]), float(p[3]))
    planets = []
    for p in (o.get("planets") or []):
        pid = int(p[0]); owner = int(p[1])
        x = float(p[2]); y = float(p[3]); radius = float(p[4])
        ships = int(p[5]); prod = int(p[6])
        is_comet = pid in comet_ids
        ix, iy = init_pos.get(pid, (x, y))
        dx = ix - CENTER[0]; dy = iy - CENTER[1]
        orb_r = math.sqrt(dx * dx + dy * dy)
        init_angle = math.atan2(dy, dx)
        is_orbiting = (not is_comet) and (orb_r + radius < ROT_LIMIT)
        planets.append(dict(id=pid, owner=owner, x=x, y=y, radius=radius,
                            ships=ships, prod=prod, orb_r=orb_r,
                            init_angle=init_angle, is_orbiting=is_orbiting,
                            is_comet=is_comet))
    fleets = []
    for f in (o.get("fleets") or []):
        fleets.append(dict(id=int(f[0]), owner=int(f[1]), x=float(f[2]),
                           y=float(f[3]), angle=float(f[4]), ships=int(f[6])))
    comets = []
    for g in (o.get("comets") or []):
        pids = [int(x) for x in g["planet_ids"]]
        paths = [[(float(pt[0]), float(pt[1])) for pt in path] for path in g["paths"]]
        comets.append(dict(planet_ids=pids, paths=paths, path_index=int(g["path_index"])))
    return dict(player=int(o.get("player", 0)), step=step, av=av,
                planets=planets, fleets=fleets, comets=comets)


def comet_group_for(state, cid):
    for g in state["comets"]:
        if cid in g["planet_ids"]:
            return g, g["planet_ids"].index(cid)
    return None


def comet_remaining(state, planet):
    if not planet["is_comet"]:
        return 0
    gi = comet_group_for(state, planet["id"])
    if gi is None:
        return 0
    g, i = gi
    return max(len(g["paths"][i]) - g["path_index"], 0)


def planet_pos_at(state, planet, dt):
    if planet["is_comet"]:
        gi = comet_group_for(state, planet["id"])
        if gi is None:
            return None
        g, i = gi
        idx = g["path_index"] + dt
        if idx < 0 or idx >= len(g["paths"][i]):
            return None
        return g["paths"][i][idx]
    if planet["is_orbiting"]:
        abs_step = max(state["step"] + dt - 1, 0)
        a = planet["init_angle"] + state["av"] * abs_step
        return (CENTER[0] + planet["orb_r"] * math.cos(a),
                CENTER[1] + planet["orb_r"] * math.sin(a))
    return (planet["x"], planet["y"])


# ----------------------------- physics -----------------------------
def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(max(s, 1.0), MAX_SPEED)


def on_board(p):
    return 0.0 <= p[0] <= BOARD and 0.0 <= p[1] <= BOARD


def pt_seg_dist(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 < 1e-12:
        return math.dist(p, v)
    t = max(0.0, min(1.0, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    proj = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return math.dist(p, proj)


def swept_pair_hit(a, b, p0, p1, r):
    d0x = a[0] - p0[0]; d0y = a[1] - p0[1]
    dvx = (b[0] - a[0]) - (p1[0] - p0[0])
    dvy = (b[1] - a[1]) - (p1[1] - p0[1])
    aq = dvx * dvx + dvy * dvy
    bq = 2.0 * (d0x * dvx + d0y * dvy)
    cq = d0x * d0x + d0y * d0y - r * r
    if aq < 1e-12:
        return cq <= 0.0
    disc = bq * bq - 4.0 * aq * cq
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-bq - sq) / (2.0 * aq)
    t2 = (-bq + sq) / (2.0 * aq)
    return t2 >= 0.0 and t1 <= 1.0


def predict_fleet_collision(state, fleet):
    speed = fleet_speed(fleet["ships"])
    dx = speed * math.cos(fleet["angle"]); dy = speed * math.sin(fleet["angle"])
    pos = (fleet["x"], fleet["y"])
    for dt in range(1, MAX_TIME + 1):
        new_pos = (pos[0] + dx, pos[1] + dy)
        for planet in state["planets"]:        # obs order matters (first hit wins)
            p_old = planet_pos_at(state, planet, dt - 1)
            if p_old is None:
                continue
            p_new = planet_pos_at(state, planet, dt)
            if p_new is None:
                continue
            if not on_board(p_old) and not on_board(p_new):
                continue
            if swept_pair_hit(pos, new_pos, p_old, p_new, planet["radius"]):
                return (planet["id"], dt)
        if not on_board(new_pos):
            return None
        if pt_seg_dist(CENTER, pos, new_pos) < SUN_RADIUS:
            return None
        pos = new_pos
    return None


def extrapolate_fleets(state):
    arrivals = {}
    for fleet in state["fleets"]:
        pred = predict_fleet_collision(state, fleet)
        if pred is not None:
            pid, dt = pred
            arrivals.setdefault(pid, []).append((dt, fleet["owner"], fleet["ships"]))
    result = {p["id"]: (p["owner"], p["ships"]) for p in state["planets"]}
    for pid, arrs in arrivals.items():
        arrs.sort(key=lambda x: x[0])
        owner, ships = result.get(pid, (-1, 0))
        for _t, f_owner, f_ships in arrs:
            if f_owner == owner:
                ships += f_ships
            elif f_ships > ships:
                owner = f_owner
                ships = f_ships - ships
            else:
                ships -= f_ships
        result[pid] = (owner, ships)
    return result


# ----------------------------- features -----------------------------
def min_dist_to(planets, ox, oy, pred):
    best = math.inf
    for q in planets:
        if not pred(q):
            continue
        dx = F32(q["x"] - ox); dy = F32(q["y"] - oy)
        d = F32(math.sqrt(F32(dx * dx + dy * dy)))
        if d < best:
            best = d
    return best


def current_block(state, p):
    planets = state["planets"]
    ships_pl = n_st = n_or = n_co = pr_st = pr_or = pr_co = 0.0
    for pl in planets:
        if pl["owner"] != p:
            continue
        ships_pl += pl["ships"]; prod = pl["prod"]
        if pl["is_comet"]:
            n_co += 1; pr_co += prod
        elif pl["is_orbiting"]:
            n_or += 1; pr_or += prod
        else:
            n_st += 1; pr_st += prod
    ships_fly = sum(f["ships"] for f in state["fleets"] if f["owner"] == p)
    n_neut = n_en = 0.0
    for o in planets:
        if o["owner"] == -1:
            dme = min_dist_to(planets, o["x"], o["y"], lambda q: q["owner"] == p)
            den = min_dist_to(planets, o["x"], o["y"], lambda q: q["owner"] != p and q["owner"] != -1)
            if dme < den:
                n_neut += 1
        elif o["owner"] != p:
            dme = min_dist_to(planets, o["x"], o["y"], lambda q: q["owner"] == p)
            doth = min_dist_to(planets, o["x"], o["y"],
                               lambda q: q["owner"] != p and q["owner"] != -1 and q["id"] != o["id"])
            if dme < doth:
                n_en += 1
    return [ships_pl, ships_fly, n_st, n_or, n_co, pr_st, pr_or, pr_co, n_neut, n_en]


def extrap_block(state, p, extrap):
    planets = state["planets"]
    def owner_of(pid):
        if pid in extrap:
            return extrap[pid][0]
        for q in planets:
            if q["id"] == pid:
                return q["owner"]
        return -1
    ships_pl = n_st = n_or = n_co = pr_st = pr_or = pr_co = 0.0
    for pl in planets:
        eo, es = extrap.get(pl["id"], (pl["owner"], pl["ships"]))
        if eo != p:
            continue
        ships_pl += es; prod = pl["prod"]
        if pl["is_comet"]:
            n_co += 1; pr_co += prod
        elif pl["is_orbiting"]:
            n_or += 1; pr_or += prod
        else:
            n_st += 1; pr_st += prod
    n_neut = n_en = 0.0
    for o in planets:
        eo = owner_of(o["id"])
        if eo == -1:
            dme = min_dist_to(planets, o["x"], o["y"], lambda q: owner_of(q["id"]) == p)
            den = min_dist_to(planets, o["x"], o["y"],
                              lambda q: owner_of(q["id"]) != p and owner_of(q["id"]) != -1)
            if dme < den:
                n_neut += 1
        elif eo != p:
            dme = min_dist_to(planets, o["x"], o["y"], lambda q: owner_of(q["id"]) == p)
            doth = min_dist_to(planets, o["x"], o["y"],
                               lambda q: owner_of(q["id"]) != p and owner_of(q["id"]) != -1 and q["id"] != o["id"])
            if dme < doth:
                n_en += 1
    return [ships_pl, n_st, n_or, n_co, pr_st, pr_or, pr_co, n_neut, n_en]


def neutral_block(state):
    ships = n_st = n_or = n_co = pr_st = pr_or = pr_co = ctime = 0.0
    for pl in state["planets"]:
        if pl["owner"] == -1:
            ships += pl["ships"]; prod = pl["prod"]
            if pl["is_comet"]:
                n_co += 1; pr_co += prod
            elif pl["is_orbiting"]:
                n_or += 1; pr_or += prod
            else:
                n_st += 1; pr_st += prod
        if pl["is_comet"]:
            ctime += comet_remaining(state, pl)
    return [ships, n_st, n_or, n_co, pr_st, pr_or, pr_co, ctime]


def dominant_enemy(state, me):
    """Faithful port of value_net.rs: strongest enemy by total ships
    (planets + fleets), ties -> first encountered in planet order. Falls
    back to the other slot when `me` has no opponents on the board."""
    totals = {}
    best_owner = None
    best_total = 0
    for pl in state["planets"]:
        o = pl["owner"]
        if o == -1 or o == me:
            continue
        if o not in totals:
            totals[o] = (sum(p["ships"] for p in state["planets"] if p["owner"] == o)
                         + sum(f["ships"] for f in state["fleets"] if f["owner"] == o))
        t = totals[o]
        if best_owner is None:
            best_owner, best_total = o, t
        elif t > best_total or best_owner == o:
            best_owner, best_total = o, t
    if best_owner is None:
        return 1 if me == 0 else 0
    return best_owner


def py_extract(o: dict):
    state = parse_state(o)
    me = state["player"]; opp = dominant_enemy(state, me)
    extrap = extrapolate_fleets(state)
    feats = (current_block(state, me) + current_block(state, opp)
             + extrap_block(state, me, extrap) + extrap_block(state, opp, extrap)
             + neutral_block(state))
    return np.array(feats, dtype=np.float32), len(state["fleets"])


# ----------------------------- driver -----------------------------
def normalize_obs(o):
    return {
        "player": int(o.get("player", 0)),
        "step": int(o.get("step", 0)),
        "planets": list(o.get("planets", []) or []),
        "fleets": list(o.get("fleets", []) or []),
        "angular_velocity": float(o.get("angular_velocity", 0.0)),
        "initial_planets": list(o.get("initial_planets", []) or []),
        "comets": list(o.get("comets", []) or []),
        "comet_planet_ids": list(o.get("comet_planet_ids", []) or []),
    }


RECORD = 8 + 4 + 4 * 46


def main():
    files = sorted(REPLAYS.glob("*.json"))
    obs_list = []
    n_skip4p = 0
    for path in files:
        data = json.loads(path.read_bytes())
        # from_replays_fast.py only keeps 2-player games (len(rewards)==2);
        # 4-player replays are never in the NPZ, so skip them here too.
        if len(data.get("rewards") or []) != 2:
            n_skip4p += 1
            continue
        for step in (data.get("steps") or []):
            if not isinstance(step, list):
                continue
            for slot in range(2):
                if slot >= len(step) or not isinstance(step[slot], dict):
                    continue
                ob = step[slot].get("observation")
                if ob and ob.get("planets"):
                    obs_list.append(normalize_obs(ob))
    print(f"{len(files)} replays ({n_skip4p} skipped: not 2-player) -> {len(obs_list)} observations")

    # ---- run the actual Rust extractor on the SAME observations ----
    proc = subprocess.Popen([str(BIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    payload = b"".join(json.dumps(o, separators=(",", ":")).encode() + b"\n" for o in obs_list)
    out, _ = proc.communicate(payload)
    n_rec = len(out) // RECORD
    rust = np.zeros((n_rec, 46), dtype=np.float32)
    rust_step = np.zeros(n_rec, dtype=np.int64)
    rust_player = np.zeros(n_rec, dtype=np.int64)
    for i in range(n_rec):
        rec = out[i * RECORD:(i + 1) * RECORD]
        rust_step[i] = struct.unpack_from("<q", rec, 0)[0]
        rust_player[i] = struct.unpack_from("<i", rec, 8)[0]
        rust[i] = np.frombuffer(rec[12:], dtype=np.float32)
    assert n_rec == len(obs_list), f"binary returned {n_rec} for {len(obs_list)} obs"

    # ---- independent Python extraction + compare ----
    BLOCKS = [("me_cur", 0, 10), ("opp_cur", 10, 20), ("me_ext", 20, 29),
              ("opp_ext", 29, 38), ("neutral", 38, 46)]
    n_fleet_rows = 0
    block_mismatch = {b[0]: 0 for b in BLOCKS}
    examples = []
    max_abs = np.zeros(46)
    for i, o in enumerate(obs_list):
        py, nf = py_extract(o)
        if nf > 0:
            n_fleet_rows += 1
        diff = np.abs(py - rust[i])
        max_abs = np.maximum(max_abs, diff)
        if diff.max() > 1e-3:
            for name, lo, hi in BLOCKS:
                if diff[lo:hi].max() > 1e-3:
                    block_mismatch[name] += 1
            if len(examples) < 8:
                bad = int(np.argmax(diff))
                examples.append((i, int(rust_player[i]), bad, float(py[bad]), float(rust[i][bad]), nf))
    n = len(obs_list)
    print(f"observations with >=1 fleet in flight: {n_fleet_rows}/{n} "
          f"({100*n_fleet_rows/max(n,1):.0f}%)  -> extrapolation is exercised")
    print("\nper-block EXACT-match check (Python re-impl vs Rust extract_v2 binary):")
    any_bad = False
    for name, lo, hi in BLOCKS:
        mm = block_mismatch[name]
        worst = max_abs[lo:hi].max()
        status = "OK" if mm == 0 else f"MISMATCH in {mm} obs"
        if mm:
            any_bad = True
        print(f"  {name:<9} cols[{lo:>2}:{hi:<2}]  max|Δ|={worst:>10.4f}   {status}")
    print(f"\noverall max|Δ| over all 46 features, all {n} obs = {max_abs.max():.6f}")
    if not any_bad:
        print("RESULT: PERFECT MATCH — the Rust extractor (which built the NPZ) is bit-exact\n"
              "        with an independent from-scratch Python re-implementation of the\n"
              "        collision predictor + fleet extrapolation + all 46 features.")
    else:
        print("RESULT: MISMATCH FOUND — examples (obs_idx, player, feat_idx, py, rust, n_fleets):")
        for ex in examples:
            print(f"   obs={ex[0]} player={ex[1]} feat={ex[2]} py={ex[3]} rust={ex[4]} fleets={ex[5]}")


if __name__ == "__main__":
    main()
