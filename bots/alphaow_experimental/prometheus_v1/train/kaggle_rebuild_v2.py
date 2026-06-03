"""KAGGLE REBUILD CELL — rebuild the summary_v2 value-net dataset on Kaggle.

Run this ON Kaggle (in a notebook attached to the orbit-wars episode datasets),
because that is where the raw replays live (/kaggle/input/**/*.json). Locally we
only have a handful of JSONs; the full ~15k-episode collection is Kaggle-side.

What it does:
  1. Scans /kaggle/input/**/*.json
  2. Keeps ONLY 2-player games (len(rewards) == 2) -> drops every 4-player game
     (the source is ~33% 4p, so this is the filter that matters)
  2b. STRONG_ONLY (default True): of those 2p games, keeps only ones where BOTH
     players' win rate is above the median — faithfully reproduces the original
     select_strong_replays.py "strong subset" (that's why the old NPZ was 2728
     games, not all ~5k 2p games). Set STRONG_ONLY=False to keep all 2p games.
  3. Extracts the 46-d summary_v2 features with a PURE-PYTHON extractor that is
     byte-for-byte identical to the Rust `extract_v2` binary that built the
     original replays_strong.npz (verified max|Δ|=0 on all 46 features). No Rust
     toolchain needed on Kaggle.
  4. Saves /kaggle/working/replays_2p_rebuilt.npz with the SAME keys/layout as the
     original: summary_v2 (N,46) f32, labels (N,) f32, meta (N,4) i32
     meta cols = (game_id, step, player, 1-player)

Then CELL 2 trains a symmetric-linear + h64 MLP on the rebuilt set (canonical
seed=42, 12% game-level val split, SmoothL1) and prints sign accuracy so you can
compare to the known baseline: linear ~84.4% / MLP ~84.8%.

SPEED NOTE: the pure-Python collision predictor is ~100x slower than the Rust
binary. Set MAX_GAMES to bound a single Kaggle session (default 4000 ≈ the
original 2728-game scale). Set MAX_GAMES = None to process everything (will take
hours; use the persistent/background session). N_WORKERS uses all CPUs.

PERSPECTIVE NOTE: like the original pipeline we strip `config`, so max fleet
speed defaults to 6.0 — this is exactly what built the deployed net. Do not
"fix" this to read config, or the features stop matching the trained net.
"""

# %% [cell 1] ----------------------------------------------------------------
import os, json, math, time, struct, pathlib, random
import multiprocessing as mp
from collections import defaultdict
import numpy as np

INPUT_ROOT = pathlib.Path("/kaggle/input")
OUT_NPZ    = pathlib.Path("/kaggle/working/replays_2p_rebuilt.npz")
MAX_GAMES  = 4000          # cap games sent to (slow) extraction; set None for all
N_WORKERS  = os.cpu_count() or 4

# --- strong-player gate (reproduces select_strong_replays.py) ---
STRONG_ONLY = True         # keep only games where BOTH players' win rate is above median
MIN_GAMES   = 3            # a player needs >= this many games for their win rate to count
SEED        = 0            # shuffle seed for sampling the strong subset (select_strong default)

# ---- engine constants (src/lib.rs, src/pathing.rs) ----
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
BOARD = 100.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0            # config stripped -> default (matches the deployed net)
MAX_TIME = 100
F32 = np.float32


# ----------------------------- parse -----------------------------
def parse_state(o):
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


def py_extract(o):
    state = parse_state(o)
    me = state["player"]; opp = dominant_enemy(state, me)
    extrap = extrapolate_fleets(state)
    feats = (current_block(state, me) + current_block(state, opp)
             + extrap_block(state, me, extrap) + extrap_block(state, opp, extrap)
             + neutral_block(state))
    return np.array(feats, dtype=np.float32)


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


# --------------------------- build driver ---------------------------
def process_game(args):
    """One replay -> (feats, labels, meta_local, kept_flag). meta_local game id
    is a placeholder (0); the parent renumbers games globally."""
    path_str, = args
    try:
        data = json.loads(pathlib.Path(path_str).read_bytes())
    except Exception:
        return None
    rewards = data.get("rewards") or []
    steps = data.get("steps") or []
    # THE FILTER: 2-player only. Drops every 4p game (source is ~33% 4p).
    if len(rewards) != 2 or not steps:
        return None
    if any(r is None for r in rewards):
        return None

    feats, labels, meta = [], [], []
    for step in steps:
        if not isinstance(step, list) or len(step) < 2:
            continue
        for slot in range(2):
            entry = step[slot]
            if not isinstance(entry, dict):
                continue
            obs = entry.get("observation")
            if not obs or not obs.get("planets"):
                continue
            norm = normalize_obs(obs)
            feats.append(py_extract(norm))
            labels.append(float(rewards[slot]))
            meta.append((0, norm["step"], norm["player"], 1 - norm["player"]))
    if not feats:
        return None
    return (np.stack(feats).astype(np.float32),
            np.array(labels, dtype=np.float32),
            np.array(meta, dtype=np.int32))


def build():
    all_json = [str(p) for p in INPUT_ROOT.rglob("*.json")]
    print(f"found {len(all_json)} json under {INPUT_ROOT}")

    # ---- pass 1: cheap scan of rewards + agent names; count 2p vs 4p ----
    games = []           # (path, name0, name1, reward0, reward1)
    n4p = nother = 0
    for p in all_json:
        try:
            d = json.loads(pathlib.Path(p).read_bytes())
        except Exception:
            nother += 1; continue
        rewards = d.get("rewards") or []
        if len(rewards) == 4:
            n4p += 1; continue                       # <-- the 4p filter
        agents = (d.get("info") or {}).get("Agents") or []
        if (len(rewards) != 2 or len(agents) != 2
                or any(r is None for r in rewards) or not d.get("steps")):
            nother += 1; continue
        names = [a.get("Name", f"p{i}") for i, a in enumerate(agents)]
        games.append((p, names[0], names[1], float(rewards[0]), float(rewards[1])))
    print(f"2-player games: {len(games)}   4-player games filtered: {n4p}   "
          f"other/invalid skipped: {nother}")

    # ---- strong-player gate: keep games where BOTH players are above median win rate ----
    if STRONG_ONLY:
        pg, pw = defaultdict(int), defaultdict(int)
        for _, n0, n1, r0, r1 in games:
            pg[n0] += 1; pg[n1] += 1
            if r0 > r1:   pw[n0] += 1
            elif r1 > r0: pw[n1] += 1
        rates = {pl: pw[pl] / pg[pl] for pl in pg if pg[pl] >= MIN_GAMES}
        sr = sorted(rates.values())
        median = sr[len(sr) // 2] if sr else 0.0
        above = {pl for pl, r in rates.items() if r > median}
        strong = [g for g in games if g[1] in above and g[2] in above]
        print(f"win-rate median={median:.3f} over {len(rates)} players "
              f"(min {MIN_GAMES} games); {len(above)} above median; "
              f"{len(strong)} games with BOTH above-median")
        random.Random(SEED).shuffle(strong)
        selected = [g[0] for g in strong]
    else:
        selected = [g[0] for g in games]

    if MAX_GAMES:
        selected = selected[:MAX_GAMES]
    print(f"selected {len(selected)} games for feature extraction")

    # ---- pass 2: expensive bit-exact feature extraction on the selected subset ----
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(process_game, [(p,) for p in selected], chunksize=4)

    feats_all, labels_all, meta_all = [], [], []
    gid = 0
    for res in results:
        if res is None:
            continue
        f, lbl, m = res
        m = m.copy(); m[:, 0] = gid; gid += 1
        feats_all.append(f); labels_all.append(lbl); meta_all.append(m)

    feats = np.concatenate(feats_all)
    labels = np.concatenate(labels_all)
    meta = np.concatenate(meta_all)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, summary_v2=feats, labels=labels, meta=meta)
    el = time.time() - t0
    print(f"\nwrote {feats.shape[0]:,} samples from {gid:,} 2p games to {OUT_NPZ} in {el:.0f}s")
    print(f"  player slots present in meta: {sorted(np.unique(meta[:,2]).tolist())}  (must be [0, 1])")
    print(f"  label values: {sorted(np.unique(labels).tolist())}")


if __name__ == "__main__":
    build()


# %% [cell 2] — TRAIN + EVAL (run after cell 1) ------------------------------
def train_and_eval(npz_path="/kaggle/working/replays_2p_rebuilt.npz"):
    import torch, torch.nn as nn
    d = np.load(npz_path)
    X = torch.tensor(d["summary_v2"], dtype=torch.float32)
    y = torch.tensor(np.sign(d["labels"]).astype(np.float32))   # ±1
    games = d["meta"][:, 0]

    # canonical seed=42, 12% game-level holdout (split by game, not by row)
    rng = np.random.default_rng(42)
    uniq = np.unique(games)
    rng.shuffle(uniq)
    n_val = int(round(0.12 * len(uniq)))
    val_games = set(uniq[:n_val].tolist())
    val_mask = np.array([g in val_games for g in games])
    tr = torch.tensor(~val_mask); va = torch.tensor(val_mask)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"train {Xtr.shape[0]:,} rows / {len(uniq)-n_val} games | "
          f"val {Xva.shape[0]:,} rows / {n_val} games")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr, Xva, yva = Xtr.to(dev), ytr.to(dev), Xva.to(dev), yva.to(dev)
    huber = nn.SmoothL1Loss()

    def fit(model, epochs, lr):
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            loss = huber(model(Xtr).squeeze(-1), ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            pv = model(Xva).squeeze(-1)
            sign = ((pv > 0) == (yva > 0)).float().mean().item()
        return sign

    # --- symmetric linear: model on difference features (mine - theirs) ---
    # mirror pairs: (i, i+10) for i in 0..10 ; (20+i, 29+i) for i in 0..9
    pairs = [(i, i + 10) for i in range(10)] + [(20 + i, 29 + i) for i in range(9)]
    mine = torch.tensor([a for a, _ in pairs]); theirs = torch.tensor([b for _, b in pairs])
    Dtr = (Xtr[:, mine] - Xtr[:, theirs]); Dva = (Xva[:, mine] - Xva[:, theirs])
    lin = nn.Sequential(nn.Linear(len(pairs), 1), nn.Tanh())
    # reuse fit() by swapping data
    opt = torch.optim.Adam(lin.to(dev).parameters(), lr=0.01)
    for ep in range(8000):
        opt.zero_grad(); loss = huber(lin(Dtr).squeeze(-1), ytr); loss.backward(); opt.step()
    with torch.no_grad():
        sym_sign = ((lin(Dva).squeeze(-1) > 0) == (yva > 0)).float().mean().item()

    # --- h64 MLP (deployed arch) on raw 46-d ---
    mlp = nn.Sequential(nn.Linear(46, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
    mlp_sign = fit(mlp, epochs=4000, lr=0.003)

    print("\n=== rebuilt-dataset val sign accuracy ===")
    print(f"  symmetric linear : {sym_sign*100:.2f}%   (baseline ~84.4%)")
    print(f"  h64 MLP          : {mlp_sign*100:.2f}%   (baseline ~84.8%)")
    print("  -> if these match the baselines, the 84% wall is the FEATURES, not the data.")


# To run training on Kaggle after cell 1:  train_and_eval()
