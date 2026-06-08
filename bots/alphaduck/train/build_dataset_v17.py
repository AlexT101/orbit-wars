"""v17 NPZ builder: planet+fleet tokens, structured attention masks, pair feats.

Output keys (per state):
  planet_feats (N_MAX, F_PLANET=9), planet_mask (N_MAX,), planet_ids (N_MAX,)
  fleet_feats  (F_MAX, F_FLEET=5),  fleet_mask  (F_MAX,), fleet_tgt_pid (F_MAX,)
                                                          fleet_tgt_idx (F_MAX,)   # index into planet array
  globals      (F_GLOBAL=9,)
  pair_feats   (N_MAX, N_MAX, F_PAIR=3)
  policy / value / noop labels matching v15 format
"""
from __future__ import annotations
import sys, io, json, os, time, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# bots/alphaduck/train -> bots/mine/{apollo, target_predictor/train}
BOTS_DIR = HERE.parents[1]
APOLLO_DIR = BOTS_DIR / "mine" / "apollo"
TP_TRAIN_DIR = BOTS_DIR / "mine" / "target_predictor" / "train"
sys.path.insert(0, str(APOLLO_DIR))
sys.path.insert(0, str(TP_TRAIN_DIR))
import build_dataset as bd  # reuse parse_state, predict_fleet_collision, planet_pos_at, fleet_speed
from build_dataset import (
    parse_state, predict_fleet_collision, planet_pos_at, fleet_speed,
    update_owner_history, comet_remaining, F32,
)
import aim_native as apollo_native  # forked aim_eta_batch lives in bots/alphaduck/aim_native

N_MAX = 50           # planets — max observed 44, p99 40
F_MAX = 128          # fleet token groups — covers ~p95; over-cap keeps top-F_MAX by sum_ships
K_MAX = 32           # per-row launch-label slots — covers ~p99 of single-turn launches
GAME_MAX_TURN = 500  # for turns_remaining

PLANET_FEAT_NAMES = [
    "production",
    "ships", "ships_log1p",
    "is_mine", "is_neutral", "is_enemy",
    "is_orbital", "is_stationary", "is_comet",
    "dist_to_edge",
    "fleet_speed_at_full_ships",  # raw, since log of it = log of log of ships (over-compressed)
]
F_PLANET = len(PLANET_FEAT_NAMES)
assert F_PLANET == 11

F_FLEET = 5
FLEET_FEAT_NAMES = [
    "sum_ships", "log1p_sum_ships",
    "turns_to_arrival",
    "is_mine", "is_enemy",
]
assert len(FLEET_FEAT_NAMES) == F_FLEET

F_GLOBAL = 9
GLOBAL_FEAT_NAMES = [
    "sum_my_ships_ground", "sum_enemy_ships_ground",
    "sum_my_prod", "sum_enemy_prod",
    "sum_my_ships_flight", "sum_enemy_ships_flight",
    "omega",
    "comet_remaining_max",   # max remaining lifetime across live comets; 0 if none
    "turns_remaining",
]
assert len(GLOBAL_FEAT_NAMES) == F_GLOBAL

F_PAIR = 4
PAIR_FEAT_NAMES = [
    "distance_now",
    "drift_distance_h1",        # d(a_now, b_at_t+1) - d(a_now, b_now) — derivative-like
    "eta_apollo",               # real intercept flight time via apollo's aim solver; 0 if unreachable
    "is_unreachable",           # one-hot: 1 if apollo's aim returns None
]
assert len(PAIR_FEAT_NAMES) == F_PAIR

BOARD_HALF = 50.0  # board is roughly [0, 100] square per existing features


def _planet_type_onehot(p):
    """orbital, stationary, comet — based on parse_state's is_comet + is_orbiting."""
    if p.get("is_comet"):
        return (0.0, 0.0, 1.0)
    if p.get("is_orbiting"):
        return (1.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0)  # stationary


def _dist_to_edge(p):
    """How far from the nearest board edge (board ~ [0,100])."""
    x, y = p["x"], p["y"]
    return min(x, 100.0 - x, y, 100.0 - y)


def extract_state_v17(state, player, owner_change_turn=None, obs_for_apollo=None, pair_cache=None):
    """Return dict with all v17 features for this state from `player`'s POV.

    obs_for_apollo: the raw kaggle obs dict for this turn — used as input to
    apollo's `aim_eta` function for the pair features.
    pair_cache: optional mutable dict to share pair_feats across both player
    POVs (pair geometry doesn't depend on which player is asking).
    """
    planets = state["planets"]
    fleets = state["fleets"]
    pids = sorted(p["id"] for p in planets)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    n_real = len(pids)

    # ---- planet features
    pf = np.zeros((N_MAX, F_PLANET), dtype=F32)
    pmask = np.zeros(N_MAX, dtype=bool)
    pid_arr = np.zeros(N_MAX, dtype=np.int32)
    for p in planets:
        i = pid_to_idx[p["id"]]
        pmask[i] = True
        pid_arr[i] = p["id"]
        owner = p["owner"]
        is_mine = float(owner == player)
        is_neutral = float(owner == -1)
        is_enemy = float(owner != player and owner != -1)
        t_o, t_s, t_c = _planet_type_onehot(p)
        pf[i, 0] = float(p["prod"])
        pf[i, 1] = float(p["ships"])
        pf[i, 2] = float(np.log1p(p["ships"]))
        pf[i, 3] = is_mine
        pf[i, 4] = is_neutral
        pf[i, 5] = is_enemy
        pf[i, 6] = t_o
        pf[i, 7] = t_s
        pf[i, 8] = t_c
        pf[i, 9] = _dist_to_edge(p)
        # raw fleet speed at current ships — formula is non-linear (log^1.5),
        # giving it as a feature saves the encoder from learning it.
        pf[i, 10] = float(fleet_speed(max(int(p["ships"]), 1)))

    # ---- fleet features: group by (target_pid, arrival_turn, owner)
    groups = defaultdict(lambda: {"sum_ships": 0.0, "owner": None,
                                   "tgt_pid": None, "eta": None})
    for f in fleets:
        pred = predict_fleet_collision(state, f)
        if pred is None:
            continue  # ignore fleets that don't hit a planet
        dst_pid, eta = pred
        if dst_pid not in pid_to_idx:
            continue
        key = (int(dst_pid), int(round(eta)), int(f["owner"]))
        g = groups[key]
        g["sum_ships"] += float(f["ships"])
        g["owner"] = int(f["owner"])
        g["tgt_pid"] = int(dst_pid)
        g["eta"] = int(round(eta))
    # Sort by sum_ships desc, keep top F_MAX
    g_list = sorted(groups.values(), key=lambda g: -g["sum_ships"])[:F_MAX]
    ff = np.zeros((F_MAX, F_FLEET), dtype=F32)
    fmask = np.zeros(F_MAX, dtype=bool)
    f_tgt_pid = np.full(F_MAX, -1, dtype=np.int32)
    f_tgt_idx = np.full(F_MAX, -1, dtype=np.int32)
    for j, g in enumerate(g_list):
        fmask[j] = True
        ff[j, 0] = g["sum_ships"]
        ff[j, 1] = float(np.log1p(g["sum_ships"]))
        ff[j, 2] = float(g["eta"])
        is_mine = float(g["owner"] == player)
        is_enemy = float(g["owner"] != player)
        ff[j, 3] = is_mine
        ff[j, 4] = is_enemy
        f_tgt_pid[j] = g["tgt_pid"]
        f_tgt_idx[j] = pid_to_idx[g["tgt_pid"]]

    # ---- globals
    my_ground = sum(p["ships"] for p in planets if p["owner"] == player)
    en_ground = sum(p["ships"] for p in planets if p["owner"] != player and p["owner"] != -1)
    my_prod = sum(p["prod"] for p in planets if p["owner"] == player)
    en_prod = sum(p["prod"] for p in planets if p["owner"] != player and p["owner"] != -1)
    my_flight = sum(f["ships"] for f in fleets if f["owner"] == player)
    en_flight = sum(f["ships"] for f in fleets if f["owner"] != player)
    # parse_state stores angular velocity under "av" (not "angular_velocity").
    omega = float(state.get("av", state.get("angular_velocity", 0.0)))
    # max remaining lifetime across all live comets (0 if none) — matches v15
    # comet_remaining_max semantics. Per user spec: "turns until comets disappear".
    comet_max = 0
    for p in planets:
        if p["is_comet"]:
            r = comet_remaining(state, p)
            if r > comet_max:
                comet_max = r
    turns_remaining = max(0.0, float(GAME_MAX_TURN - state["step"]))
    gl = np.array([
        my_ground, en_ground, my_prod, en_prod,
        my_flight, en_flight, omega, float(comet_max), turns_remaining,
    ], dtype=F32)

    # ---- pair feats (N_MAX, N_MAX, 4)
    # Pair geometry is player-independent. If we already computed it for this
    # state (other POV), reuse.
    if pair_cache is not None and "pf_pair" in pair_cache:
        pf_pair = pair_cache["pf_pair"]
        return {
            "planet_feats": pf, "planet_mask": pmask, "planet_ids": pid_arr,
            "fleet_feats": ff, "fleet_mask": fmask,
            "fleet_tgt_pid": f_tgt_pid, "fleet_tgt_idx": f_tgt_idx,
            "globals": gl,
            "pair_feats": pf_pair,
            "n_real": n_real,
        }
    pf_pair = np.zeros((N_MAX, N_MAX, F_PAIR), dtype=F32)
    # Positions now + at t+1 (h=1 for derivative-like drift)
    xy_now = np.zeros((N_MAX, 2), dtype=F32)
    xy_h1 = np.zeros((N_MAX, 2), dtype=F32)
    for p in planets:
        i = pid_to_idx[p["id"]]
        pos0 = planet_pos_at(state, p, 0)
        pos1 = planet_pos_at(state, p, 1)
        xy_now[i] = pos0 if pos0 is not None else (p["x"], p["y"])
        xy_h1[i] = pos1 if pos1 is not None else (p["x"], p["y"])
    # distance now
    dx = xy_now[:, None, 0] - xy_now[None, :, 0]
    dy = xy_now[:, None, 1] - xy_now[None, :, 1]
    dist_now = np.sqrt(dx * dx + dy * dy + 1e-12)
    pf_pair[..., 0] = dist_now
    # drift at h=1: d(a_now, b_at_h1) - d(a_now, b_now)
    dx1 = xy_now[:, None, 0] - xy_h1[None, :, 0]
    dy1 = xy_now[:, None, 1] - xy_h1[None, :, 1]
    dist_drift = np.sqrt(dx1 * dx1 + dy1 * dy1 + 1e-12)
    pf_pair[..., 1] = dist_drift - dist_now
    # apollo aim_eta in a single batched Rust call: builds EntityCache once,
    # loops over all (src, tgt, ships) triples in Rust. ~25× faster per state.
    if obs_for_apollo is not None:
        clean_obs = {
            "planets": obs_for_apollo["planets"],
            "angular_velocity": obs_for_apollo.get("angular_velocity", 0.0),
            "comets": obs_for_apollo.get("comets", []),
            "comet_planet_ids": obs_for_apollo.get("comet_planet_ids", []),
        }
        triples = []
        triple_idx = []
        for i, p_src in enumerate(planets):
            src_pid = int(p_src["id"])
            ships = max(int(p_src["ships"]), 1)
            for j, p_tgt in enumerate(planets):
                if i == j:
                    continue
                tgt_pid = int(p_tgt["id"])
                triples.append((src_pid, tgt_pid, ships))
                triple_idx.append((i, j))
        etas = apollo_native.aim_eta_batch(clean_obs, triples)
        for (i, j), eta in zip(triple_idx, etas):
            if eta is None:
                pf_pair[i, j, 2] = 0.0
                pf_pair[i, j, 3] = 1.0
            else:
                pf_pair[i, j, 2] = float(eta)
                pf_pair[i, j, 3] = 0.0
    if pair_cache is not None:
        pair_cache["pf_pair"] = pf_pair
    return {
        "planet_feats": pf, "planet_mask": pmask, "planet_ids": pid_arr,
        "fleet_feats": ff, "fleet_mask": fmask,
        "fleet_tgt_pid": f_tgt_pid, "fleet_tgt_idx": f_tgt_idx,
        "globals": gl,
        "pair_feats": pf_pair,
        "n_real": n_real,
    }


def process_game(game_id_int, game_json):
    """Same skeleton as build_dataset.process_game but writes v17 features.
    Labels: policy/value/noop reuse the v15 logic via action_labels."""
    rewards = game_json.get("rewards") or []
    if len(rewards) != 2:
        return []
    r0 = float(rewards[0]) if rewards[0] is not None else 0.0
    r1 = float(rewards[1]) if rewards[1] is not None else 0.0
    value_per_player = {0: float(np.sign(r0 - r1)), 1: float(np.sign(r1 - r0))}
    steps = game_json.get("steps") or []
    if len(steps) < 2:
        return []
    parsed = [None] * len(steps)
    for t, step in enumerate(steps):
        if step and step[0].get("observation"):
            parsed[t] = parse_state(step[0]["observation"])
    if not any(s and step[0].get("action") for s, step in zip(parsed, steps)):
        return []

    rows = []
    last_owner = {}; owner_ct = {}
    for t in range(len(steps) - 1):
        st = parsed[t]
        if st is None or parsed[t + 1] is None:
            continue
        update_owner_history(st, last_owner, owner_ct, st["step"])
        obs_t = steps[t][0].get("observation")
        # Pair feats are player-independent (apollo uses geometry only).
        # Compute once per state and reuse for both POVs.
        pair_cache = {"obs": obs_t}
        for player in (0, 1):
            try:
                feats = extract_state_v17(st, player, owner_ct, obs_for_apollo=obs_t, pair_cache=pair_cache)
            except Exception:
                continue
            if feats["n_real"] == 0 or feats["n_real"] > N_MAX:
                continue
            # action labels (per planet): noop + per-target launch.
            # IMPORTANT: kaggle stores the action that PRODUCED obs[t] at
            # steps[t][player]['action'] — i.e. it's the action taken DURING
            # turn t-1. To label the action taken DURING turn t (the one that
            # transitions obs[t] → obs[t+1]), we read steps[t+1].
            if t + 1 >= len(steps):
                continue  # no future step to read the action for turn t from
            action = (steps[t + 1][player].get("action") or [])
            pids = sorted(p["id"] for p in st["planets"])
            pid_to_idx = {pid: i for i, pid in enumerate(pids)}
            n = len(pids)
            noop_lbl = np.ones(N_MAX, dtype=F32)
            pair_src = np.full(K_MAX, -1, dtype=np.int32)
            pair_tgt = np.full(K_MAX, -1, dtype=np.int32)
            k = 0
            for act in action:
                src_pid = int(act[0])
                if src_pid not in pid_to_idx:
                    continue
                src_idx = pid_to_idx[src_pid]
                noop_lbl[src_idx] = 0.0
                # find target by predicted fleet.
                # Engine convention (bots/alphaduck/aim_native/src/engine.rs::process_moves):
                # the fleet spawns at src + (radius + LAUNCH_CLEARANCE)*u where
                # LAUNCH_CLEARANCE = 0.1, then moves +speed*u per turn.
                # Using radius+speed here put the fleet ~1 fleet-length too far
                # out, so predict_fleet_collision was reporting wrong targets.
                from build_dataset import fleet_speed as _fs   # noqa: F401 (kept for symmetry)
                import math as _math
                ang = float(act[1])
                ships = int(act[2])
                src_pl = st["planets"][src_idx]
                LAUNCH_CLEARANCE = 0.1
                launch_off = src_pl["radius"] + LAUNCH_CLEARANCE
                lx = float(src_pl["x"]) + launch_off * _math.cos(ang)
                ly = float(src_pl["y"]) + launch_off * _math.sin(ang)
                test_fleet = {
                    "id": -1, "owner": player, "ships": ships,
                    "x": lx, "y": ly,
                    "angle": ang,
                }
                pred = predict_fleet_collision(st, test_fleet)
                if pred is None:
                    continue
                tgt_pid = pred[0]
                if tgt_pid not in pid_to_idx:
                    continue
                tgt_idx = pid_to_idx[tgt_pid]
                if k < K_MAX:
                    pair_src[k] = src_idx
                    pair_tgt[k] = tgt_idx
                    k += 1
            value_lbl = float(value_per_player[player])
            rows.append({
                **feats,
                "noop_label": noop_lbl,
                "pair_src": pair_src,
                "pair_tgt": pair_tgt,
                "value_label": np.float32(value_lbl),
                "turn": np.int32(st["step"]),
                "game_id": np.int32(game_id_int),
                "player": np.int32(player),
            })
    return rows


def _process_game_worker(args):
    """Top-level worker for multiprocessing (must be picklable)."""
    game_id, game_json, max_per_game = args
    rows = process_game(game_id, game_json)
    return rows if max_per_game is None else rows[:max_per_game]


def _iter_games(zip_paths, max_games, max_per_game, skip_game_ids=None):
    if isinstance(zip_paths, (str, Path)):
        zip_paths = [zip_paths]
    if skip_game_ids is None:
        skip_game_ids = set()
    n_yielded = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted([n for n in zf.namelist() if n.endswith(".json")])
            for i, name in enumerate(names):
                if max_games is not None and n_yielded >= max_games:
                    return
                try:
                    game_id = int(Path(name).stem)
                except ValueError:
                    game_id = i
                if game_id in skip_game_ids:
                    continue
                try:
                    with zf.open(name) as f:
                        g = json.load(io.BytesIO(f.read()))
                except Exception:
                    continue
                yield (game_id, g, max_per_game)
                n_yielded += 1


# Keys whose values come from rows_buf (stacked across rows).
_STACK_KEYS = [
    ("planet_feats", "planet_feats"),
    ("planet_mask",  "planet_mask"),
    ("planet_ids",   "planet_ids"),
    ("fleet_feats",  "fleet_feats"),
    ("fleet_mask",   "fleet_mask"),
    ("fleet_tgt_pid","fleet_tgt_pid"),
    ("fleet_tgt_idx","fleet_tgt_idx"),
    ("globals",      "globals"),
    ("pair_feats",   "pair_feats"),
    ("noop_labels",  "noop_label"),
    ("pair_src",     "pair_src"),
    ("pair_tgt",     "pair_tgt"),
    ("value_labels", "value_label"),
    ("turns",        "turn"),
    ("game_ids",     "game_id"),
    ("players",      "player"),
]


def _save_chunk(rows_buf, chunk_path):
    out = {out_key: np.stack([r[row_key] for r in rows_buf])
           for out_key, row_key in _STACK_KEYS}
    np.savez_compressed(chunk_path, **out)


def _concat_chunks_and_save(chunk_paths, out_path):
    """Concat all chunks into one NPZ. WARNING: only safe for small datasets.
    For v17 at 2M+ rows, pair_feats alone would be ~88 GB uncompressed and
    cannot be materialized in RAM. Use the chunked-trainer path
    (train_v17.py --data-glob ...) instead of this concat. Kept for
    backwards-compatibility with small builds (e.g. dev smoke tests)."""
    accum = {out_key: [] for out_key, _ in _STACK_KEYS}
    n_rows = 0
    for cp in chunk_paths:
        d = np.load(cp, allow_pickle=True)
        for out_key, _ in _STACK_KEYS:
            accum[out_key].append(d[out_key])
        n_rows += d["planet_feats"].shape[0]
    out = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    out.update(dict(
        planet_feat_names=np.array(PLANET_FEAT_NAMES, dtype=object),
        fleet_feat_names=np.array(FLEET_FEAT_NAMES, dtype=object),
        global_feat_names=np.array(GLOBAL_FEAT_NAMES, dtype=object),
        pair_feat_names=np.array(PAIR_FEAT_NAMES, dtype=object),
    ))
    np.savez_compressed(out_path, **out)
    print(f"  concat: {n_rows} rows from {len(chunk_paths)} chunks -> {out_path}", flush=True)
    return n_rows


def build_npz_from_zip(zip_paths, out_path, max_games=None, max_per_game=None,
                      workers=1, chunk_size=500, no_concat=False):
    """Chunked, crash-safe build. Writes intermediate chunks to disk every
    `chunk_size` games so a server crash keeps the data processed so far.

    If no_concat=True, leave chunks on disk and skip the final NPZ concat
    (the concat step is the EC2-killer per [[ec2_oom_pattern]] — it does
    np.concatenate over all chunks and the pair_feats alone would peak at
    ~80 GB RAM on the full v17 dataset). The trainer's --data-glob handles
    chunked input natively."""
    rows_buf = []
    t0 = time.time()
    n_games = 0
    chunk_idx = 0
    chunk_paths: list[Path] = []
    chunk_dir = Path(out_path).parent
    chunk_prefix = Path(out_path).name + ".chunk_"

    # Resume support: pick up any existing chunks for this output path.
    # Also collect the game_ids those chunks cover so we skip them in iteration.
    skip_game_ids: set[int] = set()
    existing = sorted(chunk_dir.glob(f"{chunk_prefix}*.npz"))
    if existing:
        print(f"  resume: found {len(existing)} existing chunks", flush=True)
        chunk_paths.extend(existing)
        chunk_idx = len(existing)
        for cp in existing:
            d = np.load(cp, allow_pickle=True)
            skip_game_ids.update(int(g) for g in d["game_ids"])
        print(f"  resume: skipping {len(skip_game_ids)} already-processed games", flush=True)

    def maybe_flush(force=False):
        nonlocal rows_buf, chunk_idx
        if not rows_buf:
            return
        if not force and len(rows_buf) < chunk_size:
            return
        cp = chunk_dir / f"{chunk_prefix}{chunk_idx:04d}.npz"
        _save_chunk(rows_buf, cp)
        chunk_paths.append(cp)
        chunk_idx += 1
        rows_buf = []
        print(f"  saved chunk {chunk_idx} ({cp.name})", flush=True)

    games_since_flush = 0
    if workers <= 1:
        for args in _iter_games(zip_paths, max_games, max_per_game, skip_game_ids=skip_game_ids):
            rows_buf.extend(_process_game_worker(args))
            n_games += 1
            games_since_flush += 1
            if n_games % 50 == 0:
                print(f"  game {n_games}  rows={len(rows_buf)}  ({time.time()-t0:.0f}s, {n_games/max(1,time.time()-t0):.1f} g/s)", flush=True)
            if games_since_flush >= chunk_size:
                maybe_flush(force=True)
                games_since_flush = 0
    else:
        import multiprocessing as mp
        # maxtasksperchild=50: recycle each worker after 50 games to cap any
        # per-game memory accumulation (apollo entity cache, pyo3 refs, etc.).
        # Trivial overhead since worker startup is ~1s and we get hours of
        # safety. Confirmed needed after the v17 OOM-induced EC2 hang.
        with mp.Pool(workers, maxtasksperchild=50) as pool:
            for rows in pool.imap_unordered(_process_game_worker, _iter_games(zip_paths, max_games, max_per_game, skip_game_ids=skip_game_ids), chunksize=4):
                rows_buf.extend(rows)
                n_games += 1
                games_since_flush += 1
                if n_games % 50 == 0:
                    print(f"  game {n_games}  rows={len(rows_buf)}  ({time.time()-t0:.0f}s, {n_games/max(1,time.time()-t0):.1f} g/s)", flush=True)
                if games_since_flush >= chunk_size:
                    maybe_flush(force=True)
                    games_since_flush = 0

    # Final flush of leftover rows
    maybe_flush(force=True)
    if not chunk_paths:
        raise RuntimeError("no rows produced")

    if no_concat:
        print(f"skipping concat: {len(chunk_paths)} chunks left on disk for chunked training", flush=True)
        return
    print(f"concatenating {len(chunk_paths)} chunks ...", flush=True)
    n_rows = _concat_chunks_and_save(chunk_paths, out_path)
    # Cleanup chunks now that the final NPZ is on disk
    for cp in chunk_paths:
        try: cp.unlink()
        except Exception: pass
    print(f"saved {out_path}  ({Path(out_path).stat().st_size // 1024} KB, {n_rows} rows)", flush=True)


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, nargs="+",
                    help="one or more zip paths; games are concatenated in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--max-per-game", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--chunk-size", type=int, default=500,
                    help="games per checkpoint chunk. Lower = safer on crash, "
                         "slightly more disk I/O. Default 500.")
    ap.add_argument("--no-concat", action="store_true",
                    help="skip the final np.concatenate of all chunks (the EC2 OOM-killer "
                         "per [[ec2_oom_pattern]]). Recommended for any multi-day build.")
    args = ap.parse_args()
    print(f"workers={args.workers}  chunk-size={args.chunk_size}  no_concat={args.no_concat}", flush=True)
    build_npz_from_zip(args.zip, args.out, args.max_games, args.max_per_game,
                      args.workers, args.chunk_size, no_concat=args.no_concat)
