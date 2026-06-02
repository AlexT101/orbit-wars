"""Compute cheap *extra* per-row features that the Rust extract_v2
binary does NOT emit, and align them with an existing combined NPZ
(produced by build_from_zip.py).

Features added per row (16 cols):
  0  tick                    current step (0..n_steps-1)
  1  tick_frac               tick / n_steps   (game progress 0..1)
  2  n_steps                 total game length
  3  time_remaining          n_steps - tick
  4  sends_last_5_me         my fleet launches in last 5 ticks (incl this)
  5  sends_last_5_opp        opp fleet launches in last 5 ticks
  6  sends_total_me          cumulative my launches
  7  sends_total_opp         cumulative opp launches
  8  ships_delta_me_5        my total ships minus ships 5 ticks ago
  9  ships_delta_opp_5       opp delta
  10 fleets_in_flight_me     count of my in-flight fleets
  11 fleets_in_flight_opp    same for opp
  12 nearest_enemy_planet    Euclidean dist to nearest enemy planet
  13 nearest_my_planet       min dist between any of my planets and any enemy planet (frontline)
  14 angular_velocity        from obs (game speed of planet rotation)
  15 ship_dominance          (my_total - opp_total) / (my_total + opp_total + 1)

All cheap: ~3-5 ms per game in pure Python.

Alignment strategy: NPZ rows are indexed by (gid, step, slot). We build a
dict keyed by (gid, step, slot) -> 16-vec, then materialise an array
matching the existing meta order.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import zipfile
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

EXTRA_DIM = 16


def _ship_totals(obs):
    ships_p = 0; ships_p_opp = 0
    me = obs.get("player", 0); opp = 1 - me
    for pl in obs.get("planets") or []:
        if isinstance(pl, dict):
            o = pl.get("owner"); s = pl.get("ships") or 0
        else:
            # list-form: [x,y,r,prod,orbit_r,owner,ships,...]
            o = pl[5] if len(pl) > 5 else -1
            s = pl[6] if len(pl) > 6 else 0
        if o == me: ships_p += s
        elif o == opp: ships_p_opp += s
    ships_f = 0; ships_f_opp = 0; cnt_f = 0; cnt_f_opp = 0
    for fl in obs.get("fleets") or []:
        if isinstance(fl, dict):
            o = fl.get("owner"); s = fl.get("ships") or 0
        else:
            o = fl[2] if len(fl) > 2 else -1
            s = fl[3] if len(fl) > 3 else 0
        if o == me: ships_f += s; cnt_f += 1
        elif o == opp: ships_f_opp += s; cnt_f_opp += 1
    return ships_p + ships_f, ships_p_opp + ships_f_opp, cnt_f, cnt_f_opp


def _planet_positions_owned(obs):
    me = obs.get("player", 0); opp = 1 - me
    mine = []; theirs = []
    for pl in obs.get("planets") or []:
        if isinstance(pl, dict):
            x = float(pl.get("x", 0.0)); y = float(pl.get("y", 0.0))
            o = pl.get("owner", -1)
        else:
            x = float(pl[0] if len(pl) > 0 else 0)
            y = float(pl[1] if len(pl) > 1 else 0)
            o = pl[5] if len(pl) > 5 else -1
        if o == me: mine.append((x, y))
        elif o == opp: theirs.append((x, y))
    return mine, theirs


def process_game(args):
    zip_path, entry, gid = args
    zf = zipfile.ZipFile(zip_path)
    try:
        data = json.loads(zf.read(entry))
    except Exception:
        zf.close()
        return gid, []
    zf.close()
    rewards = data.get("rewards") or []
    steps = data.get("steps") or []
    if len(rewards) != 2 or not steps:
        return gid, []
    n_steps = len(steps)

    out = []  # (gid, step, slot, extras_vec)
    # roll history for sends and ships per slot
    send_hist_me = deque(maxlen=5); send_hist_op = deque(maxlen=5)
    ship_hist_me = deque(maxlen=6); ship_hist_op = deque(maxlen=6)
    cum_me = 0; cum_op = 0
    for tick_idx, step in enumerate(steps):
        if not isinstance(step, list) or len(step) < 2:
            continue
        # both slots see the same world; pull each side's obs to also derive their POV
        for slot in range(2):
            entry_obj = step[slot]
            if not isinstance(entry_obj, dict): continue
            obs = entry_obj.get("observation")
            if not obs or not obs.get("planets"): continue
            action = entry_obj.get("action") or []
            sends = len(action) if isinstance(action, list) else 0
            if slot == 0:
                send_hist_me.append(sends); cum_me += sends
            else:
                send_hist_op.append(sends); cum_op += sends
            ships_me, ships_op, fl_me, fl_op = _ship_totals(obs)
            if slot == 0:
                ship_hist_me.append(ships_me)
            else:
                ship_hist_op.append(ships_op)
            # nearest-enemy / frontline distances
            mine_xy, theirs_xy = _planet_positions_owned(obs)
            if mine_xy and theirs_xy:
                near_e = min(math.hypot(mx-tx, my-ty) for mx, my in mine_xy for tx, ty in theirs_xy)
                frontline = near_e  # same metric (min pair dist)
            else:
                near_e = 0.0; frontline = 0.0
            sld_me = (ship_hist_me[-1] - ship_hist_me[0]) if len(ship_hist_me) >= 2 else 0
            sld_op = (ship_hist_op[-1] - ship_hist_op[0]) if len(ship_hist_op) >= 2 else 0
            tot = ships_me + ships_op
            dom = (ships_me - ships_op) / (tot + 1.0)
            av = float(obs.get("angular_velocity", 0.0))
            extras = (
                float(tick_idx),
                tick_idx / max(1, n_steps),
                float(n_steps),
                float(n_steps - tick_idx),
                float(sum(send_hist_me)) if slot == 0 else float(sum(send_hist_op)),
                float(sum(send_hist_op)) if slot == 0 else float(sum(send_hist_me)),
                float(cum_me) if slot == 0 else float(cum_op),
                float(cum_op) if slot == 0 else float(cum_me),
                float(sld_me) if slot == 0 else float(sld_op),
                float(sld_op) if slot == 0 else float(sld_me),
                float(fl_me),
                float(fl_op),
                near_e,
                frontline,
                av,
                dom,
            )
            out.append((gid, tick_idx, slot, extras))
    return gid, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="combined NPZ produced by build_from_zip.py")
    p.add_argument("--zip-root", default="/", help="prefix where zip paths in game_files resolve")
    p.add_argument("--zip", nargs="+", required=True, help="all source zips by full path (will match by stem)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    meta = d["meta"]
    game_files = d["game_files"]   # global_gid -> "tag:entry"
    n_games = game_files.shape[0]
    print(f"NPZ rows={meta.shape[0]} games={n_games}")
    print(f"zips supplied: {len(args.zip)}")

    # build tag -> zip path
    tag_to_zip = {Path(z).stem: z for z in args.zip}

    # per-gid (zip_path, entry)
    tasks = []
    miss = 0
    for gid in range(n_games):
        gf = str(game_files[gid])
        if ":" not in gf:
            miss += 1; continue
        tag, entry = gf.split(":", 1)
        if tag not in tag_to_zip:
            miss += 1; continue
        tasks.append((tag_to_zip[tag], entry, gid))
    if miss:
        print(f"  WARN {miss} games missing zip match")
    print(f"  processing {len(tasks)} games with {args.workers} workers")

    t0 = time.time()
    row_extras = {}  # (gid, step, slot) -> 16-tuple
    with mp.Pool(args.workers) as pool:
        for done, (gid, rows) in enumerate(pool.imap_unordered(process_game, tasks, chunksize=32)):
            for (g, st, sl, vec) in rows:
                row_extras[(g, st, sl)] = vec
            if (done + 1) % 1000 == 0:
                print(f"    {done+1}/{len(tasks)} games processed ({time.time()-t0:.1f}s)", flush=True)
    print(f"  extras dict size: {len(row_extras)}   elapsed: {time.time()-t0:.1f}s")

    # materialise aligned with meta
    extras = np.zeros((meta.shape[0], EXTRA_DIM), dtype=np.float32)
    hit = 0
    for i in range(meta.shape[0]):
        gid = int(meta[i, 0]); step = int(meta[i, 1]); slot = int(meta[i, 2])
        v = row_extras.get((gid, step, slot))
        if v is not None:
            extras[i] = v
            hit += 1
    print(f"  aligned {hit}/{meta.shape[0]} rows ({100*hit/meta.shape[0]:.2f}%)")
    np.savez_compressed(args.out, extras=extras)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
