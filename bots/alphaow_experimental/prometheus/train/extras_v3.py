"""Focused 5-d extras (user-requested set):
  0  tick                        current step (0..n_steps-1)
  1  nearest_enemy_planet_now    min Euclidean dist between any of MY planets
                                 and any ENEMY planet at the current state
  2  nearest_enemy_planet_ext    same metric but using EXTRAPOLATED ownership
                                 (apply all in-flight fleets, resolve combat,
                                 then recompute min-dist on the updated owners)
  3  n_total_static              total stationary planets in the game (const)
  4  n_total_orbit               total rotating (orbiting non-comet) planets

All leakage-free (no n_steps / future-info). Aligned row-by-row to the meta
of a combined NPZ via (game_id, step, slot) lookup.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import zipfile
from pathlib import Path

import numpy as np

# Re-use the proven extractor's parse_state + extrapolate_fleets so this
# extra inherits the bit-exact extrapolation logic.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_extract import parse_state, extrapolate_fleets

EXTRA_DIM = 5


def _min_pair_dist(my_xy, en_xy):
    if not my_xy or not en_xy:
        return 0.0
    best = math.inf
    for mx, my in my_xy:
        for ex, ey in en_xy:
            dx = mx - ex
            dy = my - ey
            d = math.sqrt(dx * dx + dy * dy)
            if d < best:
                best = d
    return best if best != math.inf else 0.0


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

    # Game-level constants: count static / orbit planets from the FIRST
    # valid observation's planet list (planet types don't change mid-game).
    n_static = 0
    n_orbit = 0
    for st in steps:
        if not isinstance(st, list) or len(st) < 2:
            continue
        obs0 = (st[0] or {}).get("observation") if isinstance(st[0], dict) else None
        if obs0 and obs0.get("planets"):
            state = parse_state(obs0)
            for p in state["planets"]:
                if p["is_comet"]:
                    pass
                elif p["is_orbiting"]:
                    n_orbit += 1
                else:
                    n_static += 1
            break
    n_static_f = float(n_static)
    n_orbit_f = float(n_orbit)

    out = []
    for tick_idx, step in enumerate(steps):
        if not isinstance(step, list) or len(step) < 2:
            continue
        for slot in range(2):
            entry_obj = step[slot]
            if not isinstance(entry_obj, dict):
                continue
            obs = entry_obj.get("observation")
            if not obs or not obs.get("planets"):
                continue
            state = parse_state(obs)
            me = state["player"]

            # Distance NOW: min pair-distance between my-owned and enemy-owned planets
            my_xy_now = [(p["x"], p["y"]) for p in state["planets"] if p["owner"] == me]
            en_xy_now = [(p["x"], p["y"]) for p in state["planets"]
                         if p["owner"] != me and p["owner"] != -1]
            near_now = _min_pair_dist(my_xy_now, en_xy_now)

            # Distance AFTER extrapolation: same planet positions, but use
            # the predicted post-combat owners.
            ext_owners = extrapolate_fleets(state)  # {pid: (owner, ships)}
            my_xy_ext = []
            en_xy_ext = []
            for p in state["planets"]:
                o = ext_owners.get(p["id"], (p["owner"], p["ships"]))[0]
                if o == me:
                    my_xy_ext.append((p["x"], p["y"]))
                elif o != -1:
                    en_xy_ext.append((p["x"], p["y"]))
            near_ext = _min_pair_dist(my_xy_ext, en_xy_ext)

            out.append((gid, tick_idx, slot, (
                float(tick_idx),
                float(near_now),
                float(near_ext),
                n_static_f,
                n_orbit_f,
            )))
    return gid, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--zip", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    meta = d["meta"]
    game_files = d["game_files"]
    n_games = game_files.shape[0]
    print(f"NPZ rows={meta.shape[0]} games={n_games}")
    tag_to_zip = {Path(z).stem: z for z in args.zip}
    print(f"zip tags: {list(tag_to_zip.keys())}")

    tasks = []
    miss = 0
    for gid in range(n_games):
        gf = str(game_files[gid])
        if ":" not in gf:
            miss += 1
            continue
        tag, entry = gf.split(":", 1)
        z = tag_to_zip.get(tag)
        if z is None:
            miss += 1
            continue
        tasks.append((z, entry, gid))
    if miss:
        print(f"  WARN {miss} games missing zip match")
    print(f"  processing {len(tasks)} games with {args.workers} workers")

    t0 = time.time()
    row_extras = {}
    with mp.Pool(args.workers) as pool:
        for done, (gid, rows) in enumerate(pool.imap_unordered(process_game, tasks, chunksize=32)):
            for (g, st, sl, vec) in rows:
                row_extras[(g, st, sl)] = vec
            if (done + 1) % 1000 == 0:
                print(f"    {done+1}/{len(tasks)} games processed ({time.time()-t0:.1f}s)", flush=True)
    print(f"  extras dict size: {len(row_extras)}   elapsed: {time.time()-t0:.1f}s")

    extras = np.zeros((meta.shape[0], EXTRA_DIM), dtype=np.float32)
    hit = 0
    for i in range(meta.shape[0]):
        gid = int(meta[i, 0])
        step = int(meta[i, 1])
        slot = int(meta[i, 2])
        v = row_extras.get((gid, step, slot))
        if v is not None:
            extras[i] = v
            hit += 1
    print(f"  aligned {hit}/{meta.shape[0]} rows ({100 * hit / meta.shape[0]:.2f}%)")
    np.savez_compressed(args.out, extras=extras)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
