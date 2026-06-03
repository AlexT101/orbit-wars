"""Compute spatial_features.compute() for every row of an existing combined/
summary NPZ, aligned by (game_id, step, slot) via the NPZ's meta + game_files.

Output: <out>.npz with key `spatial` shape [n_rows, SPATIAL_DIM], row-aligned to
the input NPZ's meta.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from spatial_features import compute, SPATIAL_DIM


def source_tag(path):
    p = Path(path)
    return p.stem if p.is_file() and p.suffix == ".zip" else p.name


def read_entry(source_path, entry):
    p = Path(source_path)
    if p.is_dir():
        return (p / entry).read_bytes()
    with zipfile.ZipFile(p) as zf:
        return zf.read(entry)


def process_chunk(args):
    src_path, entries, worker_id, n_players = args
    out = []  # (gid, step, slot, feats)
    for (gid, entry) in entries:
        try:
            data = json.loads(read_entry(src_path, entry))
        except Exception:
            continue
        rewards = data.get("rewards") or []
        steps = data.get("steps") or []
        if len(rewards) != n_players or not steps:
            continue
        for step in steps:
            if not isinstance(step, list) or len(step) < n_players:
                continue
            for slot in range(n_players):
                entry_obj = step[slot]
                if not isinstance(entry_obj, dict):
                    continue
                obs = entry_obj.get("observation")
                if not obs or not obs.get("planets"):
                    continue
                tick = int(obs.get("step", 0))
                feats = compute(obs, slot)
                out.append((gid, tick, slot, feats))
        if worker_id == 0 and len(out) % 40000 < n_players * 2:
            print(f"  [w0] ~{len(out)} obs done", flush=True)
    return out


def build(npz_path, src_paths, out_path, workers):
    d = np.load(npz_path, allow_pickle=False)
    meta = d["meta"]
    game_files = d["game_files"]
    n_games = game_files.shape[0]
    n_players = 2
    if "game_player_count" in d.files:
        counts = np.unique(d["game_player_count"].astype(np.int32))
        if len(counts) == 1:
            n_players = int(counts[0])
    print(f"NPZ rows={meta.shape[0]} games={n_games} players={n_players}")

    tag_to_src = {source_tag(s): str(s) for s in src_paths}
    by_src = {}
    miss = 0
    for gid in range(n_games):
        gf = str(game_files[gid])
        if ":" not in gf:
            miss += 1
            continue
        tag, entry = gf.split(":", 1)
        if tag not in tag_to_src:
            miss += 1
            continue
        by_src.setdefault(tag_to_src[tag], []).append((gid, entry))
    if miss:
        print(f"  WARN {miss} games missing source match")

    t0 = time.time()
    row_feats = {}
    for src_path, entries in by_src.items():
        chunks = [entries[i::workers] for i in range(workers)]
        with mp.Pool(workers) as pool:
            results = pool.map(process_chunk, [(src_path, c, i, n_players) for i, c in enumerate(chunks)])
        for rows in results:
            for (gid, step, slot, feats) in rows:
                row_feats[(gid, step, slot)] = feats
        print(f"  done {Path(src_path).name} ({time.time()-t0:.0f}s)", flush=True)

    spatial = np.zeros((meta.shape[0], SPATIAL_DIM), dtype=np.float32)
    hit = 0
    for i in range(meta.shape[0]):
        v = row_feats.get((int(meta[i, 0]), int(meta[i, 1]), int(meta[i, 2])))
        if v is not None:
            spatial[i] = v
            hit += 1
    print(f"  aligned {hit}/{meta.shape[0]} rows ({100*hit/meta.shape[0]:.2f}%)")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spatial=spatial)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--src", nargs="+", required=True, help="replay dir(s) or zip(s)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    build(args.npz, args.src, args.out, args.workers)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
