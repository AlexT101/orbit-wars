"""Extract the 5-d extras_v3 (tick, near_now, near_ext, n_static, n_orbit)
via the Rust `extract_v3` binary and align to an existing combined NPZ.

Saves `<out>.npz` with key `extras` of shape (n_rows, 5), float32, aligned
row-by-row to the input NPZ's `meta` via (game_id, step, slot) lookup.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ALPHAOW_DIR = REPO / "bots" / "mine" / "alphaow"
BIN = ALPHAOW_DIR / "target" / "release" / "extract_v3"

SUMMARY_V2_DIM = 46
EXTRA_DIM = 5
RECORD_BYTES = 8 + 4 + 4 * SUMMARY_V2_DIM + 4 * EXTRA_DIM   # = 216


def normalize_obs(o: dict) -> dict:
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


def process_chunk(args):
    """One worker: stream all assigned games' obs through one extract_v3
    subprocess, collect 216-byte records, return (gid, step, slot, extras5)."""
    zip_path, entries, worker_id = args
    zf = zipfile.ZipFile(zip_path)
    proc = subprocess.Popen(
        [str(BIN)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    records = []   # parallel to send_meta

    def reader():
        out = proc.stdout
        while True:
            raw = out.read(RECORD_BYTES)
            if not raw or len(raw) < RECORD_BYTES:
                break
            step = int.from_bytes(raw[:8], "little", signed=True)
            player = int.from_bytes(raw[8:12], "little", signed=True)
            extras = np.frombuffer(raw[12 + 4 * SUMMARY_V2_DIM:], dtype=np.float32).copy()
            records.append((step, player, extras))

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    send_meta = []   # (gid, slot) in send order
    for (gid, entry) in entries:
        try:
            data = json.loads(zf.read(entry))
        except Exception:
            continue
        rewards = data.get("rewards") or []
        steps = data.get("steps") or []
        if len(rewards) != 2 or not steps:
            continue
        for step in steps:
            if not isinstance(step, list) or len(step) < 2:
                continue
            for slot in range(2):
                entry_obj = step[slot]
                if not isinstance(entry_obj, dict):
                    continue
                obs = entry_obj.get("observation")
                if not obs or not obs.get("planets"):
                    continue
                norm = normalize_obs(obs)
                line = json.dumps(norm, separators=(",", ":")) + "\n"
                try:
                    proc.stdin.write(line.encode())
                except BrokenPipeError:
                    break
                send_meta.append((gid, slot))
        try:
            proc.stdin.flush()
        except BrokenPipeError:
            break
        if (worker_id == 0) and len(send_meta) % 20000 == 0 and len(send_meta) > 0:
            print(f"  [w{worker_id}] {len(send_meta)} obs sent", flush=True)

    try:
        proc.stdin.close()
    except Exception:
        pass
    rt.join(timeout=300)
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    zf.close()

    n = min(len(records), len(send_meta))
    out = []
    for i in range(n):
        step, _player, extras = records[i]
        gid, slot = send_meta[i]
        out.append((gid, step, slot, extras))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="combined NPZ with meta + game_files")
    p.add_argument("--zip", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    meta = d["meta"]
    game_files = d["game_files"]
    n_games = game_files.shape[0]
    print(f"NPZ rows={meta.shape[0]} games={n_games}")

    tag_to_zip = {Path(z).stem: z for z in args.zip}
    # Build per-zip ordered list of (gid, entry) so worker chunks balance.
    by_zip: dict[str, list[tuple[int, str]]] = {}
    miss = 0
    for gid in range(n_games):
        gf = str(game_files[gid])
        if ":" not in gf:
            miss += 1
            continue
        tag, entry = gf.split(":", 1)
        if tag not in tag_to_zip:
            miss += 1
            continue
        by_zip.setdefault(tag_to_zip[tag], []).append((gid, entry))
    if miss:
        print(f"  WARN {miss} games missing zip match")
    total_games = sum(len(v) for v in by_zip.values())
    print(f"  processing {total_games} games across {len(by_zip)} zips with {args.workers} workers")

    t0 = time.time()
    row_extras: dict[tuple[int, int, int], np.ndarray] = {}
    for zip_path, entries in by_zip.items():
        chunks = [entries[i::args.workers] for i in range(args.workers)]
        with mp.Pool(args.workers) as pool:
            results = pool.map(process_chunk, [(zip_path, c, i) for i, c in enumerate(chunks)])
        for rows in results:
            for (gid, step, slot, extras) in rows:
                row_extras[(gid, step, slot)] = extras
        print(f"  done zip {Path(zip_path).stem} ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"  extras dict size: {len(row_extras)}   elapsed: {time.time()-t0:.0f}s")

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
