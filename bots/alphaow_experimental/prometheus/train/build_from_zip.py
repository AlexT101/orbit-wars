"""Build a summary_v2 NPZ by streaming replay JSONs directly out of a zip
archive (no disk extraction), feeding each observation to the long-lived
`extract_v2` Rust binary over stdin.

Designed for a ~20GB / 5000-game archive on a disk-constrained machine:
games are read one at a time via zipfile.read() and never unpacked to disk.

Single parse pass:
  * filter out 4-player games (len(rewards) != 2),
  * extract all 2p observations -> 46-d summary_v2 features,
  * collect each game's agent names + rewards so the "strong player"
    gate (both players above the median win rate) can be computed in
    memory afterward and stored as a per-sample is_strong flag.

The strong gate is NOT applied destructively here -- every 2p sample is
written, tagged with is_strong, so the trainer can compare all-2p vs the
strong subset without re-extracting.
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
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALPHAOW_DIR = HERE.parent
BIN = ALPHAOW_DIR / "target" / "release" / "extract_v2"

SUMMARY_V2_DIM = 46
RECORD_BYTES = 8 + 4 + 4 * SUMMARY_V2_DIM  # 196

MIN_GAMES = 3   # min games a player needs before their win rate counts
SEED = 0


def source_tag(path: str | Path) -> str:
    p = Path(path)
    return p.stem if p.is_file() and p.suffix == ".zip" else p.name


def list_json_entries(source_path: str | Path, limit=None) -> list[str]:
    p = Path(source_path)
    if p.is_dir():
        names = [x.relative_to(p).as_posix() for x in p.rglob("*.json") if x.is_file()]
    else:
        with zipfile.ZipFile(p) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/")]
    names.sort()
    return names[:limit] if limit else names


def read_entry(source_path: str | Path, entry: str) -> bytes:
    p = Path(source_path)
    if p.is_dir():
        return (p / entry).read_bytes()
    with zipfile.ZipFile(p) as zf:
        return zf.read(entry)


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


def _agent_names(d: dict) -> list:
    agents = (d.get("info") or {}).get("Agents") or []
    return [str(a.get("Name", f"p{i}")) for i, a in enumerate(agents)]


def process_chunk(args):
    """One worker: open its own zip handle + one extract_v2 subprocess,
    stream all assigned games through it."""
    zip_path, names, worker_id, zip_tag = args
    proc = subprocess.Popen(
        [str(BIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    records = []  # (step, player, v2) in send order

    def reader():
        out = proc.stdout
        while True:
            raw = out.read(RECORD_BYTES)
            if not raw or len(raw) < RECORD_BYTES:
                break
            step = int.from_bytes(raw[:8], "little", signed=True)
            player = int.from_bytes(raw[8:12], "little", signed=True)
            v2 = np.frombuffer(raw[12:], dtype=np.float32).copy()
            records.append((step, player, v2))

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    sent_meta = []        # parallel to records: (local_gid, slot, reward)
    game_info = []        # local_gid -> (name0, name1, r0, r1)
    game_files = []       # local_gid -> "tag:entry" source filename
    gid = -1
    skip_4p = 0
    skip_other = 0

    for entry in names:
        try:
            data = json.loads(read_entry(zip_path, entry))
        except Exception:
            skip_other += 1
            continue
        rewards = data.get("rewards") or []
        if len(rewards) == 4:
            skip_4p += 1
            continue
        steps = data.get("steps") or []
        agents = _agent_names(data)
        if (
            len(rewards) != 2
            or len(agents) != 2
            or any(r is None for r in rewards[:2])
            or not steps
        ):
            skip_other += 1
            continue

        gid += 1
        game_info.append((agents[0], agents[1], float(rewards[0]), float(rewards[1])))
        game_files.append(f"{zip_tag}:{entry}")

        wrote_any = False
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
                    return None
                sent_meta.append((gid, slot, float(rewards[slot])))
                wrote_any = True
        if wrote_any:
            try:
                proc.stdin.flush()
            except BrokenPipeError:
                return None

        if (gid + 1) % 50 == 0:
            print(f"  [w{worker_id}] {gid + 1} games sent", flush=True)

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
    n = min(len(records), len(sent_meta))
    if n < len(sent_meta):
        print(f"  [w{worker_id}] WARN got {len(records)} records for {len(sent_meta)} sent", flush=True)
    if n == 0:
        return ([], [], [], game_info, game_files, gid + 1, skip_4p, skip_other)

    feats = np.empty((n, SUMMARY_V2_DIM), dtype=np.float32)
    labels = np.empty(n, dtype=np.float32)
    meta = np.empty((n, 4), dtype=np.int32)
    for i in range(n):
        step, player, v2 = records[i]
        local_gid, slot, reward = sent_meta[i]
        feats[i] = v2
        labels[i] = reward
        meta[i] = (local_gid, step, player, 1 - player)

    return (feats, labels, meta, game_info, game_files, gid + 1, skip_4p, skip_other)


def build(zip_paths, out_npz: str, n_workers: int, limit=None):
    if isinstance(zip_paths, str):
        zip_paths = [zip_paths]
    feats_all, labels_all, meta_all = [], [], []
    game_info_all = []  # global_gid -> (name0, name1, r0, r1)
    game_files_all = [] # global_gid -> "tag:entry"
    total_games = total_4p = total_other = 0
    elapsed = 0.0

    for zi, zip_path in enumerate(zip_paths):
        names = list_json_entries(zip_path, limit)
        tag = source_tag(zip_path)
        print(f"\n>>> [{zi+1}/{len(zip_paths)}] {zip_path}: {len(names)} entries  tag={tag}")
        chunks = [names[i::n_workers] for i in range(n_workers)]
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            results = pool.map(process_chunk, [(zip_path, c, i, tag) for i, c in enumerate(chunks)])
        elapsed += time.time() - t0
        for res in results:
            if res is None:
                continue
            f, lbl, m, ginfo, gfiles, n_games, s4p, soth = res
            if len(f):
                m = m.copy()
                m[:, 0] += total_games  # offset local gids -> global
                feats_all.append(f)
                labels_all.append(lbl)
                meta_all.append(m)
            game_info_all.extend(ginfo)
            game_files_all.extend(gfiles)
            total_games += n_games
            total_4p += s4p
            total_other += soth

    if not feats_all:
        raise SystemExit("no samples extracted")

    feats = np.concatenate(feats_all)
    labels = np.concatenate(labels_all)
    meta = np.concatenate(meta_all)

    # --- strong-player gate (computed in memory, stored as a flag) ---
    pg, pw = defaultdict(int), defaultdict(int)
    for n0, n1, r0, r1 in game_info_all:
        pg[n0] += 1
        pg[n1] += 1
        if r0 > r1:
            pw[n0] += 1
        elif r1 > r0:
            pw[n1] += 1
    rates = {pl: pw[pl] / pg[pl] for pl in pg if pg[pl] >= MIN_GAMES}
    sr = sorted(rates.values())
    median = sr[len(sr) // 2] if sr else 0.0
    above = {pl for pl, r in rates.items() if r > median}
    strong_gids = {
        gid for gid, (n0, n1, _r0, _r1) in enumerate(game_info_all)
        if n0 in above and n1 in above
    }
    is_strong = np.fromiter(
        (1 if g in strong_gids else 0 for g in meta[:, 0]),
        dtype=np.uint8, count=meta.shape[0],
    )

    slots = sorted(np.unique(meta[:, 2]).tolist())
    print(f"\n=== build summary ===")
    print(f"2-player games extracted : {total_games}")
    print(f"4-player games filtered  : {total_4p}")
    print(f"other/invalid skipped    : {total_other}")
    print(f"observations             : {feats.shape[0]}")
    print(f"player slots present     : {slots}   (must be [0, 1])")
    print(f"unique players (>= {MIN_GAMES} games): {len(rates)}   median win rate: {median:.3f}")
    print(f"strong games             : {len(strong_gids)} / {total_games}")
    print(f"strong observations      : {int(is_strong.sum())} / {is_strong.shape[0]}")
    print(f"NaN/Inf in features      : {int(np.sum(~np.isfinite(feats)))}")
    print(f"label values             : {sorted(np.unique(labels).tolist())[:6]}...")
    print(f"extraction time          : {elapsed:.1f}s")

    # game_names: shape (n_games, 2) of unicode strings, indexed by global_gid
    name_arr = np.array([(g[0], g[1]) for g in game_info_all], dtype="<U64")
    file_arr = np.array(game_files_all, dtype="<U200")

    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        summary_v2=feats, labels=labels, meta=meta, is_strong=is_strong,
        game_names=name_arr, game_files=file_arr,
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, nargs="+", help="one or more zip files (built in order)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--limit", type=int, default=None, help="cap entries per zip (debug)")
    args = p.parse_args()
    build(args.zip, args.out, args.workers, args.limit)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
