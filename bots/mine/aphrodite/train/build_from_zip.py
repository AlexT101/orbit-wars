"""Build a summary_v2 NPZ by streaming replay JSONs directly out of a zip
archive (no disk extraction), feeding each observation to the long-lived
`extract_v2` Rust binary over stdin.

Designed for a ~20GB / 5000-game archive on a disk-constrained machine:
games are read one at a time via zipfile.read() and never unpacked to disk.

Single parse pass:
  * keep games matching --players (2 by default, or 4 for FFA),
  * extract all player observations -> 41-d summary_v2 features,
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
REPO = HERE.parents[3]
APHRODITE_DIR = REPO / "bots" / "mine" / "aphrodite"
BIN = APHRODITE_DIR / "target" / "release" / "extract_v2"

SUMMARY_V2_DIM = 65
RECORD_BYTES = 8 + 4 + 4 * SUMMARY_V2_DIM  # 272

MIN_GAMES = 3   # min games a player needs before their win rate counts
SEED = 0


def label_for_rewards(rewards, slot: int) -> float:
    if len(rewards) >= 4:
        vals = [float(r) for r in rewards]
        best = max(vals)
        winners = [i for i, r in enumerate(vals) if r == best]
        if len(winners) != 1:
            return 0.0
        return 1.0 if slot == winners[0] else -1.0
    return float(rewards[slot])


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
    zip_path, names, worker_id, zip_tag, n_players, keep_set = args
    zf = zipfile.ZipFile(zip_path)
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
    game_names = []       # local_gid -> tuple(names)
    game_rewards = []     # local_gid -> tuple(rewards)
    game_files = []       # local_gid -> "tag:entry" source filename
    gid = -1
    skip_format = 0
    skip_other = 0

    for entry in names:
        try:
            data = json.loads(zf.read(entry))
        except Exception:
            skip_other += 1
            continue
        rewards = data.get("rewards") or []
        if len(rewards) != n_players:
            skip_format += 1
            continue
        steps = data.get("steps") or []
        agents = _agent_names(data)
        if (
            len(agents) != n_players
            or any(r is None for r in rewards[:n_players])
            or not steps
        ):
            skip_other += 1
            continue

        gid += 1
        game_names.append(tuple(agents[:n_players]))
        game_rewards.append(tuple(float(r) for r in rewards[:n_players]))
        game_files.append(f"{zip_tag}:{entry}")

        wrote_any = False
        for step in steps:
            if not isinstance(step, list) or len(step) < n_players:
                continue
            for slot in range(n_players):
                if keep_set is not None and str(agents[slot]) not in keep_set:
                    continue  # Elo gate: skip rows for players not in the keep set
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
                sent_meta.append((gid, slot, label_for_rewards(rewards, slot)))
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
    zf.close()

    n = min(len(records), len(sent_meta))
    if n < len(sent_meta):
        print(f"  [w{worker_id}] WARN got {len(records)} records for {len(sent_meta)} sent", flush=True)
    if n == 0:
        return ([], [], [], game_names, game_rewards, game_files, gid + 1, skip_format, skip_other)

    feats = np.empty((n, SUMMARY_V2_DIM), dtype=np.float32)
    labels = np.empty(n, dtype=np.float32)
    meta = np.empty((n, 4), dtype=np.int32)
    for i in range(n):
        step, player, v2 = records[i]
        local_gid, slot, reward = sent_meta[i]
        feats[i] = v2
        labels[i] = reward
        meta[i] = (local_gid, step, player, n_players)

    return (feats, labels, meta, game_names, game_rewards, game_files, gid + 1, skip_format, skip_other)


def build(zip_paths, out_npz: str, n_workers: int, limit=None, n_players: int = 2,
          keep_players=None):
    if isinstance(zip_paths, str):
        zip_paths = [zip_paths]
    # Elo gate (Phase 2): when given, extract feature rows ONLY for these players,
    # so the expensive Rust extraction never touches rows the gate would discard.
    keep_set = None
    if keep_players is not None:
        keep_set = set(json.loads(Path(keep_players).read_text(encoding="utf-8")))
        print(f"keep-players gate: {len(keep_set)} players; other players' rows skipped")
    feats_all, labels_all, meta_all = [], [], []
    game_names_all = []  # global_gid -> tuple(names)
    game_rewards_all = [] # global_gid -> tuple(rewards)
    game_files_all = [] # global_gid -> "tag:entry"
    total_games = total_wrong_format = total_other = 0
    elapsed = 0.0

    for zi, zip_path in enumerate(zip_paths):
        zf = zipfile.ZipFile(zip_path)
        names = [n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/")]
        zf.close()
        names.sort()
        if limit:
            names = names[:limit]
        tag = Path(zip_path).stem
        print(f"\n>>> [{zi+1}/{len(zip_paths)}] {zip_path}: {len(names)} entries  tag={tag}")
        chunks = [names[i::n_workers] for i in range(n_workers)]
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            results = pool.map(process_chunk, [(zip_path, c, i, tag, n_players, keep_set) for i, c in enumerate(chunks)])
        elapsed += time.time() - t0
        for res in results:
            if res is None:
                continue
            f, lbl, m, gnames, grewards, gfiles, n_games, sfmt, soth = res
            if len(f):
                m = m.copy()
                m[:, 0] += total_games  # offset local gids -> global
                feats_all.append(f)
                labels_all.append(lbl)
                meta_all.append(m)
            game_names_all.extend(gnames)
            game_rewards_all.extend(grewards)
            game_files_all.extend(gfiles)
            total_games += n_games
            total_wrong_format += sfmt
            total_other += soth

    if not feats_all:
        raise SystemExit("no samples extracted")

    feats = np.concatenate(feats_all)
    labels = np.concatenate(labels_all)
    meta = np.concatenate(meta_all)

    # --- strong-player gate (computed in memory, stored as a flag) ---
    # Skipped under the Elo keep-players gate: every extracted row is already a
    # kept (top-rated) player, so is_strong is uniformly 1.
    if keep_set is not None:
        rates, median = {}, 0.0
        is_strong = np.ones(meta.shape[0], dtype=np.uint8)
    else:
        pg, pw = defaultdict(int), defaultdict(int)
        for names, rewards in zip(game_names_all, game_rewards_all):
            for name in names:
                pg[name] += 1
            best = max(rewards)
            winners = [i for i, reward in enumerate(rewards) if reward == best]
            if len(winners) == 1:
                pw[names[winners[0]]] += 1
        rates = {pl: pw[pl] / pg[pl] for pl in pg if pg[pl] >= MIN_GAMES}
        sr = sorted(rates.values())
        median = sr[len(sr) // 2] if sr else 0.0
        above = {pl for pl, r in rates.items() if r > median}
        strong_gids = {
            gid for gid, names in enumerate(game_names_all)
            if all(name in above for name in names)
        }
        is_strong = np.fromiter(
            (1 if g in strong_gids else 0 for g in meta[:, 0]),
            dtype=np.uint8, count=meta.shape[0],
        )

    slots = sorted(np.unique(meta[:, 2]).tolist())
    print(f"\n=== build summary ===")
    print(f"{n_players}-player games extracted : {total_games}")
    print(f"wrong-format games skipped: {total_wrong_format}")
    print(f"other/invalid skipped    : {total_other}")
    print(f"observations             : {feats.shape[0]}")
    print(f"player slots present     : {slots}   (expected {list(range(n_players))})")
    if keep_set is not None:
        print(f"elo keep-players gate    : {len(keep_set)} players; {is_strong.shape[0]} rows kept")
    else:
        print(f"unique players (>= {MIN_GAMES} games): {len(rates)}   median win rate: {median:.3f}")
        print(f"strong games             : {len(strong_gids)} / {total_games}")
        print(f"strong observations      : {int(is_strong.sum())} / {is_strong.shape[0]}")
    print(f"NaN/Inf in features      : {int(np.sum(~np.isfinite(feats)))}")
    print(f"label values             : {sorted(np.unique(labels).tolist())[:6]}...")
    print(f"extraction time          : {elapsed:.1f}s")

    # game_names: shape (n_games, n_players) of unicode strings, indexed by global_gid
    name_arr = np.array(game_names_all, dtype="<U64")
    reward_arr = np.array(game_rewards_all, dtype=np.float32)
    file_arr = np.array(game_files_all, dtype="<U200")

    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        summary_v2=feats, labels=labels, meta=meta, is_strong=is_strong,
        game_names=name_arr, game_rewards=reward_arr, game_files=file_arr,
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, nargs="+", help="one or more zip files (built in order)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--limit", type=int, default=None, help="cap entries per zip (debug)")
    p.add_argument("--players", type=int, choices=(2, 4), default=2)
    p.add_argument("--keep-players", type=Path, default=None,
                   help="JSON list of player names (from elo_topn.py); extract ONLY these players' "
                        "rows. The Elo gate, applied during extraction so skipped rows cost nothing.")
    args = p.parse_args()
    build(args.zip, args.out, args.workers, args.limit, args.players, args.keep_players)


if __name__ == "__main__":
    if os.name != "nt":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
