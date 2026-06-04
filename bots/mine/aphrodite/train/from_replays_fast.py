"""Fast feature extraction from kaggle replay JSONs using the
dedicated `extract_v2` Rust binary (no MCTS, no daemons, just feature
extraction). 100x+ faster than spawning the full bot per game.

Strategy: one long-lived extract_v2 subprocess per worker, fed every
observation across many replays as JSON lines on stdin. Reads back the
fixed-size binary records on stdout.

Supports --workers for parallel processing.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
APHRODITE_DIR = REPO / "bots" / "mine" / "aphrodite"
BIN = APHRODITE_DIR / "target" / "release" / "extract_v2"

SUMMARY_V2_DIM = 46
RECORD_BYTES = 8 + 4 + 4 * SUMMARY_V2_DIM  # 196


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


def process_chunk(args):
    files, worker_id, n_players = args
    proc = subprocess.Popen(
        [str(BIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    # A background thread drains stdout continuously so the extract_v2
    # stdout pipe never fills (which would block its stdin reads and
    # deadlock against our writes). Records come back in send order.
    records = []  # list of (step, player, v2)

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

    sent_meta = []  # parallel to records: (file_idx, slot, reward)
    n_games = 0
    skipped = 0

    for file_idx, path in enumerate(files):
        try:
            data = json.loads(path.read_bytes())
        except Exception:
            skipped += 1
            continue
        rewards = data.get("rewards") or []
        steps = data.get("steps") or []
        if len(rewards) != n_players or not steps:
            skipped += 1
            continue

        wrote_any = False
        for tick_idx, step in enumerate(steps):
            if not isinstance(step, list) or len(step) < n_players:
                continue
            for slot in range(n_players):
                entry = step[slot]
                if not isinstance(entry, dict):
                    continue
                obs = entry.get("observation")
                if not obs or not obs.get("planets"):
                    continue
                norm = normalize_obs(obs)
                line = json.dumps(norm, separators=(",", ":")) + "\n"
                try:
                    proc.stdin.write(line.encode())
                except BrokenPipeError:
                    return None
                sent_meta.append((file_idx, slot, label_for_rewards(rewards, slot)))
                wrote_any = True
        if wrote_any:
            try:
                proc.stdin.flush()
            except BrokenPipeError:
                return None

        n_games += 1
        if (file_idx + 1) % 50 == 0:
            print(f"  [w{worker_id}] {file_idx + 1}/{len(files)} games sent", flush=True)

    try:
        proc.stdin.close()
    except Exception:
        pass
    rt.join(timeout=120)
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
        return None

    feats_list = []
    labels_list = []
    meta_list = []
    for i in range(n):
        step, player, v2 = records[i]
        file_idx, slot, reward = sent_meta[i]
        feats_list.append(v2)
        labels_list.append(reward)
        meta_list.append((file_idx, step, player, n_players))

    return (
        np.stack(feats_list).astype(np.float32),
        np.array(labels_list, dtype=np.float32),
        np.array(meta_list, dtype=np.int32),
        n_games,
        skipped,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", default=str(REPO / "replays"))
    p.add_argument("--manifest", default=None, help="JSON with {'files': [...]} list, overrides --replays")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--players", type=int, choices=(2, 4), default=2)
    args = p.parse_args()

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        files = [Path(p) for p in manifest["files"]]
    else:
        replay_dir = Path(args.replays)
        files = sorted(replay_dir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no .json files found", file=sys.stderr)
        sys.exit(1)
    src = args.manifest or args.replays
    print(f"processing {len(files)} replays from {src} with {args.workers} workers")

    # Split into chunks.
    chunks = [files[i :: args.workers] for i in range(args.workers)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(process_chunk, [(c, i, args.players) for i, c in enumerate(chunks)])

    elapsed = time.time() - t0
    feats_all, labels_all, meta_all = [], [], []
    total_games = total_skipped = 0
    for res in results:
        if res is None:
            continue
        f, lbl, m, n_games, n_skip = res
        # Offset meta game_idx by worker boundary.
        m = m.copy()
        m[:, 0] += total_games  # rough offset to keep ids unique-ish
        feats_all.append(f)
        labels_all.append(lbl)
        meta_all.append(m)
        total_games += n_games
        total_skipped += n_skip

    if not feats_all:
        print("no samples extracted", file=sys.stderr)
        sys.exit(1)
    feats = np.concatenate(feats_all)
    labels = np.concatenate(labels_all)
    meta = np.concatenate(meta_all)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save with summary_v2 as the primary 'features'-like key, plus
    # legacy 'features' as zeros for trainer compat (trainer reads
    # summary_v2 directly).
    np.savez_compressed(out, summary_v2=feats, labels=labels, meta=meta)
    print(f"wrote {feats.shape[0]} samples ({total_games} games, {total_skipped} skipped) to {out} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
