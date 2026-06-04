"""Extract v2 summary features from Kaggle Orbit Wars replay JSONs.

For each replay:
  - For each tick, for each player slot, take the observation as the
    bot saw it.
  - Pipe it through `aphrodite` running at APHRODITE_BUDGET_MS=1 with
    APHRODITE_DUMP_FEATURES_PATH set so the bot dumps the 46-d
    summary_v2 feature vector (alongside the 2728-d raw block).
  - Label each row with the final reward of the player who saw that
    state.

Two-player replays only (we skip 4-player for now since the value-net
output is binary).

Output NPZ: features (2728), labels (±1 or 0), meta (game_idx, tick,
player, opp_id), summary_v2 (46).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
APHRODITE_DIR = REPO / "bots" / "mine" / "aphrodite"
BIN_PATH = APHRODITE_DIR / "target" / "release" / "aphrodite"

PER_OBJECT = 9
MAX_OBJECTS = 44
PER_BLOCK = MAX_OBJECTS * PER_OBJECT
DIST_BLOCK = MAX_OBJECTS * MAX_OBJECTS
INPUT_DIM = 2 * PER_BLOCK + DIST_BLOCK
SUMMARY_V2_DIM = 46
RECORD_BYTES = 8 + 4 + 4 * INPUT_DIM + 4 * SUMMARY_V2_DIM


def spawn_bot(dump_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop("APHRODITE_VALUE_NET_PATH", None)
    env["APHRODITE_BUDGET_MS"] = "1"  # near-zero search; we only want features
    env["APHRODITE_DUMP_FEATURES_PATH"] = str(dump_path.resolve())
    return subprocess.Popen(
        [str(BIN_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(APHRODITE_DIR),
        env=env,
        bufsize=0,
    )


def normalize_obs(o: dict) -> dict:
    """Match the wrapper's _norm shape so the bot accepts it."""
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


def read_dump(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = path.read_bytes() if path.exists() else b""
    n = len(raw) // RECORD_BYTES
    if n == 0:
        return (
            np.zeros((0, INPUT_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, SUMMARY_V2_DIM), dtype=np.float32),
        )
    arr = np.frombuffer(raw[: n * RECORD_BYTES], dtype=np.uint8).reshape(n, RECORD_BYTES)
    steps = arr[:, :8].view(np.int64).reshape(n).copy()
    feats = arr[:, 12 : 12 + 4 * INPUT_DIM].view(np.float32).reshape(n, INPUT_DIM).copy()
    v2 = arr[:, 12 + 4 * INPUT_DIM :].view(np.float32).reshape(n, SUMMARY_V2_DIM).copy()
    return feats, steps, v2


def process_replay(path: Path, scratch: Path, game_idx: int):
    data = json.loads(path.read_text())
    rewards = data.get("rewards") or []
    steps = data.get("steps") or []
    if len(rewards) != 2 or not steps:
        return None
    n_players = len(rewards)
    if n_players != 2:
        return None  # 2P only for now

    dumps = [scratch / f"dump_{game_idx}_p{i}.bin" for i in range(2)]
    for d in dumps:
        d.write_bytes(b"")
    procs = [spawn_bot(dumps[i]) for i in range(2)]

    try:
        for tick_idx, step in enumerate(steps):
            if not isinstance(step, list) or len(step) < 2:
                continue
            for slot in range(2):
                entry = step[slot]
                if not isinstance(entry, dict):
                    continue
                obs = entry.get("observation")
                if not obs:
                    continue
                # Skip terminal/done states with no planets list.
                if not obs.get("planets"):
                    continue
                line = json.dumps(normalize_obs(obs), separators=(",", ":")) + "\n"
                try:
                    procs[slot].stdin.write(line.encode())
                    procs[slot].stdin.flush()
                    procs[slot].stdout.readline()  # discard the moves
                except (BrokenPipeError, OSError):
                    return None
    finally:
        for p in procs:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    out = []
    for slot in range(2):
        feats, ticks, v2 = read_dump(dumps[slot])
        if feats.size == 0:
            continue
        label = float(rewards[slot])
        labels = np.full(feats.shape[0], label, dtype=np.float32)
        meta = np.stack(
            [
                np.full(feats.shape[0], game_idx, dtype=np.int32),
                ticks.astype(np.int32),
                np.full(feats.shape[0], slot, dtype=np.int32),
                np.full(feats.shape[0], 1 - slot, dtype=np.int32),
            ],
            axis=1,
        )
        out.append((feats, labels, meta, v2))
        try:
            dumps[slot].unlink()
        except Exception:
            pass
    return out, rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", default=str(REPO / "replays"), help="dir of replay JSONs")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="max replays")
    args = p.parse_args()

    replay_dir = Path(args.replays)
    files = sorted(f for f in replay_dir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no .json files in {replay_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"processing {len(files)} replays from {replay_dir}")

    scratch = Path(tempfile.mkdtemp(prefix="aphrodite_replay_"))
    all_feats, all_labels, all_meta, all_v2 = [], [], [], []
    total = 0
    t0 = time.time()
    for gi, path in enumerate(files):
        result = process_replay(path, scratch, gi)
        if result is None:
            print(f"  [{gi+1}/{len(files)}] {path.name} SKIP (not 2P or empty)")
            continue
        rows, rewards = result
        n_added = 0
        for feats, labels, meta, v2 in rows:
            all_feats.append(feats)
            all_labels.append(labels)
            all_meta.append(meta)
            all_v2.append(v2)
            n_added += feats.shape[0]
        total += n_added
        elapsed = time.time() - t0
        print(
            f"  [{gi+1}/{len(files)}] {path.name} rewards={rewards} +{n_added} samples (total {total}) "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    if not all_feats:
        print("no samples extracted", file=sys.stderr)
        sys.exit(1)
    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    meta = np.concatenate(all_meta, axis=0)
    v2 = np.concatenate(all_v2, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=feats, labels=labels, meta=meta, summary_v2=v2)
    print(f"wrote {feats.shape[0]} samples to {out}")


if __name__ == "__main__":
    main()
