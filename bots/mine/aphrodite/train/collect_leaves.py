"""Leaf-distributed value-net training data collector.

Unlike `collect.py` (which dumps the *observed* per-turn state — the same
distribution as replay extraction), this dumps the forward-simulated **DUCT leaf
states** aphrodite actually scores during search, labeled by the eventual game
outcome for the seat that produced them. Training on these closes the
train-on-replays / infer-on-sim-leaves gap.

Sources, mixed per `--self-play-frac`:
  * self-play  — aphrodite vs aphrodite (4p: all aphrodite). On-policy: leaves
                 come from the exact policy you're improving; win/loss balanced.
  * vs bots    — aphrodite vs a local opponent (4p: vs 3 of it, seat-shuffled).
                 Adds opponent-style diversity to the leaf distribution.

Each game runs to completion (or `--max-steps`) at the production search budget
so the leaf distribution matches deployment. Leaves are subsampled per search to
decorrelate (leaves within one search are nearly identical). Output NPZ uses the
exact schema `train_xgb.py` consumes: summary_v2[N,65], labels[N], meta[N,4]
(meta = game_id, leaf_step, seat, n_players).

    python train/collect_leaves.py --out train/data/2p/leaves_v1.npz \
        --players 2 --games 200 --budget-ms 700 --threads 12

Weights default to the live nets (xgb_2p_shapdrop / xgb_4p_shapdrop). The run
checkpoints the NPZ periodically and on SIGINT, so it can be stopped anytime.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # type: ignore  # AphroditeDaemon, load_other_agent, label_for_rewards, MAX_STEPS

SUMMARY_V2_DIM = 65
LEAF_RECORD_BYTES = 4 + 4 + 4 * SUMMARY_V2_DIM  # search_step:i32, leaf_step:i32, feats

DEFAULT_WEIGHTS = {
    2: "train/weights/xgb_2p.json",
    4: "train/weights/xgb_4p.json",
}
# The 2-players-left net (used when a 4p game collapses to 1v1) is always the 2p net.
WEIGHTS_2P_NAME = "train/weights/xgb_2p.json"


def read_leaves(path: Path):
    raw = path.read_bytes() if path.exists() else b""
    n = len(raw) // LEAF_RECORD_BYTES
    if n == 0:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32),
                np.zeros((0, SUMMARY_V2_DIM), np.float32))
    a = np.frombuffer(raw[: n * LEAF_RECORD_BYTES], np.uint8).reshape(n, LEAF_RECORD_BYTES)
    ss = a[:, 0:4].view(np.int32).reshape(n).copy()
    ls = a[:, 4:8].view(np.int32).reshape(n).copy()
    fx = a[:, 8:].view(np.float32).reshape(n, SUMMARY_V2_DIM).copy()
    return ss, ls, fx


def subsample_per_search(ss, ls, fx, k, rng):
    """Keep at most k randomly-chosen leaves per search (decorrelation)."""
    if k <= 0 or fx.shape[0] == 0:
        return ss, ls, fx
    keep = []
    for s in np.unique(ss):
        idx = np.flatnonzero(ss == s)
        if idx.size > k:
            idx = rng.choice(idx, k, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    keep.sort()
    return ss[keep], ls[keep], fx[keep]


def _play_game(job):
    """Worker: run one game, return (arrays, info). Picklable / spawn-safe."""
    (gid, bots, seed, budget_ms, weights, weights2p, players, max_steps,
     leaves_cap, per_search_sample, scratch) = job
    from engine_parity_checker.candidates.rust import RustEngine

    scratch = Path(scratch)
    dumps = [None] * players
    agents = [None] * players
    closers = []
    try:
        for i, name in enumerate(bots):
            if name == "aphrodite":
                lp = scratch / f"g{gid}_p{i}.bin"
                lp.write_bytes(b"")
                dumps[i] = lp
                d = collect.AphroditeDaemon(
                    dump_path=None, budget_ms=budget_ms, weights_path=weights,
                    weights_2p_path=weights2p, leaves_path=lp, leaves_cap=leaves_cap)
                agents[i] = d
                closers.append(d.close)
            else:
                fn, mod = collect.load_other_agent(name)
                agents[i] = fn
                closers.append(lambda m=mod: collect.teardown_other(m))

        engine = RustEngine()
        obs = engine.reset(seed, players)
        t0 = time.time()
        nsteps = 0
        for _ in range(max_steps):
            acts = [agents[i](obs[i].as_dict()) for i in range(players)]
            obs, done = engine.step(acts)
            nsteps += 1
            if done:
                break
        rewards = [float(x) for x in (engine.snapshot().rewards or [0.0] * players)]
        dt = time.time() - t0
    except Exception as e:
        for c in closers:
            try:
                c()
            except Exception:
                pass
        return None, dict(gid=gid, bots=bots, seed=seed,
                          failed=str(e), tb=traceback.format_exc())
    for c in closers:
        try:
            c()
        except Exception:
            pass

    rng = np.random.default_rng(seed * 131071 + gid)
    parts = []
    seat_labels = {}
    for i in range(players):
        if dumps[i] is None:
            continue
        ss, ls, fx = read_leaves(dumps[i])
        try:
            dumps[i].unlink()
        except Exception:
            pass
        if fx.shape[0] == 0:
            continue
        ss, ls, fx = subsample_per_search(ss, ls, fx, per_search_sample, rng)
        lab_val = float(collect.label_for_rewards(rewards, i))
        seat_labels[i] = lab_val
        lab = np.full(fx.shape[0], lab_val, np.float32)
        meta = np.stack([
            np.full(fx.shape[0], gid, np.int32),
            ls.astype(np.int32),
            np.full(fx.shape[0], i, np.int32),
            np.full(fx.shape[0], players, np.int32),
        ], axis=1)
        parts.append((fx, lab, meta))
    if not parts:
        return None, dict(gid=gid, bots=bots, seed=seed, rewards=rewards,
                          rows=0, dt=dt, nsteps=nsteps)
    fx = np.concatenate([p[0] for p in parts])
    lab = np.concatenate([p[1] for p in parts])
    meta = np.concatenate([p[2] for p in parts])
    info = dict(gid=gid, bots=bots, seed=seed, rewards=rewards, rows=fx.shape[0],
                dt=dt, nsteps=nsteps, seat_labels=seat_labels)
    return (fx, lab, meta), info


def build_jobs(args, weights, weights2p, scratch):
    rng = random.Random(args.seed)
    n_self = round(args.games * args.self_play_frac)
    n_vs = args.games - n_self
    sched = ["__self__"] * n_self
    for k in range(n_vs):
        sched.append(args.opponents[k % len(args.opponents)])
    rng.shuffle(sched)
    jobs = []
    for gid, tag in enumerate(sched):
        seed = args.seed * 100000 + gid
        if tag == "__self__":
            bots = ["aphrodite"] * args.players
        else:
            bots = ["aphrodite"] + [tag] * (args.players - 1)
            rng.shuffle(bots)  # seat balance
        jobs.append((gid, bots, seed, args.budget_ms, weights, weights2p,
                     args.players, args.max_steps, args.leaves_cap,
                     args.per_search_sample, str(scratch)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--players", type=int, choices=(2, 4), default=2)
    ap.add_argument("--budget-ms", type=int, default=700,
                    help="production search budget per turn (match deployment)")
    ap.add_argument("--weights", type=Path, default=None,
                    help="play weights (default: live xgb_{2,4}p_shapdrop)")
    ap.add_argument("--opponents", nargs="+",
                    default=["producer", "owheuristic", "apollo", "hellburner",
                             "prometheus_v2"])
    ap.add_argument("--self-play-frac", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=collect.MAX_STEPS,
                    help="cap game length; full games give correct outcome labels")
    ap.add_argument("--leaves-cap", type=int, default=128,
                    help="max leaves dumped per search in Rust (bounds I/O while "
                         "keeping depth variety); 0 = all")
    ap.add_argument("--per-search-sample", type=int, default=32,
                    help="leaves kept per search after random subsampling")
    ap.add_argument("--threads", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    wdir = collect.APHRODITE_DIR
    weights = str((args.weights if args.weights else wdir / DEFAULT_WEIGHTS[args.players]).resolve())
    weights2p = str((wdir / WEIGHTS_2P_NAME).resolve())
    if not Path(weights).is_file():
        raise SystemExit(f"weights not found: {weights}")

    scratch = Path(tempfile.mkdtemp(prefix="aphrodite_leaves_"))
    jobs = build_jobs(args, weights, weights2p, scratch)
    threads = max(1, min(args.threads, len(jobs)))
    print(f"collecting leaves: {args.games} games ({args.players}p), "
          f"self_play={args.self_play_frac:.0%}, budget={args.budget_ms}ms, "
          f"weights={Path(weights).name}, threads={threads}", flush=True)

    all_fx, all_lab, all_meta = [], [], []
    manifest = []
    total_rows = 0
    done_games = 0
    t0 = time.time()

    def flush(tag=""):
        if not all_fx:
            return
        fx = np.concatenate(all_fx)
        lab = np.concatenate(all_lab)
        meta = np.concatenate(all_meta)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, summary_v2=fx, labels=lab, meta=meta)
        winp = float((lab > 0).mean())
        print(f"{tag}wrote {fx.shape[0]:,} rows / {done_games} games "
              f"(win-label frac={winp:.2f}) -> {args.out}", flush=True)
        args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))

    def on_signal(signum, frame):
        flush("[signal] ")
        sys.exit(0)
    signal.signal(signal.SIGINT, on_signal)
    try:
        signal.signal(signal.SIGTERM, on_signal)
    except Exception:
        pass

    def handle(arrays, info):
        nonlocal total_rows, done_games
        done_games += 1
        if "failed" in info:
            print(f"[g{info['gid']}] {info['bots']} FAILED: {info['failed']}", flush=True)
            manifest.append(info)
            return
        if arrays is not None:
            fx, lab, meta = arrays
            all_fx.append(fx)
            all_lab.append(lab)
            all_meta.append(meta)
            total_rows += fx.shape[0]
        rate = done_games / max(time.time() - t0, 1e-3) * 60.0
        print(f"[g{info['gid']} {done_games}/{args.games}] {info['bots']} "
              f"steps={info['nsteps']} rewards={info['rewards']} "
              f"rows={info.get('rows', 0)} total={total_rows:,} "
              f"{info['dt']:.0f}s ({rate:.1f} games/min)", flush=True)
        manifest.append({k: info[k] for k in ("gid", "bots", "seed", "rewards",
                                              "rows", "nsteps", "dt") if k in info})

    if threads == 1:
        for job in jobs:
            arrays, info = _play_game(job)
            handle(arrays, info)
            if done_games % args.checkpoint_every == 0:
                flush("[checkpoint] ")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(_play_game, j) for j in jobs]
            for fut in as_completed(futs):
                arrays, info = fut.result()
                handle(arrays, info)
                if done_games % args.checkpoint_every == 0:
                    flush("[checkpoint] ")

    flush()
    print(f"done: {total_rows:,} rows from {done_games} games in "
          f"{time.time() - t0:.0f}s", flush=True)
    try:
        scratch.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
