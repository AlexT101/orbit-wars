"""Spawn N self-play games as bounded-concurrency subprocesses, then zip them.

Each game is its own Python subprocess (one play_game.py call) so a crash in
one game does not affect siblings. Concurrency is hard-capped by --workers to
prevent OOM (each game peaks ~3 GB RSS with alphaduck-MCTS).

Output:
  - games/{tag}/g{seed:06d}.json for each game
  - chunks/{tag}.zip — single zip containing all jsons (ready for build_dataset_v17)

Usage:
  python3 batch_play.py --tag iter0001 --games 50 --workers 4 --start-seed 1
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAY_GAME = HERE / "play_game.py"


def run_one(args_tuple):
    out_path, seed, act_timeout = args_tuple
    cmd = [
        sys.executable, str(PLAY_GAME),
        "--out", str(out_path),
        "--seed", str(seed),
        "--act-timeout", str(act_timeout),
    ]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return seed, False, "timeout (>600s)"
    dt = time.time() - t0
    ok = r.returncode == 0 and out_path.exists()
    msg = r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr.strip().splitlines()[-1] if r.stderr else ""
    return seed, ok, f"{dt:.1f}s {msg}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="output dir/zip basename, e.g. iter0001")
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2, help="concurrent game subprocesses; each ~3 GB RSS")
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--act-timeout", type=float, default=1.0)
    ap.add_argument("--games-dir", default=str(HERE / "games"))
    ap.add_argument("--chunks-dir", default=str(HERE / "chunks"))
    args = ap.parse_args()

    games_dir = Path(args.games_dir) / args.tag
    games_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.start_seed, args.start_seed + args.games))
    targets = [(games_dir / f"g{s:06d}.json", s, args.act_timeout) for s in seeds]
    # Skip any already done.
    pending = [t for t in targets if not t[0].exists()]
    print(f"[{args.tag}] {len(targets)} games, {len(pending)} pending, {args.workers} workers", flush=True)

    n_done = 0
    n_fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, t): t for t in pending}
        for fut in as_completed(futures):
            seed, ok, msg = fut.result()
            n_done += 1
            if not ok:
                n_fail += 1
            if n_done % 5 == 0 or not ok:
                print(f"  [{n_done}/{len(pending)}] seed={seed} ok={ok} {msg}", flush=True)
    print(f"[{args.tag}] play done: {n_done} runs, {n_fail} failures, {time.time()-t0:.0f}s", flush=True)

    # Zip up all jsons in games_dir/<tag>/
    zip_path = chunks_dir / f"{args.tag}.zip"
    n_in_zip = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        for js in sorted(games_dir.glob("g*.json")):
            zf.write(js, arcname=js.name)
            n_in_zip += 1
    sz = zip_path.stat().st_size / 1e6
    print(f"[{args.tag}] zipped {n_in_zip} jsons -> {zip_path} ({sz:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
