"""AlphaZero-style self-play loop for alphaduck v17.

For each iteration:
  1. Spawn N self-play games (bots/alphaduck/main.py vs itself)        [Phase A]
  2. Build a v17 NPZ chunk from the game zip                           [Phase B]
  3. Train v17 on a rolling window of recent chunks, save new weights  [Phase C]
  4. Archive the previous weights and swap in the new ones

Phases never overlap to keep peak RAM bounded.

Defaults are conservative for a 16 GB Mac. Override for bigger machines:
  --workers 2     (Mac default) → 4 (EC2 30 GB) → 8 (EC2 ≥60 GB)
  --games 50      (Mac default) → 200 (EC2)
  --buffer 5      (chunks kept in training window)
  --train-epochs 3   (Mac CPU is slow; on GPU you can use 10)
  --device cpu    (Mac default; "cuda" on EC2 GPU box; "mps" is broken for PairNetV17)

This script is the orchestrator; do NOT run it from inside a worker.
Each phase shells out so a phase crash does not poison the loop.

RAM budget per host (peak across phases, NOT additive):
  | host           | RAM   | workers | per-worker | safe peak |
  | Mac M4 16 GB   | 16 GB | 2       | ~3 GB      | ~7 GB     |
  | EC2 30 GB CPU  | 30 GB | 4–8     | ~3 GB      | ~12–24 GB |
  | EC2 GPU L4     | 30 GB | 4       | ~3 GB      | ~12 GB    |
"""
from __future__ import annotations
import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ALPHADUCK_MAIN = REPO / "bots" / "alphaduck" / "main.py"
WEIGHTS_PATH = REPO / "bots" / "alphaduck" / "train" / "weights" / "transformer_pair_v17.pt"
BUILD_DATASET_V17 = REPO / "bots" / "alphaduck" / "train" / "build_dataset_v17.py"
TRAIN_V17 = REPO / "bots" / "alphaduck" / "train" / "train_v17_chunked.py"


def log(msg: str, log_path: Path):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as fh:
        fh.write(line + "\n")


def check_ram(min_gb_free: float, log_path: Path) -> bool:
    """Refuse to start a new phase if free RAM is below the threshold."""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["vm_stat"], capture_output=True, text=True)
            # Very rough; macOS doesn't expose "free" cleanly. Defer to "memory pressure".
            pr = subprocess.run(["memory_pressure"], capture_output=True, text=True)
            pct = 100
            for ln in pr.stdout.splitlines():
                if "free percentage" in ln.lower():
                    try:
                        pct = int(ln.split(":")[-1].strip().rstrip("%"))
                    except ValueError:
                        pass
            ok = pct >= 10
            log(f"  RAM: free={pct}% (need ≥10%) → {'ok' if ok else 'LOW'}", log_path)
            return ok
        else:  # linux
            r = subprocess.run(["free", "-g"], capture_output=True, text=True)
            for ln in r.stdout.splitlines():
                if ln.startswith("Mem:"):
                    parts = ln.split()
                    avail = float(parts[6]) if len(parts) >= 7 else float(parts[3])
                    ok = avail >= min_gb_free
                    log(f"  RAM: available={avail:.1f} GB (need ≥{min_gb_free:.0f}) → {'ok' if ok else 'LOW'}", log_path)
                    return ok
        return True
    except Exception as e:
        log(f"  RAM check failed: {e}; assuming ok", log_path)
        return True


def phase_a_play(args, tag: str, log_path: Path) -> Path:
    """Run self-play games. Returns path to chunks/{tag}.zip."""
    log(f"=== Phase A: self-play {args.games} games (workers={args.workers}) ===", log_path)
    if not check_ram(args.min_ram_gb, log_path):
        raise SystemExit("aborting: low RAM before phase A")
    cmd = [
        sys.executable, str(HERE / "batch_play.py"),
        "--tag", tag,
        "--games", str(args.games),
        "--workers", str(args.workers),
        "--start-seed", str(args.start_seed),
        "--act-timeout", str(args.act_timeout),
    ]
    log(f"  $ {' '.join(cmd)}", log_path)
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"phase A failed (code {r.returncode})")
    zip_path = HERE / "chunks" / f"{tag}.zip"
    if not zip_path.exists():
        raise SystemExit(f"phase A produced no zip at {zip_path}")
    log(f"  -> {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)", log_path)
    return zip_path


def phase_b_build(args, tag: str, zip_path: Path, log_path: Path) -> Path:
    """Build a v17 NPZ chunk from the zip. Returns path to the chunk file."""
    log(f"=== Phase B: build v17 chunk from {zip_path.name} ===", log_path)
    if not check_ram(args.min_ram_gb, log_path):
        raise SystemExit("aborting: low RAM before phase B")
    chunk_out = HERE / "chunks" / f"{tag}.npz"
    cmd = [
        sys.executable, str(BUILD_DATASET_V17),
        "--zip", str(zip_path),
        "--out", str(chunk_out),
        "--workers", str(args.workers),
        "--chunk-size", str(args.games + 10),  # one chunk per iter
    ]
    log(f"  $ {' '.join(cmd)}", log_path)
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"phase B failed (code {r.returncode})")
    # build_dataset_v17 saves as <out>.chunk_0000.npz; rename to canonical
    produced = sorted((HERE / "chunks").glob(f"{tag}.npz.chunk_*.npz"))
    if not produced:
        raise SystemExit(f"phase B produced no chunks")
    # We expect exactly one chunk (chunk-size > games). Rename to {tag}.npz.
    canonical = HERE / "chunks" / f"{tag}.npz"
    if canonical.exists():
        canonical.unlink()
    shutil.move(str(produced[0]), str(canonical))
    for extra in produced[1:]:
        log(f"  warning: extra chunk {extra.name}; keeping it", log_path)
    log(f"  -> {canonical} ({canonical.stat().st_size/1e6:.1f} MB)", log_path)
    return canonical


def phase_c_train(args, tag: str, log_path: Path) -> Path:
    """Train v17 on the most recent --buffer chunks. Returns path to new weights."""
    log(f"=== Phase C: train on rolling buffer of {args.buffer} chunks ===", log_path)
    if not check_ram(args.min_ram_gb, log_path):
        raise SystemExit("aborting: low RAM before phase C")
    chunks = sorted((HERE / "chunks").glob("iter*.npz"))
    if len(chunks) == 0:
        raise SystemExit("no chunks to train on")
    window = chunks[-args.buffer:]
    log(f"  training on {len(window)} chunks: {[c.name for c in window]}", log_path)
    # Use a glob that captures exactly the rolling window via temp dir of symlinks.
    train_dir = HERE / "chunks_train"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir()
    for i, c in enumerate(window):
        (train_dir / f"win_{i:04d}.npz").symlink_to(c.resolve())
    new_weights = HERE / "weights_history" / f"{tag}.pt"
    cmd = [
        sys.executable, str(TRAIN_V17),
        "--data-glob", str(train_dir / "*.npz"),
        "--device", args.device,
        "--epochs", str(args.train_epochs),
        "--batch-size", str(args.batch_size),
        "--out", str(new_weights),
    ]
    log(f"  $ {' '.join(cmd)}", log_path)
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"phase C failed (code {r.returncode})")
    if not new_weights.exists():
        raise SystemExit("phase C produced no weights")
    log(f"  -> {new_weights} ({new_weights.stat().st_size/1e6:.1f} MB)", log_path)
    return new_weights


def phase_d_swap(args, new_weights: Path, log_path: Path):
    """Atomic swap of weights so the next phase A picks them up."""
    log(f"=== Phase D: swap weights ===", log_path)
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_PATH.exists():
        backup = HERE / "weights_history" / f"prev_{int(time.time())}.pt"
        shutil.copy2(WEIGHTS_PATH, backup)
        log(f"  backed up old weights to {backup.name}", log_path)
    shutil.copy2(new_weights, WEIGHTS_PATH)
    log(f"  installed new weights at {WEIGHTS_PATH}", log_path)


def gc_chunks(args, log_path: Path):
    """Delete chunks older than the rolling window (don't blow disk)."""
    chunks = sorted((HERE / "chunks").glob("iter*.npz"))
    if len(chunks) <= args.keep_chunks:
        return
    stale = chunks[:-args.keep_chunks]
    for c in stale:
        c.unlink()
        log(f"  gc: removed {c.name}", log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=50, help="self-play games per iter")
    ap.add_argument("--workers", type=int, default=2, help="parallel game subprocesses")
    ap.add_argument("--act-timeout", type=float, default=1.0)
    ap.add_argument("--start-seed", type=int, default=1,
                    help="incremented by --games each iter to avoid replays")
    ap.add_argument("--buffer", type=int, default=5,
                    help="rolling window of recent chunks for training")
    ap.add_argument("--keep-chunks", type=int, default=10,
                    help="chunks kept on disk beyond the training window")
    ap.add_argument("--train-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--min-ram-gb", type=float, default=4.0,
                    help="abort phase if available RAM is below this (Linux only)")
    ap.add_argument("--start-iter", type=int, default=1,
                    help="iter # to start from (resume support)")
    args = ap.parse_args()

    log_path = HERE / "logs" / f"loop_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"alphaduck self-play loop starting", log_path)
    log(f"args: {vars(args)}", log_path)

    seed_cursor = args.start_seed
    for it in range(args.start_iter, args.start_iter + args.iters):
        tag = f"iter{it:04d}"
        log(f"\n========== ITER {it} ({tag}) ==========", log_path)
        try:
            zp = phase_a_play(argparse.Namespace(**vars(args), tag=tag), tag, log_path) \
                 if False else phase_a_play_simple(args, tag, seed_cursor, log_path)
            seed_cursor += args.games
            cp = phase_b_build(args, tag, zp, log_path)
            wp = phase_c_train(args, tag, log_path)
            phase_d_swap(args, wp, log_path)
            gc_chunks(args, log_path)
            log(f"========== iter {it} OK ==========", log_path)
        except SystemExit as e:
            log(f"iter {it} aborted: {e}", log_path)
            break
        except KeyboardInterrupt:
            log(f"interrupted at iter {it}", log_path)
            break


def phase_a_play_simple(args, tag, start_seed, log_path):
    """Wrapper that calls phase_a_play with the per-iter start seed."""
    args2 = argparse.Namespace(**vars(args))
    args2.start_seed = start_seed
    return phase_a_play(args2, tag, log_path)


if __name__ == "__main__":
    main()
