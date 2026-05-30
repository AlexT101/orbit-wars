"""Extract entity-transformer token tensors from Kaggle replay JSONs.

Output NPZ:
  tokens: float32 [N, 77, 24]
  mask:   bool    [N, 77]
  summary_v2: float32 [N, 66] route-augmented summary features
  labels: float32 [N]       training target from that player's perspective
  labels_raw: float32 [N]   unshaped final reward sign
  finish_step: int32 [N]    game finish step, for time-shaped targets
  meta:   int32   [N, 4]    (game_idx, step, player, opponent)
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ALPHAOW_DIR = REPO / "bots" / "mine" / "alphaow"
BIN = ALPHAOW_DIR / "target" / "release" / "extract_tokens"

MAX_TOKENS = 77
TOKEN_DIM = 24
SUMMARY_DIM = 66
RECORD_BYTES = 8 + 4 + 4 * MAX_TOKENS * TOKEN_DIM + MAX_TOKENS + 4 * SUMMARY_DIM
EPISODE_STEPS = 500


class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"


def tag(name: str, color: str = C.CYAN) -> str:
    return f"{color}{name:>8s}{C.RESET}"


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def human_int(n: int) -> str:
    return f"{n:,}"


def verify_binary_layout():
    try:
        completed = subprocess.run(
            [str(BIN), "--record-bytes"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        raise SystemExit(f"could not query {BIN}: {exc}") from exc
    raw = completed.stdout.strip()
    if completed.returncode != 0 or not raw.isdigit():
        raise SystemExit(
            f"{BIN} does not report record layout. Rebuild it with:\n"
            f"  cd {ALPHAOW_DIR} && cargo build --release\n"
            f"stdout={raw!r} stderr={completed.stderr.strip()!r}"
        )
    got = int(raw)
    if got != RECORD_BYTES:
        raise SystemExit(
            f"extract_tokens record layout mismatch: Python expects {RECORD_BYTES} bytes "
            f"(summary_dim={SUMMARY_DIM}), binary reports {got}. Rebuild with:\n"
            f"  cd {ALPHAOW_DIR} && cargo build --release"
        )


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


def reward_sign(reward: float) -> float:
    if reward > 0:
        return 1.0
    if reward < 0:
        return -1.0
    return 0.0


def shaped_label(reward: float, finish_step: int, target_mode: str, time_coef: float, episode_steps: int) -> float:
    sign = reward_sign(reward)
    if sign == 0.0 or target_mode == "outcome":
        return sign
    finish_frac = max(0.0, min(1.0, float(finish_step) / max(float(episode_steps), 1.0)))
    # Wins faster are closer to +1; losses slower are closer to 0.
    # The sign still dominates, and magnitude never exceeds 1.
    magnitude = 1.0 - max(0.0, min(time_coef, 0.95)) * finish_frac
    return sign * magnitude


def read_replay_json(path: Path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw)


def replay_finish_step(steps: list) -> int:
    best = 0
    for step in steps:
        if not isinstance(step, list):
            continue
        for entry in step:
            if isinstance(entry, dict):
                obs = entry.get("observation")
                if isinstance(obs, dict):
                    try:
                        best = max(best, int(obs.get("step", best)))
                    except (TypeError, ValueError):
                        pass
    return best if best > 0 else max(0, len(steps) - 1)


def process_chunk(args):
    files, worker_id, target_mode, time_coef, episode_steps = args
    proc = subprocess.Popen(
        [str(BIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    records = []
    invalid_records = 0

    def reader():
        nonlocal invalid_records
        out = proc.stdout
        while True:
            raw = out.read(RECORD_BYTES)
            if not raw or len(raw) < RECORD_BYTES:
                break
            step = int.from_bytes(raw[:8], "little", signed=True)
            player = int.from_bytes(raw[8:12], "little", signed=True)
            tok_end = 12 + 4 * MAX_TOKENS * TOKEN_DIM
            tokens = np.frombuffer(raw[12:tok_end], dtype=np.float32).copy().reshape(MAX_TOKENS, TOKEN_DIM)
            mask_end = tok_end + MAX_TOKENS
            mask = np.frombuffer(raw[tok_end:mask_end], dtype=np.uint8).copy().astype(bool)
            summary = np.frombuffer(raw[mask_end:], dtype=np.float32).copy()
            if not (0 <= step <= EPISODE_STEPS + 5 and 0 <= player <= 3 and mask.shape[0] == MAX_TOKENS):
                invalid_records += 1
                continue
            records.append((step, player, tokens, mask, summary))

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    sent_meta = []
    n_games = 0
    skipped = 0
    t0 = time.time()
    for file_idx, path in enumerate(files):
        try:
            data = read_replay_json(path)
        except Exception:
            skipped += 1
            continue
        # Every observation frame from each player becomes one sample.
        # The label is the final reward for that player, so the evaluator
        # learns "does this frame eventually win?" rather than imitation.
        rewards = data.get("rewards") or []
        steps = data.get("steps") or []
        if len(rewards) != 2 or not steps:
            skipped += 1
            continue
        try:
            rewards = [float(rewards[0]), float(rewards[1])]
        except (TypeError, ValueError):
            skipped += 1
            continue
        finish_step = replay_finish_step(steps)
        wrote_any = False
        for step in steps:
            if not isinstance(step, list) or len(step) < 2:
                continue
            for slot in range(2):
                entry = step[slot]
                if not isinstance(entry, dict):
                    continue
                obs = entry.get("observation")
                if not obs or not obs.get("planets"):
                    continue
                norm = normalize_obs(obs)
                try:
                    proc.stdin.write((json.dumps(norm, separators=(",", ":")) + "\n").encode())
                except BrokenPipeError:
                    return None
                sent_meta.append((file_idx, slot, rewards[slot], finish_step))
                wrote_any = True
        if wrote_any:
            try:
                proc.stdin.flush()
            except BrokenPipeError:
                return None
        n_games += 1
        if (file_idx + 1) % 100 == 0 or file_idx + 1 == len(files):
            elapsed = time.time() - t0
            done = file_idx + 1
            rate = done / max(elapsed, 1e-6)
            eta = (len(files) - done) / max(rate, 1e-6)
            print(
                f"{tag('worker', C.DIM)} w{worker_id} {done:,}/{len(files):,} "
                f"sent={n_games:,} skipped={skipped:,} rate={rate:.1f}/s eta={fmt_time(eta)}",
                flush=True,
            )

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
    if n == 0:
        return None
    if invalid_records:
        raise RuntimeError(
            f"worker {worker_id}: got {invalid_records} invalid binary records. "
            f"This usually means extract_tokens has a stale record layout; run "
            f"`cd {ALPHAOW_DIR} && cargo build --release` and re-extract."
        )
    if n < len(sent_meta):
        print(f"  [w{worker_id}] WARN got {len(records)} records for {len(sent_meta)} sent", flush=True)

    tokens, masks, summaries, labels, labels_raw, finish_steps, meta = [], [], [], [], [], [], []
    for i in range(n):
        step, player, tok, mask, summary = records[i]
        file_idx, slot, reward, finish_step = sent_meta[i]
        tokens.append(tok)
        masks.append(mask)
        summaries.append(summary)
        labels.append(shaped_label(reward, finish_step, target_mode, time_coef, episode_steps))
        labels_raw.append(reward_sign(reward))
        finish_steps.append(finish_step)
        meta.append((file_idx, step, player, 1 - player))

    return (
        np.stack(tokens).astype(np.float32),
        np.stack(masks).astype(bool),
        np.stack(summaries).astype(np.float32),
        np.array(labels, dtype=np.float32),
        np.array(labels_raw, dtype=np.float32),
        np.array(finish_steps, dtype=np.int32),
        np.array(meta, dtype=np.int32),
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
    p.add_argument("--target-mode", choices=["time", "outcome"], default="time")
    p.add_argument("--time-coef", type=float, default=0.10)
    p.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    args = p.parse_args()

    if not BIN.exists():
        print(f"missing {BIN}; run `cargo build --release` in {ALPHAOW_DIR}", file=sys.stderr)
        sys.exit(1)
    verify_binary_layout()

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        files = [Path(p) for p in manifest["files"]]
    else:
        files = sorted(Path(args.replays).glob("*.json"))
        files += sorted(Path(args.replays).glob("*.json.gz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("no replay JSON files found", file=sys.stderr)
        sys.exit(1)

    print(
        f"{tag('extract', C.BOLD)} processing {human_int(len(files))} replays with {args.workers} workers "
        f"target={args.target_mode} time_coef={args.time_coef:g}"
    )
    chunks = [files[i :: args.workers] for i in range(args.workers)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(
            process_chunk,
            [(c, i, args.target_mode, args.time_coef, args.episode_steps) for i, c in enumerate(chunks)],
        )

    all_tokens, all_masks, all_summaries, all_labels, all_labels_raw, all_finish_steps, all_meta = [], [], [], [], [], [], []
    total_games = total_skipped = 0
    for res in results:
        if res is None:
            continue
        tokens, masks, summaries, labels, labels_raw, finish_steps, meta, n_games, skipped = res
        meta = meta.copy()
        meta[:, 0] += total_games
        all_tokens.append(tokens)
        all_masks.append(masks)
        all_summaries.append(summaries)
        all_labels.append(labels)
        all_labels_raw.append(labels_raw)
        all_finish_steps.append(finish_steps)
        all_meta.append(meta)
        total_games += n_games
        total_skipped += skipped

    if not all_tokens:
        print("no samples extracted", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tokens=np.concatenate(all_tokens),
        mask=np.concatenate(all_masks),
        summary_v2=np.concatenate(all_summaries),
        labels=np.concatenate(all_labels),
        labels_raw=np.concatenate(all_labels_raw),
        finish_step=np.concatenate(all_finish_steps),
        meta=np.concatenate(all_meta),
        target_mode=args.target_mode,
        time_coef=np.array([args.time_coef], dtype=np.float32),
        episode_steps=np.array([args.episode_steps], dtype=np.int32),
    )
    print(
        f"{tag('written', C.GREEN)} {human_int(sum(x.shape[0] for x in all_tokens))} samples "
        f"({human_int(total_games)} games, {human_int(total_skipped)} skipped) "
        f"-> {out} in {fmt_time(time.time() - t0)}"
    )


if __name__ == "__main__":
    main()
