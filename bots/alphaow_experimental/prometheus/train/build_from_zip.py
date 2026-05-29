"""Build a summary_v2 NPZ by streaming replay JSONs directly out of a zip
archive (no disk extraction), feeding each observation to the long-lived
`extract_v2` Rust binary over stdin.

Designed for a ~20GB / 5000-game archive on a disk-constrained machine:
games are read one at a time via zipfile.read() and never unpacked to disk.

Single parse pass:
  * select either 2-player or 4-player games,
  * extract all selected observations -> 46-d summary_v2 features,
  * collect each game's agent names + rewards so the "strong player"
    gate (both players above the median win rate) can be computed in
    memory afterward and stored as a per-sample is_strong flag.

The strong gate is NOT applied destructively here -- every selected sample is
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
BIN_4P_V1 = ALPHAOW_DIR / "target" / "release" / "extract_4p_v1"
BIN_4P_V2 = ALPHAOW_DIR / "target" / "release" / "extract_4p_v2"

SUMMARY_V2_DIM = 46
RECORD_BYTES = 8 + 4 + 4 * SUMMARY_V2_DIM  # 196
FEATURE_SETS = {
    "summary_v2": (SUMMARY_V2_DIM, BIN),
    "4p_v1": (236, BIN_4P_V1),
    "4p_v2": (278, BIN_4P_V2),
}

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


def balanced_4p_labels(rewards: list[float], win_value: float = 0.99) -> list[float]:
    vals = [float(r) for r in rewards]
    max_r = max(vals)
    winners = [i for i, r in enumerate(vals) if r == max_r]
    losers = [i for i in range(len(vals)) if i not in winners]
    if not winners or not losers:
        return [0.0 for _ in vals]
    out = [0.0 for _ in vals]
    win_label = win_value / len(winners)
    lose_label = -win_value / len(losers)
    for i in winners:
        out[i] = win_label
    for i in losers:
        out[i] = lose_label
    return out


def resolve_label_mode(game_mode: str, label_mode: str) -> str:
    if label_mode == "auto":
        return "ordinal" if game_mode == "4p" else "native"
    if game_mode == "2p" and label_mode != "native":
        raise ValueError("2p extraction only supports native labels")
    return label_mode


def final_observation(data: dict, n_slots: int) -> dict | None:
    steps = data.get("steps") or []
    for step in reversed(steps):
        if not isinstance(step, list):
            continue
        for slot in range(min(n_slots, len(step))):
            entry = step[slot]
            if not isinstance(entry, dict):
                continue
            obs = entry.get("observation")
            if isinstance(obs, dict) and obs.get("planets"):
                return obs
    return None


def final_ship_totals(data: dict, n_slots: int) -> list[float] | None:
    obs = final_observation(data, n_slots)
    if obs is None:
        return None
    totals = [0.0 for _ in range(n_slots)]
    for p in obs.get("planets", []) or []:
        if len(p) < 6:
            continue
        owner = int(p[1])
        if 0 <= owner < n_slots:
            totals[owner] += float(p[5])
    for f in obs.get("fleets", []) or []:
        if len(f) < 7:
            continue
        owner = int(f[1])
        if 0 <= owner < n_slots:
            totals[owner] += float(f[6])
    return totals


def ordinal_4p_labels(scores: list[float]) -> list[float]:
    rank_values = [1.0, 0.33, -0.33, -1.0]
    vals = [float(s) for s in scores[:4]]
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    out = [0.0 for _ in vals]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and vals[order[end]] == vals[order[start]]:
            end += 1
        label = float(sum(rank_values[start:end]) / (end - start))
        for j in range(start, end):
            out[order[j]] = label
        start = end
    return out


def labels_for_game(data: dict, rewards: list[float], game_mode: str, label_mode: str) -> list[float]:
    label_mode = resolve_label_mode(game_mode, label_mode)
    if game_mode == "4p" and label_mode == "balanced":
        return balanced_4p_labels(rewards[:4])
    if game_mode == "4p" and label_mode == "ordinal":
        scores = final_ship_totals(data, 4)
        if scores is not None:
            return ordinal_4p_labels(scores)
        return ordinal_4p_labels(rewards[:4])
    if game_mode == "4p":
        return [float(r) for r in rewards[:4]]
    return [float(r) for r in rewards[:2]]


def process_chunk(args):
    """One worker: open its own zip handle + one extract_v2 subprocess,
    stream all assigned games through it."""
    zip_path, names, worker_id, zip_tag, game_mode, feature_set, feature_dim, bin_path, label_mode = args
    n_slots = 4 if game_mode == "4p" else 2
    record_bytes = 8 + 4 + 4 * feature_dim
    proc = subprocess.Popen(
        [str(bin_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    records = []  # (step, player, v2) in send order

    def reader():
        out = proc.stdout
        while True:
            raw = out.read(record_bytes)
            if not raw or len(raw) < record_bytes:
                break
            step = int.from_bytes(raw[:8], "little", signed=True)
            player = int.from_bytes(raw[8:12], "little", signed=True)
            v2 = np.frombuffer(raw[12:], dtype=np.float32).copy()
            records.append((step, player, v2))

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    sent_meta = []        # parallel to records: (local_gid, slot, label)
    game_info = []        # local_gid -> (agent_names, rewards)
    game_files = []       # local_gid -> "tag:entry" source filename
    gid = -1
    skip_wrong_mode = 0
    skip_other = 0

    for entry in names:
        try:
            data = json.loads(read_entry(zip_path, entry))
        except Exception:
            skip_other += 1
            continue
        rewards = data.get("rewards") or []
        if len(rewards) != n_slots:
            skip_wrong_mode += 1
            continue
        steps = data.get("steps") or []
        agents = _agent_names(data)
        if (
            len(agents) < n_slots
            or any(r is None for r in rewards[:n_slots])
            or not steps
        ):
            skip_other += 1
            continue

        gid += 1
        game_rewards = [float(r) for r in rewards[:n_slots]]
        game_labels = labels_for_game(data, game_rewards, game_mode, label_mode)
        game_info.append((tuple(agents[:n_slots]), tuple(game_rewards)))
        game_files.append(f"{zip_tag}:{entry}")

        wrote_any = False
        for step in steps:
            if not isinstance(step, list) or len(step) < n_slots:
                continue
            for slot in range(n_slots):
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
                sent_meta.append((gid, slot, float(game_labels[slot])))
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
        return ([], [], [], game_info, game_files, gid + 1, skip_wrong_mode, skip_other)

    feats = np.empty((n, feature_dim), dtype=np.float32)
    labels = np.empty(n, dtype=np.float32)
    meta = np.empty((n, 4), dtype=np.int32)
    for i in range(n):
        step, player, v2 = records[i]
        local_gid, slot, reward = sent_meta[i]
        feats[i] = v2
        labels[i] = reward
        meta[i] = (local_gid, step, player, 1 - player if n_slots == 2 else -1)

    return (feats, labels, meta, game_info, game_files, gid + 1, skip_wrong_mode, skip_other)


def build(
    zip_paths,
    out_npz: str,
    n_workers: int,
    limit=None,
    game_mode: str = "2p",
    feature_set: str = "summary_v2",
    label_mode: str = "auto",
):
    if isinstance(zip_paths, str):
        zip_paths = [zip_paths]
    if game_mode not in {"2p", "4p"}:
        raise ValueError(f"game_mode must be '2p' or '4p', got {game_mode!r}")
    label_mode = resolve_label_mode(game_mode, label_mode)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}, got {feature_set!r}")
    feature_dim, bin_path = FEATURE_SETS[feature_set]
    n_slots = 4 if game_mode == "4p" else 2
    feats_all, labels_all, meta_all = [], [], []
    game_info_all = []  # global_gid -> (agent_names, rewards)
    game_files_all = [] # global_gid -> "tag:entry"
    total_games = total_wrong_mode = total_other = 0
    elapsed = 0.0

    for zi, zip_path in enumerate(zip_paths):
        names = list_json_entries(zip_path, limit)
        tag = source_tag(zip_path)
        print(
            f"\n>>> [{zi+1}/{len(zip_paths)}] {zip_path}: {len(names)} entries  "
            f"tag={tag} mode={game_mode} features={feature_set} labels={label_mode}"
        )
        chunks = [names[i::n_workers] for i in range(n_workers)]
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            results = pool.map(
                process_chunk,
                [
                    (zip_path, c, i, tag, game_mode, feature_set, feature_dim, bin_path, label_mode)
                    for i, c in enumerate(chunks)
                ],
            )
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
            total_wrong_mode += s4p
            total_other += soth

    if not feats_all:
        raise SystemExit("no samples extracted")

    feats = np.concatenate(feats_all)
    labels = np.concatenate(labels_all)
    meta = np.concatenate(meta_all)

    # --- strong-player gate (computed in memory, stored as a flag) ---
    pg = defaultdict(int)
    pw = defaultdict(float)
    for names, rewards in game_info_all:
        max_r = max(rewards)
        winners = [i for i, r in enumerate(rewards) if r == max_r]
        win_credit = 1.0 / len(winners) if winners else 0.0
        for i, name in enumerate(names):
            pg[name] += 1
            if i in winners:
                pw[name] += win_credit
    rates = {pl: pw[pl] / pg[pl] for pl in pg if pg[pl] >= MIN_GAMES}
    sr = sorted(rates.values())
    median = sr[len(sr) // 2] if sr else 0.0
    above = {pl for pl, r in rates.items() if r > median}
    strong_gids = {
        gid for gid, (names, _rewards) in enumerate(game_info_all)
        if all(name in above for name in names)
    }
    is_strong = np.fromiter(
        (1 if g in strong_gids else 0 for g in meta[:, 0]),
        dtype=np.uint8, count=meta.shape[0],
    )

    slots = sorted(np.unique(meta[:, 2]).tolist())
    print(f"\n=== build summary ===")
    print(f"{game_mode} games extracted      : {total_games}")
    print(f"wrong-mode games filtered: {total_wrong_mode}")
    print(f"other/invalid skipped    : {total_other}")
    print(f"observations             : {feats.shape[0]}")
    print(f"player slots present     : {slots}   (must be {list(range(n_slots))})")
    print(f"unique players (>= {MIN_GAMES} games): {len(rates)}   median win rate: {median:.3f}")
    print(f"strong games             : {len(strong_gids)} / {total_games}")
    print(f"strong observations      : {int(is_strong.sum())} / {is_strong.shape[0]}")
    print(f"NaN/Inf in features      : {int(np.sum(~np.isfinite(feats)))}")
    print(f"label values             : {sorted(np.unique(labels).tolist())[:6]}...")
    print(f"extraction time          : {elapsed:.1f}s")

    # game_names: shape (n_games, players) of unicode strings, indexed by global_gid
    name_arr = np.array([g[0] for g in game_info_all], dtype="<U64")
    reward_arr = np.array([g[1] for g in game_info_all], dtype=np.float32)
    file_arr = np.array(game_files_all, dtype="<U200")

    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        labels=labels,
        meta=meta,
        is_strong=is_strong,
        game_names=name_arr,
        game_rewards=reward_arr,
        game_files=file_arr,
        game_player_count=np.full(len(game_info_all), n_slots, dtype=np.int16),
        game_mode=np.array(game_mode, dtype="<U2"),
        feature_set=np.array(feature_set, dtype="<U16"),
        label_mode=np.array(label_mode, dtype="<U16"),
        entry_limit=np.array(-1 if limit is None else int(limit), dtype=np.int32),
    )
    if feature_set == "summary_v2":
        payload["summary_v2"] = feats
    else:
        payload["features"] = feats
    np.savez_compressed(out, **payload)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, nargs="+", help="one or more zip files (built in order)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--limit", type=int, default=None, help="cap entries per zip (debug)")
    p.add_argument("--game-mode", choices=["2p", "4p"], default="2p")
    p.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="summary_v2")
    p.add_argument("--label-mode", choices=["auto", "native", "balanced", "ordinal"], default="auto")
    args = p.parse_args()
    build(args.zip, args.out, args.workers, args.limit, args.game_mode, args.feature_set, args.label_mode)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
