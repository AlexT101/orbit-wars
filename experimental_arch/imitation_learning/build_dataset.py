from __future__ import annotations

import json
import math
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress nicety.
    tqdm = None

IL_DIR = Path(__file__).resolve().parent
EXPERIMENTAL_ARCH_DIR = IL_DIR.parent
TRAIN_DIR = EXPERIMENTAL_ARCH_DIR / "train_transformer"
REPO_ROOT = EXPERIMENTAL_ARCH_DIR.parent
TROJAN_TRAIN_DIR = REPO_ROOT / "bots" / "mine" / "trojan_horse" / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(TROJAN_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TROJAN_TRAIN_DIR))

from build_from_zip import SUMMARY_V2_DIM, normalize_obs  # noqa: E402
from constants import ACTIONS_DIM, PAIR_TURN_SHAPE, PLANET_SLOTS, SEND_ALL_ACTION  # noqa: E402
from engineered_features import append_engineered_features, append_tempo_features  # noqa: E402
from extras_v4_build import EXTRA_DIM  # noqa: E402
from features import ACTION_DIM, EncodedObs, discrete_action_index, encoded_from_feat  # noqa: E402
from orbit_wars_model import encode_obs as rust_encode_obs  # noqa: E402


REPLAY_DIRS = [
    EXPERIMENTAL_ARCH_DIR / "replays",
]
ISAIAH_NAME = "Isaiah @ Tufa Labs"
XGB_MODEL_PATH = TROJAN_TRAIN_DIR / "weights" / "xgb_46p12e88t11_latest.json"
EXTRACT_V4_BIN = REPO_ROOT / "bots" / "mine" / "trojan_horse" / "target" / "release" / "extract_v4"
OUT_DIR = IL_DIR / "data"
DATASET_FORMAT_VERSION = 4
DATASET_STATS_JSON = OUT_DIR / "isaiah_tufa_labs_2p_wins_bc_v4_stats.json"
DATASET_DIR = OUT_DIR / "isaiah_tufa_labs_2p_wins_bc_v4_chunks"
DATASET_DB = OUT_DIR / "isaiah_tufa_labs_2p_wins_bc_v4.sqlite"
DATASET_MANIFEST_JSON = OUT_DIR / "isaiah_tufa_labs_2p_wins_bc_v4_manifest.json"

MAX_REPLAYS = None
MAX_SAMPLES = None
MAX_ANGLE_ERROR = 0.08
REQUIRE_FULL_SEND = True
FULL_SEND_TOLERANCE = 0.08
VALUE_RECORD_BYTES = 8 + 4 + 4 * SUMMARY_V2_DIM + 4 * EXTRA_DIM
SHOW_PROGRESS = True


@dataclass(frozen=True)
class DatasetBuildConfig:
    player_name: str
    replay_dirs: tuple[str, ...]
    xgb_model_path: str
    extract_v4_bin: str
    action_dim: int
    max_replays: int | None
    max_samples: int | None
    max_angle_error: float
    require_full_send: bool
    full_send_tolerance: float


@dataclass(frozen=True)
class ReplayRef:
    path: Path
    agents: tuple[str, ...]
    rewards: tuple[float, ...]
    player: int


@dataclass(frozen=True)
class ReplaySample:
    encoded: EncodedObs
    label: int
    game_id: str
    step: int
    player: int
    value_obs: dict[str, Any]


@dataclass(frozen=True)
class BuildStats:
    replays_seen: int
    replays_kept: int
    rows: int
    noop_rows: int
    launch_rows: int
    skipped_invalid: int
    skipped_bad_ship_fraction: int
    multi_launch_steps: int
    action_lengths: dict[int, int]
    label_counts: dict[int, int]


@dataclass(frozen=True)
class ReplayBuildResult:
    samples: list[ReplaySample]
    stats: BuildStats


def make_config() -> DatasetBuildConfig:
    return DatasetBuildConfig(
        player_name=ISAIAH_NAME,
        replay_dirs=tuple(str(path) for path in REPLAY_DIRS),
        xgb_model_path=str(XGB_MODEL_PATH),
        extract_v4_bin=str(EXTRACT_V4_BIN),
        action_dim=ACTION_DIM,
        max_replays=MAX_REPLAYS,
        max_samples=MAX_SAMPLES,
        max_angle_error=MAX_ANGLE_ERROR,
        require_full_send=REQUIRE_FULL_SEND,
        full_send_tolerance=FULL_SEND_TOLERANCE,
    )


def progress(iterable, *, desc: str, total: int | None = None):
    if not SHOW_PROGRESS:
        yield from iterable
        return
    if tqdm is not None:
        with tqdm(total=total, desc=desc, dynamic_ncols=True) as bar:
            for item in iterable:
                bar.update(1)
                yield item
        return

    if desc:
        suffix = f" 0/{total}" if total is not None else ""
        print(f"{desc}:{suffix}", flush=True)
    every = max(1, total // 20) if total else 100
    for i, item in enumerate(iterable, start=1):
        yield item
        if i % every == 0 or (total is not None and i == total):
            suffix = f"{i}/{total}" if total is not None else str(i)
            print(f"{desc}: {suffix}", flush=True)


def iter_replay_paths(replay_dirs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for replay_dir in replay_dirs:
        print(f"discovering replay jsons under {replay_dir}", flush=True)
        paths.extend(p for p in replay_dir.rglob("*.json") if p.is_file())
    return sorted(set(paths))


def load_replay(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def replay_metadata(path: Path) -> tuple[list[str], tuple[float, ...]] | None:
    """Read replay metadata without parsing the huge steps payload."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        head = f.read(262_144)
    steps_at = head.find('"steps"')
    if steps_at > 0:
        head = head[:steps_at]
    names = re.findall(r'"Name"\s*:\s*"([^"]*)"', head)
    rewards_match = re.search(r'"rewards"\s*:\s*\[([^\]]*)\]', head)
    if not rewards_match:
        return None
    try:
        rewards = tuple(float(x.strip()) for x in rewards_match.group(1).split(",") if x.strip())
    except ValueError:
        return None
    return names, rewards


def agent_names(replay: dict[str, Any]) -> list[str]:
    agents = (replay.get("info") or {}).get("Agents") or []
    return [str(agent.get("Name", f"p{i}")) for i, agent in enumerate(agents)]


def isaiah_2p_win_ref(path: Path) -> ReplayRef | None:
    meta = replay_metadata(path)
    if meta is None:
        replay = load_replay(path)
        names = agent_names(replay)
        rewards = tuple(float(x) for x in (replay.get("rewards") or []))
    else:
        names, rewards = meta
    if len(names) != 2 or len(rewards) != 2 or ISAIAH_NAME not in names:
        return None
    player = names.index(ISAIAH_NAME)
    if rewards[player] != max(rewards) or rewards.count(rewards[player]) != 1:
        return None
    return ReplayRef(path=path, agents=tuple(names), rewards=rewards, player=player)


def find_isaiah_2p_wins(replay_dirs: Iterable[Path], limit: int | None = None) -> list[ReplayRef]:
    refs: list[ReplayRef] = []
    paths = iter_replay_paths(replay_dirs)
    for path in progress(paths, desc="filter Isaiah 2p wins", total=len(paths)):
        ref = isaiah_2p_win_ref(path)
        if ref is not None:
            refs.append(ref)
            if limit is not None and len(refs) >= limit:
                break
    return refs


def angle_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def ship_fraction(obs: dict[str, Any], move: list[Any]) -> float | None:
    if len(move) < 3:
        return None
    source_id = int(move[0])
    sent = float(move[2])
    for planet in obs.get("planets", []) or []:
        if int(planet[0]) == source_id:
            ships = float(planet[5])
            if ships <= 0.0:
                return None
            return sent / ships
    return None


def obs_with_step(obs: dict[str, Any], turn_index: int) -> dict[str, Any]:
    if "step" in obs and obs.get("step") is not None:
        return obs
    out = dict(obs)
    out["step"] = int(turn_index)
    return out


def encode_replay_move(
    obs: dict[str, Any],
    move: list[Any],
    player: int,
    max_angle_error: float,
    feat: dict[str, Any],
) -> tuple[int | None, str]:
    if len(move) < 3:
        return None, "short_move"
    source_id = int(move[0])
    move_angle = float(move[1])
    planet_ids = [int(x) for x in feat["planet_ids"]]
    try:
        source_slot = planet_ids.index(source_id)
    except ValueError:
        return None, "source_not_slotted"

    best_label: int | None = None
    best_delta = float("inf")
    for target_slot in range(PLANET_SLOTS):
        raw_idx = (source_slot * PLANET_SLOTS + target_slot) * ACTIONS_DIM + SEND_ALL_ACTION
        if raw_idx >= len(feat["mask"]) or not bool(feat["mask"][raw_idx]):
            continue
        delta = angle_delta(float(feat["angles"][raw_idx]), move_angle)
        if delta < best_delta:
            best_delta = delta
            best_label = discrete_action_index(source_slot, target_slot)
    if best_label is None:
        return None, "no_valid_target"
    if best_delta > max_angle_error:
        return None, f"angle_miss_{best_delta:.4f}"
    return best_label, "ok"


def build_replay_samples(
    ref: ReplayRef,
    max_angle_error: float,
    require_full_send: bool,
    full_send_tolerance: float,
    max_samples: int | None = None,
) -> ReplayBuildResult:
    samples: list[ReplaySample] = []
    action_lengths: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()
    skipped_invalid = 0
    skipped_bad_ship_fraction = 0
    multi_launch_steps = 0

    replay = load_replay(ref.path)
    prev_obs: dict[str, Any] | None = None
    for turn_index, turn in enumerate(replay.get("steps") or []):
        if not isinstance(turn, list) or ref.player >= len(turn):
            continue
        entry = turn[ref.player]
        if not isinstance(entry, dict):
            continue
        current_obs = obs_with_step(entry.get("observation") or {}, turn_index)
        if not current_obs.get("planets"):
            continue
        action = entry.get("action") or []
        action_lengths[len(action)] += 1
        if prev_obs is None:
            prev_obs = current_obs
            continue
        obs = prev_obs
        step_num = int(obs.get("step", len(samples)))
        if len(action) > 1:
            multi_launch_steps += 1
        if not action:
            feat = rust_encode_obs(obs, ref.player)
            encoded = encoded_from_feat(feat)
            samples.append(
                ReplaySample(
                    encoded=encoded,
                    label=0,
                    game_id=ref.path.stem,
                    step=step_num,
                    player=ref.player,
                    value_obs=obs,
                )
            )
            label_counts[0] += 1
        else:
            feat = rust_encode_obs(obs, ref.player)
            encoded = encoded_from_feat(feat)
            for move in action:
                frac = ship_fraction(obs, move)
                if require_full_send and (frac is None or abs(frac - 1.0) > full_send_tolerance):
                    skipped_bad_ship_fraction += 1
                    continue
                label, _reason = encode_replay_move(obs, move, ref.player, max_angle_error, feat)
                if label is None:
                    skipped_invalid += 1
                    continue
                if label < 0 or label >= len(encoded.action_mask) or not bool(encoded.action_mask[label]):
                    skipped_invalid += 1
                    continue
                samples.append(
                    ReplaySample(
                        encoded=encoded,
                        label=label,
                        game_id=ref.path.stem,
                        step=step_num,
                        player=ref.player,
                        value_obs=obs,
                    )
                )
                label_counts[label] += 1
        if max_samples is not None and len(samples) >= max_samples:
            break
        prev_obs = current_obs

    if max_samples is not None and len(samples) >= max_samples:
        samples = samples[:max_samples]
        label_counts = Counter(sample.label for sample in samples)

    stats = BuildStats(
        replays_seen=1,
        replays_kept=1 if samples else 0,
        rows=len(samples),
        noop_rows=int(label_counts.get(0, 0)),
        launch_rows=len(samples) - int(label_counts.get(0, 0)),
        skipped_invalid=skipped_invalid,
        skipped_bad_ship_fraction=skipped_bad_ship_fraction,
        multi_launch_steps=multi_launch_steps,
        action_lengths=dict(sorted(action_lengths.items())),
        label_counts=dict(label_counts.most_common(20)),
    )
    return ReplayBuildResult(samples=samples, stats=stats)


def stack_encoded(items: list[EncodedObs]) -> dict[str, np.ndarray]:
    return {
        "planets": np.stack([x.planets for x in items]).astype(np.float32),
        "planet_mask": np.stack([x.planet_mask for x in items]).astype(np.float32),
        "tokens": np.stack([x.tokens for x in items]).astype(np.float32),
        "presence": np.stack([x.presence for x in items]).astype(np.float32),
        "globals_": np.stack([x.globals for x in items]).astype(np.float32),
        "action_mask": np.stack([x.action_mask for x in items]).astype(np.bool_),
        "pair_turns": np.stack([x.pair_turns for x in items]).reshape((-1, *PAIR_TURN_SHAPE)).astype(np.float32),
        "pair_reachable_mask": np.stack([x.pair_reachable_mask for x in items])
        .reshape((-1, *PAIR_TURN_SHAPE))
        .astype(np.float32),
        "planet_timeline_features": np.stack([x.planet_timeline_features for x in items]).astype(np.float32),
    }


def extract_value_features(
    observations: list[dict[str, Any]],
    players: list[int],
    extract_v4_bin: Path,
) -> np.ndarray:
    if len(observations) != len(players):
        raise ValueError("observations and players length mismatch")
    if not observations:
        return np.zeros((0, 157), dtype=np.float32)

    input_lines: list[str] = []
    for obs, player in progress(zip(observations, players), desc="prepare value obs", total=len(observations)):
        norm = normalize_obs(obs)
        norm["player"] = int(player)
        input_lines.append(json.dumps(norm, separators=(",", ":")))
    proc = subprocess.run(
        [str(extract_v4_bin)],
        input=("\n".join(input_lines) + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise RuntimeError(f"extract_v4 exited with {proc.returncode}: {stderr[:500]}")
    expected_bytes = len(observations) * VALUE_RECORD_BYTES
    if len(proc.stdout) != expected_bytes:
        got = len(proc.stdout) // VALUE_RECORD_BYTES
        raise RuntimeError(
            f"extract_v4 returned {got}/{len(observations)} records "
            f"({len(proc.stdout)} bytes, expected {expected_bytes}): {stderr[:500]}"
        )

    summary = np.zeros((len(observations), SUMMARY_V2_DIM), dtype=np.float32)
    extras = np.zeros((len(observations), EXTRA_DIM), dtype=np.float32)
    for i in progress(range(len(observations)), desc="parse value features", total=len(observations)):
        offset = i * VALUE_RECORD_BYTES
        raw = proc.stdout[offset : offset + VALUE_RECORD_BYTES]
        summary[i] = np.frombuffer(raw[12 : 12 + 4 * SUMMARY_V2_DIM], dtype=np.float32)
        extras[i] = np.frombuffer(raw[12 + 4 * SUMMARY_V2_DIM :], dtype=np.float32)

    base = np.concatenate([summary, extras], axis=1).astype(np.float32)
    core = append_engineered_features(base)
    meta = np.zeros((len(observations), 4), dtype=np.int32)
    for i, obs in enumerate(observations):
        meta[i, 0] = i
        meta[i, 1] = int(obs.get("step", 0))
        meta[i, 2] = int(players[i])
        meta[i, 3] = 1 - int(players[i])
    return append_tempo_features(core, meta).astype(np.float32)


def predict_values(features: np.ndarray, xgb_model_path: Path) -> np.ndarray:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(xgb_model_path))
    return booster.predict(xgb.DMatrix(features.astype(np.float32, copy=False))).astype(np.float32)


def sample_weights_from_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    weights = np.where(values < 0.5, 1.0, np.maximum(0.1, 2.0 * (1.0 - values)))
    return weights.astype(np.float32)


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_replays (
            path TEXT PRIMARY KEY,
            stem TEXT NOT NULL,
            chunk_file TEXT,
            rows INTEGER NOT NULL,
            noop_rows INTEGER NOT NULL,
            launch_rows INTEGER NOT NULL,
            skipped_invalid INTEGER NOT NULL,
            skipped_bad_ship_fraction INTEGER NOT NULL,
            multi_launch_steps INTEGER NOT NULL,
            action_lengths_json TEXT NOT NULL,
            label_counts_json TEXT NOT NULL,
            agents_json TEXT NOT NULL,
            rewards_json TEXT NOT NULL,
            player INTEGER NOT NULL,
            value_sum REAL NOT NULL,
            weight_sum REAL NOT NULL,
            value_min REAL,
            value_max REAL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def processed_paths(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT path FROM processed_replays").fetchall()
    return {str(row[0]) for row in rows}


def total_rows(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(SUM(rows), 0) FROM processed_replays").fetchone()
    return int(row[0])


def next_chunk_index(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM processed_replays WHERE chunk_file IS NOT NULL AND chunk_file != ''"
    ).fetchone()
    return int(row[0])


def json_counter(counter: dict[int, int]) -> str:
    return json.dumps({str(k): int(v) for k, v in counter.items()}, sort_keys=True)


def merge_json_counter(dst: Counter[int], raw: str) -> None:
    for key, value in (json.loads(raw or "{}")).items():
        dst[int(key)] += int(value)


def write_samples_chunk(
    samples: list[ReplaySample],
    ref: ReplayRef,
    chunk_index: int,
    cfg: DatasetBuildConfig,
) -> tuple[str, np.ndarray, np.ndarray]:
    observations = [sample.value_obs for sample in samples]
    players = [sample.player for sample in samples]
    print(f"extracting xgb values for {ref.path.name}: {len(samples)} rows", flush=True)
    value_features = extract_value_features(observations, players, Path(cfg.extract_v4_bin))
    values = predict_values(value_features, Path(cfg.xgb_model_path))
    weights = sample_weights_from_values(values)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    encoded = stack_encoded([sample.encoded for sample in samples])
    tensors = {
        "tokens": torch.as_tensor(encoded["tokens"], dtype=torch.float16),
        "presence": torch.as_tensor(encoded["presence"], dtype=torch.bool),
        "globals_": torch.as_tensor(encoded["globals_"], dtype=torch.float32),
        "action_mask": torch.as_tensor(encoded["action_mask"], dtype=torch.bool),
        "pair_turns": torch.as_tensor(np.rint(encoded["pair_turns"] * 20.0).clip(0, 255), dtype=torch.uint8),
        "pair_reachable_mask": torch.as_tensor(encoded["pair_reachable_mask"].astype(np.bool_), dtype=torch.bool),
        "planet_timeline_features": torch.as_tensor(encoded["planet_timeline_features"], dtype=torch.float16),
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "values": torch.as_tensor(values, dtype=torch.float32),
        "weights": torch.as_tensor(weights, dtype=torch.float32),
        "steps": torch.as_tensor([sample.step for sample in samples], dtype=torch.long),
        "players": torch.as_tensor([sample.player for sample in samples], dtype=torch.long),
    }
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", ref.path.stem)[:80]
    chunk_name = f"chunk_{chunk_index:06d}_{safe_stem}.pt"
    tmp_path = DATASET_DIR / f".{chunk_name}.tmp"
    final_path = DATASET_DIR / chunk_name
    payload = {
        "format_version": DATASET_FORMAT_VERSION,
        "player_name": cfg.player_name,
        "game_id": ref.path.stem,
        "replay_path": str(ref.path),
        "agents": list(ref.agents),
        "rewards": list(ref.rewards),
        "tensors": tensors,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)
    return chunk_name, values, weights


def record_processed_replay(
    conn: sqlite3.Connection,
    ref: ReplayRef,
    stats: BuildStats,
    chunk_file: str | None,
    values: np.ndarray,
    weights: np.ndarray,
) -> None:
    stat = ref.path.stat()
    conn.execute(
        """
        INSERT OR REPLACE INTO processed_replays (
            path, stem, chunk_file, rows, noop_rows, launch_rows, skipped_invalid,
            skipped_bad_ship_fraction, multi_launch_steps, action_lengths_json,
            label_counts_json, agents_json, rewards_json, player, value_sum,
            weight_sum, value_min, value_max, size_bytes, mtime_ns
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(ref.path),
            ref.path.stem,
            chunk_file or "",
            int(stats.rows),
            int(stats.noop_rows),
            int(stats.launch_rows),
            int(stats.skipped_invalid),
            int(stats.skipped_bad_ship_fraction),
            int(stats.multi_launch_steps),
            json_counter(stats.action_lengths),
            json_counter(stats.label_counts),
            json.dumps(list(ref.agents)),
            json.dumps(list(ref.rewards)),
            int(ref.player),
            float(values.sum()) if values.size else 0.0,
            float(weights.sum()) if weights.size else 0.0,
            float(values.min()) if values.size else None,
            float(values.max()) if values.size else None,
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ),
    )
    conn.commit()


def aggregate_stats(conn: sqlite3.Connection, cfg: DatasetBuildConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT path, stem, chunk_file, rows, noop_rows, launch_rows, skipped_invalid,
               skipped_bad_ship_fraction, multi_launch_steps, action_lengths_json,
               label_counts_json, value_sum, weight_sum
        FROM processed_replays
        ORDER BY chunk_file, path
        """
    ).fetchall()
    action_lengths: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()
    chunks: list[dict[str, Any]] = []
    values_all: list[np.ndarray] = []
    weights_all: list[np.ndarray] = []
    replays_seen = len(rows)
    replays_kept = 0
    n_rows = 0
    noop_rows = 0
    launch_rows = 0
    skipped_invalid = 0
    skipped_bad_ship_fraction = 0
    multi_launch_steps = 0
    for row in rows:
        (
            path,
            stem,
            chunk_file,
            rows_i,
            noop_i,
            launch_i,
            invalid_i,
            bad_frac_i,
            multi_i,
            action_json,
            label_json,
            _value_sum,
            weight_sum,
        ) = row
        merge_json_counter(action_lengths, action_json)
        merge_json_counter(label_counts, label_json)
        rows_i = int(rows_i)
        if rows_i:
            replays_kept += 1
            chunk_path = DATASET_DIR / str(chunk_file)
            if chunk_path.exists():
                payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
                tensors = payload["tensors"]
                values_all.append(tensors["values"].numpy().astype(np.float32, copy=False))
                weights_all.append(tensors["weights"].numpy().astype(np.float32, copy=False))
            chunks.append(
                {
                    "path": str(chunk_path),
                    "rows": rows_i,
                    "weight_sum": float(weight_sum),
                    "game_id": str(stem),
                    "replay_path": str(path),
                }
            )
        n_rows += rows_i
        noop_rows += int(noop_i)
        launch_rows += int(launch_i)
        skipped_invalid += int(invalid_i)
        skipped_bad_ship_fraction += int(bad_frac_i)
        multi_launch_steps += int(multi_i)

    values = np.concatenate(values_all) if values_all else np.zeros((0,), dtype=np.float32)
    weights = np.concatenate(weights_all) if weights_all else np.zeros((0,), dtype=np.float32)
    labels_total = max(1, n_rows)
    stats = BuildStats(
        replays_seen=replays_seen,
        replays_kept=replays_kept,
        rows=n_rows,
        noop_rows=noop_rows,
        launch_rows=launch_rows,
        skipped_invalid=skipped_invalid,
        skipped_bad_ship_fraction=skipped_bad_ship_fraction,
        multi_launch_steps=multi_launch_steps,
        action_lengths=dict(sorted(action_lengths.items())),
        label_counts=dict(label_counts.most_common(20)),
    )
    stats_payload = {
        "format_version": DATASET_FORMAT_VERSION,
        "config": asdict(cfg),
        "dataset": asdict(stats),
        "value_mean": float(values.mean()) if values.size else 0.0,
        "value_p50": float(np.percentile(values, 50)) if values.size else 0.0,
        "value_p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "sample_weight_mean": float(weights.mean()) if weights.size else 0.0,
        "sample_weight_p10": float(np.percentile(weights, 10)) if weights.size else 0.0,
        "noop_fraction": float(noop_rows / labels_total),
        "launch_fraction": float(launch_rows / labels_total),
        "unique_games": int(len({chunk["game_id"] for chunk in chunks})),
        "chunks": len(chunks),
    }
    return stats_payload, chunks


def write_manifest(conn: sqlite3.Connection, cfg: DatasetBuildConfig) -> dict[str, Any]:
    stats_payload, chunks = aggregate_stats(conn, cfg)
    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "player_name": cfg.player_name,
        "config": asdict(cfg),
        "stats": stats_payload,
        "chunks": chunks,
        "db_path": str(DATASET_DB),
    }
    DATASET_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    DATASET_STATS_JSON.write_text(json.dumps(stats_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    cfg = make_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect_db(DATASET_DB)

    refs = find_isaiah_2p_wins([Path(path) for path in cfg.replay_dirs], limit=cfg.max_replays)
    bad_name_refs = [ref for ref in refs if cfg.player_name not in ref.agents]
    if bad_name_refs:
        examples = ", ".join(ref.path.name for ref in bad_name_refs[:5])
        raise RuntimeError(f"replay filter returned refs without exact player name {cfg.player_name!r}: {examples}")
    print(f"found {len(refs)} exact-name Isaiah 2p wins in {', '.join(cfg.replay_dirs)}", flush=True)

    done = processed_paths(conn)
    if done:
        print(f"resume: {len(done)} replay files already processed, rows={total_rows(conn)}", flush=True)
    chunk_index = next_chunk_index(conn)
    rows_before = total_rows(conn)
    for ref in progress(refs, desc="stream replay chunks", total=len(refs)):
        if str(ref.path) in done:
            continue
        remaining = None if cfg.max_samples is None else max(0, cfg.max_samples - total_rows(conn))
        if remaining == 0:
            break
        result = build_replay_samples(
            ref,
            max_angle_error=cfg.max_angle_error,
            require_full_send=cfg.require_full_send,
            full_send_tolerance=cfg.full_send_tolerance,
            max_samples=remaining,
        )
        if result.samples:
            chunk_file, values, weights = write_samples_chunk(result.samples, ref, chunk_index, cfg)
            chunk_index += 1
        else:
            chunk_file, values, weights = None, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        record_processed_replay(conn, ref, result.stats, chunk_file, values, weights)
        done.add(str(ref.path))
        print(
            f"recorded {ref.path.name}: rows={result.stats.rows} "
            f"launch={result.stats.launch_rows} noop={result.stats.noop_rows} "
            f"total_rows={total_rows(conn)}",
            flush=True,
        )

    manifest = write_manifest(conn, cfg)
    stats_payload = manifest["stats"]
    stats = stats_payload["dataset"]
    if stats["rows"] == 0:
        raise RuntimeError("no imitation samples produced")

    print(
        f"wrote {DATASET_MANIFEST_JSON}\n"
        f"rows={stats['rows']} (+{stats['rows'] - rows_before}) games={stats_payload['unique_games']} "
        f"chunks={stats_payload['chunks']} launch={stats['launch_rows']} noop={stats['noop_rows']} "
        f"invalid={stats['skipped_invalid']} bad_frac={stats['skipped_bad_ship_fraction']} "
        f"weight_mean={stats_payload['sample_weight_mean']:.3f}"
    )
    print(f"wrote {DATASET_STATS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
