"""Build an imitation-learning dataset by streaming ladder replay zips.

This is the policy-data counterpart to aphrodite's value-net
``train/build_from_zip.py``. It reads replay JSONs directly from
``ladder_replays/*.zip``, encodes the observation with ``orbit_wars_model``,
matches the next replay action to the discrete policy label used by
``experimental_arch/train_transformer/features.py``, and writes the chunked
NPZ manifest format consumed by ``experimental_arch/imitation_learning/train.py``.

Typical 4p run:

    # Optional but recommended: gate to strong 4p players first.
    .\\venv\\Scripts\\python.exe bots\\mine\\aphrodite\\train\\elo_topn.py ^
      --zip ladder_replays\\replays_6_*.zip --players 4 --top-n 20 ^
      --out experimental_arch\\imitation_learning\\data\\top20_4p.json

    .\\venv\\Scripts\\python.exe experimental_arch\\imitation_learning\\build_dataset_from_zips.py ^
      --zip ladder_replays\\replays_6_*.zip --players 4 ^
      --keep-players experimental_arch\\imitation_learning\\data\\top20_4p.json ^
      --out-dir experimental_arch\\imitation_learning\\data\\osteo_top20_4p

Specific-player run:

    .\\venv\\Scripts\\python.exe experimental_arch\\imitation_learning\\build_dataset_from_zips.py ^
      --zip ladder_replays\\replays_*.zip --players 4 ^
      --player-name "Isaiah @ Tufa Labs" --launch-only ^
      --out-dir experimental_arch\\imitation_learning\\data\\isaiah_tufa_labs_4p_launches

Then train with:

    $env:IL_DATASET_PATH = "experimental_arch\\imitation_learning\\data\\osteo_top20_4p\\manifest.json"
    $env:IL_OUT_DIR = "experimental_arch\\imitation_learning\\checkpoints\\osteo_bc_transformer_4p"
    $env:IL_DATASET_NAME = "osteo_top20_4p"
    .\\venv\\Scripts\\python.exe experimental_arch\\imitation_learning\\train.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

IL_DIR = Path(__file__).resolve().parent
EXPERIMENTAL_ARCH_DIR = IL_DIR.parent
REPO_ROOT = EXPERIMENTAL_ARCH_DIR.parent
TRAIN_DIR = EXPERIMENTAL_ARCH_DIR / "train_transformer"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from constants import (  # noqa: E402
    ACTIONS_DIM,
    PAIR_OUTCOME_SHAPE,
    PAIR_TURN_SHAPE,
    PLANET_TIMELINE_SHAPE,
    TOKEN_SHAPE,
)
DATASET_FORMAT_VERSION = 2
DEFAULT_OUT_DIR = IL_DIR / "data" / "osteo_il_from_zips"
MAX_ANGLE_ERROR = 0.08
SEND_FRACTION_TOLERANCE = 0.25
DEFAULT_CHUNK_ROWS = 2048
REPLAY_HEAD_BYTES = 262_144

_NAME_RE = re.compile(rb'"Name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_REWARDS_RE = re.compile(rb'"rewards"\s*:\s*\[([^\]]*)\]')
_STEPS_RE = re.compile(rb'"steps"')

ACTION_DIM = 1 + TOKEN_SHAPE[1] * TOKEN_SHAPE[1] * 2
SEND_ACTIONS: tuple[int, ...]
SEND_FRACTIONS: tuple[float, ...]
discrete_action_index: Callable[[int, int, int], int]
encoded_from_feat: Callable[..., Any]
alive_players_from_obs: Callable[[dict[str, Any]], int]
rust_encode_obs: Callable[[dict[str, Any], int], dict[str, Any]]
_RUNTIME_DEPS_LOADED = False


def load_runtime_deps() -> None:
    global SEND_ACTIONS, SEND_FRACTIONS, alive_players_from_obs, discrete_action_index, encoded_from_feat, rust_encode_obs
    global _RUNTIME_DEPS_LOADED
    if _RUNTIME_DEPS_LOADED:
        return
    try:
        from features import (  # noqa: E402
            ACTION_DIM as feature_action_dim,
            SEND_ACTIONS as feature_send_actions,
            SEND_FRACTIONS as feature_send_fractions,
            alive_players_from_obs as feature_alive_players_from_obs,
            discrete_action_index as feature_discrete_action_index,
            encoded_from_feat as feature_encoded_from_feat,
        )
        from orbit_wars_model import encode_obs as feature_rust_encode_obs  # noqa: E402
    except ModuleNotFoundError as exc:
        if exc.name == "orbit_wars_model":
            raise ModuleNotFoundError(
                "orbit_wars_model is required to encode replay observations. "
                "Build and install experimental_arch/env_model first."
            ) from exc
        raise

    if int(feature_action_dim) != ACTION_DIM:
        raise RuntimeError(f"feature ACTION_DIM={feature_action_dim}, expected {ACTION_DIM}")
    SEND_ACTIONS = tuple(int(x) for x in feature_send_actions)
    SEND_FRACTIONS = tuple(float(x) for x in feature_send_fractions)
    alive_players_from_obs = feature_alive_players_from_obs
    discrete_action_index = feature_discrete_action_index
    encoded_from_feat = feature_encoded_from_feat
    rust_encode_obs = feature_rust_encode_obs
    _RUNTIME_DEPS_LOADED = True


@dataclass(frozen=True)
class BuildConfig:
    zip_paths: tuple[str, ...]
    players: int
    keep_players: str | None
    player_names: tuple[str, ...]
    winner_only: bool
    launch_only: bool
    min_alive_players: int
    row_half_life_days: float
    reference_day: str
    max_angle_error: float
    send_fraction_tolerance: float
    chunk_rows: int
    limit_games_per_zip: int | None
    max_rows: int | None
    out_dir: str


@dataclass(frozen=True)
class ReplayMeta:
    names: tuple[str, ...]
    rewards: tuple[float, ...]


@dataclass(frozen=True)
class Sample:
    encoded: Any
    label: int
    step: int
    player: int
    player_rank: int
    opponent_rank: int
    our_ship_fraction: float
    age_weight: float
    day: str
    game_id: str


def normalize_obs(obs: dict[str, Any], *, player: int, step: int) -> dict[str, Any]:
    return {
        "player": int(player),
        "step": int(obs.get("step", step) or 0),
        "planets": list(obs.get("planets", []) or []),
        "fleets": list(obs.get("fleets", []) or []),
        "angular_velocity": float(obs.get("angular_velocity", 0.0) or 0.0),
        "initial_planets": list(obs.get("initial_planets", []) or []),
        "comets": list(obs.get("comets", []) or []),
        "comet_planet_ids": list(obs.get("comet_planet_ids", []) or []),
    }


def replay_meta(replay: dict[str, Any], players: int) -> ReplayMeta | None:
    rewards = replay.get("rewards") or []
    if len(rewards) != players or any(r is None for r in rewards[:players]):
        return None
    agents = (replay.get("info") or {}).get("Agents") or []
    if len(agents) != players:
        return None
    names = tuple(str(agent.get("Name", f"p{i}")) for i, agent in enumerate(agents[:players]))
    return ReplayMeta(names=names, rewards=tuple(float(r) for r in rewards[:players]))


def replay_meta_from_head(head: bytes, players: int) -> ReplayMeta | None:
    steps_match = _STEPS_RE.search(head)
    if steps_match is not None:
        head = head[: steps_match.start()]
    names: list[str] = []
    for raw in _NAME_RE.findall(head):
        try:
            names.append(str(json.loads(b'"' + raw + b'"')))
        except Exception:
            names.append(raw.decode("utf-8", "replace"))
    rewards_match = _REWARDS_RE.search(head)
    if len(names) != players or rewards_match is None:
        return None
    try:
        rewards = tuple(
            float(x.strip())
            for x in rewards_match.group(1).decode("ascii", "replace").split(",")
            if x.strip()
        )
    except ValueError:
        return None
    if len(rewards) != players or any(r is None for r in rewards[:players]):
        return None
    return ReplayMeta(names=tuple(names[:players]), rewards=tuple(float(r) for r in rewards[:players]))


def read_replay_meta(zf: zipfile.ZipFile, entry_name: str, players: int) -> ReplayMeta | None:
    with zf.open(entry_name) as f:
        return replay_meta_from_head(f.read(REPLAY_HEAD_BYTES), players)


def selected_slots_for(
    meta: ReplayMeta,
    args: argparse.Namespace,
    keep_set: set[str] | None,
    daily_ranks: dict[str, dict[str, int]] | None,
    day: str,
) -> tuple[int, ...] | None:
    winner = unique_winner(meta.rewards)
    if args.winner_only and winner is None:
        return None
    slots: list[int] = []
    for slot in range(args.players):
        if args.winner_only and slot != winner:
            continue
        if not player_kept_for(meta.names[slot], day, keep_set, daily_ranks):
            continue
        slots.append(slot)
    return tuple(slots)


def unique_winner(rewards: tuple[float, ...]) -> int | None:
    best = max(rewards)
    winners = [i for i, reward in enumerate(rewards) if reward == best]
    return winners[0] if len(winners) == 1 else None


def angle_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def source_ship_count(obs: dict[str, Any], source_id: int) -> float | None:
    for planet in obs.get("planets", []) or []:
        if int(planet[0]) == source_id:
            ships = float(planet[5])
            return ships if ships > 0.0 else None
    return None


def nearest_send_bin(frac: float | None, tolerance: float) -> int | None:
    if frac is None or not math.isfinite(frac) or frac <= 0.0:
        return None
    best_bin = min(range(len(SEND_FRACTIONS)), key=lambda i: abs(float(SEND_FRACTIONS[i]) - frac))
    if abs(float(SEND_FRACTIONS[best_bin]) - frac) > tolerance:
        return None
    return int(best_bin)


def encode_replay_move(
    obs: dict[str, Any],
    move: list[Any],
    feat: dict[str, Any],
    send_bin: int,
    max_angle_error: float,
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

    if send_bin < 0 or send_bin >= len(SEND_ACTIONS):
        return None, "bad_send_bin"
    raw_action = SEND_ACTIONS[send_bin]

    best_label: int | None = None
    best_delta = float("inf")
    for target_slot in range(len(planet_ids)):
        raw_idx = (source_slot * len(planet_ids) + target_slot) * ACTIONS_DIM + raw_action
        if raw_idx >= len(feat["mask"]) or not bool(feat["mask"][raw_idx]):
            continue
        delta = angle_delta(float(feat["angles"][raw_idx]), move_angle)
        if delta < best_delta:
            best_delta = delta
            best_label = discrete_action_index(source_slot, target_slot, send_bin)

    if best_label is None:
        return None, "no_valid_target"
    if best_delta > max_angle_error:
        return None, f"angle_miss_{best_delta:.4f}"
    return best_label, "ok"


def player_ship_fraction(obs: dict[str, Any], player: int) -> float:
    totals: Counter[int] = Counter()
    for planet in obs.get("planets", []) or []:
        owner = int(planet[1])
        if owner >= 0:
            totals[owner] += int(planet[5])
    for fleet in obs.get("fleets", []) or []:
        owner = int(fleet[1])
        if owner >= 0:
            totals[owner] += int(fleet[6])
    denom = sum(totals.values())
    return float(totals.get(player, 0) / denom) if denom > 0 else 0.0


def opponent_rank_for(names: tuple[str, ...], player: int, ranks: dict[str, int]) -> int:
    vals = [ranks.get(name, 30) for i, name in enumerate(names) if i != player]
    return min(vals) if vals else 30


def encode_sample(
    obs: dict[str, Any],
    action: list[Any],
    player: int,
    player_rank: int,
    opponent_rank: int,
    day: str,
    game_id: str,
    min_alive_players: int,
    age_weight: float,
    max_angle_error: float,
    send_fraction_tolerance: float,
    launch_only: bool,
) -> tuple[list[Sample], Counter[str]]:
    stats: Counter[str] = Counter()
    alive_count = alive_players_from_obs(obs)
    stats[f"alive_{alive_count}_states"] += 1
    if alive_count < min_alive_players:
        stats["skipped_alive_filter"] += 1
        return [], stats
    feat = rust_encode_obs(obs, player)
    encoded = encoded_from_feat(feat, obs=obs)
    rows: list[Sample] = []
    step = int(obs.get("step", 0) or 0)
    ship_fraction = player_ship_fraction(obs, player)

    if not action:
        if launch_only:
            stats["skipped_noop_rows"] += 1
            return rows, stats
        rows.append(
            Sample(
                encoded=encoded,
                label=0,
                step=step,
                player=player,
                player_rank=player_rank,
                opponent_rank=opponent_rank,
                our_ship_fraction=ship_fraction,
                age_weight=age_weight,
                day=day,
                game_id=game_id,
            )
        )
        stats["noop_rows"] += 1
        return rows, stats

    if len(action) > 1:
        stats["multi_launch_steps"] += 1
    for move in action:
        if len(move) < 3:
            stats["skipped_invalid"] += 1
            continue
        ships = float(move[2])
        source_ships = source_ship_count(obs, int(move[0]))
        frac = ships / source_ships if source_ships else None
        send_bin = nearest_send_bin(frac, send_fraction_tolerance)
        if send_bin is None:
            stats["skipped_bad_ship_fraction"] += 1
            continue
        label, reason = encode_replay_move(obs, move, feat, send_bin, max_angle_error)
        if label is None:
            stats[f"skipped_{reason.split('_')[0]}"] += 1
            stats["skipped_invalid"] += 1
            continue
        if label < 0 or label >= len(encoded.action_mask) or not bool(encoded.action_mask[label]):
            stats["skipped_invalid"] += 1
            continue
        rows.append(
            Sample(
                encoded=encoded,
                label=int(label),
                step=step,
                player=player,
                player_rank=player_rank,
                opponent_rank=opponent_rank,
                our_ship_fraction=ship_fraction,
                age_weight=age_weight,
                day=day,
                game_id=game_id,
            )
        )
        stats["launch_rows"] += 1
    return rows, stats


def day_from_zip(path: Path) -> str:
    match = re.search(r"replays_(\d+_\d+)", path.stem)
    return match.group(1) if match else path.stem


def day_ordinal(day: str) -> int:
    month_s, day_s = day.split("_", 1)
    return date(2026, int(month_s), int(day_s)).toordinal()


def age_weight_for(day: str, reference_day: str, half_life_days: float) -> float:
    if half_life_days <= 0.0:
        return 1.0
    age_days = max(0, day_ordinal(reference_day) - day_ordinal(day))
    return float(0.5 ** (age_days / half_life_days))


def load_player_filter(
    path: Path | None,
    player_names: list[str],
) -> tuple[set[str] | None, dict[str, int], dict[str, dict[str, int]] | None]:
    names: list[str] = []
    daily_ranks: dict[str, dict[str, int]] | None = None
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            names.extend(str(x) for x in raw)
        elif isinstance(raw, dict):
            daily_ranks = {}
            for day, day_names in raw.items():
                if not isinstance(day_names, list):
                    raise ValueError(f"--keep-players day {day!r} must map to a JSON list: {path}")
                daily_ranks[str(day)] = {str(name): i + 1 for i, name in enumerate(day_names)}
        else:
            raise ValueError(f"--keep-players must be a JSON list or day->list object: {path}")
    names.extend(str(x) for x in player_names)
    names = list(dict.fromkeys(names))
    keep_set = set(names) if names else None
    ranks = {name: i + 1 for i, name in enumerate(names)}
    return keep_set, ranks, daily_ranks


def player_rank_for(name: str, day: str, ranks: dict[str, int], daily_ranks: dict[str, dict[str, int]] | None) -> int:
    if daily_ranks is not None:
        return daily_ranks.get(day, {}).get(name, 30)
    return ranks.get(name, 1)


def player_kept_for(
    name: str,
    day: str,
    keep_set: set[str] | None,
    daily_ranks: dict[str, dict[str, int]] | None,
) -> bool:
    if daily_ranks is not None:
        return name in daily_ranks.get(day, {})
    if keep_set is None:
        return True
    return name in keep_set


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "player"


class ChunkWriter:
    def __init__(self, out_dir: Path, chunk_rows: int) -> None:
        self.out_dir = out_dir
        self.chunks_dir = out_dir / "chunks"
        self.chunk_rows = int(chunk_rows)
        self.pending: list[Sample] = []
        self.chunk_index = 0
        self.manifest_chunks: list[dict[str, Any]] = []

    def add(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self.pending.append(sample)
            if len(self.pending) >= self.chunk_rows:
                self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        samples = self.pending
        self.pending = []
        chunk_name = f"chunk_{self.chunk_index:06d}.npz"
        self.chunk_index += 1
        final_path = self.chunks_dir / chunk_name
        tmp_path = self.chunks_dir / f".{chunk_name}.tmp.npz"

        encoded_rows = [sample.encoded for sample in samples]
        tokens = np.stack([encoded.tokens for encoded in encoded_rows]).astype(np.float16)
        presence = np.stack([encoded.presence for encoded in encoded_rows]).astype(np.bool_)
        globals_ = np.stack([encoded.globals for encoded in encoded_rows]).astype(np.float32)
        action_mask = np.stack([encoded.action_mask for encoded in encoded_rows]).astype(np.bool_)
        pair_turns = np.stack([encoded.pair_turns for encoded in encoded_rows])
        pair_turns_u8 = np.rint(pair_turns * 20.0).clip(0, 255).astype(np.uint8)
        pair_reachable = np.stack([encoded.pair_reachable_mask for encoded in encoded_rows]).astype(np.bool_)
        pair_outcome = np.stack([encoded.pair_outcome_features for encoded in encoded_rows]).astype(np.float16)
        timeline = np.stack([encoded.planet_timeline_features for encoded in encoded_rows]).astype(np.float16)
        owner_ids = np.stack([encoded.owner_ids for encoded in encoded_rows]).astype(np.int8)

        labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
        steps = np.asarray([sample.step for sample in samples], dtype=np.int64)
        players = np.asarray([sample.player for sample in samples], dtype=np.int64)
        player_ids = np.asarray([sample.encoded.player_id for sample in samples], dtype=np.int64)
        alive_players = np.asarray([sample.encoded.alive_players for sample in samples], dtype=np.int64)
        player_rank = np.asarray([sample.player_rank for sample in samples], dtype=np.int64)
        opponent_rank = np.asarray([sample.opponent_rank for sample in samples], dtype=np.int64)
        our_ship_fraction = np.asarray([sample.our_ship_fraction for sample in samples], dtype=np.float32)
        age_weight = np.asarray([sample.age_weight for sample in samples], dtype=np.float32)
        games = Counter((sample.day, sample.game_id) for sample in samples)
        games_json = json.dumps(
            [
                {"day": day, "game_id": game_id, "rows": int(rows)}
                for (day, game_id), rows in sorted(games.items())
            ],
            sort_keys=True,
        )

        np.savez_compressed(
            tmp_path,
            format_version=np.asarray([DATASET_FORMAT_VERSION], dtype=np.int32),
            action_dim=np.asarray([ACTION_DIM], dtype=np.int32),
            send_fractions=np.asarray(SEND_FRACTIONS, dtype=np.float32),
            tokens=tokens,
            presence=presence,
            globals_=globals_,
            action_mask=action_mask,
            pair_turns=pair_turns_u8.reshape((-1, *PAIR_TURN_SHAPE)),
            pair_reachable_mask=pair_reachable.reshape((-1, *PAIR_TURN_SHAPE)),
            pair_outcome_features=pair_outcome.reshape((-1, *PAIR_OUTCOME_SHAPE)),
            planet_timeline_features=timeline.reshape((-1, *PLANET_TIMELINE_SHAPE)),
            owner_ids=owner_ids,
            labels=labels,
            steps=steps,
            players=players,
            player_ids=player_ids,
            alive_players=alive_players,
            player_rank=player_rank,
            opponent_rank=opponent_rank,
            our_ship_fraction=our_ship_fraction,
            age_weight=age_weight,
            games_json=np.asarray(games_json),
        )
        tmp_path.replace(final_path)
        rel = final_path.relative_to(self.out_dir)
        self.manifest_chunks.append(
            {
                "path": str(rel).replace("\\", "/"),
                "rows": int(len(samples)),
                "weight_sum": float(
                    (
                        np.clip((1.0 - our_ship_fraction) ** 2 * (1.0 - player_rank / 30.0), 0.0, None)
                        * age_weight
                    ).sum()
                ),
                "alive_counts": {str(k): int(v) for k, v in sorted(Counter(alive_players.tolist()).items())},
            }
        )
        print(f"wrote {final_path} rows={len(samples):,}", flush=True)


def iter_zip_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        out.extend(matches if matches else [Path(pattern)])
    paths = sorted({p.resolve() for p in out}, key=lambda p: p.name)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("zip path(s) not found:\n  " + "\n  ".join(missing))
    return paths


def build(args: argparse.Namespace) -> dict[str, Any]:
    load_runtime_deps()
    zip_paths = iter_zip_paths(args.zip)
    reference_day = args.reference_day or max((day_from_zip(path) for path in zip_paths), key=day_ordinal)
    out_dir = args.out_dir.resolve()
    chunks_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    keep_set, ranks, daily_ranks = load_player_filter(args.keep_players, args.player_name)

    cfg = BuildConfig(
        zip_paths=tuple(str(p) for p in zip_paths),
        players=args.players,
        keep_players=str(args.keep_players) if args.keep_players else None,
        player_names=tuple(args.player_name),
        winner_only=bool(args.winner_only),
        launch_only=bool(args.launch_only),
        min_alive_players=int(args.min_alive_players),
        row_half_life_days=float(args.row_half_life_days),
        reference_day=str(reference_day),
        max_angle_error=float(args.max_angle_error),
        send_fraction_tolerance=float(args.send_fraction_tolerance),
        chunk_rows=int(args.chunk_rows),
        limit_games_per_zip=args.limit_games_per_zip,
        max_rows=args.max_rows,
        out_dir=str(out_dir),
    )
    cfg_path = out_dir / "build_config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    writer = ChunkWriter(out_dir, args.chunk_rows)
    stats: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    t0 = time.time()

    for zi, zip_path in enumerate(zip_paths, start=1):
        day = day_from_zip(zip_path)
        age_weight = age_weight_for(day, reference_day, args.row_half_life_days)
        with zipfile.ZipFile(zip_path) as zf:
            entries = sorted(n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/"))
            if args.limit_games_per_zip is not None:
                entries = entries[: args.limit_games_per_zip]
            print(f">>> [{zi}/{len(zip_paths)}] {zip_path.name}: entries={len(entries):,} day={day}", flush=True)
            for ei, entry_name in enumerate(entries, start=1):
                stats["replays_seen"] += 1
                try:
                    meta = read_replay_meta(zf, entry_name, args.players)
                except Exception:
                    stats["skipped_parse"] += 1
                    continue
                if meta is None:
                    stats["skipped_format"] += 1
                    continue
                selected_slots = selected_slots_for(meta, args, keep_set, daily_ranks, day)
                if selected_slots is None:
                    stats["skipped_draw"] += 1
                    continue
                if not selected_slots:
                    stats["skipped_player_filter"] += 1
                    continue
                try:
                    replay = json.loads(zf.read(entry_name))
                except Exception:
                    stats["skipped_parse"] += 1
                    continue
                steps = replay.get("steps") or []
                if not steps:
                    stats["skipped_empty"] += 1
                    continue

                prev_obs: list[dict[str, Any] | None] = [None] * args.players
                replay_rows = 0
                game_id = Path(entry_name).stem
                for turn_index, turn in enumerate(steps):
                    if not isinstance(turn, list) or len(turn) < args.players:
                        continue
                    for slot in selected_slots:
                        entry_obj = turn[slot]
                        if not isinstance(entry_obj, dict):
                            continue
                        raw_obs = entry_obj.get("observation") or {}
                        if raw_obs.get("planets"):
                            current_obs = normalize_obs(raw_obs, player=slot, step=turn_index)
                        else:
                            current_obs = None
                        action = entry_obj.get("action") or []
                        obs = prev_obs[slot]
                        if obs is not None:
                            player_rank = player_rank_for(meta.names[slot], day, ranks, daily_ranks)
                            opponent_rank = opponent_rank_for(meta.names, slot, ranks)
                            try:
                                samples, row_stats = encode_sample(
                                    obs,
                                    action,
                                    slot,
                                    player_rank,
                                    opponent_rank,
                                    day,
                                    game_id,
                                    args.min_alive_players,
                                    age_weight,
                                    args.max_angle_error,
                                    args.send_fraction_tolerance,
                                    args.launch_only,
                                )
                            except Exception:
                                stats["skipped_encode_error"] += 1
                                samples = []
                                row_stats = Counter()
                            stats.update(row_stats)
                            if samples:
                                writer.add(samples)
                                replay_rows += len(samples)
                                stats["rows"] += len(samples)
                                for sample in samples:
                                    label_counts[sample.label] += 1
                                    stats[f"row_alive_{sample.encoded.alive_players}"] += 1
                                    stats[f"row_player_slot_{sample.player}"] += 1
                                if args.max_rows is not None and stats["rows"] >= args.max_rows:
                                    break
                        if current_obs is not None:
                            prev_obs[slot] = current_obs
                    if args.max_rows is not None and stats["rows"] >= args.max_rows:
                        break
                if replay_rows:
                    stats["replays_kept"] += 1
                if ei % 250 == 0:
                    print(
                        f"  {zip_path.name}: {ei:,}/{len(entries):,} entries, rows={stats['rows']:,}",
                        flush=True,
                    )
                if args.max_rows is not None and stats["rows"] >= args.max_rows:
                    break
        if args.max_rows is not None and stats["rows"] >= args.max_rows:
            break

    writer.flush()
    if not writer.manifest_chunks:
        raise SystemExit("no IL samples written")

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "action_dim": ACTION_DIM,
        "dataset_name": args.dataset_name,
        "players": args.players,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - t0,
        "config": asdict(cfg),
        "rows": int(stats["rows"]),
        "noop_rows": int(stats["noop_rows"]),
        "launch_rows": int(stats["launch_rows"]),
        "processed_replays": int(stats["replays_kept"]),
        "replays_seen": int(stats["replays_seen"]),
        "chunks": writer.manifest_chunks,
        "label_counts_top": {str(k): int(v) for k, v in label_counts.most_common(50)},
        "stats": {str(k): int(v) for k, v in sorted(stats.items())},
        "identity_features": {
            "owner_ids": "0=neutral_or_missing, 1..4=absolute_player_id_plus_one",
            "player_ids": "absolute acting player id",
            "alive_players": "live owners with ships on planets or fleets",
        },
        "feature_shapes_verified": {
            "TOKEN_SHAPE": list(TOKEN_SHAPE),
            "PAIR_TURN_SHAPE": list(PAIR_TURN_SHAPE),
            "PAIR_OUTCOME_SHAPE": list(PAIR_OUTCOME_SHAPE),
            "PLANET_TIMELINE_SHAPE": list(PLANET_TIMELINE_SHAPE),
        },
    }
    manifest_path = out_dir / "manifest.json"
    stats_path = out_dir / "dataset_stats.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stats_path.write_text(json.dumps({k: v for k, v in manifest.items() if k != "chunks"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {manifest_path}")
    print(f"rows={manifest['rows']:,} chunks={len(writer.manifest_chunks):,} replays={manifest['processed_replays']:,}")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chunked IL policy data from replay zips.")
    p.add_argument("--zip", required=True, nargs="+", help="replay zip path(s); globs are supported")
    p.add_argument("--players", type=int, choices=(2, 4), default=4)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dataset-name", default=None, help="name written into manifest; default derives from --players")
    p.add_argument("--keep-players", type=Path, default=None, help="JSON list from elo_topn.py; only these players' rows are encoded")
    p.add_argument("--player-name", action="append", default=[], help="exact player name to include; repeatable")
    p.add_argument("--winner-only", action="store_true", help="only imitate the unique game winner's actions")
    p.add_argument("--launch-only", action="store_true", help="drop noop rows and train only on launch actions")
    p.add_argument(
        "--min-alive-players",
        type=int,
        default=0,
        help="drop rows whose observation has fewer live players; use 3 for native 4p IL",
    )
    p.add_argument(
        "--row-half-life-days",
        type=float,
        default=0.0,
        help="multiply sample weights by 0.5 ** (days_old / half_life); 0 disables",
    )
    p.add_argument(
        "--reference-day",
        default=None,
        help="latest day for row age weighting, e.g. 6_17; default is latest provided zip",
    )
    p.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    p.add_argument("--limit-games-per-zip", type=int, default=None, help="debug cap per zip before format filtering")
    p.add_argument("--max-rows", type=int, default=None, help="stop after writing this many samples")
    p.add_argument("--max-angle-error", type=float, default=MAX_ANGLE_ERROR)
    p.add_argument("--send-fraction-tolerance", type=float, default=SEND_FRACTION_TOLERANCE)
    args = p.parse_args()
    if args.chunk_rows < 1:
        p.error("--chunk-rows must be >= 1")
    if args.min_alive_players < 0 or args.min_alive_players > args.players:
        p.error("--min-alive-players must be between 0 and --players")
    if args.row_half_life_days < 0.0:
        p.error("--row-half-life-days must be >= 0")
    if args.reference_day is not None:
        try:
            day_ordinal(args.reference_day)
        except Exception:
            p.error("--reference-day must look like M_D, e.g. 6_17")
    if args.dataset_name is None:
        if args.player_name:
            gate = "_".join(slugify_name(name) for name in args.player_name[:3])
            if len(args.player_name) > 3:
                gate += f"_plus{len(args.player_name) - 3}"
        else:
            gate = "kept" if args.keep_players else "all"
        win = "_winners" if args.winner_only else ""
        launches = "_launches" if args.launch_only else ""
        args.dataset_name = f"osteo_{gate}_{args.players}p{win}{launches}"
    return args


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
