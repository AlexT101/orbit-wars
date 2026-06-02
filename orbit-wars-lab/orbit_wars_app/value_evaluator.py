"""XGBoost value-function scoring for replay timelines.

The lab reuses the trojan_horse training feature path:

  observation JSON -> target/release/extract_v4 -> summary_v2[46]+extras_v4[12]
  -> engineered+tempo columns -> XGBoost Booster

This keeps the visualizer's value trace aligned with the bot's training
pipeline without copying the feature math into orbit-wars-lab.
"""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent if LAB_ROOT.name == "orbit-wars-lab" else LAB_ROOT
DEFAULT_MODEL_REL = Path("mine") / "trojan_horse" / "train" / "weights" / "xgb_46p12e88t11_latest.json"


def _candidate_zoo_roots() -> list[Path]:
    env_zoo = Path(os.environ.get("ORBIT_WARS_ZOO_DIR", "agents")).expanduser()
    raw_candidates = [
        env_zoo if env_zoo.is_absolute() else Path.cwd() / env_zoo,
        env_zoo if env_zoo.is_absolute() else LAB_ROOT / env_zoo,
        env_zoo if env_zoo.is_absolute() else REPO_ROOT / env_zoo,
        LAB_ROOT / "agents",
        REPO_ROOT / "bots",
        Path.cwd() / "agents",
        Path.cwd() / "bots",
        Path("/app/agents"),
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for cand in raw_candidates:
        resolved = cand.resolve() if cand.exists() else cand
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(cand)
    return out


def _agent_tree_suffix(path: Path) -> Path | None:
    parts = path.parts
    for marker in ("bots", "agents"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return Path(*parts[idx + 1 :])
    return None


def _resolve_agent_tree_path(path: Path) -> Path | None:
    suffix = _agent_tree_suffix(path)
    if suffix is None:
        return None
    for zoo_root in _candidate_zoo_roots():
        cand = zoo_root / suffix
        if cand.exists():
            return cand
    return None


def _default_trojan_horse_dir() -> Path:
    suffix = Path("mine") / "trojan_horse"
    for zoo_root in _candidate_zoo_roots():
        cand = zoo_root / suffix
        if cand.is_dir():
            return cand
    return REPO_ROOT / "bots" / suffix


TROJAN_HORSE_DIR = _default_trojan_horse_dir()
TROJAN_HORSE_TRAIN = TROJAN_HORSE_DIR / "train"
DEFAULT_VALUE_MODEL = TROJAN_HORSE_TRAIN / "weights" / "xgb_46p12e88t11_latest.json"

SUMMARY_DIM = 46
EXTRAS_DIM = 12
BASE_DIM = SUMMARY_DIM + EXTRAS_DIM
RECORD_BYTES = 8 + 4 + 4 * SUMMARY_DIM + 4 * EXTRAS_DIM


def default_value_model_path() -> Path:
    return DEFAULT_VALUE_MODEL


def resolve_value_model_path(path: str | os.PathLike[str] | None) -> Path:
    if path is None or str(path).strip() == "":
        return DEFAULT_VALUE_MODEL
    raw = Path(str(path).strip()).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw
        translated = _resolve_agent_tree_path(raw)
        if translated is not None:
            return translated
        return raw
    translated = _resolve_agent_tree_path(raw)
    if translated is not None:
        return translated
    for base in (REPO_ROOT, LAB_ROOT, Path.cwd()):
        cand = base / raw
        if cand.exists():
            return cand
    return REPO_ROOT / raw


def validate_value_model_path(path: str | os.PathLike[str] | None) -> Path:
    resolved = resolve_value_model_path(path)
    if not resolved.is_file():
        raise ValueError(f"value model not found: {resolved}")
    return resolved


def _extractor_path() -> Path:
    path = TROJAN_HORSE_DIR / "target" / "release" / "extract_v4"
    if path.is_file():
        return path
    cargo = os.environ.get("CARGO") or "cargo"
    try:
        subprocess.check_call(
            [cargo, "build", "--release", "--bin", "extract_v4"],
            cwd=TROJAN_HORSE_DIR,
        )
    except Exception as exc:
        raise RuntimeError(
            f"extract_v4 is missing and could not be built at {path}: {exc}"
        ) from exc
    if not path.is_file():
        raise RuntimeError(f"extract_v4 build completed but binary is missing: {path}")
    return path


def _load_engineered_module():
    module_path = TROJAN_HORSE_TRAIN / "engineered_features.py"
    spec = importlib.util.spec_from_file_location(
        "trojan_horse_engineered_features", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load engineered feature module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _normalize_obs(obs: dict[str, Any], player: int) -> dict[str, Any]:
    return {
        "player": int(player),
        "step": int(obs.get("step", 0)),
        "planets": list(obs.get("planets", []) or []),
        "fleets": list(obs.get("fleets", []) or []),
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        "initial_planets": list(obs.get("initial_planets", []) or []),
        "comets": list(obs.get("comets", []) or []),
        "comet_planet_ids": list(obs.get("comet_planet_ids", []) or []),
    }


def _replay_observations(
    replay: dict[str, Any],
    num_agents: int,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    steps = replay.get("steps") or []
    meta: list[tuple[int, int]] = []
    observations: list[dict[str, Any]] = []
    for step_idx, step in enumerate(steps):
        if not isinstance(step, list):
            continue
        for player in range(num_agents):
            entry = step[player] if player < len(step) else None
            if not isinstance(entry, dict):
                continue
            obs = entry.get("observation")
            if not isinstance(obs, dict) or not obs.get("planets"):
                continue
            norm = _normalize_obs(obs, player)
            norm["step"] = int(norm.get("step", step_idx))
            meta.append((step_idx, player))
            observations.append(norm)
    return meta, observations


def _extract_base_features(observations: list[dict[str, Any]]) -> np.ndarray:
    if not observations:
        return np.zeros((0, BASE_DIM), dtype=np.float32)

    payload = "".join(
        json.dumps(obs, separators=(",", ":")) + "\n" for obs in observations
    ).encode("utf-8")
    proc = subprocess.run(
        [str(_extractor_path())],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=TROJAN_HORSE_DIR,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"extract_v4 failed: {stderr or proc.returncode}")

    count = len(proc.stdout) // RECORD_BYTES
    if count != len(observations):
        raise RuntimeError(
            f"extract_v4 returned {count} records for {len(observations)} observations"
        )
    out = np.zeros((count, BASE_DIM), dtype=np.float32)
    for i in range(count):
        chunk = proc.stdout[i * RECORD_BYTES : (i + 1) * RECORD_BYTES]
        # First 12 bytes are step:i64, player:i32. The rest is 58 float32s.
        _step, _player = struct.unpack_from("<qi", chunk, 0)
        out[i] = np.frombuffer(chunk, dtype=np.float32, count=BASE_DIM, offset=12)
    return out


def attach_value_predictions(
    replay: dict[str, Any],
    *,
    model_path: str | os.PathLike[str] | None,
    agent_ids: list[str],
) -> dict[str, Any]:
    """Mutate and return replay with `value_function` prediction traces."""
    resolved_model = validate_value_model_path(model_path)
    num_agents = len(agent_ids)
    if num_agents <= 0:
        return replay

    meta_pairs, observations = _replay_observations(replay, num_agents)
    total_steps = len(replay.get("steps") or [])
    values = [[None for _ in range(total_steps)] for _ in range(num_agents)]
    if observations:
        base = _extract_base_features(observations)
        engineered = _load_engineered_module()
        core = engineered.append_engineered_features(base)
        # meta columns are (game_id, step, slot). There is only one game in
        # this replay, and `slot` is the player perspective.
        tempo_meta = np.array(
            [[0, step_idx, player] for step_idx, player in meta_pairs],
            dtype=np.int32,
        )
        features = engineered.append_tempo_features(core, tempo_meta)

        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is not installed in the lab Python environment"
            ) from exc
        booster = xgb.Booster()
        booster.load_model(str(resolved_model))
        dmatrix = xgb.DMatrix(features)
        preds = booster.predict(dmatrix)
        for (step_idx, player), pred in zip(meta_pairs, preds):
            if 0 <= player < num_agents and 0 <= step_idx < total_steps:
                values[player][step_idx] = round(float(pred), 6)

    replay["value_function"] = {
        "model_path": str(resolved_model),
        "agent_ids": list(agent_ids),
        "kind": "xgboost_binary_logistic",
        "label": "P(win from player perspective)",
        "values": values,
    }
    return replay
