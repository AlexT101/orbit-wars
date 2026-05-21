"""Rust candidate engine adapter.

Wraps the native PyO3 boundary exposed by `orbit_wars_rust.RustEngineCore`:

- `reset(seed, num_players, configuration) -> dict`
- `step(actions) -> dict`
- `snapshot() -> dict`
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from parity.engine import JointAction, PlayerObs, Snapshot


def _make_player_obs(raw: dict[str, Any]) -> PlayerObs:
    return PlayerObs(
        player=int(raw["player"]),
        step=int(raw["step"]),
        angular_velocity=float(raw["angular_velocity"]),
        planets=_normalize_rows(raw["planets"]),
        initial_planets=_normalize_rows(raw["initial_planets"]),
        fleets=_normalize_rows(raw["fleets"]),
        comets=_normalize_comets(raw["comets"]),
        comet_planet_ids=list(raw["comet_planet_ids"]),
    )


def _make_snapshot(raw: dict[str, Any]) -> Snapshot:
    return Snapshot(
        step=int(raw["step"]),
        angular_velocity=float(raw["angular_velocity"]),
        planets=_normalize_rows(raw["planets"]),
        initial_planets=_normalize_rows(raw["initial_planets"]),
        fleets=_normalize_rows(raw["fleets"]),
        next_fleet_id=int(raw["next_fleet_id"]),
        comet_planet_ids=list(raw["comet_planet_ids"]),
        comets=_normalize_comets(raw["comets"]),
        done=bool(raw["done"]),
        rewards=list(raw["rewards"]) if raw.get("rewards") is not None else None,
        seed=raw.get("seed"),
        info={"engine": "rust"},
    )


def _normalize_rows(rows: list[Any]) -> list[list[Any]]:
    return [list(row) for row in rows]


def _normalize_comets(comets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for comet in comets:
        normalized.append(
            {
                "planet_ids": list(comet["planet_ids"]),
                "path_index": int(comet["path_index"]),
                "paths": [[list(point) for point in path] for path in comet["paths"]],
            }
        )
    return normalized


def _load_native_module():
    module_name = "orbit_wars_rust"
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        repo_root = Path(__file__).resolve().parents[2]
        release_dir = repo_root / "rust_engine" / "target" / "release"
        dll_path = release_dir / "orbit_wars_rust.dll"
        load_candidates: list[Path] = []
        if dll_path.exists():
            temp_dir = Path(tempfile.gettempdir()) / "orbit_wars_rust"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_pyd = temp_dir / f"orbit_wars_rust_{os.getpid()}.pyd"
            shutil.copyfile(dll_path, temp_pyd)
            load_candidates.append(temp_pyd)
        else:
            pyd_path = release_dir / "orbit_wars_rust.pyd"
            if pyd_path.exists():
                load_candidates.append(pyd_path)

        for load_path in load_candidates:
            spec = importlib.util.spec_from_file_location(module_name, load_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        raise ImportError(
            "Could not import native module 'orbit_wars_rust'. Build the Rust "
            "crate first, e.g. `cargo build --release` in `rust_engine/`, or "
            "install it into the venv with `maturin develop`."
        ) from exc


class RustEngine:
    """Parity-harness adapter for the Rust engine."""

    def __init__(self) -> None:
        native = _load_native_module()
        self._core = native.RustEngineCore()
        self._last_snapshot: Snapshot | None = None

    def reset(
        self,
        seed: int,
        num_players: int,
        configuration: dict | None = None,
    ) -> list[PlayerObs]:
        payload = self._core.reset(int(seed), int(num_players), configuration or None)
        self._last_snapshot = _make_snapshot(payload["snapshot"])
        return [_make_player_obs(obs) for obs in payload["observations"]]

    def step(self, actions: JointAction) -> tuple[list[PlayerObs], bool]:
        payload = self._core.step(actions)
        self._last_snapshot = _make_snapshot(payload["snapshot"])
        return (
            [_make_player_obs(obs) for obs in payload["observations"]],
            bool(payload["done"]),
        )

    def snapshot(self) -> Snapshot:
        if self._last_snapshot is None:
            self._last_snapshot = _make_snapshot(self._core.snapshot())
        return self._last_snapshot
