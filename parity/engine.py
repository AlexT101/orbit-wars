"""Engine abstract interface and canonical state types.

A `Snapshot` is the full, engine-agnostic representation of game state that
gets diffed between the reference (Kaggle) and candidate (Rust/port) engines.
Anything that affects future state lives here; player-specific views do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# Per-player action: list of [from_planet_id, angle_radians, num_ships]
PlayerAction = list[list[Any]]
# Joint action across all players, indexed by player id
JointAction = list[PlayerAction]


@dataclass
class Snapshot:
    """Canonical, engine-agnostic state. Sortable and float-comparable.

    Lists are stored in their natural engine order (planet/fleet creation
    order) since combat resolution depends on iteration order. The diff
    layer compares element-by-element with that invariant in mind.
    """

    step: int
    angular_velocity: float
    # [id, owner, x, y, radius, ships, production]
    planets: list[list[float]]
    initial_planets: list[list[float]]
    # [id, owner, x, y, angle, from_planet_id, ships]
    fleets: list[list[float]]
    next_fleet_id: int
    comet_planet_ids: list[int]
    # [{"planet_ids": [...], "path_index": int, "paths": [[[x, y], ...], ...]}]
    comets: list[dict]
    done: bool
    rewards: list[float] | None = None
    seed: int | None = None
    # Tags for human-readable diffs; not compared
    info: dict = field(default_factory=dict)


@dataclass
class PlayerObs:
    """Per-player observation as agents see it. Mirrors Kaggle's obs schema."""

    player: int
    step: int
    angular_velocity: float
    planets: list[list[float]]
    initial_planets: list[list[float]]
    fleets: list[list[float]]
    comets: list[dict]
    comet_planet_ids: list[int]

    def as_dict(self) -> dict:
        return {
            "player": self.player,
            "step": self.step,
            "angular_velocity": self.angular_velocity,
            "planets": self.planets,
            "initial_planets": self.initial_planets,
            "fleets": self.fleets,
            "comets": self.comets,
            "comet_planet_ids": self.comet_planet_ids,
        }


class Engine(Protocol):
    """Common interface every parity-testable engine must implement."""

    def reset(
        self,
        seed: int,
        num_players: int,
        configuration: dict | None = None,
    ) -> list[PlayerObs]:
        """Initialize a new episode. Returns one observation per player."""

    def step(self, actions: JointAction) -> tuple[list[PlayerObs], bool]:
        """Advance one tick. Returns (per-player obs, done)."""

    def snapshot(self) -> Snapshot:
        """Return the full canonical state for diffing."""
