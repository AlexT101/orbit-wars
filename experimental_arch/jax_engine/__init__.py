"""JAX Orbit Wars engine."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .engine import (
    BOARD_SIZE,
    CENTER,
    MAX_PLAYERS,
    COMET_SPAWN_STEPS,
    Configuration,
    JaxOrbitWarsEngine,
    Limits,
    RewardWeights,
    State,
    actions_to_jax,
    batch_reset,
    jit_step,
    reset,
    snapshot_from_state,
    step,
)

__all__ = [
    "BOARD_SIZE",
    "CENTER",
    "MAX_PLAYERS",
    "COMET_SPAWN_STEPS",
    "Configuration",
    "JaxOrbitWarsEngine",
    "Limits",
    "RewardWeights",
    "State",
    "actions_to_jax",
    "batch_reset",
    "jit_step",
    "reset",
    "snapshot_from_state",
    "step",
]
