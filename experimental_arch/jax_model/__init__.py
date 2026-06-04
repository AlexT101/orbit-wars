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
from .features import (
    ACTIONS_DIM,
    GLOBAL_DIM,
    NUM_FRAMES,
    PLANET_SLOTS,
    TOKEN_DIM,
    FeatureConfig,
    Features,
    encode,
    jit_encode,
    jit_step_and_encode,
    step_and_encode,
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
    "ACTIONS_DIM",
    "GLOBAL_DIM",
    "NUM_FRAMES",
    "PLANET_SLOTS",
    "TOKEN_DIM",
    "FeatureConfig",
    "Features",
    "encode",
    "jit_encode",
    "jit_step_and_encode",
    "step_and_encode",
]
