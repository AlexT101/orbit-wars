"""JAX port of the Kaggle Orbit Wars engine.

Designed for batched RL self-play (1024+ games in parallel on GPU). The
init path runs on CPU using Python's `random.Random` (matches Kaggle's
MT19937 bit-exactly); the per-step physics/combat is a JIT-compiled,
vmap-able JAX function.
"""

from .state import (
    BatchState,
    MAX_PLANETS,
    MAX_FLEETS,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    MAX_ACTIONS_PER_PLAYER,
    NUM_PLAYERS_PAD,
)
from .init import init_batch, init_single
from .engine import JaxEngine
from .action import ActionBatch, encode_actions, encode_action_batch
from .step import step_batch, step_single

__all__ = [
    "BatchState",
    "MAX_PLANETS",
    "MAX_FLEETS",
    "MAX_COMET_GROUPS",
    "MAX_COMET_PATH_LEN",
    "MAX_ACTIONS_PER_PLAYER",
    "NUM_PLAYERS_PAD",
    "ActionBatch",
    "encode_actions",
    "encode_action_batch",
    "init_batch",
    "init_single",
    "step_batch",
    "step_single",
    "JaxEngine",
]
