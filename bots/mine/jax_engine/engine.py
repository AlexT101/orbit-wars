"""Engine adapter exposing the JAX/numpy engine through the
`engine_parity_checker.engine.Engine` protocol. The numpy step path is
the parity oracle; once it matches Kaggle bit-exactly we'll swap in the
JIT-compiled batched step.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .init import init_single
from .state import (
    F_ANGLE,
    F_FROM,
    F_ID,
    F_OWNER,
    F_SHIPS,
    F_X,
    F_Y,
    MAX_COMET_GROUPS,
    P_ID,
    P_OWNER,
    P_PROD,
    P_R,
    P_SHIPS,
    P_X,
    P_Y,
)
from .step_numpy import step_numpy


def _row_planet(row, mask) -> list | None:
    if not mask:
        return None
    return [
        int(row[P_ID]),
        int(row[P_OWNER]),
        float(row[P_X]),
        float(row[P_Y]),
        float(row[P_R]),
        int(row[P_SHIPS]),
        int(row[P_PROD]),
    ]


def _row_fleet(row, mask) -> list | None:
    if not mask:
        return None
    return [
        int(row[F_ID]),
        int(row[F_OWNER]),
        float(row[F_X]),
        float(row[F_Y]),
        float(row[F_ANGLE]),
        int(row[F_FROM]),
        int(row[F_SHIPS]),
    ]


def _planets_list(state):
    out = []
    for i in range(state["planets"].shape[0]):
        row = _row_planet(state["planets"][i], state["planet_mask"][i])
        if row is not None:
            out.append(row)
    return out


def _initial_planets_list(state):
    out = []
    for i in range(state["initial_planets"].shape[0]):
        row = _row_planet(state["initial_planets"][i], state["initial_mask"][i])
        if row is not None:
            out.append(row)
    return out


def _fleets_list(state):
    out = []
    for i in range(state["fleets"].shape[0]):
        row = _row_fleet(state["fleets"][i], state["fleet_mask"][i])
        if row is not None:
            out.append(row)
    # Kaggle's `obs0.fleets` is append-ordered; ids increase monotonically
    # with launch time. Our slot reuse breaks that ordering, so sort by id
    # at the serialization boundary to match the parity diff.
    out.sort(key=lambda f: f[0])
    return out


def _comets_list(state):
    out = []
    for gi in range(MAX_COMET_GROUPS):
        if not bool(state["comet_group_active"][gi]):
            continue
        planet_ids = [int(state["comet_planet_ids"][gi, qi]) for qi in range(4)]
        # Path lengths can vary per-quadrant in principle; in practice the
        # 4 symmetric copies share length. Emit each quadrant truncated to
        # its own length.
        paths = []
        for qi in range(4):
            L = int(state["comet_path_lens"][gi, qi])
            qpath = [
                [float(state["comet_paths"][gi, qi, k, 0]),
                 float(state["comet_paths"][gi, qi, k, 1])]
                for k in range(L)
            ]
            paths.append(qpath)
        out.append({
            "planet_ids": planet_ids,
            "path_index": int(state["comet_path_index"][gi]),
            "paths": paths,
        })
    return out


def _comet_planet_ids_list(state):
    out = []
    for gi in range(MAX_COMET_GROUPS):
        if not bool(state["comet_group_active"][gi]):
            continue
        for qi in range(4):
            out.append(int(state["comet_planet_ids"][gi, qi]))
    return out


class JaxEngine:
    """Parity-harness adapter for the JAX engine.

    `backend="numpy"` uses the bit-exact numpy reference step.
    `backend="jax"` uses the JIT-compiled, vmap-able JAX step. The JAX
    backend runs on whatever device JAX selects (CPU/GPU); enabling x64
    (`jax.config.update("jax_enable_x64", True)`) is required to keep
    parity at atol=0.
    """

    def __init__(self, configuration: dict | None = None,
                 backend: str = "numpy"):
        if backend not in ("numpy", "jax"):
            raise ValueError(f"unknown backend: {backend}")
        self._cfg = configuration or {}
        self._backend = backend
        self._state: dict[str, Any] | None = None  # numpy backend
        self._jax_state = None  # jax backend BatchState (no batch dim)
        self._seed: int | None = None
        self._num_players: int = 0
        self._snap_cache: dict[str, Any] | None = None

    def reset(self, seed: int, num_players: int, configuration: dict | None = None):
        cfg = {**self._cfg, **(configuration or {})}
        episode_steps = int(cfg.get("episodeSteps", 500))
        ship_speed = float(cfg.get("shipSpeed", 6.0))
        comet_speed = float(cfg.get("cometSpeed", 4.0))

        single = init_single(
            seed=seed,
            num_players=num_players,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
        )
        if self._backend == "numpy":
            self._state = single
            self._jax_state = None
        else:
            import jax.numpy as jnp
            from .state import BatchState
            self._jax_state = BatchState(**{k: jnp.asarray(v) for k, v in single.items()})
            self._state = None
        self._seed = seed
        self._num_players = num_players
        self._snap_cache = None
        return self._gather_obs()

    def step(self, actions):
        assert len(actions) == self._num_players
        padded = list(actions) + [[] for _ in range(4 - len(actions))]
        if self._backend == "numpy":
            assert self._state is not None
            self._state = step_numpy(self._state, padded)
            done = bool(self._state["done"])
        else:
            import jax.numpy as jnp
            from .action import encode_actions
            from .step import step_single
            moves, mask = encode_actions(padded, self._num_players)
            self._jax_state = step_single(
                self._jax_state, jnp.asarray(moves), jnp.asarray(mask)
            )
            done = bool(self._jax_state.done)
        self._snap_cache = None
        return self._gather_obs(), done

    def _current_dict(self):
        """Return a numpy-dict view of current state, regardless of backend."""
        if self._backend == "numpy":
            return self._state
        # JAX BatchState -> numpy dict.
        import numpy as np
        return {k: np.asarray(getattr(self._jax_state, k))
                for k in self._jax_state._fields}

    def snapshot(self):
        from engine_parity_checker.engine import Snapshot

        cur = self._current_dict()
        if self._snap_cache is None:
            self._snap_cache = {
                "planets": _planets_list(cur),
                "initial_planets": _initial_planets_list(cur),
                "fleets": _fleets_list(cur),
                "comets": _comets_list(cur),
                "comet_planet_ids": _comet_planet_ids_list(cur),
            }
        rewards = None
        done = bool(cur["done"])
        if done:
            rewards = [float(x) for x in cur["rewards"][: self._num_players]]
        # Kaggle's core.py resets obs.step to 0 once env.done is True
        # (see kaggle_engine.KaggleEngine._set_step). Mirror that so the
        # parity diff matches the post-terminal snapshot.
        step_out = 0 if done else int(cur["step"])
        return Snapshot(
            step=step_out,
            angular_velocity=float(cur["angular_velocity"]),
            planets=copy.deepcopy(self._snap_cache["planets"]),
            initial_planets=copy.deepcopy(self._snap_cache["initial_planets"]),
            fleets=copy.deepcopy(self._snap_cache["fleets"]),
            next_fleet_id=int(cur["next_fleet_id"]),
            comet_planet_ids=list(self._snap_cache["comet_planet_ids"]),
            comets=copy.deepcopy(self._snap_cache["comets"]),
            done=done,
            rewards=rewards,
            seed=self._seed,
            info={"engine": f"jax-{self._backend}"},
        )

    def _gather_obs(self):
        from engine_parity_checker.engine import PlayerObs

        cur = self._current_dict()
        if self._snap_cache is None:
            self._snap_cache = {
                "planets": _planets_list(cur),
                "initial_planets": _initial_planets_list(cur),
                "fleets": _fleets_list(cur),
                "comets": _comets_list(cur),
                "comet_planet_ids": _comet_planet_ids_list(cur),
            }
        out = []
        for i in range(self._num_players):
            out.append(PlayerObs(
                player=i,
                step=int(cur["step"]),
                angular_velocity=float(cur["angular_velocity"]),
                planets=copy.deepcopy(self._snap_cache["planets"]),
                initial_planets=copy.deepcopy(self._snap_cache["initial_planets"]),
                fleets=copy.deepcopy(self._snap_cache["fleets"]),
                comets=copy.deepcopy(self._snap_cache["comets"]),
                comet_planet_ids=list(self._snap_cache["comet_planet_ids"]),
            ))
        return out
