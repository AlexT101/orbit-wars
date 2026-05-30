"""Kaggle reference engine wrapper.

Drives `kaggle_environments.envs.orbit_wars.orbit_wars.interpreter` directly,
bypassing the full env.run() loop. We replicate exactly the step-counter
bookkeeping that Kaggle's core.py applies after each interpreter call
(see core.py:602 — `step = len(self.steps)` post-call).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter

from engine_parity_checker.engine import JointAction, PlayerObs, Snapshot


DEFAULT_CONFIG = {
    "episodeSteps": 500,
    "actTimeout": 1,
    "shipSpeed": 6.0,
    "sunRadius": 10.0,
    "boardSize": 100.0,
    "cometSpeed": 4.0,
}


def _build_initial_state(num_players: int) -> list[SimpleNamespace]:
    """Match what Kaggle's runner passes on the very first interpreter call:
    obs0 has step=0 and nothing else; other agents have only `player` set."""
    state = []
    obs0 = SimpleNamespace(step=0)
    state.append(
        SimpleNamespace(observation=obs0, action=[], status="ACTIVE", reward=0)
    )
    for i in range(1, num_players):
        obs_i = SimpleNamespace(player=i)
        state.append(
            SimpleNamespace(observation=obs_i, action=[], status="ACTIVE", reward=0)
        )
    return state


class KaggleEngine:
    """Reference engine. Treats the Kaggle interpreter as ground truth."""

    def __init__(self):
        self._state: list[SimpleNamespace] | None = None
        self._env: SimpleNamespace | None = None
        self._step_count: int = 0  # mirrors len(self.steps) in core.py
        self._seed: int | None = None
        self._num_players: int = 0
        self._done: bool = False

    # ---- internal helpers -------------------------------------------------

    def _set_step(self) -> None:
        """Replicate core.py:602 — set obs0.step = len(steps) post-interpreter."""
        if self._done:
            self._state[0].observation.step = 0
        else:
            self._state[0].observation.step = self._step_count

    def _gather_obs(self) -> list[PlayerObs]:
        obs0 = self._state[0].observation
        out = []
        for i in range(self._num_players):
            out.append(
                PlayerObs(
                    player=i,
                    step=getattr(obs0, "step", 0),
                    angular_velocity=obs0.angular_velocity,
                    planets=copy.deepcopy(obs0.planets),
                    initial_planets=copy.deepcopy(obs0.initial_planets),
                    fleets=copy.deepcopy(obs0.fleets),
                    comets=copy.deepcopy(obs0.comets),
                    comet_planet_ids=list(obs0.comet_planet_ids),
                )
            )
        return out

    # ---- Engine protocol --------------------------------------------------

    def reset(
        self,
        seed: int,
        num_players: int,
        configuration: dict | None = None,
    ) -> list[PlayerObs]:
        assert num_players in (2, 4), f"orbit_wars supports 2 or 4 players, got {num_players}"
        self._seed = seed
        self._num_players = num_players
        self._step_count = 0
        self._done = False

        cfg = {**DEFAULT_CONFIG, **(configuration or {}), "seed": seed}
        self._env = SimpleNamespace(
            configuration=SimpleNamespace(**cfg),
            done=False,
            info={"seed": seed},  # pre-set so interpreter reuses it
        )
        self._state = _build_initial_state(num_players)

        # First interpreter call performs init only and returns immediately.
        # Kaggle's core.py:602 sets obs.step = 0 post-reset (because every
        # agent is INACTIVE so env.done is True at that moment), so the very
        # first env.step() call has the interpreter read obs.step=0.
        self._state = interpreter(self._state, self._env)
        self._step_count = 0
        self._set_step()
        return self._gather_obs()

    def step(self, actions: JointAction) -> tuple[list[PlayerObs], bool]:
        assert self._state is not None, "call reset() before step()"
        assert len(actions) == self._num_players, (
            f"need {self._num_players} action lists, got {len(actions)}"
        )

        for i, act in enumerate(actions):
            self._state[i].action = act if act is not None else []
            self._state[i].status = "ACTIVE"

        self._env.done = self._done
        self._state = interpreter(self._state, self._env)

        # Detect terminal — interpreter sets status=DONE on every agent.
        self._done = all(s.status == "DONE" for s in self._state)
        self._env.done = self._done

        self._step_count += 1
        self._set_step()
        return self._gather_obs(), self._done

    def snapshot(self) -> Snapshot:
        obs0 = self._state[0].observation
        rewards = None
        if self._done:
            rewards = [float(s.reward) for s in self._state]
        return Snapshot(
            step=getattr(obs0, "step", 0),
            angular_velocity=obs0.angular_velocity,
            planets=copy.deepcopy(obs0.planets),
            initial_planets=copy.deepcopy(obs0.initial_planets),
            fleets=copy.deepcopy(obs0.fleets),
            next_fleet_id=getattr(obs0, "next_fleet_id", 0),
            comet_planet_ids=list(obs0.comet_planet_ids),
            comets=copy.deepcopy(obs0.comets),
            done=self._done,
            rewards=rewards,
            seed=self._seed,
            info={"engine": "kaggle"},
        )
