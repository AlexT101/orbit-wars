from __future__ import annotations

import contextlib
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable

@contextlib.contextmanager
def _silence_noisy_imports():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    logging.disable(logging.CRITICAL)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)
        logging.disable(logging.NOTSET)


with _silence_noisy_imports():
    from kaggle_environments import make

from .features import (
    MAX_STEPS,
    GameStats,
    decode_move,
    encode_obs,
    game_stats,
    resolve_via_env_rollout,
)
from .opponents import get_opponent


Opponent = Callable[[dict], list[list[float]]]


@dataclass
class StepResult:
    obs: dict
    reward: float
    done: bool
    info: dict


@dataclass(frozen=True)
class RewardWeights:
    """Minimal reward shaping.

    - terminal: ±1 on the final step from raw Kaggle reward.
    - terminal_time: small bonus for faster wins / extra penalty for faster losses.
    - production_delta: per-step shaping for own/enemy production change.
      Positive when we gain a producing planet, negative when the opponent does.
    - launch_penalty: tiny per-fleet-sent cost so the policy doesn't spam launches.
    """

    terminal_win: float = 1.0
    terminal_time: float = 0.10
    production_delta: float = 0.05
    launch_penalty: float = 0.001


def compute_reward(
    prev: GameStats,
    curr: GameStats,
    done: bool,
    raw_rewards: list[float] | None,
    weights: RewardWeights,
    num_launches: int = 0,
) -> tuple[float, dict[str, float]]:
    components = {
        "terminal": 0.0,
        "terminal_time": 0.0,
        "production_delta": 0.0,
        "launch_penalty": 0.0,
    }

    # Symmetric production shaping: + for our gains, - for their gains.
    own_dp = curr.own_production - prev.own_production
    enemy_dp = curr.enemy_production - prev.enemy_production
    components["production_delta"] = weights.production_delta * (own_dp - enemy_dp)

    # Per-launch cost.
    components["launch_penalty"] = -weights.launch_penalty * float(num_launches)

    if done and raw_rewards is not None and len(raw_rewards) >= 2:
        if raw_rewards[0] > raw_rewards[1]:
            outcome = weights.terminal_win
        elif raw_rewards[1] > raw_rewards[0]:
            outcome = -weights.terminal_win
        else:
            outcome = 0.0
        components["terminal"] = outcome
        if outcome != 0.0:
            remaining_frac = max(0.0, min(1.0, curr.remaining / MAX_STEPS))
            components["terminal_time"] = (
                weights.terminal_time * (1.0 if outcome > 0.0 else -1.0) * remaining_frac
            )

    reward = float(sum(components.values()))
    return reward, {k: float(v) for k, v in components.items()}


class OrbitWarsDuelEnv:
    """Two-player training wrapper around the Kaggle Orbit Wars environment."""

    def __init__(
        self,
        seed: int | None = None,
        opponent: str | Opponent = "nearest",
        reward_weights: RewardWeights | None = None,
    ) -> None:
        self.seed = seed
        self.reward_weights = reward_weights or RewardWeights()
        self.opponent = get_opponent(opponent)
        self.env = None
        self.last_stats: GameStats | None = None
        self.player = 0
        self.turn = 0
        self._resolved_cache: dict | None = None

    def _obs_for_player(self, player: int) -> dict:
        assert self.env is not None, "call reset() first"
        obs = dict(self.env.state[player].observation)
        # env.run injects the Kaggle step field before calling agents, but
        # direct env.step users only see the raw game observation. Some strong
        # bots, including hellburner, require obs["step"].
        obs.setdefault("step", self.turn)
        # Inject the per-step cached ground-truth resolution (single source
        # of truth shared by us and the opponent). encode_obs reads this if
        # present and skips its own rollout, so both sides see identical
        # `ships_resolved` features and `resolved+1` action legality.
        if self._resolved_cache is not None:
            obs["_resolved"] = self._resolved_cache
        return obs

    def _refresh_resolved_cache(self) -> None:
        """Compute and cache ground-truth resolved state once per env step."""
        # resolve_via_env_rollout deep-copies and steps the env forward; once
        # the env is done, stepping it raises FailedPrecondition. Skip — the
        # cached value isn't consumed past the terminal frame (the rollout
        # loop resets a fresh env before the next encode_obs).
        if self.env is not None and getattr(self.env, "done", False):
            return
        self._resolved_cache = resolve_via_env_rollout(self.env)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            self.seed = random.randint(1, 2**31 - 1)
        self.env = make("orbit_wars", configuration={"seed": int(self.seed)}, debug=False)
        self.env.reset(2)
        self.turn = 0
        self._refresh_resolved_cache()
        obs = self._obs_for_player(self.player)
        self.last_stats = game_stats(obs, self.player)
        return obs

    def encoded(self):
        # The obs returned by current_obs() carries `_resolved` set by
        # _refresh_resolved_cache, so encode_obs uses ground truth.
        return encode_obs(self.current_obs())

    def current_obs(self) -> dict:
        return self._obs_for_player(self.player)

    def step(self, action_index: int) -> StepResult:
        assert self.env is not None, "call reset() first"
        my_obs = self._obs_for_player(0)
        # `_resolved` is already on my_obs so decode_move's resolved+1 path
        # uses the same ground truth as encode_obs.
        return self.step_moves(decode_move(my_obs, action_index))

    def step_moves(self, my_moves: list[list[float]]) -> StepResult:
        assert self.env is not None, "call reset() first"
        opp_obs = self._obs_for_player(1)
        actions = [my_moves, self.opponent(opp_obs)]
        self.env.step(actions)
        self.turn += 1
        # State changed — invalidate, then recompute the per-step cache.
        self._refresh_resolved_cache()

        next_obs = self._obs_for_player(0)
        done = bool(self.env.done)
        raw_rewards = [float(s.reward or 0.0) for s in self.env.state]
        curr_stats = game_stats(next_obs, self.player)
        assert self.last_stats is not None
        reward, components = compute_reward(
            self.last_stats,
            curr_stats,
            done,
            raw_rewards,
            self.reward_weights,
            num_launches=len(my_moves),
        )
        self.last_stats = curr_stats
        return StepResult(
            obs=next_obs,
            reward=float(reward),
            done=done,
            info={
                "seed": self.seed,
                "raw_rewards": raw_rewards,
                "reward_components": components,
                "stats": curr_stats.__dict__,
            },
        )
