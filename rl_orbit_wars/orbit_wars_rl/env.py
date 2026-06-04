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

from .features import MAX_STEPS, GameStats, decode_move, encode_obs, game_stats
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
    # Reward is intentionally not based on final margin. Terminal win/loss is
    # the main objective; dense reward is only for signs of life.
    mode: str = "terminal"

    # Dense per-step control signal. At maximum control this contributes about
    # +1 over a full 500-step game, so it cannot dominate win/loss.
    control: float = 0.002

    # Positive-only progress signals. Losing ships, margin, and opponent growth
    # are deliberately not punished here.
    score_increase: float = 0.0015
    production_increase: float = 0.04
    planet_increase: float = 0.08
    enemy_planet_capture_bonus: float = 0.06

    # Terminal objective.
    terminal_win: float = 2.0
    terminal_time: float = 0.25


def compute_reward(
    prev: GameStats,
    curr: GameStats,
    done: bool,
    raw_rewards: list[float] | None,
    weights: RewardWeights,
) -> tuple[float, dict[str, float]]:
    own_score_delta = max(0.0, curr.own_score - prev.own_score)
    production_delta = max(0.0, curr.own_production - prev.own_production)
    planet_delta = max(0.0, float(curr.own_planets - prev.own_planets))
    enemy_planet_captures = min(planet_delta, max(0.0, float(prev.enemy_planets - curr.enemy_planets)))
    components = {
        "control": 0.0,
        "score_increase": 0.0,
        "production_increase": 0.0,
        "planet_increase": 0.0,
        "enemy_planet_capture": 0.0,
        "terminal_time": 0.0,
    }

    if weights.mode in {"score_delta", "shaped"}:
        components["score_increase"] = weights.score_increase * own_score_delta

    if weights.mode == "shaped":
        control = (
            0.50 * curr.production_share
            + 0.30 * curr.planet_share
            + 0.20 * curr.score_share
        )
        components["control"] = weights.control * control
        components["production_increase"] = weights.production_increase * production_delta
        components["planet_increase"] = weights.planet_increase * planet_delta
        components["enemy_planet_capture"] = weights.enemy_planet_capture_bonus * enemy_planet_captures

    if done and raw_rewards is not None and len(raw_rewards) >= 2:
        if raw_rewards[0] > raw_rewards[1]:
            outcome = weights.terminal_win
        elif raw_rewards[1] > raw_rewards[0]:
            outcome = -weights.terminal_win
        else:
            outcome = 0.0
        if outcome:
            remaining_frac = max(0.0, min(1.0, curr.remaining / MAX_STEPS))
            components["terminal_time"] = weights.terminal_time * (1.0 if outcome > 0.0 else -1.0) * remaining_frac
        components["terminal"] = outcome
    else:
        components["terminal"] = 0.0

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

    def _obs_for_player(self, player: int) -> dict:
        assert self.env is not None, "call reset() first"
        obs = dict(self.env.state[player].observation)
        # env.run injects the Kaggle step field before calling agents, but
        # direct env.step users only see the raw game observation. Some strong
        # bots, including hellburner, require obs["step"].
        obs.setdefault("step", self.turn)
        return obs

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            self.seed = random.randint(1, 2**31 - 1)
        self.env = make("orbit_wars", configuration={"seed": int(self.seed)}, debug=False)
        self.env.reset(2)
        self.turn = 0
        obs = self._obs_for_player(self.player)
        self.last_stats = game_stats(obs, self.player)
        return obs

    def encoded(self):
        return encode_obs(self.current_obs())

    def current_obs(self) -> dict:
        return self._obs_for_player(self.player)

    def step(self, action_index: int) -> StepResult:
        assert self.env is not None, "call reset() first"
        my_obs = self._obs_for_player(0)
        return self.step_moves(decode_move(my_obs, action_index))

    def step_moves(self, my_moves: list[list[float]]) -> StepResult:
        assert self.env is not None, "call reset() first"
        opp_obs = self._obs_for_player(1)
        actions = [my_moves, self.opponent(opp_obs)]
        self.env.step(actions)
        self.turn += 1

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
