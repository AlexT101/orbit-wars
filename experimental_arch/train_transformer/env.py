from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from orbit_wars_engine import OrbitWarsEngine

from features import action_space, decode_action, encode_features, flat_action_mask, observation_space
from opponents import BotOpponent, ModelOpponent, Opponent, get_opponent


def scores_from_obs(obs: dict[str, Any], num_players: int = 2) -> list[int]:
    scores = [0 for _ in range(num_players)]
    for planet in obs.get("planets", []):
        owner = int(planet[1])
        if 0 <= owner < num_players:
            scores[owner] += int(planet[5])
    for fleet in obs.get("fleets", []):
        owner = int(fleet[1])
        if 0 <= owner < num_players:
            scores[owner] += int(fleet[6])
    return scores


@dataclass(frozen=True)
class RewardWeights:
    mode: str = "engine"
    terminal: float = 1.0
    terminal_time: float = 1.0
    production_income: float = 0.0002
    launch_penalty: float = -0.00004

    def as_engine_dict(self) -> dict[str, float]:
        if self.mode == "terminal":
            return {
                "terminal": self.terminal,
                "terminal_time": 0.0,
                "production_income": 0.0,
                "launch_penalty": 0.0,
            }
        if self.mode in {"engine", "default", "shaped"}:
            return {
                "terminal": self.terminal,
                "terminal_time": self.terminal_time,
                "production_income": self.production_income,
                "launch_penalty": self.launch_penalty,
            }
        raise ValueError(f"unknown reward mode: {self.mode!r}")


@dataclass
class StepResult:
    obs: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class OrbitWarsDuelEnv:
    """Reference-style two-player training wrapper backed by OrbitWarsEngine."""

    def __init__(
        self,
        seed: int | None = None,
        opponent: str | Opponent = "nearest",
        reward_weights: RewardWeights | None = None,
    ) -> None:
        self.seed = seed
        self.reward_weights = reward_weights or RewardWeights()
        self.opponent = get_opponent(opponent)
        self.engine = OrbitWarsEngine(
            num_players=2,
            reward_weights=self.reward_weights.as_engine_dict(),
        )
        self.player = 0
        self.turn = 0
        self.obs_pair: list[dict[str, Any]] | None = None

    def _obs_for_player(self, player: int) -> dict[str, Any]:
        assert self.obs_pair is not None, "call reset() first"
        obs = dict(self.obs_pair[player])
        obs.setdefault("player", player)
        obs.setdefault("step", self.turn)
        return obs

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            self.seed = int(np.random.randint(1, 2**31 - 1))
        if hasattr(self.opponent, "reset"):
            self.opponent.reset()
        self.obs_pair = self.engine.reset(int(self.seed))["observations"]
        self.turn = 0
        return self._obs_for_player(self.player)

    def current_obs(self) -> dict[str, Any]:
        return self._obs_for_player(self.player)

    def encoded(self):
        from features import encode_obs

        return encode_obs(self.current_obs(), player=self.player)

    def _opponent_moves(self, obs: dict[str, Any]) -> list[list[float]]:
        if hasattr(self.opponent, "act"):
            return self.opponent.act(obs)
        return self.opponent(obs)

    def step(self, action_index: int) -> StepResult:
        from features import decode_move

        return self.step_moves(decode_move(self._obs_for_player(self.player), action_index))

    def step_moves(self, my_moves: list[list[float]]) -> StepResult:
        assert self.obs_pair is not None, "call reset() first"
        opponent_player = 1 - self.player
        actions = [[] for _ in range(2)]
        actions[self.player] = my_moves
        actions[opponent_player] = self._opponent_moves(self._obs_for_player(opponent_player))
        out = self.engine.step(actions)
        self.obs_pair = out["observations"]
        self.turn += 1

        next_obs = self._obs_for_player(self.player)
        scores = scores_from_obs(next_obs, num_players=2)
        components = {
            name: float(values[self.player])
            for name, values in (out.get("reward_components") or {}).items()
        }
        return StepResult(
            obs=next_obs,
            reward=float(out["reward"][self.player]),
            done=bool(out["done"]),
            info={
                "seed": self.seed,
                "raw_rewards": [float(x) for x in scores],
                "reward_components": components,
                "stats": {
                    "own_score": float(scores[self.player]),
                    "enemy_score": float(scores[opponent_player]),
                },
            },
        )


class OrbitWarsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: str | Opponent = "hellburner",
        seed: int = 0,
        randomize_sides: bool = False,
        side_mode: str | None = None,
        reward_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.base_seed = seed
        self.next_seed = seed
        self.reward_weights = reward_weights
        self.engine = OrbitWarsEngine(num_players=2, reward_weights=reward_weights)
        self.opponent = BotOpponent(opponent) if isinstance(opponent, str) else opponent
        self.side_mode = side_mode or ("random" if randomize_sides else "fixed")
        if self.side_mode not in {"fixed", "random", "alternate"}:
            raise ValueError(f"unknown side_mode {self.side_mode!r}")
        self.randomize_sides = self.side_mode == "random"
        self.reset_count = 0
        self.learner_player = 0
        self.opponent_player = 1
        self.obs_pair = None
        self.feat = None
        self.turn = 0

        self.action_space = action_space()
        self.observation_space = observation_space()

    def set_opponent_checkpoint(self, checkpoint: str | Path | None) -> None:
        if isinstance(self.opponent, ModelOpponent):
            self.opponent.set_checkpoint(checkpoint)

    def set_next_seed(self, seed: int) -> None:
        self.next_seed = int(seed)

    def _encode_current(self) -> dict[str, np.ndarray]:
        assert self.obs_pair is not None
        model_obs, self.feat = encode_features(
            self.obs_pair[self.learner_player],
            player=self.learner_player,
        )
        return model_obs

    def action_masks(self) -> np.ndarray:
        assert self.feat is not None
        return flat_action_mask(self.feat)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None:
            seed = self.next_seed
            self.next_seed += 1
        super().reset(seed=int(seed))
        self.opponent.reset()
        self.obs_pair = self.engine.reset(int(seed))["observations"]
        if self.side_mode == "random":
            self.learner_player = int(self.np_random.integers(0, 2))
        elif self.side_mode == "alternate":
            self.learner_player = self.reset_count % 2
        else:
            self.learner_player = 0
        self.reset_count += 1
        self.opponent_player = 1 - self.learner_player
        self.turn = 0
        return self._encode_current(), {
            "seed": int(seed),
            "learner_player": self.learner_player,
            "opponent_player": self.opponent_player,
        }

    def _decode_action(self, action: np.ndarray) -> list[list[float]]:
        assert self.feat is not None
        return decode_action(self.feat, action)

    def step(self, action: np.ndarray):
        assert self.obs_pair is not None
        learner_moves = self._decode_action(action)
        opponent_moves = self.opponent.act(self.obs_pair[self.opponent_player])
        actions = [[] for _ in range(2)]
        actions[self.learner_player] = learner_moves
        actions[self.opponent_player] = opponent_moves
        out = self.engine.step(actions)
        self.obs_pair = out["observations"]
        self.turn += 1
        terminated = bool(out["done"])
        reward = float(out["reward"][self.learner_player])
        obs = self._encode_current()
        info = {
            "turn": self.turn,
            "learner_moves": len(learner_moves),
            "opponent_moves": len(opponent_moves),
            "learner_player": self.learner_player,
            "opponent_player": self.opponent_player,
            "reward_components": out.get("reward_components"),
        }
        if terminated:
            scores = scores_from_obs(self.obs_pair[self.learner_player], num_players=2)
            learner_score = int(scores[self.learner_player])
            opponent_score = int(scores[self.opponent_player])
            info["scores"] = scores
            info["learner_score"] = learner_score
            info["opponent_score"] = opponent_score
            info["score_diff"] = learner_score - opponent_score
            if learner_score == opponent_score:
                info["winner"] = None
            else:
                info["winner"] = self.learner_player if learner_score > opponent_score else self.opponent_player
            info["learner_won"] = bool(learner_score > opponent_score)
            info["tie"] = bool(learner_score == opponent_score)
        return obs, reward, terminated, False, info
