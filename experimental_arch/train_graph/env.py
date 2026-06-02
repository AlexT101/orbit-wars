from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from orbit_wars_engine import OrbitWarsEngine

from features import action_space, decode_action, encode_features, flat_action_mask, observation_space
from opponents import BotOpponent, ModelOpponent, Opponent


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
