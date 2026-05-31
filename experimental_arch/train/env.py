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

    def __init__(self, opponent: str | Opponent = "hellburner", seed: int = 0):
        super().__init__()
        self.base_seed = seed
        self.next_seed = seed
        self.engine = OrbitWarsEngine(num_players=2)
        self.opponent = BotOpponent(opponent) if isinstance(opponent, str) else opponent
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
        model_obs, self.feat = encode_features(self.obs_pair[0], player=0)
        return model_obs

    def action_masks(self) -> np.ndarray:
        assert self.feat is not None
        return flat_action_mask(self.feat)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = self.next_seed
            self.next_seed += 1
        self.opponent.reset()
        self.obs_pair = self.engine.reset(int(seed))["observations"]
        self.turn = 0
        return self._encode_current(), {"seed": int(seed)}

    def _decode_action(self, action: np.ndarray) -> list[list[float]]:
        assert self.feat is not None
        return decode_action(self.feat, action)

    def step(self, action: np.ndarray):
        assert self.obs_pair is not None
        learner_moves = self._decode_action(action)
        opponent_moves = self.opponent.act(self.obs_pair[1])
        out = self.engine.step([learner_moves, opponent_moves])
        self.obs_pair = out["observations"]
        self.turn += 1
        terminated = bool(out["done"])
        reward = float(out["reward"][0])
        obs = self._encode_current()
        info = {
            "turn": self.turn,
            "learner_moves": len(learner_moves),
            "opponent_moves": len(opponent_moves),
            "reward_components": out.get("reward_components"),
        }
        if terminated:
            scores = scores_from_obs(self.obs_pair[0], num_players=2)
            info["scores"] = scores
            info["winner"] = int(scores[1] > scores[0])
            info["tie"] = bool(scores[0] == scores[1])
        return obs, reward, terminated, False, info
