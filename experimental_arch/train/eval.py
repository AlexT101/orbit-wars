from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sb3_contrib import MaskablePPO

from env import OrbitWarsEnv
from opponents import BotOpponent, ModelOpponent, Opponent


@dataclass
class EvalResult:
    games: int
    wins: int
    ties: int
    losses: int
    mean_reward: float
    mean_score_diff: float

    @property
    def winrate(self) -> float:
        return (self.wins + 0.5 * self.ties) / max(1, self.games)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "games": self.games,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "winrate": self.winrate,
            "mean_reward": self.mean_reward,
            "mean_score_diff": self.mean_score_diff,
        }


def hellburner_opponent() -> BotOpponent:
    return BotOpponent("hellburner")


def model_opponent(checkpoint: str | Path | None, device: str = "auto") -> ModelOpponent:
    return ModelOpponent(checkpoint, device=device, fallback="hellburner")


def evaluate_model(
    model: MaskablePPO,
    opponent_factory: Callable[[], Opponent],
    *,
    games: int,
    seed: int,
    deterministic: bool = True,
) -> EvalResult:
    wins = ties = losses = 0
    rewards: list[float] = []
    score_diffs: list[int] = []

    for i in range(games):
        env = OrbitWarsEnv(opponent=opponent_factory(), seed=seed + i)
        obs, _ = env.reset(seed=seed + i)
        done = False
        total_reward = 0.0
        info = {}
        while not done:
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=env.action_masks(),
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        scores = info.get("scores", [0, 0])
        diff = int(scores[0]) - int(scores[1])
        score_diffs.append(diff)
        rewards.append(total_reward)
        if info.get("tie"):
            ties += 1
        elif diff > 0:
            wins += 1
        else:
            losses += 1

    return EvalResult(
        games=games,
        wins=wins,
        ties=ties,
        losses=losses,
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        mean_score_diff=float(np.mean(score_diffs)) if score_diffs else 0.0,
    )
