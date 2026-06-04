from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch as th

from features import decode_action, encode_features, flat_action_mask
from sb3_wiring import make_model
from env import OrbitWarsEnv


GAMES = 4
SEED = 9999999
DEVICE = "auto"

# This matches the PPO rollout policy: False means sample from the initial policy
# distribution instead of taking its argmax action.
DETERMINISTIC = False
PRINT_EACH_GAME = True

PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 1,
    "rollout_buffer_kwargs": {},
    "target_kl": None,
    "stats_window_size": 100,
    "tensorboard_log": None,
    "policy_kwargs": None,
}


class InitialSelfOpponent:
    name = "initial_self"

    def __init__(self):
        self.model = None

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> list[list[float]]:
        if self.model is None:
            raise RuntimeError("InitialSelfOpponent.model must be assigned before play")
        player = int(obs.get("player", 1))
        model_obs, feat = encode_features(obs, player=player)
        action, _ = self.model.predict(
            model_obs,
            deterministic=DETERMINISTIC,
            action_masks=flat_action_mask(feat),
        )
        return decode_action(feat, action)


@dataclass
class GameResult:
    game: int
    seed: int
    learner_player: int
    reward: float
    final_reward: float
    turns: int
    score_diff: int
    outcome: str


def play_games() -> list[GameResult]:
    np.random.seed(SEED)
    th.manual_seed(SEED)

    opponent = InitialSelfOpponent()
    env = OrbitWarsEnv(opponent=opponent, seed=SEED, side_mode="alternate")
    model = make_model(env, device=DEVICE, verbose=0, seed=SEED, **PPO_KWARGS)
    opponent.model = model

    results: list[GameResult] = []
    for game in range(GAMES):
        seed = SEED + game
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        final_reward = 0.0
        done = False
        final_info = info
        while not done:
            action, _ = model.predict(
                obs,
                deterministic=DETERMINISTIC,
                action_masks=env.action_masks(),
            )
            obs, reward, terminated, truncated, final_info = env.step(action)
            final_reward = float(reward)
            total_reward += float(reward)
            done = bool(terminated or truncated)

        if final_info.get("tie"):
            outcome = "tie"
        elif final_info.get("score_diff", 0) > 0:
            outcome = "win"
        else:
            outcome = "loss"
        result = GameResult(
            game=game,
            seed=seed,
            learner_player=int(final_info["learner_player"]),
            reward=total_reward,
            final_reward=final_reward,
            turns=int(final_info["turn"]),
            score_diff=int(final_info.get("score_diff", 0)),
            outcome=outcome,
        )
        results.append(result)
        if PRINT_EACH_GAME:
            print(
                f"finished game={result.game} seed={result.seed} "
                f"learner_player={result.learner_player} reward={result.reward:+.6f} "
                f"final_reward={result.final_reward:+.6f} turns={result.turns} "
                f"score_diff={result.score_diff} outcome={result.outcome}",
                flush=True,
            )
    return results


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    results = play_games()
    rewards = [r.reward for r in results]
    side0 = [r.reward for r in results if r.learner_player == 0]
    side1 = [r.reward for r in results if r.learner_player == 1]

    side_counts = {0: len(side0), 1: len(side1)}
    if abs(side_counts[0] - side_counts[1]) > 1:
        raise AssertionError(f"side alternation failed: {side_counts}")

    sign_failures = []
    for r in results:
        if r.outcome == "win" and r.final_reward <= 0.0:
            sign_failures.append(r)
        elif r.outcome == "loss" and r.final_reward >= 0.0:
            sign_failures.append(r)
    if sign_failures:
        details = ", ".join(
            f"game={r.game} outcome={r.outcome} final_reward={r.final_reward:+.6f}"
            for r in sign_failures
        )
        raise AssertionError(f"terminal reward sign did not match learner outcome: {details}")

    print(f"initial self-play balance probe")
    print(f"games={GAMES} seed={SEED} deterministic={DETERMINISTIC}")
    print(f"side_counts={side_counts}")
    print(f"mean_reward={mean(rewards):+.6f}")
    print(f"mean_reward_player0={mean(side0):+.6f}")
    print(f"mean_reward_player1={mean(side1):+.6f}")
    print(f"mean_turns={mean([float(r.turns) for r in results]):.2f}")
    print()
    print("game seed learner_player reward final_reward turns score_diff outcome")
    for r in results:
        print(
            f"{r.game:>4} {r.seed:>10} {r.learner_player:>14} "
            f"{r.reward:+.6f} {r.final_reward:+.6f} "
            f"{r.turns:>5} {r.score_diff:>10} {r.outcome}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
