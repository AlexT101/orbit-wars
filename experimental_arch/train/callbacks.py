from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback

from eval import evaluate_model, hellburner_opponent, model_opponent
from opponents import Opponent

try:
    import wandb
except ImportError:  # pragma: no cover - training can still run without wandb installed.
    wandb = None


@dataclass(frozen=True)
class EvalOpponentSpec:
    name: str
    factory: Callable[[], Opponent]
    promote_on_winrate: bool = False


EVAL_POLICY_MODES: tuple[tuple[str, bool], ...] = (
    ("deterministic", True),
    ("sampled", False),
)
PROMOTION_POLICY_MODE = "deterministic"


class SelfPlayCheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        self_play_checkpoint: Path,
        checkpoint_freq: int,
        eval_freq: int,
        eval_games: int,
        promotion_winrate: float,
        seed: int,
        device: str,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.checkpoint_dir = checkpoint_dir
        self.self_play_checkpoint = self_play_checkpoint
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.eval_games = eval_games
        self.promotion_winrate = promotion_winrate
        self.seed = seed
        self.device = device
        self.metrics_path = checkpoint_dir / "eval.jsonl"
        self.opponent_generation = 0
        self.promotions = 0

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not self.self_play_checkpoint.exists():
            self.model.save(self.self_play_checkpoint)
            if hasattr(self.training_env, "env_method"):
                self.training_env.env_method("set_opponent_checkpoint", self.self_play_checkpoint)
            self._write_event(
                {
                    "step": self.num_timesteps,
                    "event": "initialized_self_play_checkpoint",
                    "path": str(self.self_play_checkpoint),
                }
            )
            if self.verbose:
                print(f"[self-play] initialized gen=0 from fresh model: {self.self_play_checkpoint}")
        self._record_self_play_state(promoted=False)

    def _write_event(self, event: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _wandb_log(self, values: dict[str, int | float | str | bool]) -> None:
        if wandb is not None and wandb.run is not None:
            wandb.log({"train/total_timesteps": self.num_timesteps, **values}, step=self.num_timesteps)

    def _eval_opponents(self) -> list[EvalOpponentSpec]:
        return [
            EvalOpponentSpec(
                name="self_play",
                factory=lambda: model_opponent(self.self_play_checkpoint, device=self.device),
                promote_on_winrate=True,
            ),
            EvalOpponentSpec(name="hellburner", factory=hellburner_opponent),
        ]

    def _record_eval(self, opponent: str, mode: str, result, promoted: bool | None = None) -> None:
        prefix = f"eval/{opponent}_{mode}"
        self.logger.record(f"{prefix}_winrate", result.winrate)
        self.logger.record(f"{prefix}_reward", result.mean_reward)
        self.logger.record(f"{prefix}_score_diff", result.mean_score_diff)
        self.logger.record(f"{prefix}_games", result.games)
        if promoted is not None:
            self.logger.record(f"{prefix}_promoted", float(promoted))

        wandb_values = {
            f"{prefix}/winrate": result.winrate,
            f"{prefix}/mean_reward": result.mean_reward,
            f"{prefix}/mean_score_diff": result.mean_score_diff,
            f"{prefix}/wins": result.wins,
            f"{prefix}/ties": result.ties,
            f"{prefix}/losses": result.losses,
            f"{prefix}/games": result.games,
        }
        if promoted is not None:
            wandb_values[f"{prefix}/promoted"] = float(promoted)
        self._wandb_log(wandb_values)

    def _record_self_play_state(self, promoted: bool) -> None:
        self.logger.record("self_play/opponent_generation", self.opponent_generation)
        self.logger.record("self_play/promotions", self.promotions)
        self.logger.record("self_play/promotion_threshold", self.promotion_winrate)
        self.logger.record("self_play/switched_opponent", float(promoted))
        self._wandb_log(
            {
                "self_play/opponent_generation": self.opponent_generation,
                "self_play/promotions": self.promotions,
                "self_play/promotion_threshold": self.promotion_winrate,
                "self_play/switched_opponent": float(promoted),
            }
        )

    def _save_latest(self) -> Path:
        path = self.checkpoint_dir / "latest.zip"
        self.model.save(path)
        return path

    def _promote_self_play(self) -> None:
        self.opponent_generation += 1
        self.promotions += 1
        self.model.save(self.self_play_checkpoint)
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_opponent_checkpoint", self.self_play_checkpoint)

    def _on_step(self) -> bool:
        if self.checkpoint_freq > 0 and self.num_timesteps % self.checkpoint_freq == 0:
            latest = self._save_latest()
            if self.verbose:
                print(f"saved checkpoint {latest}")

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            promoted = False
            opponent_summaries = []
            for opp_idx, opponent in enumerate(self._eval_opponents()):
                for mode, deterministic in EVAL_POLICY_MODES:
                    result = evaluate_model(
                        self.model,
                        opponent.factory,
                        games=self.eval_games,
                        seed=self.seed + self.num_timesteps + 10_000 * opp_idx,
                        deterministic=deterministic,
                    )
                    can_promote = opponent.promote_on_winrate and mode == PROMOTION_POLICY_MODE
                    this_promoted = can_promote and result.winrate >= self.promotion_winrate
                    promoted = promoted or this_promoted
                    event = {
                        "step": self.num_timesteps,
                        "opponent": opponent.name,
                        "policy_mode": mode,
                        "deterministic": deterministic,
                        **result.as_dict(),
                    }
                    if can_promote:
                        event["promoted"] = this_promoted
                        event["opponent_generation"] = self.opponent_generation
                    self._write_event(event)
                    self._record_eval(opponent.name, mode, result, promoted=this_promoted if can_promote else None)
                    opponent_summaries.append(
                        f"{opponent.name}/{mode}:wr={result.winrate:.3f},rew={result.mean_reward:.3f}"
                    )

            if promoted:
                self._promote_self_play()
                if self.verbose:
                    print(f"[self-play] switched to gen={self.opponent_generation}: {self.self_play_checkpoint}")
            self._record_self_play_state(promoted=promoted)
            if self.verbose:
                print(f"[eval] step={self.num_timesteps} " + " | ".join(opponent_summaries))

        return True

    def _on_training_end(self) -> None:
        self._save_latest()
