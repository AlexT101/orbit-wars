from __future__ import annotations

from typing import Any

import wandb

from callbacks import SelfPlayCheckpointCallback
from constants import TRAIN_DIR
from env import OrbitWarsEnv
from opponents import ModelOpponent
from sb3_wiring import load_or_make_model
from wandb_logging import configure_wandb_sb3_logger


SEED = 0
TOTAL_TIMESTEPS = 4096_000
LEARNING_RATE = 3e-3
N_STEPS = 512
BATCH_SIZE = 128
N_EPOCHS = 4
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
NORMALIZE_ADVANTAGE = True
ENT_COEF = 0.0001
VF_COEF = 0.5
MAX_GRAD_NORM = 1
ROLLOUT_BUFFER_CLASS = None
ROLLOUT_BUFFER_KWARGS = None
TARGET_KL = None
STATS_WINDOW_SIZE = 100
TENSORBOARD_LOG = None
POLICY_KWARGS = None
DEVICE = "auto"
VERBOSE = 1
WANDB_PROJECT = "orbit-wars"
WANDB_RUN_NAME = "galaxy-selfplay"

CHECKPOINT_DIR = TRAIN_DIR / "checkpoints" / "galaxy_selfplay"
LOG_DIR = CHECKPOINT_DIR / "logs"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.zip"
SELF_PLAY_CHECKPOINT = CHECKPOINT_DIR / "self_play_opponent.zip"
CHECKPOINT_FREQ = 4096
EVAL_FREQ = 4096
EVAL_GAMES = 10
PROMOTION_WINRATE = 0.80


def _config_value(value: Any) -> Any:
    if value is None or isinstance(value, int | float | str | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _config_value(v) for k, v in value.items()}
    return str(value)


def ppo_kwargs() -> dict[str, Any]:
    return {
        "learning_rate": LEARNING_RATE,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "clip_range_vf": CLIP_RANGE_VF,
        "normalize_advantage": NORMALIZE_ADVANTAGE,
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "rollout_buffer_class": ROLLOUT_BUFFER_CLASS,
        "rollout_buffer_kwargs": {} if ROLLOUT_BUFFER_KWARGS is None else ROLLOUT_BUFFER_KWARGS,
        "target_kl": TARGET_KL,
        "stats_window_size": STATS_WINDOW_SIZE,
        "tensorboard_log": TENSORBOARD_LOG,
        "policy_kwargs": POLICY_KWARGS,
    }


def wandb_config() -> dict:
    ppo_config = {f"ppo/{key}": _config_value(value) for key, value in ppo_kwargs().items()}
    return {
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "device": DEVICE,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "eval_freq": EVAL_FREQ,
        "eval_games": EVAL_GAMES,
        "promotion_winrate": PROMOTION_WINRATE,
        "latest_checkpoint": str(LATEST_CHECKPOINT),
        "self_play_checkpoint": str(SELF_PLAY_CHECKPOINT),
        "log_dir": str(LOG_DIR),
        **ppo_config,
    }


def main() -> int:
    wandb.login()
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=wandb_config(),
    )
    wandb.define_metric("train/total_timesteps")
    wandb.define_metric("eval/*", step_metric="train/total_timesteps")
    wandb.define_metric("self_play/*", step_metric="train/total_timesteps")

    opponent = ModelOpponent(
        SELF_PLAY_CHECKPOINT if SELF_PLAY_CHECKPOINT.exists() else None,
        device=DEVICE,
        fallback="hellburner",
    )
    env = OrbitWarsEnv(opponent=opponent, seed=SEED)
    model, resumed = load_or_make_model(
        LATEST_CHECKPOINT,
        env,
        verbose=VERBOSE,
        seed=SEED,
        device=DEVICE,
        **ppo_kwargs(),
    )
    env.set_next_seed(SEED + int(model.num_timesteps))
    model.set_logger(configure_wandb_sb3_logger(LOG_DIR))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if resumed:
        print(f"resumed learner from latest checkpoint: {LATEST_CHECKPOINT}")
    else:
        print("created fresh learner; no latest checkpoint found")
    if not SELF_PLAY_CHECKPOINT.exists():
        model.save(SELF_PLAY_CHECKPOINT)
        env.set_opponent_checkpoint(SELF_PLAY_CHECKPOINT)
        source = "resumed learner" if resumed else "fresh learner"
        print(f"initialized self-play opponent from {source}: {SELF_PLAY_CHECKPOINT}")
    callback = SelfPlayCheckpointCallback(
        checkpoint_dir=CHECKPOINT_DIR,
        self_play_checkpoint=SELF_PLAY_CHECKPOINT,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        eval_games=EVAL_GAMES,
        promotion_winrate=PROMOTION_WINRATE,
        seed=SEED,
        device=DEVICE,
        verbose=VERBOSE,
    )
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=not resumed,
        )
        model.save(CHECKPOINT_DIR / "final.zip")
        wandb.save(str(CHECKPOINT_DIR / "final.zip"))
        print(f"saved {CHECKPOINT_DIR / 'final.zip'}")
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
