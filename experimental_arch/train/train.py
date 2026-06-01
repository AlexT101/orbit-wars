from __future__ import annotations

import random
from collections import deque
from typing import Any

import wandb

from callbacks import SelfPlayCheckpointCallback
from constants import ACTIONS_DIM, ACTION_CHOICES_PER_SOURCE, LAUNCH_GATE_CHOICES, TARGET_CHOICES, TRAIN_DIR
from env import OrbitWarsEnv
from opponents import ModelOpponent
from sb3_wiring import load_or_make_model
from self_play import bootstrap_snapshot_from_legacy, read_current_snapshot, save_model_snapshot
from wandb_logging import configure_wandb_sb3_logger


SEED = random.randint(0, 1_000_000_000)
TOTAL_TIMESTEPS = 4096_000
LEARNING_RATE = 3e-5
N_STEPS = 512
BATCH_SIZE = 64
N_EPOCHS = 4
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
NORMALIZE_ADVANTAGE = True
ENT_COEF = 0.0000
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
ROLLOUT_BUFFER_CLASS = None
ROLLOUT_BUFFER_KWARGS = None
TARGET_KL = None
STATS_WINDOW_SIZE = 100
TENSORBOARD_LOG = None
POLICY_KWARGS = None
DEVICE = "auto"
VERBOSE = 1
WANDB_PROJECT = "orbit-wars"
ACTION_SPACE_TAG = f"a{ACTIONS_DIM}_g{LAUNCH_GATE_CHOICES}_t{TARGET_CHOICES}"
WANDB_RUN_NAME = f"galaxy-selfplay-{ACTION_SPACE_TAG}"

CHECKPOINT_DIR = TRAIN_DIR / "checkpoints" / f"galaxy_selfplay_{ACTION_SPACE_TAG}"
LOG_DIR = CHECKPOINT_DIR / "logs"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.zip"
SELF_PLAY_DIR = CHECKPOINT_DIR / "self_play"
SELF_PLAY_POINTER = SELF_PLAY_DIR / "current.txt"
LEGACY_SELF_PLAY_CHECKPOINT = CHECKPOINT_DIR / "self_play_opponent.zip"
TRAINING_SIDE_MODE = "alternate"
SELF_PLAY_OPPONENT_DETERMINISTIC = False
REWARD_WEIGHTS = None

CHECKPOINT_FREQ = 4096
EVAL_FREQ = 4096 * 3
EVAL_GAMES = 5
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
        "actions_dim": ACTIONS_DIM,
        "action_choices_per_source": ACTION_CHOICES_PER_SOURCE,
        "launch_gate_choices": LAUNCH_GATE_CHOICES,
        "target_choices": TARGET_CHOICES,
        "total_timesteps": TOTAL_TIMESTEPS,
        "device": DEVICE,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "eval_freq": EVAL_FREQ,
        "eval_games": EVAL_GAMES,
        "promotion_winrate": PROMOTION_WINRATE,
        "latest_checkpoint": str(LATEST_CHECKPOINT),
        "self_play_dir": str(SELF_PLAY_DIR),
        "self_play_pointer": str(SELF_PLAY_POINTER),
        "legacy_self_play_checkpoint": str(LEGACY_SELF_PLAY_CHECKPOINT),
        "training_side_mode": TRAINING_SIDE_MODE,
        "self_play_opponent_deterministic": SELF_PLAY_OPPONENT_DETERMINISTIC,
        "reward_weights": _config_value(REWARD_WEIGHTS),
        "log_dir": str(LOG_DIR),
        **ppo_config,
    }


def clear_resume_rollout_stats(model) -> None:
    """Drop saved Monitor windows so resumed logs show the current run."""
    model.ep_info_buffer = deque(maxlen=STATS_WINDOW_SIZE)
    model.ep_success_buffer = deque(maxlen=STATS_WINDOW_SIZE)


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
        None,
        device=DEVICE,
        fallback="hellburner",
        deterministic=SELF_PLAY_OPPONENT_DETERMINISTIC,
    )
    env = OrbitWarsEnv(opponent=opponent, seed=SEED, side_mode=TRAINING_SIDE_MODE, reward_weights=REWARD_WEIGHTS)
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
        current_self_play = bootstrap_snapshot_from_legacy(
            legacy_checkpoint=LEGACY_SELF_PLAY_CHECKPOINT,
            self_play_dir=SELF_PLAY_DIR,
            pointer_path=SELF_PLAY_POINTER,
        )
        if current_self_play is None:
            current_self_play = read_current_snapshot(SELF_PLAY_POINTER, SELF_PLAY_DIR)
        opponent_checkpoint = current_self_play[1] if current_self_play is not None else None
        if opponent_checkpoint is not None:
            env.set_opponent_checkpoint(opponent_checkpoint)
        clear_resume_rollout_stats(model)
        print(f"resumed learner from latest checkpoint: {LATEST_CHECKPOINT}")
        print("cleared saved SB3 rollout episode window; ep_rew_mean will reflect fresh episodes")
    else:
        current_self_play = save_model_snapshot(model, SELF_PLAY_DIR, SELF_PLAY_POINTER, generation=0)
        opponent_checkpoint = current_self_play[1]
        env.set_opponent_checkpoint(opponent_checkpoint)
        print("created fresh learner; no latest checkpoint found")
        print(f"initialized self-play opponent gen=0 from fresh learner: {opponent_checkpoint}")
    if opponent_checkpoint is not None:
        generation = current_self_play[0] if current_self_play is not None else "?"
        print(f"loaded self-play opponent gen={generation}: {opponent_checkpoint}")
    callback = SelfPlayCheckpointCallback(
        checkpoint_dir=CHECKPOINT_DIR,
        self_play_dir=SELF_PLAY_DIR,
        self_play_pointer=SELF_PLAY_POINTER,
        legacy_self_play_checkpoint=LEGACY_SELF_PLAY_CHECKPOINT,
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
