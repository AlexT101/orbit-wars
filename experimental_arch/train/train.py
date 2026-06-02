from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import wandb

from constants import ACTIONS_DIM, EXPERIMENTAL_ARCH_DIR, PLANET_SLOTS, TRAIN_DIR
from ppo import PPOConfig, train


TOTAL_STEPS = 100_000
ROLLOUT_STEPS = 512
UPDATE_EPOCHS = 2
MINIBATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE = 1.0e-4
CLIP_COEF = 0.20
ENT_COEF = 0.05
ENT_COEF_FINAL = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 1.0e-4
SEED = 7
DEVICE = "cpu"

MODEL = "entity_transformer"
HIDDEN = 128
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 4
LR_WARMUP_STEPS = 5_000
LR_SCHEDULE = "cosine"

OPPONENT = "nearest,hellburner,heuristic,snapshot_sample"
EVAL_OPPONENTS = "noop,random,nearest,hellburner,heuristic,self,snapshot_sample"
REWARD_MODE = "engine"
EVAL_EVERY_UPDATES = 25
EVAL_GAMES = 32
REPORT_EVERY_UPDATES = 1
SNAPSHOT_EVERY_UPDATES = 25
SNAPSHOT_POOL_SIZE = 4

USE_WANDB = True
WANDB_PROJECT = "orbit-wars"
RUN_TAG = f"a{ACTIONS_DIM}_p{PLANET_SLOTS}_reference_ppo_transformer_v1"
WANDB_RUN_NAME = f"galaxy-{RUN_TAG}"
CHECKPOINT_DIR = TRAIN_DIR / "checkpoints" / f"galaxy_{RUN_TAG}"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"

# The reference run used a BC checkpoint with a different feature/action schema.
# We start fresh unless this directory already has a native reference-pipeline checkpoint.
INIT_CHECKPOINT = None


def make_config() -> PPOConfig:
    return PPOConfig(
        total_steps=TOTAL_STEPS,
        rollout_steps=ROLLOUT_STEPS,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        learning_rate=LEARNING_RATE,
        clip_coef=CLIP_COEF,
        ent_coef=ENT_COEF,
        ent_coef_final=ENT_COEF_FINAL,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        opponent=OPPONENT,
        device=DEVICE,
        checkpoint_dir=str(CHECKPOINT_DIR),
        reward_mode=REWARD_MODE,
        eval_every_updates=EVAL_EVERY_UPDATES,
        eval_games=EVAL_GAMES,
        eval_opponents=EVAL_OPPONENTS,
        report_every_updates=REPORT_EVERY_UPDATES,
        init_checkpoint=INIT_CHECKPOINT,
        resume_checkpoint=str(LATEST_CHECKPOINT) if LATEST_CHECKPOINT.exists() else None,
        snapshot_every_updates=SNAPSHOT_EVERY_UPDATES,
        snapshot_pool_size=SNAPSHOT_POOL_SIZE,
        model=MODEL,
        hidden=HIDDEN,
        transformer_layers=TRANSFORMER_LAYERS,
        transformer_heads=TRANSFORMER_HEADS,
        lr_warmup_steps=LR_WARMUP_STEPS,
        lr_schedule=LR_SCHEDULE,
    )


def main() -> int:
    cfg = make_config()
    run = None
    if USE_WANDB:
        wandb.login()
        run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                **asdict(cfg),
                "experimental_arch_dir": str(EXPERIMENTAL_ARCH_DIR),
                "action_atoms": ACTIONS_DIM,
                "planet_slots": PLANET_SLOTS,
            },
            resume="allow",
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
    try:
        path = train(cfg)
        if run is not None:
            wandb.save(str(path))
            report = Path(cfg.checkpoint_dir) / "training_report.html"
            if report.exists():
                wandb.save(str(report))
        print(f"wrote {path}")
    finally:
        if run is not None:
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
