from __future__ import annotations

import argparse

from orbit_wars_rl.ppo import PPOConfig, train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a small PPO Orbit Wars policy.")
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--entropy-coef-final", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--model", choices=["mlp", "entity_transformer"], default="mlp")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-schedule", choices=["linear", "cosine", "constant"], default="linear")
    parser.add_argument(
        "--opponent",
        default="nearest",
        help=(
            "Comma-separated training opponents. Supports random/nearest/noop, bot names "
            "like hellburner/heuristic, and self/self_sample."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=["terminal", "terminal_score", "score_delta", "shaped"],
        default="terminal",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", default="rl_orbit_wars/checkpoints")
    parser.add_argument("--eval-every-updates", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--eval-opponents", default="random,nearest,baselines/starter,hellburner,heuristic")
    parser.add_argument("--report-every-updates", type=int, default=1)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Continue from a PPO checkpoint. Logs in --checkpoint-dir are appended, and --total-steps is additional.",
    )
    parser.add_argument(
        "--snapshot-every-updates",
        type=int,
        default=0,
        help="When using snapshot/snapshot_sample opponents, add a frozen copy of the current policy every N PPO updates.",
    )
    parser.add_argument(
        "--snapshot-pool-size",
        type=int,
        default=4,
        help="Maximum number of frozen self-play snapshots kept in the checkpoint.",
    )
    args = parser.parse_args()

    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        ent_coef=args.entropy_coef,
        ent_coef_final=args.entropy_coef_final,
        update_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        model=args.model,
        hidden=args.hidden,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_schedule=args.lr_schedule,
        opponent=args.opponent,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        reward_mode=args.reward_mode,
        eval_every_updates=args.eval_every_updates,
        eval_games=args.eval_games,
        eval_opponents=args.eval_opponents,
        report_every_updates=args.report_every_updates,
        init_checkpoint=args.init_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        snapshot_every_updates=args.snapshot_every_updates,
        snapshot_pool_size=args.snapshot_pool_size,
    )
    path = train(cfg)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
