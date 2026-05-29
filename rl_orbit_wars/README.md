# Orbit Wars RL Scaffold

This is a deliberately small PPO setup for getting RL signs of life before spending real GPU time.

## What It Trains

- Two-player Orbit Wars only.
- Player 0 is the learned policy.
- Player 1 is cycled by episode through the comma-separated opponent pool you pass to the trainer.
- The action space is one launch per turn: `noop` or `(source planet, target planet, send fraction)`.
- The default model is an entity-pair MLP with masking. `--model entity_transformer` enables a small planet-token transformer.

## Reward Function

The default training reward is now the clean two-player objective: `terminal`, meaning `+1` for a win, `-1` for a loss, and `0` during the episode. This follows the community note that `+1/-1` is enough for 2p mode, especially after a behavior-cloning warm start.

You can still run reward-shaping experiments with `--reward-mode`:

- `terminal`: win/loss only.
- `terminal_score`: win/loss plus final score margin.
- `score_delta`: terminal reward plus dense actual-score/share deltas.
- `shaped`: all shaping terms below.

When shaping is enabled, each PPO log line prints the average contribution from every reward term.

- `score_delta`: change in your actual competition score, ships on planets plus ships in fleets.
- `score_share_delta`: change in your share of non-neutral ships.
- `production_delta`: immediate reward for gaining production.
- `production_share_delta`: relative economy control, which helps value learning before new planets pay off.
- `planet_delta`: small capture/loss signal.
- `economy_delta`: light relative potential using score, planets, production, and turns remaining.
- `enemy_score_delta`: small penalty when the opponent grows.
- `fleet_exposure_delta`: tiny penalty for moving ships into fleets, so pointless launches are less attractive.
- `terminal`: win/loss plus a tiny time preference. Faster wins are worth up to `+0.10` more, and faster losses are up to `-0.10` worse. `terminal_score` and shaped modes also add normalized final score margin.

Do not treat these weights as sacred. If local win rate improves but leaderboard performance drops, suspect reward overfitting first.

## Quick Smoke Test

```bash
python3 rl_orbit_wars/train.py --total-steps 128 --rollout-steps 32 --opponent noop
```

## Longer First Run

I would warm-start before PPO. By default this clones the simple nearest-capture teacher against only `noop` and `nearest`:

```bash
python3 rl_orbit_wars/pretrain_bc.py \
  --teacher nearest \
  --samples 20000 \
  --opponents noop,nearest \
  --out rl_orbit_wars/checkpoints/bc_teacher.pt
```

Pretraining writes `bc_metrics.jsonl` and updates `training_report.html` in the checkpoint directory. The report shows BC loss/accuracy separately.
The collector advances through noop teacher turns but caps accepted noop labels with `--max-noop-fraction` (default `0.10`), so BC does not become mostly "stand still" examples.

To clone your stronger Rust bot, first build `bots/mine/apollo`, then run:

```bash
python3 rl_orbit_wars/pretrain_bc.py \
  --teacher apollo \
  --samples 50000 \
  --opponents noop,nearest \
  --out rl_orbit_wars/checkpoints/bc_apollo.pt
```

Then fine-tune with PPO:

```bash
python3 rl_orbit_wars/train.py \
  --total-steps 20000 \
  --rollout-steps 32 \
  --opponent nearest,hellburner,heuristic \
  --eval-opponents noop,random,nearest,hellburner,heuristic \
  --reward-mode terminal \
  --init-checkpoint rl_orbit_wars/checkpoints/bc_teacher.pt \
  --checkpoint-dir rl_orbit_wars/checkpoints
```

`--opponent` accepts comma-separated built-ins/bots, frozen checkpoints (`checkpoint:path/to/latest.pt`), and the special names `self`, `self_sample`, `snapshot`, or `snapshot_sample`. Opponents are cycled per episode, so `nearest,hellburner,heuristic,snapshot_sample` gives each one regular coverage. Metrics include `train_return_<opponent>` and `train_win_rate_<opponent>` for recent completed episodes. Eval logs also include `eval_score`, the mean win rate across all `--eval-opponents`.

For more stable self-play, prefer a frozen snapshot pool over pure live self-play:

```bash
python3 rl_orbit_wars/train.py \
  --total-steps 100000 \
  --rollout-steps 512 \
  --opponent nearest,hellburner,heuristic,snapshot_sample \
  --snapshot-every-updates 25 \
  --snapshot-pool-size 4 \
  --eval-opponents noop,random,nearest,hellburner,heuristic,self \
  --reward-mode terminal_score \
  --init-checkpoint rl_orbit_wars/checkpoints/bc_hellburner.pt \
  --checkpoint-dir rl_orbit_wars/checkpoints_mixed_snapshots
```

Transformer run template:

```bash
python3 rl_orbit_wars/pretrain_bc.py \
  --teacher hellburner \
  --samples 50000 \
  --epochs 16 \
  --model entity_transformer \
  --hidden 128 \
  --transformer-layers 3 \
  --transformer-heads 4 \
  --opponents noop,nearest \
  --max-noop-fraction 0.10 \
  --out rl_orbit_wars/checkpoints/bc_hellburner_transformer.pt

python3 rl_orbit_wars/train.py \
  --total-steps 100000 \
  --rollout-steps 512 \
  --model entity_transformer \
  --hidden 128 \
  --transformer-layers 3 \
  --transformer-heads 4 \
  --lr-schedule cosine \
  --lr-warmup-steps 5000 \
  --entropy-coef 0.05 \
  --entropy-coef-final 0.01 \
  --opponent nearest,hellburner,heuristic,snapshot_sample \
  --snapshot-every-updates 25 \
  --snapshot-pool-size 4 \
  --eval-opponents noop,random,nearest,hellburner,heuristic,self \
  --reward-mode terminal_score \
  --init-checkpoint rl_orbit_wars/checkpoints/bc_hellburner_transformer.pt \
  --checkpoint-dir rl_orbit_wars/checkpoints_transformer_v1
```

To continue an interrupted PPO run, point both paths at the same run directory. With `--resume-checkpoint`, `--total-steps` means additional training steps:

```bash
python3 rl_orbit_wars/train.py \
  --total-steps 20000 \
  --rollout-steps 32 \
  --opponent nearest,hellburner,heuristic \
  --eval-opponents noop,random,nearest,hellburner,heuristic \
  --reward-mode terminal \
  --checkpoint-dir rl_orbit_wars/checkpoints \
  --resume-checkpoint rl_orbit_wars/checkpoints/latest.pt
```

While it runs, open this file in a browser:

```text
rl_orbit_wars/checkpoints/training_report.html
```

It refreshes every 5 seconds and shows return, reward components, PPO health, action behavior, and eval win rate versus whatever names you pass with `--eval-opponents`, such as `noop,random,nearest,hellburner,heuristic`.

If PPO starts from `--init-checkpoint`, the PPO charts include a vertical `pretrain ended` marker at step 0. That marks the boundary between imitation learning and RL fine-tuning.

The defaults mirror the simple community baseline style: `clip_coef=0.2`, `ent_coef=0.05`, `gamma=0.99`, `gae_lambda=0.95`, `learning_rate=3e-5`, `ppo_epochs=1`, `max_grad_norm=1`, `weight_decay=1e-4`. For transformer runs, prefer `--lr-schedule cosine`, a nonzero `--lr-warmup-steps`, and `--entropy-coef-final` below the starting entropy coefficient.

Watch `clip_frac`, `approx_kl`, `entropy`, `entropy_launch`, `entropy_source`, `entropy_target`, `entropy_send`, `explained_var`, `noop_rate`, and `launch_rate`. A steadily rising `clip_frac` is the early warning that PPO is becoming unstable. If `explained_var` never climbs, inspect the observation features or simplify the reward. If `noop_rate` sticks near 1.0, the policy learned to turtle; if `launch_rate` sticks near 1.0 with poor margins, it is likely throwing ships away.

You can rebuild the report from an existing run with:

```bash
python3 rl_orbit_wars/monitor.py --log-dir rl_orbit_wars/checkpoints
```

## Evaluate

```bash
python3 rl_orbit_wars/evaluate.py rl_orbit_wars/checkpoints/latest.pt --games 20 --opponent nearest
```

## Export A Kaggle Agent

```bash
python3 rl_orbit_wars/export_submission.py rl_orbit_wars/checkpoints/best.pt --out bots/mine/rl_ppo/main.py
python3 run_match.py rl_ppo nearest-sniper --kaggle --seed 42
```

## Next Useful Upgrades

- Add multi-action turns by sampling repeated masked launches until `noop`.
- Add enemy fleet impact features and target ETA features.
- Port the wrapper to the Rust engine once `orbit_wars_rust` is built locally.
- Only then try a tiny entity transformer, with LR warmup/decay and per-head entropy logging.
