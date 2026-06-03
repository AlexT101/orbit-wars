# Orbit Wars RL Scaffold

This is a deliberately small PPO setup for getting RL signs of life before spending real GPU time.

## What It Trains

- Two-player Orbit Wars only.
- Player 0 is the learned policy.
- Player 1 is cycled by episode through the comma-separated opponent pool you pass to the trainer.
- The policy head scores `noop` plus `(source planet, target planet, send fraction)` actions. At inference time one forward pass can emit multiple fleet launches whose logits clear the noop threshold.
- The default model is an entity-pair MLP with masking. `--model entity_transformer` enables a small planet-token transformer.

## Feature Schema

The current observation encoder intentionally excludes arbitrary planet IDs, planet slot IDs, quadrant flags, and constant bias columns. Those signals are too easy to overfit to map-generation/order artifacts.

Planet tokens contain only gameplay-relevant state:

- centered position, radius, orbital angle, orbiting/comet flags
- owner class from our perspective
- ships, production, launchability
- 10-turn future position/motion
- predicted inbound friendly/enemy fleet pressure
- nearest owned/enemy/neutral planet distances

Global features contain step, angular velocity, score/production/planet shares, fleet shares, comet count, and orbiting count. Old checkpoints from earlier schemas are intentionally incompatible.

## Reward Function

The default training reward is now the two-player objective: `terminal`, meaning win/loss at game end plus a small faster-win/slower-loss bonus. This follows the community note that win/loss is enough for 2p mode, especially after a behavior-cloning warm start.

You can still run reward-shaping experiments with `--reward-mode`:

- `terminal`: win/loss plus the small terminal-time bonus.
- `terminal_score`: compatibility alias for terminal-style reward; no final margin is added.
- `score_delta`: terminal reward plus positive-only own score increase.
- `shaped`: all shaping terms below.

When shaping is enabled, each PPO log line prints the average contribution from every reward term.

- `control`: small per-step reward for current control, combining production share, planet share, and score share.
- `score_increase`: positive-only reward for our score increasing. Score drops and opponent growth are not directly punished.
- `production_increase`: positive-only reward for gaining production.
- `planet_increase`: positive-only reward for gaining planets.
- `enemy_planet_capture`: small extra reward when that gained planet came from an opponent.
- `terminal`: dominant win/loss reward.
- `terminal_time`: faster wins are better; slower losses are less bad.

There is no final score-margin reward. Do not treat these weights as sacred. If local win rate improves but leaderboard performance drops, suspect reward overfitting first.

## Quick Smoke Test

```bash
python3 rl_orbit_wars/train.py --total-steps 128 --rollout-steps 32 --opponent random
```

## Longer First Run

I would warm-start before PPO. By default this clones the simple nearest-capture teacher against `random`, `nearest`, and `baselines/starter`:

```bash
python3 rl_orbit_wars/pretrain_bc.py \
  --teacher nearest \
  --samples 20000 \
  --opponents random,nearest,baselines/starter \
  --out rl_orbit_wars/checkpoints/bc_teacher.pt
```

Pretraining writes `bc_metrics.jsonl` and updates `training_report.html` in the checkpoint directory. The report shows BC loss/accuracy separately.
The collector advances through noop teacher turns but caps accepted noop labels with `--max-noop-fraction` (default `0.10`), so BC does not become mostly "stand still" examples.

To clone your stronger Rust bot, first build `bots/mine/apollo`, then run:

```bash
python3 rl_orbit_wars/pretrain_bc.py \
  --teacher apollo \
  --samples 50000 \
  --opponents random,nearest,baselines/starter \
  --out rl_orbit_wars/checkpoints/bc_apollo.pt
```

## Source-Target Replay Imitation

`imitation.py` is the replay imitation path for the entity transformer. It
learns which source planet and target planet to choose from expert actions, but
does not learn the ship count/send fraction. That leaves count sizing to the
separate hand-coded logic we already trust more.

By default it copies the cached replay agent named `Isaiah @ Tufa Labs` and
reuses the same AlphaOW replay manifests/cache as the value-net work:

```bash
python3 rl_orbit_wars/imitation.py \
  --samples 50000 \
  --epochs 12 \
  --hidden 128 \
  --transformer-layers 3 \
  --transformer-heads 4 \
  --out rl_orbit_wars/checkpoints/imitation_isaiah_source_target.pt
```

It writes `imitation_metrics.jsonl`, `imitation_config.json`,
`imitation_dataset_stats.json`, `latest_source_target.pt`, and
`training_report.html` in the checkpoint directory. The report auto-refreshes
and includes source accuracy, target accuracy, pair accuracy/top-k, loss, and
throughput.

If the local cache does not contain Isaiah yet, populate it from the daily
manifest and pin the dates you want:

```bash
python3 rl_orbit_wars/imitation.py \
  --download-from-alphaow-manifest \
  --start-date 2026-05-26 \
  --end-date 2026-05-28 \
  --samples 50000 \
  --out rl_orbit_wars/checkpoints/imitation_isaiah_source_target.pt
```

For a quick parser/model smoke test on checked-in replays, use all visible
players instead of the named agent:

```bash
python3 rl_orbit_wars/imitation.py \
  --replay-dir replays \
  --no-alphaow-manifests \
  --target-mode all \
  --target-name "" \
  --samples 128 \
  --epochs 1 \
  --out rl_orbit_wars/checkpoints/imitation_smoke.pt
```

## Replay Imitation + Inverse RL

`imitation_irl.py` trains from Kaggle replay JSONs instead of calling a local teacher. By default it looks in the alphaow replay manifests/cache under `bots/mine/alphaow_transformer/train`, scans cached player win rates, picks the top cached agent, learns multi-launch labels with a sampled multi-label policy loss, and also saves a contrastive inverse-RL reward model that scores expert state-action pairs above valid alternatives.

```bash
python3 rl_orbit_wars/imitation_irl.py \
  --samples 50000 \
  --epochs 8 \
  --model entity_transformer \
  --hidden 128 \
  --target-mode top-agent \
  --out rl_orbit_wars/checkpoints/irl_policy.pt \
  --reward-out rl_orbit_wars/checkpoints/irl_reward.pt
```

If the cache is missing, let it reuse the KaggleHub dataset list from `bots/alphaow_experimental/prometheus/train/manifest.csv`:

```bash
python3 rl_orbit_wars/imitation_irl.py \
  --download-from-alphaow-manifest \
  --start-date 2026-05-26 \
  --end-date 2026-05-28 \
  --samples 50000 \
  --out rl_orbit_wars/checkpoints/irl_policy.pt \
  --reward-out rl_orbit_wars/checkpoints/irl_reward.pt
```

To pin the exact cached #1-style teacher from the current replay scan, use the replay agent name directly:

```bash
python3 rl_orbit_wars/imitation_irl.py \
  --target-mode agent-name \
  --target-name "Isaiah @ Tufa Labs" \
  --samples 50000 \
  --out rl_orbit_wars/checkpoints/irl_policy_isaiah.pt \
  --reward-out rl_orbit_wars/checkpoints/irl_reward_isaiah.pt
```

The policy checkpoint can warm-start PPO the same way as normal BC:

```bash
python3 rl_orbit_wars/train.py \
  --total-steps 100000 \
  --rollout-steps 512 \
  --model entity_transformer \
  --opponent nearest,hellburner,heuristic,snapshot_sample \
  --reward-mode terminal \
  --init-checkpoint rl_orbit_wars/checkpoints/irl_policy.pt \
  --checkpoint-dir rl_orbit_wars/checkpoints_irl_terminal
```

Then fine-tune with PPO:

```bash
python3 rl_orbit_wars/train.py \
  --total-steps 20000 \
  --rollout-steps 32 \
  --opponent nearest,hellburner,heuristic \
  --eval-opponents random,nearest,baselines/starter,hellburner,heuristic \
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
  --eval-opponents random,nearest,baselines/starter,hellburner,heuristic,self \
  --reward-mode terminal \
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
  --opponents random,nearest,baselines/starter \
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
  --eval-opponents random,nearest,baselines/starter,hellburner,heuristic,self \
  --reward-mode terminal \
  --init-checkpoint rl_orbit_wars/checkpoints/bc_hellburner_transformer.pt \
  --checkpoint-dir rl_orbit_wars/checkpoints_transformer_v1
```

To continue an interrupted PPO run, point both paths at the same run directory. With `--resume-checkpoint`, `--total-steps` means additional training steps:

```bash
python3 rl_orbit_wars/train.py \
  --total-steps 20000 \
  --rollout-steps 32 \
  --opponent nearest,hellburner,heuristic \
  --eval-opponents random,nearest,baselines/starter,hellburner,heuristic \
  --reward-mode terminal \
  --checkpoint-dir rl_orbit_wars/checkpoints \
  --resume-checkpoint rl_orbit_wars/checkpoints/latest.pt
```

While it runs, open this file in a browser:

```text
rl_orbit_wars/checkpoints/training_report.html
```

It refreshes every 5 seconds and shows return, reward components, PPO health, action behavior, and eval win rate versus whatever names you pass with `--eval-opponents`, such as `random,nearest,baselines/starter,hellburner,heuristic`.

## Overnight Curriculum

Use the curriculum runner to train against harder and harder bots from the rating ladder. It uses `terminal` reward for every phase, starts with only `random`, `nearest`, and `baselines/starter`, ignores `mine/apollo_backup`, and promotes to the next rating band only when every gate bot clears the current win-rate threshold. The threshold starts strict (`0.98`) for starter bots and decays toward `0.65` for later phases.

```bash
python3 rl_orbit_wars/curriculum_train.py \
  --checkpoint-dir rl_orbit_wars/checkpoints_curriculum_terminal \
  --init-checkpoint rl_orbit_wars/checkpoints/bc_hellburner_transformer.pt \
  --total-budget-steps 300000 \
  --chunk-steps 10000 \
  --start-gate-threshold 0.98 \
  --end-gate-threshold 0.65 \
  --gate-games 16
```

The runner writes `curriculum_config.json`, `curriculum_state.json`, and `curriculum_events.jsonl` into the checkpoint directory. Re-running the same command resumes from `latest.pt` and the saved curriculum phase. Use a fresh checkpoint directory when changing reward modes.
Trainer-side eval is disabled by default inside chunks; promotion is handled by the curriculum gate after each chunk.

For the default overnight setup, this wrapper is equivalent:

```bash
python3 rl_orbit_wars/run_overnight.py
```

If the BC checkpoint is missing on the VM, `run_overnight.py` will create a modest one first using `hellburner` as the teacher and `random,nearest,baselines/starter` as data-collection opponents.

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

- Feed the learned IRL reward model into PPO as an optional auxiliary reward.
- Add target ETA features directly into the action/pair head.
- Port the wrapper to the Rust engine once `orbit_wars_rust` is built locally.
- Keep tuning the entity transformer with LR warmup/decay and per-head entropy logging.
