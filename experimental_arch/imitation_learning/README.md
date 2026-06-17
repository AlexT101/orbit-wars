# Imitation Learning

This directory contains the policy imitation-learning tooling used by
`bots/mine/chaos`. Aphrodite's `bots/mine/aphrodite/train` scripts build value
net data; they do not train the IL policy that Chaos injects as root candidates.

## What Exists

- `train.py` trains the transformer behavior-cloning policy from a chunked
  `.npz` manifest.
- `play_checkpoint.py` runs a checkpoint directly in 2p or 4p games for quick
  policy smoke tests.
- `build_dataset.py` is the older local 2p Isaiah replay builder.
- `build_dataset_from_zips.py` streams `ladder_replays/*.zip` and writes the
  manifest/chunk format consumed by `train.py`. It supports both 2p and 4p.

## Isaiah-Specific 4p Workflow

To clone Isaiah directly for Chaos candidate injection, filter by exact player
name instead of using a top-N cohort and drop noop rows:

```powershell
.\venv\Scripts\python.exe experimental_arch\imitation_learning\build_dataset_from_zips.py `
  --zip ladder_replays\replays_*.zip --players 4 `
  --player-name "Isaiah @ Tufa Labs" --launch-only `
  --out-dir experimental_arch\imitation_learning\data\isaiah_tufa_labs_4p_launches
```

Add `--winner-only` if you want only Isaiah wins. Without it, the dataset uses
every Isaiah 4p game, including losses.

## 4p Replay-Zip Workflow

First build a strong-player allowlist from the same replay zips:

```powershell
.\venv\Scripts\python.exe bots\mine\aphrodite\train\elo_topn.py `
  --zip ladder_replays\replays_6_*.zip --players 4 --top-n 20 `
  --out experimental_arch\imitation_learning\data\top20_4p.json
```

Then stream policy samples from the zips:

```powershell
.\venv\Scripts\python.exe experimental_arch\imitation_learning\build_dataset_from_zips.py `
  --zip ladder_replays\replays_6_*.zip --players 4 `
  --keep-players experimental_arch\imitation_learning\data\top20_4p.json `
  --out-dir experimental_arch\imitation_learning\data\osteo_top20_4p
```

Train a separate 4p checkpoint:

```powershell
$env:IL_DATASET_PATH = "experimental_arch\imitation_learning\data\osteo_top20_4p\manifest.json"
$env:IL_OUT_DIR = "experimental_arch\imitation_learning\checkpoints\osteo_bc_transformer_4p"
$env:IL_DATASET_NAME = "osteo_top20_4p"
$env:IL_WANDB_RUN_NAME = "osteo-bc-transformer-4p"
.\venv\Scripts\python.exe experimental_arch\imitation_learning\train.py
```

Smoke-test the checkpoint directly:

```powershell
.\venv\Scripts\python.exe experimental_arch\imitation_learning\play_checkpoint.py `
  --players 4 `
  --checkpoint experimental_arch\imitation_learning\checkpoints\osteo_bc_transformer_4p\latest.pt `
  --opponent hellburner --num-games 5 --no-render
```

To let Chaos inject that checkpoint in 4p, point it at the checkpoint and opt in:

```powershell
$env:CHAOS_IL_CHECKPOINT = "experimental_arch\imitation_learning\checkpoints\osteo_bc_transformer_4p\latest.pt"
$env:CHAOS_IL_PLAYERS = "4"
```
