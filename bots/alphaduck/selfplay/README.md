# alphaduck self-play loop

AlphaZero-style: generate self-play games, build v17 chunks, retrain, swap weights.

## Files

- `play_game.py` — runs one self-play game, dumps kaggle-format JSON.
- `batch_play.py` — bounded-concurrency wrapper that spawns N games and zips them.
- `loop.py` — orchestrator: A) play → B) build chunk → C) train → D) swap weights.

## Output dirs

- `games/{iter}/g{seed}.json` — raw replays
- `chunks/{iter}.zip` — zipped jsons for build_dataset_v17
- `chunks/{iter}.npz` — v17 training chunk
- `weights_history/{iter}.pt` — every trained checkpoint
- `logs/loop_{ts}.log` — run log

## Phases never overlap (RAM safety)

Each iter runs Phase A → B → C → D **sequentially**. Workers are killed before
training starts so peak RAM = max(phase_peak), not sum.

## Defaults by host

| host | flags |
| --- | --- |
| Mac M4 16 GB | `--workers 2 --games 50 --buffer 5 --train-epochs 3 --device cpu` |
| EC2 c-class CPU 30 GB | `--workers 4 --games 200 --buffer 5 --train-epochs 3 --device cpu` |
| EC2 g-class GPU L4 | `--workers 4 --games 100 --buffer 5 --train-epochs 5 --device cuda` |

`--device mps` is rejected by torch path setup but DO NOT use it on Mac — MPS
is 22× slower than CPU for PairNetV17 (32k small-kernel launches/batch).

## How to run (manual)

```bash
# Mac, smoke test (one iter, 10 games)
python3 bots/alphaduck/selfplay/loop.py --iters 1 --games 10 --workers 2 --buffer 1

# EC2 GPU, ongoing
python3 bots/alphaduck/selfplay/loop.py --iters 50 --games 100 --workers 4 --device cuda --train-epochs 5
```

## How to resume

`--start-iter N` skips iters < N. Chunks are tagged by iter # so existing
chunks are picked up by the rolling window.

## Memory model

- One alphaduck process ≈ 1.5 GB RSS (Python + torch + model + chunks of MCTS state).
- `env.run([A, B])` spawns A + B + a parent → ~3 GB per game subprocess.
- Trainer ≈ 3 GB (one chunk loaded + model).

Phase peaks on a 16 GB Mac with `--workers 2`:
- Phase A: ~6 GB (2 games × 3 GB)
- Phase B: ~3 GB (build_dataset_v17 single chunk)
- Phase C: ~3 GB (trainer on 5 chunks via memmap-style)

Leaves ~10 GB headroom for the OS. The orchestrator also reads
`memory_pressure` (mac) / `free -g` (linux) before each phase and aborts if
available RAM is below `--min-ram-gb`.

## Known limits

- `play_game.py` uses kaggle_environments which forks processes per agent.
  The env's `actTimeout` default is 1 s/turn; bump with `--act-timeout 0.5`
  on slow boxes if you want shorter games for higher throughput.
- v17 chunks are large (~50 MB compressed per 50 games). 50 iters × 50 games
  → ~12 GB on disk over the run; `--keep-chunks 10` evicts older ones.
- This loop does NOT record MCTS visit distributions — it imitates the
  greedy action like build_dataset_v17 does from replays. To convert to true
  AlphaZero (visit-count policy targets), modify alphaduck/main.py to dump
  per-turn root visit counts; not done here.
