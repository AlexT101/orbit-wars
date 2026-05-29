# env_model

Rust forward model for Orbit Wars with Python bindings. Same physics as
`rust_engine/`, **minus** the seeded RNG, planet generation, and comet
spawning. Designed for "load an arbitrary state, step forward" — the
common shape for tree search, ground-truth fleet resolution, or fast RL
rollouts.

Module name: `orbit_wars_model`. Class: `OrbitWarsModel`.

## What it does NOT do

- Generate planets at game start. There is no `reset(seed)`; you must
  load state via `set_state(obs_dict)`.
- Spawn new comets at step boundaries 50 / 150 / 250 / 350 / 450. Those
  spawns use a seed kaggle scrubs from the observation, so we can't
  reproduce them. `step` is still callable on those turns — it just
  doesn't add new comets, so the model diverges from the live env after
  any crossed boundary. Resync via `set_state` if you need parity.

Already-spawned comets ARE stepped along their precomputed paths.

## API

```python
from orbit_wars_model import OrbitWarsModel

model = OrbitWarsModel(num_players=2, configuration={"shipSpeed": 6.0,
                                                     "episodeSteps": 500})

# Load from a kaggle observation. `next_fleet_id` is optional; if missing
# it defaults to max(fleet.id) + 1 (or 0 if no fleets).
model.set_state(obs_dict, num_players=2, configuration=cfg)

# Step. Actions are [[from_id, angle_rad, ships], ...] per player.
result = model.step([player0_moves, player1_moves])
# -> {"observations": [obs_p0, obs_p1], "done": bool}

# Lower-overhead variant (skips per-player obs dicts):
result = model.step_fast(actions)
# -> {"done": bool}

# No rewards here — env_model is for forward simulation only (e.g. tree
# search, ground-truth fleet resolution). For training rewards, see
# `env_engine/`.

state = model.get_state()      # full snapshot dict
model.done                     # property
model.step_count               # property
```

## Build

```bash
cd env_model
maturin build --release
pip install --force-reinstall --no-deps \
  target/wheels/orbit_wars_model-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl
```

## Validate against kaggle

```bash
python env_model/validate.py --seeds 5 --steps 499
```

For each step, the script: snapshots kaggle's obs, loads it into the
model, runs the same actions in both, and compares post-step state. Step
transitions where `kaggle.step + 1 ∈ {50, 150, 250, 350, 450}` are NOT
compared (model's `step()` is still called there — it just isn't expected
to match because kaggle spawns comets and we don't). Last verified:
**2470 / 2470 checks pass** across 5 full 499-step games.

## Benchmark

```bash
python env_model/benchmark.py --steps 5000
```

Measures raw step throughput. Sample numbers (this machine):

| Mode    | Kaggle env    | Rust model     | Speedup |
|---------|--------------:|---------------:|--------:|
| noop    |    343 step/s |  805,075 step/s | 2347×  |
| fleets  |    221 step/s |   38,480 step/s |  175×  |

`fleets` is slower than `noop` for the model because computing actions
requires `model.get_state()` each step (serializes the full state into
Python). The simulator itself is faster than the dict construction.
