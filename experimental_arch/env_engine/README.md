# env_engine

Self-contained Rust Orbit Wars engine with **configurable reward
shaping**, intended as a drop-in replacement for the kaggle env during
training. Unlike `env_model/`, this one ships its own seeded RNG and
generates planets + comets internally, so a full game runs without any
kaggle dependency.

Module: `orbit_wars_engine`. Class: `OrbitWarsEngine`.

## When to use which

| | env_model | env_engine |
|---|---|---|
| API | `set_state(obs)` + `step` | `reset(seed)` + `step` |
| Comet spawning | ✗ (no RNG) | ✓ (bit-exact to kaggle) |
| Rewards | ✗ | ✓ shaped + components |
| Use at | **test time** (tree search, fleet resolution) | **training** (PPO rollouts) |

## API

```python
from orbit_wars_engine import OrbitWarsEngine

engine = OrbitWarsEngine(
    num_players=2,
    configuration={"shipSpeed": 6.0, "episodeSteps": 500, "cometSpeed": 4.0},
    reward_weights={
        "terminal":          1.0,    # centered ships-share at game end
        "terminal_time":     1.0,    # zero-sum outcome x remaining-fraction
        "production_income": 0.0002, # per-step centered absolute production
        "launch_penalty":   -0.00004, # per successful launch
    },
)

# Start a new game. The seed controls planet layout AND comet spawns —
# bit-identical to `make("orbit_wars", configuration={"seed": seed})`.
result = engine.reset(seed=42)
obs_p0 = result["observations"][0]

# Step. Returns weighted scalar reward per player + a per-component
# breakdown for logging.
result = engine.step([player0_moves, player1_moves])
# {
#   "observations": [obs_p0, obs_p1],
#   "done": bool,
#   "reward": [r0, r1],
#   "reward_components": {
#     "terminal":          [..., ...],
#     "terminal_time":     [..., ...],
#     "production_income": [..., ...],
#     "launch_penalty":    [..., ...],
#   },
# }

# Faster variant (skips per-player observation dicts):
result = engine.step_fast(actions)   # {"done", "reward"}

# Tune shaping mid-experiment without restarting the game:
engine.set_reward_weights({"production_income": 0.0005})

state = engine.get_state()
engine.done
engine.step_count
```

### Reward math

For each player `i`:

```
baseline          = 1 / num_players
ships_share_i     = own_ships_i / Σ_j own_ships_j        # in [0, 1], sums to 1
mean_prod         = Σ_j own_prod_j / num_players         # players only, per step
launches_i        = successful launches by player i      # per step
outcome_i         = zero-sum win/loss value              # ties are 0
reward_i = w_terminal          * (ships_share_i - baseline)  # terminal turn only
         + w_terminal_time     * outcome_i * remaining_fraction
         + w_production_income * (own_prod_i - mean_prod)    # every step
         + w_launch_penalty    * launches_i                  # every step
```

For two players, `outcome_i` is `+1` for the winner and `-1` for the loser.
Ties are zero for every player. For more than two players, the winner set sums
to `+1` and the loser set sums to `-1`. The terminal terms are 0 on
non-terminal turns; `production_income` is applied every step. Each component
except `launch_penalty` is centered, so the reward sums to the launch penalties
across players.

## Build

```bash
cd env_engine
maturin build --release
pip install --user --force-reinstall --no-deps \
  target/wheels/orbit_wars_engine-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl
```

## Validate against kaggle

```bash
python env_engine/validate.py
```

Resets both engines with the same seed and steps both with the same
actions, comparing planets + fleets + comet path indices at every turn.
Last verified: **2495 / 2495 checks pass** across 5 full 499-step games
including all 5 comet spawn boundaries.

## Benchmark

```bash
python env_engine/benchmark.py
```

Sample numbers (this machine, 2-player, 50k engine steps vs 1k kaggle steps):

| Mode    | Kaggle env    | Rust engine     | Speedup |
|---------|--------------:|----------------:|--------:|
| noop    |    236 step/s |  ~40,000 step/s | ~165×   |
| fleets  |    252 step/s |  ~20,000 step/s |  ~80×   |

### Why slower than env_model?

The **steady-state per-step cost is identical to env_model** (~1-3 µs).
The throughput gap is entirely from the 5 comet-spawn boundaries per
game: `generate_comet_paths` runs up to 300 ellipse-validation attempts,
each evaluating a 5000-point swept path against every orbiting planet —
typically 3-4 ms per spawn. Averaged across 500-step games:

```
5 spawns × ~3.4 ms ÷ 500 steps ≈ 34 µs/step amortized overhead
```

Most steps run at ~600k sps; spawn-boundary steps cost ~3-4 ms each.
Tradeoff is intentional: bit-exact parity with kaggle. If a future
training run is comet-spawn-bound, add a `disable_comets` flag.

`fleets` is slower than `noop` because computing actions requires
`engine.get_state()` each step (state→Python dict round-trip). For
training, the python policy already needs the obs anyway, so the
`step()`-returns-observations path is the natural one.
