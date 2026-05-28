# duck

Orbit Wars bot using Decoupled UCT (simultaneous-move MCTS) over an
`ow2_plan`-derived candidate policy and rollout policy.

## Build

```
cargo build --release
```

## Run as a Kaggle agent

```python
from kaggle_environments import make
env = make("orbit_wars", configuration={"seed": 1})
env.run(["duck/main.py", "opponent/main.py"])
```

`main.py` spawns `target/release/duck-bot` once and pipes one JSON
observation per turn. Set `DUCK_BOT_BIN` to override the binary path.

## Layout

- `src/main.rs` — daemon (stdin obs → stdout moves).
- `src/duct.rs` — Decoupled UCT search loop.
- `src/mcts.rs` — sequential MCTS variant (also exports `evaluate_external`
  used by duct as the leaf evaluator).
- `src/ow2_plan.rs` — `ow2`-style planner used for candidate enumeration
  and rollouts. Heavily cached: thread-local `dir_to_hit` cache invalidated
  per `state.step`, split TIME/ANGLE arrays sized for L1.
- `src/pathing.rs` — `dir_to_hit` and obstacle-avoidance geometry.
- `src/policy.rs` — sampling policies and `greedy_joint_action`.
- `src/sim.rs` — engine-faithful tick simulation.
- `src/bin/validator.rs` — verifies `sim::tick` matches engine traces.
- `src/bin/bench.rs` — micro-benchmarks for `plan` and rollout costs.

## Tunables (env vars)

- `OW_DUCT_ENUMERATE` — candidate enumeration strategy.
- `OW_ROLLOUT` (default `ow2_full`) — rollout policy.
- `OW_ROLLOUT_DEPTH` — plies per rollout.
- `OW_PUCT_C` — PUCT exploration constant.
- `OW_NO_COOP` — disable cooperation in `ow2_plan`.
- `OW_DEBUG` — per-turn status line to stderr.
