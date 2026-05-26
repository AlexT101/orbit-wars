# apollo

Apollo is a Rust-first Orbit Wars bot. The Python entrypoint in `main.py` is a
thin wrapper: it imports `apollo_native`, constructs one `Bot`, and forwards
each observation into Rust.

On the Rust side, `Bot::compute_moves`:

1. parses the observation,
2. refreshes a long-lived `EntityCache`,
3. builds a per-turn `WorldState`,
4. runs the `hellburner` planner,
5. returns `[from_planet_id, angle, ships]` moves back to Python.

There is also a slower `compute_moves_with_search` path in `src/lib.rs` for
rollout-based candidate selection, but `main.py` currently uses the faster
`compute_moves` path.

## Layout

| Path | Role |
|---|---|
| `main.py` | Python wrapper that exposes the Kaggle `agent` function |
| `build_submission.py` | Builds a Kaggle-compatible Linux wheel in Docker and bundles a submission tarball |
| `src/lib.rs` | PyO3 module definition, observation parsing, and `Bot` lifecycle |
| `src/hellburner.rs` | Main strategy, ported from the open-source `hellburner` bot |
| `src/rollout.rs` | Rollout/search helpers used by `compute_moves_with_search` |
| `src/world.rs` | Strategy-agnostic per-turn snapshot built from the engine state |
| `src/entity_cache.rs` | Cached planet/comet positions and aim-cache data shared across turns |
| `src/blockers.rs` | Obstacle-aware aiming and collision/blocking logic |
| `src/helpers.rs` | Timeline simulation, aiming helpers, and shared utilities |
| `src/sim_probe.rs` | Lightweight forward-simulation interface for rollout work |
| `src/engine.rs` | Vendored Rust engine used for local simulation and timeline construction |
| `src/tests` | Correctness tests plus ignored throughput/profiling benchmarks |

The cache is built on the first turn from `initial_planets` and refreshed for
new comet groups at the engine's comet spawn steps.

## Build Locally

Apollo uses PyO3 + maturin, so the native extension must be built into your
active Python environment before `main.py` can import it:

```powershell
cd bots/mine/apollo
maturin develop --release
cd ../../..
```

Rebuild after Rust changes or `main.py` will keep importing the previously
compiled extension.

## Test

Run the regular Rust test suite from the bot directory:

```powershell
cd bots/mine/apollo
cargo test
```

Some performance checks are intentionally ignored by default. Useful ones:

```powershell
cargo test --release pure_sim_throughput -- --ignored --nocapture
cargo test --release pure_sim_with_fleets -- --ignored --nocapture
cargo test --release rollout_score_throughput -- --ignored --nocapture
cargo test --release pick_plan_throughput -- --ignored --nocapture
cargo test --release rollout_score_throughput_4p -- --ignored --nocapture
cargo test --release --features profile profile_sim_sections -- --ignored --nocapture
```

## Run Locally

From the repo root:

```powershell
python run_match.py apollo random
```

## Build A Kaggle Submission

Kaggle will not compile Rust during evaluation, so the submission must include
a prebuilt Linux shared object. `build_submission.py` handles that by running a
Docker build inside Kaggle's own Python runtime image, extracting the compiled
`.so` from the wheel, and packaging it with `main.py`.

From the repo root:

```powershell
python bots/mine/apollo/build_submission.py
```

On Windows, make sure Docker Desktop is running first.

The script writes:

- `bots/mine/apollo/apollo_native.abi3.so` (or the platform-equivalent wheel extract)
- `bots/mine/apollo/submission.tar.gz`

Submit with:

```powershell
kaggle competitions submit orbit-wars -f bots/mine/apollo/submission.tar.gz -m "apollo v1"
```

The wheel is built as `abi3` for Python 3.9+, so it does not depend on the
exact Python patch version Kaggle uses. It is still intentionally tied to the
Kaggle Linux runtime image used during the Docker build.
