# simbot

A Rust-native bot. `main.py` is a thin wrapper that instantiates one `Bot` and
forwards every observation to it; all state, parsing, simulation, and
decisions live in the Rust extension (`simbot_native`).

The current strategy is the `nearest-sniper` baseline, but the foundation —
vendored engine, forward simulator, pre-computed entity trajectories, aim
solver, timeline simulator — is in place for a stronger replacement.

## Layout

| File | Role |
|---|---|
| `src/tests` | Contains all test files |
| `src/lib.rs` | PyO3 module: `Bot` pyclass + observation parsing |
| `src/strategy.rs` | Decision layer (currently `nearest_sniper`) |
| `src/entity_cache.rs` | Pre-computed 500-turn position tables per planet/comet |
| `src/helpers.rs` | Aim solver + combat timeline simulator |
| `src/sim_probe.rs` | Forward simulator for lookahead rollouts |
| `src/engine.rs` | Vendored clone of `rust_engine` (resync when the engine changes) |
| `src/constants.rs` | Game and simulation constants |

The `Bot` owns an `EntityCache` across turns: built on the first call, comet
portion refreshed at the engine's comet-spawn steps (`50, 150, 250, 350, 450`).

## Build

The native module (`simbot_native`) must be compiled into the active
environment before `main.py` can import it:

```powershell
cd bots/simbot
maturin develop --release
cd ../..
```

## Run

```powershell
python run_match.py simbot random
```

## Submit to Kaggle

Kaggle runs agents on Linux x86_64 and won't compile Rust, so the submission
must carry a precompiled Linux `.so`. `build_submission.py` cross-builds one in
the official manylinux Docker image, extracts it, and bundles it with `main.py`:

```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
python bots/simbot/build_submission.py
kaggle competitions submit orbit-wars -f bots/simbot/submission.tar.gz -m "simbot v1"
```

The bundled `.so` is `abi3` (Python ≥ 3.9), so it doesn't need to match Kaggle's
exact interpreter version — only the OS/arch. `main.py` adds its own directory to
`sys.path`, so the bundled module imports regardless of the harness's cwd.
