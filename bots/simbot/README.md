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
must carry a precompiled Linux `.so`. `build_submission.py` builds one *inside
Kaggle's own runtime image* (`gcr.io/kaggle-images/python`) so the resulting
wheel links against the exact glibc/libstdc++ that the submission worker will
load it with. The script then extracts the `.so` and bundles it with `main.py`:

```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
python bots/simbot/build_submission.py
kaggle competitions submit orbit-wars -f bots/simbot/submission.tar.gz -m "simbot v1"
```

First build is slow — the container downloads the Rust toolchain. Subsequent
builds reuse it via `.cargo-home`/`.rustup-home` caches inside the repo.

The bundled `.so` is `abi3` (Python ≥ 3.9), so it doesn't need to match Kaggle's
exact interpreter version. It *is* tied to Kaggle's exact runtime (built with
`--compatibility off`, emitting a `linux_x86_64` wheel rather than `manylinux`)
— a deliberate trade: non-portable to other distros, but a perfect match for the
only machine that matters.

Kaggle loads `main.py` via `exec()`, which leaves `__file__` undefined, so
`main.py` detects the documented `/kaggle_simulations/agent/` path directly and
inserts it into `sys.path`. Locally it falls back to `__file__`'s directory.
