# simbot

A bot whose `main.py` is a thin wrapper that delegates decision-making to a
native Rust extension. The Python side only marshals the observation; the Rust
side (`src/lib.rs`) picks the moves.

Cloned from [`nearest-sniper-rust`](../nearest-sniper-rust/): the agent currently
reproduces the open-source [`nearest-sniper`](../_open_source/nearest-sniper/)
baseline. What makes simbot different is `src/engine.rs` — a vendored clone of
the [`rust_engine`](../../rust_engine/) simulator — so the Rust agent can run
forward simulations in-process and score candidate moves, rather than picking
heuristically. `src/helpers.rs` is a placeholder for the scoring/search code
built on top of it.

## Layout

| File | Role |
|---|---|
| `src/lib.rs` | PyO3 module + `compute_moves` agent entry point |
| `src/engine.rs` | Vendored clone of `rust_engine` for in-bot simulation (resync when the engine changes) |
| `src/helpers.rs` | Strategy helpers (simulation scoring, candidate generation) — empty for now |

## Build

The native module (`simbot_native`) must be compiled into the active
environment before `main.py` can import it:

```powershell
cd bots/simbot
maturin develop --release
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
python bots/simbot/build_submission.py
kaggle competitions submit orbit-wars -f bots/simbot/submission.tar.gz -m "simbot v1"
```

The bundled `.so` is `abi3` (Python ≥ 3.9), so it doesn't need to match Kaggle's
exact interpreter version — only the OS/arch. `main.py` adds its own directory to
`sys.path`, so the bundled module imports regardless of the harness's cwd.
