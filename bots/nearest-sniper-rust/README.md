# Rust Bot

A bot whose `main.py` is a thin wrapper that delegates decision-making to a
native Rust extension. The Python side only marshals the observation; the Rust
side (`src/lib.rs`) picks the moves.

Right now the Rust agent reproduces the open-source
[`nearest-sniper`](../_open_source/nearest-sniper/) baseline exactly, so this is
a drop-in replacement that returns identical moves — the Rust code is the place
to evolve a stronger strategy from here.

## Build

The native module (`rust_bot_native`) must be compiled into the active
environment before `main.py` can import it:

```powershell
cd bots/rust_bot
maturin develop --release
```

## Run

```powershell
python run_match.py rust_bot random
```

## Submit to Kaggle

Kaggle runs agents on Linux x86_64 and won't compile Rust, so the submission
must carry a precompiled Linux `.so`. `build_submission.py` cross-builds one in
the official manylinux Docker image, extracts it, and bundles it with `main.py`:

```powershell
python bots/rust_bot/build_submission.py
kaggle competitions submit orbit-wars -f bots/rust_bot/submission.tar.gz -m "rust_bot v1"
```

The bundled `.so` is `abi3` (Python ≥ 3.9), so it doesn't need to match Kaggle's
exact interpreter version — only the OS/arch. `main.py` adds its own directory to
`sys.path`, so the bundled module imports regardless of the harness's cwd.
