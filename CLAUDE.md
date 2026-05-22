# Context

We are writing bots to play Orbit Wars, a 2 or 4 player game by Kaggle.

## Repository Structure

| Path | Contents |
|---|---|
| [README.md](README.md) | Orbit Wars gameplay rules |
| [agents.md](agents.md) | Build, test, and submit an agent to Kaggle |
| [bots/_open_source/](bots/_open_source/) | Open-source bots used as training opponents |
| [rust_engine/](rust_engine/) | Native simulator: parity-faithful single-env API ([README](rust_engine/README.md)) |
| [parity/](parity/) | Lockstep harness comparing the Rust engine against the Kaggle reference |
| [run_match.py](run_match.py) | Run a single Kaggle-engine match between two bots |
