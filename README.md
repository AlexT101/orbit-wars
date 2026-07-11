# Overview

Competed in [Kaggle's Orbit Wars competition](https://www.kaggle.com/competitions/orbit-wars) as team "pantheon of ducks", placing 74th out of 4,729 teams (10,196 entrants) and earning a competition silver medal.

Our final submission can be found at `bots/mine/chaos`.

## Development

Though we are required to use Python for the entrypoint to our bot, all of our core logic is in Rust. Anytime the code changes, the Rust module **needs to be recompiled** like so:

```powershell
cd bots/mine/apollo
maturin develop --release
cd ../../..
```

## Submission

To compile the final bot for submission, run:

```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
python bots/apollo/build_submission.py
```

And then submit the generated `submission.tar.gz` file.

## Testing

You can run a single match using:

```powershell
python run_match.py starter starter
```

Add the `--kaggle` flag to use the Kaggle engine instead of the ported Rust engine. Seed is random by default unless you add a flag like `--seed 42`. You can also use the `run_batched.py` or `run_batched_4p.py` scripts to run many matches in parallel.

You can run the local visualizer using:

```bash
docker compose up
```

Then open <http://localhost:6001>.

## Repository Structure

| Path | Contents |
|---|---|
| [orbit-wars-lab/](orbit-wars-lab/) | Local visualizer with tournament and gauntlet modes |
| [orbit_wars_rules.md](orbit_wars_rules.md) | Orbit Wars gameplay rules |
| [agents.md](agents.md) | Build, test, and submit an agent to Kaggle |
| [bots/](bots/) | Various agents including `/mine`, `/baseline`, and `/external` (open source bots) |
| [rust_engine/](rust_engine/) | Rust-native port of the Kaggle engine ([README](rust_engine/README.md)) |
| [engine_parity_checker/](engine_parity_checker/) | Harness comparing ported Rust engine to Kaggle engine |
| [run_match.py](run_match.py) | Run a single Kaggle-engine match between two bots |
