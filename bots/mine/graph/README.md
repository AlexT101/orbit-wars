# graph

A native Rust Orbit Wars agent in the spirit of [`apollo2`](../apollo2), with one
key difference: its **model of the environment is `env_model`**
([`experimental_arch/env_model`](../../../experimental_arch/env_model), crate
`orbit_wars_model`) rather than a vendored engine clone. The bot depends on
`orbit_wars_model` as a Rust library (it ships `crate-type = ["cdylib", "rlib"]`)
and builds an `EngineState` forward model straight from the observation.

## Why

This bot is the seed for an RL setup where we train with `env_engine` and use
`env_model` as the test-time forward model. Sharing the *exact same* physics in
the bot and the trainer avoids train/test skew.

Its distinctive feature is the **pairwise planet distance matrix**, the intended
RL input feature. Every turn it computes the full NxN distance matrix between
planets (using env_model's `distance`) and:

- exposes it to Python via `Bot.distances_matrix(obs)` →
  `{"planet_ids": [...], "matrix": [[...], ...]}`, and
- overlays the planet adjacency graph as debug, drawing one `[LINE]` between
  each pair of planets colored by distance (green = near, yellow = mid,
  red = far), a `[DOT]` node at each planet, and a `[TEXT]` summary — the same
  `[LINE]`/`[DOT]`/`[TEXT]` grammar apollo2 uses, so the viewer overlay just
  works.

The move logic is currently the **nearest-sniper baseline** — a placeholder
until the RL policy replaces it.

## Build

```bash
cd bots/mine/graph
maturin develop --release
cd ../../..
```

You MUST rebuild after changing this crate **or** `env_model`, since the latter
is compiled in as a dependency.

## Test

```bash
python run_match.py graph nearest-sniper
```
