# Orbit Wars Rust Engine

Native simulator for Orbit Wars, used for exact single-environment stepping for parity and debugging

The native Python module is `orbit_wars_rust`.

## APIs

### `RustEngineCore` — parity / debugging

Single-environment API that mirrors the Kaggle game contract.

- `reset(seed, num_players, configuration=None) -> dict`
- `step(actions) -> dict`
- `snapshot() -> dict`

Action format: `[[from_planet_id, angle_radians, ships], ...]` per player.

Use this for parity checks and deterministic debugging.

## Build

```powershell
cd rust_engine
cargo test                # parity + unit tests
maturin develop --release # install Python extension into the venv
```