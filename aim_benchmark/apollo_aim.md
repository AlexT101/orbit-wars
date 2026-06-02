# Apollo aim benchmark — harness & debug bindings

This folder benchmarks Apollo's aimer against the standalone Orbit Wars aim
benchmark (`aim_benchmark.py` + `aim_samples.npz`, 10k board states scored by the
real Kaggle engine). It mirrors `benchmark-for-aiming-implementation.ipynb` but
runs from the command line.

## Layout

| File | Purpose | Needs debug bindings? |
|---|---|---|
| `aim_benchmark.py` | the dataset loader + engine scorer (`iter_samples`, `validate`, `_hit_planet`) | — |
| `aim_samples.npz` | 10k sampled board states | — |
| `test_apollo_aim.py` | build apollo, aim every sample, score against the engine | no |
| `dump_impossible_failures.py` | dump impossible-bucket shots apollo *didn't* decline, with engine outcomes | no |
| `diagnose_reachable_misses.py` | find + classify the reachable shots apollo misses (lead/blocker/turn breakdown) | **yes** (`aim_diagnose`) |

Run them with the project venv's Python so the freshly built module imports into
the same interpreter, e.g. `venv\Scripts\python.exe aim_benchmark\test_apollo_aim.py`.

## The production binding (`aim_angle`) — kept in `lib.rs`

`apollo_native.aim_angle(obs, source, target, fleet_size) -> float | None` is the
benchmark entry point and **stays in `bots/mine/apollo/src/lib.rs`**. It builds a
one-shot `EntityCache` from the obs snapshot and returns the launch angle (or
`None` to decline). This is what the Kaggle notebook and `test_apollo_aim.py`
call. Nothing to restore for the basic benchmark run.

## Debug bindings (removed from `lib.rs` after the investigation)

Three extra PyO3 functions were used to investigate aim misses. They were removed
to keep `lib.rs` clean. To re-enable deeper diagnosis (e.g. to run
`diagnose_reachable_misses.py`), paste these back into
`bots/mine/apollo/src/lib.rs` **immediately after `fn aim_angle(...)`** (before
`aim_counters_report`), then rebuild with `maturin develop --release` from
`bots/mine/apollo`.

```rust
/// Debug sibling of [`aim_angle`]: returns the full solver result tuple
/// `(angle, turns, target_x, target_y, flight_time)` (or `None`) instead of
/// just the angle, so a harness can compare apollo's predicted intercept point
/// and flight time against the engine's actual fleet trajectory.
#[pyfunction]
#[pyo3(signature = (obs, source, target, fleet_size))]
fn aim_debug(
    obs: &Bound<'_, PyDict>,
    source: i64,
    target: i64,
    fleet_size: i64,
) -> PyResult<Option<(f64, i64, f64, f64, f64)>> {
    let planets = parse_planets(&get_item(obs, "planets")?)?;
    let angular_velocity: f64 = get_item(obs, "angular_velocity")?.extract()?;
    if source_unowned(&planets, source, obs)? {
        return Ok(None);
    }
    let (comets, comet_planet_ids) = match (obs.get_item("comets")?, obs.get_item("comet_planet_ids")?) {
        (Some(c), Some(ids)) => (parse_comets(&c)?, ids.extract::<Vec<i64>>()?),
        _ => (Vec::new(), Vec::new()),
    };
    let cache = EntityCache::build(&planets, &comets, &comet_planet_ids, angular_velocity, obs_current_step(obs)?);
    Ok(crate::aim::aim_with_prediction(&cache, source, target, fleet_size, 0))
}

/// Diagnostic for reachable-miss investigation. Returns
/// `(found_lead, angle, turns, flight_time, blocked_all, blocked_comets_only)`:
///   * `found_lead` — did [`crate::aim::lead_target`] find any intercept turn?
///   * `angle`/`turns`/`flight_time` — that lead (0 if none).
///   * `blocked_all` — does the direct lead path hit any obstacle (sun+planets+comets)?
///   * `blocked_comets_only` — does *only a comet* block the direct path?
/// Lets a harness partition declines into "no intercept", "blocked by comet",
/// and "blocked by sun/planet".
#[pyfunction]
#[pyo3(signature = (obs, source, target, fleet_size))]
fn aim_diagnose(
    obs: &Bound<'_, PyDict>,
    source: i64,
    target: i64,
    fleet_size: i64,
) -> PyResult<(bool, f64, i64, f64, bool, bool)> {
    let planets = parse_planets(&get_item(obs, "planets")?)?;
    let angular_velocity: f64 = get_item(obs, "angular_velocity")?.extract()?;
    let (comets, comet_planet_ids) = match (obs.get_item("comets")?, obs.get_item("comet_planet_ids")?) {
        (Some(c), Some(ids)) => (parse_comets(&c)?, ids.extract::<Vec<i64>>()?),
        _ => (Vec::new(), Vec::new()),
    };
    let cache = EntityCache::build(&planets, &comets, &comet_planet_ids, angular_velocity, obs_current_step(obs)?);
    let v = crate::engine::fleet_speed(fleet_size.max(1), crate::constants::MAX_SHIP_SPEED);
    match crate::aim::lead_target(&cache, source, target, 0, v) {
        None => Ok((false, 0.0, 0, 0.0, false, false)),
        Some((angle, turns, _tx, _ty, ft)) => {
            let blocked = crate::aim::shot_blocked_exact(&cache, source, target, angle, ft, v, 0);
            let comet_only = crate::aim::comet_blocks_path(&cache, source, target, angle, ft, v, 0);
            Ok((true, angle, turns, ft, blocked, comet_only))
        }
    }
}

/// Diagnostic: apollo's blocker verdict for an *explicit* `(angle, flight_time)`
/// shot, returning `(blocked_all, blocked_comets_only)`. Lets a harness ask
/// "would apollo consider this exact angle blocked?" — e.g. testing apollo's
/// verdict on an engine-confirmed working angle to expose a false positive.
#[pyfunction]
#[pyo3(signature = (obs, source, target, fleet_size, angle, flight_time))]
fn blocked_at(
    obs: &Bound<'_, PyDict>,
    source: i64,
    target: i64,
    fleet_size: i64,
    angle: f64,
    flight_time: f64,
) -> PyResult<(bool, bool)> {
    let planets = parse_planets(&get_item(obs, "planets")?)?;
    let angular_velocity: f64 = get_item(obs, "angular_velocity")?.extract()?;
    let (comets, comet_planet_ids) = match (obs.get_item("comets")?, obs.get_item("comet_planet_ids")?) {
        (Some(c), Some(ids)) => (parse_comets(&c)?, ids.extract::<Vec<i64>>()?),
        _ => (Vec::new(), Vec::new()),
    };
    let cache = EntityCache::build(&planets, &comets, &comet_planet_ids, angular_velocity, obs_current_step(obs)?);
    let v = crate::engine::fleet_speed(fleet_size.max(1), crate::constants::MAX_SHIP_SPEED);
    let blocked = crate::aim::shot_blocked_exact(&cache, source, target, angle, flight_time, v, 0);
    let comet_only = crate::aim::comet_blocks_path(&cache, source, target, angle, flight_time, v, 0);
    Ok((blocked, comet_only))
}
```

And register them inside the `apollo_native` `#[pymodule]` (next to the existing
`m.add_function(wrap_pyfunction!(aim_angle, m)?)?;`):

```rust
    m.add_function(wrap_pyfunction!(aim_debug, m)?)?;
    m.add_function(wrap_pyfunction!(aim_diagnose, m)?)?;
    m.add_function(wrap_pyfunction!(blocked_at, m)?)?;
```

`aim_debug`, `aim_diagnose`, and `blocked_at` reuse the helpers `parse_planets`,
`parse_comets`, `get_item`, `source_unowned`, and `obs_current_step`, all of
which remain in `lib.rs` (used by `aim_angle`).

## Build

```powershell
cd bots/mine/apollo
maturin develop --release   # installs apollo_native into the active venv
cd ../../..
```

`test_apollo_aim.py` runs this build step for you (skip with `--no-build`).

## Usage

```powershell
# Full benchmark (build + 10k samples scored by the real engine):
venv\Scripts\python.exe aim_benchmark\test_apollo_aim.py
venv\Scripts\python.exe aim_benchmark\test_apollo_aim.py --no-build   # skip rebuild
venv\Scripts\python.exe aim_benchmark\test_apollo_aim.py --limit 500  # quick subset

# Dump the impossible-bucket shots apollo didn't decline (engine outcome each):
venv\Scripts\python.exe aim_benchmark\dump_impossible_failures.py

# Classify the reachable shots apollo misses (REQUIRES the debug bindings above):
venv\Scripts\python.exe aim_benchmark\diagnose_reachable_misses.py
```

Requires the project venv (numpy + kaggle_environments + maturin). The benchmark
was validated against `kaggle-environments==1.30.1`; the `orbit_wars` env is
byte-identical to 1.29.1.

## Notes on the obs schema the bindings expect

The benchmark obs carries `planets` (rows `[id, owner, x, y, radius, ships, prod]`),
`angular_velocity`, `step`, `player`, optional `comets`/`comet_planet_ids`, and
`initial_planets`. `aim_angle` uses `planets` + `angular_velocity` (+ comets), uses
`player` for the ownership decline, and uses `step` only to pick `current_step`
(0 at game start, else 1 — see the `obs_current_step` doc comment in `lib.rs`).
