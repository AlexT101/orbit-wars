# Adding / removing a SummaryV2 feature

The value net consumes an `N`-dim feature vector (i.e. **N = 41** when this document was written).

XGBoost addresses features **positionally** (`f0..f{N-1}`), and the Rust loader dispatches
purely on vector length (`detect_kind`: `input_dim == DIM` → SummaryV2). So two
invariants must always hold:

1. The training NPZ column order == the order emitted by `extract()`.
2. Every place that parses the binary record knows the new `N`.

A model trained on a different `N` is **incompatible** with a binary built for the
new `N` — you must re-extract and retrain together.

---

## 1. Edit the feature (Rust — the source of truth)

`src/value_net.rs`, module `summary_features_v2`:

- Add/remove the computation in the right block function:
  - `current_player_block` (me & opp, current state)
  - `extrap_player_block` (me & opp, after in-flight fleets land)
  - `neutral_block` (neutral planets)
  Edit the function's return array **and its fixed-size type** (e.g. `[f32; 9]`).
- Update `pub const DIM` to the new `N`.
- Update the slice offsets in `extract()` (the `out[a..b].copy_from_slice(...)`
  lines) so the blocks pack contiguously.

Geometry/positions are on `Planet` (`x, y, orbital_radius, initial_angle,
is_orbiting, is_comet`); engine constants live in `rust_engine/src/lib.rs`.

## 2. Update everything that depends on N

| File | What | Required? |
|---|---|---|
| `src/value_net.rs` | `DIM`, block arrays, `extract()` offsets | **yes** (the change) |
| `train/build_from_zip.py` | `SUMMARY_V2_DIM` (record byte parse) | **yes — silent corruption if stale** |
| `train/collect.py` | `SUMMARY_V2_DIM` (self-play/eval daemon record) | **yes** |
| `train/feature_importance.py` | `FEATURE_NAMES` (must match `extract()` order) | yes (for readable importances) |
| `src/bin/extract_v2.rs` | doc comment (record bytes) | cosmetic |

> The record is `step:i64 + player:i32 + [f32; N]` = `12 + 4N` bytes. Mis-set
> `SUMMARY_V2_DIM` desyncs the stream and parses garbage (e.g. int-overflow in `meta`).

## 3. Build

```bash
cd bots/mine/aphrodite && cargo build --release && cd ../../..
```

## 4. Re-extract + retrain (REQUIRED after any feature change)

`combined.npz` caches **feature vectors**, not raw obs, so a feature change needs a
fresh extraction. Delete the scratch dir first so old-`N` data can't mix in:

```bash
rm -rf bots/mine/aphrodite/train/data/2p/_ladder_work    # and .../4p/... for 4p
```

Note that it takes ~1-2 min/day of data to re-extract, so AI agents should only start this process with user permission.

2p:

```bash
./venv/Scripts/python.exe bots/mine/aphrodite/train/build_ladder.py \
  --replays-dir ladder_replays --players 2 \
  --recency-halflife 7 --rounds 2000 \
  --model-out bots/mine/aphrodite/train/weights/xgb_2p_<tag>.json --keep-temp
```

4p: same with `--players 4` and `--model-out .../xgb_4p_<tag>.json`.

Train to a **candidate** name (`_<tag>`), never straight over the deployed
`xgb_2p.json` / `xgb_4p.json`. Watch the printed val **sign-acc** + **best_iter**
(a sanity filter, not the verdict). With `--keep-temp` you can re-train weighting
variants cheaply off `combined.npz` (see README §2).

Inspect the result:

```bash
./venv/Scripts/python.exe bots/mine/aphrodite/train/feature_importance.py \
  bots/mine/aphrodite/train/weights/xgb_2p_<tag>.json --by total_gain
```

## 5. Eval (the real verdict) + promote

Test the candidate vs the apollo_fast or producer bots. Run at least 20 matches per opponent on unique seeds (don't replay on opposite side of the same seed). Note that changing the number of features makes our old weights incompatible with our model, so we need to rely on past match data against the same candidates, or test before updating features.

Promote by archiving the old weights and renaming the candidate to `xgb_2p.json` / `xgb_4p.json` (README §4).

---

## Appendix: aim-based features (apollo in-place)

The extractor can use apollo's aiming **in place** — no duplication. `apollo` and
`value_net` are the same crate, so call `crate::apollo::*` directly. Features are
computed inside `summary_features_v2::extract_with_cache` (and its per-player
block fns), which already receive `&EntityCache` (the apollo geometry/aim cache).

- **Inference**: duct builds & refreshes one shared `EntityCache` per search and
  threads it in via `predict_with_cache` — cache use is free per leaf.
- **Training**: `extract()` builds a throwaway cache per row (offline, fine).
- **Comets**: the shared inference cache is the root cache, refreshed only on the
  fixed `COMET_SPAWN_STEPS` (50/150/250/350/450). Orbit geometry (`position_abs`
  by id) is always valid; comet-*blocking* aim queries at a deep leaf may be
  slightly stale. Ignore while comets are de-emphasized.

### Resolving a shot — `resolve_shot`

For aim features, call `resolve_shot` directly on the cache. It is the body of
`HellburnerModel::plan_shot` with **none** of the model overhead (the proximity
graph + reinforcement BFS the model builds are used only by the *planner*, never
for aiming), so you skip `WorldState` and `HellburnerModel` entirely:

```rust
use crate::apollo::strategy::resolve_shot;
// `cache` is the &EntityCache already threaded into the block fns.
let shot = resolve_shot(cache, src_id, target_id, ships, launch_turn_offset, None);
```

It still uses the cache's L2 (`aim_cache`) and L3 (invariant) fast paths, which
are shared across the whole search — so repeated static→static shots stay cheap.
Pass `None` for the L1 map (an optional per-caller `ShotL1` memo); the shared
cache already memoizes. You only need a full `WorldState` + `HellburnerModel` for
the *planner's* output (`plan` / `search_candidates`) — see the world-building
pattern in `apollo_bridge::apollo_plan` — never for shots.

`resolve_shot(cache, src_id, target_id, ships, launch_turn_offset, l1) -> Option<AimResult>`

| param | type | meaning |
|---|---|---|
| `cache` | `&EntityCache` | the threaded apollo geometry/aim cache |
| `src_id` | `i64` | shooter planet id |
| `target_id` | `i64` | target planet id |
| `ships` | `i64` | fleet size (clamped to ≥1; larger = faster, affects intercept turn) |
| `launch_turn_offset` | `i64` | turns after the cache's current turn to launch (`0` = launch now) |
| `l1` | `Option<&ShotL1>` | optional per-caller memo; pass `None` (shared cache memoizes) |

Returns `None` when no clear shot exists (sun/planet/comet blocks every angle in
the target's aim cone). On success, `AimResult = (f64, i64, f64, f64, f64)`:

| field | type | meaning |
|---|---|---|
| `.0` angle | `f64` | launch bearing in radians |
| `.1` turns | `i64` | integer intercept turn (turns to impact) |
| `.2` target_x | `f64` | impact point x (target's chord position when the hit fires) |
| `.3` target_y | `f64` | impact point y |
| `.4` flight_time | `f64` | fractional flight time `= turns − 1 + s*`, always `≤ turns` |

Useful derived signals: reachability (`is_some`), time-to-hit (`.1`), and
threat/pressure (loop enemy planets within reach of one of my planets, sum ships
they could land via `plan_shot`). Read-only aim helpers in
`crate::apollo::aim` (`lead_target_from`, `shot_blocked_exact`, …) are also `pub`
if you need lower-level queries; `plan_shot` is the cached, batteries-included
path and should be the default.
