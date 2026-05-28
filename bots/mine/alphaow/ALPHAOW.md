# alphaow

A Rust Orbit Wars bot: **Decoupled-UCT (DUCT) simultaneous-move MCTS**, using
apollo's *hellburner* planner both to generate move candidates and to drive
rollouts, with a **learned value network** evaluating leaf states. This doc
describes the base bot and its two self-play variants (`alphaow_roll`,
`alphaow_noroll`). Everything below was verified against the source, not memory.

---

## 1. How a turn runs (process model)

`main.py` (`bots/mine/alphaow/main.py`) is the Kaggle entry point. It spawns the
Rust binary **once** and pipes one JSON observation per turn:

- Binary path: `$ALPHAOW_BOT_BIN`, else `<dir>/target/release/alphaow-bot`. If
  missing, it runs `cargo build --release` once (`_build_if_needed`).
- **Value-net default:** if the caller hasn't set `ALPHAOW_VALUE_NET_PATH`,
  `_ensure()` points it at `train/weights/v2_replays.bin` (relative to the bot
  dir) via `env.setdefault`. This is the deploy fix — without it the loader has
  *no* default path and the bot silently falls back to the duck heuristic.
- Per turn: normalize the obs (`_norm`), JSON-encode, write a line to the
  binary's stdin, read one line back, decode as the moves array. On a broken
  pipe it drops the proc and returns `[]`.

The Rust side (`src/main.rs`) is a stdin/stdout daemon: read one JSON obs line →
`parse_state` → `duct::best_move(&state, state.player, budget_ms)` → emit a JSON
array of `[fleet_id, angle, ships]` moves (filtered to the current player).
Budget is `$ALPHAOW_BUDGET_MS`, default **500 ms**.
(`ALPHAOW_DUMP_FEATURES_PATH` optionally dumps raw + summary_v2 features per turn
for training data collection.)

---

## 2. The search: DUCT (`src/duct.rs`)

Decoupled UCT models the game as a true simultaneous-move game. At each node
both players commit actions privately, then the joint action is applied. Each
player selects via **PUCT on their own marginal stats** (summed over what the
opponent might do); backprop updates both players' marginals plus the joint
child. Tree depth is half that of sequential MyTurn/EnemyTurn MCTS, and there's
no opp-sees-my-action info leak.

**Per node** (`Node`): the `GameState`, my/opp candidate action-lists, their
priors and per-candidate stats (`visits`, `sum_value` from MY perspective), and
a `HashMap<(my_idx, opp_idx) -> child>` of joint subtrees.

**Selection** (`select_my` / `select_opp`): PUCT,
`exploit + c·prior·√parent_N/(1+n)`. `c` = `OW_PUCT_C`, default `0.3`. Opp negates
exploit (it minimizes MY value). Priors are rank-based:
`rank_prior(i) = 0.5^i / Σ 0.5^k` — the greedy candidate gets the most prior.

**Candidates** (`ensure_candidates` → `enumerate_alternatives`): K candidates per
player (`OW_K_ROOT`=5 at root, `OW_K_NON_ROOT`=4 elsewhere). The opponent is
`dominant_enemy` (highest-score non-me player; handles 4P). By default candidates
come from **apollo's `strategy::search_candidates`** (`apollo_bridge::apollo_candidates`);
`OW_APOLLO_CANDIDATES=0` falls back to the ow2 enumerators (`enumerate_alternatives_strong`,
which is the greedy ow2 plan plus per-target exclusion variants, or `fast` via
`OW_DUCT_ENUMERATE=fast`).

**Expansion + rollout** (`select_and_expand`): on first visit to a joint
`(my_idx, opp_idx)`, apply both action sets, `tick` once, create the child, and
`rollout` from there. On revisits, recurse. Terminal when `step >= 500` or
`alive_players <= 1`.

**Rollout** (`rollout`) — *not* play-to-terminal:
- Runs up to `depth` ticks of a fixed policy, then returns `evaluate(state)` (the
  value net). The leaf value is the **net's estimate**, not a win/loss.
- `OW_ROLLOUT` mode default `ow2_full`, `OW_ROLLOUT_DEPTH` default **8** for that
  mode (`none`=0, `fast`=30, `ow2_short`=2, `ow2_fast`=12).
- **Rollout policy:** by default apollo's hellburner plan for both me and the
  dominant enemy (`apollo_bridge::apollo_plan`), with an `EntityCache` built once
  at the leaf and reused across ticks (orbiter geometry is fixed; the cache is
  only refreshed when a comet spawns/expires). `OW_APOLLO_ROLLOUT=0` falls back
  to `ow2_fast` (fast policy) or `ow2_full` (`ow2_plan::plan`).
- Early-exits the tick loop when one side holds <5% of total ships (with >30
  total) — a blowout is already decided.
- Rollout noise defaults **OFF** (`OW_ROLLOUT_NOISE`=0); it regressed DUCT 2-4
  because the same noisy value is applied to both marginals.

**Final move** (`best_move`): the candidate with the highest mean marginal value
(raw max; the robust-margin override defaults off — `OW_MARGIN`=0, because forcing
stay-with-greedy regressed 0-6). Runs until the time budget or 100k iters.

**Subtree reuse:** after each turn the root is stashed (`LAST_TREE`); next turn,
if the observed state hashes to one of the prior root's joint children, that
subtree is reused. `OW_NO_REUSE` disables it.

---

## 3. Leaf evaluation: the value net (`src/value_net.rs`)

`evaluate_inner` (in duct.rs) is the leaf eval:

```
if use_value_net() and predict(state, me) -> Some(v):
    v_scaled = (v * value_scale()).clamp(-1, 1)
    if value_blend() >= 1.0:  return v_scaled            # default — net only
    else: return blend*v_scaled + (1-blend)*heuristic     # heuristic = 15-tick lookahead
else: return heuristic
```

- `OW_VALUE_NET=0` forces the duck heuristic (`mcts::evaluate_external`, which
  ticks `OW_EVAL_LOOKAHEAD`=15 turns then scores).
- `OW_VALUE_BLEND` default **1.0** → at the default, the 15-tick heuristic
  contributes **nothing** and is skipped entirely (saves ~60µs/leaf). The blend
  knob only matters if you set it < 1.
- `OW_VALUE_SCALE` default 1.0 — multiplicative damping on the net output.

**The net** (`predict`): output is a scalar in `[-1, 1]` = MCTS value from MY
perspective. The loader (`load_weights`) reads `ALPHAOW_VALUE_NET_PATH`; if unset
it logs and returns `None` (→ heuristic, **no default path in Rust** — main.py
supplies the default).

**Weight file format** — little-endian, magic `0x564F4157` ("AOWV"):
- **v1** (single hidden layer): `[magic, version=1, input_dim, hidden, w1, b1, w2, b2]`.
  Lifted into a 2-layer stack input→hidden (ReLU) → 1 (tanh).
- **v2** (deep): `[magic, version=2, input_dim, n_layers, out_dims[n], (w,b) per layer]`.
  Arbitrary dense stack; ReLU on every layer but the last, tanh on the final
  out_dim-1 layer.

**Input feature sets** (auto-detected from `input_dim`):
- `Full` (2728-d): two-stream per-object `[is_me,is_opp,is_neutral, log1p(ships),
  radius, is_static,is_orbit,is_comet, production]` for ≤44 objects, current +
  extrapolated, plus a 44×44 pairwise distance matrix.
- `Summary` (23-d): handcrafted scalar aggregates (ship/planet/production totals,
  pressure terms, frontline distance, log-ratio).
- `SummaryV2` (**46-d**, current production net): per-player blocks for *me* and
  *dominant enemy* — 10 current features (ships on planets, ships flying, static/
  orbit/comet counts + productions, neutrals/enemies-closer-to-me) + 9 extrap
  features (same minus ships_flying) — plus an 8-d neutral block (ships, counts,
  productions, comet time-remaining). Layout: `[me_cur 10][opp_cur 10][me_ext 9]
  [opp_ext 9][neut 8]`.

"Extrapolated" = resolve all in-flight fleets onto their predicted target planets
(simple arrival-ordered combat, **no** production added) — "what the board looks
like once current flights land."

Forward pass uses a NEON-accelerated dot product on aarch64 (Apple Silicon),
scalar 8-accumulator fallback elsewhere. Inference cost is negligible
(~17µs/predict even at 512×512 ≈ 0.1% of a 500ms budget); the bot is
rollout-bound, not net-bound.

**Production net:** `train/weights/v2_replays.bin` — AOWV **v1**, SummaryV2
(input 46), hidden 64, arch `46->64->1`. This is the net behind the 8-0 / 7-1
results vs apollo_fast.

---

## 4. The three variants

All three run the **same** `alphaow-bot` binary; they differ only in the env the
wrapper pins. The relevant binary switch is `apollo_rollout_enabled()`
(duct.rs:282): **default TRUE** unless `OW_APOLLO_ROLLOUT ∈ {0, false, off}`.

| Variant | `OW_APOLLO_ROLLOUT` | Rollout policy | Notes |
|---|---|---|---|
| **alphaow** (base) | *unset* → defaults ON | apollo hellburner | Production bot. main.py also defaults the net path + builds if needed. |
| **alphaow_roll** | `"1"` (explicit ON) | apollo hellburner | Behaviorally identical to base default. Self-play wrapper. |
| **alphaow_noroll** | `"0"` (OFF) | ow2 (`ow2_full` plan) | Self-play wrapper; isolates the value of apollo-in-rollout. |

- **`alphaow`** — the deployable bot. `main.py` resolves the binary, builds if
  needed, and `env.setdefault`s `ALPHAOW_VALUE_NET_PATH` →
  `train/weights/v2_replays.bin`. Leaves `OW_APOLLO_ROLLOUT` unset, so apollo
  rollout is ON by default. This is what ships to Kaggle.

- **`alphaow_roll`** (`bots/mine/alphaow_roll/main.py`) — a thin self-play
  wrapper. It hardcodes absolute `_BIN` and `_NET` paths into the base
  `alphaow/` tree, then in `_ensure()` sets `env["OW_APOLLO_ROLLOUT"]="1"` and
  `env["ALPHAOW_VALUE_NET_PATH"]=_NET`. Because base alphaow already defaults
  rollout ON, `alphaow_roll` is functionally the same bot — it exists so it can
  run head-to-head against `alphaow_noroll` in one Python process without racing
  on global `os.environ`. No build step, no path resolution.

- **`alphaow_noroll`** (`bots/mine/alphaow_noroll/main.py`) — identical to
  `alphaow_roll` except `_ROLLOUT="0"` → `OW_APOLLO_ROLLOUT=0`. This turns OFF
  apollo-in-rollout, so rollouts use the ow2 planner instead. Used to measure how
  much apollo's planner-in-the-loop actually buys during rollouts. Candidate
  generation (the apollo hellburner child policy) is **unaffected** — that's the
  separate `OW_APOLLO_CANDIDATES` switch, left at its default ON.

Net effect: **roll vs noroll isolates exactly one factor** — whether rollouts use
apollo's hellburner (roll) or the ow2 plan (noroll). Both use apollo candidates,
the same binary, and the same value net.

---

## 5. Key tuning env vars (all have sane defaults)

| Var | Default | Effect |
|---|---|---|
| `ALPHAOW_BUDGET_MS` | 500 | Per-turn think time. |
| `ALPHAOW_VALUE_NET_PATH` | (set by main.py) | Path to AOWV weights; unset → heuristic. |
| `OW_APOLLO_ROLLOUT` | on | apollo planner in rollouts (the roll/noroll switch). |
| `OW_APOLLO_CANDIDATES` | on | apollo hellburner as the child candidate policy. |
| `OW_ROLLOUT` / `OW_ROLLOUT_DEPTH` | `ow2_full` / 8 | Rollout mode + tick depth. |
| `OW_VALUE_NET` | on | `0` forces the duck heuristic leaf eval. |
| `OW_VALUE_BLEND` | 1.0 | Net↔heuristic mix; 1.0 = net only (skips 15-tick step). |
| `OW_VALUE_SCALE` | 1.0 | Multiplicative damping on net output. |
| `OW_K_ROOT` / `OW_K_NON_ROOT` | 5 / 4 | Candidates per player at root / interior. |
| `OW_PUCT_C` | 0.3 | PUCT exploration constant. |
| `OW_MARGIN` | 0.0 | Robust-child override margin (off — raw max wins). |
| `OW_ROLLOUT_NOISE` | 0.0 | Leaf-value jitter (off — regressed DUCT). |
| `OW_NO_REUSE` | off | Disable cross-turn subtree reuse. |
| `OW_DEBUG` / `OW_PROFILE` | off | Per-turn debug line / eval-vs-rollout timing breakdown. |
