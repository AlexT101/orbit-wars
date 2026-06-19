# chaos

Aphrodite's DUCT search + XGB value net, with the osteo IL transformer policy
injected as extra root candidates. The idea: the IL policy (LB ~1250, close to
aphrodite's ~1300) proposes moves apollo's heuristics can never generate, while
the search and value net veto its tactical blunders (the failure mode that
makes search-free osteo-il-latest exploitable).

## How it works

Per turn, `main.py`:

1. Runs the osteo IL transformer once (~70ms, CPU) and takes its top-k actions
   above a probability floor (deduped, noop dropped).
2. Sends the normal aphrodite payload to the **aphrodite binary** with one
   extra field: `"il_candidates": [[from_id, angle, ships], ...]`.

On the Rust side (`bots/mine/aphrodite/src/duct.rs::inject_root_candidates`),
the IL actions are appended to the root's candidate set after deduping against
apollo's plans. Priors are rebuilt as a virtual interleave (apollo#0, il#0,
apollo#1, il#1, …) with sqrt(0.5) decay per slot, which preserves apollo's
existing 0.5-per-rank prior ratios exactly. Appending (rather than reordering)
keeps reused-subtree `children` indices valid.

The field is optional: aphrodite's own wrapper never sends it, so the shared
binary behaves identically for aphrodite.

Scope (v1): root node only (per-node IL is ~1000x too slow), my side only.
IL injection runs in 2p games only. A native 4p game plays as pure aphrodite (no
IL); if it decays to two surviving players, the 2p checkpoint loads lazily on
that turn and IL resumes. (4p IL was tried and regressed chaos, so the 4p Isaiah
policy is no longer bundled or loaded.)

**Failures are loud:** a missing checkpoint, stale `orbit_wars_model` schema,
IL runtime error, or dead binary raises immediately. If chaos is playing, the
IL injection is provably active.

**Time budgeting** is dynamic per turn: the wrapper times its IL pass and
sends the binary `budget_ms = target - il_elapsed - 100` (floor 250ms) in the
payload, overriding the binary's env budget. The source default is a
conservative 600ms target for dev runs; `build_submission.py` flips prod limits
on, giving Chaos a 900ms target. Aphrodite's Rust panic clamp also caps the
effective search budget at 900ms when the remaining overage pool is low.

## Requirements

- The aphrodite binary built from `bots/mine/aphrodite` (auto-built on first
  run if cargo is available).
- A fresh `orbit_wars_model` install (tokens `(4,44,15)`, pair outcomes
  `(44,44,3,4)`): `cd experimental_arch/env_model && maturin build --release`
  then pip install the wheel.
- The IL checkpoint at
  `experimental_arch/imitation_learning/checkpoints/osteo_bc_transformer/latest.pt`.

## Env knobs

| Var | Default | Meaning |
|---|---|---|
| `CHAOS_IL_K` | 4 | max IL candidates injected per turn |
| `CHAOS_IL_MIN_PROB` | 0.02 | drop IL suggestions below this policy prob |
| `CHAOS_IL_SKIP_TURNS` | 8 | skip IL injection on the first N turns, matching the tested Apollo-only opening |
| `CHAOS_TORCH_THREADS` | 2 | torch / OpenMP intra-op threads |
| `CHAOS_TURN_TARGET_MS` | 600 dev / 900 submission | total per-turn wall target (IL + search) |
| `CHAOS_IL_CHECKPOINT` | repo 2p checkpoint | override the 2p IL checkpoint path |

`OW_DEBUG=1` prints per-turn `[chaos]` (wrapper: IL mode, ms + candidates) and
`[chaos-il]` (Rust: offered/added/root_K) lines to stderr.

## Opening shortcuts (why IL / eval might look "disabled")

Two independent knobs deliberately bypass IL and/or DUCT on the first few turns.
If injection or search looks like it isn't running early in a game, check both
before assuming a bug — they live in **different layers**:

| Knob | Layer | Default | Effect |
|---|---|---|---|
| `CHAOS_IL_SKIP_TURNS` | this wrapper (env var, `main.py`) | 8 | skip the IL forward + injection for steps `< N`, matching the tested Apollo-only opening. `0` = inject from step 0. |
| `APOLLO_ONLY_FIRST_TURNS` | aphrodite binary (`const` in `src/duct.rs`) | 8 | skip DUCT search + leaf eval for steps `< N` and play apollo's top candidate directly. **Compile-time constant — needs a rebuild, not an env var.** `0` = search from step 0. |

They are **not** coupled in code, but the checked-in defaults are both `8` from
testing: the opening plays pure Apollo and avoids spending IL work on turns whose
candidates the binary would ignore. If you change one, consider whether the other
should change too.

With `OW_DEBUG=1`, an apollo-only turn prints `[duck-apollo-only]` (instead of
`[duck]`) and an IL-skipped turn shows `il=2p:skip` in the `[chaos]` line.

## 4p IL training

The policy IL tooling lives under `experimental_arch/imitation_learning`.
`build_dataset_from_zips.py` can stream `ladder_replays/*.zip` into the chunked
manifest format consumed by `train.py`; see that directory's README for the
full 4p workflow. Note: a 4p IL policy was trained and wired into chaos but
regressed it, so chaos no longer bundles or loads a 4p checkpoint — this tooling
remains for future retraining only.

## Future work

- Kaggle `build_submission.py` (bundle: aphrodite binary + torch + checkpoint +
  `orbit_wars_model` .so — crib from triplepoint1 + aphrodite).
- IL priors from actual policy probabilities instead of rank interleave.
- Opponent-side IL candidates (second inference per turn).
- IL value head as a mixing term in leaf evaluation.
