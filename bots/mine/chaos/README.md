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

Scope (v1): root node only (per-node IL is ~1000x too slow), my side only,
2p games only (the IL net is 2p-trained; 4p runs as pure aphrodite).

**Failures are loud:** a missing checkpoint, stale `orbit_wars_model` schema,
IL runtime error, or dead binary raises immediately. If chaos is playing, the
IL injection is provably active.

**Time budgeting** is dynamic per turn: the wrapper times its IL pass and
sends the binary `budget_ms = target - il_elapsed - 30` (floor 250ms) in the
payload, overriding the binary's env budget. The source default is a
conservative 700ms target for dev runs; `build_submission.py` flips prod limits
on, giving Chaos a 1000ms target while Aphrodite's Rust panic clamp still caps
the effective search budget at 900ms when the remaining overage pool is low.

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
| `CHAOS_IL_K` | 5 | max IL candidates injected per turn |
| `CHAOS_IL_MIN_PROB` | 0.02 | drop IL suggestions below this policy prob |
| `CHAOS_TURN_TARGET_MS` | 700 dev / 1000 submission | total per-turn wall target (IL + search) |
| `CHAOS_IL_CHECKPOINT` | repo checkpoint | override IL checkpoint path |

`OW_DEBUG=1` prints per-turn `[chaos]` (wrapper: IL ms + candidates) and
`[chaos-il]` (Rust: offered/added/root_K) lines to stderr.

## Future work

- Kaggle `build_submission.py` (bundle: aphrodite binary + torch + checkpoint +
  `orbit_wars_model` .so — crib from triplepoint1 + aphrodite).
- IL priors from actual policy probabilities instead of rank interleave.
- Opponent-side IL candidates (second inference per turn).
- IL value head as a mixing term in leaf evaluation.
