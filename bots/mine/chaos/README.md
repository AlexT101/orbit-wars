# chaos

Chaos wraps Aphrodite and adds imitation-learning policy suggestions to the
root candidate set. The shared Aphrodite binary accepts optional IL fields, so
Aphrodite itself can keep using the same binary without sending IL data.

## Runtime

Per turn, `main.py` may:

1. Run the osteo IL transformer and decode its best legal launch actions.
2. Add `il_candidates`, `il_candidate_probs`, and `il_candidate_indices` to the
   payload sent to the Aphrodite daemon.
3. Send a per-turn `budget_ms` after accounting for wrapper-side IL time.

Rust appends non-duplicate IL actions to the root candidate set in
`bots/mine/aphrodite/src/duct.rs::inject_root_candidates`. The search and value
net then choose among Apollo and IL-rooted plans.

Current runtime IL is enabled only for live two-player states. Three- and
four-player states pass through Aphrodite without IL injection. A four-player
game that later becomes a 1v1 uses the two-player IL path.

The wrapper loads IL checkpoints lazily. Submission builds stage stripped
runtime checkpoints containing only model weights and config.

## Build

Use the Docker builder for Kaggle-compatible native artifacts:

```bash
python bots/mine/chaos/build_submission_docker.py
```

For local development, the wrapper reuses `bots/mine/aphrodite/main.py` for
daemon startup, binary location, and value-net selection.

## Env Knobs

Values live in `main.py`; this table describes behavior only.

| Var | Meaning |
|---|---|
| `CHAOS_IL_K` | max IL candidates injected per turn; `0` disables the IL forward |
| `CHAOS_IL_MIN_PROB` | drop decoded IL suggestions below this policy probability |
| `CHAOS_IL_SKIP_TURNS` | skip IL injection before this step |
| `CHAOS_IL_BUSY_FAIL_MS` | fail if a timed-out IL worker remains busy too long |
| `CHAOS_TORCH_THREADS` | torch / OpenMP intra-op threads |
| `CHAOS_TURN_TARGET_MS` | total per-turn wall target for IL plus search |
| `CHAOS_IL_CHECKPOINT` | override the two-player IL checkpoint path |
| `CHAOS_IL_CHECKPOINT_4P` | override the four-player IL checkpoint path |

`OW_DEBUG=1` prints wrapper timing and candidate details to stderr. Aphrodite's
DUCT debug lines come from the Rust daemon.

## Opening Gates

There are two separate early-turn gates:

| Knob | Layer | Effect |
|---|---|---|
| `CHAOS_IL_SKIP_TURNS` | Chaos wrapper env var | skips the IL forward and root injection |
| `APOLLO_ONLY_FIRST_TURNS` | Aphrodite Rust constant | skips DUCT and leaf eval, then plays Apollo's first candidate |

They are not coupled. If either value changes, update only the source constant
or env var value rather than copying the number into docs.
