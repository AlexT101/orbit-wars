# Phase 3 design note — neutral-capture discipline

Status: **spec, not yet implemented.** Build only AFTER the phase-2 scoring study
(`apollo_scoring_2p`) finishes and a scoring config is adopted — this is a Rust
change (recompile), which would invalidate any running study.

## Problem

In replays apollo holds a slight total-ship lead (e.g. 450 vs 420), then sinks a
large fleet into a **high-garrison neutral** whose production takes many turns to
repay. We come out behind on ships and struggle to keep up.

Root cause: the **live path is greedy** (`main.py` → `compute_moves` →
`strategy::plan`; the rollout / `score_snapshot` global metric is NOT used). The
only thing valuing a capture is the per-target `timeline_delta_score`, which is
**planet-local** — it scores one target against that planet's own do-nothing
baseline and never compares our *total* ships to the enemy's, nor sees what the
enemy does with the tempo while we're sunk into a garrison. `− ships_committed`
is only a local proxy for the drain.

The phase-2 scoring knobs won't fix this on their own: against the current
training opponents the search actually trends toward *more* aggression (the
leading trial had `score_w_ship_cost ≈ 0.5`, `capture_min_score ≈ −2.5`), because
those opponents don't punish over-extension.

## The four new constants (a "neutral discipline" block)

All default to **identity no-ops** so the build reproduces current play until
tuned. Implemented inside `timeline_delta_score` (strategy.rs ~:668-695), which
already has `target`, `production`, `ships_committed`; it additionally needs the
global ship lead (see Plumbing below).

Definitions (per neutral target):
```
G          = target.ships            # garrison ≈ ships we lose taking it
p          = target.production
payback    = G / p                   # turns of the planet's own output to repay; p==0 ⇒ ∞
excess     = max(0, payback − neutral_payback_turns)
lead       = my_total_ships − strongest_enemy_total_ships   # planets + in-flight fleets
lead_after = lead − G
neutral_at_arrival = (baseline owner at the capture turn == −1)
```

### 1. Payback-ratio surcharge (3 constants)
Targets the high-garrison / slow-payoff pathology. Applied only when the target
is neutral-at-arrival:
```
if neutral_payback_penalty > 0 and excess > 0 and lead_after < lead_gate:
    score -= (w_ship_cost · ships_committed) · neutral_payback_penalty · excess
```
i.e. effective capture cost becomes `w_ship_cost·ships_committed·(1 + penalty·excess)`.
A large-enough surcharge drops the score under `capture_min_score` → natural veto,
no separate skip path.

- **`neutral_payback_turns`** — turns-to-recoup threshold. Default 999 (moot at
  penalty 0). Range ~[4, 40].
- **`neutral_payback_penalty`** — surcharge steepness per excess-turn.
  **Default 0.0 ⇒ exact no-op.** Range ~[0, 0.3].
- **`lead_gate`** — if we'd stay ahead by ≥ this many ships *after* the buy,
  waive the surcharge (we can afford the transient deficit). Default 1000
  (conservative: rarely waives). Range ~[0, 120]. `lead_after ≥ lead_gate`
  already implies "currently ahead", so it's one comparison.

**No exponent needed.** The penalty is linear in `excess` but multiplied by
`ships_committed` (large precisely for high-garrison planets), so the absolute
penalty already accelerates with garrison size — convexity for free. Add an
exponent later only if it's still too soft.

### 2. Flat neutral admission penalty (1 constant)
Targets a *different* failure mode: marginal low-value neutrals (the "+2-score
grab that burns a turn and a fleet"). Applied to ranking AND the gate (so it
deprioritizes, not just gates):
```
if neutral_at_arrival:
    score -= neutral_capture_penalty
```
- **`neutral_capture_penalty`** — flat score reduction for neutral captures.
  Default 0.0 (no-op). Range ~[0, 30] (score units ≈ production·(h−arrival)).

Why it's NOT redundant with `capture_min_score`: the global gate hits neutrals
and enemies equally, and `score_enemy_capture_bonus` (multiplicative) can only
make enemy captures *more* attractive — it can never reject a positive-score
neutral while keeping a marginal enemy capture. The flat penalty is the only knob
that yields **two independent additive admission thresholds**:
`neutral bar = capture_min_score + penalty`, `enemy bar = capture_min_score`.
It cancels in neutral-vs-neutral comparisons, so it only bites at the
neutral-vs-enemy choice and the admission bar. **Unconditional** (not lead-gated)
to keep it orthogonal to the payback surcharge.

## Locked judgment calls
- **Payback surcharge and flat penalty apply to NEUTRALS only.** Enemy captures
  also deny the enemy's production (double value) and are handled by
  `score_enemy_capture_bonus`.
- **`lead` = my total − strongest single enemy total** (planets + in-flight
  fleets). Works for 2p (the one enemy) and 4p.
- **Flat neutral penalty is unconditional** (lead_gate governs only the payback
  surcharge).

## Plumbing
Greedy scoring has never looked at global state. Add it once per `plan()` call:
compute `my_total_ships` and per-enemy totals (planets + owned fleets), derive
`strongest_enemy_total`, stash the resulting `ship_lead` on `WorldState` for
`timeline_delta_score` to read. Cheap (one pass over planets+fleets).

## Tuning plan: ALL-AT-ONCE (10 tuned, 1 pinned)

**Decision (supersedes the earlier 7/3 staged split):** tune the 6 phase-2
scoring constants AND the 4 new neutral constants together in one study.
Rationale: freezing any of the phase-2 constants risks baking in a neutral/enemy
*compromise* (e.g. `capture_min_score` was optimized as a shared bar; the new
neutral penalty should let it re-settle toward its enemy-optimal value). A staged
freeze strands that — joint tuning lets every coupled knob re-optimize together.

→ **10 tuned** (`score_w_ship_cost`, `score_w_final_ships`,
`score_per_ship_smoothing`, `capture_min_score`, `score_enemy_capture_bonus`,
`default_strategy` + the 4 neutral) **, 1 pinned** (`score_w_production = 1.0`).

Cost of all-at-once: a higher-dimensional, more correlated landscape → **~200-300
trials**, multivariate TPE (already on), and **warm-start is mandatory** — seed
the current phase-2 study's best config + the 4 neutral constants at identity, so
trial 0 reproduces current play (upside-only).

### Normalization: considered and DROPPED
We worried the additive score-unit constants (`capture_min_score`,
`neutral_capture_penalty`) "float" as the weights (`w_final_ships`,
`w_ship_cost`) move, and considered normalizing the gate (e.g.
`score > capture_min_score · production`). **Rejected**, for two reasons:
1. That specific normalizer is *backwards* — it raises the bar for high-production
   planets, the ones we most want.
2. All-at-once tuning makes it unnecessary: multivariate TPE co-adapts the
   additive constants to whatever weights it picks in the *same* trial, so the
   float self-resolves. Breakeven (`score = 0 ↔ capture_min_score = 0`) is
   scale-invariant anyway, and the constants live near breakeven.
   The gate acts on **raw score before ranking**, and the ranking key is pure
   argmax, so ScoreFirst's wide range never needs a strategy-conditional interval.
Conclusion: keep `capture_min_score` and `neutral_capture_penalty` as plain
absolute score-unit constants; just give them **generous intervals** so the
jointly-optimal value is reachable. (The payback surcharge is already
ships-relative, so unit-consistent with the cost term.)

Identifiability note (still holds): co-tuning `capture_min_score` +
`neutral_capture_penalty` isn't a degenerate ridge — the *neutral* bar is `c + P`
but the *enemy* bar is `c` alone, so enemy outcomes pin `c` and neutral outcomes
pin `P`.

## Eval-signal caveat
A restraint knob won't get tuned toward restraint unless an opponent *punishes*
over-extension. producer_v2 / apollo_baseline / apollo_tuned apparently don't.
Add a tougher eval opponent (aphrodite / prometheus) or an objective term that
rewards keeping the lead, or phase 3 may just re-learn aggression. See
[[apollo-tuning-plan]], [[apollo-scoring-tunables]].

---

# Phase 4 (SPECULATIVE) — online opponent modeling

**Treat this skeptically.** It is an unproven direction with real failure modes
(below) and it breaks apollo's stateless-per-turn invariant. It is recorded here
because, *done well*, it could be a significant edge on a diverse ladder — not
because it is justified yet. **Do not build it before the de-risking experiment.**

## Idea
Track the live opponent's behavior during a game and adapt some constants to it,
instead of using a single offline-tuned value. Motivating example: infer the
ratio at which the opponent commits to attacking a planet, then feed that into
the pressure model (`ally_pressure_ratio`, `enemy_offset_lookahead`, etc.) for a
better-calibrated threat estimate against *this* opponent.

## Why be skeptical (especially of the "attack ratio" instance)
- **Censored estimation.** We only observe attacks the opponent *did* launch
  (ratios above their threshold); declined attacks are invisible. The min
  observed attack ratio is only an upper bound on the threshold; recovering the
  lower bound means modeling "could have attacked but didn't" every turn
  (expensive, model-dependent). One-sided samples → nasty threshold estimation.
- **Tiny samples per game** (~5-15 attack decisions before the game is decided);
  variance is largest early, when it matters most.
- **Strong opponents aren't ratio-machines.** A scalar threshold fits weak
  heuristic bots; the MCTS / apollo-like bots where the payoff is highest are
  state-dependent and won't reduce to a constant — abstraction fits worst where
  it would help most.
- **Nonstationarity:** behavior shifts by phase (expand early, aggress late).
- **Architecture cost:** breaks "re-plan every turn, no carried state." And
  aphrodite calls apollo for move-gen inside DUCT/MCTS — those are *hypothetical*
  rollouts, so any online model must update on REAL turns only or it gets
  polluted by simulated futures.

## The less-bad version, if pursued
Not a from-scratch threshold estimator — **Bayesian shrinkage on a cheap, robust,
uncensored aggregate stat**:
`estimate = blend(tuned_prior, observed, weight = f(samples_seen))`.
Early game uses the tuned prior; lean on observations as they accumulate. Prefer
stats that are cheap and unambiguous (opponent ship/production growth rate;
reinforcement speed of a threatened planet; neutral-vs-player target preference)
over a censored threshold. New constants then = the prior + the blend/learning
rate, both tunable with existing infra. Needs a DIVERSE/changing eval — the
current 3 fixed training opponents wouldn't reward adaptivity at all.

## De-risk FIRST (cheap, do before any stateful code)
Measure the ceiling: with the existing tuner, fit a SEPARATE config per opponent
(2-3 distinct bots), then compare to the single blended config.
- Configs **differ a lot** → a fixed policy leaves points on the table → online
  adaptation has headroom worth chasing.
- Configs **similar** → a well-tuned fixed policy is near-optimal → adaptation is
  complexity/risk for little gain; drop the idea.
The per-opponent divergence *is* the upper bound on what any online learner could
buy. Costs a few tuner runs; tells you whether the direction is worth it before
writing a line of stateful code.

## Sequencing
After the offline wins are exhausted (phase 2 + phase 3): they're cheaper,
lower-risk, and don't touch the stateless invariant or the aphrodite/MCTS path.
Bank those, run the divergence test, and only then consider this.
