"""Validate + benchmark the training rollout against the official kaggle runner.

The training loop only ever sees `env_engine`. The question this answers: if we
take the actions the loop actually produces (our toy policy as player 0, a
`/bots` opponent as player 1) and replay them in kaggle's reference env, do the
two games stay bit-identical? If yes, what env_engine feeds the policy IS what
kaggle would produce — no train/test skew. If no, this prints the first
divergences so we can hunt them down.

It reuses the exact policy / feature / opponent code from `train_loop_example`,
so it tests the real training path, not a reimplementation.

Run (from experimental_arch/):
    python examples/validate_against_kaggle.py
"""

from __future__ import annotations

import contextlib
import io
import time

import torch

from orbit_wars_engine import OrbitWarsEngine

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make

from train_loop_example import (
    SEND_FRACTIONS,
    BotOpponent,
    TinyPolicy,
    moves_for_fraction,
    summarize,
)

# --- config (edit here) ----------------------------------------------------
SEEDS = [1, 2, 3, 4, 5]
OPPONENT = "nearest-sniper"   # deterministic -> reproducible games
PLAYERS = 2
MAX_STEPS = 500
TOL = 1e-9                     # float tolerance for state comparison
BENCH_REPEATS = 10            # env_engine replays for a stable sps number
# ---------------------------------------------------------------------------


def learner_moves(policy: TinyPolicy, obs0: dict) -> list:
    """The exact player-0 action the training loop would emit for this obs."""
    g, D, ids = summarize(obs0, player=0)
    logits = policy(torch.from_numpy(g))
    action = int(torch.distributions.Categorical(logits=logits).sample())
    return moves_for_fraction(obs0, 0, D, ids, SEND_FRACTIONS[action])


def diff(es: dict, ko: dict) -> list[str]:
    """Field-by-field state diff (same scheme as env_engine/validate.py).

    `es` = env_engine.get_state(), `ko` = kaggle player-0 observation. Handles
    rows being tuples (env_engine) or lists (kaggle)."""
    errs = []
    if es["step"] != ko["step"]:
        errs.append(f"step: {es['step']} vs {ko['step']}")
    specs = [
        ("planet", "planets", ["id", "owner", "x", "y", "radius", "ships", "production"]),
        ("fleet", "fleets", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]),
    ]
    for tag, key, names in specs:
        ef = {int(x[0]): x for x in es.get(key, [])}
        kf = {int(x[0]): x for x in ko.get(key, [])}
        if ef.keys() != kf.keys():
            errs.append(f"{tag} ids: {sorted(ef)} vs {sorted(kf)}")
        for i in ef.keys() & kf.keys():
            for j, name in enumerate(names):
                rv, kv = ef[i][j], kf[i][j]
                bad = (abs(float(rv) - float(kv)) > TOL
                       if isinstance(rv, float) or isinstance(kv, float) else rv != kv)
                if bad:
                    errs.append(f"{tag}[{i}].{name}: {rv} vs {kv}")
    # Comet path_index by group (engine-internal ordering ignored).
    eg = {tuple(g["planet_ids"]): g["path_index"] for g in es.get("comets") or []}
    kg = {tuple(g["planet_ids"]): g["path_index"] for g in ko.get("comets") or []}
    if eg.keys() != kg.keys():
        errs.append(f"comet groups: {sorted(eg)} vs {sorted(kg)}")
    for k in eg.keys() & kg.keys():
        if eg[k] != kg[k]:
            errs.append(f"comet[{k}].path_index: {eg[k]} vs {kg[k]}")
    return errs


def run_parity(seed: int, policy: TinyPolicy, opponent: BotOpponent):
    """Drive env_engine and kaggle with identical training-loop actions; diff
    state each step. Returns (checked, failures, recorded_actions)."""
    opponent.reset()
    torch.manual_seed(seed)  # reproducible policy sampling

    engine = OrbitWarsEngine(num_players=PLAYERS)
    eng_obs = engine.reset(seed=seed)["observations"]
    kenv = make("orbit_wars", configuration={"seed": seed}, debug=False)
    kenv.reset(PLAYERS)

    checked = failures = 0
    actions_seq = []
    for _ in range(MAX_STEPS):
        if kenv.done:
            break
        acts = [learner_moves(policy, eng_obs[0]), opponent.act(eng_obs[1])]
        actions_seq.append(acts)

        kenv.step(acts)
        res = engine.step(acts)
        eng_obs = res["observations"]

        errs = diff(engine.get_state(), dict(kenv.state[0].observation))
        checked += 1
        if errs:
            failures += 1
            if failures <= 3:
                print(f"    seed {seed} step {engine.step_count}: {errs[:4]}")
        if res["done"]:
            break
    return checked, failures, actions_seq


def benchmark(seed: int, actions_seq: list):
    """Replay a fixed action sequence through each engine, timing only steps."""
    # env_engine — repeat for a stable number (it's fast).
    t0 = time.perf_counter()
    eng_steps = 0
    for _ in range(BENCH_REPEATS):
        engine = OrbitWarsEngine(num_players=PLAYERS)
        engine.reset(seed=seed)
        for acts in actions_seq:
            engine.step(acts)
            eng_steps += 1
    eng_sps = eng_steps / (time.perf_counter() - t0)

    # kaggle — once (slow).
    kenv = make("orbit_wars", configuration={"seed": seed}, debug=False)
    kenv.reset(PLAYERS)
    t0 = time.perf_counter()
    k_steps = 0
    for acts in actions_seq:
        if kenv.done:
            break
        kenv.step(acts)
        k_steps += 1
    k_sps = k_steps / (time.perf_counter() - t0)
    return eng_sps, k_sps


def main() -> int:
    policy = TinyPolicy()
    opponent = BotOpponent(OPPONENT)

    print(f"Parity: training rollout (policy vs {OPPONENT}) — env_engine vs kaggle")
    total_checked = total_failures = 0
    longest_actions: list = []
    bench_seed = SEEDS[0]
    for seed in SEEDS:
        checked, failures, acts = run_parity(seed, policy, opponent)
        total_checked += checked
        total_failures += failures
        if len(acts) > len(longest_actions):
            longest_actions, bench_seed = acts, seed
        print(f"  seed={seed}  checked={checked}  failures={failures}  "
              f"[{'OK' if not failures else 'FAIL'}]")
    print(f"---\nTOTAL: checked={total_checked}  failures={total_failures}  "
          f"[{'ALL MATCH' if not total_failures else 'DIFFERENCES FOUND'}]")

    print(f"\nBenchmark: replay {len(longest_actions)} steps (seed {bench_seed})")
    eng_sps, k_sps = benchmark(bench_seed, longest_actions)
    print(f"  env_engine: {eng_sps:11,.0f} step/s")
    print(f"  kaggle:     {k_sps:11,.0f} step/s")
    print(f"  speedup:    {eng_sps / k_sps:11,.1f}x")
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
