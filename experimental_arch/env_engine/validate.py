"""Validate env_engine against kaggle's reference env.

Unlike env_model, env_engine has its own seeded RNG, so it generates
planets and spawns comets bit-identically to kaggle. We can therefore
validate the FULL game including comet spawn boundaries.
"""
import contextlib
import io
import math
import random
from orbit_wars_engine import OrbitWarsEngine
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make


SEEDS, STEPS, PLAYERS = 5, 999999, 2
BASE_SEED, ACTION_SEED = 1, 12345
TOL = 1e-9


def actions(obs, n, rng):
    out = [[] for _ in range(n)]
    for pid, owner, *_, ships, _ in obs.get("planets", []):
        if 0 <= owner < n and ships >= 2 and rng.random() < 0.30:
            out[owner].append([int(pid), rng.uniform(-math.pi, math.pi),
                              max(1, int(ships * rng.uniform(0.25, 0.75)))])
    return out


def diff(es, ko):
    errs = []
    if es["step"] != ko["step"]:
        errs.append(f"step: {es['step']} vs {ko['step']}")
    for tag, key in [("planet", "planets"), ("fleet", "fleets")]:
        ef = {int(x[0]): x for x in es[key]}
        kf = {int(x[0]): x for x in ko[key]}
        if ef.keys() != kf.keys():
            errs.append(f"{tag} ids: {sorted(ef)} vs {sorted(kf)}")
        names = ["id", "owner", "x", "y", "radius", "ships", "production"] if tag == "planet" else [
            "id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
        for i in ef.keys() & kf.keys():
            for j, n in enumerate(names):
                rv, kv = ef[i][j], kf[i][j]
                bad = abs(float(rv)-float(kv)) > TOL if isinstance(rv, float) or isinstance(kv, float) else rv != kv
                if bad:
                    errs.append(f"{tag}[{i}].{n}: {rv} vs {kv}")
    # Comet path index is engine-internal — compare separately if both have comets.
    eg = {tuple(g["planet_ids"]): g["path_index"] for g in es.get("comets", []) or []}
    kg = {tuple(g["planet_ids"]): g["path_index"] for g in ko.get("comets", []) or []}
    if eg.keys() != kg.keys():
        errs.append(f"comet groups: {sorted(eg)} vs {sorted(kg)}")
    for k in eg.keys() & kg.keys():
        if eg[k] != kg[k]:
            errs.append(f"comet[{k}].path_index: {eg[k]} vs {kg[k]}")
    return errs


total_checked = total_failures = 0
for i in range(SEEDS):
    seed = BASE_SEED + i
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(PLAYERS)
    engine = OrbitWarsEngine(num_players=PLAYERS)
    engine.reset(seed)
    rng = random.Random(ACTION_SEED + i)
    checked = failures = 0
    for _ in range(STEPS):
        if env.done:
            break
        acts = actions(dict(env.state[0].observation), PLAYERS, rng)
        env.step(acts)
        engine.step(acts)
        errs = diff(engine.get_state(), dict(env.state[0].observation))
        checked += 1
        if errs:
            failures += 1
            print(f"  step {engine.step_count}: {errs[:3]}")
    total_checked += checked
    total_failures += failures
    print(f"seed={seed}  checked={checked}  failures={failures}  [{'OK' if not failures else 'FAIL'}]")

print(f"---\nTOTAL: checked={total_checked}  failures={total_failures}")
