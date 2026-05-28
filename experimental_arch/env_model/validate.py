import math
import random
import sys
from orbit_wars_model import OrbitWarsModel
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make


SEEDS, STEPS, PLAYERS = 5, 999999, 2
BASE_SEED, ACTION_SEED = 1, 12345
TOL = 1e-9

# comet spawns need to be skipped - our env model cannot predict it
# SPAWN_STEPS = set()
SPAWN_STEPS = {50, 150, 250, 350, 450}


def actions(obs, n, rng):
    out = [[] for _ in range(n)]
    for pid, owner, *_, ships, _ in obs.get("planets", []):
        if 0 <= owner < n and ships >= 2 and rng.random() < 0.30:
            out[owner].append([int(pid), rng.uniform(-math.pi, math.pi),
                              max(1, int(ships * rng.uniform(0.25, 0.75)))])
    return out


def diff(rs, ko):
    errs = []
    if rs["step"] != ko["step"]:
        errs.append(f"step: {rs['step']} vs {ko['step']}")
    for tag, key in [("planet", "planets"), ("fleet", "fleets")]:
        rf = {int(x[0]): x for x in rs[key]}
        kf = {int(x[0]): x for x in ko[key]}
        if rf.keys() != kf.keys():
            errs.append(f"{tag} ids: {sorted(rf)} vs {sorted(kf)}")
        names = ["id", "owner", "x", "y", "radius", "ships", "production"] if tag == "planet" else [
            "id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
        for i in rf.keys() & kf.keys():
            for j, n in enumerate(names):
                rv, kv = rf[i][j], kf[i][j]
                bad = abs(float(rv)-float(kv)) > TOL if isinstance(rv,
                                                                   float) or isinstance(kv, float) else rv != kv
                if bad:
                    errs.append(f"{tag}[{i}].{n}: {rv} vs {kv}")
    return errs


total_checked = total_skipped = total_failures = 0
for i in range(SEEDS):
    seed = BASE_SEED + i
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(PLAYERS)
    cfg = {"episodeSteps": env.configuration.episodeSteps,
           "shipSpeed": env.configuration.shipSpeed}
    model = OrbitWarsModel(num_players=PLAYERS)
    rng = random.Random(ACTION_SEED + i)
    checked = skipped = failures = 0
    for _ in range(STEPS):
        if env.done:
            break
        pre = dict(env.state[0].observation)
        acts = actions(pre, PLAYERS, rng)
        model.set_state(pre, num_players=PLAYERS, configuration=cfg)
        env.step(acts)
        model.step(acts)
        if (pre["step"] + 1) in SPAWN_STEPS:
            skipped += 1
            continue
        errs = diff(model.get_state(), dict(env.state[0].observation))
        checked += 1
        if errs:
            failures += 1
            print(f"  step {pre['step']+1}: {errs[:3]}")
    total_checked += checked
    total_skipped += skipped
    total_failures += failures
    print(f"seed={seed}  checked={checked}  skipped={skipped}  failures={
          failures}  [{'OK' if not failures else 'FAIL'}]")

print(f"---\nTOTAL: checked={total_checked}  skipped={
      total_skipped}  failures={total_failures}")
