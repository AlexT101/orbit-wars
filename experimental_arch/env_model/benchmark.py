import contextlib
import io
import math
import sys
import time
from orbit_wars_model import OrbitWarsModel
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make

KAGGLE_STEPS = 1000
RUST_STEPS = 10_000
PLAYERS = 2
SEED = 67


def fresh(seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(PLAYERS)
    cfg = {"episodeSteps": env.configuration.episodeSteps,
           "shipSpeed": env.configuration.shipSpeed}
    return dict(env.state[0].observation), cfg, env


def noop(): return [[] for _ in range(PLAYERS)]


def fleet_acts(obs):
    out = [[] for _ in range(PLAYERS)]
    for pid, owner, x, y, _r, ships, _ in obs.get("planets", []):
        if 0 <= owner < PLAYERS and ships >= 2:
            out[owner].append([int(pid), math.atan2(
                y-50., x-50.) + 0.3, int(ships//2)])
    return out


def bench_kaggle(mode, steps):
    obs, _, env = fresh(SEED)
    seed = SEED
    games = completed = 0
    t0 = time.perf_counter()
    while completed < steps:
        if env.done:
            seed += 1
            obs, _, env = fresh(seed)
            games += 1
        env.step(fleet_acts(
            dict(env.state[0].observation)) if mode == "fleets" else noop())
        completed += 1
    return time.perf_counter() - t0, steps, games


def bench_model(mode, steps):
    obs, cfg, _ = fresh(SEED)
    seed = SEED
    model = OrbitWarsModel(num_players=PLAYERS)
    model.set_state(obs, num_players=PLAYERS, configuration=cfg)
    games = completed = 0
    t0 = time.perf_counter()
    while completed < steps:
        acts = fleet_acts(model.get_state()) if mode == "fleets" else noop()
        done = model.step_fast(acts)["done"]
        completed += 1
        if done:
            tp = time.perf_counter()
            seed += 1
            obs, cfg, _ = fresh(seed)
            model.set_state(obs, num_players=PLAYERS, configuration=cfg)
            t0 += time.perf_counter() - tp
            games += 1
    return time.perf_counter() - t0, steps, games


def fmt(dt, steps, games):
    return f"{dt:.3f}s  {steps/dt:>10,.0f} steps/s  ({dt*1e9/steps:.1f} ns/step)  games={games}"


for mode in ("noop", "fleets"):
    bench_kaggle(mode, 200)
    bench_model(mode, 200)  # warmup

for mode in ("noop", "fleets"):
    print(
        f"\n=== {mode} / {PLAYERS}p  kaggle={KAGGLE_STEPS} rust={RUST_STEPS} ===")
    k = bench_kaggle(mode, KAGGLE_STEPS)
    print(f"  kaggle : {fmt(*k)}")
    m = bench_model(mode, RUST_STEPS)
    print(f"  model  : {fmt(*m)}")
    k_nsps = k[0] / k[1]
    m_nsps = m[0] / m[1]
    print(f"  speedup: {k_nsps/m_nsps:.1f}×")
