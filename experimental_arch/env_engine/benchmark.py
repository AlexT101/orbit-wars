import contextlib
import io
import math
import time
from orbit_wars_engine import OrbitWarsEngine
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make

KAGGLE_STEPS = 1000
RUST_STEPS = 50000
PLAYERS = 2
SEED = 67


def noop():
    return [[] for _ in range(PLAYERS)]


def fleet_acts(obs):
    out = [[] for _ in range(PLAYERS)]
    for pid, owner, x, y, _r, ships, _ in obs.get("planets", []):
        if 0 <= owner < PLAYERS and ships >= 2:
            out[owner].append([int(pid), math.atan2(y-50., x-50.) + 0.3, int(ships//2)])
    return out


def bench_kaggle(mode, steps):
    env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
    env.reset(PLAYERS)
    seed = SEED
    games = completed = 0
    t0 = time.perf_counter()
    while completed < steps:
        if env.done:
            seed += 1
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.reset(PLAYERS)
            games += 1
        env.step(fleet_acts(dict(env.state[0].observation)) if mode == "fleets" else noop())
        completed += 1
    return time.perf_counter() - t0, steps, games


def bench_engine(mode, steps):
    engine = OrbitWarsEngine(num_players=PLAYERS)
    seed = SEED
    engine.reset(seed)
    games = completed = 0
    t0 = time.perf_counter()
    while completed < steps:
        acts = fleet_acts(engine.get_state()) if mode == "fleets" else noop()
        done = engine.step_fast(acts)["done"]
        completed += 1
        if done:
            seed += 1
            engine.reset(seed)
            games += 1
    return time.perf_counter() - t0, steps, games


def fmt(dt, steps, games):
    return f"{dt:.3f}s  {steps/dt:>10,.0f} steps/s  ({dt*1e9/steps:.1f} ns/step)  games={games}"


for mode in ("noop", "fleets"):
    bench_kaggle(mode, 200)
    bench_engine(mode, 200)  # warmup

for mode in ("noop", "fleets"):
    print(f"\n=== {mode} / {PLAYERS}p  kaggle={KAGGLE_STEPS} rust={RUST_STEPS} ===")
    k = bench_kaggle(mode, KAGGLE_STEPS)
    print(f"  kaggle : {fmt(*k)}")
    e = bench_engine(mode, RUST_STEPS)
    print(f"  engine : {fmt(*e)}")
    print(f"  speedup: {(k[0]/k[1])/(e[0]/e[1]):.1f}×")
