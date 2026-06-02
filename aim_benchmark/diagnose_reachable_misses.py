"""Find the reachable shots apollo misses and deep-diagnose each one.

REQUIRES the debug bindings (`aim_diagnose`) re-added to apollo's lib.rs — see
apollo_aim.md in this folder. Run with the project venv's python, e.g.
`venv\\Scripts\\python.exe aim_benchmark\\diagnose_reachable_misses.py`.
"""
import contextlib
import copy
import io
import logging
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments  # noqa
import aim_benchmark as ab
import apollo_native

CENTER = 50.0
ROTATION_LIMIT = 50.0
MAX_STEPS = 40


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def hit_with_tick(sample, angle):
    ow = ab._ow()
    o = sample.obs
    seat = int(o["player"]); base = int(o["step"]); fid = int(o["next_fleet_id"]); src = int(sample.source)
    obs = types.SimpleNamespace(
        planets=[list(r) for r in o["planets"]], initial_planets=[list(r) for r in o["initial_planets"]],
        fleets=[], comets=copy.deepcopy(o["comets"]), comet_planet_ids=list(o["comet_planet_ids"]),
        next_fleet_id=fid, angular_velocity=float(o["angular_velocity"]), step=base, player=seat)
    na = max(seat + 1, 2)
    state = [types.SimpleNamespace(observation=(obs if i == 0 else types.SimpleNamespace()),
                                   action=None, status="ACTIVE", reward=0) for i in range(na)]
    state[seat].action = [[src, float(angle), int(sample.fleet_size)]]
    cfg = types.SimpleNamespace(shipSpeed=float(o.get("ship_speed", 6.0)), cometSpeed=4.0,
                                episodeSteps=100000, seed=None, randomSeed=None)
    env = types.SimpleNamespace(configuration=cfg, done=False, info={"seed": 0})
    for tick in range(MAX_STEPS):
        obs.step = base + tick
        pre = {int(p[0]): (int(p[1]), int(p[5]), int(p[6])) for p in obs.planets}
        ow.interpreter(state, env)
        state[seat].action = None
        if not any(int(f[0]) == fid for f in obs.fleets):
            for p in obs.planets:
                pid = int(p[0])
                if pid == src or pid not in pre:
                    continue
                o0, s0, pr0 = pre[pid]
                if int(p[1]) != o0 or int(p[5]) != s0 + (pr0 if o0 != -1 else 0):
                    return pid, tick + 1
            return None, tick + 1
    return None, None


def kind(o, pid):
    cids = set(o.get("comet_planet_ids", []))
    if pid in cids:
        return "comet"
    for r in o["planets"]:
        if int(r[0]) == pid:
            orbit_r = math.hypot(r[2] - CENTER, r[3] - CENTER)
            return "static" if (orbit_r + r[4]) >= ROTATION_LIMIT else "orbiting"
    return "missing"


def prow(o, pid):
    for r in o["planets"]:
        if int(r[0]) == pid:
            return r
    return None


samples = list(ab.iter_samples())
misses = []
for i, s in enumerate(samples):
    if not s.meta.get("reachable", True):
        continue
    a = apollo_native.aim_angle(s.obs, s.source, s.target, s.fleet_size)
    if not (a is not None and ab._hit_planet(s, float(a)) == int(s.target)):
        misses.append((i, s, a))

print(f"remaining reachable misses: {len(misses)}\n")
N = 1440  # 0.25 deg
for n, (i, s, a) in enumerate(misses, 1):
    o = s.obs
    found, lead_angle, lead_turns, ft, blocked, comet_only = apollo_native.aim_diagnose(
        o, s.source, s.target, s.fleet_size)
    sp = prow(o, int(s.source)); tp = prow(o, int(s.target))
    sk, tk = kind(o, int(s.source)), kind(o, int(s.target))
    # dense engine scan: hitting angles + their contact turns
    hits = []
    for k in range(N):
        th = -math.pi + 2 * math.pi * k / N
        if ab._hit_planet(s, th) == int(s.target):
            hits.append(th)
    # cluster hit turns
    turn_set = {}
    closest = None
    if hits:
        closest = min(hits, key=lambda h: abs(wrap_pi(h - lead_angle)))
        for h in (hits[0], hits[len(hits) // 2], hits[-1], closest):
            _, tk2 = hit_with_tick(s, h)
            turn_set[round(math.degrees(h), 1)] = tk2
    nc = len(o.get("comet_planet_ids", []))
    print(f"--- #{n}  idx={i}  ep={s.meta.get('episode_id')} step={s.meta.get('step')} ---")
    print(f"  src {s.source}({sk}) -> tgt {s.target}({tk})  ships={s.fleet_size}  comets_on_board={nc}")
    print(f"  apollo_returned={'None' if a is None else round(a,4)}  "
          f"(decline)" if a is None else f"  apollo_returned={round(a,4)} (wrong angle)")
    print(f"  aim_diagnose: found_lead={found} lead_angle={round(lead_angle,4)} lead_turns={lead_turns} "
          f"ft={round(ft,2)} blocked={blocked} comet_only={comet_only}")
    print(f"  engine hitting angles: {len(hits)} (of {N}, {360/N:.2f}deg)")
    if hits:
        gap = math.degrees(abs(wrap_pi(closest - lead_angle)))
        print(f"    closest hit {round(math.degrees(closest),2)}deg, {round(gap,2)}deg from lead; "
              f"contact turns by angle(deg)->turn: {turn_set}")
    if a is not None:
        _, atick = hit_with_tick(s, float(a))
        ahit = ab._hit_planet(s, float(a))
        print(f"    apollo's angle -> engine hit={ahit} at turn {atick}")
    print()
