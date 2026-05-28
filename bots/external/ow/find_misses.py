"""Find ow fleets that get launched but never reach a planet.

For each fleet ow launches, walk forward through env.steps. Classify outcome:
  - HIT_PLANET: fleet disappears + a combat happens on some planet that turn.
  - SUN: fleet's last known segment crosses the sun's 10-radius bubble.
  - OOB:  fleet's last known segment exits [0, 100].
  - TIMEOUT: fleet stops appearing for unknown reason (probably swept by moving planet).

Usage: python ow/find_misses.py [--opp main.py|random] [--seeds 1 2 3 ...]
"""

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OW_BOT_DIR", os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make  # noqa: E402

BOARD = 100.0
SUN = (50.0, 50.0)
SUN_R = 10.0


def fleet_speed(ships, max_speed=6.0):
    if ships <= 1:
        return 1.0
    s = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return max(1.0, min(max_speed, s))


def point_to_segment(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0:
        return math.hypot(p[0] - v[0], p[1] - v[1])
    t = max(0.0, min(1.0, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    px = v[0] + t * (w[0] - v[0])
    py = v[1] + t * (w[1] - v[1])
    return math.hypot(p[0] - px, p[1] - py)


def swept_pair_hit(A, B, P0, P1, r):
    d0x = A[0] - P0[0]
    d0y = A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return t2 >= 0.0 and t1 <= 1.0


def hit_any_planet(prev_obs, next_obs, fleet_old, fleet_new):
    """Check if fleet hits any planet during this turn's segment."""
    planets_old = {p[0]: p for p in prev_obs.planets}
    planets_new = {p[0]: p for p in next_obs.planets}
    for pid, p_old in planets_old.items():
        p_new = planets_new.get(pid, p_old)
        old_pos = (p_old[2], p_old[3])
        new_pos = (p_new[2], p_new[3])
        if old_pos[0] < 0 and new_pos[0] < 0:
            continue
        r = p_old[4]
        if swept_pair_hit(fleet_old, fleet_new, old_pos, new_pos, r):
            return pid
    return None


def find_misses(env, my_player):
    """Yield dicts describing each ow fleet outcome."""
    fleet_history = {}  # fleet_id -> list of (step, x, y, ships, angle)
    fleet_launched = {}  # fleet_id -> (launch_step, from_id, angle, ships)
    for step_idx in range(len(env.steps)):
        state = env.steps[step_idx]
        obs = state[my_player].observation
        # Index fleets at this step
        my_fleets_now = {f[0]: f for f in obs.fleets if f[1] == my_player}
        # Detect new launches (fleet_id present now but not in prev step)
        if step_idx > 0:
            prev_obs = env.steps[step_idx - 1][my_player].observation
            prev_ids = {f[0] for f in prev_obs.fleets if f[1] == my_player}
            for fid, f in my_fleets_now.items():
                if fid not in prev_ids:
                    fleet_launched[fid] = {
                        "launch_step": step_idx,
                        "from_planet": f[5],
                        "angle": f[4],
                        "ships": f[6],
                    }
        # Record positions
        for fid, f in my_fleets_now.items():
            fleet_history.setdefault(fid, []).append(
                (step_idx, f[2], f[3], f[6], f[4])
            )
    # Classify each fleet
    results = []
    last_step = len(env.steps) - 1
    for fid, hist in fleet_history.items():
        launch = fleet_launched.get(fid)
        if launch is None:
            continue  # was already in flight at step 0 — skip
        last = hist[-1]
        last_step_seen = last[0]
        x, y, ships, angle = last[1], last[2], last[3], last[4]
        speed = fleet_speed(ships)
        nx = x + speed * math.cos(angle)
        ny = y + speed * math.sin(angle)
        # On step (last_step_seen + 1), this fleet is gone. What happened?
        # Check the combat lists of step (last_step_seen+1): hits any planet?
        # We don't have direct access; check whether any planet's owner/ships
        # changed in a way consistent with our fleet arriving, OR whether
        # any later-step planet at this fleet's position got bumped.
        # Simpler: check geometry.
        outcome = "UNKNOWN"
        hit_pid = None
        if last_step_seen < last_step:
            # Look at the turn it disappeared (last_step_seen+1 = the turn
            # that processed its final movement). Check planet hits first
            # (engine checks planets before sun/oob).
            disappear_step = last_step_seen + 1
            if disappear_step < len(env.steps):
                prev_obs_local = env.steps[last_step_seen][my_player].observation
                next_obs_local = env.steps[disappear_step][my_player].observation
                hit_pid = hit_any_planet(prev_obs_local, next_obs_local, (x, y), (nx, ny))
        if hit_pid is not None:
            outcome = "HIT_PLANET"
        elif point_to_segment(SUN, (x, y), (nx, ny)) < SUN_R:
            outcome = "SUN"
        elif not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
            outcome = "OOB"
        else:
            if last_step_seen >= last_step:
                outcome = "STILL_FLYING"
            else:
                outcome = "UNKNOWN"
        results.append({
            "fid": fid,
            "launch_step": launch["launch_step"],
            "from": launch["from_planet"],
            "ships": launch["ships"],
            "angle": launch["angle"],
            "last_pos": (x, y),
            "next_pos": (nx, ny),
            "last_step": last_step_seen,
            "outcome": outcome,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="main.py")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 123, 2024, 9999])
    ap.add_argument("--limit", type=int, default=5, help="show up to N failures per seed")
    args = ap.parse_args()

    for seed in args.seeds:
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        random.seed(seed)
        env.run(["ow/main.py", args.opp])
        results = find_misses(env, my_player=0)
        sun = [r for r in results if r["outcome"] == "SUN"]
        oob = [r for r in results if r["outcome"] == "OOB"]
        hit = [r for r in results if r["outcome"] == "HIT_PLANET"]
        flying = [r for r in results if r["outcome"] == "STILL_FLYING"]
        unknown = [r for r in results if r["outcome"] == "UNKNOWN"]
        print(f"=== seed={seed} ({len(results)} ow fleets) ===")
        print(f"  HIT_PLANET={len(hit)}  SUN={len(sun)}  OOB={len(oob)}  "
              f"STILL_FLYING={len(flying)}  UNKNOWN={len(unknown)}")
        for r in (sun + oob + unknown)[:args.limit]:
            print(f"    {r['outcome']} fid={r['fid']} launched step={r['launch_step']} "
                  f"from_planet={r['from']} ships={r['ships']} angle={r['angle']:.3f} "
                  f"last_pos=({r['last_pos'][0]:.1f},{r['last_pos'][1]:.1f}) "
                  f"next=({r['next_pos'][0]:.1f},{r['next_pos'][1]:.1f}) "
                  f"last_step={r['last_step']}")


if __name__ == "__main__":
    main()
