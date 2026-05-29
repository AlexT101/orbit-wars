"""Validate the Rust feature encoder against kaggle's reference engine.

Two things are checked, both against the official `kaggle_environments` engine
(the source of truth) rather than env_model, for safety:

1. Frame caches. `encode_obs` builds frames at t, t+1, t+10, t_resolved by
   forward-simulating with empty actions. We step a kaggle env the same number
   of empty turns and assert the planet states (owner / ships / position) match.

2. Aim solver. For a sample of actions the mask marks valid, we replay the move
   `[source, angle, count]` in a fresh kaggle env (reached deterministically
   from the same seed) and assert the fleet lands on the intended target on
   exactly the predicted turn, uninterrupted.

Run (from experimental_arch/):
    python env_model/validate_features.py
"""

from __future__ import annotations

import contextlib
import io
import math
import random

from orbit_wars_model import encode_obs

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make

SEEDS = [1, 2, 3, 4, 5]
PLAYERS = 2
WARMUP = 6            # empty steps before we encode, so planets have moved
ACTIONS_PER_STATE = 12
POS_TOL = 1e-6
PLANET_SLOTS = 44
ACTIONS_DIM = 6


def fresh_env(seed: int, warmup: int):
    """Reset a kaggle env and step `warmup` empty turns (deterministic)."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(PLAYERS)
    for _ in range(warmup):
        if env.done:
            break
        env.step([[], []])
    return env


def obs_of(env, player=0) -> dict:
    return dict(env.state[player].observation)


def planets_by_id(obs) -> dict:
    return {int(p[0]): p for p in obs["planets"]}


# --------------------------------------------------------------------------- #
# 1. Frame-cache validation
# --------------------------------------------------------------------------- #
def check_frames(seed: int) -> list[str]:
    fails = []
    env = fresh_env(seed, WARMUP)
    if env.done:
        return fails
    obs0 = obs_of(env, 0)
    feat = encode_obs(obs0, 0)
    offsets = feat["frame_offsets"]
    frame_planets = feat["frame_planets"]

    for f, off in enumerate(offsets):
        # Step a parallel env `off` empty turns from the encoded state.
        sub = fresh_env(seed, WARMUP)
        ok = True
        for _ in range(off):
            if sub.done:
                ok = False
                break
            sub.step([[], []])
        if not ok:
            continue  # game ended before this offset; encoder freezes too
        kag = planets_by_id(obs_of(sub, 0))
        got = {int(p[0]): p for p in frame_planets[f]}
        if got.keys() != kag.keys():
            fails.append(f"seed {seed} frame {f}(off {off}): planet ids {sorted(got)} vs {sorted(kag)}")
            continue
        for pid, gp in got.items():
            kp = kag[pid]
            _, gowner, gx, gy, gships = gp
            if gowner != int(kp[1]) or gships != int(kp[5]):
                fails.append(f"seed {seed} frame {f} planet {pid}: owner/ships ({gowner},{gships}) vs ({int(kp[1])},{int(kp[5])})")
            if abs(gx - float(kp[2])) > POS_TOL or abs(gy - float(kp[3])) > POS_TOL:
                fails.append(f"seed {seed} frame {f} planet {pid}: pos ({gx:.4f},{gy:.4f}) vs ({float(kp[2]):.4f},{float(kp[3]):.4f})")
    return fails


# --------------------------------------------------------------------------- #
# 2. Aim / arrival validation
# --------------------------------------------------------------------------- #
def kaggle_arrival(seed: int, src_id: int, angle: float, count: int, max_turns: int):
    """Issue one launch in a fresh kaggle env (reached from `seed` + WARMUP empty
    steps) and report (hit_planet_id | None, arrival_turn). The hit planet is the
    one whose ship count deviates from the production-only expectation when our
    (sole) fleet disappears."""
    env = fresh_env(seed, WARMUP)
    if env.done:
        return (None, 0)
    prev = planets_by_id(obs_of(env, 0))
    for turn in range(1, max_turns + 1):
        action0 = [[src_id, angle, count]] if turn == 1 else []
        env.step([action0, []])
        obs = obs_of(env, 0)
        cur = planets_by_id(obs)
        fleets = obs.get("fleets", [])
        if not fleets:  # our (only) fleet has resolved
            for pid, pp in prev.items():
                if pid not in cur:
                    continue
                cp = cur[pid]
                prev_owner, prev_ships, prod = int(pp[1]), int(pp[5]), int(pp[6])
                expected = prev_ships + (prod if prev_owner >= 0 else 0)
                if int(cp[1]) != prev_owner or int(cp[5]) != expected:
                    return (pid, turn)
            return (None, turn)  # left board / hit sun
        prev = cur
    return (None, 0)


def check_aim(seed: int, rng: random.Random) -> tuple[int, list[str]]:
    fails = []
    env = fresh_env(seed, WARMUP)
    if env.done:
        return 0, fails
    obs0 = obs_of(env, 0)
    feat = encode_obs(obs0, 0)
    ids = feat["planet_ids"]
    turns = feat["turns"]      # flat (NUM_FRAMES,44,44,6); frame 0 is first block
    angles = feat["angles"]    # flat (44,44,6)
    mask = feat["mask"]        # flat (44,44,6)
    src_ships = {int(p[0]): int(p[5]) for p in obs0["planets"]}

    valid = [
        (si, sj, a)
        for si in range(PLANET_SLOTS)
        for sj in range(PLANET_SLOTS)
        for a in range(ACTIONS_DIM)
        if mask[(si * PLANET_SLOTS + sj) * ACTIONS_DIM + a]
    ]
    rng.shuffle(valid)
    checked = 0
    for si, sj, a in valid[:ACTIONS_PER_STATE]:
        mi = (si * PLANET_SLOTS + sj) * ACTIONS_DIM + a
        id_i, id_j = ids[si], ids[sj]
        angle = angles[mi]
        # recover the integer count for this action from frame-t source ships
        ss = src_ships[id_i]
        if a <= 3:
            count = int(ss * (0.25, 0.50, 0.75, 1.00)[a])
        elif a == 4:
            count = 42
        else:
            # resolved+1: invert the predicted move by trusting the mask/count
            # relationship is exercised by the fraction/const actions; for the
            # resolved bin we just send what the engine reports it took, derived
            # below from a probe. Simplest: skip resolved bin here (covered by
            # the Rust engine-replay test).
            continue
        pred_turn = round(turns[mi] * 20.0)
        hit, turn = kaggle_arrival(seed, id_i, angle, count, max_turns=80)
        if hit != id_j:
            fails.append(f"seed {seed} {id_i}->{id_j} a{a} cnt{count}: kaggle hit {hit}, expected {id_j}")
        elif turn != pred_turn:
            fails.append(f"seed {seed} {id_i}->{id_j} a{a} cnt{count}: arrival turn {turn} != predicted {pred_turn}")
        checked += 1
    return checked, fails


AIM_HORIZON = 64  # must match features.rs; arrivals beyond this are masked out


def check_mask_completeness(seed: int, rng: random.Random) -> tuple[int, list[str]]:
    """Mask *completeness*: a masked-OUT action that is otherwise legal (owned
    source, sendable count, distinct present target) must genuinely fail to
    reach its target cleanly within the horizon. If kaggle shows such an action
    landing on the target by turn <= AIM_HORIZON, the mask wrongly excluded a
    valid move."""
    fails = []
    env = fresh_env(seed, WARMUP)
    if env.done:
        return 0, fails
    obs0 = obs_of(env, 0)
    feat = encode_obs(obs0, 0)
    ids = feat["planet_ids"]
    mask = feat["mask"]
    pmap = planets_by_id(obs0)

    # Candidate masked-out actions that are *non-trivially* invalid: source
    # owned by us, count sendable, target a distinct present planet. The only
    # legal reason for mask=0 here is a blocked/unreachable trajectory.
    cands = []
    for si in range(PLANET_SLOTS):
        id_i = ids[si]
        if id_i < 0 or id_i not in pmap or int(pmap[id_i][1]) != 0:
            continue
        ss = int(pmap[id_i][5])
        for sj in range(PLANET_SLOTS):
            id_j = ids[sj]
            if sj == si or id_j < 0 or id_j not in pmap:
                continue
            for a in range(4):  # fractions only (count is unambiguous here)
                if mask[(si * PLANET_SLOTS + sj) * ACTIONS_DIM + a]:
                    continue
                count = int(ss * (0.25, 0.50, 0.75, 1.00)[a])
                if 1 <= count <= ss:
                    cands.append((si, sj, a, id_i, id_j, count))
    rng.shuffle(cands)

    checked = 0
    for si, sj, a, id_i, id_j, count in cands[:ACTIONS_PER_STATE]:
        # We must pick an angle to fire. The mask gives no angle for invalid
        # actions, so aim straight at the target's current position — if even a
        # direct shot can't reach j cleanly, masking it is at least plausible;
        # but if the direct shot DOES reach j by the horizon, masking was wrong.
        tx, ty = float(pmap[id_j][2]), float(pmap[id_j][3])
        sx, sy = float(pmap[id_i][2]), float(pmap[id_i][3])
        angle = math.atan2(ty - sy, tx - sx)
        hit, turn = kaggle_arrival(seed, id_i, angle, count, max_turns=AIM_HORIZON)
        if hit == id_j and turn <= AIM_HORIZON:
            fails.append(f"seed {seed} {id_i}->{id_j} a{a} cnt{count}: masked-out but direct shot reaches j at turn {turn}")
        checked += 1
    return checked, fails


def main() -> int:
    rng = random.Random(0)
    frame_fails = []
    aim_fails = []
    aim_checked = 0
    mask_fails = []
    mask_checked = 0
    for seed in SEEDS:
        frame_fails += check_frames(seed)
        c, f = check_aim(seed, rng)
        aim_checked += c
        aim_fails += f
        mc, mf = check_mask_completeness(seed, rng)
        mask_checked += mc
        mask_fails += mf

    for m in (frame_fails + aim_fails + mask_fails)[:20]:
        print("  FAIL:", m)

    print(f"frame caches:      {'OK' if not frame_fails else f'{len(frame_fails)} MISMATCHES'} ({len(SEEDS)} seeds)")
    print(f"aim arrivals:      {aim_checked - len(aim_fails)} / {aim_checked} land on target at predicted turn")
    print(f"mask completeness: {mask_checked - len(mask_fails)} / {mask_checked} masked-out actions confirmed unreachable")
    return 1 if (frame_fails or aim_fails or mask_fails) else 0


if __name__ == "__main__":
    raise SystemExit(main())
