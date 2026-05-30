"""Validate env_engine's reward shaping.

Three terms (see RewardWeights in src/lib.rs):
  - terminal         = ships_share = own_ships / all_players_ships, in [0, 1]
  - terminal_time    = ±remaining/episode_steps (+ winner, − loser); max ~1 for
                       a turn-1 finish.
  - production_share = SHARE_W × (own_production / Σ all players' production),
                       a per-step dense term (0 if no player has production).

We drive games to termination with an aggressive self-play policy (so games end
both by elimination and by the step cap). The terminal terms are recomputed from
the final state (env_engine's own get_state) and checked against the engine's
reported reward_components; the per-step production_share term is recomputed from
each step's post-step observation and checked every step. Also asserts the time
bonus has the right sign and the fast-win magnitude bound (winner's bonus ==
remaining fraction).

Run (from experimental_arch/):
    python env_engine/validate_reward.py
"""

from __future__ import annotations

import math
import random

from orbit_wars_engine import OrbitWarsEngine

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
PLAYERS = 2
EPISODE_STEPS = 500
TERM_W = 1.0
TIME_W = 1.0
SHARE_W = 0.001
TOL = 1e-9


def production_share(obs: dict, n: int) -> list[float]:
    """Per-player own/Σ-player production from a post-step obs (neutral excluded),
    matching the engine's production_share term. Player obs[0] carries planets."""
    prod = [0] * n
    for p in obs.get("planets", []):
        owner = int(p[1])
        if 0 <= owner < n:
            prod[owner] += int(p[6])
    total = sum(prod)
    return [(prod[i] / total) if total > 0 else 0.0 for i in range(n)]


def aggressive(obs: dict, n: int, rng: random.Random, only_player: int | None = None) -> list:
    """Each player flings a chunk of ships from owned planets at a random angle
    — mutual attrition that tends to run to the step cap."""
    out = [[] for _ in range(n)]
    for p in obs.get("planets", []):
        pid, owner, ships = int(p[0]), int(p[1]), int(p[5])
        if only_player is not None and owner != only_player:
            continue
        if 0 <= owner < n and ships >= 2 and rng.random() < 0.6:
            out[owner].append([pid, rng.uniform(-math.pi, math.pi), max(1, ships // 2)])
    return out


def targeted(obs: dict, n: int) -> list:
    """Each player throws all ships from every owned planet straight at the
    biggest enemy planet's current position. This reliably eliminates a player
    well before the step cap, exercising the fast-win/loss time bonus with a
    large remaining fraction."""
    out = [[] for _ in range(n)]
    planets = obs.get("planets", [])
    for actor in range(n):
        mine = [p for p in planets if int(p[1]) == actor]
        enemies = [p for p in planets if 0 <= int(p[1]) < n and int(p[1]) != actor]
        if not enemies:
            continue
        tgt = max(enemies, key=lambda p: int(p[5]))
        for p in mine:
            if int(p[5]) >= 2:
                ang = math.atan2(tgt[3] - p[3], tgt[2] - p[2])
                out[actor].append([int(p[0]), ang, int(p[5])])
    return out


def final_scores(state: dict, n: int) -> list[int]:
    scores = [0] * n
    for p in state["planets"]:
        owner, ships = int(p[1]), int(p[5])
        if 0 <= owner < n:
            scores[owner] += ships
    for f in state["fleets"]:
        owner, ships = int(f[1]), int(f[6])
        if 0 <= owner < n:
            scores[owner] += ships
    return scores


def main() -> int:
    fails: list[str] = []
    checked = 0
    by_elim = by_time = 0
    # "mutual" random attrition (tends to hit the step cap) + "targeted" all-in
    # attacks (eliminate a player early, exercising a large remaining fraction).
    max_winner_bonus = 0.0
    for mode in ("mutual", "targeted"):
      for seed in SEEDS:
        engine = OrbitWarsEngine(num_players=PLAYERS)
        obs = engine.reset(seed=seed)["observations"]
        rng = random.Random(1000 + seed)
        for _ in range(EPISODE_STEPS + 5):
            turn_step = engine.step_count               # step index used in reward
            if mode == "mutual":
                acts = [aggressive(obs[pl], PLAYERS, rng) for pl in range(PLAYERS)]
            else:
                acts = targeted(obs[0], PLAYERS)
            out = engine.step(acts)
            obs = out["observations"]
            comp = out["reward_components"]

            # production_share is a per-step term: check it every step.
            exp_share = production_share(obs[0], PLAYERS)
            for i in range(PLAYERS):
                checked += 1
                got = comp["production_share"][i]
                want = SHARE_W * exp_share[i]
                if abs(got - want) > TOL:
                    fails.append(f"seed {seed} p{i} production_share {got:.9f} vs {want:.9f}")

            if not out["done"]:
                continue

            state = engine.get_state()
            scores = final_scores(state, PLAYERS)
            total = sum(scores)
            max_s = max(scores)
            remaining = max(EPISODE_STEPS - turn_step, 0)
            frac = min(max(remaining / EPISODE_STEPS, 0.0), 1.0)
            if turn_step >= EPISODE_STEPS - 2:
                by_time += 1
            else:
                by_elim += 1

            for i in range(PLAYERS):
                exp_share = (scores[i] / total) if total > 0 else 0.0
                won = scores[i] == max_s and max_s > 0
                exp_term = TERM_W * exp_share
                exp_time = TIME_W * (1.0 if won else -1.0) * frac
                checked += 2
                if abs(comp["terminal"][i] - exp_term) > TOL:
                    fails.append(f"seed {seed} p{i} terminal {comp['terminal'][i]:.6f} vs {exp_term:.6f}")
                if abs(comp["terminal_time"][i] - exp_time) > TOL:
                    fails.append(f"seed {seed} p{i} terminal_time {comp['terminal_time'][i]:.6f} vs {exp_time:.6f}")
                if won:
                    max_winner_bonus = max(max_winner_bonus, comp["terminal_time"][i])
            # Shares across players sum to 1 (or all-zero if board wiped).
            tsum = sum(comp["terminal"])
            if total > 0 and abs(tsum - TERM_W) > 1e-6:
                fails.append(f"seed {seed}: shares sum {tsum:.6f} != {TERM_W}")
            break

    for m in fails[:20]:
        print("  FAIL:", m)
    print(f"terminal shaping: {'OK' if not fails else f'{len(fails)} MISMATCHES'} "
          f"({checked} components checked; {by_elim} games by elimination, {by_time} by step cap)")
    print(f"largest winner time-bonus seen: +{max_winner_bonus:.3f} "
          f"(turn-1 elimination would give +1.000)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
