"""Validate the global + token features against an independent Python recompute.

`encode_obs` builds the `globals` vector and the per-planet tokens in Rust. Here
we recompute both straight from the kaggle observation (the source of truth for
state) and assert they match. This catches arithmetic / normalization / layout
bugs in features.rs without trusting env_model's own forward model.

Run (from experimental_arch/):
    python env_model/validate_globals.py
"""

from __future__ import annotations

import contextlib
import io
import math

from orbit_wars_model import encode_obs

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make

SEEDS = [1, 2, 3, 4, 5]
PLAYERS = 2
WARMUP = 8                  # empty steps so some fleets/positions differ
EPISODE_STEPS = 500
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
TOL = 1e-5

# Global feature layout — must match features.rs::compute_globals.
GLOBAL_NAMES = [
    "remaining", "own_score", "enemy_score", "neutral_ships", "own_planets",
    "enemy_planets", "neutral_planets", "own_production", "enemy_production",
    "own_fleet_ships", "enemy_fleet_ships", "score_share", "production_share",
    "planet_share", "score_diff", "production_diff",
]


def log_norm(x: float, full: float) -> float:
    return math.log1p(max(x, 0.0)) / math.log1p(full)


def signed_log_norm(x: float, full: float) -> float:
    return math.copysign(math.log1p(abs(x)) / math.log1p(full), x) if x != 0 else 0.0


def share(a: float, b: float) -> float:
    return a / (a + b) if (a + b) > 0 else 0.5


def expected_globals(obs: dict, player: int) -> list[float]:
    own_ps = en_ps = neu_s = 0
    own_pl = en_pl = neu_pl = 0
    own_pr = en_pr = 0
    for p in obs["planets"]:
        _, owner, _, _, _, ships, prod = (int(p[0]), int(p[1]), p[2], p[3], p[4], int(p[5]), int(p[6]))
        if owner == player:
            own_ps += ships; own_pl += 1; own_pr += prod
        elif owner >= 0:
            en_ps += ships; en_pl += 1; en_pr += prod
        else:
            neu_s += ships; neu_pl += 1
    own_fl = en_fl = 0
    for f in obs.get("fleets", []):
        owner, ships = int(f[1]), int(f[6])
        if owner == player:
            own_fl += ships
        elif owner >= 0:
            en_fl += ships
    own_score, en_score = own_ps + own_fl, en_ps + en_fl
    remaining = max(EPISODE_STEPS - int(obs["step"]), 0) / EPISODE_STEPS
    return [
        remaining,
        log_norm(own_score, 1000.0), log_norm(en_score, 1000.0), log_norm(neu_s, 1000.0),
        own_pl / 44.0, en_pl / 44.0, neu_pl / 44.0,
        log_norm(own_pr, 100.0), log_norm(en_pr, 100.0),
        log_norm(own_fl, 1000.0), log_norm(en_fl, 1000.0),
        share(own_score, en_score), share(own_pr, en_pr), share(own_pl, en_pl),
        signed_log_norm(own_score - en_score, 1000.0),
        signed_log_norm(own_pr - en_pr, 100.0),
    ]


def expected_av_token(obs: dict, pid: int) -> float:
    """Expected token[8] (angular velocity) for planet `pid`."""
    if pid in [int(c) for c in obs.get("comet_planet_ids", [])]:
        return 0.0
    init = {int(p[0]): p for p in obs["initial_planets"]}
    if pid not in init:
        return 0.0
    ip = init[pid]
    ix, iy, r = float(ip[2]), float(ip[3]), float(ip[4])
    orbital_r = math.hypot(ix - CENTER, iy - CENTER)
    orbiting = orbital_r + r < ROTATION_RADIUS_LIMIT
    return (obs["angular_velocity"] / 0.05) if orbiting else 0.0


def main() -> int:
    fails: list[str] = []
    checked = 0
    for seed in SEEDS:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(PLAYERS)
        for _ in range(WARMUP):
            if env.done:
                break
            env.step([[], []])
        # Only player 0's observation carries the full shared state; the
        # perspective is selected by the `player` arg to encode_obs.
        obs_p = dict(env.state[0].observation)
        for player in range(PLAYERS):
            feat = encode_obs(obs_p, player)
            got = list(feat["globals"])
            exp = expected_globals(obs_p, player)
            for name, g, e in zip(GLOBAL_NAMES, got, exp):
                checked += 1
                if abs(g - e) > TOL:
                    fails.append(f"seed {seed} p{player} global {name}: {g:.6f} vs {e:.6f}")

            # token[8] angular velocity, for every present planet slot.
            tokens = feat["tokens"].reshape(feat["tokens_shape"])  # (NF,44,9)
            ids = feat["planet_ids"]
            for slot, pid in enumerate(ids):
                if pid < 0:
                    continue
                got_av = float(tokens[0, slot, 8])
                exp_av = expected_av_token(obs_p, int(pid))
                checked += 1
                if abs(got_av - exp_av) > TOL:
                    fails.append(f"seed {seed} p{player} planet {pid} av: {got_av:.6f} vs {exp_av:.6f}")

    for m in fails[:20]:
        print("  FAIL:", m)
    print(f"globals + token-av: {'OK' if not fails else f'{len(fails)} MISMATCHES'} "
          f"({checked} values checked over {len(SEEDS)} seeds)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
