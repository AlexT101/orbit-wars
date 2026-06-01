"""Predict whether a planet will be one of the first ceil(N/4) acquired by
a player, given features from the first frame only.

Per (planet, perspective_player) row, features:
  - dist_to_my_home
  - dist_to_opp_home
  - dist_to_center
  - is_rotating          (1 if orbiting, else 0)
  - rot_toward_me        (1 if rotating AND tangential velocity has positive
                          dot with vector toward my home)
  - rot_away_from_me     (1 if rotating AND tangential velocity has negative
                          dot with vector toward my home)
  - production
  - neutral_ships        (initial garrison)
  - n_total_planets      (game-constant)
  - my_home_production

Label: 1 if this planet is one of the first ceil(N/4) the perspective
player owned across the whole episode (home counts as owned from t=0;
acquisitions thereafter are added in chronological order).

Usage:
    python train_first_owned_xgb.py \
        --zip /tmp/orbit_days/orbit-wars-episodes-2026-05-30.zip \
        --out bots/mine/alphaow/train/weights/first_owned.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


# ---------- constants ----------
CENTER = (50.0, 50.0)
ROTATION_RADIUS_LIMIT = 50.0
FEATURE_NAMES = [
    "dist_to_my_home",
    "dist_to_opp_home",
    "dist_to_center",
    "is_stationary",
    "is_orbiting",
    "angular_distance",          # signed, [-pi, pi]; planet vs my home
    "angular_distance_to_enemy", # signed, [-pi, pi]; planet vs opp home
    "production",
    "neutral_ships",
    "n_stationary_planets",
    "n_orbiting_planets",
    "sum_stationary_production",
    "sum_orbiting_production",
    "my_home_production",
    "closest_my_side_neighbor",
    "closest_enemy_side_neighbor",
    "dist_to_edge",              # min distance to any board boundary
    "angular_velocity",          # constant per game
    "starting_dist_from_center", # orbital radius (= dist_to_center at t=0)
    # One-hot for what kind of planet each home is. Homes can be either
    # stationary or orbiting depending on the game seed; an orbiting home
    # moves over time, which changes what neighbours stay reachable.
    "my_home_is_stationary",
    "my_home_is_orbiting",
    "opp_home_is_stationary",
    "opp_home_is_orbiting",
    # Ticks-until-alignment (signed): angular_distance / angular_velocity.
    # For an orbiting planet rotating at +ω rad/tick:
    #   * negative value = planet is rotating TOWARD alignment with my
    #     home's angle (will reach it in |value| ticks)
    #   * positive value = planet rotating AWAY from alignment
    # 0 for stationary planets (no rotation, undefined).
    "angular_dist_div_omega",
    "angular_dist_to_enemy_div_omega",
    # Approximate "turns of home production needed to amass enough ships
    # to capture this neutral, starting from 10". User-spec formula:
    #   (neutral_ships - 10) / my_home_production
    # Can be negative if neutral garrison < 10 (capture is feasible
    # right away). Home rows: trivial 0.
    "turns_to_capture",
    # Approximate travel time from my home to this planet, assuming I
    # send `neutral_ships + 1` ships (just enough to take it). Computed
    # as Euclidean distance / fleet_speed(neutral_ships + 1) — apollo's
    # full dir_to_hit pathing would adjust slightly for obstacles and
    # orbital motion, but at t=0 this is a good first approximation.
    "travel_time",
    # Mean of 1/distance from THIS planet to every other planet whose
    # home-side classification matches my side / opp's side. Captures
    # how "embedded" the target is in friendly vs hostile territory.
    "mean_inv_dist_my_side",
    "mean_inv_dist_opp_side",
    # Reciprocals (1/x) so XGBoost can split on either scale.
    "inv_dist_to_my_home",
    "inv_dist_to_opp_home",
    "inv_closest_my_side_neighbor",
    "inv_closest_enemy_side_neighbor",
    # Distance from my home (= opp's home by board symmetry) to the
    # nearest board edge. Game-level feature, equal for both perspectives.
    "my_home_dist_to_edge",
]


INV_CAP = 10.0  # caps reciprocal at 1/0.1 — distances below 0.1 are
                # essentially "same position," rare for distinct planets.


def safe_inv(x: float) -> float:
    if x <= 1e-3:
        return INV_CAP
    v = 1.0 / x
    return v if v <= INV_CAP else INV_CAP


def augment_features(X: np.ndarray, feat_names: list[str]):
    """For each non-boolean feature, append signed-log and (if no
    corresponding `inv_*` already exists) a capped reciprocal."""
    # Detect boolean columns: values restricted to {0, 1}.
    is_bool = np.zeros(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        u = np.unique(X[:, i])
        if set(u.tolist()).issubset({0.0, 1.0}):
            is_bool[i] = True
    name_set = set(feat_names)
    new_cols = []
    new_names = []
    inv_cap = 10.0
    for i, name in enumerate(feat_names):
        if is_bool[i]:
            continue
        col = X[:, i]
        # Signed log: handles 0 and negatives.
        log_col = np.sign(col) * np.log1p(np.abs(col))
        new_cols.append(log_col.astype(np.float32))
        new_names.append(f"log_{name}")
        # Reciprocal — skip if an `inv_<name>` is already a feature, or
        # if the feature is itself an inverse / mean-inverse.
        already_inverse = (
            name.startswith("inv_")
            or name.startswith("mean_inv")
            or f"inv_{name}" in name_set
        )
        if not already_inverse:
            denom = np.abs(col) + 0.1
            inv_col = np.sign(col) / denom
            inv_col = np.clip(inv_col, -inv_cap, inv_cap)
            new_cols.append(inv_col.astype(np.float32))
            new_names.append(f"inv_{name}")
    if not new_cols:
        return X, feat_names
    X_aug = np.column_stack([X] + new_cols).astype(np.float32)
    return X_aug, list(feat_names) + new_names


def fleet_speed(n_ships: float, max_speed: float = 6.0) -> float:
    """Match alphaow's fleet_speed formula (pathing.rs)."""
    if n_ships <= 1:
        return 1.0
    s = 1.0 + (max_speed - 1.0) * (math.log(n_ships) / math.log(1000.0)) ** 1.5
    return min(max_speed, max(1.0, s))

# KDE bandwidths included simultaneously as features (multi-scale density).
KDE_BANDWIDTHS = [20.0, 30.0, 40.0, 50.0]

BOARD = 100.0
NO_NEIGHBOR_DIST = 200.0


def kde_at(target_xy, planets, sigma):
    """Gaussian-kernel density at `target_xy` summed over `planets`
    (each is `(x, y, weight)`). Caller subtracts the self-term if the
    target itself is in `planets`."""
    s = 0.0
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for x, y, w in planets:
        dx = x - target_xy[0]
        dy = y - target_xy[1]
        d2 = dx * dx + dy * dy
        s += w * math.exp(-d2 * inv2s2)
    return s


# ---------- feature extraction ----------
def planet_geom(p, comet_ids, initial_pos):
    """Returns (orbital_radius, initial_angle, is_orbiting, is_comet) for planet row p."""
    pid, owner, x, y, radius, ships, prod = p
    is_comet = pid in comet_ids
    ix, iy = initial_pos.get(pid, (x, y))
    dx = ix - CENTER[0]
    dy = iy - CENTER[1]
    orbital_radius = math.sqrt(dx * dx + dy * dy)
    initial_angle = math.atan2(dy, dx)
    is_orbiting = (not is_comet) and (orbital_radius + radius < ROTATION_RADIUS_LIMIT)
    return orbital_radius, initial_angle, is_orbiting, is_comet


def signed_angle_diff(theta_a: float, theta_b: float) -> float:
    """Signed angular difference theta_a − theta_b, normalised to [-π, π]."""
    d = theta_a - theta_b
    return math.atan2(math.sin(d), math.cos(d))


def extract_game(replay_json: dict, game_id: str):
    """Returns list of (features, label, weight, game_id, perspective,
    planet_id, rank) for one 2p episode, or None if not 2p / malformed."""
    rewards = replay_json.get("rewards", [])
    if len(rewards) != 2:
        return None
    steps = replay_json.get("steps")
    if not steps or not steps[0]:
        return None

    # First-frame observation (player 0's view — planets list is identical for both).
    obs0 = steps[0][0].get("observation", {})
    planets = obs0.get("planets")
    if not planets:
        return None
    angular_vel = obs0.get("angular_velocity", 0.0)
    comet_planet_ids = set(obs0.get("comet_planet_ids", []) or [])
    initial_planets = obs0.get("initial_planets") or planets
    initial_pos = {p[0]: (p[2], p[3]) for p in initial_planets}

    n_total = len(planets)
    first_k = max(1, math.ceil(n_total / 4))

    # Find each player's home at t=0.
    homes = {}  # player_id -> (x, y, production, planet_id)
    for p in planets:
        pid, owner, x, y, radius, ships, prod = p
        if owner in (0, 1) and owner not in homes:
            homes[owner] = (x, y, prod, pid)
    if 0 not in homes or 1 not in homes:
        return None
    home0 = homes[0]
    home1 = homes[1]

    # Per-planet geometry + side-classification + KDE-style densities.
    planet_geom_cache = {}
    side_closer_to = {}
    n_stationary = 0
    n_orbiting = 0
    sum_stationary_prod = 0
    sum_orbiting_prod = 0
    for p in planets:
        pid = p[0]
        x, y, prod = p[2], p[3], p[6]
        _, _, is_orb, is_com = planet_geom(p, comet_planet_ids, initial_pos)
        is_stat = (not is_orb) and (not is_com)
        planet_geom_cache[pid] = (is_stat, is_orb, is_com)
        d0 = math.hypot(x - home0[0], y - home0[1])
        d1 = math.hypot(x - home1[0], y - home1[1])
        side_closer_to[pid] = 0 if d0 <= d1 else 1
        if is_orb:
            n_orbiting += 1
            sum_orbiting_prod += prod
        elif is_stat:
            n_stationary += 1
            sum_stationary_prod += prod

    # Walk all steps to record per-player first-acquisition tick of each planet.
    # acquisitions[player_id][planet_id] = tick first owned by this player
    acquisitions = {0: {homes[0][3]: 0}, 1: {homes[1][3]: 0}}
    prev_owners = {p[0]: p[1] for p in planets}
    for t in range(1, len(steps)):
        try:
            obs_t = steps[t][0].get("observation", {})
            planets_t = obs_t.get("planets")
            if not planets_t:
                continue
            for p in planets_t:
                pid, owner = p[0], p[1]
                if owner in (0, 1) and prev_owners.get(pid) != owner:
                    if pid not in acquisitions[owner]:
                        acquisitions[owner][pid] = t
                prev_owners[pid] = owner
        except Exception:
            break

    # For each perspective player, label the first `first_k` acquisitions
    # AND record each planet's acquisition rank (0 = home, 1 = first
    # capture, …, first_k - 1 = last in the first_k window). Anything past
    # first_k or never owned has rank = None.
    first_owned = {0: set(), 1: set()}
    rank_of = {0: {}, 1: {}}
    for pl in (0, 1):
        items = sorted(acquisitions[pl].items(), key=lambda kv: kv[1])
        for r, (pid, _t) in enumerate(items[:first_k]):
            first_owned[pl].add(pid)
            rank_of[pl][pid] = r

    # Symmetric first-capture bonus: both players' rank-1 (first non-home
    # capture) are 180°-rotational mirrors around the center → both made
    # the same "obvious" choice, strong signal.
    p0_rank1 = next((pid for pid, r in rank_of[0].items() if r == 1), None)
    p1_rank1 = next((pid for pid, r in rank_of[1].items() if r == 1), None)
    symmetric_first = False
    if p0_rank1 is not None and p1_rank1 is not None:
        ip0 = initial_pos.get(p0_rank1)
        ip1 = initial_pos.get(p1_rank1)
        if ip0 and ip1:
            mirror = (2 * CENTER[0] - ip0[0], 2 * CENTER[1] - ip0[1])
            if math.hypot(ip1[0] - mirror[0], ip1[1] - mirror[1]) < 5.0:
                symmetric_first = True
    SYM_FIRST_BONUS = 2.0


    # Pre-extract (x, y, weight=1) and (x, y, weight=prod) lists for KDE.
    pos_unit = [(p[2], p[3], 1.0) for p in planets]
    pos_prod = [(p[2], p[3], float(p[6])) for p in planets]
    # My home distance to nearest edge. Same for opp by board symmetry,
    # so we compute once. Perspective-independent for this feature.
    home0_xy = (home0[0], home0[1])
    my_home_dist_to_edge_game = min(
        home0_xy[0], home0_xy[1],
        BOARD - home0_xy[0], BOARD - home0_xy[1],
    )

    rows = []
    for p in planets:
        pid, owner, x, y, radius, ships, prod = p
        is_stationary, is_orbiting, _ = planet_geom_cache[pid]
        planet_theta = math.atan2(y - CENTER[1], x - CENTER[0])

        # Closest neighbour AND mean(1/d) per side, by perspective 0.
        # (For perspective 1 the two values just swap roles.)
        closest_side0 = NO_NEIGHBOR_DIST
        closest_side1 = NO_NEIGHBOR_DIST
        sum_inv_side0 = 0.0
        sum_inv_side1 = 0.0
        cnt_side0 = 0
        cnt_side1 = 0
        for q in planets:
            qid = q[0]
            if qid == pid:
                continue
            d = math.hypot(x - q[2], y - q[3])
            if d < 1e-6:
                continue
            inv = 1.0 / d
            if side_closer_to[qid] == 0:
                if d < closest_side0:
                    closest_side0 = d
                sum_inv_side0 += inv
                cnt_side0 += 1
            else:
                if d < closest_side1:
                    closest_side1 = d
                sum_inv_side1 += inv
                cnt_side1 += 1
        mean_inv_side0 = sum_inv_side0 / cnt_side0 if cnt_side0 else 0.0
        mean_inv_side1 = sum_inv_side1 / cnt_side1 if cnt_side1 else 0.0

        # KDE densities at this planet's location, one (num, prod) pair
        # per bandwidth. Self-term subtracted; no edge correction (the
        # edge bias is itself informative — corner planets *do* have
        # fewer neighbours, and that geometry correlates with capture).
        kde_pairs = []
        inv_kde_pairs = []
        for sigma in KDE_BANDWIDTHS:
            num_d = kde_at((x, y), pos_unit, sigma) - 1.0
            prod_d = kde_at((x, y), pos_prod, sigma) - float(prod)
            kde_pairs.extend([num_d, prod_d])
            inv_kde_pairs.extend([safe_inv(num_d), safe_inv(prod_d)])

        # New per-planet (perspective-independent) features.
        dist_to_edge = min(x, y, BOARD - x, BOARD - y)
        # orbital_radius from planet_geom = sqrt((ix-50)² + (iy-50)²),
        # i.e. starting distance from center for both orbiting and
        # stationary planets.
        orbital_radius, _initial_angle, _, _ = planet_geom(
            p, comet_planet_ids, initial_pos
        )

        import builtins as _b3
        if getattr(_b3, "OW4_WINNERS_ONLY", False):
            rewards = replay_json.get("rewards", [])
            perspectives_to_use = [
                p for p in (0, 1)
                if p < len(rewards) and rewards[p] == 1
            ]
            if not perspectives_to_use:
                return None
        else:
            perspectives_to_use = (0, 1)
        for perspective in perspectives_to_use:
            my_home_x, my_home_y, my_home_prod, my_home_pid = homes[perspective]
            opp = 1 - perspective
            opp_home_x, opp_home_y, _, opp_home_pid = homes[opp]
            my_home_stat, my_home_orb, _ = planet_geom_cache[my_home_pid]
            opp_home_stat, opp_home_orb, _ = planet_geom_cache[opp_home_pid]
            d_my = math.hypot(x - my_home_x, y - my_home_y)
            d_opp = math.hypot(x - opp_home_x, y - opp_home_y)
            d_center = math.hypot(x - CENTER[0], y - CENTER[1])
            my_home_theta = math.atan2(
                my_home_y - CENTER[1], my_home_x - CENTER[0]
            )
            opp_home_theta = math.atan2(
                opp_home_y - CENTER[1], opp_home_x - CENTER[0]
            )
            ang_to_me = signed_angle_diff(planet_theta, my_home_theta)
            ang_to_opp = signed_angle_diff(planet_theta, opp_home_theta)
            if perspective == 0:
                closest_my, closest_opp = closest_side0, closest_side1
                mean_inv_my, mean_inv_opp = mean_inv_side0, mean_inv_side1
            else:
                closest_my, closest_opp = closest_side1, closest_side0
                mean_inv_my, mean_inv_opp = mean_inv_side1, mean_inv_side0
            # Angular dist over angular speed → ticks-to-alignment.
            if is_orbiting and abs(angular_vel) > 1e-9:
                ang_div_omega_me = ang_to_me / angular_vel
                ang_div_omega_opp = ang_to_opp / angular_vel
            else:
                ang_div_omega_me = 0.0
                ang_div_omega_opp = 0.0
            # Turns of home production to amass `ships - 10` extra.
            turns_to_cap = (
                (ships - 10) / my_home_prod if my_home_prod > 0 else 0.0
            )
            # Travel time assuming I send (ships + 1) attackers.
            fleet_n = max(1, ships + 1)
            travel_t = d_my / fleet_speed(fleet_n)
            features = [
                d_my, d_opp, d_center,
                1 if is_stationary else 0,
                1 if is_orbiting else 0,
                ang_to_me, ang_to_opp,
                prod, ships,
                n_stationary, n_orbiting,
                sum_stationary_prod, sum_orbiting_prod,
                my_home_prod,
                closest_my, closest_opp,
                dist_to_edge, angular_vel, orbital_radius,
                1 if my_home_stat else 0,
                1 if my_home_orb else 0,
                1 if opp_home_stat else 0,
                1 if opp_home_orb else 0,
                ang_div_omega_me, ang_div_omega_opp,
                turns_to_cap, travel_t,
                mean_inv_my, mean_inv_opp,
                safe_inv(d_my), safe_inv(d_opp),
                safe_inv(closest_my), safe_inv(closest_opp),
                my_home_dist_to_edge_game,
            ] + kde_pairs + inv_kde_pairs
            import builtins as _b2
            _r1only = getattr(_b2, "OW4_RANK1_ONLY", False)
            r_now = rank_of[perspective].get(pid)
            if _r1only:
                label = 1 if r_now == 1 else 0
            else:
                label = 1 if pid in first_owned[perspective] else 0
            # Weight schedule: linear decay over acquisition rank, peak at
            # rank 1 (first non-home capture — what the user said is the
            # "most predictable" point before the game diverges).
            #   rank 0 (home, trivial positive): weight = 1.0
            #   rank 1 (first capture):          weight = first_k
            #   rank 2:                          weight = first_k - 1
            #   …
            #   rank first_k - 1 (last in window): weight = 2.0
            #   not in first_k (negatives):      weight = 1.0
            import builtins as _b
            _no_w = getattr(_b, "OW4_NO_WEIGHTS", False)
            _no_sb = getattr(_b, "OW4_NO_SYM_BONUS", False)
            r = rank_of[perspective].get(pid)
            if _no_w:
                weight = 1.0
            elif r is None or r == 0:
                weight = 1.0
            else:
                weight = float(first_k - r + 1)
            if r == 1 and symmetric_first and not _no_sb:
                weight *= SYM_FIRST_BONUS
            # Encode rank for downstream metrics: rank or -1 if never in
            # first_k for this perspective.
            r_encoded = r if r is not None else -1
            rows.append((features, label, weight, game_id, perspective,
                         pid, r_encoded))

    return rows


# ---------- worker ----------
def process_chunk(args):
    # Unpack flags so they propagate into subprocess (ProcessPoolExecutor
    # uses 'spawn' on macOS — bare builtins.X set in the parent doesn't
    # reach the child).
    zip_path, names, flags = args
    import builtins as _b
    _b.OW4_NO_WEIGHTS = flags.get("no_weights", False)
    _b.OW4_NO_SYM_BONUS = flags.get("no_sym_bonus", False)
    _b.OW4_RANK1_ONLY = flags.get("rank1_only_label", False)
    _b.OW4_WINNERS_ONLY = flags.get("winners_only", False)
    out_X, out_y, out_w = [], [], []
    out_meta = []  # rows of (group_id, planet_id, rank)
    group_map = {}  # (game_id, perspective) -> integer group_id
    with zipfile.ZipFile(zip_path) as z:
        for name in names:
            try:
                with z.open(name) as f:
                    data = json.load(f)
                rows = extract_game(data, name)
                if not rows:
                    continue
                for feats, label, weight, gid, persp, pid, r in rows:
                    key = (gid, persp)
                    if key not in group_map:
                        group_map[key] = len(group_map)
                    out_X.append(feats)
                    out_y.append(label)
                    out_w.append(weight)
                    out_meta.append((group_map[key], pid, r))
            except Exception:
                continue
    return (
        np.asarray(out_X, dtype=np.float32),
        np.asarray(out_y, dtype=np.int8),
        np.asarray(out_w, dtype=np.float32),
        np.asarray(out_meta, dtype=np.int64),
        # Chunk-local group_map size — needed so the main process can
        # re-offset group IDs to be globally unique.
        len(group_map),
    )


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, nargs="+",
                    help="One or more daily replay zips")
    ap.add_argument("--out", required=True, help="Output XGB JSON path")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=600,
                    help="Max num_boost_round (early-stopping may halt sooner)")
    ap.add_argument("--max-games", type=int, default=0,
                    help="Cap total games for quick runs (0 = no cap)")
    # Ablation flags
    ap.add_argument("--no-weights", action="store_true",
                    help="Uniform sample weights (1.0 for every row)")
    ap.add_argument("--no-sym-bonus", action="store_true",
                    help="Disable the symmetric-first-capture weight bonus")
    ap.add_argument("--no-home-onehot", action="store_true",
                    help="Drop my_home_is_stationary/orbiting + opp variants")
    ap.add_argument("--no-ticks-to-align", action="store_true",
                    help="Drop angular_dist*/ω features")
    ap.add_argument("--rank1-only-label", action="store_true",
                    help="Train with label = (rank == 1) — explicit "
                         "first-non-home-capture target. Sparser positives "
                         "but the right loss for argmax-per-game eval.")
    ap.add_argument("--winners-only", action="store_true",
                    help="Only emit rows for the winning perspective(s).")
    ap.add_argument("--bandwidths", type=str, default=None,
                    help="Comma-separated KDE bandwidths to sweep")
    ap.add_argument("--augment-log-inv", action="store_true",
                    help="For every non-boolean feature, append signed log "
                         "and reciprocal (if not already present).")
    args = ap.parse_args()
    # Mutate module-level globals based on flags. Crude but keeps the
    # extraction path simple.
    global KDE_BANDWIDTHS
    if args.bandwidths:
        KDE_BANDWIDTHS = [float(s) for s in args.bandwidths.split(",")]
    # Patch extract_game's behaviour by toggling global flags read inside it.
    import builtins
    builtins.OW4_NO_WEIGHTS = args.no_weights
    builtins.OW4_NO_SYM_BONUS = args.no_sym_bonus
    builtins.OW4_RANK1_ONLY = args.rank1_only_label
    builtins.OW4_WINNERS_ONLY = args.winners_only

    zip_paths = [Path(p) for p in args.zip]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flags = {
        "no_weights": args.no_weights,
        "no_sym_bonus": args.no_sym_bonus,
        "rank1_only_label": args.rank1_only_label,
        "winners_only": args.winners_only,
    }
    chunks = []
    total = 0
    for zip_path in zip_paths:
        print(f"reading game list from {zip_path}…")
        with zipfile.ZipFile(zip_path) as z:
            names = sorted(z.namelist())
        if args.max_games:
            names = names[: args.max_games]
        print(f"  {len(names)} games")
        chunk_size = max(50, len(names) // (args.workers * 4))
        chunks.extend(
            (str(zip_path), names[i:i + chunk_size], flags)
            for i in range(0, len(names), chunk_size)
        )
        total += len(names)
    print(f"\n{total} games total, {len(chunks)} chunks, "
          f"{args.workers} workers")

    Xs, ys, ws, metas = [], [], [], []
    group_offset = 0
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_chunk, c) for c in chunks]
        for fut in as_completed(futs):
            X, y, w, meta, n_groups = fut.result()
            if len(X):
                # Make group IDs globally unique.
                meta[:, 0] += group_offset
                group_offset += n_groups
                Xs.append(X)
                ys.append(y)
                ws.append(w)
                metas.append(meta)
            n_done += 1
            if n_done % 5 == 0 or n_done == len(chunks):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - n_done) / rate if rate > 0 else 0
                print(f"  chunks {n_done}/{len(chunks)} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    X_full = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    meta = np.concatenate(metas, axis=0)  # cols: group_id, planet_id, rank
    base_n = len(FEATURE_NAMES)
    print(f"\ndataset: {X_full.shape[0]} rows, {X_full.shape[1]} features "
          f"({base_n} base + {2 * len(KDE_BANDWIDTHS)} KDE)")
    print(f"  positives: {y.sum()} ({100 * y.mean():.2f}%)")
    print(f"  weight stats: min={w.min():.1f} max={w.max():.1f} "
          f"mean={w.mean():.2f}")

    # Group-aware split: holdout groups (a game-perspective pair) entirely
    # so rank-1 evaluation is honest (no group leakage into train).
    rng = np.random.default_rng(0)
    n_groups = int(meta[:, 0].max()) + 1
    perm = rng.permutation(n_groups)
    n_test_groups = max(200, n_groups // 10)
    test_groups = set(perm[:n_test_groups].tolist())
    test_mask = np.fromiter(
        (g in test_groups for g in meta[:, 0]), dtype=bool, count=len(meta)
    )
    test_i = np.where(test_mask)[0]
    train_i = np.where(~test_mask)[0]
    print(f"  group split: {n_groups - n_test_groups} train / "
          f"{n_test_groups} test groups → "
          f"{len(train_i)} train rows / {len(test_i)} test rows")

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 6,
        "eta": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
    }

    results = []
    best_sigma = None
    best_bst = None
    best_auc = -1.0
    best_feat_names = None

    DROP_NAMES = set()
    if args.no_home_onehot:
        DROP_NAMES.update([
            "my_home_is_stationary", "my_home_is_orbiting",
            "opp_home_is_stationary", "opp_home_is_orbiting",
        ])
    if args.no_ticks_to_align:
        DROP_NAMES.update([
            "angular_dist_div_omega", "angular_dist_to_enemy_div_omega",
        ])
    keep_base_idx = [
        i for i, n in enumerate(FEATURE_NAMES) if n not in DROP_NAMES
    ]
    base_feat_names = [FEATURE_NAMES[i] for i in keep_base_idx]
    # Use ALL KDE bandwidths simultaneously as features (multi-scale
    # density). Loop over a single iteration so the surrounding result
    # bookkeeping still works.
    for _ in [None]:
        sigma = "+".join(f"{s:g}" for s in KDE_BANDWIDTHS)
        n_kde = 2 * len(KDE_BANDWIDTHS)
        kde_cols = list(range(base_n, base_n + n_kde))
        inv_kde_cols = list(range(base_n + n_kde, base_n + 2 * n_kde))
        col_idx = keep_base_idx + kde_cols + inv_kde_cols
        X = X_full[:, col_idx]
        feat_names = base_feat_names + [
            n
            for s in KDE_BANDWIDTHS
            for n in (f"num_density_h{s:g}", f"prod_density_h{s:g}")
        ] + [
            n
            for s in KDE_BANDWIDTHS
            for n in (f"inv_num_density_h{s:g}", f"inv_prod_density_h{s:g}")
        ]

        if args.augment_log_inv:
            X_aug, feat_names = augment_features(X, feat_names)
            X = X_aug
            print(f"  augmented features: {X.shape[1]} total (added "
                  f"{X.shape[1] - len(col_idx)} log/inv columns)")
        Xtr, ytr, wtr = X[train_i], y[train_i], w[train_i]
        Xte, yte = X[test_i], y[test_i]
        dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr, feature_names=feat_names)
        dte = xgb.DMatrix(Xte, label=yte, feature_names=feat_names)
        print(f"\n=== training with KDE bandwidth σ={sigma} ===")
        bst = xgb.train(
            params, dtr, num_boost_round=args.rounds,
            evals=[(dtr, "train"), (dte, "test")],
            verbose_eval=200, early_stopping_rounds=50,
        )
        pred = bst.predict(dte)
        bin_pred = (pred >= 0.5).astype(int)
        acc = accuracy_score(yte, bin_pred)
        ll = log_loss(yte, pred)
        auc = roc_auc_score(yte, pred)
        imps = bst.get_score(importance_type="gain")

        # === Rank-1 metric: per (game, perspective) group, the planet
        # with the highest predicted probability among non-home planets
        # should be the actual rank-1 (first non-home capture).
        meta_te = meta[test_i]
        groups_te = meta_te[:, 0]
        ranks_te = meta_te[:, 2]
        # Mask out home rows (rank 0) when picking the model's top guess.
        non_home_mask = ranks_te != 0
        n_correct_top1 = 0
        n_groups_with_rank1 = 0
        # Iterate per group.
        # numpy sort to walk contiguous groups efficiently.
        order = np.argsort(groups_te, kind="stable")
        groups_sorted = groups_te[order]
        ranks_sorted = ranks_te[order]
        pred_sorted = pred[order]
        non_home_sorted = non_home_mask[order]
        i = 0
        n = len(order)
        while i < n:
            j = i
            while j < n and groups_sorted[j] == groups_sorted[i]:
                j += 1
            g_ranks = ranks_sorted[i:j]
            g_pred = pred_sorted[i:j]
            g_nh = non_home_sorted[i:j]
            actual_rank1 = np.where(g_ranks == 1)[0]
            if len(actual_rank1) == 1 and g_nh.any():
                n_groups_with_rank1 += 1
                # Best non-home pred.
                candidate_idx = np.where(g_nh)[0]
                cand_preds = g_pred[candidate_idx]
                top = candidate_idx[int(np.argmax(cand_preds))]
                if top == actual_rank1[0]:
                    n_correct_top1 += 1
            i = j
        top1_acc = (n_correct_top1 / n_groups_with_rank1
                    if n_groups_with_rank1 else float("nan"))
        results.append((sigma, acc, ll, auc, bst.best_iteration + 1
                        if hasattr(bst, "best_iteration") else args.rounds,
                        imps, feat_names, top1_acc, n_groups_with_rank1))
        print(f"  σ={sigma}: acc={acc:.4f}  ll={ll:.4f}  auc={auc:.4f}  "
              f"rank1_top1_acc={top1_acc:.4f} ({n_groups_with_rank1} groups)")
        if auc > best_auc:
            best_auc = auc
            best_sigma = sigma
            best_bst = bst
            best_feat_names = feat_names

    print("\n=== KDE bandwidth sweep ===")
    print(f"  {'sigma':<8} {'acc':<9} {'ll':<9} {'auc':<9} {'rank1_top1':<10}")
    for sigma, acc, ll, auc, _, _, _, t1, ng in results:
        marker = "  ←" if sigma == best_sigma else ""
        print(f"  {str(sigma):<14} {acc:<9.4f} {ll:<9.4f} {auc:<9.4f} "
              f"{t1:<10.4f}{marker}")

    print(f"\nbest σ={best_sigma}; importances (gain):")
    best_imps = next(r[5] for r in results if r[0] == best_sigma)
    for name in best_feat_names:
        print(f"  {name:<26} {best_imps.get(name, 0.0):.2f}")

    best_bst.save_model(str(out_path))
    print(f"\nsaved best (σ={best_sigma}) → {out_path}")


if __name__ == "__main__":
    main()
