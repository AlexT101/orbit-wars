"""Re-extract the summary_v2 features locally using the FIXED
`extrapolate_fleets` (production accrual + tie + same-tick aggregation),
then run a Ridge regression and compare the coefficient table side-by-side
with the BUGGY version.

Uses the 5 raw JSON replays at orbit-wars/replays/*.json since we don't
have the full ~3.6k top-10 game corpus locally (it lives on Kaggle).
5 games × ~500 steps × 2 slots is ~5,000 rows — enough to see directional
changes in the standardized coefficients even though it can't match the
1.7M-row train set's stability. Output table is "which direction the
weight moves" not "definitive new weights."
"""

import json
import pathlib
import sys
from copy import deepcopy

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Reuse all the parser + per-block helpers from the existing rebuild script.
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kaggle_rebuild_v2 as kr  # noqa: E402

# 46-d summary_v2 feature names.
PLAYER_CUR = ["ships_on_planets", "ships_flying", "n_static", "n_orbit", "n_comet",
              "prod_static", "prod_orbit", "prod_comet",
              "n_neutrals_closer", "n_enemies_closer"]
PLAYER_EXT = ["ships_on_planets", "n_static", "n_orbit", "n_comet",
              "prod_static", "prod_orbit", "prod_comet",
              "n_neutrals_closer", "n_enemies_closer"]
NEUT = ["ships", "n_static", "n_orbit", "n_comet",
        "prod_static", "prod_orbit", "prod_comet", "comet_time_total"]
NAMES = ([f"me_cur.{n}" for n in PLAYER_CUR]
         + [f"opp_cur.{n}" for n in PLAYER_CUR]
         + [f"me_ext.{n}" for n in PLAYER_EXT]
         + [f"opp_ext.{n}" for n in PLAYER_EXT]
         + [f"neut.{n}" for n in NEUT])
assert len(NAMES) == 46


def extrapolate_fixed(state):
    """Fixed version of extrapolate_fleets:
      - production accrues on owned planets between arrival ticks (Bug 1)
      - same-tick same-owner arrivals are summed before combat (Bug 2)
      - tied attackers destroy each other (Bug 3, applied at same-tick combat)
    The arrival_time is now USED (cur_t tracks prior tick, prod*(t-cur_t)
    ships added if planet is owned before combat).
    """
    arrivals = {}
    for fleet in state["fleets"]:
        pred = kr.predict_fleet_collision(state, fleet)
        if pred is not None:
            pid, dt = pred
            arrivals.setdefault(pid, []).append((dt, fleet["owner"], fleet["ships"]))
    # Index planets by id so we can look up production.
    prod_for = {p["id"]: p["prod"] for p in state["planets"]}
    result = {p["id"]: (p["owner"], p["ships"]) for p in state["planets"]}
    for pid, arrs in arrivals.items():
        arrs.sort(key=lambda x: x[0])
        owner, ships = result.get(pid, (-1, 0))
        prod = prod_for.get(pid, 0)
        cur_t = 0
        # Group consecutive same-tick arrivals so combat at each tick
        # aggregates by owner (matches the engine's Combat step 1).
        i = 0
        while i < len(arrs):
            t = arrs[i][0]
            # Accrue production from cur_t to t for owned planets.
            if owner != -1 and t > cur_t:
                ships += prod * (t - cur_t)
            # Collect all arrivals at this same tick t.
            tick_by_owner = {}
            while i < len(arrs) and arrs[i][0] == t:
                _, fo, fs = arrs[i]
                tick_by_owner[fo] = tick_by_owner.get(fo, 0) + fs
                i += 1
            # Apply combat with the planet's CURRENT (owner, ships).
            sorted_atk = sorted(tick_by_owner.items(), key=lambda kv: -kv[1])
            top_owner, top_ships = sorted_atk[0]
            if len(sorted_atk) > 1:
                second = sorted_atk[1][1]
                if top_ships == second:
                    sv_owner, sv_ships = -1, 0   # tied attackers → all destroyed
                else:
                    sv_owner, sv_ships = top_owner, top_ships - second
            else:
                sv_owner, sv_ships = top_owner, top_ships
            if sv_ships > 0:
                if sv_owner == owner:
                    ships += sv_ships
                elif sv_ships > ships:
                    owner = sv_owner
                    ships = sv_ships - ships
                else:
                    ships -= sv_ships
            cur_t = t
        result[pid] = (owner, ships)
    return result


def py_extract_with(extrap_fn, o):
    """Same as kr.py_extract but using `extrap_fn` for extrapolation."""
    state = kr.parse_state(o)
    me = state["player"]
    opp = kr.dominant_enemy(state, me)
    extrap = extrap_fn(state)
    feats = (kr.current_block(state, me) + kr.current_block(state, opp)
             + kr.extrap_block(state, me, extrap) + kr.extrap_block(state, opp, extrap)
             + kr.neutral_block(state))
    return np.array(feats, dtype=np.float32)


def process_game(path, extrap_fn):
    data = json.loads(pathlib.Path(path).read_bytes())
    rewards = data.get("rewards") or []
    steps = data.get("steps") or []
    if len(rewards) != 2 or not steps or any(r is None for r in rewards):
        return None
    feats, labels = [], []
    for step in steps:
        if not isinstance(step, list) or len(step) < 2:
            continue
        for slot in range(2):
            entry = step[slot]
            if not isinstance(entry, dict):
                continue
            obs = entry.get("observation")
            if not obs or not obs.get("planets"):
                continue
            norm = kr.normalize_obs(obs)
            feats.append(py_extract_with(extrap_fn, norm))
            labels.append(float(rewards[slot]))
    if not feats:
        return None
    return np.stack(feats).astype(np.float32), np.array(labels, dtype=np.float32)


def train_ridge(X, y, label):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xs))
    split = int(0.9 * len(Xs))
    tr, va = idx[:split], idx[split:]
    m = Ridge(alpha=1.0)
    m.fit(Xs[tr], y[tr])
    yhat = m.predict(Xs[va])
    acc = float(np.mean(np.sign(yhat) == np.sign(y[va])))
    mse = float(np.mean((yhat - y[va]) ** 2))
    print(f"  [{label}] val_acc={acc:.4f}  val_mse={mse:.4f}")
    return m.coef_


def main():
    replays_dir = HERE.parent.parent.parent.parent / "replays"
    print(f"replays dir: {replays_dir}")
    json_files = sorted(replays_dir.glob("*.json"))
    print(f"found {len(json_files)} replays")

    feats_buggy, feats_fixed, labels = [], [], []
    for jf in json_files:
        rb = process_game(jf, kr.extrapolate_fleets)
        rf = process_game(jf, extrapolate_fixed)
        if rb is None or rf is None:
            print(f"  skipped {jf.name}")
            continue
        feats_buggy.append(rb[0])
        feats_fixed.append(rf[0])
        labels.append(rb[1])
        print(f"  {jf.name}: {rb[0].shape[0]} rows")

    Xb = np.concatenate(feats_buggy, axis=0)
    Xf = np.concatenate(feats_fixed, axis=0)
    y = np.concatenate(labels, axis=0)
    print(f"\nTotal: {Xb.shape[0]} rows × {Xb.shape[1]} features, y range [{y.min()},{y.max()}]\n")

    # Sanity: difference between buggy and fixed features.
    diff = (Xb - Xf)
    nz_cols = np.where(np.abs(diff).sum(axis=0) > 1e-3)[0]
    print(f"Features that DIFFER between buggy and fixed: {len(nz_cols)} of 46")
    for i in nz_cols:
        col_b = Xb[:, i]; col_f = Xf[:, i]
        mean_diff = float((col_f - col_b).mean())
        print(f"  {i:>2} {NAMES[i]:<32}  mean(fixed-buggy)={mean_diff:+.3f}")

    print("\nTraining Ridge on BUGGY features...")
    cb = train_ridge(Xb, y, "buggy")
    print("Training Ridge on FIXED features...")
    cf = train_ridge(Xf, y, "fixed")

    print()
    print("=" * 96)
    print(f"{'idx':>4} {'feature':<32}  {'buggy':>9}  {'fixed':>9}   {'Δ':>9}")
    print("=" * 96)
    pairs = sorted(range(46), key=lambda i: -max(abs(cb[i]), abs(cf[i])))
    for i in pairs:
        d = cf[i] - cb[i]
        marker = ""
        if cb[i] * cf[i] < -1e-3:
            marker = " (SIGN FLIP)"
        elif abs(d) > 0.05:
            marker = " (large Δ)"
        print(f"{i:>4} {NAMES[i]:<32}  {cb[i]:+9.4f}  {cf[i]:+9.4f}   {d:+9.4f}{marker}")

    print()
    print("Grouped sum |coef|:")
    groups = [("me_cur (10d)", range(0, 10)),
              ("opp_cur (10d)", range(10, 20)),
              ("me_ext (9d)",   range(20, 29)),
              ("opp_ext (9d)",  range(29, 38)),
              ("neut (8d)",     range(38, 46))]
    for name, rng_ in groups:
        sb = sum(abs(cb[i]) for i in rng_)
        sf = sum(abs(cf[i]) for i in rng_)
        print(f"  {name:<14}  buggy={sb:.4f}  fixed={sf:.4f}  Δ={sf-sb:+.4f}")


if __name__ == "__main__":
    main()
