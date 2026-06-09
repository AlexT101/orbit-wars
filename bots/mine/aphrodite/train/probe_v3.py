"""Leaf-variance probe for the 4p summary_v3 features.

Feeds real 4p replay positions through aphrodite's DUCT search (`probe_leaves_v3`
bin), capturing every value-net leaf's 145-d summary_v3 vector tagged by the
search it belongs to, then measures — per feature — the fraction of variance that
occurs BETWEEN sibling leaves of the same search (the cohort DUCT actually ranks):

    sib_ratio = within_cohort_var / total_var   in [0,1]   (0=passenger, 1=ranker)

and the rigorous scale-free version: the within-cohort std of each feature's
per-leaf SHAP contribution (the typical logit swing it induces between sibling
moves). HIGH xgb-gain + ~0 impact = a TRAP (loved globally, useless in-search).

Usage (from aphrodite/):
    ../../../venv/Scripts/python.exe train/probe_v3.py \
        --zip ../../../ladder_replays/replays_6_07.zip \
        --model train/weights/xgb_4p_6_01_6_07_v3_decdrop_noablate.json \
        --games 60 --steps-per-game 4 --budget-ms 150
"""
from __future__ import annotations
import argparse, json, subprocess, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "target" / "release" / "probe_leaves_v3"
DIM = 145
REC = 4 + 4 + 4 * DIM  # search_id:i32, leaf_step:i32, feats[145]


def feature_names():
    names = ["step", "angular_velocity"]
    cur = ["ships_on", "ships_fly", "n_static", "n_orbit", "n_comet",
           "prod_static", "prod_orbit", "neut_closer", "enem_closer"]
    ext = ["e_ships_on", "e_n_static", "e_n_orbit", "e_n_comet",
           "e_prod_static", "e_prod_orbit", "e_neut_closer", "e_enem_closer"]
    names += ["me_" + c for c in cur] + ["me_" + c for c in ext]
    names += ["neut_ships", "neut_n_static", "neut_n_orbit", "neut_n_comet",
              "neut_prod_static", "neut_prod_orbit", "neut_comet_time"]
    names += ["agg_ship_share", "agg_prod_share", "agg_my_n_vuln",
              "agg_my_prod_at_risk", "agg_max_enemy_press", "agg_pw_enemy_press",
              "agg_my_fleet_frac", "agg_ally_disp", "agg_avg_ally_ships",
              "agg_leader_ratio", "agg_opp_spread", "agg_n_alive",
              "agg_tot_prod", "agg_tot_ships", "agg_my_abs_prod"]
    rel = ["pw_my_press_on_k", "pw_k_press_on_me", "centroid_dist", "k_disp",
           "ship_share_vs_k", "prod_share_vs_k", "is_alive"]
    for s in ("o1", "o2", "o3"):
        names += [f"{s}_{c}" for c in cur] + [f"{s}_{c}" for c in ext] + [f"{s}_{r}" for r in rel]
    pairs = ["me>o1", "me>o2", "me>o3", "o1>me", "o1>o2", "o1>o3", "o2>me",
             "o2>o1", "o2>o3", "o3>me", "o3>o1", "o3>o2",
             "me>neut", "o1>neut", "o2>neut", "o3>neut"]
    names += ["inflight_" + p for p in pairs] + ["vuln_" + p for p in pairs]
    assert len(names) == DIM, len(names)
    return names


def block_of(i):
    if i < 2: return "global"
    if i < 11: return "me_cur"
    if i < 19: return "me_ext"
    if i < 26: return "neutral"
    if i < 41: return "aggregate"
    if i < 65: return "opp_o1"
    if i < 89: return "opp_o2"
    if i < 113: return "opp_o3"
    if i < 129: return "inflight_mat"
    return "vuln_mat"


def select_positions(zip_path, n_games, steps_per_game):
    """Yield obs dicts (player+step+planets+fleets) for sampled mid-game 4p states."""
    zf = zipfile.ZipFile(zip_path)
    entries = sorted(n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/"))
    out = []
    used = 0
    for e in entries:
        if used >= n_games:
            break
        try:
            d = json.loads(zf.read(e))
        except Exception:
            continue
        rew = d.get("rewards") or []
        if len(rew) != 4:
            continue
        steps = d.get("steps") or []
        if len(steps) < 40:
            continue
        # sample mid-game fractions (skip opening + decided endgame)
        fracs = np.linspace(0.25, 0.8, steps_per_game)
        for fr in fracs:
            t = int(fr * len(steps))
            row = steps[t]
            if not isinstance(row, list) or len(row) < 4:
                continue
            # pick an alive searching slot (first slot with planets)
            me = None
            for slot in range(4):
                o = row[slot].get("observation") if isinstance(row[slot], dict) else None
                if o and o.get("planets"):
                    me = slot; obs = o; break
            if me is None:
                continue
            out.append({"player": me, "step": t,
                        "planets": obs.get("planets", []), "fleets": obs.get("fleets", []),
                        "angular_velocity": obs.get("angular_velocity", 0.0),
                        "initial_planets": obs.get("initial_planets", []),
                        "comets": obs.get("comets", []), "comet_planet_ids": obs.get("comet_planet_ids", [])})
        used += 1
    return out


def run_probe(positions, model, dump_path, budget_ms):
    env = {**__import__("os").environ,
           "APHRODITE_VALUE_NET_PATH": str(Path(model).resolve()),
           "APHRODITE_DUMP_LEAVES_PATH": str(Path(dump_path).resolve()),
           "APHRODITE_DUMP_FEATURES": "v3",
           "APHRODITE_PROBE_BUDGET_MS": str(budget_ms)}
    Path(dump_path).write_bytes(b"")
    inp = "\n".join(json.dumps(p, separators=(",", ":")) for p in positions).encode()
    print(f"feeding {len(positions)} positions to {BIN.name} (budget {budget_ms}ms)...", flush=True)
    r = subprocess.run([str(BIN)], input=inp, env=env, capture_output=True)
    print(r.stderr.decode()[-400:])


def read_leaves(path):
    raw = Path(path).read_bytes()
    n = len(raw) // REC
    arr = np.frombuffer(raw[:n * REC], dtype=np.uint8).reshape(n, REC)
    sid = arr[:, 0:4].view(np.int32).reshape(n).copy()
    lstep = arr[:, 4:8].view(np.int32).reshape(n).copy()
    feats = arr[:, 8:].view(np.float32).reshape(n, DIM).copy()
    return sid, lstep, feats


def decomp(sid, lstep, X):
    """within-cohort vs total variance per feature; cohort=(search_id, leaf_step)."""
    X = X.astype(np.float64)
    mean = X.mean(0)
    total_var = np.maximum((X * X).mean(0) - mean * mean, 0.0)
    key = sid.astype(np.int64) * 100000 + lstep.astype(np.int64)
    order = np.argsort(key, kind="stable")
    k = key[order]; Xo = X[order]
    bnd = np.flatnonzero(np.diff(k)) + 1
    ssw = np.zeros(X.shape[1]); nw = 0; ncoh = 0
    for sl in np.split(np.arange(len(k)), bnd):
        if sl.size < 2:
            continue
        g = Xo[sl]
        ssw += ((g - g.mean(0)) ** 2).sum(0)
        nw += sl.size; ncoh += 1
    within_var = ssw / max(nw, 1)
    ratio = within_var / np.maximum(total_var, 1e-12)
    return total_var, within_var, ratio, ncoh, len(np.unique(sid))


def shap_impact(model, sid, lstep, X):
    import xgboost as xgb
    bst = xgb.Booster(); bst.load_model(str(model))
    contribs = bst.predict(xgb.DMatrix(X.astype(np.float32)), pred_contribs=True)[:, :DIM]
    _, within, _, _, _ = decomp(sid, lstep, contribs)
    return np.sqrt(within), np.abs(contribs).mean(0)


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--steps-per-game", type=int, default=4)
    p.add_argument("--budget-ms", type=int, default=150)
    p.add_argument("--dump", default=str(HERE / "data/4p/_probe_v3_leaves.bin"))
    p.add_argument("--reuse", action="store_true")
    args = p.parse_args()
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)

    if not args.reuse:
        pos = select_positions(args.zip, args.games, args.steps_per_game)
        run_probe(pos, args.model, args.dump, args.budget_ms)

    sid, lstep, X = read_leaves(args.dump)
    print(f"\nleaves={len(X):,}  searches={len(np.unique(sid)):,}")
    tv, wv, ratio, ncoh, nsearch = decomp(sid, lstep, X)
    impact, mean_abs = shap_impact(args.model, sid, lstep, X)
    names = feature_names()

    # xgb gain
    import xgboost as xgb
    bst = xgb.Booster(); bst.load_model(str(args.model))
    g = bst.get_score(importance_type="gain"); gain = np.zeros(DIM)
    for kk, vv in g.items():
        gain[int(kk[1:])] = vv
    gpct = 100 * gain / max(gain.sum(), 1e-9)

    print(f"cohorts(>=2 siblings)={ncoh:,}  searches_with_leaves={nsearch:,}")
    print("\n=== TOP 30 move-rankers by SHAP within-sibling impact (logit swing) ===")
    print(f"{'#':>3} {'feature':22s} {'impact':>9s} {'sib_ratio':>9s} {'gain%':>6s}  block")
    for c in np.argsort(-impact)[:30]:
        print(f"{c:>3} {names[c]:22s} {impact[c]:9.4f} {ratio[c]:9.3f} {gpct[c]:6.2f}  {block_of(c)}")

    # block-level: total SHAP impact (sum) and mean sib_ratio
    from collections import defaultdict
    bimp = defaultdict(float); bgain = defaultdict(float)
    for c in range(DIM):
        bimp[block_of(c)] += impact[c]; bgain[block_of(c)] += gpct[c]
    print("\n=== by block: summed SHAP impact (move-ranking power) vs gain% ===")
    print(f"{'block':14s} {'sum_impact':>11s} {'gain%':>7s}")
    for b in sorted(bimp, key=lambda x: -bimp[x]):
        print(f"{b:14s} {bimp[b]:11.4f} {bgain[b]:7.2f}")
    new = sum(bimp[b] for b in ("opp_o1", "opp_o2", "opp_o3", "inflight_mat", "vuln_mat"))
    tot = sum(bimp.values())
    print(f"\nNEW v3 features (per-opp + 2 matrices): {100*new/max(tot,1e-9):.1f}% of total move-ranking impact")
    mat = bimp["inflight_mat"] + bimp["vuln_mat"]
    print(f"  of which the two MATRICES: {100*mat/max(tot,1e-9):.1f}% of total impact")

    # TRAP check: high gain, ~0 impact
    print("\n=== TRAPS (gain% high but ~0 sibling impact) ===")
    imax = max(impact.max(), 1e-9)
    for c in np.argsort(-gpct)[:8]:
        tag = "  <-- TRAP" if impact[c] / imax < 0.05 else ""
        print(f"  {names[c]:22s} gain%={gpct[c]:5.2f} impact={impact[c]:.4f} (rel {impact[c]/imax:.2f}){tag}")


if __name__ == "__main__":
    main()
