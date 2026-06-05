"""Search-instrumentation probe + feature-quality analysis for aphrodite.

Runs a BATCH of 2p matches (several opponents x several games), captures every
DUCT leaf's SummaryV2 feature vector (tagged by the search + depth it belongs
to) plus per-turn tree-shape stats, then compiles everything into a per-feature
report of which inputs actually help the *search*.

Why this metric and not XGBoost importance
------------------------------------------
Inside one DUCT search every leaf descends from the same root and the net's only
job is to *rank sibling leaves*. A feature can dominate global accuracy yet be
near-constant across the leaves DUCT compares (e.g. `step`, empire-wide ratios) —
it shifts the absolute score but can't separate this turn's candidate moves. We
measure, per feature, the fraction of its variance that occurs *between leaves at
the same depth of the same search* (a "sibling cohort"):

    sib_ratio = within_cohort_var / total_var  in [0, 1]

Low sib_ratio = passenger (can't rank moves); high = genuine move-discriminator.
Grouping by (match, search_step, leaf_step) removes the drift down DUCT's deep
paths that would otherwise inflate the signal. We cross-reference XGBoost gain:
the sharpest trap is HIGH gain + LOW sib_ratio (loved globally, useless in-search).

Usage
-----
    # run the full batch then analyze
    python train/probe.py --weights train/weights/xgb_2p_full65.json \
        --model train/weights/xgb_2p_full65.json \
        --opponents producer owheuristic apollo_fast apollo hellburner prometheus_v2 \
        --games 4 --budget-ms 300 --max-steps 150 --out-dir train/data/2p/_probe

    # re-analyze an existing batch without replaying
    python train/probe.py --reuse --model train/weights/xgb_2p_full65.json \
        --out-dir train/data/2p/_probe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # type: ignore

SUMMARY_V2_DIM = 65
LEAF_RECORD_BYTES = 4 + 4 + 4 * SUMMARY_V2_DIM  # search_step:i32, leaf_step:i32, feats

FEATURE_NAMES = [
    # me_cur [0:9]
    "me_ships_on_planets", "me_ships_flying", "me_n_static", "me_n_orbit",
    "me_n_comet", "me_prod_static", "me_prod_orbit", "me_n_neutrals_closer",
    "me_n_enemies_closer",
    # opp_cur [9:18]
    "opp_ships_on_planets", "opp_ships_flying", "opp_n_static", "opp_n_orbit",
    "opp_n_comet", "opp_prod_static", "opp_prod_orbit", "opp_n_neutrals_closer",
    "opp_n_enemies_closer",
    # me_ext [18:26]
    "me_ext_ships_on_planets", "me_ext_n_static", "me_ext_n_orbit",
    "me_ext_n_comet", "me_ext_prod_static", "me_ext_prod_orbit",
    "me_ext_n_neutrals_closer", "me_ext_n_enemies_closer",
    # opp_ext [26:34]
    "opp_ext_ships_on_planets", "opp_ext_n_static", "opp_ext_n_orbit",
    "opp_ext_n_comet", "opp_ext_prod_static", "opp_ext_prod_orbit",
    "opp_ext_n_neutrals_closer", "opp_ext_n_enemies_closer",
    # neut [34:41]
    "neut_ships", "neut_n_static", "neut_n_orbit", "neut_n_comet",
    "neut_prod_static", "neut_prod_orbit", "neut_comet_time",
    # rel [41:65]
    "step", "avg_ally_ships", "avg_enemy_ships", "avg_ally_support",
    "avg_enemy_support", "num_my_vuln", "num_enemy_vuln", "ship_share",
    "production_share", "my_prod_at_risk", "enemy_prod_at_opportunity",
    "max_enemy_pressure", "max_ally_pressure", "centroid_dist",
    "my_fleet_fraction", "enemy_fleet_fraction", "pw_enemy_pressure",
    "pw_ally_pressure", "ally_dispersion", "enemy_dispersion",
    "my_strength_rank", "leader_strength_ratio", "opponent_strength_spread",
    "n_alive",
]
assert len(FEATURE_NAMES) == SUMMARY_V2_DIM


# ───────────────────────────── match running ──────────────────────────────

def run_one_match(weights, opponent, seed, budget_ms, max_steps, leaves_cap,
                  match_dir, swap_side):
    """Run one 2p match aphrodite vs opponent, dumping into match_dir."""
    from engine_parity_checker.candidates.rust import RustEngine

    match_dir.mkdir(parents=True, exist_ok=True)
    leaves_path = match_dir / "leaves.bin"
    tree_stats_path = match_dir / "tree_stats.csv"
    leaves_path.write_bytes(b"")

    aphro_slot = 1 if swap_side else 0
    daemon = collect.AphroditeDaemon(
        dump_path=None, budget_ms=budget_ms, weights_path=weights,
        weights_2p_path=weights, leaves_path=leaves_path,
        tree_stats_path=tree_stats_path, leaves_cap=leaves_cap,
    )
    opp_fn, opp_mod = collect.load_other_agent(opponent)
    agents = [None, None]
    agents[aphro_slot] = daemon
    agents[1 - aphro_slot] = opp_fn

    engine = RustEngine()
    obs = engine.reset(seed, 2)
    steps = min(max_steps, collect.MAX_STEPS)
    t0 = time.time()
    try:
        for _ in range(steps):
            acts = [agents[i](obs[i].as_dict()) for i in range(2)]
            obs, done = engine.step(acts)
            if done:
                break
        snap = engine.snapshot()
        rewards = [float(x) for x in (snap.rewards or [0.0, 0.0])]
    finally:
        try:
            daemon.close()
        except Exception:
            pass
        try:
            collect.teardown_other(opp_mod)
        except Exception:
            pass
    aphro_r = rewards[aphro_slot]
    return rewards, aphro_r, aphro_slot, time.time() - t0


def _match_job(job):
    """Top-level worker (picklable for ProcessPoolExecutor). Runs one match,
    returns its manifest entry. Each match writes to its own dir, so workers
    never collide."""
    (weights, opp, seed, budget_ms, max_steps, leaves_cap, match_dir, swap,
     mid, gi, out_dir) = job
    match_dir = Path(match_dir)
    base = dict(match_id=mid, opp=opp, game=gi, seed=seed,
                dir=str(match_dir.relative_to(Path(out_dir))))
    try:
        rewards, aphro_r, slot, dt = run_one_match(
            weights, opp, seed, budget_ms, max_steps, leaves_cap, match_dir, swap)
        best = max(rewards)
        outcome = ("W" if aphro_r == best and rewards.count(best) == 1
                   else "T" if aphro_r == best else "L")
        base.update(rewards=rewards, aphro_slot=slot, aphro_r=aphro_r,
                    outcome=outcome, dt=dt)
    except Exception as e:
        base.update(failed=str(e), tb=traceback.format_exc())
    return base


def _log_match(m, games):
    if "failed" in m:
        print(f"[match {m['match_id']}] {m['opp']} g{m['game']+1} "
              f"seed={m['seed']} !! FAILED: {m['failed']}", flush=True)
    else:
        print(f"[match {m['match_id']}] {m['opp']} g{m['game']+1}/{games} "
              f"seed={m['seed']} -> {m['outcome']} rewards={m['rewards']} "
              f"{m['dt']:.1f}s", flush=True)


def run_batch(weights, opponents, games, base_seed, budget_ms, max_steps,
              leaves_cap, out_dir, threads):
    jobs = []
    mid = 0
    for opp in opponents:
        for gi in range(games):
            seed = base_seed + gi
            swap = (gi % 2 == 1)
            match_dir = out_dir / opp / f"g{gi}_s{seed}"
            jobs.append((weights, opp, seed, budget_ms, max_steps, leaves_cap,
                         str(match_dir), swap, mid, gi, str(out_dir)))
            mid += 1

    threads = max(1, min(threads, len(jobs)))
    print(f"running {len(jobs)} matches on {threads} worker(s) "
          f"(budget={budget_ms}ms, max_steps={max_steps})"
          + ("  [parallel: per-turn iters degrade under contention; fine for "
             "feature distributions]" if threads > 1 else ""), flush=True)

    manifest = []
    if threads == 1:
        for job in jobs:
            m = _match_job(job)
            _log_match(m, games)
            manifest.append(m)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(_match_job, j) for j in jobs]
            for fut in as_completed(futs):
                m = fut.result()
                _log_match(m, games)
                manifest.append(m)
    manifest.sort(key=lambda m: m["match_id"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ───────────────────────────── dump reading ───────────────────────────────

def read_leaves(path: Path):
    raw = path.read_bytes() if path.exists() else b""
    n = len(raw) // LEAF_RECORD_BYTES
    if n == 0:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32),
                np.zeros((0, SUMMARY_V2_DIM), np.float32))
    arr = np.frombuffer(raw[: n * LEAF_RECORD_BYTES], dtype=np.uint8).reshape(n, LEAF_RECORD_BYTES)
    search_step = arr[:, 0:4].view(np.int32).reshape(n).copy()
    leaf_step = arr[:, 4:8].view(np.int32).reshape(n).copy()
    feats = arr[:, 8:].view(np.float32).reshape(n, SUMMARY_V2_DIM).copy()
    return search_step, leaf_step, feats


def read_tree_stats(path: Path):
    if not path.exists():
        return None
    import csv
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    cols = ["iters", "nodes", "leaves", "max_depth", "root_visits", "my_K", "opp_K"]
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


# ─────────────────────────── variance accumulator ─────────────────────────

class VarAccum:
    """Streaming per-feature variance decomposition over many matches.

    Total variance uses all leaves; within-cohort variance pools the
    sum-of-squares of (match, search_step, leaf_step) cohorts of size >= 2 —
    i.e. leaves at the same depth of the same search (the siblings DUCT ranks).
    """

    def __init__(self, d=SUMMARY_V2_DIM):
        self.d = d
        self.sum = np.zeros(d, np.float64)
        self.sumsq = np.zeros(d, np.float64)
        self.n = 0
        self.ss_within = np.zeros(d, np.float64)
        self.n_within = 0
        self.n_cohorts = 0
        self.n_leaves = 0
        self.n_searches = set()

    def add_match(self, match_id, search_step, leaf_step, feats):
        if feats.shape[0] == 0:
            return
        fx = feats.astype(np.float64)
        self.sum += fx.sum(axis=0)
        self.sumsq += (fx * fx).sum(axis=0)
        self.n += fx.shape[0]
        self.n_leaves += fx.shape[0]
        for s in np.unique(search_step):
            self.n_searches.add((match_id, int(s)))
        key = search_step.astype(np.int64) * 100000 + leaf_step.astype(np.int64)
        order = np.argsort(key, kind="stable")
        k = key[order]
        fxo = fx[order]
        bnd = np.flatnonzero(np.diff(k)) + 1
        for sl in np.split(np.arange(len(k)), bnd):
            if sl.size < 2:
                continue
            xg = fxo[sl]
            self.ss_within += ((xg - xg.mean(axis=0)) ** 2).sum(axis=0)
            self.n_within += sl.size
            self.n_cohorts += 1

    def finalize(self):
        mean = self.sum / max(self.n, 1)
        total_var = np.maximum(self.sumsq / max(self.n, 1) - mean * mean, 0.0)
        within_var = self.ss_within / max(self.n_within, 1)
        ratio = within_var / np.maximum(total_var, 1e-12)
        return dict(total_var=total_var, within_var=within_var, ratio=ratio,
                    global_std=np.sqrt(total_var), sib_std=np.sqrt(within_var),
                    n_leaves=self.n_leaves, n_cohorts=self.n_cohorts,
                    n_searches=len(self.n_searches))


def compute_shap_impact(out_dir, manifest, model_path):
    """The rigorous metric: how much each feature actually swings the model's
    leaf score *between sibling leaves*.

    For every leaf we take the model's per-feature SHAP contribution (in logit
    units; XGBoost `pred_contribs=True`), then pool the within-cohort variance of
    those contributions exactly like the feature-variance metric. `impact` =
    sqrt(within-cohort var of contribution) = the typical logit swing this
    feature induces among the moves DUCT compares. This is unit-free across
    features (all in logits), so it directly answers "does this feature help rank
    moves" — resolving the ratio-vs-scale ambiguity of raw-feature variance.

    `mean_abs_contrib` (global SHAP importance) is reported alongside: a feature
    with high mean_abs_contrib but ~0 impact is a TRAP (drives the global score
    but not sibling ranking).
    """
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    acc = VarAccum()
    sum_abs = np.zeros(SUMMARY_V2_DIM, np.float64)
    n_abs = 0
    for m in manifest:
        if "failed" in m:
            continue
        ss, ls, fx = read_leaves(out_dir / m["dir"] / "leaves.bin")
        if fx.shape[0] == 0:
            continue
        contribs = bst.predict(xgb.DMatrix(fx), pred_contribs=True)
        contribs = contribs[:, :SUMMARY_V2_DIM].astype(np.float32)
        acc.add_match(m["match_id"], ss, ls, contribs)
        sum_abs += np.abs(contribs).sum(axis=0)
        n_abs += contribs.shape[0]
    res = acc.finalize()
    return res["sib_std"], sum_abs / max(n_abs, 1)


def load_xgb_gain(model_path: Path):
    try:
        import xgboost as xgb
    except Exception:
        print(f"(xgboost not importable; skipping gain)")
        return None
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    score = bst.get_score(importance_type="gain")
    gain = np.zeros(SUMMARY_V2_DIM, dtype=np.float64)
    for k, v in score.items():
        idx = int(k[1:])
        if idx < SUMMARY_V2_DIM:
            gain[idx] = v
    return gain


# ────────────────────────────── analysis ──────────────────────────────────

def analyze(out_dir: Path, model_path, drop_ratio):
    manifest = json.loads((out_dir / "manifest.json").read_text())
    ok = [m for m in manifest if "failed" not in m]
    opponents = sorted({m["opp"] for m in ok})
    print(f"\n=== batch: {len(ok)}/{len(manifest)} matches ok over "
          f"{len(opponents)} opponents ===")

    overall = VarAccum()
    per_opp = {opp: VarAccum() for opp in opponents}
    ts_overall = {c: [] for c in ["iters", "nodes", "leaves", "max_depth", "my_K", "opp_K"]}
    ts_per_opp = {opp: {c: [] for c in ts_overall} for opp in opponents}
    wins = {opp: [0, 0, 0] for opp in opponents}  # W,L,T

    for m in ok:
        md = out_dir / m["dir"]
        ss, ls, fx = read_leaves(md / "leaves.bin")
        overall.add_match(m["match_id"], ss, ls, fx)
        per_opp[m["opp"]].add_match(m["match_id"], ss, ls, fx)
        ts = read_tree_stats(md / "tree_stats.csv")
        if ts is not None:
            for c in ts_overall:
                ts_overall[c].append(ts[c])
                ts_per_opp[m["opp"]][c].append(ts[c])
        oc = m.get("outcome")
        if oc == "W":
            wins[m["opp"]][0] += 1
        elif oc == "L":
            wins[m["opp"]][1] += 1
        elif oc == "T":
            wins[m["opp"]][2] += 1

    # ---- W/L/T + tree shape per opponent ----
    print("\n=== outcomes + tree shape (per opponent) ===")
    print(f"{'opponent':20s} {'W/L/T':>8s} {'my_K':>6s} {'opp_K':>6s} "
          f"{'nodes':>8s} {'leaves':>7s} {'depth':>6s} {'iters':>7s}")
    def _mean(lst):
        return float(np.concatenate(lst).mean()) if lst else float("nan")
    for opp in opponents:
        w, l, t = wins[opp]
        po = ts_per_opp[opp]
        print(f"{opp:20s} {f'{w}/{l}/{t}':>8s} "
              f"{_mean(po['my_K']):6.2f} {_mean(po['opp_K']):6.2f} "
              f"{_mean(po['nodes']):8.0f} {_mean(po['leaves']):7.0f} "
              f"{_mean(po['max_depth']):6.1f} {_mean(po['iters']):7.0f}")
    print(f"{'OVERALL':20s} {'':>8s} "
          f"{_mean(ts_overall['my_K']):6.2f} {_mean(ts_overall['opp_K']):6.2f} "
          f"{_mean(ts_overall['nodes']):8.0f} {_mean(ts_overall['leaves']):7.0f} "
          f"{_mean(ts_overall['max_depth']):6.1f} {_mean(ts_overall['iters']):7.0f}")

    # ---- feature ranking ----
    res = overall.finalize()
    opp_res = {opp: per_opp[opp].finalize() for opp in opponents}
    gain = load_xgb_gain(model_path) if model_path else None
    gain_norm = gain / max(gain.max(), 1e-12) if gain is not None else None

    ratio = res["ratio"]
    sib_std = res["sib_std"]
    gstd = res["global_std"]
    # consensus: in how many opponents is this feature a passenger?
    n_opp_pass = np.zeros(SUMMARY_V2_DIM, int)
    for opp in opponents:
        r = opp_res[opp]["ratio"]
        gs = opp_res[opp]["global_std"]
        for c in range(SUMMARY_V2_DIM):
            if gs[c] >= 1e-6 and r[c] < drop_ratio:
                n_opp_pass[c] += 1

    order = np.argsort(-ratio)
    print(f"\n=== feature ranking (pooled: {res['n_leaves']:,} leaves, "
          f"{res['n_cohorts']:,} sibling cohorts, {res['n_searches']:,} searches) ===")
    print("sib_ratio = same-depth/same-search var / total var (0=passenger, 1=move-ranker)")
    print("pass@opp  = # of {} opponents where it's a passenger".format(len(opponents)))
    hdr = (f"{'#':>3} {'feature':28s} {'sib_ratio':>9s} {'sib_std':>10s} "
           f"{'global_std':>10s} {'pass@opp':>8s}")
    if gain_norm is not None:
        hdr += f" {'gain%':>7s}"
    print(hdr)
    for c in order:
        line = (f"{c:>3} {FEATURE_NAMES[c]:28s} {ratio[c]:9.3f} {sib_std[c]:10.3f} "
                f"{gstd[c]:10.3f} {n_opp_pass[c]:5d}/{len(opponents):<2d}")
        if gain_norm is not None:
            line += f" {100*gain_norm[c]:7.1f}"
        print(line)

    # ---- SHAP-contribution impact (the rigorous, scale-free ranking) ----
    impact = mean_abs = None
    if model_path:
        print("\ncomputing SHAP-contribution impact (per-feature logit swing "
              "between sibling leaves)...", flush=True)
        impact, mean_abs = compute_shap_impact(out_dir, manifest, model_path)
        iorder = np.argsort(-impact)
        print("\n=== feature IMPACT ranking (SHAP within-cohort swing, logits) ===")
        print("impact   = typical logit swing this feature causes BETWEEN sibling "
              "leaves (move-ranking power)")
        print("mean|shap| = global SHAP importance; HIGH mean|shap| + ~0 impact = TRAP")
        print(f"{'#':>3} {'feature':28s} {'impact':>10s} {'mean|shap|':>10s} "
              f"{'sib_std':>10s} {'pass@opp':>8s}")
        for c in iorder:
            print(f"{c:>3} {FEATURE_NAMES[c]:28s} {impact[c]:10.4f} "
                  f"{mean_abs[c]:10.4f} {sib_std[c]:10.3f} "
                  f"{n_opp_pass[c]:5d}/{len(opponents):<2d}")

    # ---- recommendation ----
    const_cols = [c for c in range(SUMMARY_V2_DIM) if gstd[c] < 1e-6]
    maj = (len(opponents) + 1) // 2
    print(f"\n=== recommendation ===")
    if const_cols:
        print("CONSTANT in this pool (2p-degenerate): "
              + ", ".join(f"{c}:{FEATURE_NAMES[c]}" for c in const_cols))
    if impact is not None:
        imax = max(impact.max(), 1e-9)
        rel = impact / imax
        # A feature can only genuinely rank siblings if its VALUE varies between
        # them. Where sib_ratio ~ 0 the feature is constant within a cohort, so
        # any attributed SHAP impact is TreeSHAP interaction leakage, not real
        # ranking power -> safe to drop regardless of attributed impact.
        cohort_const = [c for c in range(SUMMARY_V2_DIM)
                        if c not in const_cols and ratio[c] < 1e-3]
        low_impact = [c for c in range(SUMMARY_V2_DIM)
                      if c not in const_cols and rel[c] < 0.02]
        drop = sorted(set(low_impact) | set(cohort_const))
        print(f"DROP (near-zero ranking impact OR constant within cohort):\n  "
              + ", ".join(f"{c}:{FEATURE_NAMES[c]}" for c in drop))
        print(f"  --zero-cols {','.join(str(c) for c in drop)}")
        cc = [c for c in cohort_const if c not in low_impact]
        if cc:
            print("  (constant-within-cohort, attributed impact is interaction "
                  "leakage): " + ", ".join(f"{c}:{FEATURE_NAMES[c]}" for c in cc))
        keep = list(np.argsort(-impact)[:15])
        print("KEEP / top move-rankers (by SHAP impact):\n  "
              + ", ".join(f"{c}:{FEATURE_NAMES[c]}" for c in keep))
        print("NOTE: impact = magnitude of influence, NOT correctness. A "
              "high-impact feature mis-calibrated on the leaf distribution can "
              "actively hurt — only a gauntlet confirms a drop/keep.")
    else:
        # Fallback: variance-ratio heuristic (bounded features only; noisy for
        # high-dynamic-range raw counts — pass --model for the SHAP metric).
        drop = [c for c in range(SUMMARY_V2_DIM)
                if c not in const_cols and ratio[c] < drop_ratio and n_opp_pass[c] >= maj]
        print("(no --model: variance-ratio heuristic only; pass --model for the "
              "rigorous SHAP-impact ranking)")
        print("LOW sib_ratio passengers:\n  "
              + ", ".join(f"{c}:{FEATURE_NAMES[c]}" for c in sorted(drop)))

    npz = out_dir / "probe_summary.npz"
    np.savez_compressed(
        npz, ratio=ratio, sib_std=sib_std, global_std=gstd,
        n_opp_pass=n_opp_pass, gain=(gain if gain is not None else np.zeros(SUMMARY_V2_DIM)),
        impact=(impact if impact is not None else np.zeros(SUMMARY_V2_DIM)),
        mean_abs_shap=(mean_abs if mean_abs is not None else np.zeros(SUMMARY_V2_DIM)),
        opponents=np.array(opponents),
        per_opp_ratio=np.stack([opp_res[o]["ratio"] for o in opponents]),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"\nsaved -> {npz}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path,
                   default=Path("train/weights/xgb_2p_full65.json"),
                   help="value net aphrodite plays under during the probe")
    p.add_argument("--model", type=Path,
                   default=Path("train/weights/xgb_2p_full65.json"),
                   help="xgb json whose per-feature gain is cross-referenced")
    p.add_argument("--opponents", nargs="+",
                   default=["producer", "owheuristic", "apollo_fast", "apollo",
                            "hellburner", "prometheus_v2"])
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--seed", type=int, default=1, help="base seed (game i uses seed+i)")
    p.add_argument("--budget-ms", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--leaves-cap", type=int, default=1500,
                   help="max leaves dumped per search (0 = all)")
    p.add_argument("--out-dir", type=Path, default=Path("train/data/2p/_probe"))
    p.add_argument("--threads", type=int, default=max(1, (__import__("os").cpu_count() or 2) // 2),
                   help="matches to run concurrently in worker processes. Each "
                        "match uses ~2 cores; set as high as your machine allows.")
    p.add_argument("--reuse", action="store_true",
                   help="skip matches; analyze existing dumps in --out-dir")
    p.add_argument("--drop-ratio", type=float, default=0.05)
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.reuse:
        weights = str(args.weights.resolve())
        t0 = time.time()
        run_batch(weights, args.opponents, args.games, args.seed,
                  args.budget_ms, args.max_steps, args.leaves_cap, out_dir,
                  args.threads)
        print(f"\nbatch done in {time.time() - t0:.0f}s", flush=True)

    analyze(out_dir, args.model, args.drop_ratio)


if __name__ == "__main__":
    main()
