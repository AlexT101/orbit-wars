"""Final 4p gauntlet for tuned apollo configs.

Takes the top-K configs from a 4p study (apollo_4p_v1) and plays each over a
fixed held-out seed set, distributed EQUALLY across the 3 4p rosters
(games-per-roster each), on identical seeds (common random numbers) so configs
are compared paired. No control / no 2p baseline. Reports each config's pooled
1st-place rate with a Wilson CI, per-roster rates, and pairwise paired deltas.

Usage (repo root, in the venv), AFTER the 4p study has completed:
    python bots/mine/apollo/tuning/gauntlet_4p.py --study-name apollo_4p_v1 \
        --top 8 --games-per-roster 250
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the 4p tuner's machinery (sets APOLLO_CONFIG + APOLLO_CONFIG_4P, imports
# run_batched_4p, defines the rosters and the panic-safe match wrapper).
import tune_4p  # noqa: E402
from tune_4p import (  # noqa: E402
    CHUNK_GAMES,
    CONFIG_4P_PATH,
    DEFAULT_ROSTERS,
    OUT_DIR,
    safe_run_match_job_4p,
    write_config_4p,
)
from run_batched_4p import (  # noqa: E402
    bot_entry,
    match_winner,
    reorder_by_input,
    slot_order_for_seed,
)


def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, p, (centre + margin) / denom)


def paired_delta(diffs, z=1.96):
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else 0.0
    return (mean - z * se, mean, mean + z * se, n)


def select_top_configs(study_name, k):
    """Top-k completed 4p configs ranked by the LOWER bound of their 1st-place CI."""
    path = OUT_DIR / f"{study_name}_trials.jsonl"
    if not path.exists():
        raise SystemExit(f"no trials log at {path}")
    best = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("status") != "complete":
            continue
        wins = r["cum"]["wins"]
        n = r["cum_nonerror"]
        lcb = wilson(wins, n)[0]
        key = json.dumps(r["config"], sort_keys=True)
        if key not in best or lcb > best[key][0]:
            best[key] = (lcb, r["trial"], r["config"], r["win_rate"])
    ranked = sorted(best.values(), key=lambda t: t[0], reverse=True)
    return [(f"trial{t[1]}", t[2], t[3]) for t in ranked[:k]]


def play_block_4p(apollo_path, roster_paths, seeds, threads):
    """Play apollo (slot 0) vs one roster over explicit seeds. Returns
    {seed: {"score": 1.0/0.0/None, "outcome": w/l/d/e, "ms": float|None}}.
    score: 1.0 = sole 1st place; draws and losses are 0.0 (non-wins)."""
    jobs = [([apollo_path, *roster_paths], s, i, slot_order_for_seed(s))
            for i, s in enumerate(seeds)]
    out = {}
    for i in range(0, len(jobs), CHUNK_GAMES):
        chunk = jobs[i:i + CHUNK_GAMES]
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futures = [ex.submit(safe_run_match_job_4p, j) for j in chunk]
            for fut in as_completed(futures):
                res = fut.result()
                if res[0] == "error":
                    out[seeds[res[1]]] = {"score": None, "outcome": "e", "ms": None}
                    continue
                _idx, seed, slot_order, rewards, _steps, avg_ms = res
                rewards = reorder_by_input(rewards, slot_order)
                avg_ms = reorder_by_input(avg_ms, slot_order)
                w = match_winner(rewards)
                if w is None:
                    score, oc = None, "e"
                elif w == 0:
                    score, oc = 1.0, "w"
                elif w == "draw":
                    score, oc = 0.0, "d"
                else:
                    score, oc = 0.0, "l"
                out[seed] = {"score": score, "outcome": oc, "ms": avg_ms[0]}
    return out


def play_config_4p(apollo_path, cfg, roster_blocks, threads):
    """Write cfg to config_4p.json, then play every roster's seed block."""
    write_config_4p(cfg)
    return {name: play_block_4p(apollo_path, paths, seeds, threads)
            for name, paths, seeds in roster_blocks}


def summarise(name, by_roster):
    pooled = {"w": 0, "l": 0, "d": 0, "e": 0}
    ms_vals = []
    per = {}
    for roster, games in by_roster.items():
        rec = {"w": 0, "l": 0, "d": 0, "e": 0}
        for g in games.values():
            rec[g["outcome"]] += 1
            if g["ms"] is not None:
                ms_vals.append(g["ms"])
        nonerr = rec["w"] + rec["l"] + rec["d"]
        per[roster] = {**rec, "win_rate": (rec["w"] / nonerr) if nonerr else None}
        for kk in pooled:
            pooled[kk] += rec[kk]
    nonerr = pooled["w"] + pooled["l"] + pooled["d"]
    lo, mid, hi = wilson(pooled["w"], nonerr)
    return {"name": name, "pooled": pooled, "nonerror": nonerr,
            "win_rate": mid, "wr_lo": lo, "wr_hi": hi,
            "avg_ms": (sum(ms_vals) / len(ms_vals)) if ms_vals else 0.0,
            "per_roster": per}


def pairwise_deltas(results, names):
    out = {}
    for ai, a in enumerate(names):
        for b in names[ai + 1:]:
            diffs = []
            for roster, games in results[a].items():
                gb = results[b].get(roster, {})
                for seed, g in games.items():
                    h = gb.get(seed)
                    if g["score"] is None or h is None or h["score"] is None:
                        continue
                    diffs.append(g["score"] - h["score"])
            lo, mid, hi, n = paired_delta(diffs)
            out[f"{a} - {b}"] = {"mean": mid, "lo": lo, "hi": hi, "n": n,
                                 "significant": bool(lo > 0 or hi < 0)}
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Final paired 4p gauntlet.")
    parser.add_argument("--study-name", default="apollo_4p_v1")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--games-per-roster", type=int, default=250,
                        help="Games per roster per config (total = this x 3).")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=20_000_000,
                        help="Start of the held-out seed range (disjoint from training).")
    parser.add_argument("--configs", default=None,
                        help="JSON file {name: config} to test instead of the study top-k.")
    args = parser.parse_args()

    apollo_path = bot_entry("apollo")
    if not apollo_path.is_file():
        parser.error(f"apollo not found at {apollo_path}")
    config_snapshot = json.loads(CONFIG_4P_PATH.read_text())

    if args.configs:
        raw = json.loads(Path(args.configs).read_text())
        candidates = [(str(n), c) for n, c in raw.items()]
    else:
        candidates = [(n, c) for n, c, _wr in select_top_configs(args.study_name, args.top)]
    if not candidates:
        raise SystemExit("no configs selected; check the study name / trials log.")

    # Equal allocation: each roster gets games-per-roster contiguous, shared seeds.
    roster_blocks = []
    cursor = args.seed_base
    g = args.games_per_roster
    for name, bots in DEFAULT_ROSTERS:
        paths = [bot_entry(b) for b in bots]
        for b, p in zip(bots, paths):
            if not p.is_file():
                raise SystemExit(f"bot '{b}' not found at {p}")
        roster_blocks.append((name, paths, list(range(cursor, cursor + g))))
        cursor += g
    total = g * len(DEFAULT_ROSTERS)

    print(f"4p gauntlet: {len(candidates)} configs x {total} games "
          f"({g}/roster, equal), CRN, threads={args.threads}")
    print(f"Rosters: {[n for n, _ in DEFAULT_ROSTERS]}")

    results = {}
    for name, cfg in candidates:
        print(f"  running {name} ...", flush=True)
        results[name] = play_config_4p(apollo_path, cfg, roster_blocks, args.threads)

    summaries = sorted((summarise(n, results[n]) for n, _ in candidates),
                       key=lambda s: s["win_rate"], reverse=True)
    rosters = [n for n, _ in DEFAULT_ROSTERS]
    header = f"{'config':>10} | {'pooled 1st (95% CI)':>22} | ms | " + " ".join(f"{r[:8]:>8}" for r in rosters)
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        wr = f"{s['win_rate']*100:5.1f}% [{s['wr_lo']*100:4.1f},{s['wr_hi']*100:4.1f}]"
        cells = " ".join(
            (f"{(s['per_roster'][r]['win_rate'] or 0)*100:7.1f}%" if r in s["per_roster"] else "   -    ")
            for r in rosters)
        print(f"{s['name']:>10} | {wr:>22} | {s['avg_ms']:3.0f} | {cells}")

    pw = pairwise_deltas(results, [s["name"] for s in summaries])
    if pw:
        print("\nPairwise paired delta (A - B, mean 1st-place diff, 95% CI):")
        for k, v in pw.items():
            sig = "*" if v["significant"] else " "
            print(f"  {k:>26}: {v['mean']*100:+5.1f} [{v['lo']*100:+5.1f},{v['hi']*100:+5.1f}]{sig}  (n={v['n']})")
        print("  (* = the two configs differ significantly at 95%.)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = OUT_DIR / f"{args.study_name}_gauntlet_{stamp}.json"
    out_json.write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(), "study": args.study_name,
        "games_per_roster": g, "seed_base": args.seed_base,
        "rosters": rosters, "configs": dict(candidates),
        "summaries": summaries, "pairwise": pw,
    }, indent=2))
    if summaries:
        winner = summaries[0]
        best_path = OUT_DIR / f"{args.study_name}_gauntlet_best.json"
        best_path.write_text(json.dumps(
            {"name": winner["name"], "win_rate": winner["win_rate"],
             "config": dict(candidates)[winner["name"]]}, indent=2))
        print(f"\nTop: {winner['name']} ({winner['win_rate']*100:.1f}% 1st-place pooled)")
        print(f"Saved: {best_path.name}, {out_json.name}")

    write_config_4p(config_snapshot)
    print("Restored config_4p.json.")


if __name__ == "__main__":
    main()
