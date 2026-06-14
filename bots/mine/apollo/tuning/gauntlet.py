"""Final gauntlet for the tuned apollo configs.

Takes the top-K configs from a tuning study plus the current default config (as a
control), and plays them all against a fixed opponent set on the SAME held-out
seeds (common random numbers), so configs are compared *paired* per seed — the
biggest variance reduction available. Reports pooled win rate with confidence
intervals and each config's paired improvement over the default.

Game allocation
---------------
A config plays `--total-games` games TOTAL (default 500), split across opponents
(NOT 500 per opponent). The split is weighted toward the strongest / most
relevant bots and scaled by our win rate against each: weaker matchups (lower
win rate) get more games because they are both more informative and the
opponents we care about most. Specifically:

    weight(opp) = priority(opp) * (1 - assumed_win_rate(opp))

so e.g. apollo_baseline (≈50% win rate) gets many games while simpleagent
(≈85%) gets few — but every opponent keeps a small floor so a regression is
still detectable. Every config (and the control) plays the identical per-opponent
seed blocks, enabling the paired comparison.

Usage (from repo root, in the venv) — run AFTER a study has completed:
    python bots/mine/apollo/tuning/gauntlet.py --study-name apollo_v2 --top 5

Nothing here changes the study; it only reads its trials log and replays configs.
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

# Reuse the tuner's machinery (also sets APOLLO_CONFIG and sys.path for workers).
import tune  # noqa: E402
from tune import (  # noqa: E402
    CONFIG_PATH,
    OUT_DIR,
    bot_entry,
    reorder_by_input,
    safe_run_match_job,
    slot_order_for_seed,
    write_config,
)

# The committed default config (the control). Keep in sync with config.json.
DEFAULT_CONFIG = {
    "rotation_look_ahead_turns": 10,
    "offset_lookahead": 15,
    "enemy_offset_lookahead": 5,
    "reinforcement_pressure_turns": 20,
    "reinforcement_pressure_decay": 0.5,
    "frontier_pressure_ratio": 1.4,
    "ally_pressure_ratio": 0.8,
    "horizon": 30,
    "max_distance": 38.0,
}

# Opponent universe (confirmed-stable bots only). priority = relevance/strength
# tier; win_rate = apollo's approximate current win rate vs that bot (used only
# to weight the game split — refresh from study validation if it drifts).
#   name, priority, assumed_win_rate
OPPONENTS = [
    ("producer_v2",     1.00, 0.60),
    ("apollo_baseline", 1.00, 0.50),
    ("producer",        0.60, 0.75),
    ("simpleagent",     0.50, 0.85),
    ("owheuristic",     0.25, 0.90),
]


# ---- stats helpers ----------------------------------------------------------
def wilson(wins, n, z=1.96):
    """Wilson score interval (low, mid, high) for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, p, (centre + margin) / denom)


def paired_delta(diffs, z=1.96):
    """Mean per-game score difference vs control and its 95% CI."""
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else 0.0
    return (mean - z * se, mean, mean + z * se, n)


# ---- game running -----------------------------------------------------------
def play_block(executor, apollo_path, opp_path, seeds):
    """Play apollo (P0) vs one opponent over an explicit list of seeds.
    Returns {seed: {"score": 1.0/0.5/0.0/None, "ms": float|None}} (apollo's view;
    score None = errored game)."""
    jobs = [([apollo_path, opp_path], s, i, slot_order_for_seed(s))
            for i, s in enumerate(seeds)]
    out = {}
    futures = [executor.submit(safe_run_match_job, j) for j in jobs]
    for fut in as_completed(futures):
        res = fut.result()
        if res[0] == "error":
            out[seeds[res[1]]] = {"score": None, "ms": None}
            continue
        _idx, seed, slot_order, rewards, _steps, avg_ms = res
        rewards = reorder_by_input(rewards, slot_order)
        avg_ms = reorder_by_input(avg_ms, slot_order)
        r0, r1 = rewards[0], rewards[1]
        if r0 is None or r1 is None:
            score = None
        elif r0 > r1:
            score = 1.0
        elif r1 > r0:
            score = 0.0
        else:
            score = 0.5
        out[seed] = {"score": score, "ms": avg_ms[0]}
    return out


# Cap games per worker pool so apollo's per-process native memory is reclaimed
# when the pool is torn down. A long-lived pool accumulates over hundreds of
# matches and OOMs; `max_tasks_per_child` recycling deadlocks under that memory
# pressure on Windows. Short-lived pools (~CHUNK_GAMES each, then fully torn
# down) keep per-worker matches tiny and reclaim reliably. Fresh workers re-read
# the constant-within-config config.json, so this is safe w.r.t. the LazyLock.
CHUNK_GAMES = 60


def play_config(apollo_path, cfg, opp_blocks, threads):
    """Write cfg, then play every opponent's seed block in CHUNK_GAMES-sized
    chunks, each on its own short-lived pool (bounds native memory)."""
    write_config(cfg)
    by_opp = {}
    for opp_name, opp_path, seeds in opp_blocks:
        results = {}
        for i in range(0, len(seeds), CHUNK_GAMES):
            chunk = seeds[i:i + CHUNK_GAMES]
            with ProcessPoolExecutor(max_workers=threads) as ex:
                results.update(play_block(ex, apollo_path, opp_path, chunk))
        by_opp[opp_name] = results
    return by_opp


# ---- allocation & selection -------------------------------------------------
def allocate_games(opponents, total, min_per):
    """Split `total` games across opponents by priority*(1-win_rate), with a
    floor of `min_per` each. Returns {name: n_games} summing exactly to total."""
    names = [o[0] for o in opponents]
    weights = {o[0]: o[1] * (1.0 - o[2]) for o in opponents}
    floor = min(min_per, total // max(1, len(names)))
    alloc = {n: floor for n in names}
    remaining = total - sum(alloc.values())
    sw = sum(weights.values()) or 1.0
    raw = {n: remaining * weights[n] / sw for n in names}
    for n in names:
        alloc[n] += int(raw[n])
    # hand out the rounding remainder to the largest fractional parts
    leftover = total - sum(alloc.values())
    order = sorted(names, key=lambda n: raw[n] - int(raw[n]), reverse=True)
    for i in range(leftover):
        alloc[order[i % len(order)]] += 1
    return alloc


def select_top_configs(study_name, k):
    """Top-k completed configs from the trials log, ranked by the LOWER bound of
    their win-rate CI (so we don't promote lucky high-variance trials)."""
    path = OUT_DIR / f"{study_name}_trials.jsonl"
    if not path.exists():
        raise SystemExit(f"no trials log at {path}")
    best = {}  # config-key -> (lcb, trial_number, config, win_rate)
    for line in open(path):
        r = json.loads(line)
        if r.get("status") != "complete":
            continue
        wins, n = r["cum"]["wins"], r["cum_games"]
        lcb = wilson(wins, n)[0]
        key = json.dumps(r["config"], sort_keys=True)
        if key not in best or lcb > best[key][0]:
            best[key] = (lcb, r["trial"], r["config"], r["win_rate"])
    ranked = sorted(best.values(), key=lambda t: t[0], reverse=True)
    return [(f"trial{t[1]}", t[2], t[3]) for t in ranked[:k]]


# ---- summarise --------------------------------------------------------------
def summarise(name, by_opp, control_by_opp):
    """Pooled record + per-opponent breakdown + paired delta vs control."""
    pooled = {"wins": 0, "losses": 0, "draws": 0, "errors": 0}
    ms_vals = []
    per_opp = {}
    for opp, games in by_opp.items():
        rec = {"wins": 0, "losses": 0, "draws": 0, "errors": 0}
        for g in games.values():
            s = g["score"]
            if s is None:
                rec["errors"] += 1
            elif s == 1.0:
                rec["wins"] += 1
            elif s == 0.0:
                rec["losses"] += 1
            else:
                rec["draws"] += 1
            if g["ms"] is not None:
                ms_vals.append(g["ms"])
        decided = rec["wins"] + rec["losses"]
        rec["games"] = sum(rec[k] for k in ("wins", "losses", "draws", "errors"))
        rec["win_rate"] = (rec["wins"] / decided) if decided else None
        per_opp[opp] = rec
        for kk in ("wins", "losses", "draws", "errors"):
            pooled[kk] += rec[kk]
    decided = pooled["wins"] + pooled["losses"]
    lo, mid, hi = wilson(pooled["wins"], decided)

    diffs = []
    if control_by_opp is not None:
        for opp, games in by_opp.items():
            ctrl = control_by_opp.get(opp, {})
            for seed, g in games.items():
                cg = ctrl.get(seed)
                if g["score"] is None or cg is None or cg["score"] is None:
                    continue
                diffs.append(g["score"] - cg["score"])
    d_lo, d_mid, d_hi, d_n = paired_delta(diffs)

    return {
        "name": name,
        "pooled": pooled,
        "decided": decided,
        "win_rate": mid, "wr_lo": lo, "wr_hi": hi,
        "avg_ms": (sum(ms_vals) / len(ms_vals)) if ms_vals else 0.0,
        "per_opp": per_opp,
        "paired_delta": d_mid, "pd_lo": d_lo, "pd_hi": d_hi, "pd_n": d_n,
    }


def main():
    # Output is ASCII, but force UTF-8 so a redirected console can't raise an
    # encoding error mid-run (Windows default is cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Final paired gauntlet for tuned apollo configs.")
    parser.add_argument("--study-name", default="apollo_v2")
    parser.add_argument("--top", type=int, default=5, help="Top configs from the study.")
    parser.add_argument("--total-games", type=int, default=500, help="Games TOTAL per config.")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=9_000_000,
                        help="Start of the held-out seed range (disjoint from training).")
    parser.add_argument("--min-per-opp", type=int, default=15,
                        help="Floor games per opponent (regression detection).")
    parser.add_argument("--no-control", action="store_true",
                        help="Skip the default-config control / paired comparison.")
    parser.add_argument("--configs", default=None,
                        help="JSON file of explicit configs to test instead of the study top-k. "
                             "Either a list of config objects, or {name: config}.")
    args = parser.parse_args()

    apollo_path = bot_entry("apollo")
    if not apollo_path.is_file():
        parser.error(f"apollo not found at {apollo_path}")

    # Build the candidate list (control first, then configs under test).
    candidates = []
    if not args.no_control:
        candidates.append(("default", DEFAULT_CONFIG))
    if args.configs:
        raw = json.loads(Path(args.configs).read_text())
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        for nm, cfg in items:
            candidates.append((str(nm), cfg))
    else:
        for nm, cfg, _wr in select_top_configs(args.study_name, args.top):
            candidates.append((nm, cfg))
    if len(candidates) <= (0 if args.no_control else 1):
        raise SystemExit("no configs selected; check the study name / trials log.")

    # Allocate games per opponent and assign each a fixed, shared seed block.
    alloc = allocate_games(OPPONENTS, args.total_games, args.min_per_opp)
    opp_blocks = []
    cursor = args.seed_base
    for name, _prio, _wr in OPPONENTS:
        n = alloc[name]
        opp_path = bot_entry(name)
        if not opp_path.is_file():
            raise SystemExit(f"opponent '{name}' not found at {opp_path}")
        seeds = list(range(cursor, cursor + n))
        opp_blocks.append((name, opp_path, seeds))
        cursor += n

    print(f"Gauntlet: {len(candidates)} configs x {args.total_games} games "
          f"(CRN, threads={args.threads})")
    print("Game allocation (priority*(1-win_rate), floor "
          f"{args.min_per_opp}):")
    for name, _p, wr in OPPONENTS:
        print(f"  {name:16} {alloc[name]:>4} games   (assumed wr {wr:.0%})")
    print()

    results = {}
    control_by_opp = None
    for name, cfg in candidates:
        print(f"  running {name} ...", flush=True)
        by_opp = play_config(apollo_path, cfg, opp_blocks, args.threads)
        results[name] = by_opp
        if name == "default":
            control_by_opp = by_opp

    summaries = [summarise(name, results[name],
                           None if name == "default" else control_by_opp)
                 for name, _cfg in candidates]
    # Rank configs (control kept at top for reference) by pooled win rate.
    under_test = [s for s in summaries if s["name"] != "default"]
    under_test.sort(key=lambda s: s["win_rate"], reverse=True)
    ordered = ([s for s in summaries if s["name"] == "default"] + under_test)

    opp_names = [o[0] for o in OPPONENTS]
    header = f"{'config':>10} | {'pooled wr (95% CI)':>22} | {'delta vs default (95%CI)':>24} | ms |"
    header += " " + " ".join(f"{o[:6]:>6}" for o in opp_names)
    print("\n" + header)
    print("-" * len(header))
    for s in ordered:
        wr = f"{s['win_rate']*100:5.1f}% [{s['wr_lo']*100:4.1f},{s['wr_hi']*100:4.1f}]"
        if s["name"] == "default":
            delta = f"{'(control)':>24}"
        else:
            sig = "*" if (s["pd_lo"] > 0 or s["pd_hi"] < 0) else " "
            delta = f"{s['paired_delta']*100:+5.1f} [{s['pd_lo']*100:+4.1f},{s['pd_hi']*100:+4.1f}]{sig}"
        cells = " ".join(
            (f"{(s['per_opp'][o]['win_rate'] or 0)*100:5.0f}%" if o in s["per_opp"] else "   -  ")
            for o in opp_names)
        print(f"{s['name']:>10} | {wr:>22} | {delta:>24} | {s['avg_ms']:3.0f} | {cells}")
    print("\n(* = paired improvement over default is significant at 95%. delta is mean "
          "per-game score difference on identical seeds; +1 = win flipped from loss.)")

    # Persist full per-seed results + summaries; save the best config.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = OUT_DIR / f"{args.study_name}_gauntlet_{stamp}.json"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "study": args.study_name,
        "total_games": args.total_games,
        "seed_base": args.seed_base,
        "allocation": alloc,
        "configs": {name: cfg for name, cfg in candidates},
        "summaries": [{k: v for k, v in s.items() if k != "per_opp"} | {"per_opp": s["per_opp"]}
                      for s in ordered],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    if under_test:
        winner = under_test[0]
        best_cfg = dict(candidates)[winner["name"]]
        best_path = OUT_DIR / f"{args.study_name}_gauntlet_best.json"
        best_path.write_text(json.dumps(
            {"name": winner["name"], "win_rate": winner["win_rate"],
             "paired_delta_vs_default": winner["paired_delta"], "config": best_cfg}, indent=2))
        print(f"\nWinner: {winner['name']}  ({winner['win_rate']*100:.1f}% pooled, "
              f"delta {winner['paired_delta']*100:+.1f} vs default)")
        print(f"Saved: {best_path.name}, {out_json.name}")

    # Leave config.json at the committed defaults, not the last config played.
    write_config(DEFAULT_CONFIG)
    print("Restored config.json to defaults.")


if __name__ == "__main__":
    main()
