"""Bisect which single scoring constant makes t8's config break on this build:
identity, then identity + each one t8 override, vs apollo_baseline on fixed seeds."""
from __future__ import annotations
import json, os, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[3]))
from run_batched import bot_entry, reorder_by_input, run_match_job, slot_order_for_seed

BASE = json.loads((HERE.parent / "config.json").read_text())
IDENT = {"score_w_ship_cost": 1.0, "score_w_final_ships": 1.0, "score_per_ship_smoothing": 1.0,
         "capture_min_score": 0.0, "score_enemy_capture_bonus": 1.0, "default_strategy": 0,
         "neutral_payback_turns": 20.0, "neutral_payback_penalty": 0.0, "lead_gate": 50.0,
         "neutral_capture_penalty": 0.0}
T8 = {"score_w_ship_cost": 0.5118315444571954, "score_w_final_ships": 1.699175555297551,
      "score_per_ship_smoothing": 16.93403660002761, "capture_min_score": -2.5084693088448082,
      "score_enemy_capture_bonus": 1.6518213290467685}


def safe(job):
    try: return run_match_job(job)
    except BaseException as exc: return ("error", job[2], repr(exc))


def play(cfg, opp, seeds, threads=12):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh: json.dump(cfg, fh)
    os.environ["APOLLO_CONFIG"] = path
    apollo, opp_path = bot_entry("apollo"), bot_entry(opp)
    jobs = [([apollo, opp_path], s, i, slot_order_for_seed(s)) for i, s in enumerate(seeds)]
    w = l = d = 0
    with ProcessPoolExecutor(max_workers=threads) as ex:
        for fut in as_completed([ex.submit(safe, j) for j in jobs]):
            r = fut.result()
            if r[0] == "error": continue
            _i, _s, slot, rew, _st, _ms = r
            rew = reorder_by_input(rew, slot)
            a, b = rew[0], rew[1]
            if a > b: w += 1
            elif b > a: l += 1
            else: d += 1
    os.remove(path)
    return w, l, d


def main():
    seeds = list(range(30_000_000, 30_000_000 + 24))
    opp = "apollo_baseline"
    variants = {"identity": {**BASE, **IDENT}}
    for k, v in T8.items():
        variants[f"id+{k}"] = {**BASE, **IDENT, k: v}
    variants["t8_full"] = {**BASE, **IDENT, **T8}
    for name, cfg in variants.items():
        w, l, d = play(cfg, opp, seeds)
        print(f"  {name:30} {w}-{l}-{d}  win%={w/(w+l)*100 if w+l else 0:5.1f}")


if __name__ == "__main__":
    main()
