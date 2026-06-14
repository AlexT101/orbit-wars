"""One-off diagnostic: do the phase-2 seed configs actually regress on the
current build? Plays identity vs t8's exact config against frozen opponents on
identical seeds, each config via its OWN APOLLO_CONFIG temp file (no config.json
race, fresh pool per config so the LazyLock reads the right file)."""
from __future__ import annotations
import json, os, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from run_batched import bot_entry, reorder_by_input, run_match_job, slot_order_for_seed

BASE = json.loads((HERE.parent / "config.json").read_text())  # phase-1 + current

IDENTITY = {"score_w_ship_cost": 1.0, "score_w_final_ships": 1.0,
            "score_per_ship_smoothing": 1.0, "capture_min_score": 0.0,
            "score_enemy_capture_bonus": 1.0, "default_strategy": 0}
T8 = {"score_w_ship_cost": 0.5118315444571954, "score_w_final_ships": 1.699175555297551,
      "score_per_ship_smoothing": 16.93403660002761, "capture_min_score": -2.5084693088448082,
      "score_enemy_capture_bonus": 1.6518213290467685, "default_strategy": 0}
NEUTRAL_ID = {"neutral_payback_turns": 20.0, "neutral_payback_penalty": 0.0,
              "lead_gate": 50.0, "neutral_capture_penalty": 0.0}


def safe(job):
    try:
        return run_match_job(job)
    except BaseException as exc:  # noqa: BLE001
        return ("error", job[2], repr(exc))


def play(cfg_path, opp, seeds, threads=12):
    apollo = bot_entry("apollo")
    opp_path = bot_entry(opp)
    os.environ["APOLLO_CONFIG"] = str(cfg_path)
    jobs = [([apollo, opp_path], s, i, slot_order_for_seed(s)) for i, s in enumerate(seeds)]
    w = l = d = e = 0
    with ProcessPoolExecutor(max_workers=threads) as ex:
        for fut in as_completed([ex.submit(safe, j) for j in jobs]):
            r = fut.result()
            if r[0] == "error":
                e += 1; continue
            _i, _s, slot, rew, _st, _ms = r
            rew = reorder_by_input(rew, slot)
            a, b = rew[0], rew[1]
            if a is None or b is None: e += 1
            elif a > b: w += 1
            elif b > a: l += 1
            else: d += 1
    return w, l, d, e


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    base = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000_000
    seeds = list(range(base, base + n))
    print(f"seed range: {base}..{base+n}")
    opps = ["apollo_baseline", "producer_v2", "simpleagent"]
    configs = {"identity": {**BASE, **IDENTITY, **NEUTRAL_ID},
               "t8_phase2": {**BASE, **T8, **NEUTRAL_ID}}
    for name, cfg in configs.items():
        fd, path = tempfile.mkstemp(suffix=f"_{name}.json")
        with os.fdopen(fd, "w") as fh:
            json.dump(cfg, fh)
        print(f"\n=== {name} (APOLLO_CONFIG={Path(path).name}) ===")
        for opp in opps:
            w, l, d, e = play(path, opp, seeds)
            dec = w + l
            pts = (w + 0.5 * d) / (w + l + d) * 100 if (w + l + d) else float("nan")
            print(f"  vs {opp:16} {w}-{l}-{d} (err {e})  win%={w/dec*100 if dec else float('nan'):5.1f}  pts%={pts:5.1f}")
        os.remove(path)


if __name__ == "__main__":
    main()
