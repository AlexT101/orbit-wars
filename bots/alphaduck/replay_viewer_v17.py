"""v17 replay viewer: same UI as replay_viewer.py, but uses alphaduck/main.py's
own _root_forward() so the policy/value/noop shown in the panel are exactly
what the live bot would compute from each turn's obs.

Usage:
  python3 bots/alphaduck/replay_viewer_v17.py            # auto-pick first 2p replay
  python3 bots/alphaduck/replay_viewer_v17.py --replay <zip>:<json>
  python3 bots/alphaduck/replay_viewer_v17.py --replay path/to/replay.json
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
import webbrowser
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

# Reuse helpers/HTML from the existing viewer
spec = importlib.util.spec_from_file_location("rv", HERE / "replay_viewer.py")
_rv = importlib.util.module_from_spec(spec)
# Don't actually load the old PlanetTransformerPair (it needs old set_net etc.);
# only need a handful of functions. Inject a fake `pair_net` to skip the import.
sys.modules.setdefault("pair_net", type(sys)("pair_net"))
sys.modules["pair_net"].PlanetTransformerPair = object  # only to satisfy `from pair_net import ...`
sys.modules.setdefault("set_net", type(sys)("set_net"))
sys.modules["set_net"].apply_norm = lambda *a, **k: a[:2]
spec.loader.exec_module(_rv)

# Import alphaduck/main.py as a library — gives us _root_forward, _BD, etc.
spec2 = importlib.util.spec_from_file_location("alphaduck_main", HERE / "main.py")
duck = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(duck)


def precompute_v17(game):
    """Walk every step, call duck._root_forward, pack a viewer-shaped dict."""
    steps = game.get("steps") or []
    parsed: list[dict | None] = [None] * len(steps)
    for t, step in enumerate(steps):
        if step and step[0].get("observation"):
            parsed[t] = duck._BD.parse_state(step[0]["observation"])

    last_owner: dict[int, int] = {}
    owner_change_turn: dict[int, int] = {}

    duck._load_model()
    out_steps = []
    t0 = time.time()
    for t, state in enumerate(parsed):
        if state is None:
            out_steps.append(None); continue
        duck._BD.update_owner_history(state, last_owner, owner_change_turn, state["step"])

        per_player = {}
        for player in (0, 1):
            try:
                policy, value, pids = duck._root_forward(state, player)
            except Exception as e:
                sys.stderr.write(f"t={t} p={player} forward failed: {e}\n")
                per_player[player] = None; continue
            n = len(pids)
            # policy: (n, n+1) where col 0 = noop, cols 1..n = JOINT P(launch i->j).
            # The joint is what alphaduck's MCTS sees: noop + each launch arm sum to 1 per row.
            # Targets in early game will look very small because noop is ~0.99 — that's the
            # actual prior MCTS uses, not a viewer artifact.
            noop_probs = policy[:, 0]
            joint_probs = policy[:, 1:1 + n]
            agg = _rv.per_side_aggregates(state, player)
            per_player[player] = {
                "pids": [int(p) for p in pids],
                # `probs` here = joint P(launch i -> j); row sums to (1 - noop[i]).
                # Use 6 decimal places — early-game launch probs are routinely
                # 1e-4 to 1e-3 because noop is ~0.99; 4 places rounded them to 0.
                "probs": [[round(float(joint_probs[i, j]), 6) for j in range(n)] for i in range(n)],
                "noop": [round(float(noop_probs[i]), 4) for i in range(n)],
                "value": round(float(value), 3),
                "agg": agg,
                "eval_for_me": round(_rv.heuristic_eval(state, player), 2),
            }

        # Actual launches at this step (newly-appeared fleets vs prev step)
        actual = {0: {}, 1: {}}
        prev = parsed[t - 1] if t > 0 else None
        if prev is not None:
            old_ids = {f["id"] for f in prev["fleets"]}
            for f in state["fleets"]:
                if f["id"] in old_ids:
                    continue
                pl_acts = (steps[t][f["owner"]].get("action") or [])
                src_pid = None
                for act in pl_acts:
                    if abs(float(act[1]) - f["angle"]) < 1e-6 and int(act[2]) == f["ships"]:
                        src_pid = int(act[0]); break
                pred = duck._BD.predict_fleet_collision(state, f)
                if pred is None or src_pid is None:
                    continue
                dst_pid, _eta = pred
                actual[f["owner"]].setdefault(src_pid, []).append(int(dst_pid))

        out_steps.append({
            "planets": [
                {"id": int(p["id"]), "owner": int(p["owner"]),
                 "x": round(p["x"], 3), "y": round(p["y"], 3),
                 "r": round(p["radius"], 2),
                 "ships": int(p["ships"]), "prod": int(p["prod"]),
                 "comet": bool(p["is_comet"])}
                for p in state["planets"]
            ],
            "fleets": [
                {"x": round(f["x"], 2), "y": round(f["y"], 2),
                 "angle": round(float(f["angle"]), 4),
                 "owner": int(f["owner"]), "ships": int(f["ships"])}
                for f in state["fleets"]
            ],
            "p0": per_player[0],
            "p1": per_player[1],
            "actual0": actual[0],
            "actual1": actual[1],
        })
        if (t + 1) % 25 == 0:
            print(f"  step {t+1}/{len(steps)}  ({time.time() - t0:.1f}s)", flush=True)
    return out_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=None,
                    help="<zip>:<json> or path/to/replay.json (defaults to repo replay.html source)")
    ap.add_argument("--out", type=Path, default=HERE / "replay_viewer_v17.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    spec_str = args.replay
    if spec_str is None:
        # Look for replay.html's adjacent replay.json or kaggle default
        candidate = ROOT / "replay.json"
        if candidate.exists():
            spec_str = str(candidate)
    if spec_str is None:
        game, label = _rv.load_replay(None)
    else:
        game, label = _rv.load_replay(spec_str)
    print(f"  replay: {label}  ({len(game.get('steps', []))} steps)")
    print(f"  ckpt:   {duck.DEFAULT_CKPT}")
    steps = precompute_v17(game)
    _rv.write_html(args.out, steps, label)
    sz = args.out.stat().st_size // 1024
    print(f"wrote {args.out}  ({sz} KB)")
    if not args.no_open:
        print("opening in browser ...")
        webbrowser.open(f"file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
