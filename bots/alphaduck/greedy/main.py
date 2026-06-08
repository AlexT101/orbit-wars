"""Stochastic policy sampler: same model as alphaduck, no MCTS.

For each owned planet:
  1. Get the per-source distribution over (N+1) actions from the policy head:
     policy[i] = [P(noop), P(launch→0), ..., P(launch→N-1)]
  2. Zero out unreachable targets (apollo) and self (i==j), renormalize.
  3. Sample one action proportional to the renormalized probabilities.

No threshold, no argmax. Just the trained policy as a stochastic policy.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import inspect, importlib.util
import numpy as np


def _here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        pass
    f = inspect.currentframe()
    if f is not None and f.f_code.co_filename and f.f_code.co_filename != "<string>":
        return Path(f.f_code.co_filename).resolve().parent
    return Path.cwd() / "bots" / "alphaduck" / "greedy"


HERE = _here()
ROOT = HERE.parent.parent.parent
ALPHADUCK_MAIN = ROOT / "bots" / "alphaduck" / "main.py"
spec = importlib.util.spec_from_file_location("alphaduck_main", str(ALPHADUCK_MAIN))
duck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(duck)


# Seed via env for reproducibility; default = OS entropy.
_SEED_ENV = os.environ.get("GREEDY_SEED")
_RNG = np.random.default_rng(int(_SEED_ENV) if _SEED_ENV else None)


def agent(obs, config=None):
    try:
        state = duck._BD.parse_state(obs)
        my_player = int(obs.get("player", 0))
        duck._load_model()
        policy, _value, pids = duck._root_forward(state, my_player)
        pid_to_planet = {p["id"]: p for p in state["planets"]}
        aim = duck._aim_for_state(state)
        N = len(pids)
        actions = []
        for i, pid in enumerate(pids):
            src = pid_to_planet[pid]
            if src["owner"] != my_player or src["ships"] <= 0:
                continue
            ships = int(src["ships"])
            probs = np.array(policy[i], dtype=np.float64).copy()  # (N+1,)
            # Mask self + unreachable launches; noop (col 0) always allowed.
            for j in range(N):
                if j == i:
                    probs[1 + j] = 0.0
                    continue
                entry = aim.get((int(pid), int(pids[j]), ships))
                if entry is None:
                    probs[1 + j] = 0.0
            s = probs.sum()
            if s <= 0:
                continue  # nothing legal — treat as noop
            probs /= s
            k = int(_RNG.choice(len(probs), p=probs))
            if k == 0:
                continue  # noop
            tgt_pid = int(pids[k - 1])
            entry = aim[(int(pid), tgt_pid, ships)]
            _eta, angle = entry  # apollo's angle — use it, not _lead_angle.
            actions.append([int(pid), float(angle), ships])
        return actions
    except Exception as exc:
        sys.stderr.write(f"alphaduck_greedy error: {exc!r}\n")
        import traceback; traceback.print_exc(file=sys.stderr)
        return []


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        import io, json, zipfile, time
        i = sys.argv.index("--self-test")
        spec_arg = sys.argv[i + 1]
        zp_s, name = spec_arg.split(":", 1)
        with zipfile.ZipFile(zp_s) as zf:
            with zf.open(name) as f:
                g = json.load(io.BytesIO(f.read()))
        for t in [10, 30, 60, 90, 120]:
            if t >= len(g.get("steps", [])): break
            obs = g["steps"][t][0]["observation"]
            t0 = time.perf_counter()
            acts = agent(obs)
            dt = (time.perf_counter() - t0) * 1000
            print(f"step {t}: {len(acts)} launches in {dt:.0f}ms")
            for a in acts:
                print(f"  src={a[0]:3d}  angle={a[1]:+.3f}  ships={a[2]}")
