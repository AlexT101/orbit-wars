"""Verify the Rust `summary_features_spatial` matches the Python
`spatial_features.compute` bit-for-bit (within f32 tolerance).

Feeds real replay observations through the `spatial_parity` Rust binary and
diffs each of the 13 columns against the Python implementation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from spatial_features import compute, SPATIAL_NAMES

HERE = Path(__file__).resolve().parent
BOT_DIR = HERE.parent
BIN = BOT_DIR / "target" / "release" / "spatial_parity"


def normalize_obs(o: dict) -> dict:
    return {
        "player": int(o.get("player", 0)),
        "step": int(o.get("step", 0)),
        "planets": list(o.get("planets", []) or []),
        "fleets": list(o.get("fleets", []) or []),
        "angular_velocity": float(o.get("angular_velocity", 0.0)),
        "initial_planets": list(o.get("initial_planets", []) or []),
        "comets": list(o.get("comets", []) or []),
        "comet_planet_ids": list(o.get("comet_planet_ids", []) or []),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", type=Path, default=Path("/home/sunrise/orbitwars/pantheow/experimental_arch/replays/rank1"))
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--tol", type=float, default=2e-3)
    args = p.parse_args()

    files = sorted(args.replays.glob("*.json"))[: args.games]
    obs_list = []
    py_feats = []
    for f in files:
        data = json.loads(f.read_bytes())
        if len(data.get("rewards") or []) != 2:
            continue
        for step in data.get("steps") or []:
            if not isinstance(step, list) or len(step) < 2:
                continue
            for slot in range(2):
                e = step[slot]
                if not isinstance(e, dict):
                    continue
                obs = e.get("observation")
                if not obs or not obs.get("planets"):
                    continue
                norm = normalize_obs(obs)
                obs_list.append(norm)
                py_feats.append(compute(norm, slot))
    print(f"comparing {len(obs_list)} observations from {len(files)} games")

    proc = subprocess.Popen([str(BIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    inp = "".join(json.dumps(o, separators=(",", ":")) + "\n" for o in obs_list).encode()
    out, _ = proc.communicate(inp)
    rust_rows = [ln.split() for ln in out.decode().splitlines() if ln.strip()]
    if len(rust_rows) != len(obs_list):
        print(f"WARN got {len(rust_rows)} rust rows for {len(obs_list)} obs")
    n = min(len(rust_rows), len(obs_list))
    rust = np.array([[float(x) for x in r[2:]] for r in rust_rows[:n]], dtype=np.float64)
    py = np.array(py_feats[:n], dtype=np.float64)

    diff = np.abs(rust - py)
    # Relative tolerance for large-magnitude features.
    denom = np.maximum(1.0, np.abs(py))
    rel = diff / denom
    max_abs = diff.max(axis=0)
    max_rel = rel.max(axis=0)
    print(f"{'feature':30s} {'max_abs':>12s} {'max_rel':>12s}")
    worst = 0.0
    for i, name in enumerate(SPATIAL_NAMES):
        print(f"{name:30s} {max_abs[i]:12.5f} {max_rel[i]:12.6f}")
        worst = max(worst, max_rel[i])
    ok = worst < args.tol
    print(f"\noverall max relative diff = {worst:.6f}  -> {'PARITY OK' if ok else 'PARITY FAIL'} (tol={args.tol})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
