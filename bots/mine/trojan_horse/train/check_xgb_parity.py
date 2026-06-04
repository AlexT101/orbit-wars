"""End-to-end model parity: Python xgboost vs Rust xgb.rs on the same 170-d
feature rows. Confirms the deployed bot's value-net output matches training.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "target" / "release" / "xgb_parity"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    X = d["features"].astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=min(args.n, X.shape[0]), replace=False)
    Xs = X[idx]

    bst = xgb.Booster()
    bst.load_model(str(args.model))
    py = bst.predict(xgb.DMatrix(Xs))  # probability for binary:logistic

    csv = "\n".join(",".join(f"{v:.9e}" for v in row) for row in Xs) + "\n"
    out = subprocess.run([str(BIN), str(args.model), "/dev/stdin"], input=csv.encode(),
                         capture_output=True)
    rust_vals = []
    for ln in out.stdout.decode().splitlines():
        # "margin=...  value=..."  value = 2*prob-1 mapped; we want prob.
        for tok in ln.split():
            if tok.startswith("value="):
                rust_vals.append(float(tok.split("=")[1]))
    rust_vals = np.array(rust_vals)
    if len(rust_vals) != len(py):
        print(f"WARN {len(rust_vals)} rust vals vs {len(py)} python")
    n = min(len(rust_vals), len(py))
    # Rust predict_value returns (2*prob-1) for logistic? Compare both mappings.
    py_value = 2.0 * py[:n] - 1.0
    diff_value = np.abs(rust_vals[:n] - py_value)
    diff_prob = np.abs(rust_vals[:n] - py[:n])
    print(f"compared {n} rows")
    print(f"  vs (2*prob-1): max={diff_value.max():.6e} mean={diff_value.mean():.6e}")
    print(f"  vs prob      : max={diff_prob.max():.6e} mean={diff_prob.mean():.6e}")
    best = min(diff_value.max(), diff_prob.max())
    print(f"overall best mapping max diff = {best:.6e} -> {'PARITY OK' if best < args.tol else 'CHECK'} (tol={args.tol})")


if __name__ == "__main__":
    main()
