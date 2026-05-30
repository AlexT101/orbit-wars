"""Recompute the loss of a deployed AOWV value net over the v2 datasets.

Pure NumPy (no torch) so it's light and reproducible. Parses the AOWV binary
(normalization already folded into fc1), runs forward, and reports the same
SmoothL1 (Huber, beta=1) loss used in training, plus MSE and sign accuracy.
Reproduces the seed=42 12% game-level val split PER FILE (training's cross-file
split uses Python hash() which isn't reproducible without PYTHONHASHSEED).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_aowv(path: Path):
    b = path.read_bytes()
    off = 0

    def u32():
        nonlocal off
        v = struct.unpack_from("<I", b, off)[0]
        off += 4
        return v

    magic, version, in_dim, hidden = u32(), u32(), u32(), u32()
    assert magic == 0x564F4157, f"bad magic {magic:#x}"
    w1 = np.frombuffer(b, "<f4", in_dim * hidden, off).reshape(hidden, in_dim).copy()
    off += 4 * in_dim * hidden
    b1 = np.frombuffer(b, "<f4", hidden, off).copy()
    off += 4 * hidden
    w2 = np.frombuffer(b, "<f4", hidden, off).copy()
    off += 4 * hidden
    b2 = struct.unpack_from("<f", b, off)[0]
    return in_dim, hidden, version, w1, b1, w2, b2


def forward(X, w1, b1, w2, b2):
    h = np.maximum(X @ w1.T + b1, 0.0)
    return np.tanh(h @ w2 + b2)


def smooth_l1(pred, tgt, beta=1.0):
    d = np.abs(pred - tgt)
    return np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta).mean()


def main():
    net_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "weights/v2_replays.bin"
    in_dim, hidden, version, w1, b1, w2, b2 = load_aowv(net_path)
    print(f"net={net_path.name} in_dim={in_dim} hidden={hidden} version={version}")

    data_files = sys.argv[2:] or [
        str(p) for p in sorted((HERE / "data").glob("*.npz"))
    ]

    allX, allY = [], []
    print(f"\n{'file':<26} {'N':>8} {'huber':>9} {'mse':>9} {'sign%':>7}")
    for f in data_files:
        d = np.load(f)
        if "summary_v2" not in d.files:
            continue
        v2 = d["summary_v2"]
        if v2.ndim != 2 or v2.shape[1] != in_dim or np.abs(v2).sum() == 0:
            continue
        X = v2.astype(np.float32)
        y = d["labels"].astype(np.float32)
        p = forward(X, w1, b1, w2, b2)
        h = smooth_l1(p, y)
        mse = float(((p - y) ** 2).mean())
        sign = float(((p > 0) == (y > 0)).mean())
        print(f"{Path(f).name:<26} {len(y):>8} {h:>9.4f} {mse:>9.4f} {100*sign:>6.1f}")
        allX.append(X)
        allY.append(y)

    if not allX:
        print("no usable v2 data found")
        return
    X = np.concatenate(allX)
    y = np.concatenate(allY)
    p = forward(X, w1, b1, w2, b2)
    print(f"\n{'ALL (train+val)':<26} {len(y):>8} {smooth_l1(p,y):>9.4f} "
          f"{float(((p-y)**2).mean()):>9.4f} {100*float(((p>0)==(y>0)).mean()):>6.1f}")


if __name__ == "__main__":
    main()
