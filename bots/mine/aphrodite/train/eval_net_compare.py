"""Comparative accuracy for an AOWV value net: instead of "does the net's sign
match the outcome?", ask "if we evaluate BOTH players' perspectives of the same
state and predict the higher-scoring side as the winner, how often is that right?"

This matches how the net is actually used in MCTS (relative ranking of states),
not its absolute sign. replays_strong.npz interleaves perspectives: even rows are
player 0's view of a state, the following odd row is player 1's view of the SAME
state (opposite label). So adjacent pairs give us both sides for free.
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
    return in_dim, hidden, w1, b1, w2, b2


def forward(X, w1, b1, w2, b2):
    h = np.maximum(X @ w1.T + b1, 0.0)
    return np.tanh(h @ w2 + b2)


def main():
    net_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "weights/v2_replays.bin"
    data_path = HERE / "data/replays_strong.npz"
    in_dim, hidden, w1, b1, w2, b2 = load_aowv(net_path)
    print(f"net={net_path.name} in_dim={in_dim} hidden={hidden}")

    d = np.load(data_path)
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]

    # adjacent pairs: even row = player0 view, odd row = player1 view of same state
    assert (meta[0::2, 2] == 0).all() and (meta[1::2, 2] == 1).all()
    X0, X1 = X[0::2], X[1::2]
    y0 = y[0::2]                      # +1 if player0 won this game, else -1
    games = meta[0::2, 0].astype(np.int64)

    v0 = forward(X0, w1, b1, w2, b2)  # net's value for player 0
    v1 = forward(X1, w1, b1, w2, b2)  # net's value for player 1
    pred_p0_wins = v0 > v1
    actual_p0_wins = y0 > 0
    correct = pred_p0_wins == actual_p0_wins
    ties = np.isclose(v0, v1)

    # same game-level val split as training (seed=42, 12% of games)
    unique = np.unique(games)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_games = set(unique[:n_val].tolist())
    val = np.array([g in val_games for g in games])

    print(f"\npairs total={len(y0)}  val={val.sum()} (games={len(unique)} val={n_val})")
    print(f"\n{'split':<14} {'compare-acc':>12} {'tie%':>7}")
    for name, mask in (("VAL", val), ("TRAIN", ~val), ("ALL", np.ones(len(val), bool))):
        acc = correct[mask].mean()
        tie = 100 * ties[mask].mean()
        print(f"{name:<14} {100*acc:>11.2f}% {tie:>6.2f}%")

    # also the old absolute sign accuracy on the same val rows, for reference
    pall = forward(X, w1, b1, w2, b2)
    val_rows = np.repeat(val, 2)  # expand pair-mask back to per-row
    sign_val = ((pall[val_rows] > 0) == (y[val_rows] > 0)).mean()
    print(f"\nreference (absolute sign-acc, VAL rows): {100*sign_val:.2f}%")


if __name__ == "__main__":
    main()
