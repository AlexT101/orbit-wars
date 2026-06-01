"""Linear (logistic-style) baseline on the SAME 46-d summary_v2 features the
deployed MLP uses, with the SAME data / split / loss — so val accuracy is
directly comparable to the MLP (val SmoothL1=0.177, sign-acc 84.8%).

Model: y = tanh(w . x + b)  — i.e. multiply every input by a weight, sum, then
squash the score to a win value in (-1, 1). No hidden layer.

Trained full-batch (the whole set fits in memory) so it's near-instant. Exports
an AOWV file the bot can load directly, using the hidden=2 mirror trick
(relu(z) - relu(-z) = z) so a pure-linear model fits the MLP weight format.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 46
HERE = Path(__file__).resolve().parent


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_aowv_linear(out_path: Path, w_folded: np.ndarray, b_folded: float):
    """Encode tanh(w.x + b) in the 1-hidden-layer AOWV format via 2 mirrored
    relu units: h=[relu(z), relu(-z)], out=tanh(1*relu(z) + (-1)*relu(-z)) =
    tanh(z), z = w.x + b."""
    in_dim = w_folded.shape[0]
    hidden = 2
    w1 = np.stack([w_folded, -w_folded], axis=0).astype(np.float32)  # (2, in)
    b1 = np.array([b_folded, -b_folded], dtype=np.float32)
    w2 = np.array([1.0, -1.0], dtype=np.float32)
    b2 = 0.0
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x564F4157))
    buf.extend(struct.pack("<I", 1))
    buf.extend(struct.pack("<I", in_dim))
    buf.extend(struct.pack("<I", hidden))
    buf.extend(w1.tobytes(order="C"))
    buf.extend(b1.tobytes(order="C"))
    buf.extend(w2.tobytes(order="C"))
    buf.extend(struct.pack("<f", b2))
    out_path.write_bytes(bytes(buf))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=[str(HERE / "data/replays_strong.npz")])
    p.add_argument("--out", default=str(HERE / "weights/linear_v2.bin"))
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dev = device()
    print(f"device={dev}")

    Xs, ys, metas = [], [], []
    for path in args.data:
        d = np.load(path)
        v2 = d["summary_v2"]
        if v2.shape[1] != INPUT_DIM or np.abs(v2).sum() == 0:
            print(f"  skip {path}")
            continue
        Xs.append(v2.astype(np.float32))
        ys.append(d["labels"].astype(np.float32))
        metas.append(d["meta"])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    meta = np.concatenate(metas)

    # Same game-level split as train_summary_v2.py: seed=42, 12% of games.
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_set = set(unique[:n_val].tolist())
    val_mask = np.array([g in val_set for g in games])
    print(f"samples: total={len(y)} train={(~val_mask).sum()} val={val_mask.sum()} "
          f"(games total={len(unique)} val={n_val})")

    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    Xtr, ytr = Xt[~val_mask], yt[~val_mask]
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp(min=1e-3)
    Xtr_n = (Xtr - mean) / std
    Xv_n = (Xv - mean) / std

    torch.manual_seed(args.seed)
    lin = nn.Linear(INPUT_DIM, 1).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_sign = 0.0
    best_w = None
    best_b = None
    t0 = time.time()
    for ep in range(args.epochs):
        lin.train()
        pred = torch.tanh(lin(Xtr_n).squeeze(-1))
        loss = loss_fn(pred, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            lin.eval()
            with torch.no_grad():
                tr = torch.tanh(lin(Xtr_n).squeeze(-1))
                v = torch.tanh(lin(Xv_n).squeeze(-1))
                tr_loss = loss_fn(tr, ytr).item()
                val_loss = loss_fn(v, yv).item()
                tr_sign = ((tr > 0) == (ytr > 0)).float().mean().item()
                val_sign = ((v > 0) == (yv > 0)).float().mean().item()
                val_mse = ((v - yv) ** 2).mean().item()
            if val_loss < best_val:
                best_val = val_loss
                best_sign = val_sign
                best_w = lin.weight.detach().cpu().numpy().reshape(-1).copy()
                best_b = float(lin.bias.detach().cpu().numpy().reshape(-1)[0])
            if ep % 500 == 0 or ep == args.epochs - 1:
                print(f"ep {ep:4d}  train: huber={tr_loss:.4f} sign={tr_sign:.3f}  "
                      f"|  val: huber={val_loss:.4f} mse={val_mse:.4f} sign={val_sign:.3f}")

    print(f"\nBEST val: huber={best_val:.4f} sign={best_sign:.3f}  elapsed={time.time()-t0:.1f}s")
    print(f"(MLP h64 reference: val huber=0.177 sign=0.848)")

    # Fold normalization: z = w.((x-mean)/std) + b = (w/std).x + (b - (w/std).mean)
    std_np = std.cpu().numpy()
    mean_np = mean.cpu().numpy()
    w_folded = (best_w / std_np).astype(np.float32)
    b_folded = float(best_b - (best_w / std_np) @ mean_np)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_aowv_linear(out, w_folded, b_folded)
    print(f"wrote {out} ({out.stat().st_size} bytes, deployable AOWV linear)")


if __name__ == "__main__":
    main()
