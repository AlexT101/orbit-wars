"""Train a 2-hidden-layer MLP on summary features (Python-only — checks
whether the extra capacity moves val accuracy; if it does, we'll port
the format to Rust)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summary_features import summary_features


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--hidden1", type=int, default=64)
    p.add_argument("--hidden2", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    Xs, ys, metas = [], [], []
    for path in args.data:
        d = np.load(path)
        Xs.append(d["features"])
        ys.append(d["labels"])
        metas.append(d["meta"])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    meta = np.concatenate(metas, axis=0)
    Xs = summary_features(X).astype(np.float32)
    print(f"samples={X.shape[0]} feats={Xs.shape[1]}")

    games = meta[:, 0].astype(np.int64)
    if len(args.data) > 1:
        # disambiguate across files
        offsets = np.zeros_like(games)
        cursor = 0
        for path in args.data:
            d = np.load(path)
            n = d["features"].shape[0]
            offsets[cursor : cursor + n] = (hash(path) & 0xFFF) * 10_000
            cursor += n
        games = games + offsets
    unique = np.unique(games)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_set = set(unique[:n_val].tolist())
    val_mask = np.array([g in val_set for g in games])

    dev = device()
    Xt = torch.from_numpy(Xs).to(dev)
    yt = torch.from_numpy(y).to(dev)
    Xtr, ytr = Xt[~val_mask], yt[~val_mask]
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp(min=1e-3)

    model = nn.Sequential(
        nn.Linear(Xs.shape[1], args.hidden1),
        nn.ReLU(),
        nn.Linear(args.hidden1, args.hidden2),
        nn.ReLU(),
        nn.Linear(args.hidden2, 1),
        nn.Tanh(),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.SmoothL1Loss()

    bs = 256
    n_train = Xtr.shape[0]
    best_val = float("inf")
    best_sign = 0.0
    t0 = time.time()
    for ep in range(args.epochs):
        idx = torch.randperm(n_train, device=dev)
        model.train()
        train_loss = 0.0
        for j in range(0, n_train, bs):
            sel = idx[j : j + bs]
            xb = (Xtr[sel] - mean) / std
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, ytr[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * sel.shape[0]
        train_loss /= n_train
        model.eval()
        with torch.no_grad():
            v = model((Xv - mean) / std).squeeze(-1)
            val_loss = loss_fn(v, yv).item()
            sign = ((v > 0) == (yv > 0)).float().mean().item()
        if val_loss < best_val:
            best_val = val_loss
            best_sign = sign
        print(f"ep {ep:3d} train={train_loss:.4f} val={val_loss:.4f} sign={sign:.3f}")

    print(f"best val={best_val:.4f} sign={best_sign:.3f}  elapsed={time.time() - t0:.1f}s  "
          f"(hidden1={args.hidden1} hidden2={args.hidden2})")


if __name__ == "__main__":
    main()
