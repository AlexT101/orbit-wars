"""Compare linear / tiny MLP on raw vs handcrafted summary features.

Reports val sign-accuracy + val MSE for each (architecture, feature-set)
pair to test the user's hypothesis that the big MLP is "glorified
regression" — i.e. that a linear model on a few summary features is
nearly as good.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summary_features import summary_features


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_train_val(meta, seed):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.15 * len(unique)))
    val_games = set(unique[:n_val].tolist())
    mask = np.array([g in val_games for g in games])
    return mask  # True = val


def run(X, y, mask, dev, arch, hidden=None, epochs=80, lr=1e-3, wd=1e-3, bs=128, seed=0):
    Xt = torch.from_numpy(X).float().to(dev)
    yt = torch.from_numpy(y).float().to(dev)
    Xtr, ytr = Xt[~mask], yt[~mask]
    Xv, yv = Xt[mask], yt[mask]

    # Normalize.
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp(min=1e-3)

    in_dim = X.shape[1]
    if arch == "linear":
        model = nn.Sequential(nn.Linear(in_dim, 1), nn.Tanh())
    elif arch == "tiny":
        model = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh())
    else:
        raise SystemExit(arch)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss()
    n_train = Xtr.shape[0]
    best_val = float("inf")
    best_sign = 0.0
    torch.manual_seed(seed)
    for ep in range(epochs):
        idx = torch.randperm(n_train, device=dev)
        model.train()
        for j in range(0, n_train, bs):
            sel = idx[j : j + bs]
            xb = (Xtr[sel] - mean) / std
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, ytr[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = model((Xv - mean) / std).squeeze(-1)
            val_loss = loss_fn(v, yv).item()
            sign = ((v > 0) == (yv > 0)).float().mean().item()
        if val_loss < best_val:
            best_val = val_loss
            best_sign = sign
    return best_val, best_sign


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = device()
    print(f"device={dev}")

    arrs = [np.load(p) for p in args.data]
    X = np.concatenate([a["features"] for a in arrs], axis=0).astype(np.float32)
    y = np.concatenate([a["labels"] for a in arrs], axis=0).astype(np.float32)
    meta = np.concatenate([a["meta"] for a in arrs], axis=0)
    mask = split_train_val(meta, args.seed)
    print(f"samples: total={X.shape[0]} train={(~mask).sum()} val={mask.sum()}")

    Xs = summary_features(X).astype(np.float32)
    print(f"feature dims: raw={X.shape[1]}  summary={Xs.shape[1]}")

    print("\n== val (best across training) ==")
    for arch, hidden in [
        ("linear", None),
        ("tiny", 8),
        ("tiny", 32),
        ("tiny", 64),
    ]:
        v_raw, s_raw = run(X, y, mask, dev, arch, hidden, seed=args.seed)
        v_sum, s_sum = run(Xs, y, mask, dev, arch, hidden, seed=args.seed)
        tag = f"{arch}" + (f"-h{hidden}" if hidden else "")
        print(
            f"  {tag:12s}  raw: val_loss={v_raw:.4f} sign={s_raw:.3f}  "
            f"|  summary: val_loss={v_sum:.4f} sign={s_sum:.3f}"
        )


if __name__ == "__main__":
    main()
