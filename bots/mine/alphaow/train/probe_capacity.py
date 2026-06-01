"""Probe whether a BIGGER value net (wider and/or deeper) beats the h=256
single-layer ceiling on the 46-d summary_v2 features.

Python-only: trains several architectures on the same data + same by-game
val split (seed-fixed) and reports val SmoothL1, sign-accuracy, and the
train/val gap + rebound (overfit signals). Does NOT export weights and
does NOT touch any .bin -- this only tells us if depth/width is worth the
Rust loader work to deploy. Nothing is deployed.

Architectures are given as hidden-layer lists, e.g. [256] = one hidden
layer of 256, [256,256] = two hidden layers.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from train_summary_v2 import INPUT_DIM, device

# (name, hidden_layers, weight_decay)
ARCHS = [
    ("h256       ", [256], 5e-4),
    ("h256x256   ", [256, 256], 5e-4),
    ("h512x512   ", [512, 512], 5e-4),
    ("h256x256x256", [256, 256, 256], 1e-3),
    ("h512x256x128", [512, 256, 128], 1e-3),
]


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers = []
        prev = INPUT_DIM
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x)).squeeze(-1)


def train_one(hidden, wd, Xtr, ytr, Xv, yv, mean, std, dev, epochs, bs, lr, seed):
    torch.manual_seed(seed)
    model = MLP(hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss()
    n = Xtr.shape[0]
    best_val, best_sign, best_train, best_ep = float("inf"), 0.0, float("inf"), -1
    final_val = float("inf")
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randperm(n, device=dev)
        model.train()
        tl = 0.0
        for j in range(0, n, bs):
            sel = idx[j : j + bs]
            xb = (Xtr[sel] - mean) / std
            loss = loss_fn(model(xb), ytr[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * sel.shape[0]
        tl /= n
        model.eval()
        with torch.no_grad():
            v = model((Xv - mean) / std)
            vl = loss_fn(v, yv).item()
            sign = ((v > 0) == (yv > 0)).float().mean().item()
        final_val = vl
        if vl < best_val:
            best_val, best_sign, best_train, best_ep = vl, sign, tl, ep
    params = sum(p.numel() for p in model.parameters())
    return {
        "params": params, "best_val": best_val, "sign": best_sign,
        "train": best_train, "gap": best_val - best_train,
        "rebound": final_val - best_val, "ep": best_ep, "secs": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/replays_strong.npz"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dev = device()
    print(f"device={dev}")
    Xs, ys, metas = [], [], []
    for path in args.data:
        d = np.load(path)
        Xs.append(d["summary_v2"].astype(np.float32))
        ys.append(d["labels"].astype(np.float32))
        metas.append(d["meta"])
    X = np.concatenate(Xs); y = np.concatenate(ys); meta = np.concatenate(metas)
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_set = set(unique[:n_val].tolist())
    val_mask = np.array([g in val_set for g in games])
    print(f"samples train={(~val_mask).sum()} val={val_mask.sum()}")

    Xt = torch.from_numpy(X).to(dev); yt = torch.from_numpy(y).to(dev)
    Xtr, ytr = Xt[~val_mask], yt[~val_mask]
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(0); std = Xtr.std(0).clamp(min=1e-3)

    print(f"\n{'arch':>12} {'params':>8} {'val_loss':>9} {'sign_acc':>9} "
          f"{'train':>8} {'gap':>8} {'rebound':>8} {'ep':>4}")
    base = None
    for name, hidden, wd in ARCHS:
        r = train_one(hidden, wd, Xtr, ytr, Xv, yv, mean, std, dev,
                      args.epochs, args.batch_size, args.lr, args.seed)
        if base is None:
            base = r
        dv = r["best_val"] - base["best_val"]
        ds = r["sign"] - base["sign"]
        print(f"{name:>12} {r['params']:>8} {r['best_val']:>9.5f} "
              f"{r['sign']:>9.4f} {r['train']:>8.5f} {r['gap']:>+8.5f} "
              f"{r['rebound']:>+8.5f} {r['ep']:>4}  dVal={dv:+.5f} dSign={ds*100:+.2f}pp [{r['secs']:.0f}s]")

    print("\nIf no deeper arch beats h256 on val_loss/sign, the 46-d features")
    print("are the ceiling -> bigger model won't help; need richer features/data.")


if __name__ == "__main__":
    main()
