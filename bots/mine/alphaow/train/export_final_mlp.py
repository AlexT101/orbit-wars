"""Train a final MLP (best config from the bench) and export as AOWV
deployable weights. Optionally averages an N-seed ensemble into a single
AOWV by stacking hidden layers (output = avg of N output heads).

Single-MLP export: standard h64 AOWV.
Ensemble export: produces an MLP with hidden = N*64 whose output is
just the average of the N sub-MLPs' outputs (concat fc1 of each, divide
output weights by N).
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np


def split_mask(meta, seed=42, frac=0.12):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games])


def train_one(X, y, val_mask, hidden, epochs, bs, lr, wd, seed, max_train=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    Xt = torch.from_numpy(X).to(dev); yt = torch.from_numpy(y).to(dev)
    Xtr_full, ytr_full = Xt[~val_mask], yt[~val_mask]
    if max_train and Xtr_full.shape[0] > max_train:
        rng = torch.Generator(device=dev).manual_seed(seed)
        idx = torch.randperm(Xtr_full.shape[0], generator=rng, device=dev)[:max_train]
        Xtr, ytr = Xtr_full[idx], ytr_full[idx]
    else:
        Xtr, ytr = Xtr_full, ytr_full
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(0); std = Xtr.std(0).clamp(min=1e-3)
    in_dim = X.shape[1]
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)
        def forward(self, x):
            return torch.tanh(self.fc2(torch.relu(self.fc1(x)))).squeeze(-1)
    model = MLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss = nn.SmoothL1Loss()
    n = Xtr.shape[0]
    best_sign = 0.0; best_state = None
    for ep in range(epochs):
        idx = torch.randperm(n, device=dev)
        model.train()
        for j in range(0, n, bs):
            sel = idx[j:j+bs]
            xb = (Xtr[sel] - mean) / std
            o = model(xb); l = loss(o, ytr[sel])
            opt.zero_grad(); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = model((Xv - mean) / std)
            sign = ((v > 0) == (yv > 0)).float().mean().item()
        if sign > best_sign:
            best_sign = sign
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 10 == 0:
            print(f"  seed={seed} ep {ep+1}/{epochs} sign={sign:.3f} best={best_sign:.3f}", flush=True)
    # Fold mean/std into fc1
    W1 = best_state["fc1.weight"].numpy().astype(np.float32)
    B1 = best_state["fc1.bias"].numpy().astype(np.float32)
    W2 = best_state["fc2.weight"].numpy().reshape(-1).astype(np.float32)
    B2 = float(best_state["fc2.bias"].numpy().reshape(-1)[0])
    std_np = std.cpu().numpy().astype(np.float32)
    mean_np = mean.cpu().numpy().astype(np.float32)
    W1 = W1 / std_np
    B1 = B1 - W1 @ mean_np
    return best_sign, W1, B1, W2, B2


def write_aowv(out: Path, W1, B1, W2, B2):
    hidden, in_dim = W1.shape
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x564F4157))
    buf.extend(struct.pack("<I", 1))
    buf.extend(struct.pack("<I", in_dim))
    buf.extend(struct.pack("<I", hidden))
    buf.extend(W1.astype(np.float32).tobytes(order="C"))
    buf.extend(B1.astype(np.float32).tobytes(order="C"))
    buf.extend(W2.astype(np.float32).tobytes(order="C"))
    buf.extend(struct.pack("<f", float(B2)))
    out.write_bytes(bytes(buf))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--max-train", type=int, default=2_000_000)
    args = p.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    val_mask = split_mask(meta)
    print(f"X={X.shape}  train={(~val_mask).sum()} val={val_mask.sum()}", flush=True)

    runs = []
    for s in args.seeds:
        t0 = time.time()
        sign, W1, B1, W2, B2 = train_one(X, y, val_mask, args.hidden, args.epochs, args.bs, args.lr, args.wd, s, args.max_train)
        print(f"seed={s}  best val sign-acc = {100*sign:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
        runs.append((sign, W1, B1, W2, B2))

    if len(runs) == 1:
        sign, W1, B1, W2, B2 = runs[0]
        write_aowv(Path(args.out), W1, B1, W2, B2)
        print(f"wrote {args.out}  (single MLP h{args.hidden}, {100*sign:.2f}%)")
    else:
        # Concatenate hidden layers, average output heads
        W1_cat = np.concatenate([r[1] for r in runs], axis=0)  # (N*hidden, in_dim)
        B1_cat = np.concatenate([r[2] for r in runs], axis=0)  # (N*hidden,)
        N = len(runs)
        W2_cat = np.concatenate([r[3] / N for r in runs], axis=0)  # (N*hidden,)
        B2_avg = sum(r[4] for r in runs) / N
        write_aowv(Path(args.out), W1_cat, B1_cat, W2_cat, B2_avg)
        avg_sign = np.mean([r[0] for r in runs])
        print(f"wrote {args.out}  (ensemble of {N} MLP h{args.hidden}, mean per-seed sign-acc {100*avg_sign:.2f}%)")
        print(f"  total hidden = {W1_cat.shape[0]} (={N}*{args.hidden})")


if __name__ == "__main__":
    main()
