"""Width sweep for the v2 value net (single hidden layer, 46->H->1).

Trains several hidden widths on the SAME data and the SAME by-game val
split (seed-fixed), so val metrics are directly comparable. Reports, per
width: best val SmoothL1, sign-agreement accuracy at that checkpoint, the
train loss at that epoch (train/val gap), and how much val rebounded after
its minimum (overfitting signal). Writes each model to weights/sweep_h{H}.bin.

Does NOT touch v2_replays.bin / current.bin. Compare first, deploy never
(by hand only, after confirming a width is strictly better).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

from train_summary_v2 import INPUT_DIM, V2MLP, write_aowv, device


def load_data(paths):
    Xs, ys, metas = [], [], []
    for path in paths:
        d = np.load(path)
        if "summary_v2" not in d.files or d["summary_v2"].shape[1] != INPUT_DIM:
            print(f"  skip {path}")
            continue
        Xs.append(d["summary_v2"].astype(np.float32))
        ys.append(d["labels"].astype(np.float32))
        metas.append(d["meta"])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    meta = np.concatenate(metas, axis=0)
    return X, y, meta


def train_one(hidden, Xtr, ytr, Xv, yv, mean, std, dev, epochs, bs, lr, wd, seed):
    torch.manual_seed(seed)
    model = V2MLP(hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss()
    n_train = Xtr.shape[0]

    best_val = float("inf")
    best_sign = 0.0
    best_train = float("inf")
    best_ep = -1
    best_state = None
    final_val = float("inf")
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randperm(n_train, device=dev)
        model.train()
        tl = 0.0
        for j in range(0, n_train, bs):
            sel = idx[j : j + bs]
            xb = (Xtr[sel] - mean) / std
            pred = model(xb)
            loss = loss_fn(pred, ytr[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * sel.shape[0]
        tl /= n_train
        model.eval()
        with torch.no_grad():
            v = model((Xv - mean) / std)
            vl = loss_fn(v, yv).item()
            sign = ((v > 0) == (yv > 0)).float().mean().item()
        final_val = vl
        if vl < best_val:
            best_val, best_sign, best_train, best_ep = vl, sign, tl, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)

    # Fold normalization into fc1 for export.
    with torch.no_grad():
        std_np = std.cpu().numpy().astype(np.float32)
        mean_np = mean.cpu().numpy().astype(np.float32)
        W1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
        B1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
        model.fc1.weight.copy_(torch.from_numpy(W1 / std_np).to(dev))
        model.fc1.bias.copy_(torch.from_numpy(B1 - (W1 / std_np) @ mean_np).to(dev))

    params = sum(p.numel() for p in model.parameters())
    return {
        "hidden": hidden,
        "params": params,
        "best_val": best_val,
        "sign": best_sign,
        "train_at_best": best_train,
        "gap": best_val - best_train,
        "rebound": final_val - best_val,
        "best_ep": best_ep,
        "secs": time.time() - t0,
        "model": model,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/replays_strong.npz"])
    p.add_argument("--widths", nargs="+", type=int, default=[64, 128, 256, 512])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="weights")
    args = p.parse_args()

    dev = device()
    print(f"device={dev}")
    X, y, meta = load_data(args.data)
    print(f"loaded {X.shape[0]} samples")

    games = meta[:, 0].astype(np.int64)
    if len(args.data) > 1:
        offsets = np.zeros_like(games)
        cursor = 0
        for path in args.data:
            d = np.load(path)
            n = d["summary_v2"].shape[0]
            offsets[cursor : cursor + n] = (hash(path) & 0xFFF) * 100_000
            cursor += n
        games = games + offsets
    unique = np.unique(games)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_set = set(unique[:n_val].tolist())
    val_mask = np.array([g in val_set for g in games])
    print(f"games total={len(unique)} val={n_val} | samples train={(~val_mask).sum()} val={val_mask.sum()}")

    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    Xtr, ytr = Xt[~val_mask], yt[~val_mask]
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp(min=1e-3)

    results = []
    for h in args.widths:
        r = train_one(h, Xtr, ytr, Xv, yv, mean, std, dev,
                      args.epochs, args.batch_size, args.lr, args.wd, args.seed)
        out = Path(args.outdir) / f"sweep_h{h}.bin"
        write_aowv(out, r["model"])
        r["path"] = str(out)
        results.append(r)
        print(f"h={h:4d} params={r['params']:6d} best_val={r['best_val']:.5f} "
              f"sign={r['sign']:.4f} train@best={r['train_at_best']:.5f} "
              f"gap={r['gap']:+.5f} rebound={r['rebound']:+.5f} "
              f"ep={r['best_ep']} {r['secs']:.0f}s -> {out}")

    print("\n=== SUMMARY (lower val = better, higher sign = better) ===")
    base = results[0]
    print(f"{'hidden':>7} {'params':>7} {'val_loss':>9} {'sign_acc':>9} "
          f"{'train':>8} {'gap':>8} {'rebound':>8}")
    for r in results:
        dv = r["best_val"] - base["best_val"]
        ds = r["sign"] - base["sign"]
        print(f"{r['hidden']:>7} {r['params']:>7} {r['best_val']:>9.5f} "
              f"{r['sign']:>9.4f} {r['train_at_best']:>8.5f} {r['gap']:>+8.5f} "
              f"{r['rebound']:>+8.5f}   dVal={dv:+.5f} dSign={ds:+.4f}")
    print("\nVERDICT GUIDE: a width is a deploy candidate only if val_loss is")
    print("lower AND sign_acc >= baseline AND gap/rebound are not much larger")
    print("than baseline (small rebound = no overfit). Nothing deployed.")


if __name__ == "__main__":
    main()
