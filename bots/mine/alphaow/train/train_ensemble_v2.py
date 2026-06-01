"""Push the value net above the 84.8% ceiling by (a) a better training regime
(cosine LR, longer, best-val checkpoint) and (b) ensembling K seeds.

Same data (replays_strong.npz), same seed=42 12% game-level val split as the
deployed MLP, so val numbers are directly comparable (MLP h64: sign 84.8%,
comparative 84.7%). Reports each member, mean+/-std across seeds (to separate a
real gain from split noise), and the averaged ensemble (mean of tanh outputs).

Ensembling reduces MODEL variance, not label noise; since linear ~= MLP here the
variance is small, so don't expect much. This MEASURES whether anything clears
the bar. (Deploying an ensemble would need multi-net loading or distillation.)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 46
HERE = Path(__file__).resolve().parent


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class V2MLP(nn.Module):
    def __init__(self, hidden, depth=1):
        super().__init__()
        layers = [nn.Linear(INPUT_DIM, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return torch.tanh(self.head(self.body(x))).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "data/replays_strong.npz"))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    args = p.parse_args()

    dev = device()
    print(f"device={dev} hidden={args.hidden} depth={args.depth} K={args.k} "
          f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} wd={args.wd}")
    d = np.load(args.data)
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    games = meta[:, 0].astype(np.int64)

    # canonical seed=42 12% game-level split (matches deployed MLP)
    unique = np.unique(games)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_set = set(unique[:n_val].tolist())
    val_mask = np.array([g in val_set for g in games])
    print(f"games total={len(unique)} val={n_val}  rows train={(~val_mask).sum()} val={val_mask.sum()}")

    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    vm = torch.from_numpy(val_mask).to(dev)
    Xtr, ytr = Xt[~vm], yt[~vm]
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp(min=1e-3)
    Xtr_n = (Xtr - mean) / std
    Xall_n = (Xt - mean) / std            # all rows, for pairing-based comparative
    loss_fn = nn.SmoothL1Loss()
    n_train = Xtr.shape[0]

    # pairing for comparative-acc (even=p0 view, odd=p1 view of same state)
    y0 = y[0::2]
    vpair = val_mask[0::2]
    yv_np = y[val_mask]

    def sign_acc(pred_all_np):
        pv = pred_all_np[val_mask]
        return ((pv > 0) == (yv_np > 0)).mean()

    def comp_acc(pred_all_np):
        v0 = pred_all_np[0::2]; v1 = pred_all_np[1::2]
        return ((v0 > v1) == (y0 > 0))[vpair].mean()

    member_preds = []
    signs, comps = [], []
    t0 = time.time()
    for k in range(args.k):
        torch.manual_seed(1000 + k)
        model = V2MLP(args.hidden, args.depth).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best_val = float("inf"); best_pred = None
        for ep in range(args.epochs):
            model.train()
            idx = torch.randperm(n_train, device=dev)
            for j in range(0, n_train, args.batch_size):
                sel = idx[j:j + args.batch_size]
                loss = loss_fn(model(Xtr_n[sel]), ytr[sel])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pall = model(Xall_n)
                vloss = loss_fn(pall[vm], yt[vm]).item()
            if vloss < best_val:
                best_val = vloss
                best_pred = pall.detach().cpu().numpy()
        s, c = sign_acc(best_pred), comp_acc(best_pred)
        signs.append(s); comps.append(c)
        member_preds.append(best_pred)
        print(f"  member {k}: val sign={100*s:.2f}%  comparative={100*c:.2f}%  (best huber={best_val:.4f})")

    signs = np.array(signs); comps = np.array(comps)
    ens = np.mean(member_preds, axis=0)
    print(f"\nper-member  sign: mean={100*signs.mean():.2f}% std={100*signs.std():.2f}% "
          f"max={100*signs.max():.2f}%")
    print(f"per-member  comp: mean={100*comps.mean():.2f}% std={100*comps.std():.2f}% "
          f"max={100*comps.max():.2f}%")
    print(f"ENSEMBLE(K={args.k})  val sign={100*sign_acc(ens):.2f}%  comparative={100*comp_acc(ens):.2f}%")
    print(f"reference MLP h64: sign 84.8%  comparative 84.7%   (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
