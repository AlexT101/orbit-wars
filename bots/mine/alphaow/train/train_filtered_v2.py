"""Test whether FILTERING suspect training rows breaks the 84% ceiling.

Audit found replays_strong.npz has no NaN/Inf/bad-labels; the only suspect data:
  (1) UNRESOLVED games: hit the step-500 cap with neither side eliminated, so the
      winner label is an arbitrary tiebreak (noisy label) — 364 games (~13%).
  (2) EXTREME ship-count outlier rows (ship feature beyond p99.9 ~29k).

We train on ALL train rows vs CLEAN train rows (filters above) and evaluate BOTH
on the SAME full val split (seed=42, 12% games). If filtering the noisy rows from
TRAINING raises full-val accuracy, the gain is real (not eval-set cherry-picking).
Both the symmetric linear and the h64 MLP are tested.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 46
HERE = Path(__file__).resolve().parent
PAIRS = [(i, i + 10) for i in range(10)] + [(20 + i, 29 + i) for i in range(9)]
SHIP_COLS = [0, 1, 10, 11, 20, 29, 38]


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class V2MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return torch.tanh(self.fc2(torch.relu(self.fc1(x)))).squeeze(-1)


def main():
    dev = device()
    d = np.load(HERE / "data/replays_strong.npz")
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    games = meta[:, 0].astype(np.int64)
    N = len(y)

    # ---- build CLEAN row mask ----
    ev = np.arange(N) % 2 == 0
    g_ev, step_ev, X_ev = games[ev], meta[ev, 1], X[ev]
    uniq = np.unique(games)
    # terminal p0 row per game -> resolved? (a side eliminated)
    unresolved_games = set()
    for gi in uniq:
        rows = np.where(g_ev == gi)[0]
        li = rows[np.argmax(step_ev[rows])]
        prod_me = X_ev[li, 5:8].sum(); prod_op = X_ev[li, 15:18].sum()
        ships_me = X_ev[li, 0] + X_ev[li, 1]; ships_op = X_ev[li, 10] + X_ev[li, 11]
        elim = (prod_op == 0 and ships_op == 0) or (prod_me == 0 and ships_me == 0)
        if not elim:
            unresolved_games.add(int(gi))
    in_unresolved = np.array([int(gi) in unresolved_games for gi in games])
    thr = np.percentile(X[:, SHIP_COLS], 99.9, axis=0)
    extreme = (X[:, SHIP_COLS] > thr).any(1)
    clean = ~in_unresolved & ~extreme
    print(f"rows={N}  unresolved-game rows={in_unresolved.sum()} "
          f"({len(unresolved_games)} games)  extreme-outlier rows={extreme.sum()}  "
          f"=> clean rows={clean.sum()} ({100*clean.mean():.1f}%)")

    # ---- val split (seed 42, 12% games) — eval is FULL val, never filtered ----
    rng = np.random.default_rng(42); u = uniq.copy(); rng.shuffle(u)
    n_val = max(1, int(0.12 * len(u)))
    val_games = set(u[:n_val].tolist())
    val = np.array([int(gi) in val_games for gi in games])
    print(f"val rows={val.sum()} (full, unfiltered)\n")

    Xt = torch.from_numpy(X).to(dev); yt = torch.from_numpy(y).to(dev)
    vmask = torch.from_numpy(val).to(dev)
    y0 = y[0::2]; vpair = val[0::2]; yv = y[val]
    loss_fn = nn.SmoothL1Loss()

    def report(name, pred_all):
        s = ((pred_all[val] > 0) == (yv > 0)).mean()
        v0, v1 = pred_all[0::2], pred_all[1::2]
        c = ((v0 > v1) == (y0 > 0))[vpair].mean()
        print(f"    {name:<28} val sign={100*s:.2f}%  comparative={100*c:.2f}%")
        return s, c

    def train_sym(train_mask, tag):
        tm = torch.from_numpy(train_mask & ~val).to(dev)
        me = [i for i, _ in PAIRS]; op = [j for _, j in PAIRS]
        Dtr = (Xt[tm][:, me] - Xt[tm][:, op]); ytr = yt[tm]
        Dall = Xt[:, me] - Xt[:, op]
        torch.manual_seed(42)
        w = nn.Parameter(torch.zeros(len(PAIRS), device=dev))
        opt = torch.optim.Adam([w], lr=0.003)
        for _ in range(12000):
            loss = loss_fn(torch.tanh(Dtr @ w), ytr)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return report(f"symmetric [{tag}]", torch.tanh(Dall @ w).cpu().numpy())

    def train_mlp(train_mask, tag):
        tm_np = train_mask & ~val
        tm = torch.from_numpy(tm_np).to(dev)
        Xtr = Xt[tm]; ytr = yt[tm]
        mean = Xtr.mean(0); std = Xtr.std(0).clamp(min=1e-3)
        Xtr_n = (Xtr - mean) / std; Xall_n = (Xt - mean) / std
        torch.manual_seed(42)
        m = V2MLP(64).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        n = Xtr.shape[0]; best = 1e9; best_pred = None
        for ep in range(100):
            m.train(); idx = torch.randperm(n, device=dev)
            for j in range(0, n, 1024):
                sel = idx[j:j+1024]
                loss = loss_fn(m(Xtr_n[sel]), ytr[sel])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            m.eval()
            with torch.no_grad():
                pall = m(Xall_n)
                vl = loss_fn(pall[vmask], yt[vmask]).item()
            if vl < best: best = vl; best_pred = pall.cpu().numpy()
        return report(f"MLP h64 [{tag}]", best_pred)

    allrows = np.ones(N, bool)
    t0 = time.time()
    print("=== SYMMETRIC linear ===")
    train_sym(allrows, "ALL train")
    train_sym(clean, "CLEAN train")
    print("\n=== MLP h64 ===")
    train_mlp(allrows, "ALL train")
    train_mlp(clean, "CLEAN train")
    print(f"\nreference deployed MLP: sign 84.8% / comparative 84.7%  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
