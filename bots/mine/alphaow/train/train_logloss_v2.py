"""Two questions in one run:

(A) EXTRAPOLATION SANITY — is the extrapolated state actually computed, or did it
    silently collapse to the current state? me_cur ships-on-planets (col 0) vs
    me_ext ships-on-planets (col 20) must DIFFER whenever fleets are in flight
    (ext = after all flying fleets land + combat resolves). We report what fraction
    of rows differ and the magnitude, for both me and opp blocks. If ext == cur
    everywhere, extrapolation is a no-op (bug). If they differ, it ran.

(B) LOG LOSS vs SmoothL1 — the value net predicts P(win), so BCE (log loss) is the
    principled loss. Train symmetric-linear and h64-MLP with each loss on the SAME
    seed=42 12% val split and compare val sign / comparative accuracy. BCE trains on
    a logit z (no tanh); deploy value = tanh(z/2) = 2*sigmoid(z)-1 (monotonic, so
    sign/comparative acc = sign of z — directly comparable to the tanh+Huber models).
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


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class V2MLP(nn.Module):
    """final layer emits a LOGIT (no squash); caller applies tanh or sigmoid."""

    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def main():
    dev = device()
    d = np.load(HERE / "data/replays_strong.npz")
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    games = d["meta"][:, 0].astype(np.int64)
    N = len(y)

    # ---------- (A) extrapolation sanity ----------
    print("=== (A) EXTRAPOLATION SANITY (ext block must differ from cur when fleets fly) ===")
    # me block: cur ships-on-planets = col 0, ext ships-on-planets = col 20
    # opp block: cur = col 10, ext = col 29
    for tag, c_cur, c_ext in [("me", 0, 20), ("opp", 10, 29)]:
        diff = X[:, c_ext] - X[:, c_cur]
        nz = np.abs(diff) > 1e-6
        print(f"  {tag}: rows where ext != cur ships-on-planets = {nz.sum()} "
              f"({100*nz.mean():.1f}%)   mean|Δ|={np.abs(diff).mean():.2f}  "
              f"max|Δ|={np.abs(diff).max():.0f}")
    # ownership flips: ext n_static (col 22) vs cur n_static (col 2) etc — count
    # rows where the extrap planet-count differs (a planet changed hands on landing)
    flip = (np.abs(X[:, 22] - X[:, 2]) + np.abs(X[:, 23] - X[:, 3])) > 1e-6
    print(f"  me: rows where extrap changed my static/orbit planet COUNT (ownership "
          f"flip on landing) = {flip.sum()} ({100*flip.mean():.1f}%)\n")

    # ---------- canonical seed=42 12% game-level val split ----------
    uniq = np.unique(games)
    rng = np.random.default_rng(42); rng.shuffle(uniq)
    n_val = max(1, int(0.12 * len(uniq)))
    val_games = set(uniq[:n_val].tolist())
    val = np.array([int(g) in val_games for g in games])

    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)               # ±1
    y01 = ((yt > 0).float())                        # {0,1} for BCE
    vm = torch.from_numpy(val).to(dev)
    tr = ~vm
    y0 = y[0::2]; vpair = val[0::2]; yv = y[val]

    huber = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss()

    def report(name, logit_all):
        # logit_all: monotonic score (logit for BCE, atanh-ish for Huber — we pass
        # the pre-squash for BCE and tanh-out for Huber; sign at 0 is the boundary
        # for both since tanh(0)=0 and sigmoid(0)=0.5).
        s = ((logit_all[val] > 0) == (yv > 0)).mean()
        v0, v1 = logit_all[0::2], logit_all[1::2]
        c = ((v0 > v1) == (y0 > 0))[vpair].mean()
        print(f"    {name:<26} val sign={100*s:.2f}%  comparative={100*c:.2f}%")
        return s, c

    # ---------- symmetric linear: Huber(tanh) vs BCE(logit) ----------
    print("=== (B) SYMMETRIC linear ===")
    me = [i for i, _ in PAIRS]; op = [j for _, j in PAIRS]
    Dall = Xt[:, me] - Xt[:, op]
    Dtr = Dall[tr]; ytr = yt[tr]; y01tr = y01[tr]

    def train_sym(loss_kind):
        torch.manual_seed(42)
        w = nn.Parameter(torch.zeros(len(PAIRS), device=dev))
        opt = torch.optim.Adam([w], lr=0.003)
        for _ in range(12000):
            z = Dtr @ w
            loss = bce(z, y01tr) if loss_kind == "bce" else huber(torch.tanh(z), ytr)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return (Dall @ w).cpu().numpy()   # raw logit; sign/order unaffected by squash

    report("Huber(tanh)", train_sym("huber"))
    report("BCE(logit)", train_sym("bce"))

    # ---------- MLP h64: Huber(tanh) vs BCE(logit) ----------
    print("\n=== (B) MLP h64 ===")
    mean = Xt[tr].mean(0); std = Xt[tr].std(0).clamp(min=1e-3)
    Xtr_n = (Xt[tr] - mean) / std; Xall_n = (Xt - mean) / std
    ytr_pm = yt[tr]; ytr01 = y01[tr]
    n = Xtr_n.shape[0]

    def train_mlp(loss_kind):
        torch.manual_seed(42)
        m = V2MLP(64).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        best = 1e9; best_logit = None
        for _ in range(100):
            m.train(); idx = torch.randperm(n, device=dev)
            for j in range(0, n, 1024):
                sel = idx[j:j + 1024]
                z = m(Xtr_n[sel])
                loss = bce(z, ytr01[sel]) if loss_kind == "bce" else huber(torch.tanh(z), ytr_pm[sel])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            m.eval()
            with torch.no_grad():
                zall = m(Xall_n)
                if loss_kind == "bce":
                    vl = bce(zall[vm], y01[vm]).item()
                else:
                    vl = huber(torch.tanh(zall[vm]), yt[vm]).item()
            if vl < best:
                best = vl; best_logit = zall.cpu().numpy()
        return best_logit

    t0 = time.time()
    report("Huber(tanh)", train_mlp("huber"))
    report("BCE(logit)", train_mlp("bce"))
    print(f"\nreference deployed MLP (Huber): sign 84.8% / comparative 84.7%  "
          f"(elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
