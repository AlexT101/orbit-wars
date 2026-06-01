"""Audit + retrain of the linear value model on summary_v2 (46-d).

Three things:
  1. Sanity-check the data feed (ranges, label balance, perspective pairing).
  2. Trivial heuristic baselines (sign of ship/production differences) — to see
     what accuracy a dumb rule already gets, i.e. where the task ceiling is.
  3. Two linear fits, both reported with NAMED per-feature weights:
       (a) PLAIN  : y = tanh(w.x + b), 46 free weights + bias. Also checks whether
                    it NATURALLY learned anti-symmetry (w_opp ~= -w_me).
       (b) SYMMETRIC: enforces the zero-sum constraint v(s,me) = -v(s,opp):
                    mirror pairs share one weight with opposite sign, neutral
                    features weight 0, bias 0. Equivalent to a linear model on
                    feature DIFFERENCES (mine - theirs). 19 params.
Both export a deployable AOWV (same 46->2->1 mirror trick as train_linear_v2).
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

# ----- feature names, by block (see value_net.rs summary_features_v2) -----
CUR = ["ships_planets", "ships_flying", "n_static", "n_orbit", "n_comet",
       "prod_static", "prod_orbit", "prod_comet", "n_neut_closer", "n_enemy_closer"]
EXT = ["ships_planets", "n_static", "n_orbit", "n_comet",
       "prod_static", "prod_orbit", "prod_comet", "n_neut_closer", "n_enemy_closer"]
NEU = ["ships", "n_static", "n_orbit", "n_comet",
       "prod_static", "prod_orbit", "prod_comet", "comet_time"]
NAMES = ([f"me_cur.{n}" for n in CUR] + [f"opp_cur.{n}" for n in CUR]
         + [f"me_ext.{n}" for n in EXT] + [f"opp_ext.{n}" for n in EXT]
         + [f"neut.{n}" for n in NEU])
assert len(NAMES) == INPUT_DIM

# mirror pairs (me_idx, opp_idx) and the pair label
PAIRS = [(i, i + 10) for i in range(10)] + [(20 + i, 29 + i) for i in range(9)]
PAIR_NAMES = [f"cur.{n}" for n in CUR] + [f"ext.{n}" for n in EXT]
NEUTRAL = list(range(38, 46))


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def write_aowv_linear(out_path: Path, w_folded: np.ndarray, b_folded: float):
    in_dim = w_folded.shape[0]
    w1 = np.stack([w_folded, -w_folded], axis=0).astype(np.float32)
    b1 = np.array([b_folded, -b_folded], dtype=np.float32)
    w2 = np.array([1.0, -1.0], dtype=np.float32)
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x564F4157))
    buf.extend(struct.pack("<I", 1))
    buf.extend(struct.pack("<I", in_dim))
    buf.extend(struct.pack("<I", 2))
    buf.extend(w1.tobytes(order="C"))
    buf.extend(b1.tobytes(order="C"))
    buf.extend(w2.tobytes(order="C"))
    buf.extend(struct.pack("<f", 0.0))
    out_path.write_bytes(bytes(buf))


def val_split(games, seed=42, frac=0.12):
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_games = set(unique[:n_val].tolist())
    return np.array([g in val_games for g in games]), len(unique), n_val


def acc_report(name, w_raw, b, Xv_raw, yv, val_pairs_mask, Xpair0, Xpair1, ypair0):
    """sign-acc on val rows + comparative-acc on val pairs, for raw-space (w,b)."""
    pv = np.tanh(Xv_raw @ w_raw + b)
    sign = ((pv > 0) == (yv > 0)).mean()
    v0 = np.tanh(Xpair0 @ w_raw + b)
    v1 = np.tanh(Xpair1 @ w_raw + b)
    comp = ((v0 > v1) == (ypair0 > 0))[val_pairs_mask].mean()
    print(f"  {name:<22} val sign-acc={100*sign:.2f}%  comparative={100*comp:.2f}%")
    return sign, comp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "data/replays_strong.npz"))
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dev = device()
    d = np.load(args.data)
    X = d["summary_v2"].astype(np.float32)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    games = meta[:, 0].astype(np.int64)

    # ---------- 1. data sanity ----------
    print(f"=== DATA SANITY ({Path(args.data).name}) ===")
    print(f"rows={len(y)}  feat_shape={X.shape}  label mean={y.mean():+.4f} "
          f"(0=balanced)  label vals={np.unique(y)}")
    nan = np.isnan(X).sum() + np.isinf(X).sum()
    print(f"NaN/Inf in features: {nan}")
    negmin = X.min(0)
    print(f"any feature with negative min? {(negmin < 0).sum()} cols "
          f"(counts/ships/prod should be >=0; only diffs can be neg)")
    # perspective pairing: even=player0, odd=player1 of same state
    assert (meta[0::2, 2] == 0).all() and (meta[1::2, 2] == 1).all()
    assert (y[0::2] == -y[1::2]).all(), "pairs not opposite-labeled"
    print("perspective pairing OK (even=p0, odd=p1, opposite labels)\n")

    # ---------- 2. trivial heuristic baselines (on val pairs) ----------
    val_mask, n_games, n_val = val_split(games, args.seed)
    print(f"split: games={n_games} val={n_val}  rows train={(~val_mask).sum()} val={val_mask.sum()}")
    X0, X1 = X[0::2], X[1::2]                 # p0 view, p1 view of each state
    y0 = y[0::2]                              # +1 if p0 won
    vpair = val_mask[0::2]                    # val mask over pairs
    print("\n=== TRIVIAL BASELINES (predict winner = higher of mine-vs-theirs) ===")
    def base(name, me_idx, opp_idx):
        s0 = X0[:, me_idx].sum(1) - X0[:, opp_idx].sum(1)  # p0 advantage from p0 view
        acc = ((s0 > 0) == (y0 > 0))[vpair].mean()
        print(f"  {name:<34} val comparative-acc={100*acc:.2f}%")
    base("ships_on_planets diff", [0], [10])
    base("ships(planets+flying) diff", [0, 1], [10, 11])
    base("total production diff", [5, 6, 7], [15, 16, 17])
    base("ships + production diff", [0, 1, 5, 6, 7], [10, 11, 15, 16, 17])

    # tensors  — NO input normalization: train directly on raw features.
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    vm = torch.from_numpy(val_mask).to(dev)
    Xtr, ytr = Xt[~vm], yt[~vm]
    yv_t = yt[vm]
    loss_fn = nn.SmoothL1Loss()

    def log(tag, ep, loss, pred_val, yv):
        sign = ((pred_val > 0) == (yv > 0)).float().mean().item()
        print(f"    [{tag}] ep {ep:5d}  train_huber={loss:.4f}  val_sign={100*sign:.2f}%")

    # ---------- 3a. PLAIN linear (raw inputs) ----------
    print("\n=== (a) PLAIN linear: tanh(w.x + b), 46 weights + bias, RAW inputs ===")
    torch.manual_seed(args.seed)
    lin = nn.Linear(INPUT_DIM, 1).to(dev)
    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)  # start unsaturated
    opt = torch.optim.Adam(lin.parameters(), lr=args.lr, weight_decay=args.wd)
    Xv = Xt[vm]
    for ep in range(args.epochs):
        pred = torch.tanh(lin(Xtr).squeeze(-1))
        loss = loss_fn(pred, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % (args.epochs // 6) == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                log("plain", ep, loss.item(), torch.tanh(lin(Xv).squeeze(-1)), yv_t)
    w_plain = lin.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    b_plain = float(lin.bias.detach().cpu().numpy().reshape(-1)[0])
    acc_report("plain", w_plain, b_plain, X[val_mask], y[val_mask], vpair, X0, X1, y0)
    # natural anti-symmetry check
    w_me = np.array([w_plain[i] for i, _ in PAIRS])
    w_op = np.array([w_plain[j] for _, j in PAIRS])
    asym = np.corrcoef(w_me, -w_op)[0, 1]
    print(f"  natural anti-symmetry: corr(w_me, -w_opp)={asym:+.3f}  "
          f"(1.0=perfectly anti-symmetric); |bias|={abs(b_plain):.4f}")
    print(f"  mean |neutral weight|={np.abs(w_plain[NEUTRAL]).mean():.4f} "
          f"vs mean |pair weight|={np.abs(np.concatenate([w_me,w_op])).mean():.4f}")

    # ---------- 3b. SYMMETRIC linear (zero-sum enforced, raw inputs) ----------
    print("\n=== (b) SYMMETRIC linear: tanh(w . (mine - theirs)), 19 weights, bias=0, RAW ===")
    me_idx = [i for i, _ in PAIRS]
    op_idx = [j for _, j in PAIRS]
    Dtr = Xtr[:, me_idx] - Xtr[:, op_idx]        # raw difference features (no scaling)
    Dv = Xv[:, me_idx] - Xv[:, op_idx]
    torch.manual_seed(args.seed)
    wsym = nn.Parameter(torch.zeros(len(PAIRS), device=dev))
    opt = torch.optim.Adam([wsym], lr=args.lr, weight_decay=args.wd)
    for ep in range(args.epochs):
        pred = torch.tanh(Dtr @ wsym)
        loss = loss_fn(pred, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % (args.epochs // 6) == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                log("sym", ep, loss.item(), torch.tanh(Dv @ wsym), yv_t)
    wsym_raw = wsym.detach().cpu().numpy().astype(np.float32)   # weight on (x_me - x_opp), raw units
    # fold into a full 46-d anti-symmetric weight vector, bias 0
    w_full = np.zeros(INPUT_DIM, dtype=np.float32)
    for k, (i, j) in enumerate(PAIRS):
        w_full[i] = wsym_raw[k]
        w_full[j] = -wsym_raw[k]
    acc_report("symmetric", w_full, 0.0, X[val_mask], y[val_mask], vpair, X0, X1, y0)

    order = np.argsort(-np.abs(wsym_raw))
    print("\n  per-feature weights (weight on  mine - theirs,  raw units), by |w|:")
    print(f"  {'feature':<22}{'weight':>10}")
    for k in order:
        print(f"  {PAIR_NAMES[k]:<22}{wsym_raw[k]:>+10.4f}")

    out = HERE / "weights/linear_sym.bin"
    write_aowv_linear(out, w_full, 0.0)
    out_plain = HERE / "weights/linear_v2b.bin"
    write_aowv_linear(out_plain, w_plain, b_plain)
    print(f"\nwrote {out} ({out.stat().st_size} B, symmetric, deployable)")
    print(f"wrote {out_plain} ({out_plain.stat().st_size} B, plain retrain)")
    print("MLP h64 reference: val sign-acc 84.8%, comparative 84.7%")


if __name__ == "__main__":
    main()
