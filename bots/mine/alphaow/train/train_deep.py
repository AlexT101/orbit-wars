"""Train a DEEP value net (>=2 hidden layers) on the 46-d summary_v2
features and export it in AOWV v2 (deep) format, which the generalized
Rust loader reads as an N-layer stack.

v2 binary layout (little-endian), matching value_net.rs parse_v2:
  magic     u32 = 0x564F4157  ("AOWV")
  version   u32 = 2
  input_dim u32
  n_layers  u32                       # hidden layers + output layer
  out_dim   u32 * n_layers            # output width of each layer
  per layer, in order:
     w  f32[out_dim * in_dim]  row-major (out first)   # in_dim chains
     b  f32[out_dim]
The final layer has out_dim==1; the bot applies tanh to its output and
ReLU after every earlier layer.

Input normalization (mean/std) is folded into the FIRST layer's weights
and bias so the Rust side feeds raw features. Writes to --out; does NOT
touch v2_replays.bin.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_summary_v2 import INPUT_DIM, device

MAGIC = 0x564F4157


class DeepMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dims = list(hidden)
        layers = []
        prev = INPUT_DIM
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            prev = h
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, x):
        for lin in self.hidden_layers:
            x = torch.relu(lin(x))
        return torch.tanh(self.out(x)).squeeze(-1)


def write_aowv_v2(path: Path, model: DeepMLP, mean: np.ndarray, std: np.ndarray):
    # Fold normalization (x-mean)/std into the first hidden layer.
    lins = list(model.hidden_layers) + [model.out]
    weights = []  # list of (W[out,in], b[out])
    for i, lin in enumerate(lins):
        W = lin.weight.detach().cpu().numpy().astype(np.float32)
        b = lin.bias.detach().cpu().numpy().astype(np.float32)
        if i == 0:
            Wn = W / std
            bn = b - (W / std) @ mean
            weights.append((Wn.astype(np.float32), bn.astype(np.float32)))
        else:
            weights.append((W, b))

    out_dims = [W.shape[0] for (W, _) in weights]
    assert out_dims[-1] == 1, "final layer must have out_dim 1"

    buf = bytearray()
    buf.extend(struct.pack("<I", MAGIC))
    buf.extend(struct.pack("<I", 2))
    buf.extend(struct.pack("<I", INPUT_DIM))
    buf.extend(struct.pack("<I", len(weights)))
    for d in out_dims:
        buf.extend(struct.pack("<I", int(d)))
    for (W, b) in weights:
        buf.extend(W.tobytes(order="C"))
        buf.extend(b.tobytes(order="C"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/replays_strong.npz"])
    p.add_argument("--hidden", nargs="+", type=int, required=True,
                   help="hidden layer sizes, e.g. 256 256")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dev = device()
    print(f"device={dev} hidden={args.hidden}")
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

    torch.manual_seed(args.seed)
    model = DeepMLP(args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.SmoothL1Loss()
    n = Xtr.shape[0]
    best_val, best_sign, best_state = float("inf"), 0.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        idx = torch.randperm(n, device=dev)
        model.train()
        for j in range(0, n, args.batch_size):
            sel = idx[j : j + args.batch_size]
            xb = (Xtr[sel] - mean) / std
            loss = loss_fn(model(xb), ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = model((Xv - mean) / std)
            vl = loss_fn(v, yv).item()
            sign = ((v > 0) == (yv > 0)).float().mean().item()
        if vl < best_val:
            best_val, best_sign = vl, sign
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep < 2:
            print(f"ep {ep:3d} val={vl:.5f} sign={sign:.4f}")
    model.load_state_dict(best_state)
    print(f"best val={best_val:.5f} sign={best_sign:.4f} elapsed={time.time()-t0:.0f}s")

    out = Path(args.out)
    write_aowv_v2(out, model,
                  mean.cpu().numpy().astype(np.float32),
                  std.cpu().numpy().astype(np.float32))
    print(f"wrote {out} ({out.stat().st_size} bytes) arch=46->{'->'.join(map(str,args.hidden))}->1")


if __name__ == "__main__":
    main()
