"""Train an MLP on the 46-d summary_v2 features dumped inline by the
Rust bot (or extracted from kaggle replays via from_replays.py).

Reads `summary_v2` directly from NPZ — no Python derivation needed, so
Rust↔Python parity is automatic by construction.

Exports AOWV-format weights (input_dim=46) which the bot loader
dispatches to `value_net::summary_features_v2::extract`.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 46


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class V2MLP(nn.Module):
    def __init__(self, hidden: int, hidden2: int = 0):
        super().__init__()
        self.hidden2 = hidden2
        self.fc1 = nn.Linear(INPUT_DIM, hidden)
        if hidden2 > 0:
            self.fc2 = nn.Linear(hidden, hidden2)
            self.fc3 = nn.Linear(hidden2, 1)
        else:
            self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        if self.hidden2 > 0:
            h = torch.relu(self.fc2(h))
            y = self.fc3(h)
        else:
            y = self.fc2(h)
        return torch.tanh(y).squeeze(-1)


def write_aowv(out_path: Path, model: V2MLP):
    w1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
    b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
    hidden, input_dim = w1.shape
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x564F4157))
    if model.hidden2 > 0:
        w2 = model.fc2.weight.detach().cpu().numpy().astype(np.float32)
        b2 = model.fc2.bias.detach().cpu().numpy().astype(np.float32)
        w3 = model.fc3.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
        b3 = float(model.fc3.bias.detach().cpu().numpy().reshape(-1)[0])
        hidden2 = w2.shape[0]
        buf.extend(struct.pack("<I", 2))
        buf.extend(struct.pack("<I", input_dim))
        buf.extend(struct.pack("<I", hidden))
        buf.extend(struct.pack("<I", hidden2))
        buf.extend(w1.tobytes(order="C"))
        buf.extend(b1.tobytes(order="C"))
        buf.extend(w2.tobytes(order="C"))
        buf.extend(b2.tobytes(order="C"))
        buf.extend(w3.tobytes(order="C"))
        buf.extend(struct.pack("<f", b3))
    else:
        w2 = model.fc2.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
        b2 = float(model.fc2.bias.detach().cpu().numpy().reshape(-1)[0])
        buf.extend(struct.pack("<I", 1))
        buf.extend(struct.pack("<I", input_dim))
        buf.extend(struct.pack("<I", hidden))
        buf.extend(w1.tobytes(order="C"))
        buf.extend(b1.tobytes(order="C"))
        buf.extend(w2.tobytes(order="C"))
        buf.extend(struct.pack("<f", b2))
    out_path.write_bytes(bytes(buf))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--hidden2", type=int, default=0, help="0 = one hidden layer; >0 = two hidden layers")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dev = device()
    print(f"device={dev}")
    Xs, ys, metas = [], [], []
    skipped = 0
    for path in args.data:
        d = np.load(path)
        if "summary_v2" not in d.files:
            print(f"  skip {path}: no summary_v2 field (old NPZ)")
            skipped += 1
            continue
        v2 = d["summary_v2"]
        # Older self-play NPZs might have all-zeros if dump format was
        # pre-v2; check by sum of abs.
        if v2.shape[1] != INPUT_DIM:
            print(f"  skip {path}: summary_v2 dim={v2.shape[1]} != {INPUT_DIM}")
            skipped += 1
            continue
        if np.abs(v2).sum() == 0:
            print(f"  skip {path}: summary_v2 is all-zero (legacy dump)")
            skipped += 1
            continue
        Xs.append(v2.astype(np.float32))
        ys.append(d["labels"].astype(np.float32))
        metas.append(d["meta"])
    if not Xs:
        print(f"no usable data (skipped {skipped} files)", file=sys.stderr)
        sys.exit(1)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    meta = np.concatenate(metas, axis=0)
    print(f"loaded {X.shape[0]} samples (skipped {skipped} files)")

    games = meta[:, 0].astype(np.int64)
    if len(args.data) > 1:
        offsets = np.zeros_like(games)
        cursor = 0
        for path, x in zip(args.data, Xs):
            n = x.shape[0]
            offsets[cursor : cursor + n] = (hash(path) & 0xFFF) * 100_000
            cursor += n
        games = games + offsets
    unique = np.unique(games)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_set = set(unique[:n_val].tolist())
    val_mask = np.array([g in val_set for g in games])
    print(f"games: total={len(unique)} val={n_val} train={len(unique) - n_val}")
    print(f"samples: train={(~val_mask).sum()} val={val_mask.sum()}")

    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    Xtr, ytr = Xt[~val_mask], yt[~val_mask]
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp(min=1e-3)

    model = V2MLP(args.hidden, args.hidden2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.SmoothL1Loss()
    bs = args.batch_size
    n_train = Xtr.shape[0]
    best_val = float("inf")
    best_sign = 0.0
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        idx = torch.randperm(n_train, device=dev)
        model.train()
        train_loss = 0.0
        for j in range(0, n_train, bs):
            sel = idx[j : j + bs]
            xb = (Xtr[sel] - mean) / std
            pred = model(xb)
            loss = loss_fn(pred, ytr[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * sel.shape[0]
        train_loss /= n_train
        model.eval()
        with torch.no_grad():
            v = model((Xv - mean) / std)
            val_loss = loss_fn(v, yv).item()
            sign = ((v > 0) == (yv > 0)).float().mean().item()
        if val_loss < best_val:
            best_val = val_loss
            best_sign = sign
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"ep {ep:3d} train={train_loss:.4f} val={val_loss:.4f} sign={sign:.3f}")

    model.load_state_dict(best_state)
    print(f"best val={best_val:.4f} sign={best_sign:.3f}  elapsed={time.time() - t0:.1f}s")

    # Fold normalization into fc1.
    with torch.no_grad():
        std_np = std.cpu().numpy().astype(np.float32)
        mean_np = mean.cpu().numpy().astype(np.float32)
        W1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
        B1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
        W1_new = W1 / std_np
        B1_new = B1 - (W1 / std_np) @ mean_np
        model.fc1.weight.copy_(torch.from_numpy(W1_new).to(dev))
        model.fc1.bias.copy_(torch.from_numpy(B1_new).to(dev))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_aowv(out, model)
    arch = f"{INPUT_DIM}->{args.hidden}" + (f"->{args.hidden2}" if args.hidden2 > 0 else "") + "->1"
    print(f"wrote {out} ({out.stat().st_size} bytes, arch={arch})")


if __name__ == "__main__":
    main()
