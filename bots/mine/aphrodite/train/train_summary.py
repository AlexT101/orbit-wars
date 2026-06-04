"""Train a tiny MLP on summary features and export AOWV-compatible weights.

Input: NPZ files from collect.py (raw 2728-d features) → derives the 19-d
summary via `summary_features.summary_features` and trains a small MLP.
Output: an AOWV binary the aphrodite bot can load (the Rust loader detects
`input_dim==19` and dispatches to its native summary feature extractor,
which matches this Python derivation by construction).
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summary_features import summary_features, FEATURE_NAMES


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SummaryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        if hidden == 0:
            self.body = nn.Linear(in_dim, 1)
            self.hidden = 0
        else:
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)
            self.hidden = hidden

    def forward(self, x):
        if self.hidden == 0:
            y = self.body(x)
        else:
            h = torch.relu(self.fc1(x))
            y = self.fc2(h)
        return torch.tanh(y).squeeze(-1)


def write_aowv_weights(out_path: Path, model: SummaryMLP, in_dim: int):
    """Serialize as an AOWV file. Linear models are emitted as hidden=1
    so the Rust path can still use its `Wx + b → ReLU → Wx + b → tanh`
    structure with the inner Linear's matrix being a 1-row pass-through."""
    if model.hidden == 0:
        # Linear: y = tanh(w . x + b). Encode as hidden=1 with fc1 = identity? No.
        # Easier: encode as hidden=in_dim with fc1 = diag(1), b=0, then fc2 = w, b2 = b.
        # That's expensive on the Rust side. Instead, encode hidden=1, fc1=[w], fc2=[1],
        # which gives y = tanh(1 * relu(w . x + b) + 0). But the ReLU clips negatives → wrong.
        # Cleanest: encode hidden=1, fc1=[w], b1=[b+large_pos], fc2=[1], b2=-large_pos so the
        # ReLU never clips. Pick offset=1e6: fc1 row * x in [-100, 100] + (b + 1e6) ≈ 1e6,
        # ReLU passes through, output = 1e6 + (w.x + b) - 1e6 = w.x + b. Then tanh.
        w = model.body.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
        b = float(model.body.bias.detach().cpu().numpy().reshape(-1)[0])
        hidden = 1
        OFFSET = 1e6
        W1 = w.reshape(1, in_dim)
        B1 = np.array([b + OFFSET], dtype=np.float32)
        W2 = np.array([1.0], dtype=np.float32)
        B2 = -OFFSET
    else:
        W1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)  # [H, in_dim]
        B1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
        W2 = model.fc2.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
        B2 = float(model.fc2.bias.detach().cpu().numpy().reshape(-1)[0])
        hidden = model.hidden

    buf = bytearray()
    buf.extend(struct.pack("<I", 0x564F4157))
    buf.extend(struct.pack("<I", 1))
    buf.extend(struct.pack("<I", in_dim))
    buf.extend(struct.pack("<I", hidden))
    buf.extend(W1.tobytes(order="C"))
    buf.extend(B1.tobytes(order="C"))
    buf.extend(W2.tobytes(order="C"))
    buf.extend(struct.pack("<f", B2))
    out_path.write_bytes(bytes(buf))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, default=32, help="0 = linear")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = device()
    print(f"device={dev}")
    Xs = []
    ys = []
    metas = []
    for path in args.data:
        d = np.load(path)
        Xs.append(d["features"])
        ys.append(d["labels"])
        metas.append(d["meta"])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    meta = np.concatenate(metas, axis=0)
    print(f"loaded {X.shape[0]} samples; deriving summary features...")
    # meta layout: [game_idx, step, player, opp_id] — pass step in.
    steps = meta[:, 1].astype(np.int64)
    Xs = summary_features(X, steps=steps).astype(np.float32)
    print(f"summary feature dim = {Xs.shape[1]}")

    games = meta[:, 0]
    if len(args.data) > 1:
        # Disambiguate game ids across files via per-file offset.
        file_off = np.zeros_like(games)
        cursor = 0
        for path in args.data:
            d = np.load(path)
            n = d["features"].shape[0]
            file_off[cursor : cursor + n] = (hash(path) & 0xFFF) * 10_000
            cursor += n
        games = games + file_off
    unique = np.unique(games)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_games = set(unique[:n_val].tolist())
    val_mask = np.array([g in val_games for g in games])
    print(f"games: total={len(unique)} val={n_val} train={len(unique) - n_val}")
    print(f"samples: train={(~val_mask).sum()} val={val_mask.sum()}")

    Xt = torch.from_numpy(Xs).to(dev)
    yt = torch.from_numpy(y).to(dev)
    Xtr, ytr = Xt[~val_mask], yt[~val_mask]
    Xv, yv = Xt[val_mask], yt[val_mask]
    mean = Xtr.mean(dim=0)
    std = Xtr.std(dim=0).clamp(min=1e-3)

    model = SummaryMLP(Xs.shape[1], args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.SmoothL1Loss()
    n_train = Xtr.shape[0]
    best_val = float("inf")
    best_state = None
    best_ep = -1
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        idx = torch.randperm(n_train, device=dev)
        model.train()
        train_loss = 0.0
        for j in range(0, n_train, args.batch_size):
            sel = idx[j : j + args.batch_size]
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
        history.append((ep, train_loss, val_loss, sign))
        print(f"epoch {ep:3d} train={train_loss:.4f} val={val_loss:.4f} sign_acc={sign:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"best epoch {best_ep} val={best_val:.4f}")

    # Fold normalization into fc1: weights = W / std, bias = B - W (mean/std).
    if args.hidden > 0:
        with torch.no_grad():
            std_np = std.cpu().numpy().astype(np.float32)
            mean_np = mean.cpu().numpy().astype(np.float32)
            W1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
            B1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
            W1_new = W1 / std_np
            B1_new = B1 - (W1 / std_np) @ mean_np
            model.fc1.weight.copy_(torch.from_numpy(W1_new).to(dev))
            model.fc1.bias.copy_(torch.from_numpy(B1_new).to(dev))
    else:
        with torch.no_grad():
            std_np = std.cpu().numpy().astype(np.float32)
            mean_np = mean.cpu().numpy().astype(np.float32)
            W = model.body.weight.detach().cpu().numpy().astype(np.float32)
            B = float(model.body.bias.detach().cpu().numpy().reshape(-1)[0])
            W_new = W / std_np
            B_new = B - float((W / std_np) @ mean_np)
            model.body.weight.copy_(torch.from_numpy(W_new).to(dev))
            model.body.bias.copy_(torch.tensor([B_new]).to(dev))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_aowv_weights(out, model, Xs.shape[1])
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
