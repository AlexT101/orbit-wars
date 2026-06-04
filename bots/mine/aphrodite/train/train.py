"""Train aphrodite's value net on collected (state, outcome) data.

Loss: smooth-L1 between MLP output (in [-1, 1]) and game-outcome label
(-1 / 0 / +1).

Exports weights in the AOWV binary format consumed by
`value_net::load_weights` so the bot can load them via
`APHRODITE_VALUE_NET_PATH`.
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 2728


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ValueMLP(nn.Module):
    def __init__(self, hidden: int, input_dim: int = INPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        y = self.fc2(h)
        return torch.tanh(y).squeeze(-1)


def load_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats_list, labels_list, meta_list = [], [], []
    for p in paths:
        d = np.load(p)
        feats_list.append(d["features"])
        labels_list.append(d["labels"])
        meta_list.append(d["meta"])
    return (
        np.concatenate(feats_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(meta_list, axis=0),
    )


def write_aowv_weights(out_path: Path, model: ValueMLP):
    """Serialize a 2-layer MLP into the AOWV binary format."""
    w1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)  # [hidden, input_dim]
    b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)  # [hidden]
    w2 = model.fc2.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)  # [hidden]
    b2 = float(model.fc2.bias.detach().cpu().numpy().reshape(-1)[0])
    hidden, input_dim = w1.shape
    assert input_dim == INPUT_DIM, input_dim
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x564F4157))  # "AOWV"
    buf.extend(struct.pack("<I", 1))
    buf.extend(struct.pack("<I", input_dim))
    buf.extend(struct.pack("<I", hidden))
    buf.extend(w1.tobytes(order="C"))
    buf.extend(b1.tobytes(order="C"))
    buf.extend(w2.tobytes(order="C"))
    buf.extend(struct.pack("<f", b2))
    out_path.write_bytes(bytes(buf))


def train(args):
    dev = device()
    print(f"device={dev}")
    data_paths = [Path(p) for p in args.data]
    feats, labels, meta = load_dataset(data_paths)
    print(f"loaded {feats.shape[0]} samples from {len(data_paths)} file(s)")

    # Train/val split — game-level so the same game doesn't bleed across.
    game_ids = meta[:, 0].astype(np.int64)
    # Combine source file + game_idx for uniqueness if multiple files were passed.
    if len(data_paths) > 1:
        file_offset = np.zeros_like(game_ids)
        cursor = 0
        for p in data_paths:
            d = np.load(p)
            n = d["features"].shape[0]
            file_offset[cursor : cursor + n] = hash(str(p)) & 0xFFFF_FFFF
            cursor += n
        game_key = (file_offset.astype(np.int64) << 20) | game_ids
    else:
        game_key = game_ids
    unique_games = np.unique(game_key)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_games)
    n_val_games = max(1, int(0.1 * len(unique_games)))
    val_games = set(unique_games[:n_val_games].tolist())
    val_mask = np.array([gk in val_games for gk in game_key])
    print(
        f"games: total={len(unique_games)} val={n_val_games} train={len(unique_games) - n_val_games}"
    )

    X_train = torch.from_numpy(feats[~val_mask]).to(dev)
    y_train = torch.from_numpy(labels[~val_mask]).to(dev)
    X_val = torch.from_numpy(feats[val_mask]).to(dev)
    y_val = torch.from_numpy(labels[val_mask]).to(dev)
    print(f"train={X_train.shape[0]}  val={X_val.shape[0]}")

    # Normalize features — log1p(ships) is already compact, but radii,
    # productions, and pairwise distances aren't.
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-3)

    def norm(x):
        return (x - mean) / std

    model = ValueMLP(args.hidden).to(dev)
    # Wrap with explicit (de-)normalization so we can fold it into fc1 at export.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.SmoothL1Loss()

    bs = args.batch_size
    n_train = X_train.shape[0]
    best_val = float("inf")
    best_state = None
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        idx = torch.randperm(n_train, device=dev)
        model.train()
        train_loss = 0.0
        for j in range(0, n_train, bs):
            sel = idx[j : j + bs]
            xb = norm(X_train[sel])
            yb = y_train[sel]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * sel.shape[0]
        train_loss /= n_train
        model.eval()
        with torch.no_grad():
            val_pred = model(norm(X_val))
            val_loss = loss_fn(val_pred, y_val).item()
            # Sign accuracy as a sanity check.
            sign_correct = ((val_pred > 0) == (y_val > 0)).float().mean().item()
        history.append((epoch, train_loss, val_loss, sign_correct))
        print(
            f"epoch {epoch:3d} train={train_loss:.4f} val={val_loss:.4f} sign_acc={sign_correct:.3f} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best val_loss={best_val:.4f}")

    # Fold normalization into fc1 so the exported MLP takes raw features.
    # fc1(x) = W1 x + b1.  With norm(x) = (x - mean)/std:
    # fc1(norm(x)) = W1 ((x - mean)/std) + b1
    #             = (W1 / std) x + (b1 - W1 (mean/std))
    with torch.no_grad():
        std_cpu = std.cpu().numpy().astype(np.float32)
        mean_cpu = mean.cpu().numpy().astype(np.float32)
        W1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
        b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
        W1_new = W1 / std_cpu  # broadcast over hidden rows
        b1_new = b1 - (W1 / std_cpu) @ mean_cpu
        model.fc1.weight.copy_(torch.from_numpy(W1_new).to(dev))
        model.fc1.bias.copy_(torch.from_numpy(b1_new).to(dev))

    # Sanity: raw forward on a sample should match normalized forward.
    with torch.no_grad():
        x_sample = X_val[:8]
        x_sample_norm = (x_sample - mean) / std
        # After folding, fc1 is now applied to RAW input, not normalized.
        # Reload original W1/b1 to verify folding correctness.
        model_chk = ValueMLP(args.hidden).to(dev)
        model_chk.load_state_dict(best_state)
        ref = model_chk(x_sample_norm)
        cur = model(x_sample)
        max_err = (ref - cur).abs().max().item()
        print(f"fold sanity max_err={max_err:.2e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_aowv_weights(out_path, model)
    print(f"wrote weights to {out_path} ({out_path.stat().st_size} bytes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="NPZ file(s) from collect.py")
    p.add_argument("--out", required=True, help="output AOWV weights file")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
