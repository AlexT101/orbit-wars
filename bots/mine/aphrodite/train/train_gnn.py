"""Train a planet-only GNN value model from the same NPZ data we feed
the summary MLP. The training-time graph is reconstructed from the
2728-d raw features (PER_OBJECT slots per planet + the pairwise
distance matrix). Inference will be reimplemented in Rust later.

Architecture (concrete, intentionally small):
  - Per-node input: 18 dims (CURRENT 9-slot row + EXTRAP 9-slot row).
  - Per-edge input: 4 dims (1/(1+dist), is_close, dist_norm, same_owner).
  - Layer 1: node encoder MLP (18 → H).
  - Layer 2: one message-passing pass.
       msg_ij = MLP_msg(h_j ⊕ edge_feat_ij)         # 28 → H
       agg_i  = (sum_j msg_ij) / max_neighbors      # mean-aggregate
       h'_i   = MLP_upd(h_i ⊕ agg_i)                # 2H → H
  - Layer 3: pool over real planets (mean + max → 2H).
  - Head: 2H → H → 1, tanh.

We use the existence mask (is_me + is_opp + is_neutral > 0) to ignore
padded planet slots in the pooling/aggregation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summary_features import PER_OBJECT, MAX_OBJECTS, split_blocks


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_graph_tensors(X: np.ndarray):
    """Return node features [B, MAX, NODE_DIM], edge features
    [B, MAX, MAX, EDGE_DIM], existence mask [B, MAX]."""
    B = X.shape[0]
    cur, ext, dist = split_blocks(X)
    # NODE_DIM = 18: 9 current + 9 extrap slots
    node = np.concatenate([cur, ext], axis=2).astype(np.float32)  # [B, MAX, 18]
    # Edges: distance-derived features
    inv_d = 1.0 / (1.0 + dist)  # [B, MAX, MAX]
    is_close = (dist < 30.0).astype(np.float32)
    dist_norm = dist / 141.42
    cur_is_me = cur[..., 0]
    cur_is_opp = cur[..., 1]
    cur_is_neutral = cur[..., 2]
    # same_owner: 1 iff both planets share the same owner (one-hot match).
    same_me = cur_is_me[:, :, None] * cur_is_me[:, None, :]
    same_opp = cur_is_opp[:, :, None] * cur_is_opp[:, None, :]
    same_neutral = cur_is_neutral[:, :, None] * cur_is_neutral[:, None, :]
    same_owner = same_me + same_opp + same_neutral
    edge = np.stack([inv_d, is_close, dist_norm, same_owner], axis=-1).astype(np.float32)
    # Existence: any one-hot > 0 → real planet
    exists = ((cur_is_me + cur_is_opp + cur_is_neutral) > 0).astype(np.float32)
    return node, edge, exists


NODE_DIM = 18
EDGE_DIM = 4


class GNN(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        self.node_enc = nn.Sequential(
            nn.Linear(NODE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.msg = nn.Sequential(
            nn.Linear(hidden + EDGE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node, edge, exists):
        """node: [B, M, NODE_DIM]; edge: [B, M, M, EDGE_DIM]; exists: [B, M]"""
        B, M, _ = node.shape
        h = self.node_enc(node)  # [B, M, H]
        # Build per-edge input: pair h_j with edge_ij (broadcast h over i axis).
        h_j = h.unsqueeze(1).expand(B, M, M, self.hidden)  # [B, i, j, H]
        msg_in = torch.cat([h_j, edge], dim=-1)  # [B, i, j, H+EDGE]
        m = self.msg(msg_in)  # [B, i, j, H]
        # Mask out non-existent neighbors j.
        exists_j = exists.unsqueeze(1).unsqueeze(-1)  # [B, 1, j, 1]
        m = m * exists_j
        n_neighbors = exists.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
        agg = m.sum(dim=2) / n_neighbors.unsqueeze(-1)  # [B, i, H]
        h2 = self.upd(torch.cat([h, agg], dim=-1))  # [B, M, H]
        # Pool over existing nodes
        exists_i = exists.unsqueeze(-1)  # [B, M, 1]
        h2_masked = h2 * exists_i
        n_exist = exists.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
        mean_pool = h2_masked.sum(dim=1) / n_exist  # [B, H]
        # Max-pool: set padded entries to -inf so they don't win.
        neg_inf = torch.full_like(h2, -1e9)
        h2_for_max = torch.where(exists_i.bool(), h2, neg_inf)
        max_pool, _ = h2_for_max.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2H]
        y = self.head(pooled).squeeze(-1)
        return torch.tanh(y)


def split_train_val(meta, seed):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(0.12 * len(unique)))
    val_games = set(unique[:n_val].tolist())
    return np.array([g in val_games for g in games])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True, help="output state_dict .pt path (Rust port comes later)")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
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
    print(f"samples={X.shape[0]}; building graph tensors...")
    node, edge, exists = build_graph_tensors(X)
    print(f"node: {node.shape}  edge: {edge.shape}  exists: {exists.shape}")

    val_mask = split_train_val(meta, args.seed)
    node_t = torch.from_numpy(node).to(dev)
    edge_t = torch.from_numpy(edge).to(dev)
    exists_t = torch.from_numpy(exists).to(dev)
    y_t = torch.from_numpy(y).to(dev)
    val_idx = torch.from_numpy(val_mask).to(dev)

    model = GNN(args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.SmoothL1Loss()

    train_mask = ~val_idx
    n_train = train_mask.sum().item()
    n_val = val_idx.sum().item()
    print(f"train={n_train} val={n_val}")
    print(f"params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    best_sign = 0.0
    best_state = None
    t0 = time.time()
    bs = args.batch_size
    for ep in range(args.epochs):
        train_idx = torch.where(train_mask)[0]
        train_idx = train_idx[torch.randperm(train_idx.shape[0], device=dev)]
        model.train()
        train_loss = 0.0
        n_seen = 0
        for j in range(0, train_idx.shape[0], bs):
            sel = train_idx[j : j + bs]
            pred = model(node_t[sel], edge_t[sel], exists_t[sel])
            loss = loss_fn(pred, y_t[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * sel.shape[0]
            n_seen += sel.shape[0]
        train_loss /= n_seen
        model.eval()
        val_loss = 0.0
        val_sign_correct = 0
        val_seen = 0
        val_idxs = torch.where(val_idx)[0]
        with torch.no_grad():
            for j in range(0, val_idxs.shape[0], bs):
                sel = val_idxs[j : j + bs]
                pred = model(node_t[sel], edge_t[sel], exists_t[sel])
                l = loss_fn(pred, y_t[sel]).item()
                val_loss += l * sel.shape[0]
                val_sign_correct += ((pred > 0) == (y_t[sel] > 0)).sum().item()
                val_seen += sel.shape[0]
        val_loss /= val_seen
        sign_acc = val_sign_correct / val_seen
        print(f"epoch {ep:3d} train={train_loss:.4f} val={val_loss:.4f} sign_acc={sign_acc:.3f} elapsed={time.time() - t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            best_sign = sign_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    print(f"best val_loss={best_val:.4f} sign_acc={best_sign:.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": args.hidden, "state_dict": best_state, "val_loss": best_val, "sign_acc": best_sign}, out)
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()
