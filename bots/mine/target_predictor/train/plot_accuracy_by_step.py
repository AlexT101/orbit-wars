"""Plot per-step (turn-number) accuracy of a trained set-net on the val split.

Top-K hit rate and recall@K computed per turn, then plotted vs the turn index.

  python3 train/plot_accuracy_by_step.py
  python3 train/plot_accuracy_by_step.py --ckpt weights/transformer_v2.pt \
                                          --data data/targets_2k_v2.npz \
                                          --out train/accuracy_by_step.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from set_net import build_model, game_level_split, apply_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).resolve().parent / "weights" / "transformer_v2.pt")
    ap.add_argument("--data", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "targets_2k_v2.npz")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "accuracy_by_step.png")
    ap.add_argument("--bucket", type=int, default=5,
                    help="step bucket width (default 5)")
    ap.add_argument("--max-step", type=int, default=180)
    ap.add_argument("--use-offensive-labels", action="store_true",
                    help="evaluate against `offensive_labels` (zero support launches)")
    ap.add_argument("--max-ship-ratio", type=float, default=None,
                    help="filter val rows where max(my,enemy)/min ship-count > this")
    args = ap.parse_args()

    print(f"loading {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], ck["f_planet"], ck["f_global"],
                        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
                        n_layers=ck.get("n_layers", 2), hidden=ck.get("hidden", 64),
                        dropout=0.0)
    model.load_state_dict(ck["state_dict"]); model.eval()

    print(f"loading {args.data}")
    z = np.load(args.data, allow_pickle=True)
    pf = z["planet_feats"].astype(np.float32); gl = z["globals"].astype(np.float32)
    mk = z["mask"].astype(np.bool_); lb = z["labels"].astype(np.float32); meta = z["meta"].astype(np.int64)

    if args.use_offensive_labels:
        if "offensive_labels" not in z.files:
            raise SystemExit("--use-offensive-labels requires the NPZ to contain `offensive_labels`")
        lb = z["offensive_labels"].astype(np.float32)
        print("  using `offensive_labels` (support launches zeroed)")

    _, val_mask = game_level_split(meta, seed=42, val_frac=0.12)
    pf = pf[val_mask]; gl = gl[val_mask]; mk = mk[val_mask]; lb = lb[val_mask]; meta = meta[val_mask]
    if args.max_ship_ratio is not None:
        my_s = gl[:, 2]; en_s = gl[:, 3]
        hi = np.maximum(my_s, en_s); lo = np.maximum(np.minimum(my_s, en_s), 1.0)
        keep = (hi / lo) <= args.max_ship_ratio
        pf = pf[keep]; gl = gl[keep]; mk = mk[keep]; lb = lb[keep]; meta = meta[keep]
        print(f"  ship-ratio ≤{args.max_ship_ratio}: kept {keep.sum()}/{len(keep)} val rows")
    print(f"  val rows: {len(pf)}  unique games: {len(np.unique(meta[:, 0]))}")

    pfn, gln = apply_norm(pf, gl, ck["p_mean"], ck["p_std"], ck["g_mean"], ck["g_std"])
    print("running inference ...")
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(pfn), 1024):
            out = model(torch.from_numpy(pfn[i:i+1024]),
                        torch.from_numpy(gln[i:i+1024]),
                        torch.from_numpy(mk[i:i+1024]))
            logits_all.append(out.numpy())
    logits = np.concatenate(logits_all, 0)
    logits_masked = np.where(mk, logits, -1e9)

    # ---- compute per-row metrics ----
    steps = meta[:, 1]
    # which rows have any positives?
    n_pos_per_row = (lb * mk).sum(axis=1)
    has_pos = n_pos_per_row > 0
    print(f"  rows w/ ≥1 positive: {int(has_pos.sum())}/{len(steps)}")

    # for each row: top1 / top3 hit, recall@5 / recall@10
    K1 = 1; K3 = 3; K5 = 5; K10 = 10
    order_full = np.argsort(-logits_masked, axis=1)

    def hit_at(K):
        top = order_full[:, :K]
        # gather labels at top-K
        hits = np.take_along_axis(lb, top, axis=1).sum(axis=1)
        return hits

    hits_at_1 = hit_at(K1)
    hits_at_3 = hit_at(K3)
    hits_at_5 = hit_at(K5)
    hits_at_10 = hit_at(K10)

    # top-K hit (≥1) = "was at least one true positive captured in top-K?"
    top1_hit = (hits_at_1 >= 1).astype(np.float32)
    top3_hit = (hits_at_3 >= 1).astype(np.float32)
    recall_at_5 = np.where(n_pos_per_row > 0, hits_at_5 / np.maximum(n_pos_per_row, 1), 0.0)
    recall_at_10 = np.where(n_pos_per_row > 0, hits_at_10 / np.maximum(n_pos_per_row, 1), 0.0)

    # ---- bucket by step ----
    bw = args.bucket
    max_step = int(min(steps.max(), args.max_step))
    bins = list(range(0, max_step + bw, bw))
    centers = []
    p_top1 = []; p_top3 = []; p_r5 = []; p_r10 = []; n_per_bin = []
    pos_rate_per_bin = []
    for lo in bins[:-1]:
        hi = lo + bw
        m = (steps >= lo) & (steps < hi) & has_pos
        n = int(m.sum())
        if n < 30:  # too few to be meaningful
            continue
        centers.append((lo + hi) / 2.0)
        p_top1.append(float(top1_hit[m].mean()))
        p_top3.append(float(top3_hit[m].mean()))
        p_r5.append(float(recall_at_5[m].mean()))
        p_r10.append(float(recall_at_10[m].mean()))
        n_per_bin.append(n)
        # avg positives per row in this bucket
        pos_rate_per_bin.append(float(n_pos_per_row[m].mean()))

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # main panel
    ax1.plot(centers, p_top1, marker="o", label="top-1 hit (any positive in top 1)", color="#3a86ff")
    ax1.plot(centers, p_top3, marker="o", label="top-3 hit", color="#06b6d4")
    ax1.plot(centers, p_r5,   marker="o", label="recall@5",  color="#22c55e")
    ax1.plot(centers, p_r10,  marker="o", label="recall@10", color="#a855f7")
    ax1.set_ylabel("metric on rows with ≥1 positive")
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_title(f"per-turn accuracy   (val split, {sum(n_per_bin)} rows, buckets of {bw} turns)")

    # bottom panel: sample count + positives/row
    ax2b = ax2.twinx()
    ax2.bar(centers, n_per_bin, width=bw * 0.9, color="#1f2937", alpha=0.6, label="rows")
    ax2.set_ylabel("# rows", color="#1f2937")
    ax2.tick_params(axis="y", labelcolor="#1f2937")
    ax2b.plot(centers, pos_rate_per_bin, marker="s", color="#dc2626", label="mean positives / row")
    ax2b.set_ylabel("mean positives per row", color="#dc2626")
    ax2b.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_xlabel("turn number")
    ax2.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, facecolor="white")
    print(f"wrote {args.out}")

    # also dump the numbers
    print("\nturn-bucket   rows   pos/row   top1    top3    recall@5  recall@10")
    for c, n, pp, t1, t3, r5, r10 in zip(centers, n_per_bin, pos_rate_per_bin, p_top1, p_top3, p_r5, p_r10):
        print(f"  {int(c-bw/2):3d}-{int(c+bw/2):3d}    {n:5d}  {pp:6.3f}   {t1:.4f}  {t3:.4f}  {r5:.4f}    {r10:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
