"""Per-feature permutation importance for the pair model. Same idea as
feature_importance.py but the metric is pair_logloss / pair_recall@K instead of
per-planet recall.

For each per-planet feature column f:
  - shuffle f across all real planet slots in the val set
  - re-run pair forward, measure Δpair_logloss and Δpair_recall@8
A larger drop means the model leans on that feature more.

Usage:
  python3 train/pair_feature_importance.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from set_net import apply_norm, game_level_split, load_npz
from pair_net import (PlanetTransformerPair, dense_pair_labels, pair_mask,
                       eval_pair_metrics, predict_pair_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).resolve().parent / "weights" / "transformer_pair.pt")
    ap.add_argument("--data", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "targets_2k_v5.npz")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PlanetTransformerPair(
        ck["f_planet"], ck["f_global"],
        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
        n_layers=ck.get("n_layers", 2), ff=ck.get("ff", 128), dropout=0.0,
    )
    model.load_state_dict(ck["state_dict"]); model.eval()
    feat_names = list(ck.get("feat_names") or [f"f{i}" for i in range(ck["f_planet"])])

    z = np.load(args.data, allow_pickle=True)
    pf_raw = z["planet_feats"].astype(np.float32)
    gl_raw = z["globals"].astype(np.float32)
    mk = z["mask"].astype(np.bool_)
    meta = z["meta"].astype(np.int64)
    pair_src = z["pair_source_idx"].astype(np.int64)
    pair_tgt = z["pair_target_idx"].astype(np.int64)

    # match the trainer's filters: step < 200
    keep = meta[:, 1] < ck.get("max_step", 200)
    pf_raw = pf_raw[keep]; gl_raw = gl_raw[keep]; mk = mk[keep]
    meta = meta[keep]; pair_src = pair_src[keep]; pair_tgt = pair_tgt[keep]

    _, val_mask = game_level_split(meta, seed=args.seed, val_frac=0.12)
    pf_raw = pf_raw[val_mask]; gl_raw = gl_raw[val_mask]
    mk = mk[val_mask]; pair_src = pair_src[val_mask]; pair_tgt = pair_tgt[val_mask]

    if len(pf_raw) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pf_raw), size=args.max_rows, replace=False)
        pf_raw = pf_raw[idx]; gl_raw = gl_raw[idx]
        mk = mk[idx]; pair_src = pair_src[idx]; pair_tgt = pair_tgt[idx]
    print(f"  val rows used: {len(pf_raw)}")

    pf, gl = apply_norm(pf_raw, gl_raw, ck["p_mean"], ck["p_std"], ck["g_mean"], ck["g_std"])

    print("baseline forward ...")
    base_logits = predict_pair_all(model, pf, gl, mk, torch.device("cpu"), batch=128)
    base = eval_pair_metrics(base_logits, pair_src, pair_tgt, mk)
    print(f"baseline  pair_logloss={base['pair_logloss']:.4f}  "
          f"pair_recall@8={base['pair_recall8']:.4f}  recall@10={base['pair_recall10']:.4f}")

    rng = np.random.default_rng(args.seed)
    results = []
    F = pf.shape[-1]
    t0 = time.time()
    for f_i in range(F):
        shuffled = pf.copy()
        col = shuffled[..., f_i][mk]
        rng.shuffle(col)
        shuffled[..., f_i][mk] = col
        logits = predict_pair_all(model, shuffled, gl, mk, torch.device("cpu"), batch=128)
        m = eval_pair_metrics(logits, pair_src, pair_tgt, mk)
        d_ll = m["pair_logloss"] - base["pair_logloss"]    # higher = worse, so positive
        d_r8 = base["pair_recall8"] - m["pair_recall8"]    # drop in recall is positive
        results.append((f_i, feat_names[f_i] if f_i < len(feat_names) else f"f{f_i}",
                        d_ll, d_r8))
        if (f_i + 1) % 5 == 0:
            print(f"  [{time.time() - t0:.1f}s] {f_i+1}/{F} done", flush=True)

    results.sort(key=lambda r: -r[3])  # rank by recall drop
    print(f"\nrank  feature                            Δlogloss   Δrecall@8")
    for rank, (i, name, d_ll, d_r8) in enumerate(results):
        sig = " *" if d_r8 > 0.002 else ""
        print(f"  {rank:3d}  {name:30s}    {d_ll:+.5f}   {d_r8:+.4f}{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
