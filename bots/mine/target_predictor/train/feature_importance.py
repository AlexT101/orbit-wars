"""Per-feature permutation importance for a trained set-net.

For each per-planet feature column f:
  - shuffle that column across all real planet slots in the val set
  - re-run inference, measure (Δ AUC, Δ acc, Δ logloss) vs baseline
A high drop means the model depends on that feature.

Wraps the same idea as `model_dashboard.permutation_importance` but adapted to
the (rows × N_max × F) tensor shape — model_dashboard's version is row-wise.

Usage:
  python3 feature_importance.py --ckpt weights/setnet_latest.pt \
                                --data data/targets_2k.npz \
                                --rounds 1 --max-rows 20000
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from set_net import build_model, game_level_split, _approx_auc, apply_norm


def load_ckpt(path: Path, device: str = "cpu") -> dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        ck["arch"], ck["f_planet"], ck["f_global"],
        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
        n_layers=ck.get("n_layers", 2), hidden=ck.get("hidden", 64),
        dropout=ck.get("dropout", 0.1),
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    ck["model"] = model
    ck["device"] = torch.device(device)
    return ck


def predict_all(model, pf, gl, mk, device, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(pf), batch):
            pf_b = torch.from_numpy(pf[i:i + batch]).to(device)
            gl_b = torch.from_numpy(gl[i:i + batch]).to(device)
            mk_b = torch.from_numpy(mk[i:i + batch]).to(device)
            out.append(model(pf_b, gl_b, mk_b).cpu().numpy())
    return np.concatenate(out, axis=0)


def metrics(logits, labels, mask):
    flat_l = logits[mask]; flat_y = labels[mask]
    probs = 1.0 / (1.0 + np.exp(-flat_l))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    acc = float(((probs >= 0.5) == (flat_y >= 0.5)).mean())
    logloss = float(-(flat_y * np.log(probs) + (1 - flat_y) * np.log(1 - probs)).mean())
    auc = _approx_auc(flat_y, probs, max_n=200_000)
    return dict(acc=acc, auc=auc, logloss=logloss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=1,
                    help="how many independent shuffles to average per feature")
    ap.add_argument("--max-rows", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt, args.device)
    feat_names = list(ck.get("feat_names") or [f"f{i}" for i in range(ck["f_planet"])])

    z = np.load(args.data, allow_pickle=True)
    pf_raw = z["planet_feats"].astype(np.float32)
    gl_raw = z["globals"].astype(np.float32)
    mk = z["mask"].astype(np.bool_)
    lb = z["labels"].astype(np.float32)
    meta = z["meta"].astype(np.int64)

    _, val_mask = game_level_split(meta, seed=args.seed, val_frac=0.12)
    pf_raw = pf_raw[val_mask]; gl_raw = gl_raw[val_mask]
    mk = mk[val_mask]; lb = lb[val_mask]

    if len(pf_raw) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pf_raw), size=args.max_rows, replace=False)
        pf_raw = pf_raw[idx]; gl_raw = gl_raw[idx]; mk = mk[idx]; lb = lb[idx]

    pf, gl = apply_norm(pf_raw, gl_raw, ck["p_mean"], ck["p_std"], ck["g_mean"], ck["g_std"])

    base_logits = predict_all(ck["model"], pf, gl, mk, ck["device"])
    base = metrics(base_logits, lb, mk)
    print(f"baseline  acc={base['acc']:.4f}  auc={base['auc']:.4f}  logloss={base['logloss']:.4f}")
    print(f"val rows: {len(pf)}  real planet slots: {int(mk.sum())}")

    rng = np.random.default_rng(args.seed)
    results = []
    F = pf.shape[-1]
    t0 = time.time()
    for f_i in range(F):
        deltas = []
        for r in range(args.rounds):
            shuffled = pf.copy()
            # permute the f_i column across real planet slots only
            real = mk
            col = shuffled[..., f_i][real]
            rng.shuffle(col)
            shuffled[..., f_i][real] = col
            logits = predict_all(ck["model"], shuffled, gl, mk, ck["device"])
            m = metrics(logits, lb, mk)
            deltas.append((base["acc"] - m["acc"], base["auc"] - m["auc"], m["logloss"] - base["logloss"]))
        d_acc = float(np.mean([d[0] for d in deltas]))
        d_auc = float(np.mean([d[1] for d in deltas]))
        d_ll = float(np.mean([d[2] for d in deltas]))
        results.append((f_i, feat_names[f_i] if f_i < len(feat_names) else f"f{f_i}", d_acc, d_auc, d_ll))
        if (f_i + 1) % 5 == 0:
            print(f"  [{time.time() - t0:.1f}s] {f_i+1}/{F} done", flush=True)

    # rank by AUC drop (most important first)
    results.sort(key=lambda r: -r[3])
    print(f"\nrank  feature                          Δacc    ΔAUC    Δlogloss")
    for rank, (i, name, d_acc, d_auc, d_ll) in enumerate(results):
        sig = " *" if d_auc > 0.005 else ""
        print(f"  {rank:3d}  {name:30s}  {d_acc:+.4f}  {d_auc:+.4f}  {d_ll:+.4f}{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
