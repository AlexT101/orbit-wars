"""Inference helper: load a trained set-net checkpoint and run one forward pass.

  ckpt = load_checkpoint("weights/setnet_latest.pt")
  probs = predict(ckpt, planet_feats, globals_)  # (N_planets,) sigmoid

`planet_feats` is `(N_planets, F_planet)` unnormalized — the loader applies the
mean/std fit at training time. `globals_` is `(F_global,)`. Padding to N_max is
handled internally.

CLI: predict.py --ckpt weights/setnet_latest.pt --npz data/targets_2k.npz --rows 0,1,2
  loads rows from a built NPZ and prints (label, prob) per planet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from set_net import build_model

N_MAX_DEFAULT = 30


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        ckpt["arch"], ckpt["f_planet"], ckpt["f_global"],
        d_model=ckpt.get("d_model", 64), n_heads=ckpt.get("n_heads", 4),
        n_layers=ckpt.get("n_layers", 2), hidden=ckpt.get("hidden", 64),
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    ckpt["model"] = model
    ckpt["device"] = torch.device(device)
    return ckpt


def predict(ckpt: dict, planet_feats: np.ndarray, globals_: np.ndarray,
            n_max: int = N_MAX_DEFAULT) -> np.ndarray:
    """One row. Returns sigmoid probabilities, length == n_real_planets."""
    n_real = planet_feats.shape[0]
    if n_real > n_max:
        raise ValueError(f"got {n_real} planets, model padded to {n_max}")
    pf = (planet_feats - ckpt["p_mean"]) / ckpt["p_std"]
    gl = (globals_ - ckpt["g_mean"]) / ckpt["g_std"]
    pf_pad = np.zeros((1, n_max, pf.shape[-1]), dtype=np.float32)
    pf_pad[0, :n_real] = pf
    mask = np.zeros((1, n_max), dtype=bool)
    mask[0, :n_real] = True
    with torch.no_grad():
        pf_t = torch.from_numpy(pf_pad).to(ckpt["device"])
        gl_t = torch.from_numpy(gl.reshape(1, -1).astype(np.float32)).to(ckpt["device"])
        mk_t = torch.from_numpy(mask).to(ckpt["device"])
        logits = ckpt["model"](pf_t, gl_t, mk_t).cpu().numpy()[0, :n_real]
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--rows", default="0",
                    help="comma-separated row indices into the NPZ (default 0)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt, args.device)
    z = np.load(args.npz, allow_pickle=True)
    pf_all = z["planet_feats"]; gl_all = z["globals"]
    mask_all = z["mask"]; lb_all = z["labels"]; meta_all = z["meta"]; ids_all = z["planet_ids"]

    for r in (int(x) for x in args.rows.split(",")):
        m = mask_all[r]
        n_real = int(m.sum())
        feats = pf_all[r, :n_real]
        globals_ = gl_all[r]
        probs = predict(ckpt, feats, globals_)
        labels = lb_all[r, :n_real]
        pids = ids_all[r, :n_real]
        gid, step, player, _ = meta_all[r]
        order = np.argsort(-probs)
        print(f"\nrow {r}  game={gid} step={step} player={player}  n_planets={n_real}")
        print(f"  {'rank':>4} {'planet':>6} {'prob':>7}  {'label':>5}")
        for rank, i in enumerate(order):
            tag = " <-- target" if labels[i] > 0.5 else ""
            print(f"  {rank:4d} {int(pids[i]):6d} {probs[i]:7.3f}  {int(labels[i]):5d}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
