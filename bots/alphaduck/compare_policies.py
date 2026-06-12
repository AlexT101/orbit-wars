"""Compare policy distributions of multiple checkpoints on the same state.
Helps see which model produces useful priors vs which are stuck on noop.
"""
from __future__ import annotations
import argparse
import json
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bots" / "mine" / "target_predictor" / "train"))
sys.path.insert(0, str(ROOT / "bots" / "alphaduck" / "train"))
sys.path.insert(0, str(Path(__file__).parent))

import build_dataset as bd
import build_dataset_v0 as bd_v0
from set_net import apply_norm
from pair_net import PlanetTransformerPair


def load_ckpt(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = PlanetTransformerPair(
        ck["f_planet"], ck["f_global"],
        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
        n_layers=ck.get("n_layers", 2), ff=ck.get("ff", 128), dropout=0.0,
    ).eval()
    m.load_state_dict(ck["state_dict"], strict=False)
    return ck, m


def policy_for(ck, model, state, player):
    BD = bd_v0 if ck["f_planet"] == 46 else bd
    feats, globals_, pids = BD.extract_per_player(state, player, {})
    n = feats.shape[0]
    pf = np.zeros((1, BD.N_MAX, ck["f_planet"]), dtype=np.float32); pf[0, :n] = feats
    gl = globals_.reshape(1, -1).astype(np.float32)
    mk = np.zeros((1, BD.N_MAX), dtype=bool); mk[0, :n] = True
    raw_xy = np.zeros((1, BD.N_MAX, 7, 2), dtype=np.float32)
    raw_ships = np.zeros((1, BD.N_MAX), dtype=np.float32)
    raw_prod = np.zeros((1, BD.N_MAX), dtype=np.float32)
    for i, p in enumerate(state["planets"]):
        for j, h in enumerate((0, 1, 2, 5, 10, 20, 30)):
            try:
                pos = BD.planet_pos_at(state, p, h)
            except Exception:
                pos = None
            raw_xy[0, i, j] = pos if pos is not None else (p["x"], p["y"])
        raw_ships[0, i] = p["ships"]
        raw_prod[0, i] = p["prod"]
    pf_n, gl_n = apply_norm(pf, gl, ck["p_mean"], ck["p_std"], ck["g_mean"], ck["g_std"])
    has_raw = ck.get("f_planet", 0) > 46
    with torch.no_grad():
        kwargs = dict(return_value=True, return_noop=True)
        if has_raw:
            kwargs.update(raw_xy=torch.from_numpy(raw_xy),
                          raw_ships=torch.from_numpy(raw_ships),
                          raw_prod=torch.from_numpy(raw_prod))
        out = model(
            torch.from_numpy(pf_n),
            torch.from_numpy(gl_n),
            torch.from_numpy(mk),
            **kwargs,
        )
    pair_logits = out[0].numpy()[0][:n, :n]
    value = float(out[1].numpy()[0])
    noop_logits = out[2].numpy()[0][:n]
    np.fill_diagonal(pair_logits, -1e9)
    if ck.get("policy_loss_weight", 0.0) > 0:
        full = np.concatenate([noop_logits[:, None], pair_logits], axis=1)
        full = full - full.max(axis=1, keepdims=True)
        ex = np.exp(full)
        policy = ex / ex.sum(axis=1, keepdims=True)
    else:
        pair_probs = 1.0 / (1.0 + np.exp(-pair_logits))
        np.fill_diagonal(pair_probs, 0)
        noop_probs = 1.0 / (1.0 + np.exp(-noop_logits))
        raw = np.concatenate([noop_probs[:, None], pair_probs], axis=1)
        raw = np.clip(raw, 1e-6, None)
        policy = raw / raw.sum(axis=1, keepdims=True)
    return policy, value, pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--replay", default="/tmp/orbit_days/orbit-wars-episodes-2026-05-27.zip:77828182.json")
    ap.add_argument("--steps", nargs="+", type=int, default=[30, 60, 120])
    args = ap.parse_args()

    zp, name = args.replay.split(":", 1)
    with zipfile.ZipFile(zp) as zf:
        with zf.open(name) as f:
            g = json.load(io.BytesIO(f.read()))

    models = []
    for p in args.ckpts:
        ck, m = load_ckpt(p)
        models.append((Path(p).name, ck, m))
        print(f"loaded {p}: f_planet={ck['f_planet']} policy={ck.get('policy_loss_weight', 0)}")

    for step_t in args.steps:
        obs = g["steps"][step_t][0]["observation"]
        state = bd.parse_state(obs)
        print(f"\n=== step {step_t} ===")
        for name, ck, m in models:
            policy, v, pids = policy_for(ck, m, state, 0)
            print(f"  {name} value={v:+.3f}")
            my = [(p["id"], int(p["ships"])) for p in state["planets"] if p["owner"] == 0][:3]
            pid_to_idx = {pid: i for i, pid in enumerate(pids)}
            for pid, sh in my:
                i = pid_to_idx.get(pid)
                if i is None: continue
                noop_p = policy[i, 0]
                pair_pri = policy[i, 1:]
                tops = sorted([(pair_pri[j], pids[j]) for j in range(len(pids)) if j != i], reverse=True)[:3]
                print(f"    src={pid} ships={sh:3d}  noop={noop_p:.3f}  top: " +
                      "  ".join(f"t{pids[t[1] if isinstance(t[1], int) else int(t[1])]}={t[0]:.3f}" if False else f"t{int(t[1])}={t[0]:.3f}" for t in tops))


if __name__ == "__main__":
    main()
