"""Compute per-player win rate across the cached Kaggle zips, then bucket the
val games by player skill and report model accuracy per bucket.

This identifies whether (a) the model is differentially harder/easier on
top-player games, and (b) what's distinctive about top-player launches
(distribution over target owner-type, mean fleet size, mean planets per turn).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from set_net import build_model, game_level_split, apply_norm


def scan_player_winrates(zips: list[Path]) -> dict[str, tuple[float, int]]:
    """Returns {player_name: (win_rate, n_games)} for 2p games in the zips."""
    wins = defaultdict(int); losses = defaultdict(int)
    n_games = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                with zf.open(name) as f:
                    try:
                        d = json.load(io.BytesIO(f.read()))
                    except Exception:
                        continue
                rewards = d.get("rewards") or []
                if len(rewards) != 2:
                    continue
                info = d.get("info") or {}
                agents = info.get("Agents") or []
                if len(agents) != 2:
                    continue
                names = [a.get("Name") or a.get("submissionId") or a.get("name") or "" for a in agents]
                if not all(names):
                    continue
                r0 = rewards[0] if rewards[0] is not None else 0
                r1 = rewards[1] if rewards[1] is not None else 0
                n_games += 1
                if r0 > r1:
                    wins[names[0]] += 1; losses[names[1]] += 1
                elif r1 > r0:
                    wins[names[1]] += 1; losses[names[0]] += 1
    out = {}
    for p in set(wins) | set(losses):
        n = wins[p] + losses[p]
        out[p] = (wins[p] / max(n, 1), n)
    print(f"  scanned {n_games} 2p games, {len(out)} unique players")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", type=Path, default=Path("/tmp/orbit_days"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).resolve().parent / "weights" / "transformer_s200_owsmooth.pt")
    ap.add_argument("--data", type=Path,
                    default=Path(__file__).resolve().parent / "data" / "targets_2k_v2.npz")
    ap.add_argument("--top-frac", type=float, default=0.20,
                    help="players with win-rate above this percentile count as top")
    ap.add_argument("--min-games", type=int, default=5)
    args = ap.parse_args()

    zips = sorted(args.zip_dir.glob("*.zip"))
    print(f"scanning {len(zips)} zips for player win rates ...")
    pwr = scan_player_winrates(zips)
    valid = {p: wr for p, (wr, n) in pwr.items() if n >= args.min_games}
    print(f"  {len(valid)} players with ≥{args.min_games} games")
    if not valid:
        print("nothing to do"); return 1
    wrs = sorted(valid.values(), reverse=True)
    top_thresh = wrs[max(1, int(len(wrs) * args.top_frac)) - 1]
    print(f"  top-{int(args.top_frac*100)}% win-rate threshold: {top_thresh:.3f}")
    top_set = {p for p, wr in valid.items() if wr >= top_thresh}

    # Build game_id -> (p0_name, p1_name) map
    print("indexing game -> player mapping ...")
    g2names = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                with zf.open(name) as f:
                    try:
                        d = json.load(io.BytesIO(f.read()))
                    except Exception:
                        continue
                if len(d.get("rewards") or []) != 2:
                    continue
                agents = (d.get("info") or {}).get("Agents") or []
                if len(agents) != 2:
                    continue
                gid_str = Path(name).stem
                try:
                    gid = int(gid_str)
                except ValueError:
                    gid = abs(hash(gid_str)) % (2 ** 31 - 1)
                g2names[gid] = tuple(a.get("Name") or a.get("submissionId") or a.get("name") or "" for a in agents)

    # Load data + checkpoint, run model on val split
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
    _, val_mask = game_level_split(meta, seed=42, val_frac=0.12)
    pf = pf[val_mask]; gl = gl[val_mask]; mk = mk[val_mask]; lb = lb[val_mask]; meta = meta[val_mask]
    pfn, gln = apply_norm(pf, gl, ck["p_mean"], ck["p_std"], ck["g_mean"], ck["g_std"])

    print("inference ...")
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(pfn), 1024):
            logits_all.append(model(torch.from_numpy(pfn[i:i+1024]),
                                    torch.from_numpy(gln[i:i+1024]),
                                    torch.from_numpy(mk[i:i+1024])).numpy())
    logits = np.concatenate(logits_all, 0)
    logits_masked = np.where(mk, logits, -1e9)

    # bucket each row by "row's player skill class"
    n_pos = (lb * mk).sum(axis=1)
    has_pos = n_pos > 0
    order_full = np.argsort(-logits_masked, axis=1)

    def recall_at_K(rows_mask, K):
        if not rows_mask.any(): return float("nan")
        top = order_full[rows_mask, :K]
        hits = np.take_along_axis(lb[rows_mask], top, axis=1).sum(axis=1)
        return float(hits.sum() / max(n_pos[rows_mask].sum(), 1))

    # classify each row
    row_skill = np.zeros(len(meta), dtype=np.int8)  # 0=unknown, 1=top, 2=non-top
    for i, (gid, _, player, _) in enumerate(meta):
        names = g2names.get(int(gid))
        if names is None:
            continue
        pname = names[int(player)]
        if not pname:
            continue
        row_skill[i] = 1 if pname in top_set else 2

    for cls, name in [(1, "TOP"), (2, "NON-TOP"), (0, "unknown")]:
        sel = (row_skill == cls) & has_pos
        n = int(sel.sum())
        if n < 20:
            print(f"  {name:8s}: {n} rows w/ pos (skipped)"); continue
        print(f"  {name:8s}: {n} rows w/ pos  r@5={recall_at_K(sel,5):.4f} "
              f"r@8={recall_at_K(sel,8):.4f}  r@10={recall_at_K(sel,10):.4f}  "
              f"mean pos/row={n_pos[sel].mean():.2f}")

    # Distinctive launch patterns of top players: where do top vs non-top
    # players send fleets? Count positives by target-owner type.
    print("\nlaunch target distribution by player class:")
    print(f"  {'class':10s} {'#launches':10s} {'%mine':>7s} {'%neut':>7s} {'%enemy':>7s} {'%comet':>7s}")
    # need owner one-hot features at the positive slots
    is_mine_idx = list(ck["feat_names"]).index("is_mine")
    is_neut_idx = list(ck["feat_names"]).index("is_neutral")
    is_enem_idx = list(ck["feat_names"]).index("is_enemy")
    is_comet_idx = list(ck["feat_names"]).index("is_comet")
    for cls, name in [(1, "TOP"), (2, "NON-TOP")]:
        sel = (row_skill == cls)
        if not sel.any(): continue
        pos_mask = (lb[sel] > 0.5)  # (R, N_max)
        # planet feats at positive slots (un-normalized? we have normalized)
        # use the original `pf` (un-normalized) which we kept
        pos_rows_planets = pf[sel][pos_mask]
        n = len(pos_rows_planets)
        if n == 0: continue
        pct_mine = pos_rows_planets[:, is_mine_idx].mean() * 100
        pct_neut = pos_rows_planets[:, is_neut_idx].mean() * 100
        pct_enem = pos_rows_planets[:, is_enem_idx].mean() * 100
        pct_com = pos_rows_planets[:, is_comet_idx].mean() * 100
        print(f"  {name:10s} {n:10d} {pct_mine:7.2f} {pct_neut:7.2f} {pct_enem:7.2f} {pct_com:7.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
