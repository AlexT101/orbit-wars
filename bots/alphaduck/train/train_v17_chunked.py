"""v17 trainer that reads from multiple chunk NPZs (no concat).

For v17 at 2M+ rows, the full pair_feats can't fit in RAM, so we never
materialize a single NPZ. Each epoch iterates chunks in turn, loading one
chunk at a time.

Tiny per-row metadata (game_id, turn, player, value_label, pair_src/tgt,
noop_labels, planet_ids/mask) is pre-loaded into RAM for window-label
derivation and game-level train/val split. The big arrays (planet/fleet/
pair feats) are loaded per-chunk during training.
"""
from __future__ import annotations
import argparse, glob, math, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from pair_net_v17 import PairNetV17


# ---- the keys we load up front (small per-row data) and the big ones we
# stream per chunk
META_KEYS = ["game_ids", "turns", "players", "value_labels",
             "noop_labels", "pair_src", "pair_tgt",
             "planet_ids", "planet_mask", "fleet_mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", required=True,
                    help="glob pattern for chunk files, e.g. data/v17_full.npz.chunk_*.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--policy-loss-weight", type=float, default=1.0)
    ap.add_argument("--value-loss-weight", type=float, default=1.0)
    ap.add_argument("--noop-loss-weight", type=float, default=0.2)
    ap.add_argument("--value-target-discount", type=float, default=30.0)
    ap.add_argument("--label-window", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    chunks = sorted(glob.glob(args.data_glob))
    if not chunks:
        raise SystemExit(f"no chunks match {args.data_glob}")
    print(f"found {len(chunks)} chunks", flush=True)

    # ---- Phase 1: pre-load tiny per-row metadata across all chunks ----
    print("loading metadata across all chunks...", flush=True)
    t0 = time.time()
    meta = {k: [] for k in META_KEYS}
    chunk_row_ranges = []          # list of (start, end) row indices per chunk
    cursor = 0
    for ci, cp in enumerate(chunks):
        with np.load(cp, allow_pickle=True) as d:
            n_chunk = d["planet_feats"].shape[0]
            for k in META_KEYS:
                meta[k].append(d[k])
        chunk_row_ranges.append((cursor, cursor + n_chunk))
        cursor += n_chunk
    n_rows = cursor
    for k in META_KEYS:
        meta[k] = np.concatenate(meta[k], axis=0)
    print(f"  {n_rows} total rows; metadata took {time.time()-t0:.1f}s, "
          f"{sum(a.nbytes for a in meta.values()) // 1024 // 1024} MB", flush=True)

    # ---- normalization from FIRST chunk (cheap sample, fine for normalization) ----
    print("computing normalization from first chunk...", flush=True)
    with np.load(chunks[0], allow_pickle=True) as d0:
        pf0 = d0["planet_feats"]; pm0 = d0["planet_mask"]
        flat = pf0[pm0]
        p_mean = flat.mean(axis=0); p_std = flat.std(axis=0) + 1e-6
        gl0 = d0["globals"]
        g_mean = gl0.mean(axis=0); g_std = gl0.std(axis=0) + 1e-6
        ff0 = d0["fleet_feats"]; fm0 = d0["fleet_mask"]
        fflat = ff0[fm0] if fm0.any() else ff0.reshape(-1, ff0.shape[-1])
        f_mean = fflat.mean(axis=0); f_std = fflat.std(axis=0) + 1e-6
        paf0 = d0["pair_feats"]
        pa_mean = paf0.reshape(-1, paf0.shape[-1]).mean(axis=0)
        pa_std = paf0.reshape(-1, paf0.shape[-1]).std(axis=0) + 1e-6
        F_PLANET = d0["planet_feats"].shape[-1]
        F_FLEET = d0["fleet_feats"].shape[-1]
        F_GLOBAL = d0["globals"].shape[-1]
        F_PAIR = d0["pair_feats"].shape[-1]
        N_MAX = d0["planet_feats"].shape[1]
        F_MAX = d0["fleet_feats"].shape[1]

    # ---- value-target discount ----
    # discount <= 0 disables the per-turn tanh scaling — use raw ±1 outcome
    # targets at every turn. The model has to learn uncertainty from the input
    # state (less calibrated early-game targets but no late-game saturation bias).
    if args.value_target_discount > 0:
        value_labels = meta["value_labels"] * np.tanh(
            meta["turns"].astype(np.float32) / args.value_target_discount)
    else:
        value_labels = meta["value_labels"].astype(np.float32)

    # ---- train/val game-level split ----
    unique_games = np.unique(meta["game_ids"])
    rng = np.random.default_rng(0)
    rng.shuffle(unique_games)
    n_val_games = max(1, int(len(unique_games) * args.val_frac))
    val_set = set(unique_games[:n_val_games].tolist())
    val_mask = np.array([int(g) in val_set for g in meta["game_ids"]])
    train_mask = ~val_mask
    print(f"  train rows: {train_mask.sum()}  val rows: {val_mask.sum()}", flush=True)

    # ---- window labels ----
    K = meta["pair_src"].shape[1]
    if args.label_window > 0:
        print(f"  deriving window labels (W={args.label_window})...", flush=True)
        t0 = time.time()
        gpt_to_idx = {}
        for i in range(n_rows):
            key = (int(meta["game_ids"][i]), int(meta["players"][i]), int(meta["turns"][i]))
            gpt_to_idx[key] = i
        new_src = np.full_like(meta["pair_src"], -1)
        new_tgt = np.full_like(meta["pair_tgt"], -1)
        for i in range(n_rows):
            g = int(meta["game_ids"][i]); p = int(meta["players"][i]); t = int(meta["turns"][i])
            cur_pids = meta["planet_ids"][i]
            cur_mask = meta["planet_mask"][i]
            cur_pid_to_idx = {int(cur_pids[j]): j for j in range(cur_pids.shape[0]) if cur_mask[j]}
            seen = set(); slot = 0
            for dt in range(args.label_window + 1):
                fr = gpt_to_idx.get((g, p, t + dt))
                if fr is None: continue
                future_pids = meta["planet_ids"][fr]
                for k in range(K):
                    s_idx = int(meta["pair_src"][fr, k])
                    t_idx = int(meta["pair_tgt"][fr, k])
                    if s_idx < 0 or t_idx < 0: continue
                    src_pid = int(future_pids[s_idx]); tgt_pid = int(future_pids[t_idx])
                    if src_pid in seen: continue
                    s_now = cur_pid_to_idx.get(src_pid)
                    t_now = cur_pid_to_idx.get(tgt_pid)
                    if s_now is None or t_now is None: continue
                    if slot >= K: break
                    new_src[i, slot] = s_now; new_tgt[i, slot] = t_now
                    seen.add(src_pid); slot += 1
                if slot >= K: break
        meta["pair_src"] = new_src; meta["pair_tgt"] = new_tgt
        print(f"    done in {time.time()-t0:.1f}s", flush=True)

    # ---- model ----
    device = torch.device(args.device)
    model = PairNetV17(
        f_planet=F_PLANET, f_fleet=F_FLEET, f_global=F_GLOBAL, f_pair=F_PAIR,
        n_planet_max=N_MAX, n_fleet_max=F_MAX,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ff=args.ff, dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: PairNetV17 params={n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Map row idx → (chunk_idx, local_idx)
    def row_to_chunk(i):
        for ci, (s, e) in enumerate(chunk_row_ranges):
            if s <= i < e:
                return ci, i - s
        raise IndexError(i)

    G_MY_GROUND, G_EN_GROUND = 0, 1
    G_MY_PROD, G_EN_PROD = 2, 3
    G_MY_FLIGHT, G_EN_FLIGHT = 4, 5

    def load_chunk_subset(chunk_path, local_rows):
        """Open chunk once, slice + normalize the arrays for the given local rows."""
        local_rows = np.asarray(local_rows)
        with np.load(chunk_path, allow_pickle=True) as d:
            pf_raw = d["planet_feats"][local_rows]   # (B, N, F) raw, for is_mine
            paf_raw = d["pair_feats"][local_rows]    # (B, N, N, F_pair) raw, for is_unreachable
            arr = dict(
                pf=((pf_raw - p_mean) / p_std).astype(np.float32),
                # is_mine is planet_feats column 3 (PLANET_FEAT_NAMES[3]). Raw 0/1.
                is_mine=(pf_raw[..., 3] > 0.5).astype(bool),
                # is_unreachable is pair_feats column 3 (PAIR_FEAT_NAMES[3]). Raw 0/1.
                unreach=(paf_raw[..., 3] > 0.5).astype(bool),
                ff=((d["fleet_feats"][local_rows] - f_mean) / f_std).astype(np.float32),
                gl_raw=d["globals"][local_rows].astype(np.float32),
                pm=d["planet_mask"][local_rows].astype(bool),
                fm=d["fleet_mask"][local_rows].astype(bool),
                fti=d["fleet_tgt_idx"][local_rows].astype(np.int64),
                paf=((paf_raw - pa_mean) / pa_std).astype(np.float32),
            )
        arr["gl"] = ((arr["gl_raw"] - g_mean) / g_std).astype(np.float32)
        return arr, local_rows

    def step_chunk(ci, arr, sel, base_local_rows, train: bool):
        """Run forward + (optionally) backward on a batch slice of a pre-loaded chunk.

        arr: dict from load_chunk_subset.  sel: int array into arr's first axis (batch).
        base_local_rows: arr was loaded from these local rows; sel indexes into them too.
        """
        pf = arr["pf"][sel]
        ff = arr["ff"][sel]
        gl_raw = arr["gl_raw"][sel]
        gl = arr["gl"][sel]
        paf = arr["paf"][sel]
        pm = arr["pm"][sel]
        fm = arr["fm"][sel]
        fti = arr["fti"][sel]
        is_mine = arr["is_mine"][sel]
        unreach = arr["unreach"][sel]   # (B, N, N) bool
        global_rows = chunk_row_ranges[ci][0] + base_local_rows[sel]
        # Lopsided cutoff
        my_ships = gl_raw[:, G_MY_GROUND] + gl_raw[:, G_MY_FLIGHT]
        en_ships = gl_raw[:, G_EN_GROUND] + gl_raw[:, G_EN_FLIGHT]
        my_prod = gl_raw[:, G_MY_PROD]; en_prod = gl_raw[:, G_EN_PROD]
        big = np.maximum(my_ships, en_ships); sml = np.minimum(my_ships, en_ships) + 1e-6
        ships_lop = big > 2.0 * sml
        big_p = np.maximum(my_prod, en_prod); sml_p = np.minimum(my_prod, en_prod) + 1e-6
        prod_lop = big_p > 2.0 * sml_p
        contested = ~(ships_lop & prod_lop)
        contested_t = torch.from_numpy(contested.astype(np.float32)).to(device)
        # Labels: derive action_label[src] in [0, N] from existing pair_src/pair_tgt + noop_labels.
        # action_label = 0  → noop at src
        # action_label = 1+t → launch from src to target t
        noop_y = meta["noop_labels"][global_rows]   # 1.0 = noop, 0.0 = launched
        pair_src = meta["pair_src"][global_rows]
        pair_tgt = meta["pair_tgt"][global_rows]
        value_y = value_labels[global_rows]

        policy_logits, value_pred = model(
            torch.from_numpy(pf).float().to(device),
            torch.from_numpy(pm).bool().to(device),
            torch.from_numpy(ff).float().to(device),
            torch.from_numpy(fm).bool().to(device),
            torch.from_numpy(fti.astype(np.int64)).to(device),
            torch.from_numpy(gl).float().to(device),
            torch.from_numpy(paf).float().to(device),
        )
        B, N, K_plus_1 = policy_logits.shape  # K_plus_1 == N+1

        mask = torch.from_numpy(pm).bool().to(device)
        is_mine_t = torch.from_numpy(is_mine).bool().to(device)

        # Build action_label (B, N) and src_known (B, N)
        action_label = torch.zeros(B, N, dtype=torch.long, device=device)
        src_known = torch.zeros(B, N, dtype=torch.bool, device=device)
        ps = torch.from_numpy(pair_src).long().to(device)   # (B, K) src indices
        pt = torch.from_numpy(pair_tgt).long().to(device)   # (B, K) tgt indices
        valid_pair = (ps >= 0) & (pt >= 0)
        if valid_pair.any():
            B_idx_pair = torch.arange(B, device=device).unsqueeze(1).expand_as(ps)
            sel_b = B_idx_pair[valid_pair]; sel_src = ps[valid_pair]; sel_tgt = pt[valid_pair]
            action_label[sel_b, sel_src] = sel_tgt + 1   # 1+t → launch to tgt
            src_known[sel_b, sel_src] = True

        # Ambiguous: noop_y=0 (launched) but src has no known target in pair_src
        # (e.g. target despawned). Mask these out — neither noop nor launch label is correct.
        noop_t = torch.from_numpy(noop_y).bool().to(device)   # True = noop
        ambiguous = (~noop_t) & (~src_known)

        # Build masked policy logits: -inf where action is impossible.
        # col 0  = noop (always valid for owned src)
        # col 1+ = launch to target
        masked_logits = policy_logits.clone()
        # Diagonal (i, 1+i): can't launch i to itself
        diag = torch.arange(N, device=device)
        masked_logits[:, diag, 1 + diag] = float("-inf")
        # Non-planet targets (j >= n_real)
        masked_logits[:, :, 1:] = masked_logits[:, :, 1:].masked_fill(~mask.unsqueeze(1), float("-inf"))
        # Unreachable (src, tgt) pairs
        unreach_t = torch.from_numpy(unreach).bool().to(device)   # (B, N, N)
        masked_logits[:, :, 1:] = masked_logits[:, :, 1:].masked_fill(unreach_t, float("-inf"))

        log_probs = F.log_softmax(masked_logits, dim=-1)   # (B, N, N+1)
        B_idx_n = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        N_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        lp = log_probs[B_idx_n, N_idx, action_label]   # (B, N)
        lp = torch.nan_to_num(lp, nan=0.0, posinf=0.0, neginf=0.0)

        # Loss weight: owned + real planet + contested row + not ambiguous
        loss_weight = (is_mine_t.float()
                       * mask.float()
                       * contested_t.unsqueeze(1)
                       * (~ambiguous).float())
        pol_loss = (-lp * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)

        vy = torch.from_numpy(value_y).float().to(device)
        value_loss = F.mse_loss(value_pred, vy)

        loss = (args.policy_loss_weight * pol_loss
                + args.value_loss_weight * value_loss)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return dict(
            loss=float(loss.item()), pol=float(pol_loss.item()),
            val=float(value_loss.item()), noop=0.0,
            value_pred=value_pred.detach().cpu().numpy(),
            value_y=vy.detach().cpu().numpy(),
        )

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = tr_pol = tr_val = tr_noop = 0.0
        n_tr = 0
        vl_loss = vl_pol = vl_val = vl_noop = 0.0
        n_vl = 0
        v_mse_sum = 0.0; v_sign = 0; v_n = 0
        # iterate chunks (shuffled per epoch); load each chunk ONCE for both train and val
        chunk_order = list(range(len(chunks)))
        np.random.default_rng(ep).shuffle(chunk_order)
        for ci in chunk_order:
            cp = chunks[ci]
            s, e = chunk_row_ranges[ci]
            chunk_train_mask = train_mask[s:e]
            chunk_val_mask = val_mask[s:e]
            train_local = np.where(chunk_train_mask)[0]
            val_local = np.where(chunk_val_mask)[0]
            if len(train_local) == 0 and len(val_local) == 0:
                continue
            # Combine the two splits, load chunk once
            combined = np.sort(np.unique(np.concatenate([train_local, val_local])))
            arr, base_local = load_chunk_subset(cp, combined)
            base_local_set = {int(x): i for i, x in enumerate(base_local)}
            train_in_arr = np.array([base_local_set[int(x)] for x in train_local], dtype=np.int64)
            val_in_arr = np.array([base_local_set[int(x)] for x in val_local], dtype=np.int64)
            # Train batches
            model.train()
            if len(train_in_arr) > 0:
                np.random.default_rng(ep * 1000 + ci).shuffle(train_in_arr)
                for bs in range(0, len(train_in_arr), args.batch_size):
                    sel = train_in_arr[bs:bs + args.batch_size]
                    r = step_chunk(ci, arr, sel, base_local, train=True)
                    tr_loss += r["loss"]; tr_pol += r["pol"]; tr_val += r["val"]; tr_noop += r["noop"]
                    n_tr += 1
            # Val batches
            if len(val_in_arr) > 0:
                model.eval()
                with torch.no_grad():
                    for bs in range(0, len(val_in_arr), args.batch_size):
                        sel = val_in_arr[bs:bs + args.batch_size]
                        r = step_chunk(ci, arr, sel, base_local, train=False)
                        vl_loss += r["loss"]; vl_pol += r["pol"]; vl_val += r["val"]; vl_noop += r["noop"]
                        n_vl += 1
                        vp = r["value_pred"]; vy = r["value_y"]
                        v_mse_sum += float(np.mean((vp - vy) ** 2)) * len(vp)
                        v_sign += int(np.sum(np.sign(vp) == np.sign(vy)))
                        v_n += len(vp)
            del arr
        tr_loss /= max(1, n_tr); tr_pol /= max(1, n_tr)
        tr_val /= max(1, n_tr); tr_noop /= max(1, n_tr)
        vl_loss /= max(1, n_vl); vl_pol /= max(1, n_vl)
        vl_val /= max(1, n_vl); vl_noop /= max(1, n_vl)
        v_mse = v_mse_sum / max(1, v_n)
        v_sa = v_sign / max(1, v_n)
        dt = time.time() - t0
        print(f"epoch {ep:2d} | tr loss={tr_loss:.3f} pol={tr_pol:.3f} val={tr_val:.3f} noop={tr_noop:.3f} | "
              f"val loss={vl_loss:.3f} v_mse={v_mse:.3f} v_sa={v_sa:.3f}  ({dt:.0f}s)", flush=True)

        if vl_loss < best_val:
            best_val = vl_loss
            ck = {
                "state_dict": model.state_dict(),
                "f_planet": F_PLANET, "f_fleet": F_FLEET, "f_global": F_GLOBAL, "f_pair": F_PAIR,
                "n_planet_max": N_MAX, "n_fleet_max": F_MAX,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "ff": args.ff,
                "p_mean": p_mean, "p_std": p_std,
                "f_mean": f_mean, "f_std": f_std,
                "g_mean": g_mean, "g_std": g_std,
                "pa_mean": pa_mean, "pa_std": pa_std,
                "policy_loss_weight": args.policy_loss_weight,
                "policy_conditional": True,
                "value_target_discount": args.value_target_discount,
                "version": "v17",
            }
            torch.save(ck, args.out)
            print(f"    saved (best val_loss={vl_loss:.3f}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
