"""v17 trainer: planet+fleet tokens model with structured attention.

Recipe matches v15: per-source policy CE conditional on launch + sigmoid noop
BCE + value MSE with tanh discount + 180° mirror augmentation.
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from pair_net_v17 import PairNetV17


def apply_norm(feats, p_mean, p_std):
    return (feats - p_mean) / np.clip(p_std, 1e-6, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--policy-loss-weight", type=float, default=1.0)
    ap.add_argument("--value-loss-weight", type=float, default=1.0)
    ap.add_argument("--noop-loss-weight", type=float, default=0.2)
    ap.add_argument("--value-target-discount", type=float, default=30.0)
    ap.add_argument("--mirror-aug", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--label-window", type=int, default=2,
                    help="If > 0, derive policy CE labels from launches in [t, t+W]. "
                         "noop_labels untouched (per-turn truth). Window truncates for "
                         "comets/planets that despawn — only counts launches from sources "
                         "that exist at both turn t and the future turn.")
    args = ap.parse_args()

    print(f"loading {args.data}", flush=True)
    d = np.load(args.data, allow_pickle=True)
    n = d["planet_feats"].shape[0]
    print(f"  {n} rows", flush=True)

    # No step filter — keep all turns including endgame (lopsided cutoff handles
    # noisy demonstrations from blowout positions).
    idx_keep = np.arange(n)
    if args.max_rows and args.max_rows < n:
        rng = np.random.default_rng(42)
        idx_keep = rng.choice(idx_keep, args.max_rows, replace=False)
        print(f"  max-rows subsample: kept {len(idx_keep)}", flush=True)

    # Optional: derive policy CE labels from launches in window [t, t+W].
    # noop_labels untouched (still per-turn truth — keeps timing fidelity).
    pair_src_arr = d["pair_src"]
    pair_tgt_arr = d["pair_tgt"]
    if args.label_window > 0:
        W = args.label_window
        K = pair_src_arr.shape[1]                       # max actions per row
        N_MAX_pl = d["planet_ids"].shape[1]
        # Index: (game_id, player) -> sorted list of (turn, row_idx)
        from collections import defaultdict
        gp_to_rows = defaultdict(list)
        for i in range(n):
            gp_to_rows[(int(d["game_ids"][i]), int(d["players"][i]))].append(
                (int(d["turns"][i]), i))
        for k in gp_to_rows:
            gp_to_rows[k].sort()
        # Build (game, player, turn) -> row_idx lookup
        gpt_to_idx = {}
        for (g, p), lst in gp_to_rows.items():
            for tt, ri in lst:
                gpt_to_idx[(g, p, tt)] = ri
        # Per-row planet_id -> index lookup (for mapping launches across turns)
        # Done lazily inside the loop.
        planet_ids_all = d["planet_ids"]
        planet_mask_all = d["planet_mask"]

        new_src = np.full_like(pair_src_arr, -1)
        new_tgt = np.full_like(pair_tgt_arr, -1)
        t0 = time.time()
        any_window_actions = 0
        for i in range(n):
            g = int(d["game_ids"][i]); p = int(d["players"][i]); t = int(d["turns"][i])
            # Build pid -> idx for the current row (used to map future actions back)
            cur_pids = planet_ids_all[i]
            cur_mask = planet_mask_all[i]
            cur_pid_to_idx = {int(cur_pids[j]): j for j in range(N_MAX_pl) if cur_mask[j]}
            # Walk forward through window
            seen_src_pids = set()
            slot = 0
            for dt in range(W + 1):
                fr = gpt_to_idx.get((g, p, t + dt))
                if fr is None:
                    continue
                future_pids = planet_ids_all[fr]
                future_mask = planet_mask_all[fr]
                future_pid_to_idx = {int(future_pids[j]): j for j in range(N_MAX_pl) if future_mask[j]}
                for k in range(K):
                    s_idx = int(pair_src_arr[fr, k])
                    tgt_idx = int(pair_tgt_arr[fr, k])
                    if s_idx < 0 or tgt_idx < 0:
                        continue
                    src_pid = int(future_pids[s_idx])
                    tgt_pid = int(future_pids[tgt_idx])
                    if src_pid in seen_src_pids:
                        continue
                    # Source must exist at turn t
                    s_now = cur_pid_to_idx.get(src_pid)
                    if s_now is None:
                        continue
                    t_now = cur_pid_to_idx.get(tgt_pid)
                    if t_now is None:
                        continue
                    if slot >= K:
                        break
                    new_src[i, slot] = s_now
                    new_tgt[i, slot] = t_now
                    seen_src_pids.add(src_pid)
                    slot += 1
                if slot >= K:
                    break
            if slot > 0:
                any_window_actions += 1
        pair_src_arr = new_src
        pair_tgt_arr = new_tgt
        print(f"  label-window={W}: {any_window_actions}/{n} rows now have CE targets "
              f"(was {(d['pair_src'][:, 0] >= 0).sum()}); took {time.time()-t0:.1f}s", flush=True)

    # value-target discount
    value_labels = d["value_labels"][idx_keep].copy()
    turns = d["turns"][idx_keep].astype(np.float32)
    disc = np.tanh(turns / args.value_target_discount)
    value_labels = value_labels * disc
    print(f"  value discount applied; mean |label|={np.abs(value_labels).mean():.3f}", flush=True)

    # game-split train/val
    game_ids = d["game_ids"][idx_keep]
    unique_games = np.unique(game_ids)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_games)
    n_val_games = max(1, int(len(unique_games) * args.val_frac))
    val_set = set(unique_games[:n_val_games].tolist())
    val_mask = np.array([g in val_set for g in game_ids])
    train_mask = ~val_mask
    train_idx = idx_keep[train_mask]
    val_idx = idx_keep[val_mask]
    print(f"  train rows: {len(train_idx)}  val rows: {len(val_idx)}", flush=True)

    # normalize planet feats over training data
    pf_train = d["planet_feats"][train_idx]
    pm_train = d["planet_mask"][train_idx]
    flat = pf_train[pm_train]                            # (sum_real, F)
    p_mean = flat.mean(axis=0); p_std = flat.std(axis=0) + 1e-6
    # globals normalize too
    gl_train = d["globals"][train_idx]
    g_mean = gl_train.mean(axis=0); g_std = gl_train.std(axis=0) + 1e-6
    # fleet feats
    ff_train = d["fleet_feats"][train_idx]
    fm_train = d["fleet_mask"][train_idx]
    fflat = ff_train[fm_train] if fm_train.any() else ff_train.reshape(-1, ff_train.shape[-1])
    f_mean = fflat.mean(axis=0); f_std = fflat.std(axis=0) + 1e-6
    # pair feats normalize too (over real-real pairs)
    paf_train = d["pair_feats"][train_idx]
    pa_mean = paf_train.reshape(-1, paf_train.shape[-1]).mean(axis=0)
    pa_std = paf_train.reshape(-1, paf_train.shape[-1]).std(axis=0) + 1e-6

    F_PLANET = d["planet_feats"].shape[-1]
    F_FLEET = d["fleet_feats"].shape[-1]
    F_GLOBAL = d["globals"].shape[-1]
    F_PAIR = d["pair_feats"].shape[-1]
    N_MAX = d["planet_feats"].shape[1]
    F_MAX = d["fleet_feats"].shape[1]

    # mirror-aug indices: feats that flip sign under (x → 100-x, y → 100-y) — for v17
    # only dist_to_edge changes (no — it's symmetric); planet_feats has no x/y.
    # So mirror-aug here just needs to swap pair_feats (i,j)? Actually distance
    # is rotation-invariant; rate-of-change too. So pair_feats unchanged.
    # Conclusion: mirror-aug is a no-op for v17 feature set (everything is
    # rotation-/translation-invariant by design). Skip it.
    if args.mirror_aug:
        print("  mirror-aug: no-op for v17 (features are already invariant)", flush=True)

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

    def make_batches(indices, batch_size, shuffle=True):
        if shuffle:
            indices = indices.copy()
            np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            yield indices[i:i + batch_size]

    def to_device(arr):
        return torch.from_numpy(arr).to(device)

    # Indices into the raw globals array for the lopsided-cutoff rule.
    # global_feat_names: 0=sum_my_ships_ground, 1=sum_enemy_ships_ground,
    #                    2=sum_my_prod, 3=sum_enemy_prod,
    #                    4=sum_my_ships_flight, 5=sum_enemy_ships_flight, ...
    G_MY_GROUND, G_EN_GROUND = 0, 1
    G_MY_PROD,   G_EN_PROD   = 2, 3
    G_MY_FLIGHT, G_EN_FLIGHT = 4, 5

    def step(batch_idx, train: bool):
        gl_raw = d["globals"][batch_idx]
        pf = (d["planet_feats"][batch_idx] - p_mean) / p_std
        ff = (d["fleet_feats"][batch_idx] - f_mean) / f_std
        gl = (gl_raw - g_mean) / g_std
        paf = (d["pair_feats"][batch_idx] - pa_mean) / pa_std
        pm = d["planet_mask"][batch_idx]
        fm = d["fleet_mask"][batch_idx]
        fti = d["fleet_tgt_idx"][batch_idx]
        noop_y = d["noop_labels"][batch_idx]
        pair_src = pair_src_arr[batch_idx]
        pair_tgt = pair_tgt_arr[batch_idx]
        value_y = d["value_labels"][batch_idx] * np.tanh(d["turns"][batch_idx].astype(np.float32) / args.value_target_discount)

        # Lopsided cutoff: skip policy/noop loss for samples where one side
        # has > 2x total ships AND > 3x production. Value loss unaffected.
        my_ships = gl_raw[:, G_MY_GROUND] + gl_raw[:, G_MY_FLIGHT]
        en_ships = gl_raw[:, G_EN_GROUND] + gl_raw[:, G_EN_FLIGHT]
        my_prod  = gl_raw[:, G_MY_PROD]
        en_prod  = gl_raw[:, G_EN_PROD]
        big = np.maximum(my_ships, en_ships)
        sml = np.minimum(my_ships, en_ships) + 1e-6
        ships_lopsided = big > 2.0 * sml
        big_p = np.maximum(my_prod, en_prod)
        sml_p = np.minimum(my_prod, en_prod) + 1e-6
        prod_lopsided  = big_p > 2.0 * sml_p
        contested = ~(ships_lopsided & prod_lopsided)        # (B,) bool — True = train policy/noop
        contested_t = torch.from_numpy(contested.astype(np.float32)).to(device)

        pair_logits, value_pred, noop_logits = model(
            to_device(pf).float(),
            to_device(pm).bool(),
            to_device(ff).float(),
            to_device(fm).bool(),
            to_device(fti).long(),
            to_device(gl).float(),
            to_device(paf).float(),
        )
        B, N, _ = pair_logits.shape

        # ---- noop BCE
        noop_logits_flat = noop_logits  # (B, N)
        mask = to_device(pm).bool()
        noop_target = to_device(noop_y).float()
        noop_loss = F.binary_cross_entropy_with_logits(
            noop_logits_flat, noop_target, reduction="none"
        )
        # zero out contributions from lopsided samples
        weight = mask.float() * contested_t.unsqueeze(1)        # (B, N)
        noop_loss = (noop_loss * weight).sum() / weight.sum().clamp_min(1.0)

        # ---- policy CE conditional on launch
        # pair_src / pair_tgt: (B, 8) with -1 pad. For each valid (src, tgt),
        # softmax(pair_logits[b, src, :]) and CE against target index tgt.
        ps = to_device(pair_src).long()  # (B, 8)
        pt = to_device(pair_tgt).long()
        valid = (ps >= 0) & (pt >= 0)
        if valid.any():
            # mask diag to -inf so softmax never picks self
            pair_masked = pair_logits.clone()
            diag = torch.arange(N, device=device)
            pair_masked[:, diag, diag] = float("-inf")
            # Also mask invalid target columns
            pm_t = to_device(pm).bool()
            pair_masked = pair_masked.masked_fill(~pm_t.unsqueeze(1), float("-inf"))
            # gather logits at (src, :) per sample (just use full B,N,N)
            # CE per sample: log_softmax(pair_masked[b, src, :])[tgt]
            log_probs = F.log_softmax(pair_masked, dim=-1)   # (B, N, N)
            # gather: for each valid (b, k), -log_probs[b, src[b,k], tgt[b,k]]
            B_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ps)
            ps_safe = ps.clamp_min(0)
            pt_safe = pt.clamp_min(0)
            lp = log_probs[B_idx, ps_safe, pt_safe]   # (B, 8)
            # Replace non-finite (from masked diag lookups under invalid)
            # with a safe finite value before zeroing invalid contributions.
            lp = torch.nan_to_num(lp, nan=0.0, posinf=0.0, neginf=0.0)
            # zero out lopsided-position rows
            pol_weight = valid.float() * contested_t.unsqueeze(1)   # (B, 8)
            ce = -lp * pol_weight
            pol_loss = ce.sum() / pol_weight.sum().clamp_min(1.0)
        else:
            pol_loss = torch.zeros((), device=device)

        # ---- value MSE
        vy = to_device(value_y).float()
        value_loss = F.mse_loss(value_pred, vy)

        loss = (args.policy_loss_weight * pol_loss
                + args.value_loss_weight * value_loss
                + args.noop_loss_weight * noop_loss)

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        return dict(
            loss=float(loss.item()), pol=float(pol_loss.item()),
            val=float(value_loss.item()), noop=float(noop_loss.item()),
            value_pred=value_pred.detach().cpu().numpy(),
            value_y=vy.detach().cpu().numpy(),
            pair_logits=pair_logits.detach(),
            valid=valid.detach(), ps=ps.detach(), pt=pt.detach(),
        )

    best_pol_top3 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = tr_pol = tr_val = tr_noop = 0.0
        n_batches = 0
        for batch_idx in make_batches(train_idx, args.batch_size):
            r = step(batch_idx, train=True)
            tr_loss += r["loss"]; tr_pol += r["pol"]; tr_val += r["val"]; tr_noop += r["noop"]
            n_batches += 1
        tr_loss /= max(1, n_batches); tr_pol /= max(1, n_batches)
        tr_val /= max(1, n_batches); tr_noop /= max(1, n_batches)

        # val
        model.eval()
        with torch.no_grad():
            vl_loss = vl_pol = vl_val = vl_noop = 0.0
            top1_hit = top3_hit = top5_hit = total_valid = 0
            v_mse_sum = 0.0; v_sign = 0; v_n = 0
            for batch_idx in make_batches(val_idx, args.batch_size, shuffle=False):
                r = step(batch_idx, train=False)
                vl_loss += r["loss"]; vl_pol += r["pol"]; vl_val += r["val"]; vl_noop += r["noop"]
                # policy top-k metric
                pair_logits = r["pair_logits"]
                ps_v = r["ps"]; pt_v = r["pt"]; valid_v = r["valid"]
                B_, N_, _ = pair_logits.shape
                if valid_v.any():
                    pair_masked = pair_logits.clone()
                    diag = torch.arange(N_, device=pair_logits.device)
                    pair_masked[:, diag, diag] = float("-inf")
                    for b in range(B_):
                        for kk in range(ps_v.shape[1]):
                            if not valid_v[b, kk]: continue
                            src_i = int(ps_v[b, kk].item())
                            tgt_i = int(pt_v[b, kk].item())
                            scores = pair_masked[b, src_i, :].cpu().numpy()
                            order = np.argsort(-scores)
                            r1 = int(order[0] == tgt_i)
                            r3 = int(tgt_i in order[:3])
                            r5 = int(tgt_i in order[:5])
                            top1_hit += r1; top3_hit += r3; top5_hit += r5; total_valid += 1
                # value sign acc
                vp = r["value_pred"]; vy = r["value_y"]
                v_mse_sum += float(np.mean((vp - vy) ** 2)) * len(vp)
                v_sign += int(np.sum(np.sign(vp) == np.sign(vy)))
                v_n += len(vp)
            n_vb = max(1, len(val_idx) // args.batch_size)
            vl_loss /= n_vb; vl_pol /= n_vb; vl_val /= n_vb; vl_noop /= n_vb
            top1 = top1_hit / max(1, total_valid)
            top3 = top3_hit / max(1, total_valid)
            top5 = top5_hit / max(1, total_valid)
            v_mse = v_mse_sum / max(1, v_n)
            v_sa = v_sign / max(1, v_n)
        dt = time.time() - t0
        print(f"epoch {ep:2d} | tr loss={tr_loss:.3f} pol={tr_pol:.3f} val={tr_val:.3f} noop={tr_noop:.3f} | "
              f"val loss={vl_loss:.3f} top1={top1:.3f} top3={top3:.3f} top5={top5:.3f} "
              f"v_mse={v_mse:.3f} v_sa={v_sa:.3f}  ({dt:.0f}s)", flush=True)

        if top3 > best_pol_top3:
            best_pol_top3 = top3
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
            print(f"    saved (best top3={top3:.3f}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
