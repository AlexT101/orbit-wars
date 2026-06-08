# Aphrodite training handoff

Concise state of the value-net training work. Read before changing the pipeline
so we don't regress (e.g. reintroducing the decisiveness cut, or using the
unregularized rating).

## Live weights (what `main.py` loads)

`main.py` `_WEIGHTS_2P_NAME` / `_WEIGHTS_4P_NAME`:
- **`xgb_2p.json`** — quality(**rating**, Bradley-Terry) + 14-col ablation, **no** decisiveness, **no** drop. 8-day data (~6.5M rows).
- **`xgb_4p.json`** — quality(**rating**) + 11-col ablation, **no** decisiveness, **no** drop. 3-day 4p data (~832k rows).

These are the promoted rating models (the old `*_rating*` experiment files were deleted after they gauntletted well). Note: the data is still **win-rate-GATED** (the per-day strong-topn gate baked into `combined.npz`); only the soft *weight* uses the rating. The Elo two-phase work below replaces the **gate** itself (in progress).

CAVEAT: if these were trained before the BT Gamma-prior fix, their soft weight used the **un-regularized** rating (minor — mis-weights only a few low-game players' rows, not a gate). Regularized BT is now the default; a clean retrain is optional.

(`main.py` may currently point 2p at a `*_6_06*` POC for gauntletting — reset to `xgb_2p.json` to compare against live.)

## Ablation (`--zero-cols`) — DIFFERS by format
65-dim `summary_v2` layout: me_cur[0:9] opp_cur[9:18] me_ext[18:26] opp_ext[26:34] neutral[34:41] relational[41:65]. Key cols: 41=step, 48=ship_share, 49=production_share; 61=my_strength_rank, 62=leader_strength_ratio, 63=opponent_strength_spread, 64=n_alive.
- **2p:** `4,8,13,17,21,25,29,33,37,40,41,61,63,64` (14 cols). Drops step + 4p-standing 61/63/64 (near-constant in 2p). 62 kept (it degrades to a valid 1v1 strength ratio).
- **4p:** `4,8,13,17,21,25,29,33,37,40,41` (11 cols). **KEEP 61/62/63/64** — the 4p-standing features carry real signal with 4 players.

## Quality weighting: winrate vs Elo (Bradley-Terry)
`train_xgb.py --quality-metric {winrate,rating}` (default `winrate`).
- `rating` = opponent-adjusted Bradley-Terry, fit from `game_names`+`game_rewards`. Better than win rate (Elo-matched ladder compresses win rates toward 0.5; rating un-compresses).
- **BT is REGULARIZED** (Gamma(K,K) prior, `prior_strength` default **3.0**). REQUIRED: without it a low-game undefeated player (e.g. 5g/4-0) diverges to rank #1. Don't remove the prior.
- `bradley_terry_ratings()` / `per_row_strength_from_rating()` live in `train_xgb.py`.

## DO NOT use decisiveness/drop on 4p (regression risk)
`--decisiveness-weight` and `--drop-decided` use `lead = max(adv, 1-adv)` assuming 0.5 = even (true for 2p). In 4p "even" is 0.25, so ordinary positions read as decided → the drop nuked **38.6%** of legit 4p data. **2p-only** until the lead/claimed metrics are made player-count-aware. Neither is used by the live `xgb_2p.json`/`xgb_4p.json`.

## Two-phase Elo-gated extraction (NEW — the main build)
Per-day BT (each day independent → adding a day never forces re-extracting old days).
- **Phase 1 `elo_topn.py`** — scans a day's outcomes ONLY (names+rewards, no Rust extraction), fits regularized per-day BT, writes top-N keep-list JSON. Fast.
- **Phase 2 `build_from_zip.py --keep-players <json>`** — extracts `summary_v2` ONLY for kept players' rows (gate applied during extraction, so skipped rows cost nothing). Writes gated npz with `game_names`/`game_rewards` but **NO `win_rate`** → downstream must use `--quality-metric rating`.
- `combine_npz.py` now carries `game_names`/`game_rewards` (gid-aligned) for the rating fit.

Per-day commands (paths relative to `train/`):
```
python elo_topn.py --zip ../../../../ladder_replays/replays_<DAY>.zip \
  --players 2 --top-n 15 --min-games 5 --out data/2p/_ladder_work/topn_<DAY>.json
python build_from_zip.py --zip ../../../../ladder_replays/replays_<DAY>.zip \
  --out data/2p/_ladder_work/gated_replays_<DAY>.npz --players 2 \
  --keep-players data/2p/_ladder_work/topn_<DAY>.json
```
Combine inputs **oldest→newest** (combine assigns `source` = input order, which recency reads as the day rank). Train (matches live recipe minus the winrate→rating swap, plus recency):
```
python combine_npz.py --out <combined>.npz gated_replays_<oldest>.npz ... gated_replays_<newest>.npz
python train_xgb.py --data <combined>.npz --no-filter \
  --quality-weight --quality-metric rating --quality-floor 0.25 \
  --recency-halflife 6 \
  --zero-cols 4,8,13,17,21,25,29,33,37,40,41,61,63,64 \
  --rounds 2000 --early-stopping 50 --model-out weights/<name>.json
```
- **`--recency-halflife 6`** = exponential day-decay on `source`: newest day weight 1.0, each 6 days older halves it. Over a 7-day window (source 0..6) the oldest day = 0.500, newest = 1.000 (gentle — week-old data still counts at half). Multiplies into the rating quality weight.
- **Ablation differs by format** (see §Ablation): the `--zero-cols` above is **2p**. For **4p** use `4,8,13,17,21,25,29,33,37,40,41` (11 cols, KEEP 61/62/63/64).
Extraction ≈ 9-14 min/day (gated to top-15). `build_ladder.py --gate elo-topn` is **not yet wired** — the two commands above are run manually per day.

## Weights in progress (Elo-gated, REGULARIZED BT, recency hl=6)
- **`xgb_2p_5_31_6_06.json`** — 2p, 7 days `5_31…6_06` either-top-15 gate (6.45M rows). 2p 14-col ablation. val sign-acc 89.9%.
- **`xgb_4p_6_01_6_07.json`** — 4p, 7 days `6_01…6_07` either-top-15 gate (2.60M rows). 4p 11-col ablation (keeps 61-64). val sign-acc 89.0%.
- Both pending gauntlet vs live `xgb_2p.json` / `xgb_4p.json` (val sign-acc not comparable across day-sets — judge by gauntlet).
- **NOT in use:** `filter_both_topn.py` + `xgb_2p_5_31_6_06_both10.json` — an experiment restricting 2p to games where BOTH players are top-10 that day (post-filters the combined npz, no re-extraction; compacts gids so `game_names` stays gid-aligned). Kept for reference only; not gauntletted, not deployed.

## Gotchas
- val sign-acc is NOT comparable across drop/no-drop or different day-sets (val composition changes) — judge models by **gauntlet**, not val acc.
- Elo-gated npz lacks `win_rate`; `--quality-metric winrate` would silently no-op.
- Gauntlet by swapping `APHRODITE_VALUE_NET_PATH` (or `main.py` names). To deploy, rename winner onto `xgb_2p.json`/`xgb_4p.json`.
- Rust rebuild (`cargo build --release --bin aphrodite`) only needed for `.rs` changes, not weight swaps.
