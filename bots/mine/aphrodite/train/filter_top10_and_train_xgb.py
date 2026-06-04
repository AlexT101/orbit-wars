"""Take a `build_from_zip.py` output NPZ, apply the top-10 player filter,
and train XGBoost with the same params the deployed `xgb_top10_d6.json`
used (binary:logistic, max_depth=6, lr=0.08, n_est=600, early stop=40).
Saves the booster as `weights/<out_stem>.json`.

This collapses filtering and XGBoost training into one script for the
"rebuild + retrain" workflow.

Usage:
    python filter_top10_and_train_xgb.py \\
        --input data/2p/old_top10.npz \\
        --top10-out data/2p/old_top10_rebuilt.npz \\
        --model-out weights/xgb_2p_old_top10.json

    python filter_top10_and_train_xgb.py \\
        --data data/4p/train_4p_mixed.npz \\
        --no-filter \\
        --model-out weights/xgb_4p.json
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Player names can contain emoji / non-Latin characters; force UTF-8 so
# printing the ranking never dies on a cp1252 console or redirected pipe.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def compute_rates(d, min_games=5):
    """Per-player win rate over the dataset's games.

    Returns (rates, pw, pg) where rates maps player name -> win rate for
    players with at least `min_games` games. A win counts only when the
    game has a single winner (ties credit nobody).
    """
    meta = d["meta"]
    y = d["labels"]
    game_names = d["game_names"]
    n_games = game_names.shape[0]
    n_players = game_names.shape[1]
    game_rewards = d["game_rewards"] if "game_rewards" in d.files else None

    pg = defaultdict(int)
    pw = defaultdict(int)
    for gid in range(n_games):
        for slot in range(n_players):
            pg[str(game_names[gid, slot])] += 1

    rewards_by_gid: dict[int, list[float]] = {}
    if game_rewards is not None:
        for gid in range(n_games):
            rewards_by_gid[gid] = [float(x) for x in game_rewards[gid]]
    else:
        for i in range(meta.shape[0]):
            gid = int(meta[i, 0])
            slot = int(meta[i, 2])
            if gid not in rewards_by_gid:
                rewards_by_gid[gid] = [0.0] * n_players
            rewards_by_gid[gid][slot] = float(y[i])

    for gid, rewards in rewards_by_gid.items():
        if gid >= n_games or not rewards:
            continue
        best = max(rewards)
        winners = [slot for slot, reward in enumerate(rewards) if reward == best]
        if len(winners) == 1:
            pw[str(game_names[gid, winners[0]])] += 1

    rates = {pl: pw[pl] / pg[pl] for pl in pg if pg[pl] >= min_games}
    return rates, pw, pg


def compute_top_n(d, n_top=10, min_games=5):
    """Compute per-player win rates and return the set of top-N player names."""
    rates, pw, pg = compute_rates(d, min_games)
    sorted_rates = sorted(rates.items(), key=lambda kv: -kv[1])
    print(f"players with >= {min_games} games: {len(rates)}")
    print(f"top 15:")
    for pl, r in sorted_rates[:15]:
        print(f"  {r:.3f} ({pw[pl]:>4}/{pg[pl]:<4})  {pl[:60]}")
    return {pl for pl, _ in sorted_rates[:n_top]}


def strong_set_for(d, gate: str, top_n: int, min_games: int):
    """Resolve the set of 'strong' player names for a per-perspective gate.

    gate == "strong-topn": the top-`top_n` players by win rate.
    gate == "strong-median": every player strictly above the median win rate.
    Prints the relevant slice for visibility. Returns (strong_set, rates).
    """
    rates, pw, pg = compute_rates(d, min_games)
    if not rates:
        raise SystemExit(f"no players reached --min-games={min_games}; cannot gate")
    ordered = sorted(rates.items(), key=lambda kv: -kv[1])
    if gate == "strong-median":
        sr = sorted(rates.values())
        median = sr[len(sr) // 2]
        strong = {pl for pl, r in rates.items() if r > median}
        print(f"  median win rate = {median:.3f} over {len(rates)} qualified players; "
              f"{len(strong)} above it")
    else:  # strong-topn
        strong = {pl for pl, _ in ordered[:top_n]}
        print(f"  top-{top_n} of {len(rates)} qualified players (>= {min_games} games):")
        for pl, r in ordered[:top_n]:
            print(f"    {r:.3f} ({pw[pl]:>4}/{pg[pl]:<4})  {pl[:60]}")
    return strong, rates


def filter_strong_side(d, out_path: Path, strong_set, rates):
    """Per-perspective gate: keep a ROW iff that row's player is in
    `strong_set`. Unlike the both-players top-N gate this keeps the strong
    side of a strong-vs-weak game and drops only the weak side, so a
    1st-vs-30th game still contributes the strong player's positions.

    Records each kept row's player win rate as `win_rate` so the trainer can
    soft-weight by strength within the kept set.
    """
    meta = d["meta"]
    game_names = d["game_names"]
    n_games = game_names.shape[0]
    n_players = game_names.shape[1]

    # Per-(game, slot) win-rate grid + membership; unqualified players -> -1.
    rate_grid = np.full((n_games, n_players), -1.0, dtype=np.float64)
    in_strong = np.zeros((n_games, n_players), dtype=bool)
    for g in range(n_games):
        for slot in range(n_players):
            nm = str(game_names[g, slot])
            rate_grid[g, slot] = rates.get(nm, -1.0)
            in_strong[g, slot] = nm in strong_set

    gid = meta[:, 0].astype(np.int64)
    slot = meta[:, 2].astype(np.int64)
    sub = in_strong[gid, slot]

    Xs = d["summary_v2"][sub]
    ys = d["labels"][sub]
    ms = meta[sub]
    wr = rate_grid[gid, slot][sub].astype(np.float32)
    n_kept = int(np.unique(ms[:, 0]).size)
    print(f"  strong-side gate kept {len(Xs):,} / {meta.shape[0]:,} rows "
          f"({100 * sub.mean():.1f}%) across {n_kept} games")
    np.savez_compressed(
        out_path,
        summary_v2=Xs.astype(np.float32),
        labels=ys.astype(np.float32),
        meta=ms.astype(np.int32),
        win_rate=wr,
        game_names=game_names,
        **({"game_files": d["game_files"]} if "game_files" in d.files else {}),
        **({"game_rewards": d["game_rewards"]} if "game_rewards" in d.files else {}),
    )
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return Xs, ys, ms, sub, wr


def filter_top_n(d, top_set, out_path: Path):
    meta = d["meta"]
    game_names = d["game_names"]
    n_games = game_names.shape[0]
    n_players = game_names.shape[1]
    game_in_top = np.array([
        all(str(game_names[g, slot]) in top_set for slot in range(n_players))
        for g in range(n_games)
    ])
    sub = game_in_top[meta[:, 0].astype(np.int64)]
    Xs = d["summary_v2"][sub]
    ys = d["labels"][sub]
    ms = meta[sub]
    n_kept = int(np.unique(ms[:, 0]).size)
    print(f"  top-N filter kept {n_kept} games / {len(Xs):,} rows / "
          f"{len(Xs) * 196 / 1e6:.1f} MB raw")
    np.savez_compressed(
        out_path,
        summary_v2=Xs.astype(np.float32),
        labels=ys.astype(np.float32),
        meta=ms.astype(np.int32),
        game_names=game_names,
        **({"game_files": d["game_files"]} if "game_files" in d.files else {}),
        **({"game_rewards": d["game_rewards"]} if "game_rewards" in d.files else {}),
    )
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return Xs, ys, ms, sub


def game_level_split_mask(meta, frac=0.12, seed=42):
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games])


def recency_weights(source, halflife_days: float):
    """Per-row sample weight from the combine_npz `source` column.

    `source` is the input-file index assigned by combine_npz. When the
    per-day NPZs are combined oldest->newest, `source` is the day rank, so
    one step == one day. Newest rows get weight 1.0; each `halflife_days`
    older halves the weight. Returns None if weighting is disabled or no
    source column is present.
    """
    if halflife_days is None or halflife_days <= 0:
        return None
    if source is None:
        print("  [recency] WARN no `source` column (combine inputs oldest->newest "
              "to enable recency weighting); using uniform weights")
        return None
    source = source.astype(np.float64)
    newest = source.max()
    age = newest - source  # in source-steps == days for consecutive daily zips
    w = 0.5 ** (age / float(halflife_days))
    uniq = np.unique(source).astype(int)
    print(f"  [recency] halflife={halflife_days}d  sources={uniq.min()}..{uniq.max()}  "
          f"weight newest=1.000 oldest={float(0.5 ** ((newest - source.min()) / halflife_days)):.3f}")
    return w.astype(np.float32)


def quality_weights(win_rate, floor: float, enabled: bool):
    """Per-row sample weight from player strength, ranked WITHIN the kept set.

    Maps each row's player win rate to its percentile among kept rows, then
    onto [floor, 1.0]: the strongest kept players weigh ~1.0, the weakest
    (just above the median gate) weigh `floor`. Rank-based so it is robust
    to per-day median differences and absolute win-rate scale. Returns None
    if disabled or no `win_rate` column is present.
    """
    if not enabled:
        return None
    if win_rate is None:
        print("  [quality] WARN no `win_rate` column (gate with --gate strong-median "
              "to enable strength weighting); using uniform quality")
        return None
    wr = win_rate.astype(np.float64)
    # Tie-aware percentile: rows with equal win rate (== equal strength) get
    # the same weight. `average` ranking shares the rank across ties.
    from scipy.stats import rankdata
    pct = rankdata(wr, method="average") / wr.shape[0]
    w = floor + (1.0 - floor) * pct
    print(f"  [quality] floor={floor}  win_rate range [{wr.min():.3f}, {wr.max():.3f}]  "
          f"weight range [{w.min():.3f}, {w.max():.3f}]")
    return w.astype(np.float32)


def combine_sample_weights(*weights):
    """Element-wise product of optional per-row weight arrays (None == all-ones).
    Returns None if every input is None."""
    present = [w for w in weights if w is not None]
    if not present:
        return None
    out = np.ones_like(present[0], dtype=np.float32)
    for w in present:
        out = out * w
    return out


def train_xgb(X, y, val_mask, out_json: Path,
              max_depth=6, learning_rate=0.08, n_est=600, early_stopping=40, weight=None):
    import xgboost as xgb
    yb = (y > 0).astype(np.float32)
    w_tr = weight[~val_mask] if weight is not None else None
    dtr = xgb.DMatrix(X[~val_mask], label=yb[~val_mask], weight=w_tr)
    dva = xgb.DMatrix(X[val_mask], label=yb[val_mask])
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        verbosity=0,
    )
    t0 = time.time()
    bst = xgb.train(
        params, dtr, num_boost_round=n_est,
        evals=[(dva, "val")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    pred = bst.predict(dva)
    sign_acc = float(((pred > 0.5) == (yb[val_mask] > 0.5)).mean())
    elapsed = time.time() - t0
    print(f"  XGB val sign-acc = {100*sign_acc:.3f}%  "
          f"best_iter={bst.best_iteration}  t={elapsed:.1f}s")
    if bst.best_iteration >= n_est - 1:
        print(f"  NOTE best_iter hit the {n_est}-round cap (no early stop) — "
              f"raise --rounds for a likely better model")
    bst.save_model(str(out_json))
    print(f"  saved {out_json} ({out_json.stat().st_size / 1e6:.2f} MB)")
    return sign_acc, bst.best_iteration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "--data", required=True, type=Path,
                   help="combined NPZ from build_from_zip.py")
    p.add_argument("--top10-out", type=Path)
    p.add_argument("--model-out", type=Path)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--gate", choices=("both-topn", "strong-topn", "strong-median"), default="both-topn",
                   help="both-topn: keep games where every player is in the top-N (strict). "
                        "strong-topn: per-perspective, keep a row iff that row's player is in the "
                        "top-N by win rate (keeps the strong side of mismatches; records win_rate). "
                        "strong-median: same but the bar is the median win rate instead of top-N.")
    p.add_argument("--no-filter", action="store_true", help="train on all rows; useful for combined self-play/candidate datasets")
    p.add_argument("--filter-only", action="store_true",
                   help="write the filtered NPZ (--top10-out) and skip training; used for the per-day ladder step")
    p.add_argument("--recency-halflife", type=float, default=0.0,
                   help="down-weight older rows by 0.5 per this many days (reads combine_npz `source`; 0 = uniform). Use with --no-filter.")
    p.add_argument("--quality-weight", action="store_true",
                   help="soft-weight rows by player strength (reads `win_rate` from the strong-median gate). "
                        "Multiplies into the recency weight. Use with --no-filter.")
    p.add_argument("--quality-floor", type=float, default=0.25,
                   help="weakest kept player's quality weight (strongest = 1.0); only with --quality-weight")
    p.add_argument("--rounds", type=int, default=600, help="max XGBoost boosting rounds")
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--early-stopping", type=int, default=40,
                   help="stop if val logloss hasn't improved in this many rounds")
    args = p.parse_args()

    if not args.filter_only and args.model_out is None:
        raise SystemExit("--model-out is required unless --filter-only is set")

    print(f"Loading {args.input}...")
    d = np.load(args.input, allow_pickle=False)
    n_games = len(np.unique(d["meta"][:, 0])) if "game_names" not in d.files else d["game_names"].shape[0]
    n_rows = d["summary_v2"].shape[0]
    print(f"  {n_games} games / {n_rows:,} rows")

    source = d["source"] if "source" in d.files else None
    win_rate = d["win_rate"] if "win_rate" in d.files else None

    if args.no_filter or "game_names" not in d.files:
        print("\n=== STEP 1: no filter ===")
        if not args.no_filter and "game_names" not in d.files:
            print("  input has no game_names; training on all rows")
        Xs = d["summary_v2"].astype(np.float32)
        ys = d["labels"].astype(np.float32)
        ms = d["meta"].astype(np.int32)
        row_source = source
        row_win_rate = win_rate
    else:
        if args.top10_out is None:
            raise SystemExit("--top10-out is required unless --no-filter is set")
        if args.gate in ("strong-topn", "strong-median"):
            label = f"top-{args.top_n}" if args.gate == "strong-topn" else "median"
            print(f"\n=== STEP 1: strong-side {label} gate (min {args.min_games} games) ===")
            strong_set, rates = strong_set_for(d, args.gate, args.top_n, args.min_games)
            Xs, ys, ms, sub, row_win_rate = filter_strong_side(d, args.top10_out, strong_set, rates)
        else:
            print(f"\n=== STEP 1: both-players top-{args.top_n} filter (min {args.min_games} games) ===")
            top_set = compute_top_n(d, n_top=args.top_n, min_games=args.min_games)
            Xs, ys, ms, sub = filter_top_n(d, top_set, args.top10_out)
            row_win_rate = win_rate[sub] if win_rate is not None else None
        row_source = source[sub] if source is not None else None

    if args.filter_only:
        print("\n--filter-only set; skipping training.")
        print("Done.")
        return

    weight = combine_sample_weights(
        recency_weights(row_source, args.recency_halflife),
        quality_weights(row_win_rate, args.quality_floor, args.quality_weight),
    )

    print(f"\n=== STEP 2: train XGB (binary:logistic d=6 lr=0.08 n_est=600) ===")
    val_mask = game_level_split_mask(ms, frac=0.12, seed=42)
    n_train_games = len(np.unique(ms[~val_mask, 0]))
    n_val_games = len(np.unique(ms[val_mask, 0]))
    print(f"  split: train games={n_train_games}, val games={n_val_games}, "
          f"train rows={(~val_mask).sum():,}, val rows={val_mask.sum():,}")
    train_xgb(Xs, ys, val_mask, args.model_out, weight=weight,
              max_depth=args.max_depth, learning_rate=args.learning_rate,
              n_est=args.rounds, early_stopping=args.early_stopping)

    print("\nDone.")


if __name__ == "__main__":
    main()
