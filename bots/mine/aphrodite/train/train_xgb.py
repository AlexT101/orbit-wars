"""The canonical Aphrodite value-net trainer: optional win-rate filtering, then
sample weighting (recency / quality / decisiveness), decided-row dropping, and
optional feature zeroing, then XGBoost (binary:logistic, max_depth=6, lr=0.08).
Saves the booster as `weights/<out_stem>.json`.

Pass `--no-filter` for an already-gated dataset (e.g. build_ladder.py's
`combined.npz`); the weighting / drop / zero-cols steps still apply.

Usage:
    # train on an already-gated combined NPZ with the full preprocessing
    python train_xgb.py \\
        --data data/2p/_ladder_work/combined.npz --no-filter \\
        --quality-weight --decisiveness-weight --drop-decided \\
        --zero-cols 4,8,13,17,21,25,29,33,37,40,41,61,63,64 \\
        --rounds 2000 --model-out weights/xgb_2p.json

    # gate a raw build_from_zip.py NPZ by top-N win rate, then train
    python train_xgb.py \\
        --input data/2p/old_top10.npz \\
        --top10-out data/2p/old_top10_rebuilt.npz \\
        --model-out weights/xgb_2p_old_top10.json
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
          f"{len(Xs) * 176 / 1e6:.1f} MB raw")
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
    print(f"  [quality] floor={floor}  strength range [{wr.min():.3f}, {wr.max():.3f}]  "
          f"weight range [{w.min():.3f}, {w.max():.3f}]")
    return w.astype(np.float32)


def bradley_terry_ratings(game_names, game_rewards, min_games: int, iters: int = 200,
                          prior_strength: float = 3.0):
    """Fit a (regularized) Bradley-Terry skill rating per player from outcomes.

    Unlike raw win rate, this is OPPONENT-ADJUSTED: a player who goes ~.500
    against strong opponents outrates one who goes ~.500 against weak ones —
    important on an Elo-matched ladder where win rates compress toward 0.5.

    Each game is decomposed into pairwise "higher reward beats lower reward"
    comparisons (so it handles 2p and 4p). Strengths theta_i are fit by an MM
    iteration with a Gamma(K, K) prior (`prior_strength=K`):

        theta_i <- (wins_i + K) / (sum_j n_ij/(theta_i+theta_j) + K)

    The prior == K pseudo-games against an average (strength-1) opponent, half
    won. It both BOUNDS the estimate (an undefeated few-game player no longer
    diverges to +inf) and SHRINKS low-sample players toward the field mean, so a
    5-game 4-0 run can't outrank a 3000-game veteran. Renormalized to geometric
    mean 1. Returns {name: log(theta)} (higher = stronger) for players with
    >= `min_games` appearances; others are omitted so the caller falls back to
    the median rating for them.
    """
    from collections import defaultdict

    names = sorted({str(n) for n in game_names.reshape(-1).tolist()})
    idx = {n: i for i, n in enumerate(names)}
    P = len(names)
    wins = np.zeros(P)
    appearances = np.zeros(P)
    pair_count: dict = defaultdict(float)  # (lo_idx, hi_idx) -> games between them

    n_players = game_names.shape[1]
    for gi in range(game_names.shape[0]):
        row = game_names[gi]
        rew = game_rewards[gi]
        ids = [idx[str(row[s])] for s in range(n_players)]
        for s in ids:
            appearances[s] += 1
        for a in range(n_players):
            for b in range(a + 1, n_players):
                ra, rb = float(rew[a]), float(rew[b])
                if ra == rb:
                    continue  # tie: no information
                ia, ib = ids[a], ids[b]
                wins[ia if ra > rb else ib] += 1
                pair_count[(min(ia, ib), max(ia, ib))] += 1

    adj: dict = defaultdict(list)
    for (i, j), c in pair_count.items():
        adj[i].append((j, c))
        adj[j].append((i, c))

    K = float(prior_strength)
    theta = np.ones(P)
    for _ in range(iters):
        new = theta.copy()
        for i in range(P):
            denom = 0.0
            ti = theta[i]
            for (j, c) in adj[i]:
                denom += c / (ti + theta[j])
            # Gamma(K, K) prior: bounds undefeated divergence + shrinks low-game
            # players toward the strength-1 (average) anchor.
            new[i] = (wins[i] + K) / (denom + K)
        new = np.where(new <= 0, 1e-9, new)
        theta = new / np.exp(np.mean(np.log(new)))  # renormalize geo-mean -> 1

    rating = {names[i]: float(np.log(theta[i])) for i in range(P) if appearances[i] >= min_games}
    if rating:
        vals = np.array(list(rating.values()))
        print(f"  [rating] Bradley-Terry over {len(rating)}/{P} players (>= {min_games} games); "
              f"log-strength range [{vals.min():.3f}, {vals.max():.3f}]")
    return rating


def per_row_strength_from_rating(game_names, meta, rating: dict):
    """Map each row to its player's rating via meta (gid, _, slot, _). Players
    not in `rating` (too few games) get the median, so they weigh neutrally."""
    n_games, n_players = game_names.shape
    med = float(np.median(list(rating.values()))) if rating else 0.0
    grid = np.empty((n_games, n_players), dtype=np.float32)
    for g in range(n_games):
        for s in range(n_players):
            grid[g, s] = rating.get(str(game_names[g, s]), med)
    gid = meta[:, 0].astype(np.int64)
    slot = meta[:, 2].astype(np.int64)
    return grid[gid, slot]


# ── decisiveness down-weighting ─────────────────────────────────────────────
# summary_v2 (65-dim) column indices. Layout: me_cur[0:9] opp_cur[9:18]
# me_ext[18:26] opp_ext[26:34] neutral[34:41] relational[41:65] (see value_net.rs).
_COL_ME_PROD_STATIC = 5      # me: current static-planet production
_COL_ME_PROD_ORBIT = 6       # me: current orbiting-planet production
_COL_OPP_PROD_STATIC = 14    # dominant enemy: current static-planet production
_COL_OPP_PROD_ORBIT = 15     # dominant enemy: current orbiting-planet production
_COL_NEUT_PROD_STATIC = 38   # neutral: static-planet production
_COL_NEUT_PROD_ORBIT = 39    # neutral: orbiting-planet production
_COL_SHIP_SHARE = 48         # ship_share: my ships / (my + enemy) ships
_COL_PRODUCTION_SHARE = 49   # production_share: my prod / (my + enemy) prod
# Tuning knobs for the decisiveness weight (down-weights decided positions):
_DEC_ADV_SHIP_W = 0.5        # blend weight of ship_share in the advantage
_DEC_ADV_PROD_W = 0.5        # blend weight of production_share in the advantage
_DEC_LEAD_TAU = 0.70         # advantage at which a side counts as clearly "ahead"
_DEC_LEAD_K = 15.0           # steepness of the "ahead" sigmoid (higher = sharper)
_DEC_MATURE_TAU = 0.70       # claimed-production fraction at which it's "endgame"
_DEC_MATURE_K = 12.0         # steepness of the "endgame" sigmoid (higher = sharper)
_DEC_ALPHA = 1.0             # exponent shaping mid-range weight falloff (higher = steeper)
_DEC_FLOOR = 0.2             # min weight kept for blowouts (preserves extreme calibration)
# Hard-drop knobs for fully decided rows (--drop-decided); see decided_keep_mask:
_DEC_DROP_LEAD = 0.80        # drop rows where a side's advantage exceeds this
_DEC_DROP_MATURE = 0.75      # ...AND this much production is claimed (never drops early leads)

# ── player-count-correct decided/decisiveness from `decisiveness_aux` ────────
# The summary_v2-column metric above uses d_rel = max(adv, 1-adv) on a [0.5, 1]
# scale (0.5 = even), which is ONLY valid for 2p — in 4p "even" is 0.25, so it
# mis-flags ordinary positions as decided. With `decisiveness_aux` present
# (per-player ship strength + production + neutral prod) we instead use the
# seat-invariant top-two strength gap `lead = (s1 - s2)/(s1 + s2)` in [0, 1]
# (0 = even regardless of player count) and all-player `claimed`. Thresholds are
# the [0.5,1]→[0,1] remaps of the ones above (drop_lead 0.80→0.60, lead_tau
# 0.70→0.40), so 2p reduces to the same behaviour.
_DEC_LEAD_TAU_AUX = 0.40
_DEC_DROP_LEAD_AUX = 0.60


def _has_summary_cols(X, who: str) -> bool:
    """True if X is wide enough to index the summary_v2 columns used below."""
    need = _COL_PRODUCTION_SHARE + 1
    if X.shape[1] < need:
        print(f"  [{who}] WARN feature matrix has {X.shape[1]} cols (<{need}); "
              f"skipping (need full 65-dim summary_v2)")
        return False
    return True


def _advantage_and_claimed(X):
    """Per-row (d_rel, claimed) from summary_v2 columns, shared by the
    decisiveness weight and the decided-drop.

      d_rel   = max(adv, 1-adv)  — symmetric lead in [0.5, 1] (which side is
                ahead doesn't matter; a blowout is a blowout from either seat).
      claimed = player_prod / (player + neutral prod) — endgame proxy: how much
                of the map is taken vs still neutral, so an early lead over a
                mostly-neutral board reads as low-maturity, not decided.

    NOTE: `claimed` uses only the dominant enemy's production (the opp_cur
    block), so in 4p it under-counts other enemies and reads conservatively low
    (maturity gate won't over-trigger). Exact in 2p.
    """
    Xf = X.astype(np.float64)
    adv = _DEC_ADV_SHIP_W * Xf[:, _COL_SHIP_SHARE] + _DEC_ADV_PROD_W * Xf[:, _COL_PRODUCTION_SHARE]
    d_rel = np.maximum(adv, 1.0 - adv)
    me_prod = Xf[:, _COL_ME_PROD_STATIC] + Xf[:, _COL_ME_PROD_ORBIT]
    en_prod = Xf[:, _COL_OPP_PROD_STATIC] + Xf[:, _COL_OPP_PROD_ORBIT]
    neu_prod = Xf[:, _COL_NEUT_PROD_STATIC] + Xf[:, _COL_NEUT_PROD_ORBIT]
    claimed = (me_prod + en_prod) / np.maximum(me_prod + en_prod + neu_prod, 1e-6)
    return d_rel, claimed


def decisiveness_weights(X, enabled: bool):
    """Per-row sample weight that down-weights DECIDED positions so the model
    spends its capacity on contested midgame states (where DUCT actually has to
    discriminate between candidate moves). A position is decided only when BOTH
    a side is far ahead AND most of the map is claimed:

        decisiveness = sigmoid(k1*(d_rel - tau1)) * sigmoid(k2*(claimed - tau2))
        weight       = floor + (1 - floor) * (1 - decisiveness)^alpha

    Computed from summary_v2 columns (no re-extraction). Returns None when
    disabled or the matrix is too narrow.
    """
    if not enabled:
        return None
    if not _has_summary_cols(X, "decisiveness"):
        return None
    d_rel, claimed = _advantage_and_claimed(X)
    lead_term = 1.0 / (1.0 + np.exp(-_DEC_LEAD_K * (d_rel - _DEC_LEAD_TAU)))
    mature_term = 1.0 / (1.0 + np.exp(-_DEC_MATURE_K * (claimed - _DEC_MATURE_TAU)))
    decisiveness = lead_term * mature_term
    w = _DEC_FLOOR + (1.0 - _DEC_FLOOR) * (1.0 - decisiveness) ** _DEC_ALPHA
    print(f"  [decisiveness] floor={_DEC_FLOOR} alpha={_DEC_ALPHA} "
          f"lead(tau={_DEC_LEAD_TAU},k={_DEC_LEAD_K}) mature(tau={_DEC_MATURE_TAU},k={_DEC_MATURE_K})  "
          f"weight range [{w.min():.3f}, {w.max():.3f}] mean={w.mean():.3f}  "
          f"rows<0.5w: {100 * (w < 0.5).mean():.1f}%")
    return w.astype(np.float32)


def decided_keep_mask(X, enabled: bool):
    """Boolean KEEP-mask (True = keep) that hard-drops fully decided rows: a
    side's advantage exceeds `_DEC_DROP_LEAD` AND the board is mature
    (`claimed > _DEC_DROP_MATURE`). The maturity condition guarantees an early
    lead over a mostly-neutral board is never dropped. Returns None (keep all)
    when disabled or the matrix is too narrow.

    Safe for the MCTS eval as long as the 0.7-0.9 band is retained (it is, via
    the decisiveness floor): XGBoost trees saturate to the most-winning leaf for
    >0.9 leaf states seen in search, so ordering stays monotonic (winning >
    contested) without needing fine resolution in the dropped tail.
    """
    if not enabled:
        return None
    if not _has_summary_cols(X, "drop-decided"):
        return None
    d_rel, claimed = _advantage_and_claimed(X)
    drop = (d_rel > _DEC_DROP_LEAD) & (claimed > _DEC_DROP_MATURE)
    keep = ~drop
    print(f"  [drop-decided] lead>{_DEC_DROP_LEAD} & claimed>{_DEC_DROP_MATURE}: "
          f"dropping {int(drop.sum()):,}/{len(drop):,} rows ({100 * drop.mean():.1f}%)")
    return keep


def _lead_claimed_from_aux(aux):
    """Player-count-correct (lead, claimed) from `decisiveness_aux`
    (`[ship[0..4], prod[0..4], neutral_prod]` per row).

      lead    = (s1 - s2)/(s1 + s2) over the top-two ship strengths — in [0, 1],
                0 = even REGARDLESS of player count (fixes the 2p-only metric).
      claimed = Σ player prod / (Σ player prod + neutral prod).
    """
    aux = aux.astype(np.float64)
    ship = aux[:, 0:4]
    prod = aux[:, 4:8]
    neutral = aux[:, 8]
    top2 = np.sort(ship, axis=1)[:, ::-1][:, :2]  # two strongest per row
    s1, s2 = top2[:, 0], top2[:, 1]
    lead = np.where(s1 + s2 > 1e-9, (s1 - s2) / np.maximum(s1 + s2, 1e-9), 0.0)
    pl = prod.sum(axis=1)
    claimed = np.where(pl + neutral > 1e-9, pl / np.maximum(pl + neutral, 1e-9), 0.0)
    return lead.astype(np.float32), claimed.astype(np.float32)


def decisiveness_weights_aux(aux, enabled: bool):
    """Player-count-correct decisiveness weight from `decisiveness_aux`. Same
    sigmoid shape as `decisiveness_weights` but on the [0,1] top-two-gap `lead`
    (tau remapped to `_DEC_LEAD_TAU_AUX`)."""
    if not enabled or aux is None:
        return None
    lead, claimed = _lead_claimed_from_aux(aux)
    lead_term = 1.0 / (1.0 + np.exp(-_DEC_LEAD_K * (lead - _DEC_LEAD_TAU_AUX)))
    mature_term = 1.0 / (1.0 + np.exp(-_DEC_MATURE_K * (claimed - _DEC_MATURE_TAU)))
    decisiveness = lead_term * mature_term
    w = _DEC_FLOOR + (1.0 - _DEC_FLOOR) * (1.0 - decisiveness) ** _DEC_ALPHA
    print(f"  [decisiveness/aux] floor={_DEC_FLOOR} alpha={_DEC_ALPHA} "
          f"lead(tau={_DEC_LEAD_TAU_AUX},k={_DEC_LEAD_K}) mature(tau={_DEC_MATURE_TAU},k={_DEC_MATURE_K})  "
          f"weight range [{w.min():.3f}, {w.max():.3f}] mean={w.mean():.3f}  "
          f"rows<0.5w: {100 * (w < 0.5).mean():.1f}%")
    return w.astype(np.float32)


def decided_keep_mask_aux(aux, enabled: bool):
    """Player-count-correct decided-drop from `decisiveness_aux`."""
    if not enabled or aux is None:
        return None
    lead, claimed = _lead_claimed_from_aux(aux)
    drop = (lead > _DEC_DROP_LEAD_AUX) & (claimed > _DEC_DROP_MATURE)
    keep = ~drop
    print(f"  [drop-decided/aux] lead>{_DEC_DROP_LEAD_AUX} & claimed>{_DEC_DROP_MATURE}: "
          f"dropping {int(drop.sum()):,}/{len(drop):,} rows ({100 * drop.mean():.1f}%)")
    return keep


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


def roc_auc(y_true, score):
    """Rank-based ROC AUC (no sklearn dependency). `y_true` is 0/1; `score` is
    any monotone-with-confidence value (e.g. predicted win prob)."""
    order = np.argsort(score, kind="mergesort")
    yr = np.asarray(y_true)[order]
    npos = float(yr.sum())
    nneg = float(len(yr) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.arange(1, len(yr) + 1, dtype=np.float64)
    return float((ranks[yr == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def summary_v2_monotone():
    """Monotone-constraint vector for the 65-d summary_v2 layout (used by BOTH
    the 2p net and the 4p v2aux net; see value_net.rs::summary_features_v2).
    +1 = win-prob non-decreasing in the feature, -1 = non-increasing, 0 = free.

    ONLY features with an unconditional sign prior are constrained: my/opponent
    ship + production totals, the share/pressure/vulnerability relational
    columns. Ambiguous features (planet COUNTS, fleet fractions, dispersion,
    centroid distance, step) are left at 0 — a wrong constraint forbids a real
    relationship and hurts both AUC and play.
    """
    c = [0] * 65
    # me current (0:9) + me extrap (18:26): more of MY ships / production helps.
    #   cur:  ships_on=0, prod_static=5, prod_orbit=6
    #   ext:  ships_on=18, prod_static=22, prod_orbit=23
    for i in (0, 5, 6, 18, 22, 23):
        c[i] = 1
    # dominant-enemy current (9:18) + extrap (26:34): more of THEIRS hurts.
    #   cur:  ships_on=9, prod_static=14, prod_orbit=15
    #   ext:  ships_on=26, prod_static=30, prod_orbit=31
    for i in (9, 14, 15, 26, 30, 31):
        c[i] = -1
    # relational block (41:65); index = 41 + slot in relational_block()
    c[46] = -1  # num_my_vulnerable_planets
    c[47] = 1   # num_enemy_vulnerable_planets
    c[48] = 1   # ship_share (me / me+enemy)
    c[49] = 1   # production_share
    c[50] = -1  # my_production_at_risk
    c[51] = 1   # enemy_production_at_opportunity
    c[52] = -1  # max_enemy_pressure (on my planets)
    c[53] = 1   # max_ally_pressure (on enemy planets)
    c[57] = -1  # prod_weighted_enemy_pressure
    c[58] = 1   # prod_weighted_ally_pressure
    c[61] = -1  # my_strength_rank (0 = leader, higher = worse)
    c[62] = 1   # leader_strength_ratio
    return c


def summary_v3_monotone():
    """Monotone-constraint vector for the 145-d summary_v3 layout (4p FFA; see
    value_net.rs::summary_features_v3 assembly). v3 is share-normalized, so the
    sign priors are clean: MY share/economy/aim ↑ = win-prob ↑; any OPPONENT's
    share/economy or their threat on me ↑ = win-prob ↓.

    Only unconditional-sign features are constrained: my vs opponent ship +
    production SHARES, my-vs-k pairwise shares, vulnerability (me attacking =
    +, me defending = -), and threat-on-me aggregates. Planet-count fractions,
    fleet fraction, dispersion, centroid distance, in-flight direction, board
    anchors, step/angular_velocity, and opp-vs-opp cells are left free.
    """
    c = [0] * 145
    # globals 0 step, 1 angular_velocity → free
    # me_cur (2..11): ships_on=2, ships_fly=3, prod_static=7, prod_orbit=8
    for i in (2, 3, 7, 8):
        c[i] = 1
    # me_ext (11..19): ships_on=11, prod_static=15, prod_orbit=16
    for i in (11, 15, 16):
        c[i] = 1
    # neutral block (19..26) → free
    # aggregate (26..41)
    c[26] = 1   # ship_share_me_vs_all
    c[27] = 1   # production_share
    c[28] = -1  # my_n_vuln
    c[29] = -1  # my_prod_at_risk
    c[30] = -1  # my_threat_max
    c[31] = -1  # my_pw_threat
    c[34] = 1   # avg_ally_ships
    c[35] = 1   # leader_strength_ratio
    c[40] = 1   # production[me] (absolute anchor — more of my own prod helps)
    # per-opponent blocks: 3 × 24 at base 41, 65, 89.
    for slot in range(3):
        b = 41 + slot * 24
        c[b + 0] = -1   # k ships_on share
        c[b + 1] = -1   # k ships_fly share
        c[b + 5] = -1   # k prod_static share
        c[b + 6] = -1   # k prod_orbit share
        c[b + 9] = -1   # k e_ships_on share
        c[b + 13] = -1  # k e_prod_static share
        c[b + 14] = -1  # k e_prod_orbit share
        c[b + 17] = 1   # pw_my_on_k  (my prod-weighted pressure on k)
        c[b + 18] = -1  # pw_k_on_me  (k's pressure on me)
        c[b + 21] = 1   # strength share me/(me+k)
        c[b + 22] = 1   # production share me/(me+k)
    # vulnerability matrix (129..145), slot order S = [me, o1, o2, o3].
    #   me-as-attacker = +1 (I can take their prod), me-as-defender = -1.
    for i in (129, 130, 131, 141):  # me->o1, me->o2, me->o3, me->neutral
        c[i] = 1
    for i in (132, 135, 138):       # o1->me, o2->me, o3->me
        c[i] = -1
    # in-flight matrix (113..129) → free (fleet direction is not sign-definite)
    return c


def train_xgb(X, y, val_mask, out_json: Path,
              max_depth=6, learning_rate=0.08, n_est=600, early_stopping=40, weight=None,
              min_child_weight=1.0, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
              subsample=0.85, colsample_bytree=0.85, max_bin=256,
              monotone=False, early_stop_metric="logloss"):
    import xgboost as xgb
    yb = (y > 0).astype(np.float32)
    w_tr = weight[~val_mask] if weight is not None else None
    dtr = xgb.DMatrix(X[~val_mask], label=yb[~val_mask], weight=w_tr)
    dva = xgb.DMatrix(X[val_mask], label=yb[val_mask])
    # Early stopping watches the LAST eval_metric, so order the list to put the
    # chosen driver last; the other is still printed each round.
    eval_metric = (["auc", "logloss"] if early_stop_metric == "logloss"
                   else ["logloss", "auc"])
    params = dict(
        objective="binary:logistic",
        eval_metric=eval_metric,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_bin=max_bin,
        tree_method="hist",
        verbosity=0,
    )
    if monotone:
        cons = ({65: summary_v2_monotone, 145: summary_v3_monotone}
                .get(X.shape[1], lambda: None)())
        if cons is None:
            print(f"  [monotone] WARN feature matrix is {X.shape[1]}-d, not 65-d "
                  f"summary_v2 or 145-d summary_v3; skipping constraints")
        else:
            params["monotone_constraints"] = "(" + ",".join(map(str, cons)) + ")"
            n_con = sum(1 for v in cons if v != 0)
            kind = "summary_v2" if X.shape[1] == 65 else "summary_v3"
            print(f"  [monotone] constraining {n_con}/{X.shape[1]} {kind} columns "
                  f"(+1 {sum(v == 1 for v in cons)} / -1 {sum(v == -1 for v in cons)})")
    t0 = time.time()
    bst = xgb.train(
        params, dtr, num_boost_round=n_est,
        evals=[(dva, "val")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    pred = bst.predict(dva)
    sign_acc = float(((pred > 0.5) == (yb[val_mask] > 0.5)).mean())
    auc = roc_auc(yb[val_mask], pred)
    elapsed = time.time() - t0
    print(f"  XGB val AUC = {auc:.4f}  sign-acc = {100*sign_acc:.3f}%  "
          f"best_iter={bst.best_iteration}  (early-stop on {early_stop_metric})  t={elapsed:.1f}s")
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
                   help="soft-weight rows by player strength. Multiplies into the recency weight. "
                        "Use with --no-filter.")
    p.add_argument("--quality-metric", choices=("winrate", "rating"), default="winrate",
                   help="player-strength signal for --quality-weight. winrate: precomputed `win_rate` "
                        "column (opponent-dependent). rating: opponent-adjusted Bradley-Terry rating fit "
                        "from game_names+game_rewards (better on an Elo-matched ladder).")
    p.add_argument("--quality-floor", type=float, default=0.25,
                   help="weakest kept player's quality weight (strongest = 1.0); only with --quality-weight")
    p.add_argument("--decisiveness-weight", action="store_true",
                   help="down-weight DECIDED positions (a side far ahead AND map mostly claimed) so "
                        "training focuses on contested midgame states. Computed from summary_v2 "
                        "columns; tune via the _DEC_* / _COL_* constants. Multiplies into the other weights.")
    p.add_argument("--drop-decided", action="store_true",
                   help="hard-drop rows where a side's advantage exceeds _DEC_DROP_LEAD AND the board "
                        "is mature (claimed > _DEC_DROP_MATURE) — removes degenerate blowout/throwing "
                        "play. Applied before weighting and the train/val split.")
    p.add_argument("--rounds", type=int, default=600, help="max XGBoost boosting rounds")
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--early-stopping", type=int, default=40,
                   help="stop if the val early-stop metric hasn't improved in this many rounds")
    # ── booster regularization / sampling knobs (for tuning toward higher AUC) ──
    p.add_argument("--min-child-weight", type=float, default=1.0,
                   help="min sum hessian per leaf; raise (e.g. 5) to regularize")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="min split-loss reduction; raise (e.g. 0.1) to prune weak splits")
    p.add_argument("--reg-alpha", type=float, default=0.0, help="L1 weight penalty")
    p.add_argument("--reg-lambda", type=float, default=1.0, help="L2 weight penalty")
    p.add_argument("--subsample", type=float, default=0.85)
    p.add_argument("--colsample-bytree", type=float, default=0.85)
    p.add_argument("--max-bin", type=int, default=256,
                   help="hist bins; raise (e.g. 512) for finer splits at some cost")
    p.add_argument("--early-stop-metric", choices=("logloss", "auc"), default="logloss",
                   help="val metric that drives early stopping (both are always printed)")
    p.add_argument("--monotone", action="store_true",
                   help="apply unconditional monotone constraints on the summary_v2 "
                        "ship/production/share/pressure columns (see summary_v2_monotone). "
                        "Regularizes and gives DUCT a saner value surface; 65-d only.")
    p.add_argument("--zero-cols", type=str, default="",
                   help="comma-separated summary_v2 column indices to zero before training "
                        "(SHAP-dropped features). A constant column has zero split gain so XGBoost "
                        "ignores it — drops the feature while keeping the model 65-d so the Rust "
                        "runtime loads it unchanged. Applied AFTER drop/weighting (disjoint columns).")
    args = p.parse_args()

    if not args.filter_only and args.model_out is None:
        raise SystemExit("--model-out is required unless --filter-only is set")

    print(f"Loading {args.input}...")
    d = np.load(args.input, allow_pickle=False)
    feat_key = "summary_v3" if "summary_v3" in d.files else "summary_v2"
    n_games = len(np.unique(d["meta"][:, 0])) if "game_names" not in d.files else d["game_names"].shape[0]
    n_rows = d[feat_key].shape[0]
    print(f"  {n_games} games / {n_rows:,} rows  (features: {feat_key}, {d[feat_key].shape[1]}-d)")

    source = d["source"] if "source" in d.files else None
    win_rate = d["win_rate"] if "win_rate" in d.files else None
    # player-count-correct decided/decisiveness inputs (4p v3); None for v2.
    aux = d["decisiveness_aux"] if "decisiveness_aux" in d.files else None
    row_aux = None

    if args.no_filter or "game_names" not in d.files:
        print("\n=== STEP 1: no filter ===")
        if not args.no_filter and "game_names" not in d.files:
            print("  input has no game_names; training on all rows")
        Xs = d[feat_key].astype(np.float32)
        ys = d["labels"].astype(np.float32)
        ms = d["meta"].astype(np.int32)
        row_source = source
        row_win_rate = win_rate
        row_aux = aux
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
        row_aux = aux[sub] if aux is not None else None

    if args.filter_only:
        print("\n--filter-only set; skipping training.")
        print("Done.")
        return

    # When quality-weighting by Bradley-Terry rating, replace the per-row
    # strength signal (win_rate) with an opponent-adjusted rating fit from the
    # game outcomes. Done before the drop so it rides the same row-align path.
    if args.quality_weight and args.quality_metric == "rating":
        if "game_names" not in d.files or "game_rewards" not in d.files:
            raise SystemExit("--quality-metric rating needs game_names+game_rewards in the NPZ; "
                             "rebuild combined.npz with the updated combine_npz.py")
        ratings = bradley_terry_ratings(d["game_names"], d["game_rewards"], args.min_games)
        row_win_rate = per_row_strength_from_rating(d["game_names"], ms, ratings)

    # Hard-drop fully decided rows before weighting and the split, so every
    # downstream array (weights from win_rate/features, val_mask from meta)
    # stays row-aligned.
    # Player-count-correct decided/decisiveness when `decisiveness_aux` is present
    # (4p v3); otherwise the summary_v2-column metric (valid for 2p).
    keep = (decided_keep_mask_aux(row_aux, args.drop_decided) if row_aux is not None
            else decided_keep_mask(Xs, args.drop_decided))
    if keep is not None:
        Xs, ys, ms = Xs[keep], ys[keep], ms[keep]
        row_source = row_source[keep] if row_source is not None else None
        row_win_rate = row_win_rate[keep] if row_win_rate is not None else None
        row_aux = row_aux[keep] if row_aux is not None else None

    weight = combine_sample_weights(
        recency_weights(row_source, args.recency_halflife),
        quality_weights(row_win_rate, args.quality_floor, args.quality_weight),
        (decisiveness_weights_aux(row_aux, args.decisiveness_weight) if row_aux is not None
         else decisiveness_weights(Xs, args.decisiveness_weight)),
    )

    # Zero SHAP-dropped feature columns last, after the drop/weights have read
    # their (disjoint) columns. The model stays 65-d; zeroed columns get no
    # split gain so XGBoost ignores them, and the Rust runtime loads it unchanged.
    if args.zero_cols:
        cols = [int(c) for c in args.zero_cols.split(",") if c.strip() != ""]
        Xs = Xs.copy()
        Xs[:, cols] = 0.0
        print(f"  zeroed columns {cols} (treated as dropped; model stays {Xs.shape[1]}-d)")

    print(f"\n=== STEP 2: train XGB (binary:logistic d={args.max_depth} "
          f"lr={args.learning_rate} n_est={args.rounds} mcw={args.min_child_weight} "
          f"gamma={args.gamma} L1={args.reg_alpha} L2={args.reg_lambda} "
          f"max_bin={args.max_bin} monotone={args.monotone}) ===")
    val_mask = game_level_split_mask(ms, frac=0.12, seed=42)
    n_train_games = len(np.unique(ms[~val_mask, 0]))
    n_val_games = len(np.unique(ms[val_mask, 0]))
    print(f"  split: train games={n_train_games}, val games={n_val_games}, "
          f"train rows={(~val_mask).sum():,}, val rows={val_mask.sum():,}")
    train_xgb(Xs, ys, val_mask, args.model_out, weight=weight,
              max_depth=args.max_depth, learning_rate=args.learning_rate,
              n_est=args.rounds, early_stopping=args.early_stopping,
              min_child_weight=args.min_child_weight, gamma=args.gamma,
              reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
              subsample=args.subsample, colsample_bytree=args.colsample_bytree,
              max_bin=args.max_bin, monotone=args.monotone,
              early_stop_metric=args.early_stop_metric)

    print("\nDone.")


if __name__ == "__main__":
    main()
