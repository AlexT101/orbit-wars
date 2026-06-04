"""Derive scalar summary features from the raw 2728-d feature vector.

These are spatially-blind hand-coded features retained for parity checks and
historical comparison against SummaryV2 extraction.
"""

from __future__ import annotations

import numpy as np

PER_OBJECT = 9
MAX_OBJECTS = 44
PER_BLOCK = MAX_OBJECTS * PER_OBJECT  # 396
DIST_BLOCK = MAX_OBJECTS * MAX_OBJECTS  # 1936
INPUT_DIM = 2 * PER_BLOCK + DIST_BLOCK  # 2728

# Per-object slot:
#   [is_me, is_opp, is_neutral, log1p(ships), radius,
#    is_static, is_orbit, is_comet, production]


def split_blocks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x: [B, INPUT_DIM] → (current[B, MAX, PER], extrap[B, MAX, PER], dist[B, MAX, MAX])"""
    B = x.shape[0]
    cur = x[:, :PER_BLOCK].reshape(B, MAX_OBJECTS, PER_OBJECT)
    ext = x[:, PER_BLOCK : 2 * PER_BLOCK].reshape(B, MAX_OBJECTS, PER_OBJECT)
    dist = x[:, 2 * PER_BLOCK :].reshape(B, MAX_OBJECTS, MAX_OBJECTS)
    return cur, ext, dist


def summary_features(x: np.ndarray, steps: np.ndarray | None = None) -> np.ndarray:
    """Hand-coded scalar summary. Returns [B, F] with named F dimensions.

    `steps`, if given, is a per-sample int step (0..500) appended as
    extra time-remaining features. If None, those slots are zero —
    inference paths that don't have step info still work.
    """
    cur, ext, dist = split_blocks(x)
    is_me = cur[..., 0]
    is_opp = cur[..., 1]
    is_neutral = cur[..., 2]
    ships_log = cur[..., 3]
    radius = cur[..., 4]
    production = cur[..., 8]

    is_me_ext = ext[..., 0]
    is_opp_ext = ext[..., 1]
    is_neutral_ext = ext[..., 2]
    ships_log_ext = ext[..., 3]

    # Existence mask: a real planet has any one-hot row sum > 0.
    exists = (is_me + is_opp + is_neutral) > 0
    n_planets = exists.sum(axis=1, keepdims=False)

    def total(mask, ships):
        return (mask * np.expm1(ships)).sum(axis=1)  # invert log1p

    feats = []
    feats.append(total(is_me, ships_log))  # my total ships (current)
    feats.append(total(is_opp, ships_log))  # opp total ships (current)
    feats.append(total(is_neutral, ships_log))  # neutral total
    feats.append(total(is_me_ext, ships_log_ext))  # my total ships (extrap)
    feats.append(total(is_opp_ext, ships_log_ext))  # opp total ships (extrap)

    feats.append(is_me.sum(axis=1))  # my planet count
    feats.append(is_opp.sum(axis=1))  # opp planet count
    feats.append(is_neutral.sum(axis=1))
    feats.append(is_me_ext.sum(axis=1) - is_me.sum(axis=1))  # planet flips toward me
    feats.append(is_opp_ext.sum(axis=1) - is_opp.sum(axis=1))  # planet flips toward opp

    feats.append((is_me * production).sum(axis=1))  # my prod
    feats.append((is_opp * production).sum(axis=1))  # opp prod

    feats.append((is_me * radius).sum(axis=1))
    feats.append((is_opp * radius).sum(axis=1))

    # Spatial: distance-weighted "pressure" on each side.
    # For each cell, "im_close_to_opp" = sum over j of is_opp[j] / (1+dist[i,j]).
    # Then sum over my planets.
    inv_d = 1.0 / (1.0 + dist)
    np.einsum  # placate linters
    pressure_me_to_opp = np.einsum("bi,bij,bj->b", is_me, inv_d, is_opp)
    pressure_opp_to_me = np.einsum("bi,bij,bj->b", is_opp, inv_d, is_me)
    pressure_me_to_neutral = np.einsum("bi,bij,bj->b", is_me, inv_d, is_neutral)
    pressure_opp_to_neutral = np.einsum("bi,bij,bj->b", is_opp, inv_d, is_neutral)
    feats.append(pressure_me_to_opp)
    feats.append(pressure_opp_to_me)
    feats.append(pressure_me_to_neutral)
    feats.append(pressure_opp_to_neutral)

    feats.append(n_planets)

    # Single-biggest planet ships per side (raw ships, not log).
    ships_raw = np.expm1(ships_log)
    feats.append((is_me * ships_raw).max(axis=1))
    feats.append((is_opp * ships_raw).max(axis=1))
    # Frontline distance: min dist from any of my planets to any opp planet.
    # If either side has no planets, fall back to board diagonal.
    BIG = 200.0
    me_mask = is_me[:, :, None]
    opp_mask = is_opp[:, None, :]
    pair_mask = (me_mask * opp_mask) > 0  # [B, i, j]
    masked_dist = np.where(pair_mask, dist, BIG)
    front = masked_dist.reshape(masked_dist.shape[0], -1).min(axis=1)
    feats.append(front)
    # Log-ratio of total ships (compact relative-strength signal).
    log_ratio = np.log1p(total(is_me, ships_log)) - np.log1p(total(is_opp, ships_log))
    feats.append(log_ratio.astype(np.float32))

    out = np.stack(feats, axis=1).astype(np.float32)
    return out


FEATURE_NAMES = [
    "my_ships",
    "opp_ships",
    "neutral_ships",
    "my_ships_extrap",
    "opp_ships_extrap",
    "my_planets",
    "opp_planets",
    "neutral_planets",
    "delta_my_planets",
    "delta_opp_planets",
    "my_production",
    "opp_production",
    "my_radius_sum",
    "opp_radius_sum",
    "pressure_me_to_opp",
    "pressure_opp_to_me",
    "pressure_me_to_neutral",
    "pressure_opp_to_neutral",
    "n_planets",
    "max_my_planet_ships",
    "max_opp_planet_ships",
    "frontline_dist",
    "log_ship_total_ratio",
]


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:]
    arrs = [np.load(p) for p in paths]
    xs = np.concatenate([a["features"] for a in arrs], axis=0)
    ys = np.concatenate([a["labels"] for a in arrs], axis=0)
    feats = summary_features(xs)
    print("feats:", feats.shape, "labels:", ys.shape)
    for i, name in enumerate(FEATURE_NAMES):
        # correlation with label
        v = feats[:, i]
        std = v.std()
        if std < 1e-6:
            corr = 0.0
        else:
            corr = float(np.corrcoef(v, ys)[0, 1])
        print(f"  {name:24s}  mean={v.mean():.2f}  std={std:.2f}  corr_with_label={corr:+.3f}")
