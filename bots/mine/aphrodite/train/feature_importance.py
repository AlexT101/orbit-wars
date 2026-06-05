"""Print XGBoost feature importances for a SummaryV2 value net, with the
41 feature slots mapped to human-readable names in `extract()` order
(see value_net.rs::summary_features_v2).

The booster addresses features positionally as f0..f40, so this name list
MUST stay in sync with the order emitted by `extract()`. If you change the
feature layout, update FEATURE_NAMES to match.

Usage:
    ./venv/Scripts/python.exe bots/mine/aphrodite/train/feature_importance.py \\
        bots/mine/aphrodite/train/weights/xgb_2p.json

    # all three importance types, sorted by gain:
    ./venv/Scripts/python.exe bots/mine/aphrodite/train/feature_importance.py \\
        bots/mine/aphrodite/train/weights/xgb_2p.json --by gain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Per-block field names. Keep these aligned with value_net.rs.
# (prod_comet was dropped from every block: comets always have production 1,
# so it was an exact duplicate of n_comet.)
_CUR = [
    "ships_on_planets", "ships_flying", "n_static", "n_orbit", "n_comet",
    "prod_static", "prod_orbit", "n_neutrals_closer", "n_enemies_closer",
]
# extrap block omits ships_flying (nothing in flight post-extrapolation).
_EXT = [f for f in _CUR if f != "ships_flying"]
_NEUT = [
    "ships", "n_static", "n_orbit", "n_comet",
    "prod_static", "prod_orbit", "comet_time",
]

# Relational / positional block (shared, not per-player). Enemy = all
# non-me non-neutral planets. Keep order aligned with relational_block().
# Unweighted avg_enemy_pressure/avg_ally_pressure and ally/enemy_separation
# were dropped — their production-weighted counterparts (prod_weighted_*_pressure,
# *_economic_dispersion) fully subsumed them at equal play strength.
_REL = [
    "step",
    "avg_ally_ships_per_planet", "avg_enemy_ships_per_planet",
    "avg_ally_support", "avg_enemy_support",
    "num_my_vulnerable_planets", "num_enemy_vulnerable_planets",
    "ship_share", "production_share",
    "my_production_at_risk", "enemy_production_at_opportunity",
    "max_enemy_pressure", "max_ally_pressure",
    "centroid_to_centroid",
    "my_fleet_fraction", "enemy_fleet_fraction",
    "prod_weighted_enemy_pressure", "prod_weighted_ally_pressure",
    "ally_economic_dispersion", "enemy_economic_dispersion",
    # 4p / FFA standing (sparse arrivals-derived + border features ablated out)
    "my_strength_rank", "leader_strength_ratio", "opponent_strength_spread",
    "n_alive_players",
]

FEATURE_NAMES = (
    [f"me_{f}" for f in _CUR]          # 0..8
    + [f"opp_{f}" for f in _CUR]       # 9..17
    + [f"me_ext_{f}" for f in _EXT]    # 18..25
    + [f"opp_ext_{f}" for f in _EXT]   # 26..33
    + [f"neut_{f}" for f in _NEUT]     # 34..40
    + _REL                            # 41..59
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="path to a saved xgb *.json booster")
    p.add_argument("--by", choices=("gain", "weight", "cover", "total_gain"),
                   default="gain", help="sort key (default: gain)")
    args = p.parse_args()

    import xgboost as xgb

    bst = xgb.Booster()
    bst.load_model(str(args.model))

    if len(FEATURE_NAMES) != 65:
        print(f"WARN: FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected 65 "
              f"— update it to match the current extract() layout.", file=sys.stderr)

    # XGBoost keys importances by feature name; default booster uses f0..fN.
    # Pull every importance type so we can show them side by side.
    scores = {t: bst.get_score(importance_type=t)
              for t in ("gain", "weight", "cover", "total_gain")}

    def name_for(idx: int) -> str:
        return FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"f{idx}"

    rows = []
    for i in range(len(FEATURE_NAMES)):
        key = f"f{i}"
        rows.append((
            i,
            name_for(i),
            scores["gain"].get(key, 0.0),
            scores["total_gain"].get(key, 0.0),
            scores["weight"].get(key, 0.0),
            scores["cover"].get(key, 0.0),
        ))

    rank_idx = {"gain": 2, "total_gain": 3, "weight": 4, "cover": 5}[args.by]
    rows.sort(key=lambda r: -r[rank_idx])

    print(f"model: {args.model}")
    print(f"sorted by: {args.by}  (features never used by the booster show 0.0)\n")
    print(f"{'idx':>3}  {'feature':<26} {'gain':>12} {'total_gain':>14} "
          f"{'weight':>8} {'cover':>10}")
    print("-" * 78)
    for idx, name, gain, tgain, weight, cover in rows:
        flag = "  <- UNUSED" if weight == 0.0 else ""
        print(f"{idx:>3}  {name:<26} {gain:>12.4f} {tgain:>14.2f} "
              f"{weight:>8.0f} {cover:>10.2f}{flag}")

    unused = [name_for(i) for i in range(len(FEATURE_NAMES))
              if scores["weight"].get(f"f{i}", 0.0) == 0.0]
    if unused:
        print(f"\n{len(unused)} feature(s) never split on: {', '.join(unused)}")


if __name__ == "__main__":
    main()
