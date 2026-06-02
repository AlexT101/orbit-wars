"""Feature names for the 4-player evaluator feature contract.

Keep this file as the source of truth for labels/descriptions when the Rust
extractor layout changes. The actual feature values are produced by
`value_net::summary_features_4p_v1` and `src/bin/extract_4p_v*.rs`.
"""

from __future__ import annotations


GLOBAL_NAMES = [
    "4p.global.tick",
    "4p.global.tick_frac",
    "4p.global.remaining_frac",
    "4p.global.angular_velocity",
    "4p.global.n_planets",
    "4p.global.n_static",
    "4p.global.n_orbit",
    "4p.global.n_comet",
    "4p.global.n_neutral",
    "4p.global.neutral_ships",
    "4p.global.neutral_prod",
    "4p.global.neutral_comet_time",
    "4p.global.total_ships_planets",
    "4p.global.total_ships_flying",
    "4p.global.total_prod",
    "4p.global.total_planets_owned",
    "4p.global.enemy_count",
    "4p.global.orbit_rotation_pressure",
]

PLAYER_BLOCK = [
    "cur.ships_planets",
    "cur.ships_flying",
    "cur.ships_total",
    "cur.n_static",
    "cur.n_orbit",
    "cur.n_comet",
    "cur.planets",
    "cur.prod_static",
    "cur.prod_orbit",
    "cur.prod_comet",
    "cur.prod_total",
    "cur.neutrals_closer",
    "cur.enemies_closer",
    "cur.avg_garrison",
    "cur.prod_per_planet",
    "cur.flying_commitment",
    "cur.total_speed",
    "ext.ships_planets",
    "ext.n_static",
    "ext.n_orbit",
    "ext.n_comet",
    "ext.planets",
    "ext.prod_total",
    "ext.delta_ships_planets",
    "ext.delta_planets",
    "ext.projected_100",
]

ENEMY_AGG_NAMES = [
    "4p.enemy.sum.cur.ships_planets",
    "4p.enemy.sum.cur.ships_flying",
    "4p.enemy.sum.cur.ships_total",
    "4p.enemy.sum.cur.planets",
    "4p.enemy.sum.cur.prod_total",
    "4p.enemy.sum.cur.n_static",
    "4p.enemy.sum.cur.n_orbit",
    "4p.enemy.sum.cur.n_comet",
    "4p.enemy.sum.cur.neutrals_closer",
    "4p.enemy.sum.cur.enemies_closer",
    "4p.enemy.sum.ext.ships_planets",
    "4p.enemy.sum.ext.planets",
    "4p.enemy.sum.ext.prod_total",
    "4p.enemy.best.cur.ships_total",
    "4p.enemy.best.cur.prod_total",
    "4p.enemy.best.ext.projected_100",
    "4p.enemy.mean.cur.ships_total",
    "4p.enemy.mean.cur.prod_total",
    "4p.enemy.max.threat_score",
    "4p.enemy.sum.threat_score",
    "4p.enemy.max.pressure_to_me",
    "4p.enemy.sum.pressure_to_me",
]

OPP_REL_BLOCK = [
    "present",
    "rank",
    "threat_score",
    "nearest_dist_now",
    "nearest_dist_ext",
    "my_to_opp_pressure",
    "opp_to_my_pressure",
    "cur.ships_total_diff",
    "cur.ships_total_share",
    "cur.prod_diff",
    "cur.prod_share",
    "cur.planet_diff",
    "cur.static_diff",
    "cur.orbit_diff",
    "cur.comet_diff",
    "cur.flying_diff",
    "cur.adv_25",
    "cur.adv_50",
    "cur.adv_100",
    "cur.adv_remaining",
    "ext.ships_diff",
    "ext.prod_diff",
    "ext.planet_diff",
    "ext.adv_100",
]

RANK_NAMES = [
    "4p.rank.ships_total",
    "4p.rank.prod_total",
    "4p.rank.planets",
    "4p.rank.adv50",
    "4p.rank.is_ship_leader",
    "4p.rank.is_prod_leader",
    "4p.rank.is_planet_leader",
    "4p.rank.is_adv50_leader",
    "4p.gap.ships_to_leader",
    "4p.gap.prod_to_leader",
    "4p.gap.planets_to_leader",
    "4p.gap.adv50_to_leader",
    "4p.gap.adv50_to_second",
    "4p.gap.adv50_over_last",
    "4p.field.ships_total_diff",
    "4p.field.prod_diff",
    "4p.field.planet_diff",
    "4p.field.adv50_diff",
    "4p.field.adv100_diff",
    "4p.field.contested_neutral_balance",
]

V2_EXTRA_NAMES = []
for prefix in ("4p.opp1.pair", "4p.opp2.pair", "4p.opp3.pair"):
    V2_EXTRA_NAMES.extend(
        [
            f"{prefix}.adv_25",
            f"{prefix}.adv_50",
            f"{prefix}.adv_100",
            f"{prefix}.adv_remaining",
            f"{prefix}.ext_adv_100",
            f"{prefix}.p_above_50",
            f"{prefix}.p_above_100",
            f"{prefix}.pressure_balance",
            f"{prefix}.pressure_balance_share",
        ]
    )

V2_AGG_NAMES = [
    "4p.pair.mean_p_above_50",
    "4p.pair.mean_p_above_100",
    "4p.pair.n_above_50",
    "4p.pair.n_above_100",
    "4p.pair.worst_adv_50",
    "4p.pair.worst_adv_100",
    "4p.pair.best_adv_50",
    "4p.pair.mean_adv_50",
    "4p.pair.mean_ext_adv_100",
    "4p.pair.mean_pressure_balance",
    "4p.exposure.incoming_pressure",
    "4p.exposure.outgoing_pressure",
    "4p.exposure.dogpile_pressure",
    "4p.exposure.leader_dogpile_pressure",
    "4p.gap.adv50_to_relevant_next",
]


def feature_names() -> list[str]:
    names = list(GLOBAL_NAMES)
    for prefix in ("4p.me", "4p.opp1", "4p.opp2", "4p.opp3"):
        names.extend(f"{prefix}.{name}" for name in PLAYER_BLOCK)
    names.extend(ENEMY_AGG_NAMES)
    for prefix in ("4p.opp1.rel", "4p.opp2.rel", "4p.opp3.rel"):
        names.extend(f"{prefix}.{name}" for name in OPP_REL_BLOCK)
    names.extend(RANK_NAMES)
    return names


def feature_names_v2() -> list[str]:
    return feature_names() + V2_EXTRA_NAMES + V2_AGG_NAMES


FOURP_V1_NAMES = feature_names()
FOURP_V1_DIM = len(FOURP_V1_NAMES)
FOURP_V2_NAMES = feature_names_v2()
FOURP_V2_DIM = len(FOURP_V2_NAMES)
