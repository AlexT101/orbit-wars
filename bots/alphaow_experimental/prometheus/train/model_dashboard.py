"""Compact model/feature diagnostics for training scripts.

The terminal dashboard is intentionally dense: it is meant to sit next to
training output and answer "is this model better, calibrated, and which
features are carrying it?" without opening a notebook.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from engineered_features import ENGINEERED_NAMES
except ImportError:
    ENGINEERED_NAMES = []


# Feature naming is deliberately centralized here. When you add, remove, or
# reorder feature columns, update the relevant *_NAMES list plus either
# FEATURE_METADATA (reused summary metric names) or FEATURE_OVERRIDES
# (one-off/extras columns). The rest of the dashboard reads from this table.
SUMMARY_V2_NAMES = [
    "me.cur.ships_planets",
    "me.cur.ships_flying",
    "me.cur.n_static",
    "me.cur.n_orbit",
    "me.cur.n_comet",
    "me.cur.prod_static",
    "me.cur.prod_orbit",
    "me.cur.prod_comet",
    "me.cur.neutrals_closer",
    "me.cur.enemies_closer",
    "opp.cur.ships_planets",
    "opp.cur.ships_flying",
    "opp.cur.n_static",
    "opp.cur.n_orbit",
    "opp.cur.n_comet",
    "opp.cur.prod_static",
    "opp.cur.prod_orbit",
    "opp.cur.prod_comet",
    "opp.cur.neutrals_closer",
    "opp.cur.enemies_closer",
    "me.ext.ships_planets",
    "me.ext.n_static",
    "me.ext.n_orbit",
    "me.ext.n_comet",
    "me.ext.prod_static",
    "me.ext.prod_orbit",
    "me.ext.prod_comet",
    "me.ext.neutrals_closer",
    "me.ext.enemies_closer",
    "opp.ext.ships_planets",
    "opp.ext.n_static",
    "opp.ext.n_orbit",
    "opp.ext.n_comet",
    "opp.ext.prod_static",
    "opp.ext.prod_orbit",
    "opp.ext.prod_comet",
    "opp.ext.neutrals_closer",
    "opp.ext.enemies_closer",
    "neutral.ships",
    "neutral.n_static",
    "neutral.n_orbit",
    "neutral.n_comet",
    "neutral.prod_static",
    "neutral.prod_orbit",
    "neutral.prod_comet",
    "neutral.comet_time_left",
]

EXTRA_16_NAMES = [
    "extra.tick",
    "extra.tick_frac",
    "extra.n_steps",
    "extra.time_remaining",
    "extra.sends_last_5_me",
    "extra.sends_last_5_opp",
    "extra.sends_total_me",
    "extra.sends_total_opp",
    "extra.ships_delta_me_5",
    "extra.ships_delta_opp_5",
    "extra.fleets_in_flight_me",
    "extra.fleets_in_flight_opp",
    "extra.nearest_enemy_planet",
    "extra.frontline_dist",
    "extra.angular_velocity",
    "extra.ship_dominance",
]

EXTRA_12_NAMES = [
    "extra.tick",
    "extra.now_ss",
    "extra.now_so",
    "extra.now_os",
    "extra.now_oo",
    "extra.ext_ss",
    "extra.ext_so",
    "extra.ext_os",
    "extra.ext_oo",
    "extra.n_static",
    "extra.n_orbit",
    "extra.angular_velocity",
]

FEATURE_METADATA = {
    # summary_v2 current/extrapolated per-player blocks
    "ships_planets": ("ships on planets", "Ships currently stationed on owned planets."),
    "ships_flying": ("ships flying", "Ships currently committed to fleets in flight."),
    "n_static": ("static planets", "Count of owned non-orbiting, non-comet planets."),
    "n_orbit": ("orbiting planets", "Count of owned orbiting planets."),
    "n_comet": ("comets", "Count of owned comet planets."),
    "prod_static": ("static production", "Total production from owned static planets."),
    "prod_orbit": ("orbit production", "Total production from owned orbiting planets."),
    "prod_comet": ("comet production", "Total production from owned comets."),
    "neutrals_closer": ("nearby neutrals", "Neutral planets closer to this side than to the opposing side."),
    "enemies_closer": ("reachable enemies", "Enemy planets closer to this side than to other enemies."),
    # neutral summary_v2 block
    "ships": ("neutral ships", "Total ships on neutral planets."),
    "comet_time_left": ("comet time left", "Total remaining lifetime across comet planets."),
}

FEATURE_OVERRIDES = {
    # Extra features. Add or edit rows here when the feature layout changes.
    "extra.tick": ("Time", "tick", "Current replay/game tick."),
    "extra.tick_frac": ("Time", "game progress", "Current tick divided by total game length."),
    "extra.n_steps": ("Time", "game length", "Total number of ticks in the source game."),
    "extra.time_remaining": ("Time", "time remaining", "Ticks remaining in the source game."),
    "extra.sends_last_5_me": ("Actions", "my sends, last 5", "My fleet launches during the last five ticks."),
    "extra.sends_last_5_opp": ("Actions", "opp sends, last 5", "Opponent fleet launches during the last five ticks."),
    "extra.sends_total_me": ("Actions", "my sends total", "My cumulative fleet launches so far."),
    "extra.sends_total_opp": ("Actions", "opp sends total", "Opponent cumulative fleet launches so far."),
    "extra.ships_delta_me_5": ("Momentum", "my ship delta 5", "My ship total change over the last five ticks."),
    "extra.ships_delta_opp_5": ("Momentum", "opp ship delta 5", "Opponent ship total change over the last five ticks."),
    "extra.fleets_in_flight_me": ("Fleets", "my fleets in flight", "Count of my currently in-flight fleets."),
    "extra.fleets_in_flight_opp": ("Fleets", "opp fleets in flight", "Count of opponent currently in-flight fleets."),
    "extra.nearest_enemy_planet": ("Geometry", "nearest enemy", "Nearest distance between one of my planets and an enemy planet."),
    "extra.frontline_dist": ("Geometry", "frontline distance", "Minimum planet-to-enemy-planet distance."),
    "extra.angular_velocity": ("Map", "angular velocity", "Current rotation speed of inner planets."),
    "extra.ship_dominance": ("Balance", "ship dominance", "(my ships - opponent ships) divided by total owned/flying ships."),
    "extra.now_ss": ("Geometry now", "static -> static", "Nearest current distance from my static planets to enemy static planets."),
    "extra.now_so": ("Geometry now", "static -> orbit", "Nearest current distance from my static planets to enemy orbiting planets."),
    "extra.now_os": ("Geometry now", "orbit -> static", "Nearest current distance from my orbiting planets to enemy static planets."),
    "extra.now_oo": ("Geometry now", "orbit -> orbit", "Nearest current distance from my orbiting planets to enemy orbiting planets."),
    "extra.ext_ss": ("Geometry extrapolated", "ext static -> static", "Nearest static-to-static distance after extrapolated ownership."),
    "extra.ext_so": ("Geometry extrapolated", "ext static -> orbit", "Nearest static-to-orbit distance after extrapolated ownership."),
    "extra.ext_os": ("Geometry extrapolated", "ext orbit -> static", "Nearest orbit-to-static distance after extrapolated ownership."),
    "extra.ext_oo": ("Geometry extrapolated", "ext orbit -> orbit", "Nearest orbit-to-orbit distance after extrapolated ownership."),
    "extra.n_static": ("Map", "static planet count", "Count of non-comet, non-orbiting planets on the board."),
    "extra.n_orbit": ("Map", "orbiting planet count", "Count of orbiting planets on the board, excluding comets."),
}


def feature_names(dim: int, names: list[str] | None = None) -> list[str]:
    if names is not None:
        out = list(names)
    elif dim == 46:
        out = list(SUMMARY_V2_NAMES)
    elif dim == 58:
        out = list(SUMMARY_V2_NAMES) + list(EXTRA_12_NAMES)
    elif dim == 58 + len(ENGINEERED_NAMES):
        out = list(SUMMARY_V2_NAMES) + list(EXTRA_12_NAMES) + list(ENGINEERED_NAMES)
    elif dim == 62:
        out = list(SUMMARY_V2_NAMES) + list(EXTRA_16_NAMES)
    else:
        out = []
    if len(out) < dim:
        out.extend(f"f{i}" for i in range(len(out), dim))
    return out[:dim]


def feature_info(dim: int, names: list[str] | None = None) -> list[dict[str, str | int]]:
    return [_describe_feature(i, name) for i, name in enumerate(feature_names(dim, names))]


def _describe_feature(index: int, name: str) -> dict[str, str | int]:
    if name in FEATURE_OVERRIDES:
        group, label, description = FEATURE_OVERRIDES[name]
        return {"index": index, "name": name, "label": label, "group": group, "description": description}
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "eng":
        group, label, description = _describe_engineered(parts[1], "_".join(parts[2:]))
        return {"index": index, "name": name, "label": label, "group": group, "description": description}
    if len(parts) >= 3 and parts[0] in {"me", "opp"}:
        side = "Me" if parts[0] == "me" else "Opponent"
        phase = "current" if parts[1] == "cur" else "extrapolated"
        metric = "_".join(parts[2:])
        label, desc = FEATURE_METADATA.get(metric, (metric.replace("_", " "), metric.replace("_", " ")))
        return {
            "index": index,
            "name": name,
            "label": f"{side.lower()} {phase} {label}",
            "group": f"{side} {phase}",
            "description": f"{side} {phase} block: {desc}",
        }
    if len(parts) >= 2 and parts[0] == "neutral":
        metric = "_".join(parts[1:])
        label, desc = FEATURE_METADATA.get(metric, (metric.replace("_", " "), metric.replace("_", " ")))
        return {
            "index": index,
            "name": name,
            "label": label,
            "group": "Neutral",
            "description": f"Neutral block: {desc}",
        }
    return {"index": index, "name": name, "label": name, "group": "Feature", "description": f"Feature column {index}."}


def _describe_engineered(scope: str, metric: str) -> tuple[str, str, str]:
    scope_label = {
        "cur": "Engineered current",
        "ext": "Engineered extrapolated",
        "pending": "Engineered pending",
        "geo": "Engineered geometry",
        "time": "Engineered time",
        "forecast": "Engineered forecast",
        "margin": "Engineered margin",
        "speed": "Engineered speed",
        "comet": "Engineered comet",
        "map": "Engineered map",
        "phase": "Engineered phase",
        "strategy": "Engineered strategy",
        "area": "Engineered area",
        "density": "Engineered density",
    }.get(scope, "Engineered")
    labels = {
        "ships_planets_diff": ("ships on planets diff", "My ships on planets minus opponent ships on planets."),
        "ships_flying_diff": ("ships flying diff", "My in-flight ships minus opponent in-flight ships."),
        "ships_total_diff": ("total ships diff", "My planet plus flying ships minus opponent planet plus flying ships."),
        "ships_total_share": ("total ships share", "Total ship difference normalized by both sides' total ships."),
        "static_count_diff": ("static planets diff", "My static planet count minus opponent static planet count."),
        "orbit_count_diff": ("orbiting planets diff", "My orbiting planet count minus opponent orbiting planet count."),
        "comet_count_diff": ("comets diff", "My comet count minus opponent comet count."),
        "planet_count_diff": ("planet count diff", "My total owned planet count minus opponent total owned planet count."),
        "prod_static_diff": ("static production diff", "My static production minus opponent static production."),
        "prod_orbit_diff": ("orbit production diff", "My orbit production minus opponent orbit production."),
        "prod_comet_diff": ("comet production diff", "My comet production minus opponent comet production."),
        "prod_total_diff": ("total production diff", "My total production minus opponent total production."),
        "prod_total_share": ("production share", "Total production difference normalized by both sides' total production."),
        "ships_swing": ("pending ship swing", "Extrapolated ship diff minus current total ship diff."),
        "static_count_swing": ("pending static swing", "Extrapolated static planet diff minus current static planet diff."),
        "orbit_count_swing": ("pending orbit swing", "Extrapolated orbiting planet diff minus current orbiting planet diff."),
        "prod_total_swing": ("pending production swing", "Extrapolated production diff minus current production diff."),
        "now_min_dist": ("nearest current enemy", "Nearest current distance between any owned static/orbit bucket and enemy bucket."),
        "ext_min_dist": ("nearest extrapolated enemy", "Nearest distance after resolving in-flight fleets."),
        "dist_change": ("nearest distance change", "Extrapolated nearest enemy distance minus current nearest enemy distance."),
        "now_mixed_gap": ("current mixed gap", "Mixed static/orbit distance minus same-type static/orbit distance, current ownership."),
        "ext_mixed_gap": ("extrapolated mixed gap", "Mixed static/orbit distance minus same-type static/orbit distance, extrapolated ownership."),
        "tick_frac": ("game progress", "Current tick divided by the 500-turn episode length."),
        "remaining_frac": ("game remaining", "Fraction of the 500-turn episode remaining."),
        "early_phase": ("early phase", "One when tick is in the first sixth of the game."),
        "transition_phase": ("transition phase", "One when tick is between one sixth and one third of the game."),
        "midgame_phase": ("midgame phase", "One when tick is between one third and two thirds of the game."),
        "endgame_phase": ("endgame phase", "One when tick is in the last third of the game."),
        "cur_prod_remaining": ("current production remaining", "Current production diff multiplied by configured turns remaining."),
        "cur_prod_100": ("current production next 100", "Current production diff multiplied by up to the next 100 configured turns."),
        "cur_adv_remaining": ("current projected advantage", "Current ship diff plus production diff over configured turns remaining."),
        "cur_adv_100": ("current projected 100", "Current ship diff plus production diff over up to the next 100 turns."),
        "ext_prod_remaining": ("extrapolated production remaining", "Extrapolated production diff multiplied by configured turns remaining."),
        "ext_adv_remaining": ("extrapolated projected advantage", "Extrapolated ship diff plus production diff over configured turns remaining."),
        "cur_prod_remaining_frac": ("production share remaining", "Current production share weighted by configured game time remaining."),
        "cur_ships_elapsed_frac": ("ship share elapsed", "Current ship share weighted by configured game progress."),
        "flying_total": ("ships flying total", "Total ships currently committed to fleets by both sides."),
        "my_flying_commitment": ("my flying commitment", "My in-flight ships divided by my in-flight plus stationed ships."),
        "opp_flying_commitment": ("opp flying commitment", "Opponent in-flight ships divided by opponent in-flight plus stationed ships."),
        "flying_commitment_diff": ("flying commitment diff", "My flying commitment minus opponent flying commitment."),
        "cur_ship_pressure": ("current ship pressure", "Current total ship diff divided by nearest current enemy distance."),
        "cur_prod_pressure": ("current production pressure", "Current production diff divided by nearest current enemy distance."),
        "ext_ship_pressure": ("extrapolated ship pressure", "Extrapolated ship diff divided by nearest extrapolated enemy distance."),
        "ext_prod_pressure": ("extrapolated production pressure", "Extrapolated production diff divided by nearest extrapolated enemy distance."),
        "cur_prod_25": ("current production next 25", "Current production diff multiplied by up to the next 25 turns."),
        "cur_prod_50": ("current production next 50", "Current production diff multiplied by up to the next 50 turns."),
        "cur_adv_25": ("current projected 25", "Current ship diff plus production diff over up to the next 25 turns."),
        "cur_adv_50": ("current projected 50", "Current ship diff plus production diff over up to the next 50 turns."),
        "cur_adv_150": ("current projected 150", "Current ship diff plus production diff over up to the next 150 turns."),
        "ext_adv_100": ("extrapolated projected 100", "Extrapolated ship diff plus production diff over up to the next 100 turns."),
        "cur_adv_100_log": ("current projected 100 log", "Signed log transform of current projected 100-turn advantage."),
        "cur_adv_remaining_log": ("current projected remaining log", "Signed log transform of projected configured-game remaining advantage."),
        "ships_total_diff_log": ("total ships diff log", "Signed log transform of current total ship diff."),
        "prod_total_diff_log": ("production diff log", "Signed log transform of current production diff."),
        "my_stationed_speed": ("my stationed speed", "Estimated fleet speed if all my stationed ships were launched as one fleet."),
        "opp_stationed_speed": ("opp stationed speed", "Estimated fleet speed if all opponent stationed ships were launched as one fleet."),
        "stationed_speed_diff": ("stationed speed diff", "My estimated stationed fleet speed minus opponent estimated stationed speed."),
        "my_total_speed": ("my total speed", "Estimated fleet speed from my stationed plus in-flight ship total."),
        "opp_total_speed": ("opp total speed", "Estimated fleet speed from opponent stationed plus in-flight ship total."),
        "total_speed_diff": ("total speed diff", "My total estimated fleet speed minus opponent total estimated speed."),
        "now_travel_turns_my": ("my nearest travel now", "Nearest current enemy distance divided by estimated my stationed fleet speed."),
        "ext_travel_turns_my": ("my nearest travel extrapolated", "Nearest extrapolated enemy distance divided by estimated my stationed fleet speed."),
        "now_travel_turns_opp": ("opp nearest travel now", "Nearest current enemy distance divided by estimated opponent stationed fleet speed."),
        "ticks_to_next_spawn": ("ticks to comet spawn", "Configured turns until the next comet group spawns."),
        "next_spawn_frac": ("comet spawn fraction", "Ticks to next comet spawn divided by configured episode length."),
        "spawn_soon_25": ("comet spawn soon 25", "One when a comet group spawns within 25 configured turns."),
        "spawn_soon_50": ("comet spawn soon 50", "One when a comet group spawns within 50 configured turns."),
        "orbit_rotation_pressure": ("orbit rotation pressure", "Current orbiting planet count diff multiplied by angular velocity."),
        "orbit_count_rotation": ("orbit count rotation", "Current total orbiting planet count multiplied by angular velocity."),
        "angular_velocity_scaled": ("angular velocity scaled", "Angular velocity multiplied by 100 for easier tree thresholds."),
        "transition_adv_50": ("transition projected 50", "Projected 50-turn advantage, only active from one sixth to one third of the game."),
        "transition_adv_100": ("transition projected 100", "Projected 100-turn advantage, only active from one sixth to one third of the game."),
        "transition_flying_diff": ("transition flying diff", "In-flight ship difference, only active from one sixth to one third of the game."),
        "transition_speed_diff": ("transition speed diff", "Estimated total fleet-speed difference, only active from one sixth to one third of the game."),
        "transition_distance": ("transition nearest enemy", "Nearest current enemy distance, only active from one sixth to one third of the game."),
        "early_prod_share": ("early production share", "Production share, only active in the first sixth of the game."),
        "midgame_ship_share": ("midgame ship share", "Total ship share, only active from one third to two thirds of the game."),
        "endgame_ship_share": ("endgame ship share", "Total ship share, only active in the final third of the game."),
        "prod_payback_turns": ("production payback turns", "Turns for positive production diff to cover a current ship deficit; clipped at 500."),
        "ship_deficit_prod_cover": ("ship deficit production cover", "Current ship deficit divided by positive production diff."),
        "prod_adv_ship_deficit": ("economy ahead, ships behind", "Positive production share multiplied by current ship-share deficit."),
        "ship_adv_prod_deficit": ("ships ahead, economy behind", "Positive ship share multiplied by current production-share deficit."),
        "balanced_trade_value": ("balanced trade value", "Production share weighted toward ship-balanced positions where trading can matter most."),
        "neutral_closer_diff": ("nearby neutral diff", "Neutral-closer count for me minus opponent."),
        "enemy_reach_diff": ("reachable enemy diff", "Enemy-reachable count for me minus opponent."),
        "frontier_area_adv": ("frontier area advantage", "Planet-count, neutral-closeness, and enemy-reach count advantage combined."),
        "neutral_closer_prod_share": ("neutral area economy", "Neutral-closeness advantage multiplied by production share."),
        "enemy_reach_ship_share": ("enemy reach ship share", "Enemy-reach advantage multiplied by ship share."),
        "my_ships_per_planet": ("my ships per planet", "My stationed plus flying ships divided by current owned planet count."),
        "opp_ships_per_planet": ("opp ships per planet", "Opponent stationed plus flying ships divided by current owned planet count."),
        "ships_per_planet_diff": ("ships per planet diff", "My ships per planet minus opponent ships per planet."),
        "my_prod_per_planet": ("my production density", "My production divided by current owned planet count."),
        "opp_prod_per_planet": ("opp production density", "Opponent production divided by current owned planet count."),
        "prod_per_planet_diff": ("production density diff", "My production density minus opponent production density."),
        "my_flying_per_planet": ("my flying per planet", "My in-flight ships divided by current owned planet count."),
        "opp_flying_per_planet": ("opp flying per planet", "Opponent in-flight ships divided by current owned planet count."),
    }
    label, description = labels.get(metric, (metric.replace("_", " "), f"Engineered {scope} feature: {metric.replace('_', ' ')}."))
    return scope_label, label, description


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)


def binary_metrics(y_true: np.ndarray, pred_prob: np.ndarray) -> dict[str, float | int]:
    yb = (y_true > 0).astype(np.int32)
    p = _clip_prob(pred_prob)
    hard = (p >= 0.5).astype(np.int32)
    tp = int(((hard == 1) & (yb == 1)).sum())
    tn = int(((hard == 0) & (yb == 0)).sum())
    fp = int(((hard == 1) & (yb == 0)).sum())
    fn = int(((hard == 0) & (yb == 1)).sum())
    logloss = float(-(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean())
    brier = float(((p - yb) ** 2).mean())
    return {
        "n": int(yb.shape[0]),
        "positive_rate": float(yb.mean()),
        "accuracy": float((hard == yb).mean()),
        "logloss": logloss,
        "brier": brier,
        "auc": auc_score(yb, p),
        "avg_confidence": float(np.abs(p - 0.5).mean() * 2.0),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def auc_score(yb: np.ndarray, score: np.ndarray) -> float:
    pos = int(yb.sum())
    neg = int(yb.shape[0] - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_score = score[order]
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = float(ranks[yb == 1].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def calibration_bins(y_true: np.ndarray, pred_prob: np.ndarray, bins: int = 5) -> list[dict[str, float | int]]:
    yb = (y_true > 0).astype(np.float64)
    p = _clip_prob(pred_prob)
    out = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (p >= lo) & ((p <= hi) if i + 1 == bins else (p < hi))
        n = int(mask.sum())
        if n == 0:
            out.append({"lo": float(lo), "hi": float(hi), "n": 0, "pred": float("nan"), "actual": float("nan")})
            continue
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "pred": float(p[mask].mean()),
                "actual": float(yb[mask].mean()),
            }
        )
    return out


def phase_accuracy_buckets(
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    X: np.ndarray,
    names: list[str],
    *,
    phase_frac: np.ndarray | None = None,
    episode_steps: float = 500.0,
) -> list[dict[str, float | int | str]]:
    yb = (y_true > 0).astype(np.int32)
    p = _clip_prob(pred_prob)
    hard = (p >= 0.5).astype(np.int32)

    frac = None
    source = "none"
    if phase_frac is not None:
        frac = phase_frac.astype(np.float64, copy=False)
        source = "observed_game_progress"
    elif "extra.tick_frac" in names:
        frac = X[:, names.index("extra.tick_frac")].astype(np.float64, copy=False)
        source = "extra.tick_frac"
    elif "extra.tick" in names and "extra.n_steps" in names:
        ticks = X[:, names.index("extra.tick")].astype(np.float64, copy=False)
        steps = np.maximum(1.0, X[:, names.index("extra.n_steps")].astype(np.float64, copy=False))
        frac = ticks / steps
        source = "extra.tick/extra.n_steps"
    elif "extra.tick" in names:
        ticks = X[:, names.index("extra.tick")].astype(np.float64, copy=False)
        frac = ticks / max(1.0, float(episode_steps))
        source = f"extra.tick/{int(episode_steps)}"

    phases = [
        ("early", 0.0, 1.0 / 6.0),
        ("transition", 1.0 / 6.0, 1.0 / 3.0),
        ("midgame", 1.0 / 3.0, 2.0 / 3.0),
        ("endgame", 2.0 / 3.0, 1.0),
    ]
    if frac is None:
        return [
            {
                "phase": name,
                "lo": float(lo),
                "hi": float(hi),
                "n": 0,
                "pred": float("nan"),
                "actual": float("nan"),
                "gap": float("nan"),
                "accuracy": float("nan"),
                "pos_pred_n": 0,
                "pos_pred_accuracy": float("nan"),
                "neg_pred_n": 0,
                "neg_pred_accuracy": float("nan"),
                "source": source,
            }
            for name, lo, hi in phases
        ]

    frac = np.clip(frac, 0.0, 1.0)
    out = []
    for i, (name, lo, hi) in enumerate(phases):
        mask = (frac >= lo) & ((frac <= hi) if i + 1 == len(phases) else (frac < hi))
        n = int(mask.sum())
        if n == 0:
            out.append(
                {
                    "phase": name,
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": 0,
                    "pred": float("nan"),
                    "actual": float("nan"),
                    "gap": float("nan"),
                    "accuracy": float("nan"),
                    "pos_pred_n": 0,
                    "pos_pred_accuracy": float("nan"),
                    "neg_pred_n": 0,
                    "neg_pred_accuracy": float("nan"),
                    "source": source,
                }
            )
            continue
        pos_mask = mask & (hard == 1)
        neg_mask = mask & (hard == 0)
        pos_n = int(pos_mask.sum())
        neg_n = int(neg_mask.sum())
        actual = float(yb[mask].mean())
        pred = float(p[mask].mean())
        out.append(
            {
                "phase": name,
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "pred": pred,
                "actual": actual,
                "gap": float(actual - pred),
                "accuracy": float((hard[mask] == yb[mask]).mean()),
                "pos_pred_n": pos_n,
                "pos_pred_accuracy": float(yb[pos_mask].mean()) if pos_n else float("nan"),
                "neg_pred_n": neg_n,
                "neg_pred_accuracy": float((yb[neg_mask] == 0).mean()) if neg_n else float("nan"),
                "source": source,
            }
        )
    return out


def xgb_importance(booster, dim: int) -> dict[str, np.ndarray]:
    out = {}
    for typ in ("gain", "weight", "cover", "total_gain"):
        raw = booster.get_score(importance_type=typ)
        vals = np.zeros(dim, dtype=np.float64)
        for key, val in raw.items():
            if key.startswith("f"):
                try:
                    idx = int(key[1:])
                except ValueError:
                    continue
                if 0 <= idx < dim:
                    vals[idx] = float(val)
        out[typ] = vals
    return out


def feature_correlations(X: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y = (y_true > 0).astype(np.float64)
    yc = y - y.mean()
    yden = math.sqrt(float((yc * yc).sum()))
    vals = np.zeros(X.shape[1], dtype=np.float64)
    if yden < 1e-12:
        return vals
    for i in range(X.shape[1]):
        v = X[:, i].astype(np.float64)
        vc = v - v.mean()
        den = math.sqrt(float((vc * vc).sum())) * yden
        vals[i] = 0.0 if den < 1e-12 else float((vc * yc).sum() / den)
    return vals


def univariate_best_acc(X: np.ndarray, y_true: np.ndarray, max_rows: int = 20000, seed: int = 123) -> np.ndarray:
    y = (y_true > 0).astype(np.int32)
    Xs, ys = sample_rows(X, y, max_rows, seed)
    out = np.zeros(X.shape[1], dtype=np.float64)
    n = ys.shape[0]
    total_pos = int(ys.sum())
    total_neg = n - total_pos
    for i in range(Xs.shape[1]):
        order = np.argsort(Xs[:, i], kind="mergesort")
        yy = ys[order]
        cum_pos = np.cumsum(yy)
        seen = np.arange(1, n + 1)
        left_pos = cum_pos
        left_neg = seen - left_pos
        right_pos = total_pos - left_pos
        right_neg = total_neg - left_neg
        # Rule A: low -> negative, high -> positive.
        acc_hi_pos = (left_neg + right_pos).max() / n
        # Rule B: low -> positive, high -> negative.
        acc_hi_neg = (left_pos + right_neg).max() / n
        out[i] = max(float(acc_hi_pos), float(acc_hi_neg))
    return out


def sample_rows(X: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    return X[idx], y[idx]


def permutation_importance(
    X: np.ndarray,
    y_true: np.ndarray,
    pred_fn: Callable[[np.ndarray], np.ndarray],
    max_rows: int = 20000,
    seed: int = 123,
) -> dict[str, np.ndarray | float | int]:
    y = y_true.astype(np.float32)
    Xs, ys = sample_rows(X, y, max_rows, seed)
    rng = np.random.default_rng(seed + 1)
    base = pred_fn(Xs)
    base_m = binary_metrics(ys, base)
    acc_drop = np.zeros(X.shape[1], dtype=np.float64)
    logloss_delta = np.zeros(X.shape[1], dtype=np.float64)
    work = Xs.copy()
    for i in range(X.shape[1]):
        old = work[:, i].copy()
        work[:, i] = rng.permutation(work[:, i])
        m = binary_metrics(ys, pred_fn(work))
        acc_drop[i] = float(base_m["accuracy"]) - float(m["accuracy"])
        logloss_delta[i] = float(m["logloss"]) - float(base_m["logloss"])
        work[:, i] = old
    return {
        "sample_rows": int(Xs.shape[0]),
        "base_accuracy": float(base_m["accuracy"]),
        "base_logloss": float(base_m["logloss"]),
        "acc_drop": acc_drop,
        "logloss_delta": logloss_delta,
    }


def _bar(value: float, width: int = 10) -> str:
    v = max(0.0, min(1.0, value))
    n = int(round(v * width))
    return "#" * n + "." * (width - n)


def _fmt_pct(x: float, width: int = 6) -> str:
    if math.isnan(x):
        return " " * (width - 3) + "nan"
    return f"{100.0 * x:{width}.2f}"


def training_curve_from_xgb(evals_result: dict | None, best_iteration: int | None = None) -> list[dict[str, float | int]]:
    if not evals_result:
        return []

    def metric(split: str, name: str) -> list[float]:
        vals = evals_result.get(split, {}).get(name, [])
        return [float(v) for v in vals]

    train_logloss = metric("train", "logloss")
    val_logloss = metric("val", "logloss")
    train_error = metric("train", "error")
    val_error = metric("val", "error")
    n = max(len(train_logloss), len(val_logloss), len(train_error), len(val_error), 0)
    out = []
    for i in range(n):
        row: dict[str, float | int] = {"iteration": i}
        if i < len(train_logloss):
            row["train_logloss"] = train_logloss[i]
        if i < len(val_logloss):
            row["val_logloss"] = val_logloss[i]
        if i < len(train_error):
            row["train_accuracy"] = 1.0 - train_error[i]
        if i < len(val_error):
            row["val_accuracy"] = 1.0 - val_error[i]
        if best_iteration is not None and i == int(best_iteration):
            row["best"] = 1
        out.append(row)
    return out


def render_xgb_dashboard(
    *,
    title: str,
    booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pred_prob: np.ndarray,
    feature_name_list: list[str] | None = None,
    phase_frac: np.ndarray | None = None,
    top_n: int = 18,
    permutation: bool = True,
    permutation_rows: int = 20000,
    training_curve: list[dict] | None = None,
    json_out: str | Path | None = None,
    html_out: str | Path | None = None,
) -> dict:
    names = feature_names(X_val.shape[1], feature_name_list)
    infos = feature_info(X_val.shape[1], feature_name_list)
    metrics = binary_metrics(y_val, pred_prob)
    cal = calibration_bins(y_val, pred_prob)
    phases = phase_accuracy_buckets(y_val, pred_prob, X_val, names, phase_frac=phase_frac)
    imp = xgb_importance(booster, X_val.shape[1])
    gain = imp["gain"]
    weight = imp["weight"]
    gain_share = gain / gain.sum() if gain.sum() > 0 else gain
    weight_share = weight / weight.sum() if weight.sum() > 0 else weight
    corr = feature_correlations(X_val, y_val)
    solo = univariate_best_acc(X_val, y_val, max_rows=permutation_rows)

    perm = None
    if permutation:
        try:
            import xgboost as xgb

            def pred_fn(x: np.ndarray) -> np.ndarray:
                return booster.predict(xgb.DMatrix(x.astype(np.float32, copy=False)))

            perm = permutation_importance(X_val, y_val, pred_fn, max_rows=permutation_rows)
        except Exception as exc:  # keep training output alive if diagnostics fail
            perm = {"error": str(exc)}

    acc_drop = np.zeros(X_val.shape[1], dtype=np.float64)
    ll_delta = np.zeros(X_val.shape[1], dtype=np.float64)
    if isinstance(perm, dict) and "acc_drop" in perm:
        acc_drop = perm["acc_drop"]  # type: ignore[assignment]
        ll_delta = perm["logloss_delta"]  # type: ignore[assignment]

    score = gain_share + np.maximum(acc_drop, 0.0) * 8.0 + np.maximum(ll_delta, 0.0)
    order = np.argsort(-score)[:top_n]

    print(f"\n=== model dashboard: {title} ===")
    print(
        "rows={n:,} pos={pos}%  acc={acc}%  auc={auc}%  logloss={ll:.4f}  "
        "brier={br:.4f}  conf={conf}%  cm=[tp {tp} fp {fp} | fn {fn} tn {tn}]".format(
            n=int(metrics["n"]),
            pos=_fmt_pct(float(metrics["positive_rate"])).strip(),
            acc=_fmt_pct(float(metrics["accuracy"])).strip(),
            auc=_fmt_pct(float(metrics["auc"])).strip(),
            ll=float(metrics["logloss"]),
            br=float(metrics["brier"]),
            conf=_fmt_pct(float(metrics["avg_confidence"])).strip(),
            tp=int(metrics["tp"]),
            fp=int(metrics["fp"]),
            fn=int(metrics["fn"]),
            tn=int(metrics["tn"]),
        )
    )
    print("calibration  bin       n    pred   actual   gap")
    for b in cal:
        if int(b["n"]) == 0:
            print(f"             {b['lo']:.1f}-{b['hi']:.1f}      0      -      -      -")
        else:
            gap = float(b["actual"]) - float(b["pred"])
            print(
                f"             {b['lo']:.1f}-{b['hi']:.1f} {int(b['n']):6d} "
                f"{100*float(b['pred']):6.1f} {100*float(b['actual']):7.1f} {100*gap:+6.1f}"
            )
    print("phases       bucket       n    pred   actual   gap    acc   +acc   -acc")
    for b in phases:
        if int(b["n"]) == 0:
            print(f"             {b['phase']:10.10s}      0      -      -      -      -      -      -")
        else:
            print(
                f"             {str(b['phase']):10.10s} {int(b['n']):6d} "
                f"{100*float(b['pred']):6.1f} {100*float(b['actual']):7.1f} "
                f"{100*float(b['gap']):+6.1f} {100*float(b['accuracy']):6.1f} "
                f"{100*float(b['pos_pred_accuracy']):6.1f} {100*float(b['neg_pred_accuracy']):6.1f}"
            )

    if isinstance(perm, dict) and "sample_rows" in perm:
        print(
            f"importance  permutation_rows={perm['sample_rows']} "
            f"base_acc={100*float(perm['base_accuracy']):.2f}% base_logloss={float(perm['base_logloss']):.4f}"
        )
    elif isinstance(perm, dict) and "error" in perm:
        print(f"importance  permutation skipped: {perm['error']}")
    else:
        print("importance  permutation disabled")
    print(" rank feature                          gain%  split%  perm_acc  perm_ll  solo%   corr   gain")
    for rank, idx in enumerate(order, 1):
        print(
            f"{rank:5d} {names[idx]:30.30s} "
            f"{100*gain_share[idx]:6.2f} {100*weight_share[idx]:7.2f} "
            f"{100*acc_drop[idx]:+8.2f} {ll_delta[idx]:+8.4f} "
            f"{100*solo[idx]:6.2f} {corr[idx]:+7.3f} {_bar(float(gain_share[idx] / max(gain_share[order[0]], 1e-12)))}"
        )

    payload = {
        "kind": "model",
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "calibration": cal,
        "phase_buckets": phases,
        "training_curve": training_curve or [],
        "features": [
            {
                "index": int(i),
                "name": names[i],
                "label": str(infos[i]["label"]),
                "group": str(infos[i]["group"]),
                "description": str(infos[i]["description"]),
                "gain": float(gain[i]),
                "gain_share": float(gain_share[i]),
                "split_share": float(weight_share[i]),
                "permutation_acc_drop": float(acc_drop[i]),
                "permutation_logloss_delta": float(ll_delta[i]),
                "univariate_best_accuracy": float(solo[i]),
                "correlation": float(corr[i]),
            }
            for i in range(X_val.shape[1])
        ],
    }
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"dashboard_json={json_out}")
    if html_out:
        append_html_dashboard(html_out, payload)
        print(f"dashboard_html={html_out}")
    return payload


def append_html_dashboard(path: str | Path, record: dict) -> None:
    path = Path(path)
    records = _read_dashboard_records(path)
    records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_html(records), encoding="utf-8")


def append_eval_dashboard(path: str | Path, *, title: str, summaries: list[dict]) -> None:
    append_html_dashboard(
        path,
        {
            "kind": "eval",
            "title": title,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "summaries": summaries,
        },
    )


def _read_dashboard_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    start_tag = '<script id="ow-dashboard-data" type="application/json">'
    end_tag = "</script>"
    start = text.find(start_tag)
    if start < 0:
        return []
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end < 0:
        return []
    try:
        data = json.loads(text[start:end])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _render_html(records: list[dict]) -> str:
    data = json.dumps(records, default=_json_default, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Orbit Wars Training Dashboard</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #101418;
  --panel: #171d22;
  --panel2: #1d252b;
  --ink: #e9f0f3;
  --muted: #92a0a8;
  --line: #2d3942;
  --a: #74c0fc;
  --b: #ffd166;
  --c: #8ce99a;
  --d: #ff8787;
  --e: #b197fc;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 12px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
header {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; padding: 14px 18px; border-bottom: 1px solid var(--line); background: #0d1115; position: sticky; top: 0; z-index: 2; }}
h1 {{ font-size: 16px; margin: 0; letter-spacing: 0; }}
main {{ padding: 14px 18px 24px; display: grid; gap: 14px; }}
.muted {{ color: var(--muted); }}
.grid {{ display: grid; gap: 10px; }}
.cards {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
.label {{ color: var(--muted); font-size: 11px; }}
.value {{ font-size: 19px; font-weight: 700; margin-top: 2px; }}
.two {{ grid-template-columns: minmax(360px, 1.1fr) minmax(420px, 1.4fr); align-items: start; }}
section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
.section-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 9px 10px; border-bottom: 1px solid var(--line); background: var(--panel2); }}
h2 {{ font-size: 12px; margin: 0; text-transform: uppercase; letter-spacing: .08em; color: #cdd6dc; }}
select {{ background: #0d1115; color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; max-width: 100%; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 5px 7px; border-bottom: 1px solid #222c33; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 600; background: #12181d; position: sticky; top: 0; z-index: 1; }}
tr:hover td {{ background: #1a2329; }}
.scroll {{ max-height: 420px; overflow: auto; position: relative; }}
.barcell {{ min-width: 110px; }}
.bar {{ height: 8px; background: #26323a; border-radius: 999px; overflow: hidden; }}
.fill {{ height: 100%; background: var(--a); }}
.fill.b {{ background: var(--b); }}
.fill.c {{ background: var(--c); }}
svg {{ width: 100%; display: block; }}
.axis {{ stroke: #53616b; stroke-width: 1; }}
.lineA {{ fill: none; stroke: var(--a); stroke-width: 2; }}
.lineB {{ fill: none; stroke: var(--b); stroke-width: 2; }}
.lineTrain {{ fill: none; stroke: var(--a); stroke-width: 2; opacity: .55; }}
.lineEval {{ fill: none; stroke: var(--c); stroke-width: 2.5; }}
.lineLoss {{ fill: none; stroke: var(--b); stroke-width: 1.7; stroke-dasharray: 5 4; opacity: .75; }}
.dotA {{ fill: var(--a); }}
.dotB {{ fill: var(--b); }}
.dotTrain {{ fill: var(--a); opacity: .65; }}
.dotEval {{ fill: var(--c); }}
.dotLoss {{ fill: var(--b); opacity: .85; }}
.feature-name {{ display: grid; gap: 1px; min-width: 250px; }}
.feature-label {{ font-weight: 650; color: var(--ink); }}
.feature-tech {{ color: var(--muted); font-size: 11px; }}
.legend {{ display: flex; gap: 14px; align-items: center; color: var(--muted); }}
.sw {{ display:inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
.trend-note {{ padding: 0 10px 9px; color: var(--muted); font-size: 11px; }}
.empty {{ padding: 24px; color: var(--muted); }}
@media (max-width: 980px) {{ .two {{ grid-template-columns: 1fr; }} th {{ position: static; }} }}
</style>
</head>
<body>
<header>
  <h1>Orbit Wars Training Dashboard</h1>
  <div class="muted" id="stamp"></div>
</header>
<main>
  <div class="grid cards" id="cards"></div>
  <div class="grid two">
    <section>
      <div class="section-head"><h2>Boosting Curve</h2><div class="legend"><span><i class="sw" style="background:var(--a);opacity:.55"></i>train logloss</span><span><i class="sw" style="background:var(--b)"></i>val logloss</span><span><i class="sw" style="background:var(--c)"></i>val acc</span></div></div>
      <div id="trend"></div>
    </section>
    <section>
      <div class="section-head"><h2>Training Trials</h2></div>
      <div class="scroll"><table id="runs"></table></div>
    </section>
  </div>
  <section>
    <div class="section-head"><h2>Feature Importance</h2><select id="modelSelect"></select></div>
    <div class="scroll"><table id="features"></table></div>
  </section>
  <div class="grid two">
    <section>
      <div class="section-head"><h2>Calibration</h2></div>
      <div id="calibration"></div>
    </section>
    <section>
      <div class="section-head"><h2>Phase Buckets</h2></div>
      <div id="phaseBuckets"></div>
    </section>
  </div>
  <section>
      <div class="section-head"><h2>Eval Runs</h2></div>
      <div class="scroll"><table id="evals"></table></div>
  </section>
</main>
<script id="ow-dashboard-data" type="application/json">{data}</script>
<script>
const records = JSON.parse(document.getElementById('ow-dashboard-data').textContent);
const models = records.filter(r => r.kind === 'model');
const evals = records.filter(r => r.kind === 'eval');
const pct = v => Number.isFinite(v) ? (100*v).toFixed(2) + '%' : '-';
const num = (v, d=3) => Number.isFinite(v) ? (+v).toFixed(d) : '-';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
document.getElementById('stamp').textContent = `${{records.length}} records | updated ${{new Date().toLocaleString()}}`;

function latestModel() {{ return models[models.length - 1]; }}
function card(label, value, sub='') {{ return `<div class="card"><div class="label">${{label}}</div><div class="value">${{value}}</div><div class="muted">${{sub}}</div></div>`; }}
function renderCards() {{
  const m = latestModel();
  const lastEval = evals[evals.length - 1];
  let html = card('model runs', models.length, m ? esc(m.title) : '') + card('eval runs', evals.length, lastEval ? esc(lastEval.title) : '');
  if (m) {{
    html += card('latest acc', pct(m.metrics.accuracy), `auc ${{pct(m.metrics.auc)}}`);
    html += card('latest logloss', num(m.metrics.logloss, 4), `brier ${{num(m.metrics.brier, 4)}}`);
  }}
  if (lastEval && lastEval.summaries?.length) {{
    const avg = lastEval.summaries.reduce((a, x) => a + x.score, 0) / lastEval.summaries.length;
    html += card('eval avg score', pct(avg), `${{lastEval.summaries.length}} opponents`);
  }}
  document.getElementById('cards').innerHTML = html;
}}

function renderRuns() {{
  if (!models.length) {{ document.getElementById('runs').innerHTML = '<tr><td class="empty">No model records yet.</td></tr>'; return; }}
  let html = '<thead><tr><th>time</th><th>title</th><th>n</th><th>pos</th><th>acc</th><th>auc</th><th>logloss</th><th>brier</th><th>top feature</th></tr></thead><tbody>';
  models.forEach(r => {{
    const top = [...(r.features || [])].sort((a,b) => (b.gain_share + Math.max(0,b.permutation_acc_drop)*8) - (a.gain_share + Math.max(0,a.permutation_acc_drop)*8))[0];
    html += `<tr><td>${{esc(r.created_at?.slice(11) || '')}}</td><td>${{esc(r.title)}}</td><td>${{r.metrics.n.toLocaleString()}}</td><td>${{pct(r.metrics.positive_rate)}}</td><td>${{pct(r.metrics.accuracy)}}</td><td>${{pct(r.metrics.auc)}}</td><td>${{num(r.metrics.logloss,4)}}</td><td>${{num(r.metrics.brier,4)}}</td><td>${{esc(top?.label || top?.name || '-')}}</td></tr>`;
  }});
  document.getElementById('runs').innerHTML = html + '</tbody>';
}}

function rangeText(points, key, suffix='') {{
  const vals = points.map(p => p[key]).filter(Number.isFinite);
  if (!vals.length) return null;
  return `${{num(Math.min(...vals), 4)}}-${{num(Math.max(...vals), 4)}}${{suffix}}`;
}}
function line(points, key, cls, dotCls, invert=false) {{
  if (!points.length) return '';
  const W=720, H=210, L=42, R=14, T=16, B=28;
  const vals = points.map(p => p[key]).filter(Number.isFinite);
  const minY = Math.min(...vals), maxY = Math.max(...vals);
  const sx = p => L + p._idx * (W-L-R) / Math.max(1, records.length-1);
  const norm = v => {{
    if (!Number.isFinite(v)) return .5;
    if (Math.abs(maxY - minY) < 1e-9) return .5;
    const n = (v - minY) / (maxY - minY);
    return invert ? 1 - n : n;
  }};
  const sy = v => T + (1 - norm(v)) * (H-T-B);
  const path = points.length > 1 ? `<path class="${{cls}}" d="` + points.map((p,i) => `${{i?'L':'M'}}${{sx(p).toFixed(1)}},${{sy(p[key]).toFixed(1)}}`).join(' ') + '"/>' : '';
  return path + points.map(p => `<circle class="${{dotCls}}" cx="${{sx(p).toFixed(1)}}" cy="${{sy(p[key]).toFixed(1)}}" r="3.2"><title>${{esc(p.title)}} ${{key}}=${{num(p[key],4)}}</title></circle>`).join('');
}}
function curveLine(points, key, cls, invert=false) {{
  const pts = points.filter(p => Number.isFinite(p[key]));
  if (!pts.length) return '';
  const W=720, H=210, L=42, R=14, T=16, B=28;
  const vals = pts.map(p => p[key]);
  const minY = Math.min(...vals), maxY = Math.max(...vals);
  const maxIter = Math.max(...pts.map(p => p.iteration), 1);
  const sx = p => L + p.iteration * (W-L-R) / maxIter;
  const norm = v => {{
    if (Math.abs(maxY - minY) < 1e-9) return .5;
    const n = (v - minY) / (maxY - minY);
    return invert ? 1 - n : n;
  }};
  const sy = v => T + (1 - norm(v)) * (H-T-B);
  return `<path class="${{cls}}" d="` + pts.map((p,i) => `${{i?'L':'M'}}${{sx(p).toFixed(1)}},${{sy(p[key]).toFixed(1)}}`).join(' ') + '"/>'
}}
function renderBoostingCurve() {{
  const r = currentModel();
  const curve = r?.training_curve || [];
  if (!r) {{ document.getElementById('trend').innerHTML = '<div class="empty">No model selected.</div>'; return; }}
  if (!curve.length) {{ document.getElementById('trend').innerHTML = '<div class="empty">No per-iteration curve for this older record. New pipeline training runs will include one point per XGBoost boosting round.</div>'; return; }}
  const best = curve.find(p => p.best);
  const bestLine = best ? `<line class="axis" x1="${{(42 + best.iteration * (720-42-14) / Math.max(1, curve[curve.length-1].iteration)).toFixed(1)}}" y1="16" x2="${{(42 + best.iteration * (720-42-14) / Math.max(1, curve[curve.length-1].iteration)).toFixed(1)}}" y2="182"><title>best iteration ${{best.iteration}}</title></line>` : '';
  const notes = [
    `rounds ${{curve.length}}`,
    rangeText(curve, 'train_logloss') ? `train logloss ${{rangeText(curve, 'train_logloss')}}` : null,
    rangeText(curve, 'val_logloss') ? `val logloss ${{rangeText(curve, 'val_logloss')}}` : null,
    rangeText(curve, 'val_accuracy') ? `val acc ${{rangeText(curve, 'val_accuracy')}}` : null,
  ].filter(Boolean).join(' | ');
  document.getElementById('trend').innerHTML = `<svg viewBox="0 0 720 210" role="img"><line class="axis" x1="42" y1="182" x2="706" y2="182"/><line class="axis" x1="42" y1="16" x2="42" y2="182"/>${{curveLine(curve,'train_logloss','lineTrain',true)}}${{curveLine(curve,'val_logloss','lineLoss',true)}}${{curveLine(curve,'val_accuracy','lineEval')}}${{bestLine}}</svg><div class="trend-note">One point per XGBoost boosting round/tree; each series is independently normalized, and logloss is inverted so up is better. ${{esc(notes)}}</div>`;
}}
function renderTrend() {{
  const train = [], loss = [], evalScore = [];
  records.forEach((r, i) => {{
    if (r.kind === 'model') {{
      train.push({{ _idx: i, title: `train: ${{r.title}}`, accuracy: r.metrics.accuracy }});
      loss.push({{ _idx: i, title: `train: ${{r.title}}`, logloss: r.metrics.logloss }});
    }} else if (r.kind === 'eval' && r.summaries?.length) {{
      const score = r.summaries.reduce((a, x) => a + x.score, 0) / r.summaries.length;
      evalScore.push({{ _idx: i, title: `eval: ${{r.title}}`, score }});
    }}
  }});
  if (!train.length && !evalScore.length) {{ document.getElementById('trend').innerHTML = '<div class="empty">No train/eval records yet.</div>'; return; }}
  const notes = [
    train.length ? `val acc ${{rangeText(train, 'accuracy')}}` : null,
    evalScore.length ? `eval score ${{rangeText(evalScore, 'score')}}` : null,
    loss.length ? `logloss ${{rangeText(loss, 'logloss')}}` : null,
  ].filter(Boolean).join(' | ');
  document.getElementById('trend').innerHTML = `<svg viewBox="0 0 720 210" role="img"><line class="axis" x1="42" y1="182" x2="706" y2="182"/><line class="axis" x1="42" y1="16" x2="42" y2="182"/>${{line(train,'accuracy','lineTrain','dotTrain')}}${{line(evalScore,'score','lineEval','dotEval')}}${{line(loss,'logloss','lineLoss','dotLoss',true)}}</svg><div class="trend-note">Each series is independently normalized; logloss is inverted so up is better. ${{esc(notes)}}</div>`;
}}

function renderFeatureSelect() {{
  const sel = document.getElementById('modelSelect');
  sel.innerHTML = models.map((r,i) => `<option value="${{i}}" ${{i===models.length-1?'selected':''}}>${{esc(r.title)}} | ${{esc(r.created_at || '')}}</option>`).join('');
  sel.onchange = () => {{ renderBoostingCurve(); renderFeatures(); renderCalibration(); renderPhaseBuckets(); }};
}}
function currentModel() {{
  const i = +(document.getElementById('modelSelect').value || models.length - 1);
  return models[i];
}}
function renderFeatures() {{
  const r = currentModel();
  if (!r) {{ document.getElementById('features').innerHTML = '<tr><td class="empty">No model selected.</td></tr>'; return; }}
  const feats = [...r.features].sort((a,b) => (b.gain_share + Math.max(0,b.permutation_acc_drop)*8 + Math.max(0,b.permutation_logloss_delta)) - (a.gain_share + Math.max(0,a.permutation_acc_drop)*8 + Math.max(0,a.permutation_logloss_delta))).slice(0, 32);
  const maxGain = Math.max(...feats.map(f => f.gain_share), 1e-12);
  const maxPerm = Math.max(...feats.map(f => Math.max(0, f.permutation_acc_drop)), 1e-12);
  let html = '<thead><tr><th>#</th><th>feature</th><th>gain</th><th class="barcell">gain bar</th><th>split</th><th>perm acc</th><th class="barcell">perm bar</th><th>perm ll</th><th>solo</th><th>corr</th></tr></thead><tbody>';
  feats.forEach((f, i) => {{
    const label = f.label || f.name;
    const desc = f.description || '';
    html += `<tr title="${{esc(desc)}}"><td>${{i+1}}</td><td><div class="feature-name"><span class="feature-label">${{esc(label)}}</span><span class="feature-tech">${{esc(f.name)}}</span></div></td><td>${{pct(f.gain_share)}}</td><td><div class="bar"><div class="fill" style="width:${{100*f.gain_share/maxGain}}%"></div></div></td><td>${{pct(f.split_share)}}</td><td>${{(100*f.permutation_acc_drop).toFixed(2)}}pp</td><td><div class="bar"><div class="fill b" style="width:${{100*Math.max(0,f.permutation_acc_drop)/maxPerm}}%"></div></div></td><td>${{num(f.permutation_logloss_delta,4)}}</td><td>${{pct(f.univariate_best_accuracy)}}</td><td>${{num(f.correlation,3)}}</td></tr>`;
  }});
  document.getElementById('features').innerHTML = html + '</tbody>';
}}
function renderCalibration() {{
  const r = currentModel();
  if (!r) {{ document.getElementById('calibration').innerHTML = '<div class="empty">No model selected.</div>'; return; }}
  let html = '<table><thead><tr><th>bin</th><th>n</th><th>pred</th><th>actual</th><th>gap</th><th class="barcell">actual</th></tr></thead><tbody>';
  r.calibration.forEach(b => {{
    const actual = Number.isFinite(b.actual) ? b.actual : 0;
    html += `<tr><td>${{b.lo.toFixed(1)}}-${{b.hi.toFixed(1)}}</td><td>${{b.n}}</td><td>${{Number.isFinite(b.pred)?(100*b.pred).toFixed(1)+'%':'-'}}</td><td>${{Number.isFinite(b.actual)?(100*b.actual).toFixed(1)+'%':'-'}}</td><td>${{Number.isFinite(b.actual)?(100*(b.actual-b.pred)).toFixed(1)+'pp':'-'}}</td><td><div class="bar"><div class="fill c" style="width:${{100*actual}}%"></div></div></td></tr>`;
  }});
  document.getElementById('calibration').innerHTML = html + '</tbody></table>';
}}
function renderPhaseBuckets() {{
  const r = currentModel();
  if (!r) {{ document.getElementById('phaseBuckets').innerHTML = '<div class="empty">No model selected.</div>'; return; }}
  if (!r.phase_buckets?.length) {{ document.getElementById('phaseBuckets').innerHTML = '<div class="empty">No phase bucket data yet. New training runs will include it.</div>'; return; }}
  let html = '<table><thead><tr><th>phase</th><th>range</th><th>n</th><th>pred</th><th>actual</th><th>gap</th><th>acc</th><th>+ pred acc</th><th>- pred acc</th></tr></thead><tbody>';
  r.phase_buckets.forEach(b => {{
    const range = `${{(b.lo*100).toFixed(0)}}-${{(b.hi*100).toFixed(0)}}%`;
    const gap = Number.isFinite(b.gap) ? (100*b.gap).toFixed(1) + 'pp' : '-';
    const posAcc = Number.isFinite(b.pos_pred_accuracy) ? `${{pct(b.pos_pred_accuracy)}} (${{b.pos_pred_n}})` : '-';
    const negAcc = Number.isFinite(b.neg_pred_accuracy) ? `${{pct(b.neg_pred_accuracy)}} (${{b.neg_pred_n}})` : '-';
    html += `<tr title="phase source: ${{esc(b.source || '')}}"><td>${{esc(b.phase)}}</td><td>${{range}}</td><td>${{b.n}}</td><td>${{pct(b.pred)}}</td><td>${{pct(b.actual)}}</td><td>${{gap}}</td><td>${{pct(b.accuracy)}}</td><td>${{posAcc}}</td><td>${{negAcc}}</td></tr>`;
  }});
  document.getElementById('phaseBuckets').innerHTML = html + '</tbody></table>';
}}
function renderEvals() {{
  if (!evals.length) {{ document.getElementById('evals').innerHTML = '<tr><td class="empty">No eval records yet.</td></tr>'; return; }}
  let html = '<thead><tr><th>time</th><th>title</th><th>opp</th><th>n</th><th>score</th><th>W-L-T</th><th>dReward</th><th>ms avg/p95</th></tr></thead><tbody>';
  evals.forEach(r => (r.summaries || []).forEach(s => {{
    html += `<tr><td>${{esc(r.created_at?.slice(11) || '')}}</td><td>${{esc(r.title)}}</td><td>${{esc(s.opponent)}}</td><td>${{s.n}}</td><td>${{pct(s.score)}}</td><td>${{s.wins}}-${{s.losses}}-${{s.ties}}</td><td>${{num(s.reward_diff,3)}}+-${{num(s.reward_diff_std,3)}}</td><td>${{num(s.alpha_ms,1)}}/${{num(s.alpha_ms_p95,1)}}</td></tr>`;
  }}));
  document.getElementById('evals').innerHTML = html + '</tbody>';
}}
renderCards(); renderRuns(); renderFeatureSelect(); renderBoostingCurve(); renderFeatures(); renderCalibration(); renderPhaseBuckets(); renderEvals();
</script>
</body>
</html>
"""
