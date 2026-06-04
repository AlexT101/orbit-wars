"""Combine a summary_v2 NPZ + extras_v4 NPZ (same row order) into the full
157-d combined dataset used by the trainer:

    summary_v2[46] + extras_v4[12] + engineered[88] + tempo[11]

This mirrors pipeline.combine_days but for a single already-built source
(e.g. the rank1 replay directory), without the manifest/day machinery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from engineered_features import append_engineered_features, append_tempo_features

SUMMARY_DIM = 46
EXTRAS_DIM = 12


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--extras", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--date-tag", default="rank1", help="value stored in game_dates")
    args = p.parse_args()

    s = np.load(args.summary, allow_pickle=False)
    e = np.load(args.extras, allow_pickle=False)
    summary = s["summary_v2"].astype(np.float32)
    extras = e["extras"].astype(np.float32)
    if summary.shape[0] != extras.shape[0]:
        raise SystemExit(f"row mismatch: summary {summary.shape[0]} vs extras {extras.shape[0]}")
    if summary.shape[1] != SUMMARY_DIM or extras.shape[1] != EXTRAS_DIM:
        raise SystemExit(f"expected {SUMMARY_DIM}+{EXTRAS_DIM}, got {summary.shape[1]}+{extras.shape[1]}")

    labels = s["labels"].astype(np.float32)
    meta = s["meta"].astype(np.int32)
    base = np.concatenate([summary, extras], axis=1).astype(np.float32)
    core = append_engineered_features(base)
    features = append_tempo_features(core, meta)

    from model_dashboard import EXTRA_12_NAMES, SUMMARY_V2_NAMES
    from engineered_features import ENGINEERED_NAMES, TEMPO_NAMES

    feature_names = (
        list(SUMMARY_V2_NAMES) + list(EXTRA_12_NAMES) + list(ENGINEERED_NAMES) + list(TEMPO_NAMES)
    )
    if len(feature_names) != features.shape[1]:
        raise SystemExit(f"name/feature mismatch {len(feature_names)} vs {features.shape[1]}")

    n_games = int(s["game_names"].shape[0]) if "game_names" in s.files else int(meta[:, 0].max() + 1)
    game_dates = np.full(n_games, args.date_tag, dtype="<U10")
    is_strong = s["is_strong"].astype(np.uint8) if "is_strong" in s.files else np.ones(len(labels), np.uint8)

    payload = dict(
        features=features,
        summary_v2=summary,
        extras_v4=extras,
        base_features=base,
        feature_names=np.array(feature_names, dtype="<U96"),
        feature_set=np.array("summary_v2", dtype="<U16"),
        label_mode=np.array("native", dtype="<U16"),
        feature_layout=np.array(
            "summary_v2[46] + extras_v4[12] + engineered[88] + tempo[11]", dtype="<U160"
        ),
        labels=labels,
        meta=meta,
        is_strong=is_strong,
        game_dates=game_dates,
    )
    for k in ("game_names", "game_rewards", "game_player_count", "game_files"):
        if k in s.files:
            payload[k] = s[k]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(
        f"combined rows={features.shape[0]:,} games={n_games:,} dim={features.shape[1]} "
        f"strong_rows={int(is_strong.sum()):,} -> {args.out} ({args.out.stat().st_size/1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
