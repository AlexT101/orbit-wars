"""Master XGBoost evaluator pipeline.

One command handles the normal loop:
  1. read manifest.csv,
  2. download KaggleHub episode datasets if needed,
  3. build per-day summary_v2 + extras_v4 NPZs if missing,
  4. combine them into the engineered evaluator dataset,
  5. train an XGBoost evaluator and update dashboard.html.

The feature contract is:
  - summary_v2: 46 columns from src/bin/extract_v2.rs
  - extras_v4:  12 columns from src/bin/extract_v4.rs
  - engineered: matchup deltas/shares from train/engineered_features.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engineered_features import (
    ENGINEERED_DIM,
    ENGINEERED_NAMES,
    TEMPO_DIM,
    TEMPO_NAMES,
    append_engineered_features,
    append_tempo_features,
)

HERE = Path(__file__).resolve().parent
BOT_DIR = HERE.parent
DEFAULT_WORK = HERE / "data" / "pipeline"
DEFAULT_MANIFEST = HERE / "manifest.csv"
DEFAULT_DASHBOARD = HERE / "dashboard.html"

SUMMARY_DIM = 46
EXTRAS_DIM = 12
BASE_FEATURE_DIM = SUMMARY_DIM + EXTRAS_DIM
FEATURE_DIM = BASE_FEATURE_DIM + ENGINEERED_DIM + TEMPO_DIM
FEATURE_TAG = f"46p12e{ENGINEERED_DIM}t{TEMPO_DIM}"
FEATURE_LAYOUT = "summary_v2[46] + extras_v4[12] + engineered[%d] + tempo[%d]" % (ENGINEERED_DIM, TEMPO_DIM)
DEFAULT_MODEL = BOT_DIR / "train" / "weights" / f"xgb_{FEATURE_TAG}_latest.json"
FOURP_FEATURE_SET = "4p_v2"
FOURP_FEATURE_DIMS = {
    "4p_v1": 236,
    "4p_v2": 278,
}
FOURP_FEATURE_DIM = FOURP_FEATURE_DIMS[FOURP_FEATURE_SET]
FOURP_FEATURE_TAG = FOURP_FEATURE_SET
FOURP_DEFAULT_LABEL_MODE = "ordinal"
FOURP_LABEL_TAGS = {
    "balanced": "bal",
    "ordinal": "rank4",
    "native": "native",
}


@dataclass(frozen=True)
class ManifestRow:
    date: str
    slug: str
    url: str
    episodes: int
    bytes: int

    @property
    def kagglehub_ref(self) -> str:
        return self.slug if "/" in self.slug else f"kaggle/{self.slug}"


def read_manifest(path: Path, start_date: str | None, end_date: str | None, limit_days: int | None) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            date = r.get("date", "")
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            rows.append(
                ManifestRow(
                    date=date,
                    slug=r.get("daily_dataset_slug", "").strip(),
                    url=r.get("daily_dataset_url", "").strip(),
                    episodes=int(float(r.get("episode_count") or 0)),
                    bytes=int(float(r.get("total_bytes") or 0)),
                )
            )
    rows.sort(key=lambda x: x.date)
    return rows[:limit_days] if limit_days else rows


def resolve_label_mode(game_mode: str, label_mode: str) -> str:
    if label_mode == "auto":
        return FOURP_DEFAULT_LABEL_MODE if game_mode == "4p" else "native"
    if game_mode == "2p" and label_mode != "native":
        raise SystemExit("2p pipeline only supports --label-mode native")
    return label_mode


def ensure_rust_bins(force: bool = False, game_mode: str = "2p") -> None:
    bins = [BOT_DIR / "target" / "release" / "extract_v2", BOT_DIR / "target" / "release" / "extract_v4"]
    build_args = ["cargo", "build", "--release", "--bin", "extract_v2", "--bin", "extract_v4"]
    if game_mode == "4p":
        bins.append(BOT_DIR / "target" / "release" / f"extract_{FOURP_FEATURE_SET}")
        build_args.extend(["--bin", f"extract_{FOURP_FEATURE_SET}"])
    if not force and all(p.exists() for p in bins):
        print("rust extractors: already built")
        return
    print("building Rust extractors...")
    subprocess.check_call(
        build_args,
        cwd=BOT_DIR,
    )


def download_dataset(row: ManifestRow, download_root: Path, skip_download: bool) -> Path:
    marker = download_root / row.slug / ".kagglehub_path"
    if marker.exists():
        cached = Path(marker.read_text(encoding="utf-8").strip())
        if cached.exists():
            print(f"{row.date}: dataset cached at {cached}")
            return cached
    if skip_download:
        local = download_root / row.slug
        if local.exists():
            return local
        raise FileNotFoundError(f"{row.slug} is not cached and --skip-download was set")
    try:
        # kagglehub 1.0.1 imports get_web_endpoint, while some current
        # kagglesdk wheels expose the same function as get_endpoint. Patch the
        # alias before importing kagglehub so the pipeline works across both.
        try:
            import kagglesdk.kaggle_env as kaggle_env

            if not hasattr(kaggle_env, "get_web_endpoint") and hasattr(kaggle_env, "get_endpoint"):
                kaggle_env.get_web_endpoint = kaggle_env.get_endpoint
        except ImportError:
            pass
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub could not be imported by this Python environment.\n"
            f"Python: {sys.executable}\n"
            f"Import error: {exc}\n"
            "Try: python3 -m pip install --upgrade kagglehub kagglesdk"
        ) from exc
    print(f"{row.date}: downloading {row.kagglehub_ref}")
    path = Path(kagglehub.dataset_download(row.kagglehub_ref))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(path) + "\n", encoding="utf-8")
    return path


def find_episode_sources(dataset_path: Path, slug: str) -> list[Path]:
    zips = sorted(dataset_path.rglob("*.zip"))
    if zips:
        return zips
    jsons = sorted(p for p in dataset_path.rglob("*.json") if p.is_file())
    if not jsons:
        raise FileNotFoundError(f"no .zip or .json episode files found in {dataset_path}")
    print(f"{slug}: using unzipped dataset directory directly ({len(jsons)} json files)")
    return [dataset_path]


def extracted_cache_matches(path: Path, feature_set: str, label_mode: str, limit_entries: int | None) -> bool:
    if not path.exists():
        return False
    try:
        cached = np.load(path, allow_pickle=False)
        cached_feature_set = str(cached["feature_set"]) if "feature_set" in cached.files else "summary_v2"
        cached_label_mode = str(cached["label_mode"]) if "label_mode" in cached.files else "native"
        if cached_feature_set != feature_set or cached_label_mode != label_mode:
            return False
        if "entry_limit" not in cached.files:
            return limit_entries is None
        cached_limit = int(cached["entry_limit"])
        want_limit = -1 if limit_entries is None else int(limit_entries)
        return cached_limit == want_limit
    except Exception:
        return False


def build_day(
    row: ManifestRow,
    zips: list[Path],
    work: Path,
    workers: int,
    limit_entries: int | None,
    force: bool,
    game_mode: str,
    label_mode: str,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    import build_from_zip
    import extras_v4_build

    daily_root = f"daily_4p_{FOURP_LABEL_TAGS[label_mode]}" if game_mode == "4p" else "daily"
    day_dir = work / daily_root / row.date
    summary_npz = day_dir / "summary_v2.npz"
    extras_npz = day_dir / "extras_v4.npz"
    feature_set = FOURP_FEATURE_SET if game_mode == "4p" else "summary_v2"
    summary_rebuilt = force or not extracted_cache_matches(summary_npz, feature_set, label_mode, limit_entries)
    if summary_rebuilt:
        print(f"{row.date}: building features ({game_mode}, {feature_set})")
        build_from_zip.build(
            [str(p) for p in zips],
            str(summary_npz),
            workers,
            limit_entries,
            game_mode,
            feature_set,
            label_mode,
        )
    else:
        print(f"{row.date}: features cached")
    if game_mode == "4p":
        legacy_summary_npz = day_dir / "legacy_summary_v2.npz"
        legacy_extras_npz = day_dir / "legacy_extras_v4.npz"
        legacy_summary_rebuilt = force or not extracted_cache_matches(
            legacy_summary_npz,
            "summary_v2",
            label_mode,
            limit_entries,
        )
        if legacy_summary_rebuilt:
            print(f"{row.date}: building legacy 2p-view summary_v2 baseline features (4p rows)")
            build_from_zip.build(
                [str(p) for p in zips],
                str(legacy_summary_npz),
                workers,
                limit_entries,
                "4p",
                "summary_v2",
                label_mode,
            )
        else:
            print(f"{row.date}: legacy summary_v2 cached")
        if force or legacy_summary_rebuilt or not legacy_extras_npz.exists():
            print(f"{row.date}: building legacy 2p-view extras_v4 baseline features")
            extras_v4_build.build(legacy_summary_npz, zips, legacy_extras_npz, workers)
        else:
            print(f"{row.date}: legacy extras_v4 cached")
        return summary_npz, None, legacy_summary_npz, legacy_extras_npz
    if force or summary_rebuilt or not extras_npz.exists():
        print(f"{row.date}: building extras_v4")
        extras_v4_build.build(summary_npz, zips, extras_npz, workers)
    else:
        print(f"{row.date}: extras_v4 cached")
    return summary_npz, extras_npz, None, None


def combine_days(
    day_artifacts: list[tuple[ManifestRow, Path, Path | None, Path | None, Path | None]],
    out_npz: Path,
    force: bool,
    expected_label_mode: str | None,
) -> Path:
    if out_npz.exists() and not force:
        try:
            cached = np.load(out_npz, allow_pickle=False)
            cached_dim = cached["features"].shape[1]
            cached_feature_set = str(cached["feature_set"]) if "feature_set" in cached.files else "summary_v2"
            cached_label_mode = str(cached["label_mode"]) if "label_mode" in cached.files else "native"
            want_dim = FOURP_FEATURE_DIMS.get(cached_feature_set, FEATURE_DIM)
            label_ok = expected_label_mode is None or cached_label_mode == expected_label_mode
            if (
                cached_dim == want_dim
                and label_ok
                and not (cached_feature_set in FOURP_FEATURE_DIMS and "legacy_2p_features" not in cached.files)
            ):
                print(f"combined dataset cached: {out_npz}")
                return out_npz
            print(f"combined dataset has stale dim={cached_dim} label={cached_label_mode}; rebuilding")
        except Exception as exc:
            print(f"combined dataset cache unreadable ({exc}); rebuilding")

    summary_all = []
    extras_all = []
    direct_features_all = []
    legacy_2p_features_all = []
    labels_all = []
    meta_all = []
    strong_all = []
    game_names_all = []
    game_rewards_all = []
    game_player_count_all = []
    game_files_all = []
    game_dates_all = []
    offset = 0

    feature_set = None
    label_mode = expected_label_mode
    for row, summary_path, extras_path, legacy_summary_path, legacy_extras_path in day_artifacts:
        s = np.load(summary_path, allow_pickle=False)
        row_feature_set = str(s["feature_set"]) if "feature_set" in s.files else "summary_v2"
        row_label_mode = str(s["label_mode"]) if "label_mode" in s.files else "native"
        feature_set = feature_set or row_feature_set
        label_mode = label_mode or row_label_mode
        if row_feature_set != feature_set:
            raise ValueError(f"mixed feature sets in combine: {feature_set} and {row_feature_set}")
        if row_label_mode != label_mode:
            raise ValueError(f"mixed label modes in combine: {label_mode} and {row_label_mode}")
        if row_feature_set in FOURP_FEATURE_DIMS:
            direct_features = s["features"].astype(np.float32)
            summary = np.zeros((direct_features.shape[0], 0), dtype=np.float32)
            extras = np.zeros((direct_features.shape[0], 0), dtype=np.float32)
            if legacy_summary_path is not None and legacy_extras_path is not None:
                legacy_s = np.load(legacy_summary_path, allow_pickle=False)
                legacy_e = np.load(legacy_extras_path, allow_pickle=False)
                legacy_summary = legacy_s["summary_v2"].astype(np.float32)
                legacy_extras = legacy_e["extras"].astype(np.float32)
                if legacy_summary.shape[0] != direct_features.shape[0] or legacy_extras.shape[0] != direct_features.shape[0]:
                    raise ValueError(
                        f"{row.date}: legacy rows {legacy_summary.shape[0]}/{legacy_extras.shape[0]} "
                        f"do not match 4p rows {direct_features.shape[0]}"
                    )
                legacy_base = np.concatenate([legacy_summary, legacy_extras], axis=1).astype(np.float32)
                legacy_core = append_engineered_features(legacy_base)
                legacy_2p_features_all.append(append_tempo_features(legacy_core, legacy_s["meta"].astype(np.int32)))
        else:
            if extras_path is None:
                raise ValueError(f"{row.date}: summary_v2 feature set requires extras")
            e = np.load(extras_path, allow_pickle=False)
            summary = s["summary_v2"].astype(np.float32)
            extras = e["extras"].astype(np.float32)
            if summary.shape[0] != extras.shape[0]:
                raise ValueError(f"{row.date}: summary rows {summary.shape[0]} != extras rows {extras.shape[0]}")
            if summary.shape[1] != SUMMARY_DIM or extras.shape[1] != EXTRAS_DIM:
                raise ValueError(f"{row.date}: expected {SUMMARY_DIM}+{EXTRAS_DIM}, got {summary.shape[1]}+{extras.shape[1]}")
            direct_features = None
        meta = s["meta"].astype(np.int32).copy()
        meta[:, 0] += offset
        n_games = int(s["game_names"].shape[0]) if "game_names" in s.files else int(meta[:, 0].max() + 1 - offset)
        offset += n_games

        summary_all.append(summary)
        extras_all.append(extras)
        if direct_features is not None:
            direct_features_all.append(direct_features)
        labels_all.append(s["labels"].astype(np.float32))
        meta_all.append(meta)
        strong_all.append(s["is_strong"].astype(np.uint8) if "is_strong" in s.files else np.ones(summary.shape[0], dtype=np.uint8))
        if "game_names" in s.files:
            game_names_all.append(s["game_names"])
        if "game_rewards" in s.files:
            game_rewards_all.append(s["game_rewards"].astype(np.float32))
        if "game_player_count" in s.files:
            game_player_count_all.append(s["game_player_count"].astype(np.int16))
        if "game_files" in s.files:
            game_files_all.append(np.array([f"{row.date}:{x}" for x in s["game_files"].astype(str)], dtype="<U240"))
        game_dates_all.append(np.full(n_games, row.date, dtype="<U10"))

    feature_set = feature_set or "summary_v2"
    summary = np.concatenate(summary_all, axis=0)
    extras = np.concatenate(extras_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    meta = np.concatenate(meta_all, axis=0)
    if feature_set in FOURP_FEATURE_DIMS:
        from fourp_features import FOURP_V1_NAMES, FOURP_V2_NAMES

        features = np.concatenate(direct_features_all, axis=0).astype(np.float32)
        base_features = features
        feature_name_list = list(FOURP_V2_NAMES if feature_set == "4p_v2" else FOURP_V1_NAMES)
        feature_layout = (
            "4p_v2[278]: 4p_v1 + pairwise placement margins/probabilities + exposure"
            if feature_set == "4p_v2"
            else "4p_v1[236]: global + me + three threat-ordered opponents + aggregate/rank"
        )
    else:
        base_features = np.concatenate([summary, extras], axis=1).astype(np.float32)
        core_features = append_engineered_features(base_features)
        features = append_tempo_features(core_features, meta)
        feature_name_list = feature_names_for_dim(features.shape[1])
        feature_layout = FEATURE_LAYOUT
    is_strong = np.concatenate(strong_all, axis=0)
    game_dates = np.concatenate(game_dates_all, axis=0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        features=features,
        summary_v2=summary,
        extras_v4=extras,
        base_features=base_features,
        feature_names=np.array(feature_name_list, dtype="<U96"),
        feature_set=np.array(feature_set, dtype="<U16"),
        label_mode=np.array(label_mode or "native", dtype="<U16"),
        feature_layout=np.array(feature_layout, dtype="<U160"),
        labels=labels,
        meta=meta,
        is_strong=is_strong,
        game_dates=game_dates,
    )
    if game_names_all:
        payload["game_names"] = np.concatenate(game_names_all, axis=0)
    if game_rewards_all:
        payload["game_rewards"] = np.concatenate(game_rewards_all, axis=0)
    if game_player_count_all:
        payload["game_player_count"] = np.concatenate(game_player_count_all, axis=0)
    if game_files_all:
        payload["game_files"] = np.concatenate(game_files_all, axis=0)
    if legacy_2p_features_all:
        payload["legacy_2p_features"] = np.concatenate(legacy_2p_features_all, axis=0).astype(np.float32)
    np.savez_compressed(out_npz, **payload)
    print(f"combined rows={features.shape[0]:,} games={game_dates.shape[0]:,} dim={features.shape[1]} -> {out_npz}")
    return out_npz


def feature_names_for_dim(dim: int) -> list[str]:
    from model_dashboard import EXTRA_12_NAMES, SUMMARY_V2_NAMES

    if dim == BASE_FEATURE_DIM:
        return list(SUMMARY_V2_NAMES) + list(EXTRA_12_NAMES)
    if dim == BASE_FEATURE_DIM + ENGINEERED_DIM:
        return list(SUMMARY_V2_NAMES) + list(EXTRA_12_NAMES) + list(ENGINEERED_NAMES)
    if dim == FEATURE_DIM:
        return list(SUMMARY_V2_NAMES) + list(EXTRA_12_NAMES) + list(ENGINEERED_NAMES) + list(TEMPO_NAMES)
    return [f"f{i}" for i in range(dim)]


def split_mask(meta: np.ndarray, seed: int, frac: float) -> tuple[np.ndarray, int, int]:
    games = meta[:, 0].astype(np.int64)
    unique = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(frac * len(unique)))
    val_set = set(unique[:n_val].tolist())
    return np.array([g in val_set for g in games]), len(unique), n_val


def maybe_sample_train(X: np.ndarray, y: np.ndarray, mask: np.ndarray, max_rows: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    Xtr = X[~mask]
    ytr = y[~mask]
    if not max_rows or Xtr.shape[0] <= max_rows:
        return Xtr, ytr
    rng = np.random.default_rng(seed)
    idx = rng.choice(Xtr.shape[0], size=max_rows, replace=False)
    return Xtr[idx], ytr[idx]


def observed_game_progress(meta: np.ndarray) -> np.ndarray:
    """Return per-row tick normalized by that game's observed final tick."""
    games = meta[:, 0].astype(np.int64)
    ticks = meta[:, 1].astype(np.float64)
    order = np.argsort(games, kind="mergesort")
    sorted_games = games[order]
    sorted_ticks = ticks[order]
    out = np.zeros(meta.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_games.shape[0]:
        end = start + 1
        while end < sorted_games.shape[0] and sorted_games[end] == sorted_games[start]:
            end += 1
        denom = max(1.0, float(sorted_ticks[start:end].max()))
        out[order[start:end]] = sorted_ticks[start:end] / denom
        start = end
    return np.clip(out, 0.0, 1.0)


def train_xgb(
    dataset_npz: Path,
    model_out: Path,
    dashboard_html: Path | None,
    filter_strong: bool,
    seed: int,
    val_frac: float,
    max_train_rows: int | None,
    force: bool,
    tune: bool,
    objective: str,
    model_prefix: str,
    model_tag: str,
    baseline_2p_model: Path | None,
) -> Path:
    import xgboost as xgb
    from model_dashboard import render_xgb_dashboard, training_curve_from_xgb

    if model_out.exists() and not force:
        print(f"model cached: {model_out}")
        return model_out

    d = np.load(dataset_npz, allow_pickle=False)
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    feature_set = str(d["feature_set"]) if "feature_set" in d.files else "summary_v2"
    label_mode = str(d["label_mode"]) if "label_mode" in d.files else "native"
    feature_layout = str(d["feature_layout"]) if "feature_layout" in d.files else FEATURE_LAYOUT
    Xraw = d["features"].astype(np.float32)
    if feature_set in FOURP_FEATURE_DIMS:
        X = Xraw
        legacy_2p_X = d["legacy_2p_features"].astype(np.float32) if "legacy_2p_features" in d.files else None
    else:
        X = append_tempo_features(Xraw, meta)
        legacy_2p_X = None
    if filter_strong and "is_strong" in d.files:
        keep = d["is_strong"].astype(bool)
        X, y, meta = X[keep], y[keep], meta[keep]
        if legacy_2p_X is not None:
            legacy_2p_X = legacy_2p_X[keep]
        print(f"strong filter: kept {keep.sum():,} / {keep.shape[0]:,} rows")
    if feature_set in FOURP_FEATURE_DIMS:
        expected_dim = FOURP_FEATURE_DIMS[feature_set]
        if X.shape[1] != expected_dim:
            raise ValueError(f"expected {expected_dim} 4p features, got {X.shape[1]}")
    elif X.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM} features, got {X.shape[1]}")
    if "feature_names" in d.files:
        feature_name_list = d["feature_names"].astype(str).tolist()
    else:
        feature_name_list = feature_names_for_dim(X.shape[1])

    val_mask, n_games, n_val = split_mask(meta, seed, val_frac)
    Xtr, ytr = maybe_sample_train(X, y, val_mask, max_train_rows, seed)
    y_va_raw = y[val_mask]
    yb_va = (y_va_raw > 0).astype(np.float32)
    if objective == "regression":
        dtr = xgb.DMatrix(Xtr, label=ytr)
        dva = xgb.DMatrix(X[val_mask], label=y_va_raw)
    else:
        yb_tr = (ytr > 0).astype(np.float32)
        dtr = xgb.DMatrix(Xtr, label=yb_tr)
        dva = xgb.DMatrix(X[val_mask], label=yb_va)
    print(
        f"training rows={Xtr.shape[0]:,} val_rows={val_mask.sum():,} "
        f"games={n_games:,} val_games={n_val:,} objective={objective}"
    )

    configs = [
        dict(name=f"{model_prefix}_{model_tag}_d6_lr008", max_depth=6, learning_rate=0.08, n_est=900),
    ]
    if tune:
        configs.extend(
            [
                dict(name=f"{model_prefix}_{model_tag}_d8_lr005", max_depth=8, learning_rate=0.05, n_est=1600),
                dict(name=f"{model_prefix}_{model_tag}_d8_lr003", max_depth=8, learning_rate=0.03, n_est=2400),
                dict(name=f"{model_prefix}_{model_tag}_d6_lr005_bin512", max_depth=6, learning_rate=0.05, n_est=1800, max_bin=512),
            ]
        )

    best = None
    for cfg in configs:
        params = dict(
            objective="reg:squarederror" if objective == "regression" else "binary:logistic",
            eval_metric=["rmse"] if objective == "regression" else ["error", "logloss"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            verbosity=0,
        )
        if "max_bin" in cfg:
            params["max_bin"] = cfg["max_bin"]
        t0 = time.time()
        evals_result: dict = {}
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=cfg["n_est"],
            evals=[(dtr, "train"), (dva, "val")],
            early_stopping_rounds=60,
            verbose_eval=False,
            evals_result=evals_result,
        )
        pred_raw = bst.predict(dva)
        if objective == "regression":
            def regression_prob(v: np.ndarray) -> np.ndarray:
                return np.clip(0.5 + 0.5 * v, 1e-6, 1.0 - 1e-6)

            pred_prob = regression_prob(pred_raw)
            acc = float(((pred_raw > 0.0) == (y_va_raw > 0.0)).mean())
            rmse = float(np.sqrt(np.mean((pred_raw - y_va_raw) ** 2)))
            print(
                f"{cfg['name']}: win_acc={100*acc:.2f}% rmse={rmse:.4f} "
                f"iter={bst.best_iteration} sec={time.time()-t0:.1f}"
            )
        else:
            regression_prob = None
            pred_prob = pred_raw
            acc = float(((pred_prob > 0.5) == (yb_va > 0.5)).mean())
            rmse = None
            print(f"{cfg['name']}: acc={100*acc:.2f}% iter={bst.best_iteration} sec={time.time()-t0:.1f}")
        extra_baselines = []
        if feature_set in FOURP_FEATURE_DIMS and baseline_2p_model is not None:
            if legacy_2p_X is None:
                print("WARN 2p baseline skipped: combined dataset has no legacy_2p_features")
            elif not baseline_2p_model.exists():
                print(f"WARN 2p baseline skipped: model not found at {baseline_2p_model}")
            else:
                try:
                    baseline_bst = xgb.Booster()
                    baseline_bst.load_model(str(baseline_2p_model))
                    p2 = baseline_bst.predict(xgb.DMatrix(legacy_2p_X[val_mask]))
                    hard2 = (p2 >= 0.5).astype(np.int32)
                    yb2 = (y_va_raw > 0.0).astype(np.int32)
                    extra_baselines.append(
                        {
                            "kind": "baseline",
                            "name": "xgb_2p_latest_on_4p_legacy_view",
                            "label": "latest 2P XGBoost",
                            "description": "Latest 2-player XGBoost value net evaluated on the same 4P validation rows using the legacy dominant-opponent feature view.",
                            "accuracy": float((hard2 == yb2).mean()),
                            "positive_prediction_rate": float(hard2.mean()),
                            "tie_rate": 0.0,
                            "n": int(yb2.shape[0]),
                        }
                    )
                except Exception as exc:
                    print(f"WARN 2p baseline skipped: {exc}")
        render_xgb_dashboard(
            title=cfg["name"],
            booster=bst,
            X_val=X[val_mask],
            y_val=y[val_mask],
            pred_prob=pred_prob,
            pred_transform=regression_prob,
            feature_name_list=feature_name_list,
            phase_frac=observed_game_progress(meta[val_mask]),
            top_n=24,
            permutation=True,
            permutation_rows=20000,
            training_curve=training_curve_from_xgb(evals_result, getattr(bst, "best_iteration", None)),
            extra_baselines=extra_baselines,
            html_out=dashboard_html,
        )
        score = (acc, -rmse if rmse is not None else 0.0)
        if best is None or score > best[0]:
            best = (score, cfg, bst, acc, rmse)

    assert best is not None
    model_out.parent.mkdir(parents=True, exist_ok=True)
    best[2].save_model(str(model_out))
    sidecar = model_out.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "dataset": str(dataset_npz),
                "features": feature_layout,
                "feature_set": feature_set,
                "label_mode": label_mode,
                "feature_dim": int(X.shape[1]),
                "objective": objective,
                "accuracy": best[3],
                "rmse": best[4],
                "baseline_2p_model": str(baseline_2p_model) if baseline_2p_model is not None else None,
                "config": best[1],
                "filter_strong": filter_strong,
                "seed": seed,
                "val_frac": val_frac,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rmse_suffix = "" if best[4] is None else f" rmse={best[4]:.4f}"
    print(f"saved best model: {model_out}  acc={100*best[3]:.2f}%{rmse_suffix}")
    return model_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--limit-days", type=int, default=None, help="debug cap on manifest rows")
    p.add_argument("--limit-entries", type=int, default=None, help="debug cap on zip entries per day")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--force", action="store_true", help="rebuild cached extractors/datasets and retrain")
    p.add_argument("--force-train", action="store_true", help="retrain even if --model-out already exists, without rebuilding data")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--game-mode", choices=["2p", "4p"], default="2p", help="select 2-player or 4-player episodes")
    p.add_argument("--objective", choices=["auto", "binary", "regression"], default="auto")
    p.add_argument("--label-mode", choices=["auto", "native", "balanced", "ordinal"], default="auto")
    p.add_argument("--filter-strong", action="store_true")
    p.add_argument("--tune", action="store_true", help="train several XGB configs and save the best")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--max-train-rows", type=int, default=None, help="optional training-row sample cap")
    p.add_argument("--combined-out", type=Path, default=None)
    p.add_argument("--model-out", type=Path, default=None)
    p.add_argument("--baseline-2p-model", type=Path, default=DEFAULT_MODEL, help="2P XGBoost model to compare on 4P validation rows")
    p.add_argument("--dashboard-html", type=Path, default=DEFAULT_DASHBOARD)
    args = p.parse_args()

    label_mode = resolve_label_mode(args.game_mode, args.label_mode)
    objective = ("regression" if args.game_mode == "4p" else "binary") if args.objective == "auto" else args.objective
    model_prefix = "xgb"
    fourp_tag = f"{FOURP_FEATURE_TAG}_{FOURP_LABEL_TAGS[label_mode]}" if args.game_mode == "4p" else None
    combined_stem = f"combined_{fourp_tag}" if args.game_mode == "4p" else f"combined_{FEATURE_TAG}"
    combined_out = args.combined_out or (args.work_dir / f"{combined_stem}.npz")
    model_tag = fourp_tag if args.game_mode == "4p" else FEATURE_TAG
    model_out = args.model_out or (BOT_DIR / "train" / "weights" / f"{model_prefix}_{model_tag}_latest.json")
    if not args.train_only:
        rows = read_manifest(args.manifest, args.start_date, args.end_date, args.limit_days)
        if not rows:
            raise SystemExit("manifest selection is empty")
        print(f"selected {len(rows)} dataset day(s): {rows[0].date}..{rows[-1].date}")
        ensure_rust_bins(force=args.force, game_mode=args.game_mode)
        day_artifacts = []
        for row in rows:
            dataset_path = download_dataset(row, args.work_dir / "downloads", args.skip_download)
            sources = find_episode_sources(dataset_path, row.slug)
            summary_npz, extras_npz, legacy_summary_npz, legacy_extras_npz = build_day(
                row,
                sources,
                args.work_dir,
                args.workers,
                args.limit_entries,
                args.force,
                args.game_mode,
                label_mode,
            )
            day_artifacts.append((row, summary_npz, extras_npz, legacy_summary_npz, legacy_extras_npz))
        combine_days(day_artifacts, combined_out, args.force, label_mode)
    if args.build_only:
        print(f"build complete: {combined_out}")
        return
    train_xgb(
        combined_out,
        model_out,
        args.dashboard_html,
        args.filter_strong,
        args.seed,
        args.val_frac,
        args.max_train_rows,
        args.force or args.force_train,
        args.tune,
        objective,
        model_prefix,
        model_tag,
        args.baseline_2p_model if args.game_mode == "4p" else None,
    )


if __name__ == "__main__":
    main()
