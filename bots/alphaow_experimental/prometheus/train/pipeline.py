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

from engineered_features import ENGINEERED_DIM, ENGINEERED_NAMES, append_engineered_features

HERE = Path(__file__).resolve().parent
BOT_DIR = HERE.parent
DEFAULT_WORK = HERE / "data" / "pipeline"
DEFAULT_MANIFEST = HERE / "manifest.csv"
DEFAULT_DASHBOARD = HERE / "dashboard.html"
DEFAULT_MODEL = BOT_DIR / "train" / "weights" / "xgb_46p12_latest.json"

SUMMARY_DIM = 46
EXTRAS_DIM = 12
BASE_FEATURE_DIM = SUMMARY_DIM + EXTRAS_DIM
FEATURE_DIM = BASE_FEATURE_DIM + ENGINEERED_DIM
FEATURE_LAYOUT = "summary_v2[46] + extras_v4[12] + engineered[%d]" % ENGINEERED_DIM


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


def ensure_rust_bins(force: bool = False) -> None:
    bins = [BOT_DIR / "target" / "release" / "extract_v2", BOT_DIR / "target" / "release" / "extract_v4"]
    if not force and all(p.exists() for p in bins):
        print("rust extractors: already built")
        return
    print("building Rust extractors...")
    subprocess.check_call(
        ["cargo", "build", "--release", "--bin", "extract_v2", "--bin", "extract_v4"],
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


def build_day(row: ManifestRow, zips: list[Path], work: Path, workers: int, limit_entries: int | None, force: bool) -> tuple[Path, Path]:
    import build_from_zip
    import extras_v4_build

    day_dir = work / "daily" / row.date
    summary_npz = day_dir / "summary_v2.npz"
    extras_npz = day_dir / "extras_v4.npz"
    if force or not summary_npz.exists():
        print(f"{row.date}: building summary_v2")
        build_from_zip.build([str(p) for p in zips], str(summary_npz), workers, limit_entries)
    else:
        print(f"{row.date}: summary_v2 cached")
    if force or not extras_npz.exists():
        print(f"{row.date}: building extras_v4")
        extras_v4_build.build(summary_npz, zips, extras_npz, workers)
    else:
        print(f"{row.date}: extras_v4 cached")
    return summary_npz, extras_npz


def combine_days(day_artifacts: list[tuple[ManifestRow, Path, Path]], out_npz: Path, force: bool) -> Path:
    if out_npz.exists() and not force:
        try:
            cached = np.load(out_npz, allow_pickle=False)
            if cached["features"].shape[1] == FEATURE_DIM:
                print(f"combined dataset cached: {out_npz}")
                return out_npz
            print(f"combined dataset has stale dim={cached['features'].shape[1]} (want {FEATURE_DIM}); rebuilding")
        except Exception as exc:
            print(f"combined dataset cache unreadable ({exc}); rebuilding")

    summary_all = []
    extras_all = []
    labels_all = []
    meta_all = []
    strong_all = []
    game_names_all = []
    game_files_all = []
    game_dates_all = []
    offset = 0

    for row, summary_path, extras_path in day_artifacts:
        s = np.load(summary_path, allow_pickle=False)
        e = np.load(extras_path, allow_pickle=False)
        summary = s["summary_v2"].astype(np.float32)
        extras = e["extras"].astype(np.float32)
        if summary.shape[0] != extras.shape[0]:
            raise ValueError(f"{row.date}: summary rows {summary.shape[0]} != extras rows {extras.shape[0]}")
        if summary.shape[1] != SUMMARY_DIM or extras.shape[1] != EXTRAS_DIM:
            raise ValueError(f"{row.date}: expected {SUMMARY_DIM}+{EXTRAS_DIM}, got {summary.shape[1]}+{extras.shape[1]}")
        meta = s["meta"].astype(np.int32).copy()
        meta[:, 0] += offset
        n_games = int(s["game_names"].shape[0]) if "game_names" in s.files else int(meta[:, 0].max() + 1 - offset)
        offset += n_games

        summary_all.append(summary)
        extras_all.append(extras)
        labels_all.append(s["labels"].astype(np.float32))
        meta_all.append(meta)
        strong_all.append(s["is_strong"].astype(np.uint8) if "is_strong" in s.files else np.ones(summary.shape[0], dtype=np.uint8))
        if "game_names" in s.files:
            game_names_all.append(s["game_names"])
        if "game_files" in s.files:
            game_files_all.append(np.array([f"{row.date}:{x}" for x in s["game_files"].astype(str)], dtype="<U240"))
        game_dates_all.append(np.full(n_games, row.date, dtype="<U10"))

    summary = np.concatenate(summary_all, axis=0)
    extras = np.concatenate(extras_all, axis=0)
    base_features = np.concatenate([summary, extras], axis=1).astype(np.float32)
    features = append_engineered_features(base_features)
    labels = np.concatenate(labels_all, axis=0)
    meta = np.concatenate(meta_all, axis=0)
    is_strong = np.concatenate(strong_all, axis=0)
    game_dates = np.concatenate(game_dates_all, axis=0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        features=features,
        summary_v2=summary,
        extras_v4=extras,
        base_features=base_features,
        feature_names=np.array(feature_names_for_dim(features.shape[1]), dtype="<U80"),
        labels=labels,
        meta=meta,
        is_strong=is_strong,
        game_dates=game_dates,
    )
    if game_names_all:
        payload["game_names"] = np.concatenate(game_names_all, axis=0)
    if game_files_all:
        payload["game_files"] = np.concatenate(game_files_all, axis=0)
    np.savez_compressed(out_npz, **payload)
    print(f"combined rows={features.shape[0]:,} games={game_dates.shape[0]:,} dim={features.shape[1]} -> {out_npz}")
    return out_npz


def feature_names_for_dim(dim: int) -> list[str]:
    from model_dashboard import EXTRA_12_NAMES, SUMMARY_V2_NAMES

    if dim == BASE_FEATURE_DIM:
        return list(SUMMARY_V2_NAMES) + list(EXTRA_12_NAMES)
    if dim == FEATURE_DIM:
        return list(SUMMARY_V2_NAMES) + list(EXTRA_12_NAMES) + list(ENGINEERED_NAMES)
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


def train_xgb(dataset_npz: Path, model_out: Path, dashboard_html: Path | None, filter_strong: bool, seed: int, val_frac: float, max_train_rows: int | None, force: bool, tune: bool) -> Path:
    import xgboost as xgb
    from model_dashboard import render_xgb_dashboard, training_curve_from_xgb

    if model_out.exists() and not force:
        print(f"model cached: {model_out}")
        return model_out

    d = np.load(dataset_npz, allow_pickle=False)
    X = append_engineered_features(d["features"].astype(np.float32))
    y = d["labels"].astype(np.float32)
    meta = d["meta"]
    if filter_strong and "is_strong" in d.files:
        keep = d["is_strong"].astype(bool)
        X, y, meta = X[keep], y[keep], meta[keep]
        print(f"strong filter: kept {keep.sum():,} / {keep.shape[0]:,} rows")
    if X.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM} features, got {X.shape[1]}")

    val_mask, n_games, n_val = split_mask(meta, seed, val_frac)
    Xtr, ytr = maybe_sample_train(X, y, val_mask, max_train_rows, seed)
    yb_tr = (ytr > 0).astype(np.float32)
    yb_va = (y[val_mask] > 0).astype(np.float32)
    dtr = xgb.DMatrix(Xtr, label=yb_tr)
    dva = xgb.DMatrix(X[val_mask], label=yb_va)
    print(f"training rows={Xtr.shape[0]:,} val_rows={val_mask.sum():,} games={n_games:,} val_games={n_val:,}")

    configs = [
        dict(name=f"xgb_46p12e{ENGINEERED_DIM}_d6_lr008", max_depth=6, learning_rate=0.08, n_est=900),
    ]
    if tune:
        configs.extend(
            [
                dict(name=f"xgb_46p12e{ENGINEERED_DIM}_d8_lr005", max_depth=8, learning_rate=0.05, n_est=1600),
                dict(name=f"xgb_46p12e{ENGINEERED_DIM}_d8_lr003", max_depth=8, learning_rate=0.03, n_est=2400),
                dict(name=f"xgb_46p12e{ENGINEERED_DIM}_d6_lr005_bin512", max_depth=6, learning_rate=0.05, n_est=1800, max_bin=512),
            ]
        )

    best = None
    for cfg in configs:
        params = dict(
            objective="binary:logistic",
            eval_metric=["error", "logloss"],
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
        pred = bst.predict(dva)
        acc = float(((pred > 0.5) == (yb_va > 0.5)).mean())
        print(f"{cfg['name']}: acc={100*acc:.2f}% iter={bst.best_iteration} sec={time.time()-t0:.1f}")
        render_xgb_dashboard(
            title=cfg["name"],
            booster=bst,
            X_val=X[val_mask],
            y_val=y[val_mask],
            pred_prob=pred,
            feature_name_list=feature_names_for_dim(X.shape[1]),
            phase_frac=observed_game_progress(meta[val_mask]),
            top_n=24,
            permutation=True,
            permutation_rows=20000,
            training_curve=training_curve_from_xgb(evals_result, getattr(bst, "best_iteration", None)),
            html_out=dashboard_html,
        )
        if best is None or acc > best[0]:
            best = (acc, cfg, bst)

    assert best is not None
    model_out.parent.mkdir(parents=True, exist_ok=True)
    best[2].save_model(str(model_out))
    sidecar = model_out.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "dataset": str(dataset_npz),
                "features": FEATURE_LAYOUT,
                "feature_dim": FEATURE_DIM,
                "accuracy": best[0],
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
    print(f"saved best model: {model_out}  acc={100*best[0]:.2f}%")
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
    p.add_argument("--filter-strong", action="store_true")
    p.add_argument("--tune", action="store_true", help="train several XGB configs and save the best")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--max-train-rows", type=int, default=None, help="optional training-row sample cap")
    p.add_argument("--combined-out", type=Path, default=None)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dashboard-html", type=Path, default=DEFAULT_DASHBOARD)
    args = p.parse_args()

    combined_out = args.combined_out or (args.work_dir / f"combined_46p12e{ENGINEERED_DIM}.npz")
    if not args.train_only:
        rows = read_manifest(args.manifest, args.start_date, args.end_date, args.limit_days)
        if not rows:
            raise SystemExit("manifest selection is empty")
        print(f"selected {len(rows)} dataset day(s): {rows[0].date}..{rows[-1].date}")
        ensure_rust_bins(force=args.force)
        day_artifacts = []
        for row in rows:
            dataset_path = download_dataset(row, args.work_dir / "downloads", args.skip_download)
            sources = find_episode_sources(dataset_path, row.slug)
            summary_npz, extras_npz = build_day(row, sources, args.work_dir, args.workers, args.limit_entries, args.force)
            day_artifacts.append((row, summary_npz, extras_npz))
        combine_days(day_artifacts, combined_out, args.force)
    if args.build_only:
        print(f"build complete: {combined_out}")
        return
    train_xgb(
        combined_out,
        args.model_out,
        args.dashboard_html,
        args.filter_strong,
        args.seed,
        args.val_frac,
        args.max_train_rows,
        args.force or args.force_train,
        args.tune,
    )


if __name__ == "__main__":
    main()
