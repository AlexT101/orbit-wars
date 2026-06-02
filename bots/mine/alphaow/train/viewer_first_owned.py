"""Generic tabbed HTML viewer for first-owned predictions.

Loads a trained XGBoost model, scans replays for cases where the model's
top non-home prediction differs from the actual first non-home capture,
and emits a self-contained HTML file with a tab per replay. Each tab
shows two side-by-side SVG boards (perspective 0 / perspective 1) with
homes, actual first capture, and predicted first capture highlighted.

Usage:
    python viewer_first_owned.py \
        --model bots/mine/alphaow/train/weights/first_owned_v9.json \
        --zip /tmp/orbit_days/orbit-wars-episodes-2026-05-30.zip \
        --out /tmp/first_owned_viewer.html \
        --n 10
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from train_first_owned_xgb import (  # noqa: E402
    CENTER,
    FEATURE_NAMES,
    KDE_BANDWIDTHS,
    extract_game,
)


# ──────────────────────────────────────────────────────────────────────────
# SVG rendering
# ──────────────────────────────────────────────────────────────────────────

BOARD_PX = 420
PADDING_PX = 20
SCALE = (BOARD_PX - 2 * PADDING_PX) / 100.0  # game coord 0..100 → px


def to_svg_xy(x: float, y: float) -> tuple[float, float]:
    return (PADDING_PX + x * SCALE, PADDING_PX + y * SCALE)


def render_board(
    title: str,
    planets: list,
    home_my: int,
    home_opp: int,
    actual_ranks: dict[int, int],  # planet_id -> rank (1..n_show_ranks)
    predicted_rank1: int | None,
) -> str:
    parts = [
        f'<svg width="{BOARD_PX}" height="{BOARD_PX}" viewBox="0 0 {BOARD_PX} {BOARD_PX}" class="board">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#101418"/>',
        f'<rect x="{PADDING_PX}" y="{PADDING_PX}" width="{BOARD_PX - 2*PADDING_PX}" height="{BOARD_PX - 2*PADDING_PX}" fill="#1a1f25" stroke="#2a2f35"/>',
    ]
    # Sun
    sx, sy = to_svg_xy(CENTER[0], CENTER[1])
    parts.append(
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{10 * SCALE:.1f}" '
        f'fill="#3a2a18" stroke="#ff8c44" stroke-width="1.5" opacity="0.7"/>'
    )
    # Planets
    for p in planets:
        pid, owner, x, y, radius, ships, prod = p
        cx, cy = to_svg_xy(x, y)
        r = max(4.0, radius * SCALE)
        # Base colour: home vs neutral.
        if pid == home_my:
            fill = "#4a8cff"
            stroke = "#9cc3ff"
        elif pid == home_opp:
            fill = "#ff5050"
            stroke = "#ffa0a0"
        else:
            fill = "#3a3a3a"
            stroke = "#5a5a5a"
        # Overlay markers.
        marker_layers = []
        actual_rank_here = actual_ranks.get(pid)
        if actual_rank_here is not None:
            # Real first-few captures: solid green ring, with a rank
            # number badge in the top-right corner of the planet.
            marker_layers.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 6:.1f}" '
                f'fill="none" stroke="#3fd06d" stroke-width="3"/>'
            )
            bx = cx + r * 0.95
            by = cy - r * 0.95
            marker_layers.append(
                f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8" '
                f'fill="#0a1a10" stroke="#3fd06d" stroke-width="1.5"/>'
                f'<text x="{bx:.1f}" y="{by:.1f}" dy="0.32em" '
                f'fill="#3fd06d" font-size="10" font-weight="700" '
                f'text-anchor="middle">{actual_rank_here}</text>'
            )
        if pid == predicted_rank1:
            # Model's predicted rank-1 — dashed gold ring (outside the
            # green if both apply).
            marker_layers.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 11:.1f}" '
                f'fill="none" stroke="#ffd23f" stroke-width="2.5" '
                f'stroke-dasharray="4 3"/>'
            )
        parts.extend(marker_layers)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )
        # Ship count inside the planet (the headline number).
        # Centred; vertically nudged by 0.32em for crude baseline middling.
        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" dy="0.32em" '
            f'fill="#fff" font-size="10" font-weight="600" '
            f'text-anchor="middle">{ships}</text>'
        )
        # Label below: id + production
        parts.append(
            f'<text x="{cx:.1f}" y="{cy + r + 11:.1f}" '
            f'fill="#aaa" font-size="9" text-anchor="middle">'
            f'{pid} • p{prod}</text>'
        )
    parts.append(
        f'<text x="{BOARD_PX/2:.1f}" y="14" fill="#ddd" font-size="13" '
        f'font-weight="600" text-anchor="middle">{html.escape(title)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Model loading + per-game prediction
# ──────────────────────────────────────────────────────────────────────────


def build_feature_names() -> list[str]:
    # Mirror the runtime layout in the training script.
    names = list(FEATURE_NAMES)
    for sigma in KDE_BANDWIDTHS:
        names.append(f"num_density_h{sigma:g}")
        names.append(f"prod_density_h{sigma:g}")
    return names


def predict_per_perspective(
    bst: xgb.Booster, X_full: np.ndarray, feat_names: list[str]
) -> np.ndarray:
    d = xgb.DMatrix(X_full, feature_names=feat_names)
    return bst.predict(d)


def planets_in_obs(replay_json: dict) -> list:
    """Return raw planets list from the first frame."""
    return replay_json["steps"][0][0]["observation"]["planets"]


def collect_games(
    bst: xgb.Booster,
    zip_path: Path,
    n_target: int,
    max_scan: int,
    feat_names: list[str],
    mismatches_only: bool,
    n_actual_ranks: int,
) -> list[dict]:
    """Scan games, collect up to `n_target` for display.
    If `mismatches_only` is True, only includes games where the model's
    rank-1 prediction differs from the actual for ≥1 perspective."""
    found: list[dict] = []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(z.namelist())[:max_scan]
        for game_name in names:
            if len(found) >= n_target:
                break
            try:
                with z.open(game_name) as f:
                    data = json.load(f)
            except Exception:
                continue
            rows = extract_game(data, game_name)
            if not rows:
                continue
            X = np.array([r[0] for r in rows], dtype=np.float32)
            preds = predict_per_perspective(bst, X, feat_names)
            persps = np.array([r[4] for r in rows])
            pids = np.array([r[5] for r in rows])
            ranks = np.array([r[6] for r in rows])

            per_persp_info = {}
            mismatch = False
            ok = True
            for persp in (0, 1):
                mask = persps == persp
                p_pred = preds[mask]
                p_pids = pids[mask]
                p_ranks = ranks[mask]
                # Map planet_id → rank for ranks 1..n_actual_ranks.
                actual_ranks_map: dict[int, int] = {}
                for i, r in enumerate(p_ranks):
                    if 1 <= int(r) <= n_actual_ranks:
                        actual_ranks_map[int(p_pids[i])] = int(r)
                # Skip games where the perspective didn't actually have
                # a rank-1 (rare).
                if 1 not in actual_ranks_map.values():
                    ok = False
                    break
                cand_mask = p_ranks != 0
                if not cand_mask.any():
                    ok = False
                    break
                cand_pids = p_pids[cand_mask]
                cand_pred = p_pred[cand_mask]
                top_idx = int(np.argmax(cand_pred))
                pred_pid = int(cand_pids[top_idx])
                pred_prob = float(cand_pred[top_idx])
                actual_rank1_pid = next(
                    pid for pid, r in actual_ranks_map.items() if r == 1
                )
                actual_prob = float(p_pred[np.where(p_pids == actual_rank1_pid)[0][0]])
                per_persp_info[persp] = {
                    "actual_ranks": actual_ranks_map,
                    "actual_rank1": actual_rank1_pid,
                    "predicted_rank1": pred_pid,
                    "predicted_prob": pred_prob,
                    "actual_prob": actual_prob,
                }
                if pred_pid != actual_rank1_pid:
                    mismatch = True
            if not ok:
                continue
            if mismatches_only and not mismatch:
                continue

            planets = planets_in_obs(data)
            homes = {}
            for p in planets:
                pid, owner = p[0], p[1]
                if owner in (0, 1) and owner not in homes:
                    homes[owner] = pid
            if 0 not in homes or 1 not in homes:
                continue
            found.append({
                "name": game_name,
                "planets": planets,
                "homes": homes,
                "per_persp": per_persp_info,
                "rewards": data.get("rewards", []),
            })
    return found


# ──────────────────────────────────────────────────────────────────────────
# HTML emission
# ──────────────────────────────────────────────────────────────────────────


HTML_HEAD = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Orbit Wars — first-owned mispredictions</title>
<style>
  body { background: #0a0d10; color: #d0d7df; font-family: -apple-system,
         Helvetica, sans-serif; margin: 0; padding: 20px; }
  h1 { margin: 0 0 12px 0; font-size: 18px; color: #fff; }
  .legend { font-size: 12px; color: #8a929a; margin-bottom: 16px; }
  .legend .swatch { display: inline-block; width: 10px; height: 10px;
                    margin: 0 4px 0 12px; vertical-align: middle;
                    border-radius: 50%; }
  .tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 14px;
          border-bottom: 1px solid #2a2f35; }
  .tab-btn { background: #1a1f25; color: #aab2bb; border: 1px solid #2a2f35;
             padding: 6px 12px; cursor: pointer; font-size: 12px;
             border-bottom: none; }
  .tab-btn.active { background: #2a313a; color: #fff; }
  .tab-body { display: none; padding: 12px 0; }
  .tab-body.active { display: block; }
  .pair { display: flex; gap: 16px; flex-wrap: wrap; }
  .panel { background: #14181d; padding: 12px; border-radius: 6px;
           min-width: 460px; }
  .panel h3 { margin: 0 0 8px 0; font-size: 13px; color: #fff; }
  .stats { font-size: 11px; color: #aaa; line-height: 1.5;
           margin-top: 6px; font-family: monospace; }
  .stats .ok { color: #3fd06d; }
  .stats .bad { color: #ffd23f; }
</style>
</head>
<body>
<h1>first-owned model viewer (v12)</h1>
<div class="legend">
  <span class="swatch" style="background:#4a8cff;"></span>my home
  <span class="swatch" style="background:#ff5050;"></span>opponent home
  <span class="swatch" style="background:none;border:2px solid #3fd06d;"></span>actual first captures (#1, #2, #3 in green)
  <span class="swatch" style="background:none;border:2px dashed #ffd23f;"></span>model's predicted first capture
</div>
"""

HTML_TAIL = """\
<script>
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tid = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-body').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + tid).classList.add('active');
    });
  });
  document.querySelector('.tab-btn')?.click();
</script>
</body>
</html>
"""


def render_game_tab(idx: int, game: dict) -> tuple[str, str]:
    name_short = Path(game["name"]).stem
    rewards = game.get("rewards", [])
    btn = (
        f'<button class="tab-btn" data-tab="{idx}">'
        f'{html.escape(name_short)}</button>'
    )
    panels = []
    for persp in (0, 1):
        info = game["per_persp"][persp]
        home_my = game["homes"][persp]
        home_opp = game["homes"][1 - persp]
        actual_ranks = info["actual_ranks"]
        pred = info["predicted_rank1"]
        actual = info["actual_rank1"]
        title = (
            f"perspective {persp} "
            f"(P{persp}={'won' if rewards[persp] == 1 else 'lost'})"
        )
        svg = render_board(
            title=title,
            planets=game["planets"],
            home_my=home_my,
            home_opp=home_opp,
            actual_ranks=actual_ranks,
            predicted_rank1=pred,
        )
        match_html = (
            '<span class="ok">match</span>' if actual == pred
            else '<span class="bad">MISMATCH</span>'
        )
        ranks_str = ", ".join(
            f"#{r}=planet {pid}"
            for pid, r in sorted(actual_ranks.items(), key=lambda kv: kv[1])
        )
        stats = (
            f'<div class="stats">'
            f'  actual order: {ranks_str}<br/>'
            f'  actual rank-1 model p={info["actual_prob"]:.3f}<br/>'
            f'  model picked: planet {pred} '
            f'(p={info["predicted_prob"]:.3f}) — {match_html}'
            f'</div>'
        )
        panels.append(f'<div class="panel"><h3>{title}</h3>{svg}{stats}</div>')
    body = (
        f'<div id="tab-{idx}" class="tab-body">'
        f'  <div class="pair">{"".join(panels)}</div>'
        f'</div>'
    )
    return btn, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10,
                    help="Number of games to include")
    ap.add_argument("--max-scan", type=int, default=2000,
                    help="Max games to scan")
    ap.add_argument("--mismatches-only", action="store_true",
                    help="Only include games where model rank-1 ≠ actual")
    ap.add_argument("--show-ranks", type=int, default=3,
                    help="How many actual first captures to highlight (1..k)")
    args = ap.parse_args()

    bst = xgb.Booster()
    bst.load_model(args.model)
    feat_names = build_feature_names()
    n_feats_expected = len(feat_names)
    n_feats_model = bst.num_features()
    if n_feats_expected != n_feats_model:
        # Model trained on different KDE bandwidths — fall back to generic
        # names so DMatrix matches model expectations.
        feat_names = [f"f{i}" for i in range(n_feats_model)]
        print(
            f"WARN: build_feature_names produced {n_feats_expected} names "
            f"but model expects {n_feats_model}; using anonymous names",
            file=sys.stderr,
        )

    mode = "mismatches" if args.mismatches_only else "all"
    print(f"scanning {args.zip} ({mode}, target n={args.n}, "
          f"max scan={args.max_scan})…")
    games = collect_games(
        bst, Path(args.zip), args.n, args.max_scan, feat_names,
        mismatches_only=args.mismatches_only,
        n_actual_ranks=args.show_ranks,
    )
    print(f"  found {len(games)} games")

    btns, bodies = [], []
    for i, g in enumerate(games):
        b, body = render_game_tab(i, g)
        btns.append(b)
        bodies.append(body)
    html_out = (
        HTML_HEAD
        + f'<div class="tabs">{"".join(btns)}</div>'
        + "".join(bodies)
        + HTML_TAIL
    )
    out_path = Path(args.out)
    out_path.write_text(html_out, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
