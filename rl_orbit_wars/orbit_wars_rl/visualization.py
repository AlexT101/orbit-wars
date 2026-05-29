from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Iterable


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _series(rows: list[dict], key: str, smooth: int = 1) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for row in rows:
        if key not in row or "step" not in row:
            continue
        value = row[key]
        if value is None:
            continue
        try:
            xs.append(float(row["step"]))
            ys.append(float(value))
        except (TypeError, ValueError):
            continue
    if smooth > 1 and len(ys) >= smooth:
        smoothed = []
        for i in range(len(ys)):
            lo = max(0, i - smooth + 1)
            smoothed.append(sum(ys[lo : i + 1]) / (i - lo + 1))
        ys = smoothed
    return xs, ys


def _nice_bounds(values: list[float], domain: tuple[float, float] | None) -> tuple[float, float]:
    if domain is not None:
        return domain
    if not values:
        return 0.0, 1.0
    min_y = min(values)
    max_y = max(values)
    if not math.isfinite(min_y) or not math.isfinite(max_y):
        return 0.0, 1.0
    if abs(max_y - min_y) < 1e-9:
        pad = max(1e-3, abs(max_y) * 0.1)
        return min_y - pad, max_y + pad
    pad = (max_y - min_y) * 0.08
    return min_y - pad, max_y + pad


def _format_num(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _points(
    xs: list[float],
    ys: list[float],
    width: int,
    height: int,
    pad_l: int,
    pad_r: int,
    pad_t: int,
    pad_b: int,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
) -> list[tuple[float, float]]:
    if not xs:
        return []
    min_x, max_x = x_domain
    min_y, max_y = y_domain
    if abs(max_x - min_x) < 1e-9:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-9:
        max_y = min_y + 1.0
    out = []
    for x, y in zip(xs, ys):
        px = pad_l + (x - min_x) / (max_x - min_x) * (width - pad_l - pad_r)
        py = height - pad_b - (y - min_y) / (max_y - min_y) * (height - pad_t - pad_b)
        out.append((px, py))
    return out


def _polyline(xs: list[float], ys: list[float], *args) -> str:
    if len(xs) < 2:
        return ""
    return " ".join(f"{px:.1f},{py:.1f}" for px, py in _points(xs, ys, *args))


def _chart(
    rows: list[dict],
    title: str,
    keys: Iterable[str],
    colors: list[str],
    *,
    y_domain: tuple[float, float] | None = None,
    smooth: int = 1,
    points: bool = False,
    vlines: list[dict] | None = None,
) -> str:
    width = 760
    height = 260
    pad_l = 60
    pad_r = 18
    pad_t = 24
    pad_b = 38
    keys = list(keys)
    series = []
    all_x = []
    all_y = []
    for key in keys:
        xs, ys = _series(rows, key, smooth=smooth)
        series.append((key, xs, ys))
        all_x.extend(xs)
        all_y.extend(ys)
    for marker in vlines or []:
        try:
            all_x.append(float(marker["step"]))
        except (KeyError, TypeError, ValueError):
            continue
    if all_x:
        x_domain = (min(all_x), max(all_x))
    else:
        x_domain = (0.0, 1.0)
    y_domain_actual = _nice_bounds(all_y, y_domain)
    min_y, max_y = y_domain_actual

    parts = [
        '<section class="chart">',
        f"<h2>{html.escape(title)}</h2>",
        f'<svg viewBox="0 0 {width} {height}" role="img">',
        f'<line class="axis" x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}"></line>',
        f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}"></line>',
    ]
    for i in range(5):
        y_value = min_y + (max_y - min_y) * i / 4.0
        y = height - pad_b - i / 4.0 * (height - pad_t - pad_b)
        parts.append(f'<line class="gridline" x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"></line>')
        parts.append(f'<text class="tick" x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end">{_format_num(y_value)}</text>')
    if all_x:
        parts.append(f'<text class="tick" x="{pad_l}" y="{height - 12}" text-anchor="middle">{int(x_domain[0])}</text>')
        parts.append(f'<text class="tick" x="{width - pad_r}" y="{height - 12}" text-anchor="middle">{int(x_domain[1])}</text>')

    legend = []
    for marker in vlines or []:
        try:
            marker_x = float(marker["step"])
        except (KeyError, TypeError, ValueError):
            continue
        x = _points(
            [marker_x],
            [min_y],
            width,
            height,
            pad_l,
            pad_r,
            pad_t,
            pad_b,
            x_domain,
            y_domain_actual,
        )[0][0]
        label = html.escape(str(marker.get("label", "marker")))
        parts.append(
            f'<line class="marker" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height - pad_b}"></line>'
        )
        parts.append(
            f'<text class="marker-label" x="{x + 5:.1f}" y="{pad_t + 12}" transform="rotate(90 {x + 5:.1f},{pad_t + 12})">{label}</text>'
        )
    for i, (key, xs, ys) in enumerate(series):
        color = colors[i % len(colors)]
        line = _polyline(xs, ys, width, height, pad_l, pad_r, pad_t, pad_b, x_domain, y_domain_actual)
        if line:
            parts.append(
                f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.2" '
                'stroke-linejoin="round" stroke-linecap="round"></polyline>'
            )
        if points:
            for px, py in _points(xs, ys, width, height, pad_l, pad_r, pad_t, pad_b, x_domain, y_domain_actual):
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{color}"></circle>')
        if ys:
            legend.append(
                f'<span><i style="background:{color}"></i>{html.escape(key)}: {_format_num(ys[-1])}</span>'
            )
    parts.append("</svg>")
    parts.append(f'<div class="legend">{"".join(legend)}</div>')
    parts.append("</section>")
    return "\n".join(parts)


def _latest_table(row: dict) -> str:
    if not row:
        return "<p>No metrics yet.</p>"
    keys = [
        key
        for key in sorted(row)
        if key.startswith("reward_")
        or key
        in {
            "step",
            "episode",
            "mean_return_25",
            "reward_mean",
            "explained_var",
            "entropy",
            "clip_frac",
            "approx_kl",
            "sps",
            "lr",
            "noop_rate",
            "launch_rate",
            "avg_send_bin",
            "accuracy",
            "loss",
            "teacher",
            "samples",
            "noop_fraction",
            "unique_labels",
            "phase",
        }
    ]
    rows = []
    for key in keys:
        value = row[key]
        if isinstance(value, float):
            value = f"{value:.6g}"
        rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def write_training_report(log_dir: Path) -> Path:
    metrics = load_jsonl(log_dir / "metrics.jsonl")
    evals = load_jsonl(log_dir / "eval.jsonl")
    bc_metrics = load_jsonl(log_dir / "bc_metrics.jsonl")
    phase_events = load_jsonl(log_dir / "phase_events.jsonl")
    latest = metrics[-1] if metrics else {}
    latest_eval = evals[-1] if evals else {}
    latest_bc = next((row for row in reversed(bc_metrics) if row.get("phase") == "bc"), {})
    colors = ["#2f80ed", "#27ae60", "#eb5757", "#9b51e0", "#f2994a", "#00a3a3"]
    ppo_markers = [
        row
        for row in phase_events
        if row.get("kind") in {"pretrain_end", "ppo_start"} and "step" in row
    ]

    eval_summary = "No eval yet."
    if latest_eval:
        win_keys = [key for key in sorted(latest_eval) if key.startswith("win_rate_")]
        parts = [f"{key.replace('win_rate_', '')} {float(latest_eval.get(key, 0.0)):.1%}" for key in win_keys]
        eval_summary = f"step {latest_eval.get('step')} · " + " · ".join(parts)

    bc_summary = "No BC log yet."
    if latest_bc:
        bc_summary = (
            f"BC {latest_bc.get('teacher', '?')} · "
            f"epoch {latest_bc.get('epoch')} · "
            f"acc {float(latest_bc.get('accuracy', 0.0)):.1%} · "
            f"loss {float(latest_bc.get('loss', 0.0)):.4g}"
        )

    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="5">
  <title>Orbit Wars RL Training</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 28px; background: #f7f7f4; color: #202124; }}
    header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 24px; border-bottom: 1px solid #d8d8d0; padding-bottom: 14px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    h2 {{ font-size: 15px; margin: 0 0 8px; }}
    .subtle {{ color: #5f6368; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; margin-top: 18px; }}
    .chart, .panel {{ background: #ffffff; border: 1px solid #d8d8d0; border-radius: 8px; padding: 14px; }}
    svg {{ width: 100%; height: 260px; display: block; background: #fbfbf8; border-radius: 6px; }}
    .axis {{ stroke: #b8b8ad; stroke-width: 1; }}
    .gridline {{ stroke: #e4e4dc; stroke-width: 1; }}
    .marker {{ stroke: #111827; stroke-width: 1.5; stroke-dasharray: 5 5; opacity: 0.7; }}
    .marker-label {{ fill: #111827; font-size: 10px; }}
    .tick {{ fill: #74766f; font-size: 10px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; font-size: 12px; color: #3c4043; }}
    .legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 5px; vertical-align: -1px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ text-align: left; padding: 5px 6px; border-bottom: 1px solid #ecece5; }}
    th {{ color: #5f6368; font-weight: 600; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Orbit Wars RL Training</h1>
      <div class="subtle">Auto-refreshes every 5 seconds from {html.escape(str(log_dir))}</div>
    </div>
    <div class="subtle">{html.escape(eval_summary)}</div>
  </header>
  <p class="subtle">{html.escape(bc_summary)}</p>
  <main class="grid">
    {_chart(bc_metrics, "BC Pretrain Loss", ["loss"], colors, smooth=1, points=True)}
    {_chart(bc_metrics, "BC Pretrain Accuracy", ["accuracy"], colors, y_domain=(0.0, 1.0), smooth=1, points=True)}
    {_chart(metrics, "Return", ["mean_return_25"], colors, smooth=5, vlines=ppo_markers)}
    {_chart(metrics, "Reward Mean", ["reward_mean"], colors, smooth=8, vlines=ppo_markers)}
    {_chart(metrics, "Entropy", ["entropy", "entropy_launch", "entropy_source", "entropy_target", "entropy_send", "ent_coef"], colors, smooth=5, vlines=ppo_markers)}
    {_chart(metrics, "PPO Stability", ["clip_frac", "approx_kl", "explained_var"], colors, y_domain=(-0.05, 1.0), smooth=5, vlines=ppo_markers)}
    {_chart(metrics, "Action Rates", ["noop_rate", "launch_rate"], colors, y_domain=(0.0, 1.0), smooth=5, vlines=ppo_markers)}
    {_chart(metrics, "Average Send Bin", ["avg_send_bin"], colors, y_domain=(0.0, 3.0), smooth=5, vlines=ppo_markers)}
    {_chart(evals, "Evaluation Win Rate", [key for key in sorted(latest_eval) if key.startswith("win_rate_")] or ["win_rate_noop", "win_rate_random", "win_rate_nearest"], colors, y_domain=(0.0, 1.0), points=True, vlines=ppo_markers)}
    {_chart(evals, "Evaluation Aggregate", ["eval_score"], colors, y_domain=(0.0, 1.0), points=True, vlines=ppo_markers)}
    {_chart(evals, "Evaluation Margin", [key.replace("win_rate_", "avg_margin_") for key in sorted(latest_eval) if key.startswith("win_rate_")] or ["avg_margin_noop", "avg_margin_random", "avg_margin_nearest"], colors, points=True, vlines=ppo_markers)}
    {_chart(metrics, "Reward Components", ["reward_score_delta", "reward_score_share_delta", "reward_production_delta", "reward_production_share_delta", "reward_economy_delta", "reward_terminal"], colors, smooth=8, vlines=ppo_markers)}
    <section class="panel">
      <h2>Latest Metrics</h2>
      {_latest_table(latest)}
    </section>
    <section class="panel">
      <h2>Latest BC Metrics</h2>
      {_latest_table(latest_bc)}
    </section>
  </main>
</body>
</html>
"""
    out = log_dir / "training_report.html"
    out.write_text(body, encoding="utf-8")
    return out
