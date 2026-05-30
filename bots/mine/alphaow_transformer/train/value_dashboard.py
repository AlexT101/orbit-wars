from __future__ import annotations

import html
import json
import math
import sys
import time
from pathlib import Path


COLORS = ["#3b82f6", "#16a34a", "#ef4444", "#8b5cf6", "#f59e0b", "#0f766e"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def num(v, default=0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def fmt(v) -> str:
    v = num(v)
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}"


def bounds(values: list[float], fixed: tuple[float, float] | None = None):
    if fixed:
        return fixed
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-9:
        pad = max(abs(hi) * 0.1, 1e-3)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def series(rows: list[dict], key: str):
    xs, ys = [], []
    for row in rows:
        if key not in row:
            continue
        xs.append(num(row.get("epoch")))
        ys.append(num(row.get(key)))
    return xs, ys


def points(xs, ys, w, h, pl, pr, pt, pb, xd, yd):
    x0, x1 = xd
    y0, y1 = yd
    if abs(x1 - x0) < 1e-9:
        x1 = x0 + 1.0
    if abs(y1 - y0) < 1e-9:
        y1 = y0 + 1.0
    out = []
    for x, y in zip(xs, ys):
        px = pl + (x - x0) / (x1 - x0) * (w - pl - pr)
        py = h - pb - (y - y0) / (y1 - y0) * (h - pt - pb)
        out.append((px, py))
    return out


def chart(
    rows: list[dict],
    title: str,
    keys: list[str],
    fixed_y=None,
    height=280,
    colors: list[str] | None = None,
    widths: list[float] | None = None,
    opacities: list[float] | None = None,
):
    w, h = 760, height
    pl, pr, pt, pb = 58, 18, 22, 34
    colors = colors or COLORS
    widths = widths or [2.3] * len(keys)
    opacities = opacities or [1.0] * len(keys)
    all_x, all_y = [], []
    data = []
    for key in keys:
        xs, ys = series(rows, key)
        data.append((key, xs, ys))
        all_x.extend(xs)
        all_y.extend(ys)
    xd = bounds(all_x)
    yd = bounds(all_y, fixed_y)
    parts = [
        '<section class="card">',
        f"<h2>{html.escape(title)}</h2>",
        f'<svg viewBox="0 0 {w} {h}" style="height:{h}px">',
        f'<line class="axis" x1="{pl}" y1="{h-pb}" x2="{w-pr}" y2="{h-pb}"></line>',
        f'<line class="axis" x1="{pl}" y1="{pt}" x2="{pl}" y2="{h-pb}"></line>',
    ]
    for i in range(5):
        yv = yd[0] + (yd[1] - yd[0]) * i / 4.0
        y = h - pb - i / 4.0 * (h - pt - pb)
        parts.append(f'<line class="grid" x1="{pl}" y1="{y:.1f}" x2="{w-pr}" y2="{y:.1f}"></line>')
        parts.append(f'<text class="tick" x="{pl-8}" y="{y+4:.1f}" text-anchor="end">{fmt(yv)}</text>')
    for i, (key, xs, ys) in enumerate(data):
        pts = points(xs, ys, w, h, pl, pr, pt, pb, xd, yd)
        color = colors[i % len(colors)]
        width = widths[i] if i < len(widths) else widths[-1]
        opacity = opacities[i] if i < len(opacities) else opacities[-1]
        if len(pts) >= 2:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(
                f'<polyline points="{poly}" fill="none" stroke="{color}" opacity="{opacity:.2f}" '
                f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"></polyline>'
            )
        elif len(pts) == 1:
            x, y = pts[0]
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" opacity="{opacity:.2f}"></circle>')
    legends = []
    for i, (key, _xs, ys) in enumerate(data):
        if ys:
            color = colors[i % len(colors)]
            opacity = opacities[i] if i < len(opacities) else opacities[-1]
            legends.append(
                f'<span><i style="background:{color};opacity:{opacity:.2f}"></i>{html.escape(key)}: {fmt(ys[-1])}</span>'
            )
    parts.append("</svg>")
    parts.append(f'<div class="legend">{"".join(legends)}</div>')
    parts.append("</section>")
    return "\n".join(parts)


def table(rows: list[dict]):
    latest = list(reversed(rows[-12:]))
    parts = [
        '<section class="card wide">',
        "<h2>Epoch History</h2>",
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Epoch</th><th>Train</th><th>Val</th><th>Sign</th><th>Open</th><th>Mid</th><th>End</th><th>Rate</th><th>Elapsed</th>"
        "</tr></thead><tbody>",
    ]
    for row in latest:
        parts.append(
            "<tr>"
            f"<td>{int(num(row.get('epoch')))}</td>"
            f"<td>{fmt(row.get('train_loss'))}</td>"
            f"<td>{fmt(row.get('val_loss'))}</td>"
            f"<td>{fmt(row.get('sign'))}</td>"
            f"<td>{fmt(row.get('bucket_opener_sign'))}</td>"
            f"<td>{fmt(row.get('bucket_mid_sign'))}</td>"
            f"<td>{fmt(row.get('bucket_end_sign'))}</td>"
            f"<td>{fmt(row.get('samples_per_sec'))}</td>"
            f"<td>{fmt(row.get('total_seconds'))}s</td>"
            "</tr>"
        )
    parts.append("</tbody></table></div></section>")
    return "\n".join(parts)


def render(metrics_path: Path, out_path: Path):
    rows = load_jsonl(metrics_path)
    latest = rows[-1] if rows else {}
    title = html.escape(str(latest.get("run_name") or "AlphaOW Value Net"))
    baseline_keys = ["base_ship", "base_ext", "base_prod"]
    for row in rows:
        for key in row:
            if key.startswith("baseline_") and key not in baseline_keys:
                baseline_keys.append(key)
    sign_keys = ["sign"] + baseline_keys
    muted_colors = ["#94a3b8", "#a8a29e", "#c4b5a5", "#a1a1aa", "#b8b3aa", "#9ca3af", "#cbd5e1"]
    sign_colors = ["#2563eb"] + muted_colors
    sign_widths = [3.8] + [1.8] * len(baseline_keys)
    sign_opacities = [1.0] + [0.58] * len(baseline_keys)

    def phase_chart(label: str, bucket: str) -> str:
        keys = [f"bucket_{bucket}_sign"] + [f"bucket_{bucket}_{key}" for key in baseline_keys]
        present = []
        for key in keys:
            if any(key in row for row in rows):
                present.append(key)
        colors = ["#2563eb"] + muted_colors
        widths = [3.8] + [1.8] * max(0, len(present) - 1)
        opacities = [1.0] + [0.58] * max(0, len(present) - 1)
        return chart(rows, label, present, fixed_y=(0.5, 1), colors=colors, widths=widths, opacities=opacities)

    cards = [
        ("Epoch", latest.get("epoch", 0)),
        ("Train Loss", latest.get("train_loss", 0)),
        ("Val Loss", latest.get("val_loss", 0)),
        ("Sign", latest.get("sign", 0)),
        ("Best Val", latest.get("best_val", 0)),
        ("Samples/s", latest.get("samples_per_sec", 0)),
    ]
    summary = "".join(f"<div><b>{html.escape(k)}</b><strong>{fmt(v)}</strong></div>" for k, v in cards)
    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="20">
<title>{title}</title>
<style>
body{{margin:0;background:#f7f7f4;color:#252525;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}}
main{{padding:18px;display:grid;grid-template-columns:repeat(3,minmax(320px,1fr));gap:18px}}
header{{grid-column:1/-1;display:flex;align-items:end;justify-content:space-between}}
h1{{font-size:24px;margin:0}} h2{{font-size:18px;margin:0 0 10px}}
.muted{{color:#777;font-size:13px}} .summary{{grid-column:1/-1;display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}
.summary div,.card{{background:white;border:1px solid #ddd9d0;border-radius:8px;padding:16px;box-shadow:0 1px 2px #0000000a}}
.summary b{{display:block;color:#666;font-size:12px;font-weight:600}} .summary strong{{font-size:22px}}
.card svg{{width:100%;display:block;background:#fbfbf9;border-radius:6px}} .wide{{grid-column:1/-1}}
.axis{{stroke:#c9c2b8;stroke-width:1}} .grid{{stroke:#ece7df;stroke-width:1}} .tick{{fill:#777;font-size:11px}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;color:#555;font-size:13px}} .legend i{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
.table-wrap{{max-height:300px;overflow:auto}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:8px;border-bottom:1px solid #eee;text-align:right}} th:first-child,td:first-child{{text-align:left}}
@media(max-width:1100px){{main{{grid-template-columns:1fr}}.summary{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body><main>
<header><h1>{title}</h1><div class="muted">Updated {html.escape(time.strftime('%Y-%m-%d %H:%M:%S'))}</div></header>
<section class="summary">{summary}</section>
{chart(rows, "Loss", ["train_loss", "val_loss"])}
{chart(rows, "Sign Accuracy vs Baselines", sign_keys, fixed_y=(0.5, 1), colors=sign_colors, widths=sign_widths, opacities=sign_opacities)}
{phase_chart("Opener Sign Accuracy", "opener")}
{phase_chart("Midgame Sign Accuracy", "mid")}
{phase_chart("Endgame Sign Accuracy", "end")}
{table(rows)}
</main></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: value_dashboard.py metrics.jsonl dashboard.html", file=sys.stderr)
        return 2
    render(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
