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


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


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
    height: int = 260,
    extra_class: str = "",
    embedded: bool = False,
) -> str:
    width = 760
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

    tag = "div" if embedded else "section"
    section_class = ("embedded-chart" if embedded else "chart") + (f" {extra_class}" if extra_class else "")
    parts = [
        f'<{tag} class="{html.escape(section_class)}">',
        f"<h2>{html.escape(title)}</h2>",
        f'<svg viewBox="0 0 {width} {height}" role="img" style="height:{height}px">',
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
    for marker_index, marker in enumerate(vlines or []):
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
        marker_type = str(marker.get("marker_type", ""))
        if marker_type == "phase_switch" and x <= pad_l + 0.5:
            x = pad_l + 12 + (marker_index % 3) * 5
        label = html.escape(str(marker.get("label", "marker")))
        line_class = "marker phase-marker" if marker_type == "phase_switch" else "marker"
        label_class = "marker-label phase-marker-label" if marker_type == "phase_switch" else "marker-label"
        parts.append(
            f'<line class="{line_class}" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height - pad_b}"></line>'
        )
        label_y = pad_t + 12 + (marker_index % 5) * 14
        parts.append(
            f'<text class="{label_class}" x="{x + 5:.1f}" y="{label_y}" transform="rotate(90 {x + 5:.1f},{label_y})">{label}</text>'
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
    parts.append(f"</{tag}>")
    return "\n".join(parts)


def _dual_axis_chart(
    rows: list[dict],
    title: str,
    left_key: str,
    right_key: str,
    colors: list[str],
    *,
    left_smooth: int = 1,
    right_smooth: int = 1,
    vlines: list[dict] | None = None,
    height: int = 320,
    extra_class: str = "",
) -> str:
    width = 760
    pad_l = 60
    pad_r = 72
    pad_t = 24
    pad_b = 38
    left_xs, left_ys = _series(rows, left_key, smooth=left_smooth)
    right_xs, right_ys = _series(rows, right_key, smooth=right_smooth)
    all_x = [*left_xs, *right_xs]
    for marker in vlines or []:
        try:
            all_x.append(float(marker["step"]))
        except (KeyError, TypeError, ValueError):
            continue
    x_domain = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    left_domain = _nice_bounds(left_ys, None)
    right_domain = _nice_bounds(right_ys, None)

    section_class = "chart" + (f" {extra_class}" if extra_class else "")
    left_color = colors[0 % len(colors)]
    right_color = colors[1 % len(colors)]
    parts = [
        f'<section class="{html.escape(section_class)}">',
        f"<h2>{html.escape(title)}</h2>",
        f'<svg viewBox="0 0 {width} {height}" role="img" style="height:{height}px">',
        f'<line class="axis" x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}"></line>',
        f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}"></line>',
        f'<line class="axis" x1="{width - pad_r}" y1="{pad_t}" x2="{width - pad_r}" y2="{height - pad_b}"></line>',
    ]
    left_min, left_max = left_domain
    right_min, right_max = right_domain
    for i in range(5):
        y = height - pad_b - i / 4.0 * (height - pad_t - pad_b)
        left_value = left_min + (left_max - left_min) * i / 4.0
        right_value = right_min + (right_max - right_min) * i / 4.0
        parts.append(f'<line class="gridline" x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"></line>')
        parts.append(
            f'<text class="tick" fill="{left_color}" x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end">{_format_num(left_value)}</text>'
        )
        parts.append(
            f'<text class="tick" fill="{right_color}" x="{width - pad_r + 8}" y="{y + 4:.1f}" text-anchor="start">{_format_num(right_value)}</text>'
        )
    if all_x:
        parts.append(f'<text class="tick" x="{pad_l}" y="{height - 12}" text-anchor="middle">{int(x_domain[0])}</text>')
        parts.append(f'<text class="tick" x="{width - pad_r}" y="{height - 12}" text-anchor="middle">{int(x_domain[1])}</text>')

    for marker_index, marker in enumerate(vlines or []):
        try:
            marker_x = float(marker["step"])
        except (KeyError, TypeError, ValueError):
            continue
        x = _points(
            [marker_x],
            [left_min],
            width,
            height,
            pad_l,
            pad_r,
            pad_t,
            pad_b,
            x_domain,
            left_domain,
        )[0][0]
        marker_type = str(marker.get("marker_type", ""))
        if marker_type == "phase_switch" and x <= pad_l + 0.5:
            x = pad_l + 12 + (marker_index % 3) * 5
        label = html.escape(str(marker.get("label", "marker")))
        line_class = "marker phase-marker" if marker_type == "phase_switch" else "marker"
        label_class = "marker-label phase-marker-label" if marker_type == "phase_switch" else "marker-label"
        parts.append(
            f'<line class="{line_class}" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height - pad_b}"></line>'
        )
        label_y = pad_t + 12 + (marker_index % 5) * 14
        parts.append(
            f'<text class="{label_class}" x="{x + 5:.1f}" y="{label_y}" transform="rotate(90 {x + 5:.1f},{label_y})">{label}</text>'
        )

    left_line = _polyline(left_xs, left_ys, width, height, pad_l, pad_r, pad_t, pad_b, x_domain, left_domain)
    if left_line:
        parts.append(
            f'<polyline points="{left_line}" fill="none" stroke="{left_color}" stroke-width="2.2" '
            'stroke-linejoin="round" stroke-linecap="round"></polyline>'
        )
    right_line = _polyline(right_xs, right_ys, width, height, pad_l, pad_r, pad_t, pad_b, x_domain, right_domain)
    if right_line:
        parts.append(
            f'<polyline points="{right_line}" fill="none" stroke="{right_color}" stroke-width="2.2" '
            'stroke-linejoin="round" stroke-linecap="round"></polyline>'
        )

    legend = []
    if left_ys:
        legend.append(f'<span><i style="background:{left_color}"></i>{html.escape(left_key)} left: {_format_num(left_ys[-1])}</span>')
    if right_ys:
        legend.append(f'<span><i style="background:{right_color}"></i>{html.escape(right_key)} right: {_format_num(right_ys[-1])}</span>')
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
            "train_loss",
            "val_loss",
            "train_pair_accuracy",
            "val_pair_accuracy",
            "train_source_accuracy",
            "val_source_accuracy",
            "train_target_accuracy",
            "val_target_accuracy",
            "train_pair_top5",
            "val_pair_top5",
            "teacher",
            "samples",
            "train_samples",
            "val_samples",
            "noop_fraction",
            "launch_fraction",
            "unique_labels",
            "unique_pairs",
            "target_name",
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


def _phase_badge(text: str) -> str:
    return f'<span class="badge">{html.escape(text)}</span>'


def _curriculum_context(log_dir: Path) -> str:
    config = load_json(log_dir / "curriculum_config.json")
    state = load_json(log_dir / "curriculum_state.json")
    events = load_jsonl(log_dir / "curriculum_events.jsonl")
    metrics = load_jsonl(log_dir / "metrics.jsonl")
    phases = config.get("phases", []) if isinstance(config.get("phases", []), list) else []
    if not phases and not events:
        return ""

    current_idx = int(state.get("phase_index", 0) or 0) if state else 0
    last_phase_event = next(
        (row for row in reversed(events) if isinstance(row.get("phase"), dict)),
        {},
    )
    phase = phases[current_idx] if current_idx < len(phases) else last_phase_event.get("phase", {})
    if not isinstance(phase, dict):
        phase = {}

    last_gate = next((row for row in reversed(events) if row.get("kind") == "gate"), {})
    gate = last_gate.get("gate", {}) if isinstance(last_gate.get("gate"), dict) else {}
    gate_events = [event for event in events if event.get("kind") == "gate"]
    opponent_rows = []
    for name, result in sorted((gate.get("opponents") or {}).items()):
        if not isinstance(result, dict):
            continue
        opponent_rows.append(
            "<tr>"
            f"<th>{html.escape(str(name))}</th>"
            f"<td>{float(result.get('win_rate', 0.0)):.1%}</td>"
            f"<td>{int(result.get('wins', 0))}/{int(result.get('games', 0))}</td>"
            "</tr>"
        )
    older_gate_events = gate_events[:-1] if gate_events and gate_events[-1] is last_gate else gate_events
    for event in reversed(older_gate_events):
        event_gate = event.get("gate", {})
        if not isinstance(event_gate, dict):
            continue
        for name, result in sorted((event_gate.get("opponents") or {}).items()):
            if not isinstance(result, dict):
                continue
            opponent_rows.append(
                '<tr class="muted-row">'
                f"<th>{html.escape(str(name))}</th>"
                f"<td>{float(result.get('win_rate', 0.0)):.1%}</td>"
                f"<td>{int(result.get('wins', 0))}/{int(result.get('games', 0))}</td>"
                "</tr>"
            )
    gate_table = (
        '<div class="scroll-table curriculum-gates-table"><table><tr><th>Gate Bot</th><th>WR</th><th>Wins</th></tr>'
        + "".join(opponent_rows)
        + "</table></div>"
        if opponent_rows
        else '<p class="subtle">No curriculum gate has completed yet.</p>'
    )

    phase_rows = []
    for p in phases:
        idx = int(p.get("index", len(phase_rows)) or 0)
        cls = ' class="current-row"' if idx == current_idx else ""
        phase_rows.append(
            f"<tr{cls}>"
            f"<th>{idx}</th>"
            f"<td>{html.escape(str(p.get('name', '?')))}</td>"
            f"<td>{float(p.get('threshold', 0.0)):.1%}</td>"
            f"<td>{html.escape(', '.join(str(x) for x in p.get('gate_opponents', [])))}</td>"
            "</tr>"
        )
    phase_table = (
        "<table><tr><th>Phase</th><th>Name</th><th>Gate</th><th>Opponents</th></tr>"
        + "".join(phase_rows)
        + "</table>"
        if phase_rows
        else ""
    )

    train_badges = "".join(_phase_badge(str(x)) for x in phase.get("train_opponents", []))
    gate_badges = "".join(_phase_badge(str(x)) for x in phase.get("gate_opponents", []))
    args = config.get("args") or {}
    total_budget = int(args.get("total_budget_steps", 0) or 0)
    chunk_steps = int(args.get("chunk_steps", 0) or 0)
    steps_requested = int(state.get("steps_requested", 0) or 0) if state else 0
    latest_metric = metrics[-1] if metrics else {}
    latest_step = int(latest_metric.get("step", 0) or 0)
    latest_update = int(latest_metric.get("update", 0) or 0)
    latest_sps = float(latest_metric.get("sps", 0.0) or 0.0)
    phase_start = max(0, steps_requested - (steps_requested % max(1, chunk_steps)))
    chunk_done = min(max(0, latest_step - phase_start), max(1, chunk_steps)) if chunk_steps else 0
    phase_progress = min(1.0, max(0.0, chunk_done / max(1, chunk_steps))) if chunk_steps else 0.0
    budget_progress = min(1.0, max(0.0, steps_requested / max(1, total_budget))) if total_budget else 0.0
    promotion_count = sum(1 for event in events if event.get("kind") == "phase_promoted")
    last_event = events[-1].get("kind", "none") if events else "none"
    summary = f"""
    <section class="panel curriculum-panel wide">
      <h2>Curriculum</h2>
      <div class="summary-grid">
        <div><span class="label">Current Phase</span><strong>{html.escape(str(phase.get("index", current_idx)))} · {html.escape(str(phase.get("name", "?")))}</strong></div>
        <div><span class="label">Gate Threshold</span><strong>{float(phase.get("threshold", 0.0)):.1%}</strong></div>
        <div><span class="label">Gate Mode</span><strong>{html.escape(str(args.get("gate_mode", "min")))}</strong></div>
        <div><span class="label">Promotions</span><strong>{promotion_count}</strong></div>
        <div><span class="label">Latest PPO</span><strong>step {latest_step} · update {latest_update}</strong></div>
        <div><span class="label">Speed</span><strong>{latest_sps:.1f} SPS</strong></div>
        <div><span class="label">Last Event</span><strong>{html.escape(str(last_event))}</strong></div>
      </div>
      <div class="progress-block">
        <div><span class="label">Budget</span><span>{steps_requested}/{total_budget or "?"} requested</span></div>
        <div class="bar"><i style="width:{budget_progress * 100:.1f}%"></i></div>
        <div><span class="label">Current Chunk</span><span>{chunk_done}/{chunk_steps or "?"} PPO steps</span></div>
        <div class="bar"><i style="width:{phase_progress * 100:.1f}%"></i></div>
      </div>
      <div class="badge-row"><span class="label">Training</span>{train_badges}</div>
      <div class="badge-row"><span class="label">Gate</span>{gate_badges}</div>
      <div class="curriculum-grid">
        <div>
          <h2>Curriculum Gates</h2>
          <p class="subtle">mean {float(gate.get("mean_win_rate", 0.0)):.1%} · min {float(gate.get("min_win_rate", 0.0)):.1%} · {'passed' if gate.get("passed") else 'not passed'}</p>
          {gate_table}
        </div>
        <div>
          <h2>Phase Ladder</h2>
          {phase_table}
        </div>
      </div>
    </section>
    """
    return summary


def _curriculum_switch_markers(log_dir: Path) -> list[dict]:
    events = load_jsonl(log_dir / "curriculum_events.jsonl")
    markers = []
    seen_phases = set()
    for event in events:
        if event.get("kind") != "phase_chunk_start":
            continue
        phase = event.get("phase", {})
        state = event.get("state", {})
        if not isinstance(phase, dict) or not isinstance(state, dict):
            continue
        try:
            phase_index = int(phase.get("index", 0))
            step = int(state.get("steps_requested", 0))
        except (TypeError, ValueError):
            continue
        if phase_index <= 0 or phase_index in seen_phases:
            continue
        seen_phases.add(phase_index)
        name = str(phase.get("name", f"phase {phase_index}"))
        markers.append({"step": step, "label": f"phase {phase_index}: {name}", "marker_type": "phase_switch"})
    return markers


def write_training_report(log_dir: Path) -> Path:
    metrics = load_jsonl(log_dir / "metrics.jsonl")
    evals = load_jsonl(log_dir / "eval.jsonl")
    bc_metrics = load_jsonl(log_dir / "bc_metrics.jsonl")
    imitation_metrics = load_jsonl(log_dir / "imitation_metrics.jsonl")
    phase_events = load_jsonl(log_dir / "phase_events.jsonl")
    latest = metrics[-1] if metrics else {}
    latest_eval = evals[-1] if evals else {}
    latest_bc = next((row for row in reversed(bc_metrics) if row.get("phase") == "bc"), {})
    latest_imitation = next((row for row in reversed(imitation_metrics) if row.get("phase") == "imitation"), {})
    colors = ["#2f80ed", "#27ae60", "#eb5757", "#9b51e0", "#f2994a", "#00a3a3"]
    curriculum_panel = _curriculum_context(log_dir)
    ppo_markers = [
        row
        for row in phase_events
        if row.get("kind") in {"pretrain_end", "ppo_start"} and "step" in row
    ]
    curriculum_markers = _curriculum_switch_markers(log_dir)
    training_markers = [*ppo_markers, *curriculum_markers]

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
    imitation_summary = "No source-target imitation log yet."
    if latest_imitation:
        imitation_summary = (
            f"Imitation {latest_imitation.get('target_name') or latest_imitation.get('target_mode', '?')} · "
            f"epoch {latest_imitation.get('epoch')} · "
            f"pair {float(latest_imitation.get('val_pair_accuracy', 0.0)):.1%} · "
            f"source {float(latest_imitation.get('val_source_accuracy', 0.0)):.1%} · "
            f"target {float(latest_imitation.get('val_target_accuracy', 0.0)):.1%}"
        )
    reward_keys = [key for key in sorted(latest) if key.startswith("reward_")]
    if not reward_keys:
        reward_keys = [
            "reward_control",
            "reward_score_increase",
            "reward_production_increase",
            "reward_planet_increase",
            "reward_enemy_planet_capture",
            "reward_terminal",
            "reward_terminal_time",
        ]

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
    .wide {{ grid-column: 1 / -1; }}
    .curriculum-panel {{ min-height: 420px; }}
    details.panel {{ padding: 0; overflow: hidden; }}
    details.panel > summary {{ cursor: pointer; list-style: none; padding: 14px; font-weight: 700; }}
    details.panel > summary::-webkit-details-marker {{ display: none; }}
    details.panel > summary::before {{ content: "▸"; display: inline-block; margin-right: 8px; transition: transform 0.15s ease; }}
    details.panel[open] > summary::before {{ transform: rotate(90deg); }}
    details.panel > .details-body {{ padding: 0 14px 14px; }}
    svg {{ width: 100%; height: 260px; display: block; background: #fbfbf8; border-radius: 6px; }}
    .axis {{ stroke: #b8b8ad; stroke-width: 1; }}
    .gridline {{ stroke: #e4e4dc; stroke-width: 1; }}
    .marker {{ stroke: #111827; stroke-width: 1.5; stroke-dasharray: 5 5; opacity: 0.7; }}
    .marker-label {{ fill: #111827; font-size: 10px; }}
    .phase-marker {{ stroke: #f2994a; stroke-width: 2.4; opacity: 0.95; }}
    .phase-marker-label {{ fill: #b35c00; font-weight: 700; }}
    .tick {{ fill: #74766f; font-size: 10px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; font-size: 12px; color: #3c4043; }}
    .legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 5px; vertical-align: -1px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ text-align: left; padding: 5px 6px; border-bottom: 1px solid #ecece5; }}
    th {{ color: #5f6368; font-weight: 600; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 8px 0 12px; }}
    .summary-grid strong {{ display: block; font-size: 18px; margin-top: 3px; }}
    .label {{ color: #5f6368; font-size: 12px; font-weight: 600; margin-right: 8px; }}
    .badge-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 8px 0; }}
    .badge {{ display: inline-block; border: 1px solid #d8d8d0; background: #fbfbf8; border-radius: 999px; padding: 3px 8px; font-size: 12px; }}
    .split {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; margin-top: 12px; }}
    .curriculum-grid {{ display: grid; grid-template-columns: minmax(360px, 0.9fr) minmax(520px, 1.1fr); gap: 18px; margin-top: 12px; align-items: start; }}
    @media (max-width: 1050px) {{ .curriculum-grid {{ grid-template-columns: 1fr; }} }}
    .current-row {{ background: #fff8dc; }}
    .muted-row {{ color: #8a8c86; opacity: 0.68; }}
    .scroll-table {{ max-height: 220px; overflow: auto; border-top: 1px solid #ecece5; border-bottom: 1px solid #ecece5; }}
    .curriculum-gates-table {{ max-height: 300px; }}
    .progress-block {{ display: grid; gap: 6px; margin: 10px 0 14px; }}
    .bar {{ height: 9px; background: #ecece5; border-radius: 999px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; background: linear-gradient(90deg, #2f80ed, #27ae60); }}
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
  <p class="subtle">{html.escape(bc_summary)}<br>{html.escape(imitation_summary)}</p>
  <main class="grid">
    {curriculum_panel}
    <details class="panel wide">
      <summary>Source-Target Imitation</summary>
      <div class="details-body grid">
        {_chart(imitation_metrics, "Imitation Loss", ["train_loss", "val_loss", "train_pair_loss", "val_pair_loss"], colors, smooth=1, points=True)}
        {_chart(imitation_metrics, "Pair Accuracy", ["train_pair_accuracy", "val_pair_accuracy", "train_pair_top5", "val_pair_top5"], colors, y_domain=(0.0, 1.0), smooth=1, points=True)}
        {_chart(imitation_metrics, "Source And Target Accuracy", ["train_source_accuracy", "val_source_accuracy", "train_target_accuracy", "val_target_accuracy"], colors, y_domain=(0.0, 1.0), smooth=1, points=True)}
        {_chart(imitation_metrics, "Imitation Throughput", ["samples_per_sec", "epoch_seconds"], colors, smooth=1, points=True)}
        <section class="panel">
          <h2>Latest Imitation Metrics</h2>
          {_latest_table(latest_imitation)}
        </section>
      </div>
    </details>
    <details class="panel wide">
      <summary>BC Pretrain</summary>
      <div class="details-body grid">
        {_chart(bc_metrics, "BC Pretrain Loss", ["loss"], colors, smooth=1, points=True)}
        {_chart(bc_metrics, "BC Pretrain Accuracy", ["accuracy"], colors, y_domain=(0.0, 1.0), smooth=1, points=True)}
        <section class="panel">
          <h2>Latest BC Metrics</h2>
          {_latest_table(latest_bc)}
        </section>
      </div>
    </details>
    {_dual_axis_chart(metrics, "Return And Reward", "mean_return_25", "reward_mean", colors, left_smooth=5, right_smooth=8, vlines=training_markers)}
    {_chart(metrics, "Entropy", ["entropy", "entropy_launch", "entropy_source", "entropy_target", "entropy_send", "ent_coef"], colors, smooth=5, vlines=training_markers)}
    {_chart(metrics, "PPO Stability", ["clip_frac", "approx_kl", "explained_var"], colors, y_domain=(-0.05, 1.0), smooth=5, vlines=training_markers)}
    {_chart(metrics, "Action Rates", ["noop_rate", "launch_rate"], colors, y_domain=(0.0, 1.0), smooth=5, vlines=ppo_markers)}
    {_chart(metrics, "Average Send Bin", ["avg_send_bin"], colors, y_domain=(0.0, 3.0), smooth=5, vlines=ppo_markers)}
    {_chart(evals, "Evaluation Win Rate", [key for key in sorted(latest_eval) if key.startswith("win_rate_")] or ["win_rate_noop", "win_rate_random", "win_rate_nearest"], colors, y_domain=(0.0, 1.0), points=True, vlines=ppo_markers)}
    {_chart(evals, "Evaluation Aggregate", ["eval_score"], colors, y_domain=(0.0, 1.0), points=True, vlines=ppo_markers)}
    {_chart(evals, "Evaluation Margin", [key.replace("win_rate_", "avg_margin_") for key in sorted(latest_eval) if key.startswith("win_rate_")] or ["avg_margin_noop", "avg_margin_random", "avg_margin_nearest"], colors, points=True, vlines=ppo_markers)}
    {_chart(metrics, "Reward Components", reward_keys, colors, smooth=8, vlines=training_markers)}
    <section class="panel">
      <h2>Latest Metrics</h2>
      {_latest_table(latest)}
    </section>
  </main>
</body>
</html>
"""
    out = log_dir / "training_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out
