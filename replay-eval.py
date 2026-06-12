#!/usr/bin/env python3
"""Orbit Wars ladder-replay stat analyzer.

Usage:
    python replay-eval.py 2p "Team Name"
    python replay-eval.py 4p "Team Name" [--zip ladder_replays/replays_6_07.zip]

Finds the most recent replay zip in ladder_replays/, filters to games containing
the given team at the chosen player count, computes distribution stats for several
per-fleet / per-turn metrics (for that team's player only), and opens a self-contained
dark-mode HTML report in the browser. Stdlib only.
"""

import argparse
import json
import math
import re
import statistics
import sys
import tempfile
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LADDER_DIR = ROOT / "ladder_replays"

BOARD_SIZE = 100.0
SUN_CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
ARRIVAL_EPS = 1.5      # slack added to planet radius when matching arrival
EDGE_EPS = 2.0         # how close to the wall counts as "off-screen"
HIST_BINS = 12
INT_BIN_MAX_RANGE = 30   # integer metrics with <= this span get one bin per value


# --------------------------------------------------------------------------- #
# Zip selection
# --------------------------------------------------------------------------- #
def find_latest_zip(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"Zip not found: {p}")
        return p

    pat = re.compile(r"replays_(\d+)_(\d+)\.zip$")
    best = None
    for z in LADDER_DIR.glob("replays_*.zip"):
        m = pat.search(z.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if best is None or key > best[0]:
            best = (key, z)
    if best is None:
        sys.exit(f"No replays_*.zip found in {LADDER_DIR}")
    return best[1]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def classify_endpoint(last_pos, planets_at_end, comet_ids):
    """Return ('arrive', planet) | ('sun', None) | ('offscreen', None) | ('unknown', None)."""
    x, y = last_pos
    best = None
    for p in planets_at_end:
        pid, owner, px, py, radius = p[0], p[1], p[2], p[3], p[4]
        d = dist(x, y, px, py)
        if d <= radius + ARRIVAL_EPS:
            if best is None or d < best[1]:
                best = (p, d)
    if best is not None:
        return "arrive", best[0]
    if dist(x, y, *SUN_CENTER) <= SUN_RADIUS + ARRIVAL_EPS:
        return "sun", None
    if x <= EDGE_EPS or x >= BOARD_SIZE - EDGE_EPS or y <= EDGE_EPS or y >= BOARD_SIZE - EDGE_EPS:
        return "offscreen", None
    return "unknown", None


# --------------------------------------------------------------------------- #
# Per-replay extraction
# --------------------------------------------------------------------------- #
def extract_replay(replay, pid, acc):
    """Pull metric values for player `pid` out of one replay into `acc` (dict of lists)."""
    steps = replay["steps"]
    n_turns = len(steps)

    # Canonical world state per turn from player 0's full-visibility observation.
    world = []        # world[t] = {fleet_id: fleet_list}
    planets = []      # planets[t] = list of planet lists
    planet_idx = []   # planets[t] indexed by id
    for t in range(n_turns):
        obs = steps[t][0]["observation"]
        world.append({f[0]: f for f in obs.get("fleets", [])})
        pl = obs.get("planets", [])
        planets.append(pl)
        planet_idx.append({p[0]: p for p in pl})

    comet_ids = set(steps[0][0]["observation"].get("comet_planet_ids", []))

    def player_alive(t):
        # alive if owns a planet or has a fleet this turn
        if any(p[1] == pid for p in planets[t]):
            return True
        return any(f[1] == pid for f in world[t].values())

    for t in range(n_turns):
        entry = steps[t][pid]
        action = entry.get("action") or []
        if not isinstance(action, list):
            action = []

        if player_alive(t):
            acc["fleets_per_turn"].append(len(action))

        if not action:
            continue

        prev_ids = set(world[t - 1].keys()) if t > 0 else set()
        # New fleets owned by pid that appeared this turn, in id order (spawn order).
        new_fleets = sorted(
            (f for fid, f in world[t].items() if fid not in prev_ids and f[1] == pid),
            key=lambda f: f[0],
        )
        used = set()

        for mv in action:
            try:
                from_id, angle, ships = mv[0], mv[1], mv[2]
            except (TypeError, IndexError):
                continue
            acc["ships_per_fleet"].append(ships)

            # Fraction of the source planet's garrison that was committed. The
            # agent decided from the *previous* turn's state (obs[t-1]); that's
            # the garrison it actually had available before this turn's launches.
            if t > 0:
                src_prev = planet_idx[t - 1].get(from_id)
                if src_prev is not None and src_prev[5] > 0:
                    acc["ships_pct_source"].append(100.0 * ships / src_prev[5])

            # Match the launch to its spawned fleet (same from_planet_id + ships).
            match = None
            for f in new_fleets:
                if f[0] in used:
                    continue
                if f[5] == from_id and f[6] == ships:
                    match = f
                    break
            if match is None:  # looser match: from_planet only
                for f in new_fleets:
                    if f[0] in used:
                        continue
                    if f[5] == from_id:
                        match = f
                        break
            if match is None:
                continue
            used.add(match[0])
            fid = match[0]

            # Trace forward until it vanishes.
            t_end = None
            last_pos = (match[2], match[3])
            for tt in range(t + 1, n_turns):
                f = world[tt].get(fid)
                if f is None:
                    t_end = tt
                    break
                last_pos = (f[2], f[3])
            if t_end is None:
                continue  # still alive at game end; ignore for travel metrics

            kind, target = classify_endpoint(last_pos, planets[t_end - 1], comet_ids)

            # Only fleets that actually reach a planet contribute to distance and
            # travel-time stats. Fleets that fly off-map or hit the sun are dropped.
            if kind == "arrive":
                src = planet_idx[t].get(from_id)
                if src is not None:
                    d = dist(src[2], src[3], target[2], target[3])
                    acc["attack_distance"].append(d)
                acc["travel_time"].append(t_end - t)


# --------------------------------------------------------------------------- #
# Metric definitions
# --------------------------------------------------------------------------- #
METRICS = [
    ("ships_per_fleet", "Ships sent in a fleet", "ships"),
    ("ships_pct_source", "Ships in fleet (% of source)", "%"),
    ("fleets_per_turn", "Fleets sent in a turn", "fleets/turn"),
    ("attack_distance", "Distance between planets (straight-line source→target)", "units"),
    ("travel_time", "Travel time to arrive (turns)", "turns"),
]


def summarize(name, label, unit, values):
    n = len(values)
    out = {"name": label, "unit": unit, "n": n}
    if n == 0:
        out.update(min=None, max=None, mean=None, median=None, variance=None,
                   bins=[], tick_mode="center", ticks=[])
        return out
    vmin, vmax = min(values), max(values)
    out.update(
        min=vmin,
        max=vmax,
        mean=statistics.mean(values),
        median=statistics.median(values),
        variance=statistics.pvariance(values) if n > 1 else 0.0,
    )
    # Histogram. Each bar gets one tick label (a clean number) computed here so
    # the front-end just renders provided strings.
    is_int = all(float(v).is_integer() for v in values)

    if vmax == vmin:
        # Single value: one centered bar labeled with the value itself.
        out["bins"] = [n]
        out["tick_mode"] = "center"
        out["ticks"] = [_fmt_tick(vmin, 0)]
    elif is_int and (vmax - vmin) <= INT_BIN_MAX_RANGE:
        # One bin per integer value — each bar IS an exact value, so center-label it.
        lo, hi = int(vmin), int(vmax)
        counts = [0] * (hi - lo + 1)
        for v in values:
            counts[int(v) - lo] += 1
        out["bins"] = counts
        out["tick_mode"] = "center"
        out["ticks"] = [str(lo + i) for i in range(hi - lo + 1)]
    else:
        # Ranged buckets: label the bin *boundaries* (n+1 edges) so each bar
        # clearly reads as a range, not a single point.
        width = (vmax - vmin) / HIST_BINS
        counts = [0] * HIST_BINS
        for v in values:
            b = int((v - vmin) / width)
            if b >= HIST_BINS:
                b = HIST_BINS - 1
            counts[b] += 1
        dec = 0 if width >= 3 else (1 if width >= 0.3 else 2)
        out["bins"] = counts
        out["tick_mode"] = "edge"
        out["ticks"] = [_fmt_tick(vmin + i * width, dec) for i in range(HIST_BINS + 1)]
    return out


def _fmt_tick(value, decimals):
    """Format an axis tick with thousands separators and fixed decimals."""
    return f"{value:,.{decimals}f}"


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def render_html(header, summaries):
    payload = json.dumps({"header": header, "metrics": summaries})
    return HTML_TEMPLATE.replace("__PAYLOAD__", payload)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Orbit Wars — Replay Eval</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --border:#21262d;
    --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --bar:#3fb950;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1080px; margin:0 auto; padding:32px 24px 64px; }
  header h1 { margin:0 0 4px; font-size:22px; font-weight:600; }
  header .sub { color:var(--muted); font-size:13px; }
  header .meta { margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }
  .chip { background:var(--panel); border:1px solid var(--border); border-radius:6px;
    padding:5px 10px; font-size:12px; color:var(--muted); }
  .chip b { color:var(--text); font-weight:600; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
    padding:18px 22px 14px; margin-top:16px; }
  .chead { display:flex; align-items:flex-start; justify-content:space-between;
    gap:24px; flex-wrap:wrap; }
  .name .label { font-size:15px; font-weight:600; }
  .name .unit { color:var(--muted); font-size:12px; margin-top:2px; }
  .stats { display:flex; gap:22px; }
  .stat .k { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.04em; }
  .stat .v { font-size:15px; font-weight:600; font-variant-numeric:tabular-nums; margin-top:2px;
    text-align:right; }
  .chart { margin-top:14px; }
  .chart svg { display:block; width:100%; height:auto; }
  .empty { color:var(--muted); font-size:13px; font-style:italic; padding:8px 0; }
  .bar { fill:var(--bar); }
  .bar:hover { fill:var(--accent); }
  .axis { stroke:var(--border); stroke-width:1; }
  .grid { stroke:var(--border); stroke-width:1; stroke-dasharray:2 3; opacity:.6; }
  .ylbl { fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
  .xlbl { fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
  .ctitle { fill:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.05em; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title"></h1>
    <div class="sub" id="subtitle"></div>
    <div class="meta" id="meta"></div>
  </header>
  <div id="rows"></div>
</div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const H = DATA.header;
document.getElementById('title').textContent = `Replay Eval — ${H.team}`;
document.getElementById('subtitle').textContent =
  `${H.mode.toUpperCase()} games · ${H.zip}`;
const meta = document.getElementById('meta');
const chips = [
  ['Matching games', H.matching],
  ['Replays scanned', H.scanned],
  ['Skipped', H.skipped],
  ['Player slot', H.pid],
];
for (const [k,v] of chips) {
  const c = document.createElement('div');
  c.className = 'chip';
  c.innerHTML = `${k} <b>${v}</b>`;
  meta.appendChild(c);
}

function fmt(x) {
  if (x === null || x === undefined) return '—';
  if (Number.isInteger(x)) return x.toLocaleString();
  return (Math.abs(x) >= 100 ? x.toFixed(1) : x.toFixed(2));
}

// Approx pixel width of a label in the SVG user-space coordinate system.
function labelWidth(s) { return s.length * 6.6 + 8; }

// A "nice" number (1/2/5 x 10^k) >= x, for round axis steps.
function niceStep(x) {
  if (x <= 0) return 1;
  const exp = Math.floor(Math.log10(x));
  const f = x / Math.pow(10, exp);
  const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nf * Math.pow(10, exp);
}

// Round y-axis ticks 0..>=maxCount, ~5 intervals. Returns {ticks, axisMax}.
function yAxis(maxCount) {
  if (maxCount <= 0) return { ticks: [0, 1], axisMax: 1 };
  const step = niceStep(maxCount / 5);
  const axisMax = Math.ceil(maxCount / step) * step;
  const ticks = [];
  for (let v = 0; v <= axisMax + 1e-6; v += step) ticks.push(Math.round(v));
  return { ticks, axisMax };
}

function histogram(m) {
  if (!m.bins || m.bins.length === 0) return '<div class="empty">no data</div>';
  const n = m.bins.length;
  const maxCount = Math.max(...m.bins) || 1;
  const edgeMode = m.tick_mode === 'edge';

  // SVG user-space canvas; CSS scales it to the card width.
  const W = 1000;
  const mL = 56, mR = 18, mT = 24, mB = 48;
  const plotW = W - mL - mR;
  const plotH = 150;
  const H = mT + plotH + mB;
  const base = mT + plotH;
  const right = W - mR;
  const midX = (mL + right) / 2;
  const slot = plotW / n;
  const barW = Math.min(slot - 6, 46);
  const barOff = (slot - barW) / 2;

  // y gridlines + round count labels.
  const { ticks: yticks, axisMax } = yAxis(maxCount);
  let yaxis = '';
  for (const c of yticks) {
    const y = base - (c / axisMax) * plotH;
    yaxis += `<line class="grid" x1="${mL}" y1="${y.toFixed(1)}" x2="${right}" y2="${y.toFixed(1)}"></line>`;
    yaxis += `<text class="ylbl" x="${mL - 9}" y="${(y + 3.6).toFixed(1)}" text-anchor="end">${c.toLocaleString()}</text>`;
  }

  // Bars.
  let bars = '';
  for (let i = 0; i < n; i++) {
    const h = (m.bins[i] / axisMax) * plotH;
    const x = mL + i * slot + barOff;
    const y = base - h;
    const tip = edgeMode ? `${m.ticks[i]}–${m.ticks[i + 1]}` : m.ticks[i];
    bars += `<rect class="bar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="1.5"><title>${tip}: ${m.bins[i].toLocaleString()}</title></rect>`;
  }

  // x labels: edges sit on bin boundaries, centers sit under each bar.
  // Thin them out if they'd ever collide.
  let xlabels = '';
  const positions = [];
  if (edgeMode) {
    for (let i = 0; i <= n; i++) positions.push({ x: mL + i * slot, t: m.ticks[i], end: i });
  } else {
    for (let i = 0; i < n; i++) positions.push({ x: mL + i * slot + slot / 2, t: m.ticks[i], end: i });
  }
  let need = 0;
  for (const p of positions) need = Math.max(need, labelWidth(p.t));
  const stride = Math.max(1, Math.ceil(need / slot));
  const last = positions.length - 1;
  for (let i = 0; i < positions.length; i++) {
    if (i % stride !== 0 && i !== last) continue;
    const p = positions[i];
    const anchor = (edgeMode && i === 0) ? 'start' : (edgeMode && i === last) ? 'end' : 'middle';
    xlabels += `<text class="xlbl" x="${p.x.toFixed(1)}" y="${base + 17}" text-anchor="${anchor}">${p.t}</text>`;
  }

  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">
    <text class="ctitle" x="${mL - 9}" y="13" text-anchor="end">count</text>
    ${yaxis}
    <line class="axis" x1="${mL}" y1="${base}" x2="${right}" y2="${base}"></line>
    ${bars}
    ${xlabels}
    <text class="ctitle" x="${midX.toFixed(1)}" y="${H - 4}" text-anchor="middle">${m.unit}</text>
  </svg>`;
}

const rows = document.getElementById('rows');
for (const m of DATA.metrics) {
  const card = document.createElement('div');
  card.className = 'card';
  const stats = [
    ['min', fmt(m.min)], ['max', fmt(m.max)], ['mean', fmt(m.mean)],
    ['median', fmt(m.median)], ['variance', fmt(m.variance)], ['n', fmt(m.n)],
  ].map(([k,v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  card.innerHTML = `
    <div class="chead">
      <div class="name"><div class="label">${m.name}</div><div class="unit">${m.unit}</div></div>
      <div class="stats">${stats}</div>
    </div>
    <div class="chart">${histogram(m)}</div>`;
  rows.appendChild(card);
}
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Orbit Wars replay stat analyzer")
    ap.add_argument("mode", choices=["2p", "4p"], help="player count to filter")
    ap.add_argument("team", help="team name to filter by (exact match in TeamNames)")
    ap.add_argument("--zip", help="explicit zip path (default: latest in ladder_replays/)")
    ap.add_argument("--out", help="output HTML path (default: temp file)")
    args = ap.parse_args()

    want_players = 2 if args.mode == "2p" else 4
    zip_path = find_latest_zip(args.zip)
    print(f"Using {zip_path.name}; filtering {args.mode} games with team '{args.team}'...")

    acc = {key: [] for key, _, _ in METRICS}
    scanned = matching = skipped = 0
    pid_seen = set()

    team_bytes = args.team.encode("utf-8")
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        for name in names:
            scanned += 1
            # Cheap prefilter: decompress raw bytes and skip parsing unless the
            # team name appears at all (most replays won't contain it).
            raw = z.read(name)
            if team_bytes not in raw:
                continue
            try:
                replay = json.loads(raw)
                teams = replay["info"]["TeamNames"]
            except Exception:
                skipped += 1
                continue
            if len(teams) != want_players or args.team not in teams:
                continue
            pid = teams.index(args.team)
            try:
                extract_replay(replay, pid, acc)
                matching += 1
                pid_seen.add(pid)
            except Exception as e:
                skipped += 1
                print(f"  warn: skipped {name}: {e}")
            if scanned % 500 == 0:
                print(f"  scanned {scanned}/{len(names)} (matched {matching})...", flush=True)

    if matching == 0:
        print(f"\nNo {args.mode} games found with team '{args.team}' in {zip_path.name}.")
        sys.exit(1)

    summaries = [summarize(key, label, unit, acc[key]) for key, label, unit in METRICS]
    header = {
        "team": args.team,
        "mode": args.mode,
        "zip": zip_path.name,
        "matching": matching,
        "scanned": scanned,
        "skipped": skipped,
        "pid": ", ".join(str(p) for p in sorted(pid_seen)),
    }
    html = render_html(header, summaries)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix="_replay_eval.html", delete=False, encoding="utf-8")
        out_path = Path(fd.name)
        fd.close()
    out_path.write_text(html, encoding="utf-8")

    print(f"\nMatched {matching} games. Report: {out_path}")
    webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
