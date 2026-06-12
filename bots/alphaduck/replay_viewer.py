"""Interactive HTML replay viewer:
  - SVG board with planets, fleets, comets, sun
  - Click a planet to see its pair-model P(launch -> every other planet)
  - Heuristic eval (alphaduck's) at each step
  - Per-side stats: standing ships, in-flight ships, production
  - Step controls (slider, prev/next, play, arrows)

Usage:
  python3 bots/alphaduck/replay_viewer.py            # auto-pick first 2p replay
  python3 bots/alphaduck/replay_viewer.py --replay <zip>:<json>
  python3 bots/alphaduck/replay_viewer.py --out my.html --no-open
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import webbrowser
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "bots" / "mine" / "target_predictor" / "train"))
sys.path.insert(0, str(ROOT / "bots" / "alphaduck" / "train"))
sys.path.insert(0, str(HERE))

import build_dataset as bd
from set_net import apply_norm
from pair_net import PlanetTransformerPair


def pick_default_replay() -> tuple[Path, str]:
    root = Path("/tmp/orbit_days")
    for zp in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(zp) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith(".json"):
                    continue
                with zf.open(name) as f:
                    g = json.load(io.BytesIO(f.read()))
                if len(g.get("rewards", [])) == 2 and len(g.get("steps", [])) > 30:
                    return zp, name
    raise SystemExit("no 2p replays found under /tmp/orbit_days")


def load_replay(spec: str | None) -> tuple[dict, str]:
    if spec is None:
        zp, name = pick_default_replay()
        with zipfile.ZipFile(zp) as zf:
            with zf.open(name) as f:
                g = json.load(io.BytesIO(f.read()))
        return g, f"{Path(zp).name}::{name}"
    if ":" in spec:
        zp_s, name = spec.split(":", 1)
        zp = Path(zp_s)
        with zipfile.ZipFile(zp) as zf:
            with zf.open(name) as f:
                g = json.load(io.BytesIO(f.read()))
        return g, f"{Path(zp).name}::{name}"
    # Direct .json file path
    p = Path(spec)
    g = json.loads(p.read_text())
    return g, p.name


def load_pair_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PlanetTransformerPair(
        ck["f_planet"], ck["f_global"],
        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
        n_layers=ck.get("n_layers", 2), ff=ck.get("ff", 128), dropout=0.0,
    ).eval()
    m.load_state_dict(ck["state_dict"])
    return ck, m


def per_side_aggregates(state, player):
    standing = sum(p["ships"] for p in state["planets"] if p["owner"] == player)
    in_flight = sum(f["ships"] for f in state["fleets"] if f["owner"] == player)
    prod = sum(p["prod"] for p in state["planets"] if p["owner"] == player)
    n_planets = sum(1 for p in state["planets"] if p["owner"] == player)
    return dict(standing=int(standing), in_flight=int(in_flight),
                prod=int(prod), n_planets=int(n_planets))


def heuristic_eval(state, player):
    """Same heuristic alphaduck uses for leaf evaluation."""
    my_p = my_pr = my_s = 0; en_p = en_pr = en_s = 0
    for p in state["planets"]:
        if p["owner"] == player:
            my_p += 1; my_pr += p["prod"]; my_s += p["ships"]
        elif p["owner"] >= 0:
            en_p += 1; en_pr += p["prod"]; en_s += p["ships"]
    return (my_p - en_p) * 5.0 + (my_pr - en_pr) * 8.0 + (my_s - en_s) * 0.05


def precompute(game, ck, model):
    steps = game.get("steps") or []
    parsed: list[dict | None] = [None] * len(steps)
    for t, step in enumerate(steps):
        if step and step[0].get("observation"):
            parsed[t] = bd.parse_state(step[0]["observation"])

    # Persistent owner history (player↔player only) matches build_dataset behavior.
    last_owner: dict[int, int] = {}
    owner_change_turn: dict[int, int] = {}

    out_steps = []
    t0 = time.time()
    for t, state in enumerate(parsed):
        if state is None:
            out_steps.append(None); continue
        bd.update_owner_history(state, last_owner, owner_change_turn, state["step"])

        per_player = {}
        for player in (0, 1):
            try:
                feats, globals_, pids = bd.extract_per_player(state, player, owner_change_turn)
            except Exception:
                per_player[player] = None; continue
            n = feats.shape[0]
            pf = np.zeros((1, bd.N_MAX, ck["f_planet"]), dtype=np.float32); pf[0, :n] = feats
            gl = globals_.reshape(1, -1).astype(np.float32)
            mk = np.zeros((1, bd.N_MAX), dtype=bool); mk[0, :n] = True
            # raw inputs for the pair-feature head
            raw_xy = np.zeros((1, bd.N_MAX, 7, 2), dtype=np.float32)
            raw_ships = np.zeros((1, bd.N_MAX), dtype=np.float32)
            raw_prod = np.zeros((1, bd.N_MAX), dtype=np.float32)
            for i, p in enumerate(state["planets"]):
                for j, h in enumerate((0, 1, 2, 5, 10, 20, 30)):
                    pos = bd.planet_pos_at(state, p, h)
                    raw_xy[0, i, j] = pos if pos is not None else (p["x"], p["y"])
                raw_ships[0, i] = p["ships"]
                raw_prod[0, i] = p["prod"]
            pf_n, gl_n = apply_norm(pf, gl, ck["p_mean"], ck["p_std"], ck["g_mean"], ck["g_std"])
            with torch.no_grad():
                pair_logits, value, noop_logits = model(
                    torch.from_numpy(pf_n),
                    torch.from_numpy(gl_n),
                    torch.from_numpy(mk),
                    raw_xy=torch.from_numpy(raw_xy),
                    raw_ships=torch.from_numpy(raw_ships),
                    raw_prod=torch.from_numpy(raw_prod),
                    return_value=True, return_noop=True,
                )
            pair_logits = pair_logits.numpy()[0]
            value = float(value.numpy()[0])
            noop_logits = noop_logits.numpy()[0]
            # If model was trained with policy CE (v10+), the row of
            # softmax([noop_logit_i, pair_logits[i, :]]) is the actual prior.
            # For v9 and earlier we show the calibrated sigmoid marginals
            # (those don't sum to 1 — that was the bug).
            pair_logits_n = pair_logits[:n, :n].copy()
            np.fill_diagonal(pair_logits_n, -1e9)
            has_policy = ck.get("policy_loss_weight", 0.0) > 0
            is_conditional = bool(ck.get("policy_conditional", False))
            if has_policy and is_conditional:
                # Conditional: P(target|launch) = softmax(pair_logits); P(noop) from sigmoid head.
                # Joint: pair_probs[i, j] = (1 - noop[i]) * cond[i, j].  Matches main.py exactly.
                flat = pair_logits_n - pair_logits_n.max(axis=1, keepdims=True)
                ex = np.exp(flat)
                cond_pair = ex / ex.sum(axis=1, keepdims=True)
                noop_probs = 1.0 / (1.0 + np.exp(-noop_logits[:n]))
                probs = (1.0 - noop_probs)[:, None] * cond_pair
            elif has_policy:
                full = np.concatenate([noop_logits[:n, None], pair_logits_n], axis=1)
                full = full - full.max(axis=1, keepdims=True)
                ex = np.exp(full)
                policy = ex / ex.sum(axis=1, keepdims=True)
                probs = policy[:, 1:]
                noop_probs = policy[:, 0]
            else:
                probs = 1.0 / (1.0 + np.exp(-pair_logits[:n, :n]))
                np.fill_diagonal(probs, 0)
                noop_probs = 1.0 / (1.0 + np.exp(-noop_logits[:n]))
            agg = per_side_aggregates(state, player)
            per_player[player] = {
                "pids": [int(p) for p in pids],
                # Round to 4 places to keep the JSON compact
                "probs": [[round(float(probs[i, j]), 4) for j in range(n)] for i in range(n)],
                "noop": [round(float(noop_probs[i]), 4) for i in range(n)],
                "value": round(value, 3),
                "agg": agg,
                "eval_for_me": round(heuristic_eval(state, player), 2),
            }

        # actual launches at this step (newly-appeared fleets vs prev step)
        actual = {0: {}, 1: {}}
        prev = parsed[t - 1] if t > 0 else None
        if prev is not None:
            old_ids = {f["id"] for f in prev["fleets"]}
            for f in state["fleets"]:
                if f["id"] in old_ids:
                    continue
                # find the source planet whose action created it (best-effort:
                # the action recorded at this step has source planet pid)
                pl_acts = (steps[t][f["owner"]].get("action") or [])
                src_pid = None
                for act in pl_acts:
                    if abs(float(act[1]) - f["angle"]) < 1e-6 and int(act[2]) == f["ships"]:
                        src_pid = int(act[0]); break
                # predict its destination
                pred = bd.predict_fleet_collision(state, f)
                if pred is None or src_pid is None:
                    continue
                dst_pid, _eta = pred
                actual[f["owner"]].setdefault(src_pid, []).append(int(dst_pid))

        out_steps.append({
            "planets": [
                {"id": int(p["id"]), "owner": int(p["owner"]),
                 "x": round(p["x"], 3), "y": round(p["y"], 3),
                 "r": round(p["radius"], 2),
                 "ships": int(p["ships"]), "prod": int(p["prod"]),
                 "comet": bool(p["is_comet"])}
                for p in state["planets"]
            ],
            "fleets": [
                {"x": round(f["x"], 2), "y": round(f["y"], 2),
                 "angle": round(float(f["angle"]), 4),
                 "owner": int(f["owner"]), "ships": int(f["ships"])}
                for f in state["fleets"]
            ],
            "p0": per_player[0],
            "p1": per_player[1],
            "actual0": actual[0],
            "actual1": actual[1],
        })
        if (t + 1) % 25 == 0:
            print(f"  step {t+1}/{len(steps)}  ({time.time() - t0:.1f}s)", flush=True)
    return out_steps


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>alphaduck replay viewer</title>
<style>
:root {
  --bg: #05070b; --panel: #0e1218; --fg: #e2e8f0; --muted: #94a3b8;
  --p0: #38bdf8; --p0d: #0c4a6e; --p1: #f43f5e; --p1d: #881337;
  --neut: #94a3b8; --sun: #fde047; --comet: #ffffff;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); margin: 0;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }
.wrap { display: grid; grid-template-columns: 760px 1fr;
  grid-template-rows: 1fr auto; gap: 8px; padding: 8px; height: 100vh; }
#board { background: radial-gradient(ellipse at center, #0a1424 0%, #03070d 70%);
  border-radius: 8px; cursor: pointer; box-shadow: 0 0 30px rgba(56, 189, 248, 0.06) inset; }
.side { display: grid; grid-template-rows: auto 1fr; gap: 8px; min-width: 380px; }
.panel { background: var(--panel); border-radius: 6px; padding: 10px; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stats .col { padding: 6px; }
.stats .col.p0 { border-left: 3px solid var(--p0); }
.stats .col.p1 { border-left: 3px solid var(--p1); }
.stats h4 { margin: 0 0 4px 0; font-size: 11px; color: var(--muted); }
.stats .v { font-size: 18px; }
#detail { overflow-y: auto; max-height: calc(100vh - 280px); }
#detail h3 { margin: 0 0 6px 0; font-size: 13px; }
#detail.p0 h3 { color: var(--p0); }
#detail.p1 h3 { color: var(--p1); }
#detail .hint { color: var(--muted); margin-bottom: 8px; }
#detail table { width: 100%; border-collapse: collapse; }
#detail th, #detail td { padding: 2px 6px; text-align: left; }
#detail th { color: var(--muted); border-bottom: 1px solid #1f2937; }
.bar { display: inline-block; height: 6px; background: linear-gradient(90deg, var(--p0), #80b3ff); vertical-align: middle; margin-right: 4px; border-radius: 2px; }
#detail.p1 .bar { background: linear-gradient(90deg, var(--p1), #ff80b3); }
.controls { grid-column: 1 / -1; background: var(--panel); border-radius: 6px;
  padding: 8px 12px; display: flex; align-items: center; gap: 10px; }
.controls button { background: #1f2937; border: 0; color: var(--fg); padding: 5px 9px;
  border-radius: 4px; cursor: pointer; font-family: inherit; }
.controls button:hover { background: #374151; }
.controls input[type=range] { flex: 1; }
.controls select { background: #1f2937; color: var(--fg); border: 0; padding: 4px 6px; border-radius: 4px; }
#status { color: var(--muted); font-size: 11px; }
.row.actual { color: #34d399; }
.row.actual td:first-child::after { content: " ●"; }
</style></head>
<body>
<div class="wrap">
  <svg id="board" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet"></svg>
  <div class="side">
    <div class="panel">
      <div class="stats">
        <div class="col p0">
          <h4>P0 (blue)</h4>
          <div class="v">ships: <span id="p0-ships">-</span> (<span id="p0-flight">0</span> flight)</div>
          <div class="v">prod: <span id="p0-prod">-</span>  planets: <span id="p0-planets">-</span></div>
          <div class="v" style="color:#fde047">eval: <span id="p0-eval">-</span>  V<sub>net</sub>: <span id="p0-value">-</span></div>
        </div>
        <div class="col p1">
          <h4>P1 (red)</h4>
          <div class="v">ships: <span id="p1-ships">-</span> (<span id="p1-flight">0</span> flight)</div>
          <div class="v">prod: <span id="p1-prod">-</span>  planets: <span id="p1-planets">-</span></div>
          <div class="v" style="color:#fde047">eval: <span id="p1-eval">-</span>  V<sub>net</sub>: <span id="p1-value">-</span></div>
        </div>
      </div>
    </div>
    <div class="panel">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
        <h3 style="margin:0;" id="detail-title">click a planet</h3>
        <span>POV: <select id="pov"><option value="0">P0 (blue)</option><option value="1">P1 (red)</option></select></span>
      </div>
      <div id="detail" class="p0">
        <div class="hint">Click any planet on the board to see how likely the model thinks that planet's owner is to launch a fleet from it to each other planet. ● marks the actual destination(s) launched this turn.</div>
      </div>
    </div>
  </div>
  <div class="controls">
    <button id="prev10">«</button>
    <button id="prev">‹</button>
    <button id="play">play</button>
    <button id="next">›</button>
    <button id="next10">»</button>
    <input id="slider" type="range" min="0" max="0" value="0"/>
    <span id="status"></span>
  </div>
</div>
<script>
const STEPS = __STEPS__;
const LABEL = "__LABEL__";
const PCOL = {0:"#3a86ff", 1:"#ff006e", "-1":"#94a3b8"};
const PRING = {0:"#1d4ed8", 1:"#b91c1c"};

let t = 0; let playing = false; let timer = null;
let selectedPid = null;
let pov = 0;
const svg = document.getElementById("board");
const slider = document.getElementById("slider");
slider.max = STEPS.length - 1;

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

function setStats(s) {
  for (const pp of [["p0", s.p0], ["p1", s.p1]]) {
    const k = pp[0], v = pp[1];
    if (!v) {
      document.getElementById(k + "-ships").textContent = "-";
      document.getElementById(k + "-flight").textContent = "0";
      document.getElementById(k + "-prod").textContent = "-";
      document.getElementById(k + "-planets").textContent = "-";
      document.getElementById(k + "-eval").textContent = "-";
      document.getElementById(k + "-value").textContent = "-";
      continue;
    }
    document.getElementById(k + "-ships").textContent = v.agg.standing;
    document.getElementById(k + "-flight").textContent = v.agg.in_flight;
    document.getElementById(k + "-prod").textContent = v.agg.prod;
    document.getElementById(k + "-planets").textContent = v.agg.n_planets;
    document.getElementById(k + "-eval").textContent = v.eval_for_me.toFixed(2);
    document.getElementById(k + "-value").textContent = (v.value != null ? v.value.toFixed(3) : "-");
  }
}

function renderDetail(s) {
  const detail = document.getElementById("detail");
  const title = document.getElementById("detail-title");
  detail.className = pov === 0 ? "p0" : "p1";
  const pdata = pov === 0 ? s.p0 : s.p1;
  if (selectedPid === null || !pdata) {
    title.textContent = "click a planet";
    detail.innerHTML = '<div class="hint">Click any planet on the board to see how likely the model thinks that planet\\'s owner is to launch a fleet from it to each other planet. ● marks the actual destination(s) launched this turn.</div>';
    return;
  }
  const idx = pdata.pids.indexOf(selectedPid);
  if (idx < 0) {
    title.textContent = "planet " + selectedPid + " (not in this POV)";
    detail.innerHTML = '<div class="hint">This planet wasn\\'t in the POV\\'s feature view (rare; only happens if N>50).</div>';
    return;
  }
  const row = pdata.probs[idx];
  const noopP = (pdata.noop && pdata.noop[idx] != null) ? pdata.noop[idx] : null;
  // sort targets by probability
  const order = row.map((p, j) => [p, j]).sort((a, b) => b[0] - a[0]);
  // actual destinations launched from this source this turn
  const actualKey = pov === 0 ? "actual0" : "actual1";
  const actualDests = (s[actualKey] && s[actualKey][selectedPid]) || [];
  const planet = s.planets.find(p => p.id === selectedPid);
  const noopStr = noopP != null ? `  P(noop)=${noopP.toFixed(3)}` : "";
  title.textContent = `planet ${selectedPid} (${planet.owner === 0 ? "P0" : planet.owner === 1 ? "P1" : "neut"}, ships=${planet.ships}, prod=${planet.prod})${noopStr}`;
  let html = '<table><thead><tr><th>tgt</th><th>prob</th><th>owner</th><th>ships</th></tr></thead><tbody>';
  for (const [p, j] of order) {
    if (p < 1e-4) continue;
    const tgtPid = pdata.pids[j];
    const tgt = s.planets.find(pl => pl.id === tgtPid);
    if (!tgt) continue;
    const isActual = actualDests.includes(tgtPid);
    const ownerStr = tgt.owner === 0 ? "P0" : tgt.owner === 1 ? "P1" : "neut";
    const barW = Math.min(60, Math.round(p * 60 / 0.5));
    html += `<tr class="row${isActual ? ' actual' : ''}"><td>${tgtPid}</td>` +
            `<td><span class="bar" style="width:${barW}px"></span>${p.toFixed(4)}</td>` +
            `<td>${ownerStr}</td><td>${tgt.ships}</td></tr>`;
  }
  html += '</tbody></table>';
  if (order.length === 0 || order[0][0] < 1e-4) {
    html += '<div class="hint">All target probabilities are below 0.0001 (no clear launch predicted from this planet).</div>';
  }
  detail.innerHTML = html;
}

function render() {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const s = STEPS[t];
  if (!s) {
    document.getElementById("status").textContent = `step ${t}/${STEPS.length - 1} (no data)`;
    return;
  }

  // <defs> for gradients + filters
  const defs = svgEl("defs", {});
  defs.innerHTML = `
    <radialGradient id="sun-grad">
      <stop offset="0%" stop-color="#fff5a0"/>
      <stop offset="40%" stop-color="#fde047"/>
      <stop offset="80%" stop-color="#facc15" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#facc15" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="p0-grad"><stop offset="0%" stop-color="#bae6fd"/><stop offset="60%" stop-color="#38bdf8"/><stop offset="100%" stop-color="#0369a1"/></radialGradient>
    <radialGradient id="p1-grad"><stop offset="0%" stop-color="#fecdd3"/><stop offset="60%" stop-color="#f43f5e"/><stop offset="100%" stop-color="#9f1239"/></radialGradient>
    <radialGradient id="neut-grad"><stop offset="0%" stop-color="#e2e8f0"/><stop offset="60%" stop-color="#94a3b8"/><stop offset="100%" stop-color="#475569"/></radialGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="0.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  `;
  svg.appendChild(defs);

  // Sun
  svg.appendChild(svgEl("circle", {cx:50, cy:50, r:14, fill:"url(#sun-grad)", opacity:"0.6"}));
  svg.appendChild(svgEl("circle", {cx:50, cy:50, r:10, fill:"#fde047", filter:"url(#glow)"}));

  // Orbit rings for planets (faint dashed circles at planet's center distance)
  const ringSet = new Set();
  for (const p of s.planets) {
    if (p.comet) continue;
    const dx = p.x - 50, dy = p.y - 50;
    const dist = Math.hypot(dx, dy);
    const key = dist.toFixed(1);
    if (!ringSet.has(key) && dist > 11) {
      ringSet.add(key);
      svg.appendChild(svgEl("circle", {cx:50, cy:50, r:dist, fill:"none",
        stroke:"#1e293b", "stroke-width":0.12, "stroke-dasharray":"0.5,0.5"}));
    }
  }

  for (const p of s.planets) {
    const r = Math.max(1.5, p.r * 1.1);
    const grad = p.owner === 0 ? "url(#p0-grad)" : p.owner === 1 ? "url(#p1-grad)" : "url(#neut-grad)";
    const stroke = p.comet ? "#ffffff" : (p.owner === 0 ? "#0369a1" : p.owner === 1 ? "#9f1239" : "#475569");
    const sel = p.id === selectedPid;
    const sw = sel ? 0.6 : (p.comet ? 0.4 : 0.2);
    // Outer halo for owned planets
    if (p.owner >= 0 && !p.comet) {
      svg.appendChild(svgEl("circle", {cx:p.x, cy:p.y, r:r + 0.8, fill:grad, opacity:0.2}));
    }
    const c = svgEl("circle", {cx:p.x, cy:p.y, r:r, fill:grad,
      stroke: sel ? "#fde047" : stroke, "stroke-width": sw, filter: sel ? "url(#glow)" : ""});
    c.style.cursor = "pointer";
    c.addEventListener("click", (ev) => { ev.stopPropagation(); selectedPid = p.id; render(); });
    svg.appendChild(c);
    if (p.ships > 0) {
      const txt = svgEl("text", {x:p.x, y:p.y + 0.5, "text-anchor":"middle", "dominant-baseline":"middle",
                                  "font-size":1.7, "font-weight":"bold",
                                  fill: p.owner >= 0 ? "#0a0f1a" : "#0a0f1a"});
      txt.textContent = p.ships;
      txt.style.pointerEvents = "none";
      svg.appendChild(txt);
    }
  }
  // Fleets: rotate triangle to face heading
  for (const f of s.fleets) {
    const fc = PCOL[f.owner] || "#94a3b8";
    const sz = 0.9 + 0.05 * Math.min(20, Math.sqrt(Math.max(1, f.ships || 1)));
    // Triangle pointing up; we'll rotate via SVG transform
    const angleDeg = (f.angle || 0) * 180 / Math.PI + 90; // +90 because triangle points up by default
    const points = `0,${-sz} ${sz * 0.7},${sz * 0.6} ${-sz * 0.7},${sz * 0.6}`;
    const tri = svgEl("polygon", {points, fill: fc, stroke:"#0a0f1a", "stroke-width":0.1,
      transform: `translate(${f.x} ${f.y}) rotate(${angleDeg})`, filter:"url(#glow)"});
    svg.appendChild(tri);
    // Small ship count label next to fleet
    if (f.ships >= 5) {
      const lbl = svgEl("text", {x:f.x, y:f.y - sz - 0.3, "text-anchor":"middle",
                                  "font-size":1.0, fill:"#cbd5e1", opacity:0.7});
      lbl.textContent = f.ships;
      lbl.style.pointerEvents = "none";
      svg.appendChild(lbl);
    }
  }
  setStats(s);
  renderDetail(s);
  slider.value = t;
  document.getElementById("status").textContent =
    `step ${t}/${STEPS.length - 1}   ${s.planets.length} planets, ${s.fleets.length} fleets in flight`;
}
function goto(n) { t = Math.max(0, Math.min(STEPS.length - 1, n)); render(); }
function togglePlay() {
  playing = !playing;
  document.getElementById("play").textContent = playing ? "pause" : "play";
  if (playing) {
    timer = setInterval(() => {
      if (t + 1 >= STEPS.length) { playing = false; clearInterval(timer);
        document.getElementById("play").textContent = "play"; return; }
      goto(t + 1);
    }, 200);
  } else if (timer) { clearInterval(timer); timer = null; }
}

document.getElementById("prev").onclick = () => goto(t - 1);
document.getElementById("next").onclick = () => goto(t + 1);
document.getElementById("prev10").onclick = () => goto(t - 10);
document.getElementById("next10").onclick = () => goto(t + 10);
document.getElementById("play").onclick = togglePlay;
document.getElementById("pov").onchange = (e) => { pov = parseInt(e.target.value); render(); };
slider.oninput = () => goto(parseInt(slider.value));
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") goto(t + (e.shiftKey ? 10 : 1));
  else if (e.key === "ArrowLeft") goto(t - (e.shiftKey ? 10 : 1));
  else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "Home") goto(0);
  else if (e.key === "End") goto(STEPS.length - 1);
});
svg.addEventListener("click", () => { selectedPid = null; render(); });
document.title = `alphaduck viewer — ${LABEL}`;
render();
</script>
</body></html>
"""


def write_html(out_path: Path, steps, label: str):
    body = HTML.replace("__STEPS__", json.dumps(steps, separators=(",", ":"))) \
               .replace("__LABEL__", label.replace('"', '\\"'))
    out_path.write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=None)
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "bots" / "mine" / "target_predictor" / "train" / "weights" / "transformer_pair_v15_cond.pt")
    ap.add_argument("--out", type=Path, default=HERE / "replay_viewer.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    print(f"loading checkpoint {args.ckpt}")
    ck, model = load_pair_model(args.ckpt)
    print(f"loading replay {args.replay or '(auto)'}")
    game, label = load_replay(args.replay)
    print(f"  using: {label}  ({len(game.get('steps', []))} steps)")
    steps = precompute(game, ck, model)
    write_html(args.out, steps, label)
    print(f"wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    if not args.no_open:
        print("opening in browser ...")
        webbrowser.open(f"file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
