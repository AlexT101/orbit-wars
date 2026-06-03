"""Replay viewer: pre-compute per-(step, player) top-K predictions from a
trained set-net, then emit a self-contained HTML file (SVG board + JS step
controller). Opens in any browser.

  python3 train/gui.py                 # auto-pick replay, write viewer.html, open it
  python3 train/gui.py --topk 10
  python3 train/gui.py --replay /tmp/orbit_days/<slug>.zip:<id>.json
  python3 train/gui.py --out my.html --no-open
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import webbrowser
import zipfile
from pathlib import Path

import numpy as np
import torch

import build_dataset as bd
from set_net import apply_norm, build_model


# ---------------------------------------------------------------------------
# Replay loading
# ---------------------------------------------------------------------------


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
    else:
        if ":" not in spec:
            raise SystemExit("--replay must be <zip>:<json> or omitted")
        zp_s, name = spec.split(":", 1)
        zp = Path(zp_s)
    with zipfile.ZipFile(zp) as zf:
        with zf.open(name) as f:
            g = json.load(io.BytesIO(f.read()))
    return g, f"{Path(zp).name}::{name}"


# ---------------------------------------------------------------------------
# Pre-compute
# ---------------------------------------------------------------------------


def load_ckpt(path: Path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        ck["arch"], ck["f_planet"], ck["f_global"],
        d_model=ck.get("d_model", 64), n_heads=ck.get("n_heads", 4),
        n_layers=ck.get("n_layers", 2), hidden=ck.get("hidden", 64), dropout=0.0,
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ck["model"] = model
    return ck


def precompute(game: dict, ck: dict, topk: int, log_every: int = 25):
    steps = game.get("steps") or []
    n = len(steps)
    print(f"  parsing {n} step observations ...", flush=True)
    parsed: list[dict | None] = [None] * n
    for t, step in enumerate(steps):
        if step and step[0].get("observation"):
            parsed[t] = bd.parse_state(step[0]["observation"])
    print("  computing features + predictions per step ...", flush=True)

    # Kaggle replay convention: action recorded at step t was decided by the
    # player observing state(t-1); it shows up in state(t) as newly-appeared
    # fleets. So at GUI step t we display:
    #   board state    = state(t)             (the post-action snapshot the user sees)
    #   model prediction = model(state(t-1))  (what was computed at decision time)
    #   actual targets = destinations of fleets newly in state(t) vs state(t-1)
    # This way the marker lands on the step the user sees the launch, and the
    # prediction matches the decision that produced it.
    last_owner: dict[int, int] = {}
    owner_change_turn: dict[int, int] = {}
    out_steps = []
    t0 = time.time()
    for t in range(n):
        state_t = parsed[t]
        if state_t is None:
            out_steps.append(None); continue
        bd.update_owner_history(state_t, last_owner, owner_change_turn, t)

        prev = parsed[t - 1] if t > 0 else None
        per_player = []
        for player in (0, 1):
            # planet ids in state(t) — the board we're rendering. We map predictions
            # (made on state(t-1)) into this id space.
            cur_pids = [p["id"] for p in state_t["planets"]]
            cur_pid_to_idx = {pid: i for i, pid in enumerate(cur_pids)}

            probs_aligned = np.zeros(len(cur_pids), dtype=np.float32) - 1.0  # -1 = no prediction
            if prev is not None:
                try:
                    feats, globals_, pids = bd.extract_per_player(prev, player, owner_change_turn)
                except Exception:
                    feats = None
                if feats is not None:
                    pf = np.zeros((1, bd.N_MAX, ck["f_planet"]), dtype=np.float32)
                    pf[0, :len(feats)] = feats
                    gl = globals_.reshape(1, -1).astype(np.float32)
                    mk = np.zeros((1, bd.N_MAX), dtype=bool); mk[0, :len(feats)] = True
                    pf_n, gl_n = apply_norm(pf, gl, ck["p_mean"], ck["p_std"], ck["g_mean"], ck["g_std"])
                    with torch.no_grad():
                        logits = ck["model"](torch.from_numpy(pf_n),
                                             torch.from_numpy(gl_n),
                                             torch.from_numpy(mk)).numpy()[0, :len(feats)]
                    probs_prev = 1.0 / (1.0 + np.exp(-logits))
                    for j, pid in enumerate(pids):
                        i = cur_pid_to_idx.get(int(pid))
                        if i is not None:
                            probs_aligned[i] = probs_prev[j]
                # actual targets: destinations of fleets newly in state(t)
                labels = bd.labels_for_step(prev, state_t, player, np.array(cur_pids, dtype=np.int32))
            else:
                labels = np.zeros(len(cur_pids), dtype=np.float32)

            # rank by aligned probabilities (slots with -1 sink to the bottom)
            order = np.argsort(-probs_aligned)[:topk]
            order = [i for i in order if probs_aligned[i] >= 0.0]  # drop unpredicted slots
            per_player.append({
                "topk_pids":  [int(cur_pids[i]) for i in order],
                "topk_probs": [round(float(probs_aligned[i]), 4) for i in order],
                "actual_pids": [int(cur_pids[i]) for i, lab in enumerate(labels) if lab > 0.5],
            })

        out_steps.append({
            "planets": [
                {"id": p["id"], "owner": p["owner"], "x": round(p["x"], 3), "y": round(p["y"], 3),
                 "r": round(p["radius"], 2), "ships": p["ships"], "prod": p["prod"],
                 "comet": p["is_comet"]}
                for p in state_t["planets"]
            ],
            "fleets": [
                {"x": round(f["x"], 2), "y": round(f["y"], 2),
                 "owner": f["owner"], "ships": f["ships"]}
                for f in state_t["fleets"]
            ],
            "p0": per_player[0],
            "p1": per_player[1],
        })
        if (t + 1) % log_every == 0:
            print(f"    step {t+1}/{n}  ({time.time() - t0:.1f}s)", flush=True)
    return out_steps


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>target_predictor — {label}</title>
<style>
:root {
  --bg: #0b0d12; --panel: #11151c; --fg: #dbeafe; --muted: #94a3b8;
  --p0: #3a86ff; --p0d: #1d4ed8; --p1: #ff006e; --p1d: #b91c1c;
  --neut: #94a3b8; --sun: #fde047; --comet: #ffffff;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); margin: 0;
        font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 13px; }
.wrap { display: grid; grid-template-columns: 760px 1fr;
         grid-template-rows: 1fr auto; gap: 10px; padding: 10px; height: 100vh; }
#board { background: var(--panel); border-radius: 6px; }
.side { display: grid; grid-template-rows: 1fr 1fr; gap: 10px; min-width: 380px; }
.panel { background: var(--panel); border-radius: 6px; padding: 12px; overflow-y: auto; }
.panel h3 { margin: 0 0 8px 0; font-size: 13px; font-weight: 600; }
.panel.p0 h3 { color: var(--p0); }
.panel.p1 h3 { color: var(--p1); }
.row.actual td { font-weight: 700; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 3px 8px; text-align: left; color: var(--fg); }
th { color: var(--muted); font-weight: 500; border-bottom: 1px solid #1f2937; }
.bar { display: inline-block; height: 8px; background: linear-gradient(90deg, var(--p0), #66b3ff); border-radius: 2px; vertical-align: middle; margin-right: 6px; }
.panel.p1 .bar { background: linear-gradient(90deg, var(--p1), #ff66a3); }
.controls { grid-column: 1 / -1; background: var(--panel); border-radius: 6px;
             padding: 10px 14px; display: flex; align-items: center; gap: 12px; }
.controls button { background: #1f2937; border: 0; color: var(--fg); padding: 6px 10px;
                    border-radius: 4px; cursor: pointer; font-family: inherit; }
.controls button:hover { background: #374151; }
.controls input[type=range] { flex: 1; }
#status { color: var(--muted); font-size: 11px; }
.legend { color: var(--muted); font-size: 11px; }
.legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: middle; margin: 0 3px 1px 8px; }
.summary { color: var(--muted); margin-bottom: 8px; }
.hit { color: #34d399; }
.miss { color: #94a3b8; }
</style></head>
<body>
<div class="wrap">
  <svg id="board" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet"></svg>
  <div class="side">
    <div class="panel p0"><h3 id="p0head"></h3><div id="p0body"></div></div>
    <div class="panel p1"><h3 id="p1head"></h3><div id="p1body"></div></div>
  </div>
  <div class="controls">
    <button id="prev10">«</button>
    <button id="prev">‹</button>
    <button id="play">play</button>
    <button id="next">›</button>
    <button id="next10">»</button>
    <input id="slider" type="range" min="0" max="0" value="0"/>
    <span id="status"></span>
    <span class="legend">
      P0<span class="sw" style="background:var(--p0)"></span>
      P1<span class="sw" style="background:var(--p1)"></span>
      neutral<span class="sw" style="background:var(--neut)"></span>
      comet<span style="color:#fff; margin:0 4px 0 8px;">○</span>
    </span>
  </div>
</div>
<script>
const STEPS = __STEPS__;
const TOPK = __TOPK__;
const LABEL = "__LABEL__";

const PCOL = {0:"#3a86ff", 1:"#ff006e", "-1":"#94a3b8"};
const PRING = {0:"#1d4ed8", 1:"#b91c1c"};
let t = 0; let playing = false; let timer = null;

const svg = document.getElementById("board");
const slider = document.getElementById("slider");
slider.max = STEPS.length - 1;

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

function panelHtml(side, pp, planets) {
  if (!pp) return "<div class=summary>(no state)</div>";
  const actual = new Set(pp.actual_pids);
  const nHit = pp.topk_pids.filter(p => actual.has(p)).length;
  const rows = pp.topk_pids.map((pid, i) => {
    const p = planets.find(q => q.id === pid) || {owner:-1, ships:0, comet:false};
    const ownerS = p.owner === 0 ? "P0" : p.owner === 1 ? "P1" : "neut";
    const isHit = actual.has(pid);
    const prob = pp.topk_probs[i];
    const barW = Math.round(prob * 60);
    const tag = isHit ? '<span class="hit">●</span>' : '';
    const cm = p.comet ? '<span style="color:#fff">○</span>' : '';
    return `<tr class="row${isHit ? ' actual' : ''}"><td>${pid}</td>` +
           `<td><span class="bar" style="width:${barW}px"></span>${prob.toFixed(3)}</td>` +
           `<td>${ownerS}</td><td>${p.ships}</td><td>${tag}${cm}</td></tr>`;
  }).join("");
  const cover = actual.size > 0 ? `${nHit}/${actual.size}` : "—";
  return `<div class="summary">actual targets this turn: ${actual.size} &nbsp; covered in top-${TOPK}: ${cover}</div>` +
         `<table><thead><tr><th>pid</th><th>prob</th><th>owner</th><th>ships</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
}

function render() {
  const s = STEPS[t];
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  if (!s) {
    document.getElementById("status").textContent = `step ${t}/${STEPS.length - 1} (no data)`;
    return;
  }

  // background border
  svg.appendChild(svgEl("rect", {x:0, y:0, width:100, height:100, fill:"#0b0d12"}));

  // sun
  svg.appendChild(svgEl("circle", {cx:50, cy:50, r:10, fill:"#fde047"}));

  // top-K sets per side
  const topk0 = s.p0 ? new Set(s.p0.topk_pids) : new Set();
  const topk1 = s.p1 ? new Set(s.p1.topk_pids) : new Set();
  const act0  = s.p0 ? new Set(s.p0.actual_pids) : new Set();
  const act1  = s.p1 ? new Set(s.p1.actual_pids) : new Set();

  // planets
  for (const p of s.planets) {
    const r = Math.max(1.2, p.r);
    const fill = PCOL[p.owner] || "#94a3b8";
    const stroke = p.comet ? "#ffffff" : "#1f2937";
    svg.appendChild(svgEl("circle", {cx:p.x, cy:p.y, r:r, fill:fill, stroke:stroke,
                                       "stroke-width":(p.comet?0.5:0.25)}));
    if (p.ships > 0) {
      const txt = svgEl("text", {x:p.x, y:p.y + 0.5, "text-anchor":"middle",
                                  "dominant-baseline":"middle",
                                  "font-size":1.6, "font-weight":"bold", fill:"#0b0d12"});
      txt.textContent = p.ships;
      svg.appendChild(txt);
    }
    // rings
    if (topk0.has(p.id)) {
      const rr = r + 0.8;
      svg.appendChild(svgEl("circle", {cx:p.x, cy:p.y, r:rr, fill:"none",
                                         stroke:PRING[0],
                                         "stroke-width":(act0.has(p.id)?0.8:0.3)}));
    }
    if (topk1.has(p.id)) {
      const rr = r + 1.7;
      svg.appendChild(svgEl("circle", {cx:p.x, cy:p.y, r:rr, fill:"none",
                                         stroke:PRING[1],
                                         "stroke-width":(act1.has(p.id)?0.8:0.3)}));
    }
  }

  // fleets as small triangles
  for (const f of s.fleets) {
    const fc = PCOL[f.owner] || "#94a3b8";
    const points = `${f.x},${f.y - 0.7} ${f.x + 0.6},${f.y + 0.5} ${f.x - 0.6},${f.y + 0.5}`;
    svg.appendChild(svgEl("polygon", {points: points, fill: fc, stroke:"white", "stroke-width":0.08}));
  }

  // text panels
  document.getElementById("p0head").textContent = `P0 (blue) — top ${TOPK}`;
  document.getElementById("p1head").textContent = `P1 (red) — top ${TOPK}`;
  document.getElementById("p0body").innerHTML = panelHtml(0, s.p0, s.planets);
  document.getElementById("p1body").innerHTML = panelHtml(1, s.p1, s.planets);
  document.getElementById("status").textContent =
    `step ${t}/${STEPS.length - 1}   ${s.planets.length} planets, ${s.fleets.length} fleets in flight`;
  slider.value = t;
}

function goto(n) {
  if (n < 0) n = 0;
  if (n >= STEPS.length) n = STEPS.length - 1;
  t = n; render();
}
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
slider.oninput = () => goto(parseInt(slider.value));
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") goto(t + (e.shiftKey ? 10 : 1));
  else if (e.key === "ArrowLeft") goto(t - (e.shiftKey ? 10 : 1));
  else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "Home") goto(0);
  else if (e.key === "End") goto(STEPS.length - 1);
});

document.title = `target_predictor — ${LABEL}`;
render();
</script>
</body></html>
"""


def write_html(out_path: Path, steps, label: str, topk: int):
    body = HTML.replace("__STEPS__", json.dumps(steps, separators=(",", ":"))) \
               .replace("__TOPK__", str(topk)) \
               .replace("__LABEL__", label.replace('"', '\\"'))
    out_path.write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=None,
                    help="<zip>:<json>; omit to auto-pick from /tmp/orbit_days")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).resolve().parent / "weights" / "transformer_v2.pt")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "viewer.html")
    ap.add_argument("--no-open", action="store_true",
                    help="don't auto-open in browser")
    args = ap.parse_args()

    print(f"loading checkpoint {args.ckpt}")
    ck = load_ckpt(args.ckpt)
    print(f"loading replay {args.replay or '(auto)'}")
    game, label = load_replay(args.replay)
    print(f"  using: {label}  ({len(game.get('steps', []))} steps)")
    steps = precompute(game, ck, args.topk)
    write_html(args.out, steps, label, args.topk)
    print(f"wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    if not args.no_open:
        print("opening in browser ...")
        webbrowser.open(f"file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
