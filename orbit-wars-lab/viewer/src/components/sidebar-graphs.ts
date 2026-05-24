/**
 * Graphs accordion for the Quick Match sidebar.
 *
 * Reads the per-step series the iframe renderer publishes into
 * localStorage["ow-series"], the live cursor from "ow-live-match", and
 * draws four small panels (Ships / Production / Planets / Ship gain Δ)
 * styled to match the rest of the sidebar.
 */

const PLAYER_COLORS = ["#5EA5FF", "#FF8A4C", "#5EED9F", "#C084FC"] as const;

interface Series {
  totalSteps: number;
  numAgents: number;
  ships: number[][];
  production: number[][];
  planets: number[][];
  shipDelta: number[];
}

interface Live {
  step?: number;
  playerNames?: string[];
}

interface ThemeColors {
  bg: string;
  text: string;
  muted: string;
  grid: string;
  gridStrong: string;
  cursor: string;
}

function readVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function themeColors(): ThemeColors {
  return {
    bg: readVar("--surface-2", "#14141a"),
    text: readVar("--text-primary", "#e4e4e7"),
    muted: readVar("--text-muted", "#71717a"),
    grid: readVar("--border", "#1a1a22"),
    gridStrong: readVar("--border-strong", "#27272a"),
    cursor: readVar("--accent", "#8ac4ff"),
  };
}

function drawSeriesPanel(
  canvas: HTMLCanvasElement,
  arrays: number[][],
  step: number,
  _totalSteps: number,
  numAgents: number,
  theme: ThemeColors,
) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(1, Math.floor(rect.width));
  const cssH = Math.max(1, Math.floor(rect.height));
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, W, H);

  const padL = 28, padR = 6, padT = 6, padB = 16;
  const left = padL, right = W - padR, top = padT, bot = H - padB;
  const innerW = right - left, innerH = bot - top;
  // X-axis spans 0..current step, the same anti-spoiler view used by the
  // standalone visualize.py sidebar.
  const viewedT = Math.max(1, Math.min(step + 1, arrays[0]?.length ?? 1));

  let maxV = 1;
  for (const arr of arrays) {
    for (let i = 0; i < viewedT; i++) if (arr[i] > maxV) maxV = arr[i];
  }

  ctx.font = "9px " + readVar("--font-mono", "ui-monospace, monospace");
  ctx.fillStyle = theme.muted;
  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let g = 0; g <= 3; g++) {
    const y = bot - (g / 3) * innerH;
    ctx.beginPath();
    ctx.moveTo(left, y + 0.5);
    ctx.lineTo(right, y + 0.5);
    ctx.stroke();
    ctx.fillText(String(Math.round((g / 3) * maxV)), left - 4, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let g = 0; g <= 4; g++) {
    const x = left + (g / 4) * innerW;
    const s = Math.round((g / 4) * Math.max(0, viewedT - 1));
    ctx.fillText(String(s), x, bot + 3);
  }

  const xOf = (i: number) => left + (i / Math.max(1, viewedT - 1)) * innerW;
  for (let p = 0; p < numAgents; p++) {
    const arr = arrays[p];
    if (!arr) continue;
    ctx.strokeStyle = PLAYER_COLORS[p] ?? theme.text;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < viewedT; i++) {
      const x = xOf(i);
      const y = bot - (arr[i] / maxV) * innerH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const px = xOf(step);
  ctx.strokeStyle = theme.cursor;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px + 0.5, top);
  ctx.lineTo(px + 0.5, bot);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawDeltaPanel(
  canvas: HTMLCanvasElement,
  delta: number[],
  step: number,
  _totalSteps: number,
  theme: ThemeColors,
) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(1, Math.floor(rect.width));
  const cssH = Math.max(1, Math.floor(rect.height));
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, W, H);

  const padL = 32, padR = 6, padT = 6, padB = 16;
  const left = padL, right = W - padR, top = padT, bot = H - padB;
  const innerW = right - left, innerH = bot - top;
  const viewedT = Math.max(1, Math.min(step + 1, delta.length));

  let maxAbs = 1;
  for (let i = 0; i < viewedT; i++) {
    const v = Math.abs(delta[i] ?? 0);
    if (v > maxAbs) maxAbs = v;
  }
  const yOf = (v: number) => top + innerH / 2 - (v / maxAbs) * (innerH / 2);
  const zeroY = yOf(0);
  const xOf = (i: number) => left + (i / Math.max(1, viewedT - 1)) * innerW;

  ctx.font = "9px " + readVar("--font-mono", "ui-monospace, monospace");
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of [maxAbs, maxAbs / 2, 0, -maxAbs / 2, -maxAbs]) {
    const y = yOf(v);
    ctx.strokeStyle = v === 0 ? theme.gridStrong : theme.grid;
    ctx.lineWidth = v === 0 ? 1.2 : 1;
    ctx.beginPath();
    ctx.moveTo(left, y + 0.5); ctx.lineTo(right, y + 0.5);
    ctx.stroke();
    ctx.fillStyle = theme.muted;
    ctx.fillText((v > 0 ? "+" : "") + Math.round(v), left - 4, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillStyle = theme.muted;
  for (let g = 0; g <= 4; g++) {
    const x = left + (g / 4) * innerW;
    const s = Math.round((g / 4) * Math.max(0, viewedT - 1));
    ctx.fillText(String(s), x, bot + 3);
  }

  // Filled area above zero (P0 color), below zero (P1 color).
  const drawFilled = (sign: 1 | -1) => {
    const color = sign > 0 ? PLAYER_COLORS[0] : PLAYER_COLORS[1];
    ctx.fillStyle = color + "44";
    ctx.beginPath();
    ctx.moveTo(xOf(0), zeroY);
    for (let i = 0; i < viewedT; i++) {
      const v = delta[i] ?? 0;
      const x = xOf(i);
      const y = sign > 0 ? yOf(Math.max(0, v)) : yOf(Math.min(0, v));
      ctx.lineTo(x, y);
    }
    ctx.lineTo(xOf(viewedT - 1), zeroY);
    ctx.closePath();
    ctx.fill();
  };
  drawFilled(1);
  drawFilled(-1);

  ctx.strokeStyle = theme.text;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < viewedT; i++) {
    const x = xOf(i);
    const y = yOf(delta[i] ?? 0);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const px = xOf(step);
  ctx.strokeStyle = theme.cursor;
  ctx.globalAlpha = 0.55;
  ctx.beginPath();
  ctx.moveTo(px + 0.5, top); ctx.lineTo(px + 0.5, bot);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Current-value label, colored toward the leading side.
  const curIdx = Math.min(step, delta.length - 1);
  const curVal = delta[curIdx] ?? 0;
  ctx.fillStyle = curVal >= 0 ? PLAYER_COLORS[0] : PLAYER_COLORS[1];
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.font = "bold 10px " + readVar("--font-mono", "ui-monospace, monospace");
  ctx.fillText((curVal >= 0 ? "+" : "") + curVal, left + 4, top);
}

export interface GraphsHandle {
  render: () => void;
}

export function mountGraphs(host: HTMLElement): GraphsHandle {
  host.innerHTML = `
    <div class="qm-graphs">
      <div class="qm-graph">
        <div class="qm-graph-title">Ships</div>
        <canvas data-graph="ships"></canvas>
      </div>
      <div class="qm-graph">
        <div class="qm-graph-title">Production</div>
        <canvas data-graph="production"></canvas>
      </div>
      <div class="qm-graph">
        <div class="qm-graph-title">Planets</div>
        <canvas data-graph="planets"></canvas>
      </div>
      <div class="qm-graph" data-delta-panel>
        <div class="qm-graph-title" data-delta-title>Ship gain Δ</div>
        <canvas data-graph="shipDelta"></canvas>
      </div>
    </div>
    <div class="qm-graphs-empty" data-empty hidden>No replay loaded yet.</div>
  `;

  const canvases = {
    ships: host.querySelector<HTMLCanvasElement>('canvas[data-graph="ships"]')!,
    production: host.querySelector<HTMLCanvasElement>('canvas[data-graph="production"]')!,
    planets: host.querySelector<HTMLCanvasElement>('canvas[data-graph="planets"]')!,
    shipDelta: host.querySelector<HTMLCanvasElement>('canvas[data-graph="shipDelta"]')!,
  };
  const deltaPanel = host.querySelector<HTMLElement>("[data-delta-panel]")!;
  const deltaTitle = host.querySelector<HTMLElement>("[data-delta-title]")!;
  const graphsRoot = host.querySelector<HTMLElement>(".qm-graphs")!;
  const emptyEl = host.querySelector<HTMLElement>("[data-empty]")!;

  function render() {
    const sRaw = localStorage.getItem("ow-series");
    if (!sRaw) {
      graphsRoot.hidden = true;
      emptyEl.hidden = false;
      return;
    }
    let series: Series;
    let live: Live = {};
    try {
      series = JSON.parse(sRaw);
      const lRaw = localStorage.getItem("ow-live-match");
      if (lRaw) live = JSON.parse(lRaw);
    } catch {
      return;
    }
    graphsRoot.hidden = false;
    emptyEl.hidden = true;
    const step = Math.max(0, Math.min(series.totalSteps - 1, live.step ?? 0));
    const names = live.playerNames || [];

    if (series.numAgents >= 2) {
      deltaPanel.style.display = "";
      // Agent IDs are bucket-prefixed paths like "baselines/random" — strip
      // to the leaf so we don't end up with "baselines… − baselines…".
      const short = (n: string) => {
        const leaf = n.includes("/") ? n.split("/").pop()! : n;
        return leaf.length > 12 ? leaf.slice(0, 11) + "…" : leaf;
      };
      const p0 = short(names[0] || "P1");
      const p1 = short(names[1] || "P2");
      deltaTitle.textContent = `Ship gain Δ (${p0} − ${p1})`;
    } else {
      deltaPanel.style.display = "none";
    }

    const theme = themeColors();
    const T = series.totalSteps;
    drawSeriesPanel(canvases.ships, series.ships, step, T, series.numAgents, theme);
    drawSeriesPanel(canvases.production, series.production, step, T, series.numAgents, theme);
    drawSeriesPanel(canvases.planets, series.planets, step, T, series.numAgents, theme);
    if (series.numAgents >= 2) {
      drawDeltaPanel(canvases.shipDelta, series.shipDelta, step, T, theme);
    }
  }

  return { render };
}
