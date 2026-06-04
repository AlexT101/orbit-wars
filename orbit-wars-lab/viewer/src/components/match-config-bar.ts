/**
 * Match config bar — inline pill toggles for games / mode / seed / format.
 * Emits onChange(config) on each change.
 */

import {
  replayMapFromFile,
  replayMapLabel,
  ReplayMapConfig,
} from "../utils/replay-map";
import { escapeHtml } from "../utils/escape";

export const DEFAULT_VALUE_MODEL_PATH =
  "/home/sunrise/orbitwars/pantheow/bots/mine/trojan_horse/train/weights/xgb_46p12e88t11_latest.json";

export interface MatchConfig {
  games: number;
  mode: "fast" | "faithful" | "ultrafast" | "value";
  seed: "random" | "replay" | number;
  replayMap?: ReplayMapConfig | null;
  format: "2p" | "4p";
  valueModelPath: string;
}

export interface MatchConfigHandle {
  getConfig(): MatchConfig;
}

export function mountMatchConfigBar(
  root: HTMLElement,
  onChange: (cfg: MatchConfig) => void,
  onError?: (msg: string) => void,
): MatchConfigHandle {
  const config: MatchConfig = {
    games: 1,
    mode: "fast",
    seed: "random",
    replayMap: null,
    format: "2p",
    valueModelPath: DEFAULT_VALUE_MODEL_PATH,
  };
  let customSeed = 42;

  function clearReplayForNativeModes() {
    if (config.mode !== "ultrafast" && config.mode !== "value") return;
    if (config.seed === "replay" || config.replayMap) {
      config.seed = "random";
      config.replayMap = null;
    }
  }

  function render() {
    clearReplayForNativeModes();
    const replayLabel = replayMapLabel(config.replayMap);
    const canUseReplay = config.mode !== "ultrafast" && config.mode !== "value";
    root.innerHTML = `
      <div class="config-bar">
        <div class="config-group">
          <span class="config-label">format</span>
          <button class="config-pill ${config.format === "2p" ? "on" : ""}" data-k="format" data-v="2p">2p</button>
          <button class="config-pill ${config.format === "4p" ? "on" : ""}" data-k="format" data-v="4p">4p</button>
        </div>
        <div class="config-group">
          <span class="config-label">games</span>
          ${[1, 3, 5, 10, 20]
            .map(
              (n) =>
                `<button class="config-pill ${config.games === n ? "on" : ""}" data-k="games" data-v="${n}">${n}</button>`,
            )
            .join("")}
        </div>
        <div class="config-group">
          <span class="config-label">mode</span>
          <button class="config-pill ${config.mode === "ultrafast" ? "on" : ""}" data-k="mode" data-v="ultrafast" title="Native Rust engine, no replays (tournament throughput)">ultrafast</button>
          <button class="config-pill ${config.mode === "value" ? "on" : ""}" data-k="mode" data-v="value" title="Native Rust engine with XGBoost value trace">value</button>
          <button class="config-pill ${config.mode === "fast" ? "on" : ""}" data-k="mode" data-v="fast" title="In-process kaggle-environments">fast</button>
          <button class="config-pill ${config.mode === "faithful" ? "on" : ""}" data-k="mode" data-v="faithful" title="Subprocess + HTTP (Kaggle protocol)">faithful</button>
        </div>
        ${config.mode === "value"
          ? `<div class="config-group config-group-wide">
              <span class="config-label">model</span>
              <input
                id="config-value-model"
                class="config-input config-value-path"
                type="text"
                spellcheck="false"
                value="${escapeHtml(config.valueModelPath)}"
                title="${escapeHtml(config.valueModelPath)}"
              >
            </div>`
          : ""}
        <div class="config-group">
          <span class="config-label">seed</span>
          <button class="config-pill ${config.seed === "random" ? "on" : ""}" data-k="seed" data-v="random">random</button>
          <button class="config-pill ${typeof config.seed === "number" ? "on" : ""}" data-k="seed" data-v="custom">custom</button>
          ${canUseReplay
            ? `<button class="config-pill ${config.seed === "replay" ? "on" : ""}" data-k="seed" data-v="replay">replay</button>`
            : ""}
          <input
            id="config-custom-seed"
            class="config-input"
            type="number"
            inputmode="numeric"
            value="${customSeed}"
            ${typeof config.seed !== "number" ? "disabled" : ""}
          >
          <input id="config-replay-file" type="file" accept=".json,application/json" hidden>
          <span
            class="config-file-label"
            title="${escapeHtml(replayLabel)}"
            ${canUseReplay && (config.seed === "replay" || config.replayMap) ? "" : "hidden"}
          >${escapeHtml(replayLabel)}</span>
        </div>
      </div>
    `;

    root.querySelectorAll<HTMLButtonElement>(".config-pill").forEach((el) => {
      el.addEventListener("click", () => {
        const k = el.dataset.k as keyof MatchConfig;
        const v = el.dataset.v!;
        if (k === "games") config.games = parseInt(v, 10);
        else if (k === "mode") {
          config.mode = v as "fast" | "faithful" | "ultrafast" | "value";
          clearReplayForNativeModes();
        }
        else if (k === "seed") {
          if (v === "replay") {
            if (config.mode === "ultrafast" || config.mode === "value") return;
            root.querySelector<HTMLInputElement>("#config-replay-file")?.click();
            return;
          }
          config.seed = v === "random" ? "random" : customSeed;
          config.replayMap = null;
        }
        else if (k === "format") config.format = v as "2p" | "4p";
        onChange({ ...config });
        render();
      });
    });

    root.querySelector<HTMLInputElement>("#config-custom-seed")?.addEventListener("input", (e) => {
      const next = parseInt((e.target as HTMLInputElement).value, 10);
      if (Number.isNaN(next)) return;
      customSeed = next;
      if (typeof config.seed === "number") {
        config.seed = customSeed;
        onChange({ ...config });
      }
    });

    root.querySelector<HTMLInputElement>("#config-replay-file")?.addEventListener("change", async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      if (config.mode === "ultrafast" || config.mode === "value") {
        clearReplayForNativeModes();
        render();
        return;
      }
      try {
        config.replayMap = await replayMapFromFile(file);
        config.seed = "replay";
        onChange({ ...config });
        render();
      } catch (err) {
        config.replayMap = null;
        onError?.((err as Error).message);
      }
    });

    root.querySelector<HTMLInputElement>("#config-value-model")?.addEventListener("input", (e) => {
      config.valueModelPath = (e.target as HTMLInputElement).value.trim();
      onChange({ ...config });
    });
  }

  render();
  return { getConfig: () => ({ ...config }) };
}
