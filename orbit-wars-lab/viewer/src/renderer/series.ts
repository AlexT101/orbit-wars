/**
 * Per-step time series derived from a replay JSON — shared by:
 *   - the iframe renderer (publishes via storage on each render), and
 *   - the Quick Match parent (computes directly when it loads a replay,
 *     so the Graphs accordion works even if the iframe hasn't written yet
 *     or is running stale cached code).
 *
 * Shape mirrors what the standalone visualize.py sidebar consumes.
 */

export interface ReplaySeries {
  totalSteps: number;
  numAgents: number;
  ships: number[][];
  production: number[][];
  planets: number[][];
  shipDelta: number[];
  valueFunction?: {
    label?: string;
    modelPath?: string;
    values: Array<Array<number | null>>;
    error?: string;
  };
}

/** `replay` should be the env.toJSON()-shaped object with `.steps`. */
export function computeReplaySeries(replay: unknown, numAgentsHint?: number): ReplaySeries | null {
  const steps = (replay as any)?.steps;
  if (!Array.isArray(steps) || steps.length === 0) return null;

  let numAgents = numAgentsHint ?? 0;
  if (!numAgents) {
    numAgents =
      ((replay as any)?.info?.TeamNames?.length as number | undefined) ||
      ((replay as any)?.info?.Agents?.length as number | undefined) ||
      0;
  }
  if (!numAgents) {
    // Fallback: scan first observation for max owner index.
    const obs0 = steps[0]?.[0]?.observation;
    let max = 1;
    for (const p of (obs0?.planets || [])) if (p[1] > max) max = p[1];
    for (const f of (obs0?.fleets || [])) if (f[1] > max) max = f[1];
    numAgents = max + 1;
  }

  const T = steps.length;
  const ships: number[][] = [];
  const production: number[][] = [];
  const planetsCnt: number[][] = [];
  for (let p = 0; p < numAgents; p++) {
    ships.push(new Array(T).fill(0));
    production.push(new Array(T).fill(0));
    planetsCnt.push(new Array(T).fill(0));
  }
  for (let t = 0; t < T; t++) {
    const obs = steps[t]?.[0]?.observation;
    if (!obs) continue;
    for (const pl of (obs.planets || [])) {
      const owner = pl[1];
      if (owner >= 0 && owner < numAgents) {
        ships[owner][t] += pl[5];
        production[owner][t] += pl[6];
        planetsCnt[owner][t] += 1;
      }
    }
    for (const fl of (obs.fleets || [])) {
      const owner = fl[1];
      if (owner >= 0 && owner < numAgents) ships[owner][t] += fl[6];
    }
  }
  const shipDelta = new Array(T).fill(0);
  if (numAgents >= 2) {
    for (let i = 1; i < T; i++) {
      const p0 = ships[0][i] - ships[0][i - 1];
      const p1 = ships[1][i] - ships[1][i - 1];
      shipDelta[i] = p0 - p1;
    }
  }
  const vf = (replay as any)?.value_function;
  const vfError = (replay as any)?.value_function_error;
  let valueFunction: ReplaySeries["valueFunction"] | undefined;
  if (vf && Array.isArray(vf.values)) {
    valueFunction = {
      label: typeof vf.label === "string" ? vf.label : undefined,
      modelPath: typeof vf.model_path === "string" ? vf.model_path : undefined,
      values: vf.values.map((row: any) =>
        Array.isArray(row)
          ? row.slice(0, T).map((v: any) => typeof v === "number" ? v : null)
          : new Array(T).fill(null),
      ),
      error: typeof vfError === "string" ? vfError : undefined,
    };
  } else if (typeof vfError === "string") {
    valueFunction = {
      values: [],
      error: vfError,
    };
  }
  return {
    totalSteps: T,
    numAgents,
    ships,
    production,
    planets: planetsCnt,
    shipDelta,
    valueFunction,
  };
}
