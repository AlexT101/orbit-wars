const COMET_SPAWN_STEPS = new Set([50, 150, 250, 350, 450]);

export interface ReplayCometGroupConfig {
  spawn_step: number;
  paths: number[][][];
  ships: number;
}

export interface ReplayMapConfig {
  planets: number[][];
  initial_planets: number[][];
  angular_velocity: number;
  source_seed?: number | null;
  source_name?: string | null;
  num_players?: number | null;
  comet_schedule?: ReplayCometGroupConfig[];
}

function unwrapReplay(raw: any): any | null {
  if (raw?.replay?.steps) return raw.replay;
  if (raw?.environment?.steps) return raw.environment;
  if (raw?.steps) return raw;
  return null;
}

function normalizePlanetRows(value: unknown, fieldName: string): number[][] {
  if (!Array.isArray(value)) {
    throw new Error(`${fieldName} is missing`);
  }
  return value.map((row, idx) => {
    if (!Array.isArray(row) || row.length < 7) {
      throw new Error(`${fieldName}[${idx}] is not a planet row`);
    }
    const nums = row.slice(0, 7).map((v) => Number(v));
    if (nums.some((v) => !Number.isFinite(v))) {
      throw new Error(`${fieldName}[${idx}] contains non-numeric values`);
    }
    return nums;
  });
}

function readSourceSeed(replay: any): number | null {
  const seed = replay?.info?.seed ?? replay?.configuration?.seed ?? null;
  if (seed === null || seed === undefined) return null;
  const n = Number(seed);
  return Number.isSafeInteger(n) ? n : null;
}

function normalizeCometPaths(value: unknown, fieldName: string): number[][][] {
  if (!Array.isArray(value) || value.length !== 4) {
    throw new Error(`${fieldName} must contain 4 comet paths`);
  }
  return value.map((path, pathIdx) => {
    if (!Array.isArray(path) || path.length === 0) {
      throw new Error(`${fieldName}[${pathIdx}] is not a path`);
    }
    return path.map((point, pointIdx) => {
      if (!Array.isArray(point) || point.length < 2) {
        throw new Error(`${fieldName}[${pathIdx}][${pointIdx}] is not an x/y point`);
      }
      const x = Number(point[0]);
      const y = Number(point[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        throw new Error(`${fieldName}[${pathIdx}][${pointIdx}] contains non-numeric values`);
      }
      return [x, y];
    });
  });
}

function readCometShips(planets: unknown, planetIds: number[]): number | null {
  if (!Array.isArray(planets) || planetIds.length === 0) return null;
  const idSet = new Set(planetIds);
  for (const row of planets) {
    if (!Array.isArray(row) || row.length < 6) continue;
    const id = Number(row[0]);
    if (!idSet.has(id)) continue;
    const ships = Number(row[5]);
    return Number.isFinite(ships) && ships > 0 ? ships : null;
  }
  return null;
}

function extractCometSchedule(replay: any): ReplayCometGroupConfig[] {
  const steps = replay?.steps;
  if (!Array.isArray(steps)) return [];
  const schedule: ReplayCometGroupConfig[] = [];
  const seen = new Set<number>();
  for (let stepIdx = 0; stepIdx < steps.length; stepIdx += 1) {
    if (!COMET_SPAWN_STEPS.has(stepIdx) || seen.has(stepIdx)) continue;
    const step = steps[stepIdx];
    const firstState = Array.isArray(step) ? step[0] : null;
    const obs = firstState?.observation;
    const groups = obs?.comets;
    if (!Array.isArray(groups)) continue;
    const group = groups.find((g) => Number(g?.path_index) === 0);
    if (!group) continue;

    const planetIds = Array.isArray(group.planet_ids)
      ? group.planet_ids.map((v: unknown) => Number(v)).filter((v: number) => Number.isFinite(v))
      : [];
    const ships = readCometShips(obs?.planets, planetIds);
    if (ships === null) {
      throw new Error(`Replay comet at step ${stepIdx} is missing its planet ship count`);
    }
    schedule.push({
      spawn_step: stepIdx,
      paths: normalizeCometPaths(group.paths, `steps[${stepIdx}].observation.comets[0].paths`),
      ships,
    });
    seen.add(stepIdx);
  }
  return schedule;
}

export async function replayMapFromFile(file: File): Promise<ReplayMapConfig> {
  let raw: any;
  try {
    raw = JSON.parse(await file.text());
  } catch {
    throw new Error("Replay file is not valid JSON");
  }

  const replay = unwrapReplay(raw);
  const firstStep = replay?.steps?.[0];
  const firstState = Array.isArray(firstStep) ? firstStep[0] : null;
  const obs = firstState?.observation;
  if (!obs) {
    throw new Error("Replay is missing steps[0][0].observation");
  }

  const planets = normalizePlanetRows(obs.planets, "observation.planets");
  const initialPlanets = normalizePlanetRows(
    obs.initial_planets ?? obs.planets,
    "observation.initial_planets",
  );
  if (initialPlanets.length !== planets.length) {
    throw new Error("Replay initial_planets length does not match planets");
  }
  const angularVelocity = Number(obs.angular_velocity);
  if (!Number.isFinite(angularVelocity)) {
    throw new Error("Replay is missing observation.angular_velocity");
  }

  return {
    planets,
    initial_planets: initialPlanets,
    angular_velocity: angularVelocity,
    source_seed: readSourceSeed(replay),
    source_name: file.name,
    num_players: Array.isArray(firstStep) ? firstStep.length : null,
    comet_schedule: extractCometSchedule(replay),
  };
}

export function replayMapLabel(map: ReplayMapConfig | null | undefined): string {
  if (!map) return "no replay selected";
  const name = map.source_name || "replay";
  const players = map.num_players ? `${map.num_players}p` : "?p";
  const cometCount = map.comet_schedule?.length ?? 0;
  const cometText = cometCount ? ` - ${cometCount} comet spawns` : "";
  return `${name} - ${players} - ${map.planets.length} planets${cometText}`;
}
