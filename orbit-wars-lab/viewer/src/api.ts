/** Typed API client. Uses /api prefix (Vite dev proxy or prod same-origin). */

export interface AgentInfo {
  id: string;
  name: string;
  bucket: "baselines" | "external" | "mine";
  description?: string | null;
  author?: string | null;
  source_url?: string | null;
  version?: string | null;
  date_fetched?: string | null;
  tags: string[];
  disabled: boolean;
  has_yaml: boolean;
  path: string;
  last_error?: string | null;
}

export interface Rating {
  agent_id: string;
  mu: number;
  sigma: number;
  conservative: number;
  games_played: number;
  rank: number;
}

export interface RunSummary {
  id: string;
  started_at: string;
  finished_at?: string | null;
  mode: "fast" | "faithful" | "ultrafast";
  format: "2p" | "4p";
  status: "queued" | "running" | "completed" | "aborted";
  total_matches: number;
  matches_done: number;
  is_quick_match?: boolean;
  shape?: "round-robin" | "gauntlet";
  challenger_id?: string | null;
}

export interface RunningMatch {
  run_id: string;
  match_id: string;
  agent_ids: string[];
  mode: string;
  started_at: string;
  elapsed_s: number;
}

export interface SchedulerTournament {
  id: string;
  status: "queued" | "running" | "completed" | "aborted";
  mode: string;
  format: "2p" | "4p";
  shape: "round-robin" | "gauntlet";
  challenger_id?: string | null;
  is_quick_match: boolean;
  total_matches: number;
  matches_done: number;
  queued: number;
  running: number;
  started_at: string;
}

export interface SchedulerStatus {
  concurrency: number;
  running_count: number;
  queued_total: number;
  tournaments: SchedulerTournament[];
  running: RunningMatch[];
}

export interface KaggleSubmission {
  submission_id: number;
  description: string;
  date: string;
  status: string;
  mu: number | null;
  sigma: number | null;
  rank: number | null;
  games_played: number | null;
}

export interface AgentLogsResponse {
  submission_id: number;
  episode_id: number;
  agent_idx: number;
  text: string;
}

export interface ScrapeJob {
  job_id: string;
  submission_id: number;
  count: number;
  status: "pending" | "running" | "completed" | "failed";
  total: number;
  downloaded: number;
  error: string | null;
}

export interface AgentRuntime {
  agent_id: string;
  total_turns: number;
  total_seconds: number;
  avg_ms: number;
  last_updated: string;
}

export interface MatchResult {
  match_id: string;
  agent_ids: string[];
  winner: string | null;
  scores: number[];
  turns: number;
  duration_s: number;
  status: string;
  seed: number;
  replay_path: string;
  // Populated when status != "ok"; surfaced as a tooltip + short hint in the
  // result card so a failed match is debuggable from the UI.
  error?: string | null;
}

async function j<T>(path: string, opts?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, opts);
  if (!r.ok) {
    const err = new Error(
      `${r.status} ${r.statusText}: ${opts?.method ?? "GET"} /api${path}`,
    ) as Error & { status?: number };
    err.status = r.status;
    throw err;
  }
  return r.json();
}

export const api = {
  listAgents: () => j<AgentInfo[]>("/agents"),
  getAgent: (id: string) => j<AgentInfo>(`/agents/${id}`),
  getRatings: (format: "2p" | "4p" = "2p") =>
    j<Rating[]>(`/ratings?format=${format}`),
  listRuns: (opts?: { excludeQuickMatch?: boolean }) => {
    const qs = opts?.excludeQuickMatch ? "?exclude_quick_match=true" : "";
    return j<RunSummary[]>(`/runs${qs}`);
  },
  getRun: (id: string) => j<{
    id: string;
    config?: any;
    results?: { matches: MatchResult[]; summary: any; total_matches: number };
    trueskill?: any;
    run?: RunSummary;
  }>(`/runs/${id}`),
  getRunProgress: (id: string) =>
    j<{ status: string; matches_done: number; total_matches: number }>(
      `/runs/${id}/progress`,
    ),
  getReplay: (runId: string, matchId: string) =>
    j<any>(`/replays/${runId}/${matchId}`),
  startTournament: (cfg: {
    agents: string[];
    games_per_pair: number;
    mode: string;
    format: string;
    save_replays?: boolean;
    seed_base?: number;
    seed_mode?: "fixed" | "random" | "replay";
    replay_map?: any;
    is_quick_match?: boolean;
    shape?: "round-robin" | "gauntlet";
    challenger_id?: string | null;
  }) =>
    j<{ run_id: string; status: string }>("/tournaments", {
      method: "POST",
      body: JSON.stringify(cfg),
      headers: { "Content-Type": "application/json" },
    }),
  // Stop a tournament: drop its queued matches + kill its in-flight ones.
  stopTournament: (runId: string) =>
    j<{ run_id: string; status: string }>(`/tournaments/${runId}/stop`, {
      method: "POST",
    }),
  getScheduler: () => j<SchedulerStatus>(`/scheduler`),
  getRunningMatches: () => j<RunningMatch[]>(`/scheduler/running`),
  setConcurrency: (concurrency: number) =>
    j<{ concurrency: number }>(`/scheduler/concurrency`, {
      method: "PUT",
      body: JSON.stringify({ concurrency }),
      headers: { "Content-Type": "application/json" },
    }),
  // Recycle the worker pool so rebuilt native bot binaries get picked up.
  restartPool: () =>
    j<{ restarted: boolean }>(`/scheduler/restart-pool`, { method: "POST" }),
  deleteLocalReplay: (runId: string, matchId: string) =>
    j<{ deleted: boolean }>(`/replays/${runId}/${matchId}`, { method: "DELETE" }),
  deleteKaggleReplay: (submissionId: number, episodeId: number) =>
    j<{ deleted: boolean }>(`/kaggle-replays/${submissionId}/${episodeId}`, {
      method: "DELETE",
    }),
  deleteRun: (runId: string) =>
    j<{ deleted: boolean }>(`/runs/${runId}`, { method: "DELETE" }),
  deleteAgent: (agentId: string) =>
    j<{ deleted: boolean }>(`/agents/${agentId}`, { method: "DELETE" }),
  resetRatings: (format: "2p" | "4p" | "all" = "all") =>
    j<{ reset: boolean }>(`/ratings/reset?format=${format}`, { method: "POST" }),
  listRuntimes: () => j<AgentRuntime[]>(`/runtimes`),
  clearAgentRuntime: (agentId: string) =>
    j<{ cleared: boolean; agent_id: string }>(`/runtimes/${agentId}`, {
      method: "DELETE",
    }),
  listKaggleSubmissions: () =>
    j<KaggleSubmission[]>(`/kaggle-submissions`),
  submitKaggleAgent: (agentId: string, description: string) =>
    j<{ ok: boolean; message: string }>(`/kaggle-submissions`, {
      method: "POST",
      body: JSON.stringify({ agent_id: agentId, description }),
      headers: { "Content-Type": "application/json" },
    }),
  getAgentLogs: (submissionId: number, episodeId: number) =>
    j<AgentLogsResponse>(
      `/kaggle-submissions/${submissionId}/episodes/${episodeId}/logs`,
    ),
  startScrape: (submissionId: number, count: number) =>
    j<{ job_id: string; status: string }>(`/replays/scrape`, {
      method: "POST",
      body: JSON.stringify({ submission_id: submissionId, count }),
      headers: { "Content-Type": "application/json" },
    }),
  getScrapeStatus: (jobId: string) => j<ScrapeJob>(`/replays/scrape/${jobId}`),
  getKaggleAuthStatus: () => j<KaggleAuthStatus>(`/kaggle-auth`),
  saveKaggleAuth: (token: string, signal?: AbortSignal) =>
    j<KaggleAuthStatus>(`/kaggle-auth`, {
      method: "POST",
      body: JSON.stringify({ token }),
      headers: { "Content-Type": "application/json" },
      signal,
    }),
  clearKaggleAuth: () =>
    j<KaggleAuthStatus>(`/kaggle-auth`, { method: "DELETE" }),
};

export interface KaggleAuthStatus {
  connected: boolean;
  username: string | null;
  source: "file" | "env" | null;
  shadowed?: boolean;
  saved_username?: string | null;
  deleted?: boolean;
}
