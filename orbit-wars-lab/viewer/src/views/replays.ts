/**
 * Replays view - unified list of local tournament matches + Kaggle-scraped
 * episodes, with an import panel to fetch more episodes for a given
 * submission_id.
 */

import { installHeaderNav } from "../components/header-nav";
import { navigate } from "../router";
import { api, AgentInfo } from "../api";
import { escapeHtml } from "../utils/escape";

interface LocalReplay {
  source: "local";
  run_id: string;
  match_id: string;
  agent_ids: string[];
  winner: string | null;
  turns: number;
  duration_s: number;
  status: string;
  started_at?: string;
}

interface KaggleReplay {
  source: "kaggle";
  submission_id: number;
  episode_id: number;
  path: string;
  agents?: Array<{ name?: string; submissionId?: number }>;
  team_names?: string[];
  winner?: string | null;
  type?: string;
  endTime?: string;
}

type Replay = LocalReplay | KaggleReplay;
type Source = "all" | "local" | "kaggle";
type SortOrder = "newest" | "oldest" | "turns-desc" | "turns-asc";
type AgentBucket = AgentInfo["bucket"];

interface ReplayParticipant {
  rawName: string;
  displayName: string;
  bucket: AgentBucket | null;
  searchText: string;
}

type ReplaysFilter = {
  source: Source;
  sort: SortOrder;
  searchA: string;
  searchB: string;
};

const KNOWN_BUCKETS = new Set<AgentBucket>(["baselines", "external", "mine"]);
const FILTER_KEY = "ow-replays-filter";

let pollInterval: number | null = null;

function normalizeSearchTerm(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function splitAgentId(raw: string): { bucket: AgentBucket | null; shortId: string } {
  const parts = raw.split("/");
  if (parts.length >= 2 && KNOWN_BUCKETS.has(parts[0] as AgentBucket)) {
    return {
      bucket: parts[0] as AgentBucket,
      shortId: parts.slice(1).join("/"),
    };
  }
  return { bucket: null, shortId: raw };
}

function agentLookupKeys(agent: AgentInfo): string[] {
  const { shortId } = splitAgentId(agent.id);
  return [
    agent.id,
    agent.name,
    shortId,
    `${agent.name} (${agent.bucket})`,
    `${agent.bucket}/${shortId}`,
  ];
}

function makeParticipant(
  rawName: string,
  agentsById: Map<string, AgentInfo>,
  agentsByKey: Map<string, AgentInfo>,
): ReplayParticipant {
  const direct = agentsById.get(rawName);
  const normalizedRaw = normalizeSearchTerm(rawName);
  const known = direct ?? agentsByKey.get(normalizedRaw) ?? null;
  if (known) {
    const { shortId } = splitAgentId(known.id);
    return {
      rawName,
      displayName: known.name,
      bucket: known.bucket,
      searchText: [
        known.id,
        known.name,
        shortId,
        `${known.name} (${known.bucket})`,
        rawName,
      ]
        .map(normalizeSearchTerm)
        .filter(Boolean)
        .join("\n"),
    };
  }

  const { bucket, shortId } = splitAgentId(rawName);
  const displayName = shortId.trim() || rawName.trim() || "?";
  return {
    rawName,
    displayName,
    bucket,
    searchText: [rawName, displayName, bucket ? `${displayName} (${bucket})` : ""]
      .map(normalizeSearchTerm)
      .filter(Boolean)
      .join("\n"),
  };
}

function getReplayParticipants(
  replay: Replay,
  agentsById: Map<string, AgentInfo>,
  agentsByKey: Map<string, AgentInfo>,
): ReplayParticipant[] {
  if (replay.source === "local") {
    return replay.agent_ids.map((agentId) =>
      makeParticipant(agentId, agentsById, agentsByKey),
    );
  }
  const names =
    replay.team_names && replay.team_names.length > 0
      ? replay.team_names
      : ((replay.agents || []).map((agent) => agent.name).filter(Boolean) as string[]);
  return names.map((name) => makeParticipant(name, agentsById, agentsByKey));
}

function renderParticipantHtml(
  participant: ReplayParticipant,
  winnerRawName?: string | null,
): string {
  let cls = "replay-name";
  if (winnerRawName) {
    cls += participant.rawName === winnerRawName ? " replay-name-winner" : " replay-name-loser";
  }
  return `<span class="${cls}">${escapeHtml(participant.displayName)}</span>`;
}

function formatRunId(id: string): string {
  const m = id.match(/^\d{4}-(\d{2}-\d{2})-(.+)$/);
  return m ? `${m[1]} (${m[2]})` : id;
}

function participantText(participant: ReplayParticipant): string {
  return participant.displayName;
}

function readFilter(): ReplaysFilter {
  try {
    const raw = sessionStorage.getItem(FILTER_KEY);
    if (!raw) return { source: "all", sort: "newest", searchA: "", searchB: "" };
    const parsed = JSON.parse(raw);
    return {
      source: (["all", "local", "kaggle"] as const).includes(parsed.source)
        ? parsed.source
        : "all",
      sort: (["newest", "oldest", "turns-desc", "turns-asc"] as const).includes(parsed.sort)
        ? parsed.sort
        : "newest",
      searchA: typeof parsed.searchA === "string" ? parsed.searchA : "",
      searchB: typeof parsed.searchB === "string" ? parsed.searchB : "",
    };
  } catch {
    return { source: "all", sort: "newest", searchA: "", searchB: "" };
  }
}

export async function renderReplays(
  root: HTMLElement,
  subFilter?: string,
): Promise<void> {
  root.innerHTML = `
    <main class="dashboard replays-view">
      <section>
        <div class="replays-toolbar">
          <div class="source-pills">
            <button class="source-pill on" data-source="all">All</button>
            <button class="source-pill" data-source="local">Local</button>
            <button class="source-pill" data-source="kaggle">Kaggle LB</button>
          </div>
          <div class="replays-match-search" aria-label="Replay bot matchup filters">
            <input id="replays-search-a" class="picker-search" type="text" placeholder="Bot A">
            <span class="replays-match-search-vs">vs</span>
            <input id="replays-search-b" class="picker-search" type="text" placeholder="Bot B">
          </div>
          <label class="replays-sort">
            Submission
            <select id="replays-sub-select">
              <option value="">All submissions</option>
            </select>
          </label>
          <label class="replays-sort">
            Sort
            <select id="replays-sort-select">
              <option value="newest" selected>Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="turns-desc">Turns: most first</option>
              <option value="turns-asc">Turns: least first</option>
            </select>
          </label>
        </div>
        <div id="scrape-panel" class="scrape-panel">
          <div class="scrape-row">
            <label>Kaggle replay URL
              <input type="text" id="scrape-url"
                     placeholder="https://www.kaggle.com/.../episodes/70123456?submissionId=51799179">
            </label>
            <div class="scrape-actions">
              <button class="scrape-btn go" id="scrape-go">Import</button>
            </div>
          </div>
          <div class="scrape-hint">
            Paste a Kaggle replay URL - either an episode page or a leaderboard link containing <code>?episodeId=&lt;id&gt;</code>.
            The <code>submissionId</code> in the URL flags which bot you care about; we fetch that single episode.
          </div>
          <div id="scrape-status" class="scrape-status" hidden></div>
        </div>
        <div id="replays-list" class="replays-list"></div>
      </section>
    </main>
  `;
  installHeaderNav(root, "replays");

  const agentsById = new Map<string, AgentInfo>();
  const agentsByKey = new Map<string, AgentInfo>();

  function indexAgents(agents: AgentInfo[]): void {
    agentsById.clear();
    agentsByKey.clear();
    for (const agent of agents) {
      agentsById.set(agent.id, agent);
      for (const key of agentLookupKeys(agent)) {
        const normalized = normalizeSearchTerm(key);
        if (normalized && !agentsByKey.has(normalized)) {
          agentsByKey.set(normalized, agent);
        }
      }
    }
  }

  const agentDirectoryLoad = api.listAgents({ includeDisabled: true })
    .then((agents) => {
      indexAgents(agents);
    })
    .catch(() => {
      agentsById.clear();
      agentsByKey.clear();
    });

  const subSelect = document.getElementById("replays-sub-select") as HTMLSelectElement;
  void (async () => {
    try {
      const subs = await api.listKaggleSubmissions();
      for (const s of subs) {
        const opt = document.createElement("option");
        opt.value = String(s.submission_id);
        const shortDesc = (s.description || "").slice(0, 50);
        opt.textContent = `${s.submission_id}${shortDesc ? " - " + shortDesc : ""}`;
        if (subFilter && opt.value === subFilter) opt.selected = true;
        subSelect.appendChild(opt);
      }
    } catch {
      // Leave only "All submissions" if Kaggle auth is not configured.
    }
  })();
  subSelect.addEventListener("change", () => {
    const value = subSelect.value;
    location.hash = value ? `#/replays?sub=${encodeURIComponent(value)}` : "#/replays";
  });

  if (subFilter) {
    const digits = /^\d+$/.test(subFilter) ? subFilter : escapeHtml(subFilter);
    const chip = document.createElement("div");
    chip.className = "filter-chip";
    chip.innerHTML =
      `Filtered by submission <strong>${digits}</strong> ` +
      `<button class="filter-chip-x" title="Clear filter">&times;</button>`;
    root.querySelector<HTMLElement>(".replays-toolbar")!.prepend(chip);
    chip.querySelector<HTMLButtonElement>(".filter-chip-x")!.addEventListener(
      "click",
      () => {
        location.hash = "#/replays";
      },
    );
  }

  const restored = readFilter();
  let currentSource: Source = restored.source;
  let currentSort: SortOrder = restored.sort;
  let currentSearchA = restored.searchA;
  let currentSearchB = restored.searchB;
  let hasLoadedList = false;
  let listRequestId = 0;
  let loadedItems: Replay[] = [];

  root.querySelectorAll<HTMLButtonElement>("[data-source]").forEach((button) =>
    button.classList.toggle("on", button.dataset.source === currentSource),
  );
  (document.getElementById("replays-sort-select") as HTMLSelectElement).value = currentSort;
  (document.getElementById("replays-search-a") as HTMLInputElement).value = currentSearchA;
  (document.getElementById("replays-search-b") as HTMLInputElement).value = currentSearchB;

  function saveFilter(): void {
    try {
      sessionStorage.setItem(
        FILTER_KEY,
        JSON.stringify({
          source: currentSource,
          sort: currentSort,
          searchA: currentSearchA,
          searchB: currentSearchB,
        }),
      );
    } catch {
      // Ignore storage quota errors.
    }
  }

  function playedAtMs(replay: Replay): number {
    if (replay.source === "kaggle" && replay.endTime) {
      const t = Date.parse(replay.endTime);
      if (!Number.isNaN(t)) return t;
    }
    if (replay.source === "local" && replay.started_at) {
      const t = Date.parse(replay.started_at);
      if (!Number.isNaN(t)) return t;
    }
    const ts = (replay as Replay & { ts?: unknown }).ts;
    return typeof ts === "number" ? ts * 1000 : 0;
  }

  function formatRelative(ms: number): string {
    if (!ms) return "";
    const diff = Math.max(0, Date.now() - ms);
    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 14) return `${days}d ago`;
    const weeks = Math.floor(days / 7);
    if (weeks < 8) return `${weeks} wk ago`;
    const months = Math.floor(days / 30);
    return `${months} mo ago`;
  }

  function sortItems(items: Replay[]): Replay[] {
    const copy = items.slice();
    if (currentSort === "newest" || currentSort === "oldest") {
      const sign = currentSort === "newest" ? -1 : 1;
      copy.sort((a, b) => sign * (playedAtMs(a) - playedAtMs(b)));
    } else {
      const sign = currentSort === "turns-desc" ? -1 : 1;
      copy.sort((a, b) => {
        const turnsA = a.source === "local" ? a.turns : 0;
        const turnsB = b.source === "local" ? b.turns : 0;
        return sign * (turnsA - turnsB);
      });
    }
    return copy;
  }

  function participantMatches(participant: ReplayParticipant, query: string): boolean {
    return participant.searchText.includes(query);
  }

  function replayMatchesParticipantFilters(replay: Replay): boolean {
    const queries = [currentSearchA, currentSearchB]
      .map(normalizeSearchTerm)
      .filter(Boolean);
    if (queries.length === 0) return true;

    const participants = getReplayParticipants(replay, agentsById, agentsByKey);
    if (participants.length === 0) return false;

    const used = new Set<number>();
    const matchQueryAt = (queryIdx: number): boolean => {
      if (queryIdx >= queries.length) return true;
      const query = queries[queryIdx];
      for (let participantIdx = 0; participantIdx < participants.length; participantIdx += 1) {
        if (used.has(participantIdx)) continue;
        if (!participantMatches(participants[participantIdx], query)) continue;
        used.add(participantIdx);
        if (matchQueryAt(queryIdx + 1)) return true;
        used.delete(participantIdx);
      }
      return false;
    };

    return matchQueryAt(0);
  }

  function applyClientFilters(items: Replay[]): Replay[] {
    let filtered = items;
    if (subFilter) {
      const sub = Number(subFilter);
      filtered = filtered.filter(
        (item) => item.source === "kaggle" && item.submission_id === sub,
      );
    }
    filtered = filtered.filter(replayMatchesParticipantFilters);
    return sortItems(filtered);
  }

  function renderList(items: Replay[]): void {
    const listEl = document.getElementById("replays-list")!;
    if (items.length === 0) {
      const hasParticipantFilters = Boolean(
        normalizeSearchTerm(currentSearchA) || normalizeSearchTerm(currentSearchB),
      );
      const message = hasParticipantFilters
        ? "No replays match those bot filters."
        : "No replays yet. Play a match in Quick Match or import from Kaggle.";
      listEl.innerHTML = `<div class="loading">${message}</div>`;
      return;
    }

    listEl.innerHTML = items
      .map((replay, idx) => {
        const playedMs = playedAtMs(replay);
        const relative = formatRelative(playedMs);
        const absolute = playedMs
          ? `${new Date(playedMs).toISOString().replace("T", " ").slice(0, 16)} UTC`
          : "";
        const timeCell = relative
          ? `<span class="replay-time" title="${absolute}">${relative}</span>`
          : "";
        const participants = getReplayParticipants(replay, agentsById, agentsByKey);
        const winnerRaw = replay.winner ?? null;
        const agentsHtml = participants.length > 0
          ? participants.map((p) => renderParticipantHtml(p, winnerRaw)).join(`<span class="replay-vs">vs</span>`)
          : "?";

        if (replay.source === "local") {
          const winner = replay.winner
            ? escapeHtml(participantText(makeParticipant(replay.winner, agentsById, agentsByKey)))
            : "draw";
          return `
            <div class="replay-item" data-idx="${idx}" data-kind="local"
                 data-run-id="${escapeHtml(replay.run_id)}" data-match-id="${escapeHtml(replay.match_id)}">
              <div class="replay-meta-row">
                <span class="replay-source local">local</span>
                <span class="replay-title">${agentsHtml}</span>
                <span class="replay-winner">winner: <strong>${winner}</strong></span>
                ${timeCell}
              </div>
              <div class="replay-meta-sub">
                run ${escapeHtml(formatRunId(replay.run_id))} - match ${escapeHtml(replay.match_id)} - ${replay.turns} turns - ${replay.duration_s.toFixed(1)}s - ${escapeHtml(replay.status)}
              </div>
              <button class="replay-delete" title="Delete replay">&times;</button>
            </div>
          `;
        }

        const winner = replay.winner
          ? `winner: <strong>${escapeHtml(
            participantText(makeParticipant(replay.winner, agentsById, agentsByKey)),
          )}</strong>`
          : (replay.type ? escapeHtml(replay.type) : "");
        return `
          <div class="replay-item" data-idx="${idx}" data-kind="kaggle"
               data-submission-id="${replay.submission_id}" data-episode-id="${replay.episode_id}">
            <div class="replay-meta-row">
              <span class="replay-source kaggle">kaggle</span>
              <span class="replay-title">${agentsHtml}</span>
              <span class="replay-winner">${winner}</span>
              ${timeCell}
            </div>
            <div class="replay-meta-sub">
              submission ${replay.submission_id} - episode ${replay.episode_id}
            </div>
            <button class="replay-delete" title="Delete replay">&times;</button>
          </div>
        `;
      })
      .join("");

    listEl.querySelectorAll<HTMLElement>(".replay-item").forEach((el) => {
      el.addEventListener("click", (ev) => {
        if ((ev.target as HTMLElement).closest(".replay-delete")) return;
        const kind = el.dataset.kind;
        const payload = kind === "local"
          ? { kind: "local", runId: el.dataset.runId!, matchId: el.dataset.matchId! }
          : {
              kind: "kaggle",
              submissionId: parseInt(el.dataset.submissionId!, 10),
              episodeId: parseInt(el.dataset.episodeId!, 10),
            };
        sessionStorage.setItem("ow-pending-replay", JSON.stringify(payload));
        navigate({ view: "quick-match" });
      });
    });

    listEl.querySelectorAll<HTMLButtonElement>(".replay-delete").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const item = btn.closest(".replay-item") as HTMLElement | null;
        if (!item) return;
        if (!confirm("Delete this replay?")) return;
        try {
          if (item.dataset.kind === "local") {
            await api.deleteLocalReplay(item.dataset.runId!, item.dataset.matchId!);
          } else {
            await api.deleteKaggleReplay(
              parseInt(item.dataset.submissionId!, 10),
              parseInt(item.dataset.episodeId!, 10),
            );
          }
          await loadList();
        } catch (e) {
          alert(`Delete failed: ${(e as Error).message}`);
        }
      });
    });
  }

  function renderVisibleList(): void {
    renderList(applyClientFilters(loadedItems));
  }

  async function loadList(options?: { showLoading?: boolean }): Promise<void> {
    const listEl = document.getElementById("replays-list")!;
    const requestId = ++listRequestId;
    const showLoading = options?.showLoading ?? !hasLoadedList;
    if (showLoading) {
      listEl.innerHTML = `<div class="loading">Loading...</div>`;
    }
    try {
      const response = await fetch(`/api/replays?source=${currentSource}`);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const items: Replay[] = await response.json();
      if (requestId !== listRequestId) return;
      loadedItems = items;
      renderVisibleList();
      hasLoadedList = true;
    } catch (e) {
      if (requestId !== listRequestId) return;
      if (!hasLoadedList) {
        listEl.innerHTML = `<div class="loading">Error: ${(e as Error).message}</div>`;
      }
    }
  }

  root.querySelectorAll<HTMLButtonElement>("[data-source]").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentSource = btn.dataset.source as Source;
      saveFilter();
      root.querySelectorAll<HTMLButtonElement>("[data-source]").forEach((button) => {
        button.classList.toggle("on", button === btn);
      });
      void loadList();
    });
  });

  (document.getElementById("replays-sort-select") as HTMLSelectElement)
    .addEventListener("change", (e) => {
      currentSort = (e.target as HTMLSelectElement).value as SortOrder;
      saveFilter();
      renderVisibleList();
    });

  (document.getElementById("replays-search-a") as HTMLInputElement)
    .addEventListener("input", (e) => {
      currentSearchA = (e.target as HTMLInputElement).value;
      saveFilter();
      renderVisibleList();
    });

  (document.getElementById("replays-search-b") as HTMLInputElement)
    .addEventListener("input", (e) => {
      currentSearchB = (e.target as HTMLInputElement).value;
      saveFilter();
      renderVisibleList();
    });

  document.getElementById("scrape-go")!.addEventListener("click", async () => {
    const urlInput = document.getElementById("scrape-url") as HTMLInputElement;
    const url = urlInput.value.trim();
    if (!url) {
      alert("Paste a Kaggle URL");
      return;
    }
    const statusEl = document.getElementById("scrape-status")!;
    statusEl.hidden = false;
    statusEl.textContent = "Fetching...";
    try {
      const response = await fetch("/api/replays/scrape-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || response.statusText);
      }
      const data = await response.json();
      statusEl.textContent =
        `Imported: episode ${data.episode_id} (submission ${data.submission_id || "?"})`;
      urlInput.value = "";
      await loadList();
    } catch (e) {
      statusEl.textContent = `Error: ${(e as Error).message}`;
    }
  });

  await agentDirectoryLoad;
  await loadList({ showLoading: true });

  if (pollInterval !== null) window.clearInterval(pollInterval);
  pollInterval = window.setInterval(() => {
    if (document.hidden) return;
    void loadList();
  }, 10000);
}
