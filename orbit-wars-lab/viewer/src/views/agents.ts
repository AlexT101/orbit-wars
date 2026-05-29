/**
 * Agents view — browse all agents in the zoo (baselines / external / mine),
 * filter, search, click for details, delete from UI.
 */

import { api, AgentInfo, AgentRuntime } from "../api";
import { installHeaderNav } from "../components/header-nav";
import { navigate } from "../router";
import { escapeHtml } from "../utils/escape";

function fmtAvgMs(avgMs: number): string {
  if (!Number.isFinite(avgMs) || avgMs <= 0) return "—";
  if (avgMs < 1) return `${avgMs.toFixed(2)} ms`;
  if (avgMs < 10) return `${avgMs.toFixed(1)} ms`;
  return `${Math.round(avgMs)} ms`;
}

let pollInterval: number | null = null;

// Keys for per-view filter state in sessionStorage. Survives
// navigation-within-tab (same-tab only, by design — fresh tab = fresh view).
const FILTER_KEY = "ow-agents-filter";
type AgentsFilter = {
  bucket: "all" | "baselines" | "external" | "mine";
  search: string;
};

function readFilter(): AgentsFilter {
  try {
    const raw = sessionStorage.getItem(FILTER_KEY);
    if (!raw) return { bucket: "all", search: "" };
    const p = JSON.parse(raw);
    return {
      bucket: (["all", "baselines", "external", "mine"] as const).includes(p.bucket) ? p.bucket : "all",
      search: typeof p.search === "string" ? p.search : "",
    };
  } catch {
    return { bucket: "all", search: "" };
  }
}

function writeFilter(f: AgentsFilter): void {
  try { sessionStorage.setItem(FILTER_KEY, JSON.stringify(f)); } catch { /* quota */ }
}

export async function renderAgents(root: HTMLElement): Promise<void> {
  const restored = readFilter();
  root.innerHTML = `
    <main class="dashboard">
      <section>
        <div class="section-head">
          <h2>Agents</h2>
          <span class="td-label" id="agents-count" style="margin-left: auto;"></span>
        </div>
        <div class="replays-toolbar">
          <div class="source-pills">
            <button class="source-pill on" data-bucket="all">All</button>
            <button class="source-pill" data-bucket="baselines">Baselines</button>
            <button class="source-pill" data-bucket="external">External</button>
            <button class="source-pill" data-bucket="mine">Mine</button>
          </div>
          <input id="agents-search" class="picker-search" placeholder="search…" style="flex: 1; max-width: 300px;">
        </div>
        <div id="agents-list" class="replays-list"></div>
      </section>
    </main>
  `;
  installHeaderNav(root, "agents");

  let bucketFilter: AgentsFilter["bucket"] = restored.bucket;
  let searchTerm = restored.search;
  let hasLoadedList = false;
  let listRequestId = 0;

  // Apply restored filter state to the just-rendered toolbar.
  root.querySelectorAll<HTMLButtonElement>("[data-bucket]").forEach((b) => {
    b.classList.toggle("on", b.dataset.bucket === bucketFilter);
  });
  (document.getElementById("agents-search") as HTMLInputElement).value = searchTerm;

  async function loadList(options?: { showLoading?: boolean }) {
    const listEl = document.getElementById("agents-list")!;
    const requestId = ++listRequestId;
    const showLoading = options?.showLoading ?? !hasLoadedList;
    if (showLoading) {
      listEl.innerHTML = `<div class="loading">Loading…</div>`;
    }
    try {
      // Runtimes failure must not blank the agents tab — fall back to an
      // empty map and render the list with '—' in the runtime column.
      const [agents, runtimes] = await Promise.all([
        api.listAgents(),
        api.listRuntimes().catch(() => [] as AgentRuntime[]),
      ]);
      if (requestId !== listRequestId) return;
      const runtimeByAgent = new Map(runtimes.map((r) => [r.agent_id, r]));
      renderList(agents, runtimeByAgent);
      hasLoadedList = true;
    } catch (e) {
      if (requestId !== listRequestId) return;
      if (!hasLoadedList) {
        listEl.innerHTML = `<div class="loading">Error: ${(e as Error).message}</div>`;
      }
    }
  }

  function renderList(agents: AgentInfo[], runtimes: Map<string, AgentRuntime>) {
    const listEl = document.getElementById("agents-list")!;
    const filtered = agents.filter((a) => {
      if (bucketFilter !== "all" && a.bucket !== bucketFilter) return false;
      if (searchTerm) {
        const t = searchTerm.toLowerCase();
        if (!a.id.toLowerCase().includes(t) && !a.name.toLowerCase().includes(t))
          return false;
      }
      return true;
    });
    document.getElementById("agents-count")!.textContent =
      `${filtered.length} / ${agents.length}`;
    if (filtered.length === 0) {
      listEl.innerHTML = `<div class="loading">No agents match this filter.</div>`;
      return;
    }
    listEl.innerHTML = filtered
      .map((a) => {
        const desc = a.description ? escapeHtml(a.description.slice(0, 160)) : "";
        const errBadge = a.last_error?.trim()
          ? `<span class="replay-source" style="color: var(--error); background: rgba(255,138,138,0.08);">error</span>`
          : "";
        const disabledBadge = a.disabled
          ? `<span class="replay-source" style="color: var(--warning); background: rgba(255,184,74,0.08);">disabled</span>`
          : "";
        const safeId = escapeHtml(a.id);
        const rt = runtimes.get(a.id);
        const hasSamples = rt && rt.total_turns > 0;
        const rtText = hasSamples ? fmtAvgMs(rt!.avg_ms) : "—";
        const rtTooltip = hasSamples
          ? `avg of ${rt!.total_turns} turns (${rt!.total_seconds.toFixed(2)}s total)`
          : "no samples yet";
        const clearBtn = hasSamples
          ? `<button class="replay-delete runtime-clear" data-id="${safeId}" title="Clear runtime samples for this agent">↺</button>`
          : "";
        const author = a.author
          ? `<span class="replay-winner" style="margin-left: 8px;">by <strong>${escapeHtml(a.author)}</strong></span>`
          : "";
        return `
          <div class="replay-item" data-id="${safeId}">
            <div class="replay-meta-row">
              <span class="replay-source ${escapeHtml(a.bucket)}">${escapeHtml(a.bucket)}</span>
              ${errBadge}${disabledBadge}
              <span class="replay-title">${escapeHtml(a.name)}${author}</span>
              <span class="agent-runtime" title="${rtTooltip}">${rtText}/turn</span>
            </div>
            <div class="replay-meta-sub">
              ${desc || "&mdash;"}
            </div>
            ${clearBtn}
            <button class="replay-delete" data-id="${safeId}" title="Delete agent">×</button>
          </div>
        `;
      })
      .join("");

    listEl.querySelectorAll<HTMLElement>(".replay-item").forEach((row) => {
      row.addEventListener("click", (ev) => {
        if ((ev.target as HTMLElement).closest(".replay-delete")) return;
        const id = row.dataset.id!;
        navigate({ view: "agent", agentId: id });
      });
    });
    listEl.querySelectorAll<HTMLButtonElement>(".runtime-clear").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const id = btn.dataset.id!;
        if (!confirm(`Clear runtime samples for "${id}"?\n\nUse this after editing the agent — old samples will keep biasing the per-turn average otherwise. Ratings + replay history are untouched.`)) {
          return;
        }
        try {
          await api.clearAgentRuntime(id);
          await loadList();
        } catch (e) {
          alert(`Clear failed: ${(e as Error).message}`);
        }
      });
    });
    // Per-row delete (the × button). Skip if click originated on the
    // runtime-clear button — those two share the .replay-delete class for
    // positioning, but each has its own click handler with different semantics.
    listEl.querySelectorAll<HTMLButtonElement>(".replay-delete:not(.runtime-clear)").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const id = btn.dataset.id!;
        // Baseline agents seed TrueSkill history + every tournament references
        // them. Accidentally deleting breaks a bunch of downstream assumptions.
        // Require typing the id to confirm.
        if (id.startsWith("baselines/")) {
          const typed = prompt(
            `Deleting baseline agent "${id}" is rarely what you want — tournaments that reference it will fail, and the seeded leaderboard loses meaning.\n\nType the agent id to confirm:`,
          );
          if (typed !== id) {
            if (typed !== null) alert("Mismatch — baseline not deleted.");
            return;
          }
        } else if (!confirm(`Delete agent "${id}"?\n\nRemoves the folder from disk. Ratings + replay history kept.`)) {
          return;
        }
        try {
          await api.deleteAgent(id);
          await loadList();
        } catch (e) {
          alert(`Delete failed: ${(e as Error).message}`);
        }
      });
    });
  }

  root.querySelectorAll<HTMLButtonElement>("[data-bucket]").forEach((btn) => {
    btn.addEventListener("click", () => {
      bucketFilter = btn.dataset.bucket as typeof bucketFilter;
      writeFilter({ bucket: bucketFilter, search: searchTerm });
      root.querySelectorAll<HTMLButtonElement>("[data-bucket]").forEach((b) =>
        b.classList.toggle("on", b === btn),
      );
      void loadList();
    });
  });

  (document.getElementById("agents-search") as HTMLInputElement).addEventListener(
    "input",
    (e) => {
      searchTerm = (e.target as HTMLInputElement).value;
      writeFilter({ bucket: bucketFilter, search: searchTerm });
      void loadList();
    },
  );

  await loadList({ showLoading: true });

  if (pollInterval !== null) window.clearInterval(pollInterval);
  pollInterval = window.setInterval(() => {
    // Self-gc: clear when view is no longer mounted.
    if (!document.getElementById("agents-list")) {
      if (pollInterval !== null) window.clearInterval(pollInterval);
      pollInterval = null;
      return;
    }
    if (document.hidden) return;
    void loadList();
  }, 10000);
}
