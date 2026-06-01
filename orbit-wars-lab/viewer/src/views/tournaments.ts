/**
 * Tournaments view — create a new tournament + list historical ones.
 * Create panel: multi-select agents (bucket filter + search), config
 * (games_per_pair, mode, format), Start button.
 */

import { api, AgentInfo, Rating, RunSummary, SchedulerStatus } from "../api";
import { installHeaderNav } from "../components/header-nav";
import { navigate } from "../router";
import { escapeHtml } from "../utils/escape";

let pollInterval: number | null = null;

export async function renderTournaments(root: HTMLElement): Promise<void> {
  root.innerHTML = `
    <main class="dashboard">
      <section>
        <div class="section-head">
          <h2>Create Tournament</h2>
        </div>
        <div id="create-panel" class="scrape-panel">
          <div class="create-grid">
            <div class="create-agents">
              <div class="create-agents-head">
                <input id="create-search" class="picker-search" placeholder="search agents…">
                <div class="picker-tags">
                  <button class="picker-pill on" data-bucket="all">all</button>
                  <button class="picker-pill" data-bucket="baselines">baselines</button>
                  <button class="picker-pill" data-bucket="external">external</button>
                  <button class="picker-pill" data-bucket="mine">mine</button>
                </div>
                <div class="create-count"><span id="create-count-num">0</span> selected</div>
              </div>
              <ul id="create-agent-list" class="create-agent-list"></ul>
            </div>
            <div class="create-config">
              <div class="cfg-row">
                <span>Shape</span>
                <div class="seg-group" id="cfg-shape">
                  <button class="config-pill on" data-v="round-robin">round-robin</button>
                  <button class="config-pill" data-v="gauntlet">gauntlet</button>
                </div>
              </div>
              <div class="cfg-row" id="cfg-challenger-wrap" hidden>
                <span>Challenger</span>
                <select id="cfg-challenger" style="min-width: 200px;"></select>
              </div>
              <div class="cfg-row">
                <span>Games per pair</span>
                <div class="seg-group" id="cfg-games">
                  <button class="config-pill" data-v="1">1</button>
                  <button class="config-pill on" data-v="3">3</button>
                  <button class="config-pill" data-v="5">5</button>
                  <button class="config-pill" data-v="10">10</button>
                  <button class="config-pill" data-v="20">20</button>
                  <button class="config-pill" data-v="50">50</button>
                </div>
              </div>
              <div class="cfg-row">
                <span>Mode</span>
                <div class="seg-group" id="cfg-mode">
                  <button class="config-pill" data-v="ultrafast" title="Native Rust engine, no replays (tournament throughput)">ultrafast</button>
                  <button class="config-pill on" data-v="fast" title="In-process kaggle-environments">fast</button>
                  <button class="config-pill" data-v="faithful" title="Subprocess + HTTP (Kaggle protocol)">faithful</button>
                </div>
              </div>
              <div class="cfg-row">
                <span>Format</span>
                <div class="seg-group" id="cfg-format">
                  <button class="config-pill on" data-v="2p">2p</button>
                  <button class="config-pill" data-v="4p">4p</button>
                </div>
              </div>
              <div class="cfg-row">
                <span>Seed</span>
                <div class="create-seed-controls">
                  <div class="seg-group" id="cfg-seed-mode">
                    <button class="config-pill on" data-v="random">random</button>
                    <button class="config-pill" data-v="custom">custom</button>
                  </div>
                  <input id="cfg-seed" class="seed-input" type="number" value="42" inputmode="numeric" disabled>
                </div>
              </div>
              <label class="cfg-row create-config-checkbox" title="Skip writing per-match replay JSON files (5-10MB each). Ratings are still computed.">
                <input id="cfg-save-replays" type="checkbox" checked>
                <span>Save replays</span>
              </label>
              <div id="cfg-total-matches" class="cfg-total-matches"></div>
              <div class="create-actions">
                <div id="create-status" class="scrape-status" hidden></div>
                <button class="scrape-btn go" id="create-start">Start tournament</button>
              </div>
            </div>
          </div>
        </div>
      </section>
      <section>
        <div class="section-head">
          <h2>Active now</h2>
          <span id="sched-concurrency" class="sched-concurrency"></span>
        </div>
        <div id="scheduler-panel"></div>
      </section>
      <section>
        <h2>Recent tournaments</h2>
        <div id="runs-list"></div>
      </section>
    </main>
  `;
  installHeaderNav(root, "tournaments");

  // =========================================================
  // Agent selection state
  // =========================================================
  const selected = new Set<string>();
  let agents: AgentInfo[] = [];
  let ratingsByAgent = new Map<string, number>();
  let bucketFilter: "all" | "baselines" | "external" | "mine" = "all";
  let searchTerm = "";

  function currentFormat(): "2p" | "4p" {
    return getFormat() as "2p" | "4p";
  }

  async function loadRatings(format: "2p" | "4p" = currentFormat()) {
    const ratings = await api.getRatings(format);
    ratingsByAgent = new Map(ratings.map((r: Rating) => [r.agent_id, r.mu]));
  }

  async function loadAgents() {
    const [freshAgents] = await Promise.all([
      api.listAgents(),
      loadRatings(),
    ]);
    agents = freshAgents;
    renderAgentList();
  }

  function renderAgentList() {
    const listEl = document.getElementById("create-agent-list")!;
    const filtered = agents.filter((a) => {
      if (a.disabled) return false;
      if (bucketFilter !== "all" && a.bucket !== bucketFilter) return false;
      if (searchTerm) {
        const t = searchTerm.toLowerCase();
        if (!a.id.toLowerCase().includes(t) && !a.name.toLowerCase().includes(t))
          return false;
      }
      return true;
    }).sort((a, b) => {
      const aMu = ratingsByAgent.get(a.id);
      const bMu = ratingsByAgent.get(b.id);
      const aKnown = aMu !== undefined;
      const bKnown = bMu !== undefined;
      if (aKnown !== bKnown) return aKnown ? 1 : -1;
      if (!aKnown && !bKnown) return a.name.localeCompare(b.name);
      if (aMu !== bMu) return (bMu ?? 0) - (aMu ?? 0);
      return a.name.localeCompare(b.name);
    });
    if (filtered.length === 0) {
      listEl.innerHTML = `<li class="picker-empty">No agents match this filter.</li>`;
    } else {
      listEl.innerHTML = filtered
        .map((a) => {
          const mu = ratingsByAgent.get(a.id);
          const muStr = mu !== undefined ? `${mu.toFixed(0)} elo` : "unknown elo";
          return `
          <li class="create-agent ${selected.has(a.id) ? "picked" : ""}" data-id="${a.id}">
            <span class="create-check">${selected.has(a.id) ? "✓" : ""}</span>
            <span class="agent-name-wrap">
              <span class="agent-name">${a.name}</span>
              <span class="agent-bucket">(${a.bucket})</span>
            </span>
            <span class="agent-elo">${muStr}</span>
          </li>
        `;
        })
        .join("");
    }
    listEl.querySelectorAll<HTMLElement>(".create-agent").forEach((el) => {
      el.addEventListener("click", () => {
        const id = el.dataset.id!;
        if (selected.has(id)) selected.delete(id);
        else selected.add(id);
        updateCount();
        renderAgentList();
      });
    });
  }

  function updateCount() {
    document.getElementById("create-count-num")!.textContent = String(selected.size);
    refreshChallengerDropdown();
    updateTotalMatches();
    refreshStartButton();
  }

  function refreshStartButton() {
    const btn = document.getElementById("create-start") as HTMLButtonElement | null;
    if (!btn) return;
    btn.disabled = selected.size < 2;
  }

  document.getElementById("create-search")!.addEventListener("input", (e) => {
    searchTerm = (e.target as HTMLInputElement).value;
    renderAgentList();
  });

  root.querySelectorAll<HTMLButtonElement>("[data-bucket]").forEach((btn) => {
    btn.addEventListener("click", () => {
      bucketFilter = btn.dataset.bucket as typeof bucketFilter;
      root.querySelectorAll<HTMLButtonElement>("[data-bucket]").forEach((b) =>
        b.classList.toggle("on", b === btn),
      );
      renderAgentList();
    });
  });

  // Config seg-groups
  function wireSegGroup(groupId: string): () => string {
    const group = document.getElementById(groupId)!;
    group.querySelectorAll<HTMLButtonElement>(".config-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        group.querySelectorAll<HTMLButtonElement>(".config-pill").forEach((b) =>
          b.classList.toggle("on", b === btn),
        );
      });
    });
    return () =>
      group.querySelector<HTMLButtonElement>(".config-pill.on")?.dataset.v ?? "";
  }
  const getMode = wireSegGroup("cfg-mode");
  function refreshSaveReplays() {
    const saveReplaysEl = document.getElementById("cfg-save-replays") as HTMLInputElement;
    const wrap = saveReplaysEl.closest(".create-config-checkbox") as HTMLElement | null;
    const isUltrafast = getMode() === "ultrafast";
    saveReplaysEl.disabled = isUltrafast;
    if (isUltrafast) {
      saveReplaysEl.checked = false;
    }
    if (wrap) wrap.style.opacity = isUltrafast ? "0.55" : "";
  }
  document.getElementById("cfg-mode")!
    .querySelectorAll<HTMLButtonElement>(".config-pill")
    .forEach((btn) => btn.addEventListener("click", () => setTimeout(refreshSaveReplays, 0)));
  const getFormat = wireSegGroup("cfg-format");
  const getShape = wireSegGroup("cfg-shape");
  const getGamesValue = wireSegGroup("cfg-games");
  const getSeedMode = wireSegGroup("cfg-seed-mode");

  const challengerWrap = document.getElementById("cfg-challenger-wrap")!;
  const challengerSel = document.getElementById("cfg-challenger") as HTMLSelectElement;
  const totalMatchesEl = document.getElementById("cfg-total-matches")!;
  const seedInput = document.getElementById("cfg-seed") as HTMLInputElement;

  function getGames(): number {
    return parseInt(getGamesValue(), 10);
  }

  function randomSeedBase(): number {
    return crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
  }

  function refreshSeedInput() {
    seedInput.disabled = getSeedMode() === "random";
  }

  function refreshChallengerDropdown() {
    const picked = Array.from(selected);
    const prev = challengerSel.value;
    if (picked.length === 0) {
      challengerSel.innerHTML = `<option value="">(pick agents first)</option>`;
      return;
    }
    const options = picked.map((id) => {
      const a = agents.find((x) => x.id === id);
      const label = a ? `${a.name} (${id})` : id;
      return `<option value="${id}">${label}</option>`;
    }).join("");
    challengerSel.innerHTML = options;
    if (picked.includes(prev)) challengerSel.value = prev;
  }

  function updateTotalMatches() {
    const shape = getShape();
    const format = getFormat();
    const n = selected.size;
    const K = getGames();
    let pairs = 0;
    let note = "";
    if (shape === "gauntlet") {
      const opponents = Math.max(0, n - 1); // minus challenger
      if (format === "2p") {
        pairs = opponents;
      } else {
        // C(opponents, 3): challenger + 3 opponents per match
        if (opponents >= 3) {
          pairs = (opponents * (opponents - 1) * (opponents - 2)) / 6;
        }
      }
      if (n < 2) note = "select ≥2 agents + choose challenger";
    } else {
      if (format === "2p") {
        pairs = n < 2 ? 0 : (n * (n - 1)) / 2;
      } else {
        pairs = n < 4 ? 0 : (n * (n - 1) * (n - 2) * (n - 3)) / 24;
        if (n < 4) note = "4p needs ≥4 agents";
      }
    }
    const total = pairs * K;
    if (note) {
      totalMatchesEl.textContent = note;
    } else {
      totalMatchesEl.textContent = `${pairs} pair${pairs === 1 ? "" : "s"} × ${K} = ${total} games`;
    }
  }

  function onShapeChange() {
    const shape = getShape();
    challengerWrap.hidden = shape !== "gauntlet";
    if (shape === "gauntlet") refreshChallengerDropdown();
    updateTotalMatches();
  }

  // Wire shape pill clicks to show/hide challenger + recompute totals.
  document.getElementById("cfg-shape")!
    .querySelectorAll<HTMLButtonElement>(".config-pill")
    .forEach((btn) => btn.addEventListener("click", () => {
      // wireSegGroup already handled the .on toggle; we just react after.
      setTimeout(onShapeChange, 0);
    }));
  document.getElementById("cfg-format")!
    .querySelectorAll<HTMLButtonElement>(".config-pill")
    .forEach((btn) => btn.addEventListener("click", () => {
      setTimeout(updateTotalMatches, 0);
      void loadRatings(currentFormat()).then(() => renderAgentList()).catch(() => {
        ratingsByAgent.clear();
        renderAgentList();
      });
    }));
  document.getElementById("cfg-games")!
    .querySelectorAll<HTMLButtonElement>(".config-pill")
    .forEach((btn) => btn.addEventListener("click", () => setTimeout(updateTotalMatches, 0)));
  document.getElementById("cfg-seed-mode")!
    .querySelectorAll<HTMLButtonElement>(".config-pill")
    .forEach((btn) => btn.addEventListener("click", () => setTimeout(refreshSeedInput, 0)));

  // Start tournament
  document.getElementById("create-start")!.addEventListener("click", async () => {
    const statusEl = document.getElementById("create-status")!;
    statusEl.hidden = true; // clear any prior message; success shows nothing (no layout shift)
    const shape = getShape();
    if (selected.size < 2) {
      statusEl.hidden = false;
      statusEl.textContent = "Select at least 2 agents.";
      return;
    }
    let challengerId: string | null = null;
    if (shape === "gauntlet") {
      challengerId = challengerSel.value || null;
      if (!challengerId || !selected.has(challengerId)) {
        statusEl.hidden = false;
        statusEl.textContent = "Gauntlet: pick a challenger from the dropdown.";
        return;
      }
    }
    const games = getGames();
    const mode = getMode();
    const format = getFormat();
    const useRandomSeed = getSeedMode() === "random";
    const seed = parseInt(seedInput.value, 10);
    const saveReplays = (document.getElementById("cfg-save-replays") as HTMLInputElement).checked;
    const startBtn = document.getElementById("create-start") as HTMLButtonElement;
    // Feedback via the button label (no layout shift). Interactions unchanged.
    startBtn.textContent = "Starting…";
    try {
      await api.startTournament({
        agents: Array.from(selected),
        games_per_pair: games,
        mode,
        format,
        save_replays: saveReplays,
        seed_base: useRandomSeed ? randomSeedBase() : (isNaN(seed) ? 42 : seed),
        seed_mode: useRandomSeed ? "random" : "fixed",
        is_quick_match: false,
        shape: shape as "round-robin" | "gauntlet",
        challenger_id: challengerId,
      });
      // Success: no status message — the queued tournament shows up in
      // "Active now" immediately, which is the feedback.
      await Promise.all([loadRuns(), loadScheduler()]);
    } catch (e: any) {
      statusEl.hidden = false;
      statusEl.textContent = `Error: ${e?.message || "unknown error"}`;
    } finally {
      startBtn.textContent = "Start tournament";
    }
  });

  // =========================================================
  // Runs list
  // =========================================================
  const listEl = document.getElementById("runs-list")!;

  function formatRunId(id: string): string {
    const m = id.match(/^\d{4}-(\d{2}-\d{2})-(.+)$/);
    return m ? `${m[1]} (${m[2]})` : id;
  }

  function trimmedAgentName(agentId: string | null | undefined): string {
    if (!agentId) return "";
    const parts = agentId.split("/").filter(Boolean);
    return parts[parts.length - 1] || agentId;
  }

  function tournamentName(r: RunSummary): string {
    if (r.shape === "gauntlet") {
      const challenger = trimmedAgentName(r.challenger_id);
      return challenger ? `gauntlet - ${challenger}` : "gauntlet";
    }
    return "round robin";
  }

  async function loadRuns() {
    // Queued/running tournaments live in the "Active now" panel until they
    // finish; Recent shows only finished (completed/aborted) runs. This also
    // avoids the same run appearing in both lists with mismatched status
    // (disk run.json lags the scheduler's live in-memory state).
    const runs = (await api.listRuns({ excludeQuickMatch: true })).filter(
      (r) => r.status !== "running" && r.status !== "queued",
    );
    if (runs.length === 0) {
      listEl.innerHTML = `<div class="loading">No finished tournaments yet.</div>`;
      return;
    }
    listEl.innerHTML = `
      <ul class="runs">
        ${runs
          .map(
            (r: RunSummary) => `
          <li data-run-id="${escapeHtml(r.id)}">
            <span class="run-name">${escapeHtml(tournamentName(r))}</span>
            <span class="run-meta">${escapeHtml(r.mode)} &middot; ${escapeHtml(r.format)} &middot; ${r.matches_done}/${r.total_matches}</span>
            <span class="run-id">${escapeHtml(formatRunId(r.id))}</span>
            <span class="run-status status-${r.status}">${escapeHtml(r.status)}</span>
            <button class="replay-delete" data-run-id="${escapeHtml(r.id)}" title="Delete tournament">&times;</button>
          </li>
        `,
          )
          .join("")}
      </ul>
    `;
    listEl.querySelectorAll<HTMLLIElement>("li").forEach((li) => {
      li.addEventListener("click", (ev) => {
        if ((ev.target as HTMLElement).closest(".replay-delete")) return;
        const runId = li.getAttribute("data-run-id");
        if (!runId) return;
        navigate({ view: "tournament-detail", runId });
      });
    });
    listEl.querySelectorAll<HTMLButtonElement>(".replay-delete").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const runId = btn.dataset.runId!;
        if (!confirm(`Delete tournament ${runId} and all its replays?`)) return;
        try {
          await api.deleteRun(runId);
          await loadRuns();
        } catch (e) {
          alert(`Delete failed: ${(e as Error).message}`);
        }
      });
    });
  }

  // =========================================================
  // Scheduler: live "Active now" panel (queue + running matches)
  // =========================================================
  const schedPanel = document.getElementById("scheduler-panel")!;
  const concEl = document.getElementById("sched-concurrency")!;
  // Tournaments the user clicked Stop on — kept visible as "stopping" until the
  // backend finishes tearing them down (worker-kill latency), so the row never
  // appears to linger as "running". Survives re-renders (closure-scoped).
  const stoppingIds = new Set<string>();

  async function loadScheduler() {
    let s: SchedulerStatus;
    try {
      s = await api.getScheduler();
    } catch {
      return; // transient — keep last render
    }
    concEl.textContent = `${s.concurrency} worker${s.concurrency === 1 ? "" : "s"} · ${s.running_count} running · ${s.queued_total} queued`;
    // Drop stopping-markers for tournaments the backend has fully torn down
    // (no longer reported by the scheduler) — they now show in Recent.
    for (const id of [...stoppingIds]) {
      if (!s.tournaments.some((t) => t.id === id)) stoppingIds.delete(id);
    }
    const active = s.tournaments.filter(
      (t) => t.status === "running" || t.status === "queued" || stoppingIds.has(t.id),
    );
    if (active.length === 0) {
      schedPanel.innerHTML = `<div class="loading">Idle — no tournaments queued or running.</div>`;
      return;
    }
    schedPanel.innerHTML = `
      <ul class="runs">
        ${active
          .map((t) => {
            const running = s.running.filter((m) => m.run_id === t.id);
            const runningStr = running
              .map(
                (m) =>
                  `<span class="sched-match" title="${escapeHtml(m.agent_ids.join(" vs "))}">${escapeHtml(m.match_id)} (${m.elapsed_s.toFixed(0)}s)</span>`,
              )
              .join(" ");
            let name = "round robin";
            if (t.shape === "gauntlet") {
              const c = trimmedAgentName(t.challenger_id);
              name = c ? `gauntlet - ${c}` : "gauntlet";
            }
            const stopping = stoppingIds.has(t.id);
            const status = stopping ? "stopping" : t.status;
            return `
          <li data-run-id="${escapeHtml(t.id)}">
            <span class="run-name">${escapeHtml(name)}</span>
            <span class="run-meta">${escapeHtml(t.mode)} &middot; ${escapeHtml(t.format)} &middot; ${t.matches_done}/${t.total_matches} · ${t.queued} queued ${runningStr}</span>
            <span class="run-id">${escapeHtml(formatRunId(t.id))}</span>
            <span class="run-status status-${status}">${escapeHtml(status)}</span>
            <button class="replay-delete" ${stopping ? "disabled" : ""} data-run-id="${escapeHtml(t.id)}" title="Stop tournament">&times;</button>
          </li>`;
          })
          .join("")}
      </ul>`;
    schedPanel.querySelectorAll<HTMLLIElement>("li").forEach((li) => {
      li.addEventListener("click", (ev) => {
        if ((ev.target as HTMLElement).closest(".replay-delete")) return;
        const runId = li.getAttribute("data-run-id");
        if (runId) navigate({ view: "tournament-detail", runId });
      });
    });
    schedPanel.querySelectorAll<HTMLButtonElement>(".replay-delete").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const runId = btn.dataset.runId!;
        // Optimistically mark stopping so the row flips immediately and stays
        // put (as "stopping") until the backend tears it down.
        stoppingIds.add(runId);
        btn.disabled = true;
        const statusCell = btn.closest("li")?.querySelector(".run-status");
        if (statusCell) {
          statusCell.textContent = "stopping";
          statusCell.className = "run-status status-stopping";
        }
        try {
          await api.stopTournament(runId);
        } catch (e) {
          stoppingIds.delete(runId);
          alert(`Stop failed: ${(e as Error).message}`);
        }
        await Promise.all([loadScheduler(), loadRuns()]);
      });
    });
  }

  onShapeChange(); // initial: hide challenger + compute totals
  refreshSeedInput();
  refreshSaveReplays();
  refreshStartButton();

  await loadAgents();
  await Promise.all([loadRuns(), loadScheduler()]);

  if (pollInterval !== null) window.clearInterval(pollInterval);
  pollInterval = window.setInterval(() => {
    if (document.hidden) return;
    void loadRuns();
    void loadScheduler();
  }, 3000);
}
