/**
 * Submissions view - table of my Kaggle submissions.
 * Click a row -> /replays?sub=<id> (filter existing view).
 */

import { api, AgentInfo, KaggleSubmission } from "../api";
import { installHeaderNav } from "../components/header-nav";
import { navigate } from "../router";
import { escapeHtml } from "../utils/escape";

let pollInterval: number | null = null;

export async function renderSubmissions(root: HTMLElement): Promise<void> {
  root.innerHTML = `
    <main class="dashboard submissions-view">
      <section>
        <div class="section-head">
          <h2>My submissions</h2>
          <span id="sub-updated" class="sub-updated" title="Last list refresh"></span>
          <button
            id="sub-fetch-all"
            class="source-pill"
            style="margin-left:auto"
            title="Fetch new episodes for every submission (sequential, Kaggle rate-limited)"
          >&#x27F3; Fetch all new episodes</button>
          <button
            id="sub-refresh"
            class="source-pill"
            title="Reload status list (mu, sigma, status)"
          >&#x27F3; Reload list</button>
        </div>
        <div id="sub-fetch-all-progress" class="sub-fetch-all-progress" hidden></div>
        <details id="sub-submit-panel" class="sub-submit-panel">
          <summary>+ Submit new agent</summary>
          <div class="sub-submit-form">
            <label>
              Agent
              <select id="sub-submit-agent"></select>
            </label>
            <label>
              Description
              <input type="text" id="sub-submit-desc" placeholder="e.g. v4 sun-dodge + predicted combat">
            </label>
            <div class="sub-submit-row">
              <button id="sub-submit-go" class="sub-fetch-btn sub-submit-go">Submit</button>
              <span id="sub-submit-status" class="sub-submit-status" hidden></span>
            </div>
          </div>
        </details>
        <div id="sub-banner" class="sub-banner" hidden></div>
        <div id="sub-table-wrap"></div>
      </section>
    </main>
  `;
  installHeaderNav(root, "submissions");

  await wireSubmitPanel();

  let lastLoadedAt = 0;
  let hasLoadedTable = false;
  let tableRequestId = 0;

  function stampUpdated(): void {
    const el = document.getElementById("sub-updated");
    if (!el || !lastLoadedAt) return;
    const s = Math.floor((Date.now() - lastLoadedAt) / 1000);
    el.textContent =
      s < 5 ? "updated just now"
      : s < 60 ? `updated ${s}s ago`
      : s < 3600 ? `updated ${Math.floor(s / 60)} min ago`
      : `updated ${Math.floor(s / 3600)}h ago`;
  }

  async function loadTable(options?: { showLoading?: boolean }): Promise<void> {
    const wrap = document.getElementById("sub-table-wrap")!;
    const banner = document.getElementById("sub-banner")!;
    const requestId = ++tableRequestId;
    const showLoading = options?.showLoading ?? !hasLoadedTable;

    banner.hidden = true;
    if (showLoading) {
      wrap.innerHTML = `<div class="loading">Loading...</div>`;
    }

    try {
      const items = await api.listKaggleSubmissions();
      if (requestId !== tableRequestId) return;

      lastLoadedAt = Date.now();
      stampUpdated();

      if (items.length === 0) {
        wrap.innerHTML = `<div class="loading">No submissions yet. Use <strong>+ Submit new agent</strong> above to upload one.</div>`;
        hasLoadedTable = true;
        return;
      }

      wrap.innerHTML = renderTable(items);
      wire();
      hasLoadedTable = true;
    } catch (e) {
      if (requestId !== tableRequestId) return;

      const err = e as Error & { status?: number };
      banner.hidden = false;
      if (err.status === 401) {
        banner.innerHTML = `Connect your Kaggle account in <a href="#/settings">Settings -></a> to see your submissions.`;
      } else if (err.status === 500) {
        banner.innerHTML = `Kaggle CLI unavailable: ${err.message}`;
      } else {
        banner.innerHTML = `Error: ${err.message}`;
      }

      if (!hasLoadedTable) {
        wrap.innerHTML = "";
      }
    }
  }

  function formatRelative(iso: string): string {
    if (!iso) return "";
    const ms = Date.parse(iso);
    if (isNaN(ms)) return "";
    const diff = Math.max(0, Date.now() - ms);
    const s = Math.floor(diff / 1000);
    if (s < 60) return `${s}s temu`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m} min temu`;
    const h = Math.floor(m / 60);
    if (h < 48) return `${h}h temu`;
    const d = Math.floor(h / 24);
    if (d < 14) return `${d}d temu`;
    const w = Math.floor(d / 7);
    if (w < 8) return `${w} tyg. temu`;
    const mo = Math.floor(d / 30);
    return `${mo} mies. temu`;
  }

  function renderTable(items: KaggleSubmission[]): string {
    const rows = items
      .map((s) => {
        const dateTitle = s.date ? escapeHtml(s.date.replace("T", " ").slice(0, 19) + " UTC") : "";
        const relDate = s.date ? formatRelative(s.date) : "-";
        const statusClass = s.status === "COMPLETE" ? "ok" : s.status === "FAILED" ? "err" : "pending";
        return `
        <tr data-sub-id="${s.submission_id}" data-status="${escapeHtml(s.status)}">
          <td class="sub-cell-id">${s.submission_id}</td>
          <td class="sub-cell-desc" title="${escapeHtml(s.description)}">${escapeHtml(s.description) || "-"}</td>
          <td class="sub-cell-mu">${s.mu != null ? s.mu.toFixed(1) : "-"}</td>
          <td class="sub-cell-date" title="${dateTitle}">${relDate}</td>
          <td class="sub-cell-status"><span class="sub-status-pill ${statusClass}">${escapeHtml(s.status)}</span></td>
          <td class="sub-cell-fetch">
            <button class="sub-fetch-btn" data-sub-id="${s.submission_id}" title="Fetch all missing episodes">&#x27F3;</button>
            <span class="sub-fetch-status" hidden></span>
          </td>
          <td class="sub-cell-go">&rsaquo;</td>
        </tr>`;
      })
      .join("");
    return `
      <table class="submissions-table">
        <thead>
          <tr>
            <th class="th-id">ID</th>
            <th class="th-desc">Description</th>
            <th class="th-mu">mu</th>
            <th class="th-date">Date</th>
            <th class="th-status">Status</th>
            <th class="th-fetch"></th>
            <th class="th-go"></th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function wire(): void {
    document.querySelectorAll<HTMLTableRowElement>("tr[data-sub-id]").forEach((tr) => {
      tr.addEventListener("click", (ev) => {
        if ((ev.target as HTMLElement).closest(".sub-cell-fetch")) return;
        const subId = tr.dataset.subId!;
        navigate({ view: "replays", subFilter: subId });
      });
    });

    document.querySelectorAll<HTMLButtonElement>(".sub-fetch-btn").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const subId = parseInt(btn.dataset.subId!, 10);
        const row = btn.closest("tr")!;
        const statusEl = row.querySelector<HTMLSpanElement>(".sub-fetch-status")!;
        btn.disabled = true;
        statusEl.hidden = false;
        statusEl.textContent = "starting...";
        try {
          const { job_id } = await api.startScrape(subId, 1000);
          while (true) {
            await new Promise((r) => setTimeout(r, 1500));
            const s = await api.getScrapeStatus(job_id);
            if (s.status === "completed") {
              statusEl.textContent = s.total === 0
                ? "up to date"
                : `OK ${s.downloaded} new`;
              break;
            }
            if (s.status === "failed") {
              statusEl.textContent = `Error: ${s.error ?? "failed"}`;
              break;
            }
            if (s.total > 0) {
              statusEl.textContent = `${s.downloaded}/${s.total}...`;
            } else {
              statusEl.textContent = s.status === "pending" ? "queued..." : "listing...";
            }
          }
        } catch (e) {
          statusEl.textContent = `Error: ${(e as Error).message}`;
        } finally {
          btn.disabled = false;
        }
      });
    });
  }

  async function wireSubmitPanel(): Promise<void> {
    const select = document.getElementById("sub-submit-agent") as HTMLSelectElement;
    const descInput = document.getElementById("sub-submit-desc") as HTMLInputElement;
    const goBtn = document.getElementById("sub-submit-go") as HTMLButtonElement;
    const statusEl = document.getElementById("sub-submit-status") as HTMLSpanElement;

    try {
      const agents = await api.listAgents();
      const bucketOrder = { mine: 0, baselines: 1, external: 2 } as const;
      const submittable = agents
        .slice()
        .sort((a: AgentInfo, b: AgentInfo) => {
          const ra = bucketOrder[a.bucket] ?? 99;
          const rb = bucketOrder[b.bucket] ?? 99;
          if (ra !== rb) return ra - rb;
          return a.id.localeCompare(b.id);
        });
      if (submittable.length === 0) {
        select.innerHTML = `<option value="">(no agents found)</option>`;
        goBtn.disabled = true;
      } else {
        select.innerHTML = submittable
          .map((a: AgentInfo) => `<option value="${escapeHtml(a.id)}">${escapeHtml(a.id)}</option>`)
          .join("");
      }
    } catch {
      select.innerHTML = `<option value="">(error loading agents)</option>`;
      goBtn.disabled = true;
    }

    goBtn.addEventListener("click", async () => {
      const agentId = select.value;
      const description = descInput.value.trim();
      if (!agentId) return;
      if (!description) {
        statusEl.hidden = false;
        statusEl.textContent = "Description required";
        return;
      }
      goBtn.disabled = true;
      select.disabled = true;
      descInput.disabled = true;
      statusEl.hidden = false;
      statusEl.textContent = "uploading...";
      try {
        const res = await api.submitKaggleAgent(agentId, description);
        statusEl.textContent = `OK ${res.message}`;
        descInput.value = "";
        setTimeout(() => void loadTable(), 4000);
      } catch (e) {
        const err = e as Error & { status?: number };
        statusEl.textContent = `Error: ${err.message}`;
      } finally {
        goBtn.disabled = false;
        select.disabled = false;
        descInput.disabled = false;
      }
    });
  }

  document.getElementById("sub-refresh")!.addEventListener("click", () => void loadTable());

  document.getElementById("sub-fetch-all")!.addEventListener("click", async () => {
    const subs = await api.listKaggleSubmissions().catch(() => [] as KaggleSubmission[]);
    if (subs.length === 0) return;

    const progressEl = document.getElementById("sub-fetch-all-progress")!;
    const allBtn = document.getElementById("sub-fetch-all") as HTMLButtonElement;
    const refreshBtn = document.getElementById("sub-refresh") as HTMLButtonElement;
    allBtn.disabled = true;
    refreshBtn.disabled = true;
    progressEl.hidden = false;

    let totalNew = 0;
    let totalErrors = 0;
    try {
      for (let idx = 0; idx < subs.length; idx++) {
        const s = subs[idx];
        progressEl.textContent = `Fetching ${idx + 1}/${subs.length} (sub ${s.submission_id}) - ${totalNew} new so far`;
        try {
          const { job_id } = await api.startScrape(s.submission_id, 1000);
          while (true) {
            if (!document.querySelector(".submissions-view")) return;
            await new Promise((r) => setTimeout(r, 1500));
            const st = await api.getScrapeStatus(job_id);
            if (st.status === "completed") {
              totalNew += st.downloaded || 0;
              break;
            }
            if (st.status === "failed") {
              totalErrors += 1;
              break;
            }
            const phase = st.total > 0 ? `${st.downloaded}/${st.total}` : (st.status === "pending" ? "queued" : "listing");
            progressEl.textContent =
              `Fetching ${idx + 1}/${subs.length} (sub ${s.submission_id}, ${phase}) - ${totalNew} new so far`;
          }
        } catch {
          totalErrors += 1;
        }
      }

      progressEl.textContent =
        `Done - ${totalNew} new episodes` + (totalErrors > 0 ? ` | ${totalErrors} errored` : "");
      await loadTable();
    } finally {
      allBtn.disabled = false;
      refreshBtn.disabled = false;
      window.setTimeout(() => { progressEl.hidden = true; }, 10000);
    }
  });

  await loadTable({ showLoading: true });

  if (pollInterval !== null) window.clearInterval(pollInterval);
  let tick = 0;
  pollInterval = window.setInterval(() => {
    if (!document.querySelector(".submissions-view")) {
      if (pollInterval !== null) window.clearInterval(pollInterval);
      pollInterval = null;
      return;
    }
    stampUpdated();
    if (document.hidden) return;
    tick += 5;
    const hasPending = !!document.querySelector('tr[data-status]:not([data-status="COMPLETE"]):not([data-status="FAILED"])');
    const threshold = hasPending ? 10 : 60;
    if (tick >= threshold) {
      tick = 0;
      void loadTable();
    }
  }, 5000);
}
