import { api, AgentInfo, AgentRuntime, Rating } from "../api";
import { navigate } from "../router";

function renderAgentCell(agentId: string, agentsById: Map<string, AgentInfo>): string {
  const agent = agentsById.get(agentId);
  if (!agent) return `<span class="agent-name">${agentId}</span>`;
  return `<span class="agent-name">${agent.name}</span> <span class="agent-bucket">(${agent.bucket})</span>`;
}

async function getAgentsById(): Promise<Map<string, AgentInfo>> {
  const agents = await api.listAgents().catch(() => [] as AgentInfo[]);
  return new Map(agents.map((a) => [a.id, a]));
}

export type RatingsFormat = "2p" | "4p" | "all";

function fmtMs(avgMs: number | undefined): string {
  return avgMs !== undefined && Number.isFinite(avgMs) && avgMs > 0
    ? avgMs.toFixed(1)
    : "&mdash;";
}

async function getRuntimeByAgent(): Promise<Map<string, AgentRuntime>> {
  const runtimes = await api.listRuntimes().catch(() => [] as AgentRuntime[]);
  return new Map(runtimes.map((r) => [r.agent_id, r]));
}

export async function mountRatingsTable(
  el: HTMLElement,
  format: RatingsFormat = "2p",
): Promise<void> {
  const requestId = String((Number(el.dataset.requestId ?? "0") || 0) + 1);
  el.dataset.requestId = requestId;
  if (format === "all") {
    await renderCombined(el, requestId);
    return;
  }
  await renderSingle(el, format, requestId);
}

async function renderSingle(
  el: HTMLElement,
  format: "2p" | "4p",
  requestId: string,
): Promise<void> {
  const [ratings, runtimes, agentsById] = await Promise.all([
    api.getRatings(format),
    getRuntimeByAgent(),
    getAgentsById(),
  ]);
  if (el.dataset.requestId !== requestId) return;
  el.innerHTML = `
    <table class="ratings">
      <thead>
        <tr>
          <th>#</th><th>Agent</th><th>Elo</th><th>&sigma;</th><th>N</th><th>ms/step</th>
        </tr>
      </thead>
      <tbody>
        ${ratings
          .map(
            (r: Rating) => `
          <tr data-agent-id="${r.agent_id}">
            <td>${r.rank}</td>
            <td class="agent-id">${renderAgentCell(r.agent_id, agentsById)}</td>
            <td class="rating-elo">${r.mu.toFixed(0)}</td>
            <td>${r.sigma.toFixed(0)}</td>
            <td>${r.games_played}</td>
            <td>${fmtMs(runtimes.get(r.agent_id)?.avg_ms)}</td>
          </tr>
        `,
          )
          .join("")}
      </tbody>
    </table>
  `;
  wireRowClicks(el);
}

async function renderCombined(el: HTMLElement, requestId: string): Promise<void> {
  const [r2p, r4p, runtimes, agentsById] = await Promise.all([
    api.getRatings("2p"),
    api.getRatings("4p"),
    getRuntimeByAgent(),
    getAgentsById(),
  ]);
  if (el.dataset.requestId !== requestId) return;

  interface Row {
    agentId: string;
    mu2p?: number;
    n2p?: number;
    rank2p?: number;
    mu4p?: number;
    n4p?: number;
    rank4p?: number;
    avgMs?: number;
    avgRank: number;
  }

  const map = new Map<string, Row>();
  const ensure = (aid: string): Row => {
    let r = map.get(aid);
    if (!r) {
      r = { agentId: aid, avgRank: 0 };
      map.set(aid, r);
    }
    return r;
  };

  r2p.forEach((r) => {
    const row = ensure(r.agent_id);
    row.mu2p = r.mu;
    row.n2p = r.games_played;
    row.rank2p = r.rank;
  });
  r4p.forEach((r) => {
    const row = ensure(r.agent_id);
    row.mu4p = r.mu;
    row.n4p = r.games_played;
    row.rank4p = r.rank;
  });

  const rows = Array.from(map.values()).map((r) => {
    const ranks = [r.rank2p, r.rank4p].filter(
      (x): x is number => x !== undefined,
    );
    r.avgRank = ranks.length > 0 ? ranks.reduce((s, v) => s + v, 0) / ranks.length : Infinity;
    r.avgMs = runtimes.get(r.agentId)?.avg_ms;
    return r;
  });
  rows.sort((a, b) => a.avgRank - b.avgRank);

  const fmt = (v: number | undefined) => (v === undefined ? "&mdash;" : v.toFixed(0));
  const fmtAvg = (v: number) =>
    Number.isFinite(v) ? (Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1)) : "&mdash;";

  el.innerHTML = `
    <table class="ratings ratings-combined">
      <thead>
        <tr>
          <th rowspan="2">#</th>
          <th rowspan="2">Agent</th>
          <th colspan="2" class="group-2p">2p</th>
          <th colspan="2" class="group-4p">4p</th>
          <th rowspan="2">ms</th>
          <th rowspan="2" title="Average rank across both formats (lower is better)">avg rank</th>
        </tr>
        <tr>
          <th class="group-2p">Elo</th>
          <th class="group-2p">N</th>
          <th class="group-4p">Elo</th>
          <th class="group-4p">N</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (r, i) => `
          <tr data-agent-id="${r.agentId}">
            <td>${i + 1}</td>
            <td class="agent-id">${renderAgentCell(r.agentId, agentsById)}</td>
            <td class="rating-elo">${fmt(r.mu2p)}</td>
            <td>${r.n2p ?? "&mdash;"}</td>
            <td class="rating-elo">${fmt(r.mu4p)}</td>
            <td>${r.n4p ?? "&mdash;"}</td>
            <td>${fmtMs(r.avgMs)}</td>
            <td>${fmtAvg(r.avgRank)}</td>
          </tr>
        `,
          )
          .join("")}
      </tbody>
    </table>
  `;
  wireRowClicks(el);
}

function wireRowClicks(el: HTMLElement): void {
  el.querySelectorAll<HTMLTableRowElement>("tbody tr").forEach((row) => {
    row.addEventListener("click", () => {
      const id = row.getAttribute("data-agent-id");
      if (id) navigate({ view: "agent", agentId: id });
    });
  });
}
