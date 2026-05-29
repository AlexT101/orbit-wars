"""Pure tournament-expansion helpers: agent resolution + pair generation.

Extracted from `tournament.Tournament` so both the scheduler (which turns a
`TournamentConfig` into a flat list of matches to enqueue) and the CLI agent
filters can share one implementation with no import cycle. Nothing here does
match execution or disk IO beyond `scan_zoo`.
"""
from __future__ import annotations

import itertools
from pathlib import Path

from .discovery import scan_zoo
from .schemas import AgentInfo, TournamentConfig


def filter_agents_by_tags(
    agents: list[AgentInfo],
    include: list[str],
    exclude: list[str],
) -> list[AgentInfo]:
    """Filter agents by tags.

    Semantics:
    - `include=[]` → all (then exclude can still trim)
    - `include=[a, b, ...]` → any tag from the list must be present (OR)
    - `exclude=[a, b, ...]` → none of these tags may be present (AND)
    - `disabled: true` → always skipped

    Returns the list in original order.
    """
    out = []
    inc_set = set(include)
    exc_set = set(exclude)
    for a in agents:
        if a.disabled:
            continue
        tag_set = set(a.tags)
        if inc_set and not (tag_set & inc_set):
            continue
        if exc_set and (tag_set & exc_set):
            continue
        out.append(a)
    return out


def resolve_agents(config: TournamentConfig, zoo_root: Path) -> list[dict]:
    """Look up AgentInfo for each requested agent_id.

    Returns list of dicts with `id` and `path` fields. Raises ValueError on an
    unknown or disabled agent — callers run this BEFORE creating a run dir so an
    invalid config never leaves an orphan directory.
    """
    all_agents = {a.id: a for a in scan_zoo(zoo_root)}
    out: list[dict] = []
    for aid in config.agents:
        info = all_agents.get(aid)
        if info is None:
            raise ValueError(
                f"Agent {aid!r} not found in zoo {zoo_root}. "
                f"Available: {sorted(all_agents)}"
            )
        if info.disabled:
            raise ValueError(
                f"Agent {aid!r} is disabled; remove from config or un-disable"
            )
        out.append({"id": info.id, "path": info.path})
    return out


def generate_pairs(config: TournamentConfig, agents: list[dict]) -> list[tuple[dict, ...]]:
    """round-robin: C(n,2) pairs (2p) or C(n,4) 4-tuples (4p).
    gauntlet: challenger × each opponent (2p), or challenger + C(n-1,3) (4p)."""
    if config.shape == "gauntlet":
        return _generate_gauntlet_pairs(config, agents)
    if config.format == "2p":
        return list(itertools.combinations(agents, 2))
    # 4p round-robin
    if len(agents) < 4:
        raise ValueError(f"4p format needs ≥4 agents, got {len(agents)}")
    return list(itertools.combinations(agents, 4))


def _generate_gauntlet_pairs(
    config: TournamentConfig, agents: list[dict]
) -> list[tuple[dict, ...]]:
    cid = config.challenger_id
    if cid is None:
        raise ValueError("gauntlet requires challenger_id")
    challenger = next((a for a in agents if a["id"] == cid), None)
    if challenger is None:
        raise ValueError(f"challenger {cid!r} not in selected agents")
    opponents = [a for a in agents if a["id"] != cid]
    if config.format == "2p":
        if not opponents:
            raise ValueError("gauntlet needs ≥1 opponent")
        return [(challenger, opp) for opp in opponents]
    # 4p gauntlet: challenger + every 3-tuple of opponents
    if len(opponents) < 3:
        raise ValueError(f"4p gauntlet needs ≥3 opponents, got {len(opponents)}")
    return [(challenger,) + triple for triple in itertools.combinations(opponents, 3)]
