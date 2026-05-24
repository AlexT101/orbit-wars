"""Match runner — fast mode (in-process) and faithful mode (subprocess+HTTP).

Task 8 implements fast; Task 9 adds faithful.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class MatchOutcome:
    agent_ids: list[str]
    winner: Optional[str]           # agent_id or None (draw / error)
    scores: list[int]               # final ship sum per player (planets + fleets)
    turns: int
    duration_s: float
    seed: int = 0                   # logged for audit; engine currently ignores it
    status: Literal["ok", "timeout", "crashed", "agent_failed_to_start", "invalid_action", "draw"] = "ok"
    # agent_failed_to_start is reserved for faithful mode (Task 9)
    replay: dict = field(default_factory=dict)
    # Per-agent per-turn wallclock samples (seconds). Index aligns with agent_ids.
    # Empty inner list = agent never took a turn (match crashed before its move).
    # Sourced from kaggle-environments' own Agent.act() timing (core.py logs),
    # so this is the same number Kaggle would deadline-check against.
    per_agent_turn_seconds: list[list[float]] = field(default_factory=list)


def _crashed_replay_skeleton(error: str) -> dict:
    """Produce a replay dict with the same top-level keys as env.toJSON()
    so downstream code (save_replay, viewer) doesn't KeyError on missing keys."""
    return {"error": error, "steps": [], "rewards": [], "statuses": []}


def _per_agent_durations_from_logs(env_logs: list, num_agents: int) -> list[list[float]]:
    """Extract per-agent per-turn durations from `env.logs`.

    Shape of env.logs: list[step][agent_idx] -> dict with at least one of
    {"duration", "stdout", "stderr"}. The initial reset step has no
    "duration" key (no agent has acted yet) — those are skipped.
    """
    out: list[list[float]] = [[] for _ in range(num_agents)]
    for step in env_logs or []:
        if not isinstance(step, list):
            continue
        for idx, entry in enumerate(step):
            if idx >= num_agents:
                continue
            if isinstance(entry, dict):
                d = entry.get("duration")
                if isinstance(d, (int, float)):
                    out[idx].append(float(d))
    return out


def _per_agent_streams_from_logs(
    env_logs: list, num_agents: int
) -> tuple[list[list[str]], list[list[str]]]:
    """Extract per-agent stdout/stderr chunks from `env.logs` (fast mode)."""
    outs: list[list[str]] = [[] for _ in range(num_agents)]
    errs: list[list[str]] = [[] for _ in range(num_agents)]
    for step in env_logs or []:
        if not isinstance(step, list):
            continue
        for idx, entry in enumerate(step):
            if idx >= num_agents or not isinstance(entry, dict):
                continue
            so = entry.get("stdout")
            if so:
                outs[idx].append(so if isinstance(so, str) else str(so))
            se = entry.get("stderr")
            if se:
                errs[idx].append(se if isinstance(se, str) else str(se))
    return outs, errs


def _write_agent_logs(
    log_dir: Optional[Path],
    log_prefix: str,
    agent_ids: list[str],
    stdouts: list[list[str]],
    stderrs: list[list[str]],
) -> None:
    """Persist per-agent stdout/stderr to <log_dir>/<prefix>-<safe_id>.{stdout,stderr}.log.

    Empty streams are skipped — no zero-byte files. Failures are swallowed so
    one bad disk write can't fail the match.
    """
    if log_dir is None:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    prefix = f"{log_prefix}-" if log_prefix else ""
    for aid, so_chunks, se_chunks in zip(agent_ids, stdouts, stderrs):
        safe = aid.replace("/", "_")
        for ext, chunks in (("stdout", so_chunks), ("stderr", se_chunks)):
            if not chunks:
                continue
            text = "".join(chunks)
            if not text.endswith("\n"):
                text += "\n"
            try:
                (log_dir / f"{prefix}{safe}.{ext}.log").write_text(
                    text, encoding="utf-8", errors="replace"
                )
            except OSError:
                pass


def run_match_fast(
    agent_ids: list[str],
    agent_paths: list[Path],
    *,
    seed: int = 0,
    log_dir: Optional[Path] = None,
    log_prefix: str = "",
) -> MatchOutcome:
    """Run a single match in fast mode (kaggle-envs in-process).

    `agent_ids` order = player order (index 0 = player 0 = Q1 home).
    `agent_paths` must correspond 1:1 to agent_ids.

    `seed` is stored in the outcome for audit; kaggle-environments engine
    currently ignores the seed internally (per postmortem 2026-04-20).
    """
    if len(agent_ids) != len(agent_paths):
        raise ValueError(
            f"agent_ids and agent_paths length mismatch: "
            f"{len(agent_ids)} vs {len(agent_paths)}"
        )
    from kaggle_environments import make

    from .agent_extract import ensure_extracted

    env = make("orbit_wars", debug=False)

    # kaggle-environments loads main.py directly from each path. For
    # Kaggle-style `submission.tar.gz` agents, materialize the tarball to its
    # cached `.extracted/` dir first so a real main.py exists where kaggle-envs
    # looks. ensure_extracted is a no-op for loose-source agents.
    resolved_paths = [ensure_extracted(p) for p in agent_paths]

    start = time.monotonic()
    try:
        env.run([str(p / "main.py") for p in resolved_paths])
    except Exception as e:
        duration = time.monotonic() - start
        # env.logs may hold partial per-turn timings even when run() raised
        # (e.g. a deadline late in the match) — surface what we have rather
        # than dropping every sample, so a flaky agent's stats still update.
        env_logs = getattr(env, "logs", []) or []
        partial_timings = _per_agent_durations_from_logs(env_logs, len(agent_ids))
        outs, errs = _per_agent_streams_from_logs(env_logs, len(agent_ids))
        _write_agent_logs(log_dir, log_prefix, agent_ids, outs, errs)
        return MatchOutcome(
            agent_ids=agent_ids,
            winner=None,
            scores=[],
            turns=0,
            duration_s=duration,
            seed=seed,
            status="crashed",
            replay=_crashed_replay_skeleton(str(e)),
            per_agent_turn_seconds=partial_timings,
        )
    duration = time.monotonic() - start
    replay = env.toJSON()
    winner, scores, turns, status = _extract_outcome(replay, agent_ids)
    per_agent_timings = _per_agent_durations_from_logs(env.logs, len(agent_ids))
    outs, errs = _per_agent_streams_from_logs(env.logs, len(agent_ids))
    _write_agent_logs(log_dir, log_prefix, agent_ids, outs, errs)
    return MatchOutcome(
        agent_ids=agent_ids,
        winner=winner,
        scores=scores,
        turns=turns,
        duration_s=duration,
        seed=seed,
        status=status,  # type: ignore[arg-type]
        replay=replay,
        per_agent_turn_seconds=per_agent_timings,
    )


def _extract_outcome(
    replay: dict, agent_ids: list[str]
) -> tuple[Optional[str], list[int], int, str]:
    """Parse terminal state: winner, per-player scores, turn count, status."""
    steps = replay.get("steps") or []
    if not steps:
        return None, [], 0, "crashed"
    final_step = steps[-1]
    if not final_step:
        return None, [], 0, "crashed"

    num_players = len(agent_ids)
    rewards = [s.get("reward") for s in final_step]

    # Extract scores from last observation in state[0]
    state0 = final_step[0]
    obs = state0.get("observation", {})
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])

    scores = [0] * num_players
    for p in planets:
        owner = p[1] if len(p) > 1 else -1
        ships = p[5] if len(p) > 5 else 0
        if 0 <= owner < num_players:
            scores[owner] += int(ships)
    for f in fleets:
        owner = f[1] if len(f) > 1 else -1
        ships = f[6] if len(f) > 6 else 0
        if 0 <= owner < num_players:
            scores[owner] += int(ships)

    # Winner: exactly one reward == 1
    winners_idx = [i for i, r in enumerate(rewards) if r == 1]
    if len(winners_idx) == 1:
        winner = agent_ids[winners_idx[0]]
    else:
        winner = None

    # Status based on any agent's final status
    statuses = [s.get("status") for s in final_step]
    if "ERROR" in statuses:
        status = "crashed"
    elif "TIMEOUT" in statuses:
        status = "timeout"
    elif "INVALID" in statuses:
        status = "invalid_action"
    elif winner is None:
        status = "draw"
    else:
        status = "ok"

    turns = len(steps)
    return winner, scores, turns, status


def run_match(
    agent_ids: list[str],
    agent_paths: list[Path],
    *,
    mode: Literal["fast", "faithful"] = "fast",
    seed: int = 0,
    log_dir: Optional[Path] = None,
    log_prefix: str = "",
) -> MatchOutcome:
    """Dispatcher: fast (in-process) vs faithful (subprocess+HTTP).

    If `log_dir` is set, per-agent stdout/stderr is written to
    `<log_dir>/<log_prefix>-<safe_agent_id>.{stdout,stderr}.log`.
    """
    if mode == "fast":
        return run_match_fast(
            agent_ids, agent_paths, seed=seed,
            log_dir=log_dir, log_prefix=log_prefix,
        )
    return run_match_faithful(
        agent_ids, agent_paths, seed=seed,
        log_dir=log_dir, log_prefix=log_prefix,
    )


def run_match_faithful(
    agent_ids: list[str],
    agent_paths: list[Path],
    *,
    seed: int = 0,
    log_dir: Optional[Path] = None,
    log_prefix: str = "",
) -> MatchOutcome:
    """Run match with each agent in its own subprocess + HTTP server.

    Uses kaggle-envs UrlAgent path — identical protocol to Kaggle production.
    """
    if len(agent_ids) != len(agent_paths):
        raise ValueError(
            f"agent_ids and agent_paths length mismatch: "
            f"{len(agent_ids)} vs {len(agent_paths)}"
        )
    from kaggle_environments import make

    from .agent_subprocess import spawn_agent, shutdown

    handles: list = []
    try:
        for aid, apath in zip(agent_ids, agent_paths):
            try:
                h = spawn_agent(apath, agent_id=aid)
                handles.append(h)
            except Exception as e:
                # One agent's spawn failed; report and abort this match
                return MatchOutcome(
                    agent_ids=agent_ids,
                    winner=None,
                    scores=[],
                    turns=0,
                    duration_s=0.0,
                    seed=seed,
                    status="agent_failed_to_start",
                    replay=_crashed_replay_skeleton(f"{aid}: {e}"),
                )

        urls = [h.url for h in handles]
        env = make("orbit_wars", debug=False)

        start = time.monotonic()
        try:
            env.run(urls)
        except Exception as e:
            duration = time.monotonic() - start
            partial_timings = _per_agent_durations_from_logs(
                getattr(env, "logs", []) or [], len(agent_ids)
            )
            return MatchOutcome(
                agent_ids=agent_ids,
                winner=None,
                scores=[],
                turns=0,
                duration_s=duration,
                seed=seed,
                status="crashed",
                replay=_crashed_replay_skeleton(str(e)),
                per_agent_turn_seconds=partial_timings,
            )
        duration = time.monotonic() - start
        replay = env.toJSON()
        winner, scores, turns, status = _extract_outcome(replay, agent_ids)
        per_agent_timings = _per_agent_durations_from_logs(env.logs, len(agent_ids))
        return MatchOutcome(
            agent_ids=agent_ids,
            winner=winner,
            scores=scores,
            turns=turns,
            duration_s=duration,
            seed=seed,
            status=status,  # type: ignore[arg-type]
            replay=replay,
            per_agent_turn_seconds=per_agent_timings,
        )
    finally:
        for h in handles:
            shutdown(h)
        # Persist per-agent stdout/stderr captured by shutdown(). Aligns
        # `handles` with `agent_ids` by order; if a spawn failed earlier we
        # have fewer handles, so zip naturally truncates.
        if log_dir is not None and handles:
            stdouts = [h.stdout_lines for h in handles]
            stderrs = [h.stderr_lines for h in handles]
            _write_agent_logs(
                log_dir, log_prefix, agent_ids[:len(handles)], stdouts, stderrs,
            )
