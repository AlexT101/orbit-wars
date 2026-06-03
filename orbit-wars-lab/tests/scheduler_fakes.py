"""Top-level fake match jobs for scheduler tests.

These must live in an importable module (not inside a test function) so they
survive the pickle/spawn boundary into a pebble worker process on Windows.
None of them touch kaggle-environments, so worker startup stays fast.
"""
from __future__ import annotations

import time

from orbit_wars_app.scheduler import MatchJobResult, QueuedMatch


def ok_job(job: QueuedMatch) -> MatchJobResult:
    """Deterministic win for player 0 — no real engine involved."""
    n = len(job.agent_ids)
    return MatchJobResult(
        match_counter=job.match_counter,
        agent_ids=list(job.agent_ids),
        winner=job.agent_ids[0],
        scores=[1] + [0] * (n - 1),
        turns=10,
        duration_s=0.01,
        seed=job.seed,
        status="ok",
        replay_path="",
        worker_pid=0,
        per_agent_turn_seconds=[[0.001] for _ in job.agent_ids],
    )


def delayed_job(job: QueuedMatch) -> MatchJobResult:
    """Slow enough for API tests to observe a partially completed tournament."""
    time.sleep(0.2)
    return ok_job(job)


def slow_job(job: QueuedMatch) -> MatchJobResult:
    """Sleeps long enough that the test always kills it (timeout or cancel)."""
    time.sleep(30)
    return ok_job(job)


def crash_job(job: QueuedMatch) -> MatchJobResult:
    raise RuntimeError("boom in worker")
