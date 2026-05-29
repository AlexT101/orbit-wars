"""Scheduler core: fair RR, completion, crash/timeout/stop, concurrency.

Uses fake job functions (tests/scheduler_fakes.py) so the real kaggle engine
never runs — these exercise the scheduling/lifecycle machinery, not matches.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from orbit_wars_app.scheduler import QueuedMatch, Scheduler, match_timeout_for
from orbit_wars_app.schemas import TournamentConfig
from tests.scheduler_fakes import crash_job, ok_job, slow_job


def test_match_timeout_formula():
    # (500 + 60) * players + 20
    assert match_timeout_for(2) == 1140
    assert match_timeout_for(4) == 2260
    assert match_timeout_for(1) == 580  # floor at 1 player


def _make_zoo(tmp_path: Path, names: list[str]) -> Path:
    """Minimal zoo: baselines/<name>/main.py for each name. Fake jobs never
    execute the file, so a stub is enough for resolve_agents to succeed."""
    zoo = tmp_path / "agents"
    bucket = zoo / "baselines"
    bucket.mkdir(parents=True)
    for n in names:
        d = bucket / n
        d.mkdir()
        (d / "main.py").write_text("def agent(obs, config=None):\n    return []\n")
    return zoo


def _wait_inactive(sched: Scheduler, run_id: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not sched.is_active(run_id):
            return
        time.sleep(0.02)
    raise AssertionError(f"tournament {run_id} still active after {timeout}s")


@pytest.fixture
def zoo(tmp_path: Path) -> Path:
    return _make_zoo(tmp_path, ["a", "b", "c", "d"])


# --------------------------------------------------------------------------
# Fair round-robin (white-box: no pool, inspect _next_job_locked directly)
# --------------------------------------------------------------------------


def test_fair_round_robin_interleaves_tournaments(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(runs_root=runs, zoo_root=zoo)  # pool NOT started → no dispatch
    a = sched.submit(
        TournamentConfig(agents=["baselines/a", "baselines/b"], games_per_pair=2, mode="fast")
    )
    b = sched.submit(
        TournamentConfig(agents=["baselines/c", "baselines/d"], games_per_pair=2, mode="fast")
    )

    picked: list[tuple[str, int]] = []
    with sched._lock:
        for _ in range(4):
            nxt = sched._next_job_locked()
            assert nxt is not None
            ts, job = nxt
            picked.append((ts.id, job.match_counter))

    # One match from each tournament in turn, FIFO within each.
    assert picked == [(a, 1), (b, 1), (a, 2), (b, 2)]


# --------------------------------------------------------------------------
# End-to-end lifecycle with a real (killable) pool but fake jobs
# --------------------------------------------------------------------------


def test_completion_writes_artifacts(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(runs_root=runs, zoo_root=zoo, concurrency=2, job_fn=ok_job)
    sched.start()
    try:
        run_id = sched.submit(
            TournamentConfig(
                agents=["baselines/a", "baselines/b", "baselines/c"],
                games_per_pair=2,
                mode="fast",
            )
        )
        _wait_inactive(sched, run_id)
    finally:
        sched.shutdown()

    run = json.loads((runs / run_id / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["total_matches"] == 6  # C(3,2)=3 pairs × 2
    assert run["matches_done"] == 6

    results = json.loads((runs / run_id / "results.json").read_text())
    assert len(results["matches"]) == 6
    assert results["status"] == "completed"
    assert all(m["status"] == "ok" for m in results["matches"])

    assert (runs / "trueskill.json").is_file()


def test_concurrent_tournaments_both_complete(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(runs_root=runs, zoo_root=zoo, concurrency=4, job_fn=ok_job)
    sched.start()
    try:
        a = sched.submit(
            TournamentConfig(agents=["baselines/a", "baselines/b"], games_per_pair=3, mode="fast")
        )
        b = sched.submit(
            TournamentConfig(agents=["baselines/c", "baselines/d"], games_per_pair=3, mode="fast")
        )
        assert a != b
        _wait_inactive(sched, a)
        _wait_inactive(sched, b)
    finally:
        sched.shutdown()

    for rid in (a, b):
        assert json.loads((runs / rid / "run.json").read_text())["status"] == "completed"


def test_worker_crash_recorded_not_fatal(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(runs_root=runs, zoo_root=zoo, concurrency=2, job_fn=crash_job)
    sched.start()
    try:
        run_id = sched.submit(
            TournamentConfig(agents=["baselines/a", "baselines/b"], games_per_pair=2, mode="fast")
        )
        _wait_inactive(sched, run_id)
    finally:
        sched.shutdown()

    results = json.loads((runs / run_id / "results.json").read_text())
    assert len(results["matches"]) == 2
    assert all(m["status"] == "crashed" for m in results["matches"])
    assert all(m["error"] for m in results["matches"])
    # The run still completes despite every match crashing.
    assert json.loads((runs / run_id / "run.json").read_text())["status"] == "completed"


def test_match_timeout_kills_and_records(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(
        runs_root=runs, zoo_root=zoo, concurrency=2, match_timeout_s=1.0, job_fn=slow_job
    )
    sched.start()
    try:
        run_id = sched.submit(
            TournamentConfig(agents=["baselines/a", "baselines/b"], games_per_pair=1, mode="fast")
        )
        _wait_inactive(sched, run_id, timeout=15.0)
    finally:
        sched.shutdown()

    results = json.loads((runs / run_id / "results.json").read_text())
    assert len(results["matches"]) == 1
    assert results["matches"][0]["status"] == "timeout"
    assert json.loads((runs / run_id / "run.json").read_text())["status"] == "completed"


def test_stop_drops_queued_and_kills_running(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    # concurrency=1 so only one slow match runs while the rest wait in queue.
    sched = Scheduler(runs_root=runs, zoo_root=zoo, concurrency=1, job_fn=slow_job)
    sched.start()
    try:
        run_id = sched.submit(
            TournamentConfig(
                agents=["baselines/a", "baselines/b", "baselines/c", "baselines/d"],
                games_per_pair=3,
                mode="fast",
            )
        )
        # Wait until one match is actually running.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not sched.running_matches():
            time.sleep(0.02)
        assert sched.running_matches(), "no match started"

        assert sched.stop(run_id) is True
        _wait_inactive(sched, run_id, timeout=15.0)
    finally:
        sched.shutdown()

    run = json.loads((runs / run_id / "run.json").read_text())
    assert run["status"] == "aborted"
    # Not all matches ran — queue was dropped and the in-flight one was killed.
    assert run["matches_done"] < 18
    assert sched.running_matches() == []


def test_empty_tournament_finalizes_immediately(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(runs_root=runs, zoo_root=zoo, concurrency=2, job_fn=ok_job)
    sched.start()
    try:
        run_id = sched.submit(
            TournamentConfig(agents=["baselines/a", "baselines/b"], games_per_pair=0, mode="fast")
        )
        _wait_inactive(sched, run_id, timeout=5.0)
    finally:
        sched.shutdown()

    run = json.loads((runs / run_id / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["total_matches"] == 0
    assert run["matches_done"] == 0


def test_set_concurrency_while_idle(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(runs_root=runs, zoo_root=zoo, concurrency=2, job_fn=ok_job)
    sched.start()
    try:
        assert sched.concurrency == 2
        assert sched.set_concurrency(4) == 4
        assert sched._pool_size == 4
        # Still functional after a pool recreate.
        run_id = sched.submit(
            TournamentConfig(agents=["baselines/a", "baselines/b"], games_per_pair=1, mode="fast")
        )
        _wait_inactive(sched, run_id)
        assert json.loads((runs / run_id / "run.json").read_text())["status"] == "completed"
    finally:
        sched.shutdown()


def test_restart_pool_requeues_running_match(tmp_path: Path, zoo: Path):
    """Restarting the pool kills in-flight matches but re-queues them (so none
    are lost) and keeps serving from the fresh workers."""
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(runs_root=runs, zoo_root=zoo, concurrency=1, job_fn=slow_job)
    sched.start()
    try:
        run_id = sched.submit(
            TournamentConfig(
                agents=["baselines/a", "baselines/b", "baselines/c"],
                games_per_pair=1,
                mode="fast",
            )
        )
        total = sched._tournaments[run_id].total_matches  # C(3,2) = 3
        # Wait for the first match to be running.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not sched.running_matches():
            time.sleep(0.02)
        assert sched.running_matches()

        sched.restart_pool()

        # No match lost: pending + running + completed still accounts for all,
        # and nothing has completed.
        ts = sched._tournaments[run_id]
        with sched._lock:
            accounted = len(ts.pending) + len(ts.running) + ts.completed_count
        assert accounted == total
        assert ts.completed_count == 0
        assert sched.running_matches()  # a fresh worker resumed the work
    finally:
        sched.shutdown()


def test_submit_rejects_unknown_agent(tmp_path: Path, zoo: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    sched = Scheduler(runs_root=runs, zoo_root=zoo, job_fn=ok_job)
    with pytest.raises(ValueError):
        sched.submit(
            TournamentConfig(agents=["baselines/a", "baselines/nope"], games_per_pair=1, mode="fast")
        )
    # No orphan run dir created on a bad config.
    assert list(runs.iterdir()) == []
