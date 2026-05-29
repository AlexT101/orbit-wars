"""API: scheduler-backed /tournaments + /scheduler endpoints.

Injects fake match jobs (no real engine) by wrapping the Scheduler ctor so the
API plumbing is exercised without kaggle-environments or a real zoo.
"""
from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from orbit_wars_app import api
from orbit_wars_app.main import app
from orbit_wars_app.scheduler import Scheduler
from tests.scheduler_fakes import ok_job


def _make_zoo(tmp_path: Path, names: list[str]) -> Path:
    zoo = tmp_path / "agents"
    bucket = zoo / "baselines"
    bucket.mkdir(parents=True)
    for n in names:
        d = bucket / n
        d.mkdir()
        (d / "main.py").write_text("def agent(obs, config=None):\n    return []\n")
    return zoo


@pytest.fixture
def fake_env(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    zoo = _make_zoo(tmp_path, ["a", "b", "c", "d"])
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(runs))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(zoo))
    # Inject fake jobs into any scheduler the API builds this test.
    monkeypatch.setattr(api, "Scheduler", partial(Scheduler, job_fn=ok_job))
    yield runs
    api._shutdown_scheduler()


async def _wait_completed(ac: AsyncClient, run_id: str, timeout: float = 20.0) -> dict:
    for _ in range(int(timeout / 0.1)):
        p = await ac.get(f"/api/runs/{run_id}/progress")
        if p.status_code == 200 and p.json()["status"] in ("completed", "aborted"):
            return p.json()
        await asyncio.sleep(0.1)
    raise AssertionError(f"{run_id} never finished")


@pytest.mark.asyncio
async def test_queue_and_complete(fake_env):
    payload = {
        "agents": ["baselines/a", "baselines/b"],
        "games_per_pair": 2,
        "mode": "fast",
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        run_id = r.json()["run_id"]
        prog = await _wait_completed(ac, run_id)
        assert prog["status"] == "completed"
        assert prog["matches_done"] == 2


@pytest.mark.asyncio
async def test_two_tournaments_concurrently_no_409(fake_env):
    payload_a = {"agents": ["baselines/a", "baselines/b"], "games_per_pair": 3, "mode": "fast"}
    payload_b = {"agents": ["baselines/c", "baselines/d"], "games_per_pair": 3, "mode": "fast"}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r1 = await ac.post("/api/tournaments", json=payload_a)
        r2 = await ac.post("/api/tournaments", json=payload_b)
        assert r1.status_code == 200
        assert r2.status_code == 200  # no single-tournament 409 anymore
        await _wait_completed(ac, r1.json()["run_id"])
        await _wait_completed(ac, r2.json()["run_id"])


@pytest.mark.asyncio
async def test_bad_agent_returns_400(fake_env):
    payload = {"agents": ["baselines/a", "baselines/nope"], "games_per_pair": 1, "mode": "fast"}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_scheduler_status_and_concurrency(fake_env):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Set concurrency (also forces scheduler creation).
        r = await ac.put("/api/scheduler/concurrency", json={"concurrency": 6})
        assert r.status_code == 200
        assert r.json()["concurrency"] == 6

        s = await ac.get("/api/scheduler")
        assert s.status_code == 200
        body = s.json()
        assert body["concurrency"] == 6
        assert "running" in body and "tournaments" in body

        # Persisted to disk.
        assert (fake_env / "scheduler-settings.json").is_file()


@pytest.mark.asyncio
async def test_restart_pool_endpoint(fake_env):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.get("/api/scheduler")  # force scheduler creation
        r = await ac.post("/api/scheduler/restart-pool")
        assert r.status_code == 200
        assert r.json()["restarted"] is True


@pytest.mark.asyncio
async def test_stop_unknown_returns_409(fake_env):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Force scheduler creation so /stop has something to query.
        await ac.get("/api/scheduler")
        r = await ac.post("/api/tournaments/2099-01-01-001/stop")
        assert r.status_code == 409
