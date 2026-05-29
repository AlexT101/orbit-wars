"""API: POST /api/tournaments (scheduler-backed).

Real-engine cases use the repo's bots/ zoo (see tests/zoo.py). The stop case
injects a fake slow job so it doesn't depend on a genuinely-slow agent.
"""
from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from orbit_wars_app import api
from orbit_wars_app.main import app
from orbit_wars_app.scheduler import Scheduler
from orbit_wars_app.schemas import TournamentConfig
from orbit_wars_app.tournament import Tournament
from tests.scheduler_fakes import slow_job
from tests.zoo import REAL_ZOO


@pytest.fixture(autouse=True)
def _shutdown_scheduler_after():
    yield
    api._shutdown_scheduler()


@pytest.mark.asyncio
async def test_post_tournament_starts_and_completes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(REAL_ZOO))

    payload = {
        "agents": ["baselines/random", "baselines/random"],
        "games_per_pair": 1,
        "mode": "fast",
        "format": "2p",
        "seed_base": 42,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        run_id = r.json()["run_id"]

        for _ in range(60):
            p = await ac.get(f"/api/runs/{run_id}/progress")
            if p.status_code == 200 and p.json()["status"] == "completed":
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Tournament never completed within 30 s")

        d = await ac.get(f"/api/runs/{run_id}")
        assert d.status_code == 200
        assert d.json()["run"]["status"] == "completed"


@pytest.mark.asyncio
async def test_post_two_tournaments_run_concurrently(tmp_path: Path, monkeypatch):
    """The single-tournament 409 is gone — a second POST is accepted and both
    tournaments run."""
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(REAL_ZOO))

    payload = {
        "agents": ["baselines/random", "baselines/starter"],
        "games_per_pair": 2,
        "mode": "fast",
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r1 = await ac.post("/api/tournaments", json=payload)
        r2 = await ac.post("/api/tournaments", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        ids = {r1.json()["run_id"], r2.json()["run_id"]}
        assert len(ids) == 2

        for rid in ids:
            for _ in range(120):
                p = await ac.get(f"/api/runs/{rid}/progress")
                if p.status_code == 200 and p.json()["status"] in ("completed", "aborted"):
                    break
                await asyncio.sleep(0.5)
            else:
                pytest.fail(f"{rid} never finished")


@pytest.mark.asyncio
async def test_post_tournament_returns_new_run_id_even_with_stale_running_run(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(REAL_ZOO))

    cfg = TournamentConfig(
        agents=["baselines/random", "baselines/random"],
        games_per_pair=1,
        mode="fast",
        format="2p",
        seed_base=42,
        is_quick_match=True,
    )
    t = Tournament(config=cfg, runs_root=tmp_path, zoo_root=REAL_ZOO)
    stale_run_id = t.next_run_id()
    stale_dir = tmp_path / stale_run_id
    stale_dir.mkdir()
    (stale_dir / "run.json").write_text(json.dumps({
        "id": stale_run_id,
        "started_at": "2026-05-25T17:39:53.603702+00:00",
        "finished_at": None,
        "mode": "fast",
        "format": "2p",
        "status": "running",
        "total_matches": 1,
        "matches_done": 0,
        "is_quick_match": True,
    }))

    payload = {
        "agents": ["baselines/random", "baselines/random"],
        "games_per_pair": 1,
        "mode": "fast",
        "format": "2p",
        "seed_base": 42,
        "is_quick_match": True,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        assert run_id != stale_run_id

        # The stale "running" run is reconciled to aborted on scheduler startup.
        stale_after = json.loads((stale_dir / "run.json").read_text())
        assert stale_after["status"] == "aborted"

        for _ in range(60):
            p = await ac.get(f"/api/runs/{run_id}/progress")
            if p.status_code == 200 and p.json()["status"] == "completed":
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Reserved run id never completed within 30 s")


@pytest.mark.asyncio
async def test_post_tournament_stop_aborts_run(tmp_path: Path, monkeypatch):
    """Stop drops queued matches and kills in-flight ones. Uses a fake slow job
    + a real bots/ zoo for resolution, concurrency=1 so most stay queued."""
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(REAL_ZOO))
    monkeypatch.setattr(api, "Scheduler", partial(Scheduler, job_fn=slow_job, concurrency=1))

    payload = {
        "agents": ["baselines/random", "baselines/starter"],
        "games_per_pair": 20,
        "mode": "fast",
        "format": "2p",
        "seed_base": 42,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        # Wait until a match is actually running, then stop.
        for _ in range(100):
            run = await ac.get("/api/scheduler/running")
            if run.status_code == 200 and run.json():
                break
            await asyncio.sleep(0.05)

        rc = await ac.post(f"/api/tournaments/{run_id}/stop")
        assert rc.status_code == 200
        assert rc.json()["status"] == "stopping"

        for _ in range(100):
            p = await ac.get(f"/api/runs/{run_id}/progress")
            if p.status_code == 200 and p.json()["status"] == "aborted":
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("Tournament never aborted within 10 s after stop")


@pytest.mark.asyncio
async def test_cancel_alias_still_works(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(REAL_ZOO))
    monkeypatch.setattr(api, "Scheduler", partial(Scheduler, job_fn=slow_job, concurrency=1))

    payload = {
        "agents": ["baselines/random", "baselines/starter"],
        "games_per_pair": 20,
        "mode": "fast",
        "seed_base": 42,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        run_id = r.json()["run_id"]
        for _ in range(100):
            run = await ac.get("/api/scheduler/running")
            if run.status_code == 200 and run.json():
                break
            await asyncio.sleep(0.05)
        rc = await ac.post(f"/api/tournaments/{run_id}/cancel")
        assert rc.status_code == 200
