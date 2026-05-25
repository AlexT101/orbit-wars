"""API: POST /api/tournaments."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from orbit_wars_app.main import app
from orbit_wars_app.match import MatchOutcome
from orbit_wars_app.schemas import TournamentConfig
from orbit_wars_app.tournament import Tournament


PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.asyncio
async def test_post_tournament_starts_and_completes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(PROJECT_ROOT / "agents"))

    payload = {
        "agents": ["baselines/random", "baselines/random"],
        "games_per_pair": 1,
        "mode": "fast",
        "format": "2p",
        "parallel": 1,
        "seed_base": 42,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        # Poll progress until completed (timeout 30 s)
        for _ in range(60):
            p = await ac.get(f"/api/runs/{run_id}/progress")
            if p.status_code == 200 and p.json()["status"] == "completed":
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Tournament never completed within 30 s")

        # Get full run details
        d = await ac.get(f"/api/runs/{run_id}")
        assert d.status_code == 200
        assert d.json()["run"]["status"] == "completed"


@pytest.mark.asyncio
async def test_post_tournament_rejects_second_while_running(tmp_path: Path, monkeypatch):
    """Start a slow tournament, immediately POST another, expect 409."""
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(PROJECT_ROOT / "agents"))

    # 3 agents × K=5 games = 30 matches; gives enough window for second POST
    payload = {
        "agents": [
            "baselines/random",
            "baselines/starter",
            "baselines/nearest-sniper",
        ],
        "games_per_pair": 5,
        "mode": "fast",
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r1 = await ac.post("/api/tournaments", json=payload)
        assert r1.status_code == 200
        run_id = r1.json()["run_id"]

        # Immediately post again — expect 409
        r2 = await ac.post("/api/tournaments", json=payload)
        assert r2.status_code == 409, f"Expected 409; got {r2.status_code}: {r2.text}"

        # Wait for r1 to complete (or abort) so next test isn't blocked
        for _ in range(120):  # 60 s budget
            p = await ac.get(f"/api/runs/{run_id}/progress")
            if p.status_code == 200 and p.json()["status"] in ("completed", "aborted"):
                break
            await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_post_tournament_returns_new_run_id_even_with_stale_running_run(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(PROJECT_ROOT / "agents"))

    cfg = TournamentConfig(
        agents=["baselines/random", "baselines/random"],
        games_per_pair=1,
        mode="fast",
        format="2p",
        parallel=1,
        seed_base=42,
        is_quick_match=True,
    )
    t = Tournament(config=cfg, runs_root=tmp_path, zoo_root=PROJECT_ROOT / "agents")
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
        "parallel": 1,
        "seed_base": 42,
        "is_quick_match": True,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        assert run_id != stale_run_id

        for _ in range(60):
            p = await ac.get(f"/api/runs/{run_id}/progress")
            if p.status_code == 200 and p.json()["status"] == "completed":
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Reserved run id never completed within 30 s")


@pytest.mark.asyncio
async def test_post_tournament_cancel_aborts_run(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(PROJECT_ROOT / "agents"))

    def slow_run_match(*args, **kwargs):
        time.sleep(0.05)
        agent_ids = kwargs.get("agent_ids") or args[0]
        return MatchOutcome(
            agent_ids=agent_ids,
            winner=agent_ids[0],
            scores=[1, 0],
            turns=1,
            duration_s=0.05,
            status="ok",
            replay={},
            per_agent_turn_seconds=[[], []],
        )

    monkeypatch.setattr("orbit_wars_app.tournament.run_match", slow_run_match)

    payload = {
        "agents": ["baselines/random", "baselines/starter"],
        "games_per_pair": 20,
        "mode": "fast",
        "format": "2p",
        "parallel": 1,
        "seed_base": 42,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/tournaments", json=payload)
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        rc = await ac.post(f"/api/tournaments/{run_id}/cancel")
        assert rc.status_code == 200
        assert rc.json()["status"] == "cancelling"

        for _ in range(60):
            p = await ac.get(f"/api/runs/{run_id}/progress")
            if p.status_code == 200 and p.json()["status"] == "aborted":
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("Tournament never aborted within 6 s after cancel")
