"""API: /api/ratings leaderboard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from orbit_wars_app.main import app
from orbit_wars_app.schemas import TournamentConfig
from orbit_wars_app.tournament import Tournament
from tests.conftest import copy_fixture_agent


PROJECT_ROOT = Path(__file__).parent.parent.parent

from tests.zoo import REAL_ZOO


@pytest.mark.asyncio
async def test_ratings_empty_before_any_tournament(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/ratings?format=2p")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_ratings_after_tournament(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    cfg = TournamentConfig(
        agents=["baselines/random", "baselines/nearest-sniper"],
        games_per_pair=2,
        mode="fast",
    )
    Tournament(config=cfg, runs_root=tmp_path, zoo_root=REAL_ZOO).run()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/ratings?format=2p")
    assert r.status_code == 200
    ratings = r.json()
    ids = {x["agent_id"] for x in ratings}
    assert ids == {"baselines/random", "baselines/nearest-sniper"}
    for rating in ratings:
        assert "mu" in rating
        assert "sigma" in rating
        assert "conservative" in rating
        assert "games_played" in rating
        assert "rank" in rating
    # Ranked 1..N
    ranks = sorted(r["rank"] for r in ratings)
    assert ranks == [1, 2]


@pytest.mark.asyncio
async def test_ratings_hides_disabled_agents_by_default(
    tmp_path: Path,
    tmp_zoo: Path,
    monkeypatch,
):
    copy_fixture_agent("agent_ok", tmp_zoo / "mine")
    copy_fixture_agent("agent_disabled", tmp_zoo / "mine")
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(tmp_zoo))
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    (tmp_path / "trueskill.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ratings": {
                    "mine/agent_disabled": {
                        "2p": {"mu": 900.0, "sigma": 40.0, "games_played": 5},
                    },
                    "mine/agent_ok": {
                        "2p": {"mu": 700.0, "sigma": 80.0, "games_played": 3},
                    },
                },
            }
        )
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        visible = await ac.get("/api/ratings?format=2p")
        all_ratings = await ac.get("/api/ratings?format=2p&include_disabled=true")

    assert visible.status_code == 200
    assert all_ratings.status_code == 200
    assert visible.json() == [
        {
            "agent_id": "mine/agent_ok",
            "mu": 700.0,
            "sigma": 80.0,
            "conservative": 460.0,
            "games_played": 3,
            "rank": 1,
        }
    ]
    assert {r["agent_id"] for r in all_ratings.json()} == {
        "mine/agent_disabled",
        "mine/agent_ok",
    }


@pytest.mark.asyncio
async def test_ratings_invalid_format_422(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_RUNS_DIR", str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/ratings?format=8p")
    assert r.status_code == 422
