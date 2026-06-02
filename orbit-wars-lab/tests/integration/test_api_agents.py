"""API: /api/agents endpoints."""
from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from orbit_wars_app.main import app
from tests.conftest import copy_fixture_agent
from tests.zoo import REAL_ZOO


@pytest.fixture(autouse=True)
def _use_real_zoo(monkeypatch):
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(REAL_ZOO))


@pytest.mark.asyncio
async def test_api_agents_lists_baselines():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/agents")

    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    ids = {a["id"] for a in data}
    assert "baselines/random" in ids
    assert "baselines/nearest-sniper" in ids
    assert "baselines/starter" in ids


@pytest.mark.asyncio
async def test_api_agents_hides_disabled_by_default(tmp_zoo: Path, monkeypatch):
    copy_fixture_agent("agent_ok", tmp_zoo / "mine")
    copy_fixture_agent("agent_disabled", tmp_zoo / "mine")
    monkeypatch.setenv("ORBIT_WARS_ZOO_DIR", str(tmp_zoo))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        visible = await ac.get("/api/agents")
        all_agents = await ac.get("/api/agents?include_disabled=true")

    assert visible.status_code == 200
    assert all_agents.status_code == 200
    assert {a["id"] for a in visible.json()} == {"mine/agent_ok"}
    assert {a["id"] for a in all_agents.json()} == {
        "mine/agent_ok",
        "mine/agent_disabled",
    }


@pytest.mark.asyncio
async def test_api_agents_detail_returns_metadata():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/agents/baselines/nearest-sniper")

    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "baselines/nearest-sniper"
    assert data["name"] == "Nearest Planet Sniper"
    assert data["bucket"] == "baselines"
    assert "rule-based" in data["tags"]


@pytest.mark.asyncio
async def test_api_agents_detail_404_for_missing():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/agents/nonexistent/agent")
    assert r.status_code == 404
