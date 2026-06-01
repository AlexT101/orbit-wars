"""Shared pytest fixtures."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_zoo(tmp_path: Path) -> Path:
    """Tmp directory with `agents/` skeleton. Tests populate it per-case."""
    zoo = tmp_path / "agents"
    (zoo / "baselines").mkdir(parents=True)
    (zoo / "external").mkdir(parents=True)
    (zoo / "mine").mkdir(parents=True)
    return zoo


@pytest.fixture
def tmp_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs


def copy_fixture_agent(fixture_name: str, dest: Path) -> Path:
    """Copy tests/fixtures/<fixture_name>/ into dest/<fixture_name>/."""
    src = FIXTURES_DIR / fixture_name
    target = dest / fixture_name
    shutil.copytree(src, target)
    return target


@pytest.fixture(scope="session", autouse=True)
def _shutdown_api_scheduler_at_session_end():
    """Tear down any process-wide scheduler the API built so its pebble worker
    pool doesn't outlive the test session."""
    yield
    try:
        from orbit_wars_app import api

        api._shutdown_scheduler()
    except Exception:
        pass
