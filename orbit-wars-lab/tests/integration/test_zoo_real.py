"""Sanity check: scan_zoo na prawdziwym agents/ znajduje wszystkie baselines."""
from pathlib import Path

import pytest

from orbit_wars_app.discovery import scan_zoo


PROJECT_ROOT = Path(__file__).parent.parent.parent

from tests.zoo import REAL_ZOO


def test_real_zoo_has_core_baselines():
    zoo = REAL_ZOO
    agents = scan_zoo(zoo)

    baseline_ids = {a.id for a in agents if a.bucket == "baselines"}
    # The three core baselines must always be present; a zoo may carry extras
    # (e.g. bots/baselines also ships hellburner), so this is a subset check.
    assert {
        "baselines/nearest-sniper",
        "baselines/random",
        "baselines/starter",
    } <= baseline_ids


def test_real_zoo_has_externals():
    """Externals są gitignored — skip jeśli nie pobrane (fresh clone).

    Po Tasks 11-13 (bulk fetch 2026-04-21) docelowe external agents:
    tamrazov-starwars, lakhindar-agent, pilkwang-structured, sigmaborov-*,
    dylanxue-phoenix, yuriygreben-architect, ichigoe-score828, etc (22 total).
    """
    zoo = REAL_ZOO
    if not (zoo / "external" / "tamrazov-starwars").exists():
        pytest.skip("externals not downloaded (gitignored, expected on fresh clone)")

    agents = scan_zoo(zoo)

    external_ids = {a.id for a in agents if a.bucket == "external"}
    assert "external/tamrazov-starwars" in external_ids


def test_real_zoo_no_broken_yaml():
    zoo = REAL_ZOO
    agents = scan_zoo(zoo)

    errors = [(a.id, a.last_error) for a in agents if a.last_error]
    assert errors == [], f"Found agents with errors: {errors}"
