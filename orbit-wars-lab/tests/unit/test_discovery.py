"""Tests for orbit_wars_app.discovery."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from orbit_wars_app.discovery import scan_zoo
from tests.conftest import copy_fixture_agent


def _write_agent(zoo: Path, bucket: str, name: str, yaml_data: dict | None, has_main: bool = True):
    """Helper: stworzenie folderu agenta z agent.yaml + main.py."""
    adir = zoo / bucket / name
    adir.mkdir(parents=True)
    if has_main:
        (adir / "main.py").write_text("def agent(obs):\n    return []\n")
    if yaml_data is not None:
        (adir / "agent.yaml").write_text(yaml.safe_dump(yaml_data))


def test_scan_empty_zoo(tmp_zoo: Path):
    agents = scan_zoo(tmp_zoo)
    assert agents == []


def test_scan_finds_agent_with_yaml(tmp_zoo: Path):
    copy_fixture_agent("agent_ok", tmp_zoo / "mine")

    agents = scan_zoo(tmp_zoo)

    assert len(agents) == 1
    a = agents[0]
    assert a.id == "mine/agent_ok"
    assert a.name == "Agent OK"
    assert a.bucket == "mine"
    assert a.description == "Test fixture — empty action agent"
    assert a.author == "test"
    assert "baseline" in a.tags
    assert a.disabled is False
    assert a.has_yaml is True


def test_scan_finds_agent_without_yaml(tmp_zoo: Path):
    copy_fixture_agent("agent_no_yaml", tmp_zoo / "baselines")

    agents = scan_zoo(tmp_zoo)

    assert len(agents) == 1
    a = agents[0]
    assert a.id == "baselines/agent_no_yaml"
    assert a.name == "agent_no_yaml"  # default: folder name
    assert a.bucket == "baselines"
    assert a.has_yaml is False
    assert a.tags == []


def test_scan_respects_disabled_flag(tmp_zoo: Path):
    copy_fixture_agent("agent_disabled", tmp_zoo / "mine")

    agents = scan_zoo(tmp_zoo)

    assert len(agents) == 1
    assert agents[0].disabled is True


def test_scan_handles_broken_yaml(tmp_zoo: Path):
    copy_fixture_agent("agent_broken_yaml", tmp_zoo / "external")

    agents = scan_zoo(tmp_zoo)

    assert len(agents) == 1
    a = agents[0]
    assert a.id == "external/agent_broken_yaml"
    # Broken YAML → fall back to folder name, flag error
    assert a.name == "agent_broken_yaml"
    assert a.last_error is not None
    assert "yaml" in a.last_error.lower()


def test_scan_ignores_missing_main_py(tmp_zoo: Path):
    # Folder bez main.py — nie ma agenta
    (tmp_zoo / "mine" / "just_a_folder").mkdir()

    agents = scan_zoo(tmp_zoo)

    assert agents == []


def test_scan_finds_agent_with_only_submission_tarball(tmp_zoo: Path):
    """Kaggle-style packaging: a folder with submission.tar.gz but no
    loose main.py should still be discovered."""
    agent_dir = tmp_zoo / "mine" / "tar-bot"
    agent_dir.mkdir(parents=True)
    # Placeholder file — discovery only checks existence, not validity;
    # extraction happens lazily at load time.
    (agent_dir / "submission.tar.gz").write_bytes(b"")

    agents = scan_zoo(tmp_zoo)

    assert len(agents) == 1
    assert agents[0].id == "mine/tar-bot"


def test_scan_sorts_by_id(tmp_zoo: Path):
    copy_fixture_agent("agent_ok", tmp_zoo / "mine")
    copy_fixture_agent("agent_no_yaml", tmp_zoo / "baselines")

    agents = scan_zoo(tmp_zoo)

    ids = [a.id for a in agents]
    assert ids == sorted(ids)


def test_scan_rejects_invalid_bucket(tmp_zoo: Path):
    # Agent w nieznanym buckecie (poza baselines/external/mine) — pomijany
    (tmp_zoo / "nonsense").mkdir()
    copy_fixture_agent("agent_ok", tmp_zoo / "nonsense")

    agents = scan_zoo(tmp_zoo)

    assert agents == []


def test_scan_handles_tags_as_string(tmp_zoo: Path):
    """YAML `tags: hello` (string, not list) should NOT become ['h','e','l','l','o']."""
    agent_dir = tmp_zoo / "mine" / "bogus_tags"
    agent_dir.mkdir(parents=True)
    (agent_dir / "main.py").write_text("def agent(obs):\n    return []\n")
    (agent_dir / "agent.yaml").write_text("name: BogusTags\ntags: hello\n")

    agents = scan_zoo(tmp_zoo)

    assert len(agents) == 1
    a = agents[0]
    assert a.tags == []
    assert a.last_error is not None
    assert "tags" in a.last_error.lower()


def test_scan_zoo_parses_new_external_fields(tmp_path: Path):
    """External agent z kernel_slug/version/license/lb_score — wszystkie pola parsowane."""
    zoo = tmp_path / "agents"
    _write_agent(zoo, "external", "tamrazov-starwars", {
        "name": "Tamrazov Starwars",
        "description": "Strong rule-based",
        "author": "Roman Tamrazov",
        "kernel_slug": "romantamrazov/orbit-star-wars-lb-max-1224",
        "kernel_version": 6,
        "date_fetched": "2026-04-21",
        "license": "Apache 2.0",
        "author_claimed_lb_score": 1224,
        "tags": ["rule-based", "forward-sim", "benchmark"],
    })

    agents = scan_zoo(zoo)
    assert len(agents) == 1
    a = agents[0]
    assert a.kernel_slug == "romantamrazov/orbit-star-wars-lb-max-1224"
    assert a.kernel_version == 6
    assert a.license == "Apache 2.0"
    assert a.author_claimed_lb_score == 1224.0
    assert a.date_fetched == "2026-04-21"
    assert a.tags == ["rule-based", "forward-sim", "benchmark"]


def test_scan_zoo_warns_on_deprecated_fields(tmp_path: Path, caplog):
    """Stary agent.yaml z `source_url`/`version` → warning log, pola zachowane."""
    zoo = tmp_path / "agents"
    _write_agent(zoo, "external", "legacy-bot", {
        "name": "Legacy",
        "source_url": "https://www.kaggle.com/old",
        "version": "3",
        "tags": ["external"],
    })

    with caplog.at_level(logging.WARNING):
        agents = scan_zoo(zoo)
    assert len(agents) == 1
    a = agents[0]
    assert a.source_url == "https://www.kaggle.com/old"
    assert a.version == "3"
    assert any("deprecated" in r.message.lower() for r in caplog.records)


def test_scan_zoo_baseline_without_kernel_slug_ok(tmp_path: Path):
    """Baseline bez kernel_slug — żaden warning, wszystkie nowe pola None."""
    zoo = tmp_path / "agents"
    _write_agent(zoo, "baselines", "random", {
        "name": "Random",
        "description": "Engine built-in",
        "tags": ["baseline", "reference"],
    })

    agents = scan_zoo(zoo)
    assert len(agents) == 1
    a = agents[0]
    assert a.kernel_slug is None
    assert a.kernel_version is None
    assert a.license is None


def test_scan_zoo_blank_optional_metadata_does_not_set_error(tmp_path: Path):
    """Blank optional metadata should be treated as missing, not invalid."""
    zoo = tmp_path / "agents"
    _write_agent(zoo, "external", "blank-meta", {
        "name": "Blank Meta",
        "description": "",
        "author": " ",
        "kernel_slug": "",
        "kernel_version": "",
        "date_fetched": " ",
        "license": "",
        "author_claimed_lb_score": "",
        "source_url": "",
        "version": " ",
    })

    agents = scan_zoo(zoo)

    assert len(agents) == 1
    a = agents[0]
    assert a.last_error is None
    assert a.description is None
    assert a.author is None
    assert a.kernel_slug is None
    assert a.kernel_version is None
    assert a.date_fetched is None
    assert a.license is None
    assert a.author_claimed_lb_score is None
    assert a.source_url is None
    assert a.version is None


def test_scan_zoo_unknown_optional_metadata_does_not_set_error(tmp_path: Path):
    """Common placeholder strings like 'unknown' should behave like missing."""
    zoo = tmp_path / "agents"
    _write_agent(zoo, "external", "unknown-meta", {
        "name": "Unknown Meta",
        "author": "unknown",
        "kernel_slug": "unknown",
        "kernel_version": "unknown",
        "license": "unknown",
        "source_url": "N/A",
        "version": "-",
        "author_claimed_lb_score": "none",
    })

    agents = scan_zoo(zoo)

    assert len(agents) == 1
    a = agents[0]
    assert a.last_error is None
    assert a.author is None
    assert a.kernel_slug is None
    assert a.kernel_version is None
    assert a.license is None
    assert a.source_url is None
    assert a.version is None
    assert a.author_claimed_lb_score is None
