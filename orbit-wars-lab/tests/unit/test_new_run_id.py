"""Tests for Tournament._new_run_id — id generation must be collision-safe
even when prior runs have been deleted out of sequence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orbit_wars_app.schemas import TournamentConfig
from orbit_wars_app.tournament import Tournament


def _make_tournament(runs_root: Path) -> Tournament:
    cfg = TournamentConfig(agents=["a", "b"], games_per_pair=1, mode="fast")
    return Tournament(config=cfg, runs_root=runs_root, zoo_root=Path("agents"))


def test_new_run_id_starts_at_001_when_empty(tmp_path: Path):
    t = _make_tournament(tmp_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert t._new_run_id() == f"{today}-001"


def test_new_run_id_skips_existing_indices(tmp_path: Path):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Simulate a deleted run mid-sequence: 001, 002, 004 exist; 003 was removed.
    # Count-based numbering would return 004 (collision); max-based returns 005.
    (tmp_path / f"{today}-001").mkdir()
    (tmp_path / f"{today}-002").mkdir()
    (tmp_path / f"{today}-004").mkdir()

    t = _make_tournament(tmp_path)
    assert t._new_run_id() == f"{today}-005"


def test_new_run_id_ignores_other_days(tmp_path: Path):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (tmp_path / "2020-01-01-099").mkdir()
    (tmp_path / f"{today}-001").mkdir()

    t = _make_tournament(tmp_path)
    assert t._new_run_id() == f"{today}-002"


def test_new_run_id_ignores_non_numeric_suffixes(tmp_path: Path):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (tmp_path / f"{today}-001").mkdir()
    (tmp_path / f"{today}-notanumber").mkdir()
    (tmp_path / f"{today}-").mkdir()

    t = _make_tournament(tmp_path)
    assert t._new_run_id() == f"{today}-002"


def test_new_run_id_ignores_loose_files(tmp_path: Path):
    """`runs/latest.txt` and `runs/runtimes.json` sit alongside run dirs."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (tmp_path / f"{today}-001").mkdir()
    (tmp_path / "latest.txt").write_text("whatever")
    (tmp_path / "runtimes.json").write_text("{}")
    (tmp_path / "trueskill.json").write_text("{}")

    t = _make_tournament(tmp_path)
    assert t._new_run_id() == f"{today}-002"
