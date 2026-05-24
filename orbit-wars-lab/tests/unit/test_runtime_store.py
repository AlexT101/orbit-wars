"""Tests for orbit_wars_app.runtime_store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit_wars_app.runtime_store import RuntimeStore


def test_fresh_store_is_empty(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtimes.json")
    assert store.list_all() == []
    assert store.get("baselines/random") is None


def test_record_computes_avg_ms(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtimes.json")
    # 3 samples: 0.001s, 0.002s, 0.003s → avg = 2 ms
    store.record("baselines/random", [0.001, 0.002, 0.003])

    entry = store.get("baselines/random")
    assert entry is not None
    assert entry["total_turns"] == 3
    assert entry["total_seconds"] == pytest.approx(0.006)
    assert entry["avg_ms"] == pytest.approx(2.0)


def test_record_empty_list_is_noop(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtimes.json")
    store.record("baselines/random", [])
    assert store.get("baselines/random") is None


def test_record_accumulates_across_calls(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtimes.json")
    store.record("a", [0.001, 0.001])      # 2 samples, 2ms total
    store.record("a", [0.003])              # +1 sample, +3ms

    entry = store.get("a")
    assert entry["total_turns"] == 3
    assert entry["total_seconds"] == pytest.approx(0.005)
    assert entry["avg_ms"] == pytest.approx(5.0 / 3)


def test_clear_removes_one_agent(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtimes.json")
    store.record("a", [0.001])
    store.record("b", [0.002])

    assert store.clear("a") is True
    assert store.get("a") is None
    assert store.get("b") is not None
    # Clearing a missing agent is idempotent
    assert store.clear("does-not-exist") is False


def test_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "runtimes.json"

    store = RuntimeStore(path)
    store.record("agent-x", [0.001, 0.002])
    store.save()

    reloaded = RuntimeStore(path)
    entry = reloaded.get("agent-x")
    assert entry is not None
    assert entry["total_turns"] == 2
    assert entry["avg_ms"] == pytest.approx(1.5)


def test_list_all_shape_matches_api_contract(tmp_path: Path):
    """Rows returned by list_all() must contain the keys the UI reads."""
    store = RuntimeStore(tmp_path / "runtimes.json")
    store.record("a", [0.001])

    rows = store.list_all()
    assert len(rows) == 1
    row = rows[0]
    for key in ("agent_id", "total_turns", "total_seconds", "avg_ms", "last_updated"):
        assert key in row, f"missing key {key!r} from list_all() row"
    assert row["agent_id"] == "a"


def test_load_rejects_unknown_schema_version(tmp_path: Path):
    path = tmp_path / "runtimes.json"
    path.write_text(json.dumps({"schema_version": 99, "runtimes": {}}))
    with pytest.raises(ValueError, match="schema_version"):
        RuntimeStore(path)


def test_load_tolerates_corrupted_json(tmp_path: Path):
    """A truncated/garbled file shouldn't blow up app startup."""
    path = tmp_path / "runtimes.json"
    path.write_text("{not json")

    store = RuntimeStore(path)
    assert store.list_all() == []
