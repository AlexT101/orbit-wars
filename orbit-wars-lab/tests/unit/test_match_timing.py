"""Tests for per-agent per-turn runtime extraction from env.logs.

`run_match_fast` reads kaggle-environments' own per-call timing (recorded in
`env.logs`) and surfaces it on MatchOutcome.per_agent_turn_seconds. Two layers
covered here: the pure log-parsing helper (no kaggle-envs dep) and an
end-to-end fast-mode match (loads baselines/random).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit_wars_app.match import _per_agent_durations_from_logs, run_match_fast
from orbit_wars_app.runtime_store import RuntimeStore
from orbit_wars_app.schemas import TournamentConfig
from orbit_wars_app.tournament import Tournament


PROJECT_ROOT = Path(__file__).parent.parent.parent

from tests.zoo import REAL_ZOO
RANDOM_AGENT_DIR = REAL_ZOO / "baselines" / "random"


def test_per_agent_durations_from_logs_extracts_only_dicts_with_duration():
    """Reset-step logs lack `duration`; act-step logs include it. Helper must
    skip the former and keep the latter, indexed by agent slot."""
    env_logs = [
        # Reset step: interpreter-only, no agent acted yet.
        [{"stdout": "", "stderr": ""}, {"stdout": "", "stderr": ""}],
        # Step 1: both agents acted.
        [
            {"duration": 0.001, "stdout": "", "stderr": ""},
            {"duration": 0.002, "stdout": "", "stderr": ""},
        ],
        # Step 2: only agent 0 acted (agent 1 was already done).
        [
            {"duration": 0.003, "stdout": "", "stderr": ""},
            {"stdout": "", "stderr": ""},
        ],
    ]
    result = _per_agent_durations_from_logs(env_logs, num_agents=2)
    assert result == [[0.001, 0.003], [0.002]]


def test_per_agent_durations_from_logs_handles_empty():
    assert _per_agent_durations_from_logs([], num_agents=2) == [[], []]
    assert _per_agent_durations_from_logs(None, num_agents=2) == [[], []]


def test_per_agent_durations_ignores_non_numeric_duration():
    env_logs = [
        [{"duration": "not-a-number"}, {"duration": 0.001}],
    ]
    assert _per_agent_durations_from_logs(env_logs, 2) == [[], [0.001]]


@pytest.mark.skipif(
    not RANDOM_AGENT_DIR.is_dir(),
    reason="baselines/random not present — skipping fast-mode integration timing test",
)
def test_run_match_fast_populates_per_agent_turn_seconds():
    """End-to-end: fast match between two random agents must produce a
    non-empty timing sample for each agent."""
    outcome = run_match_fast(
        agent_ids=["baselines/random#0", "baselines/random#1"],
        agent_paths=[RANDOM_AGENT_DIR, RANDOM_AGENT_DIR],
        seed=42,
    )
    assert len(outcome.per_agent_turn_seconds) == 2
    # Both random agents took at least one turn in a normal match.
    assert all(len(samples) > 0 for samples in outcome.per_agent_turn_seconds)
    # Durations are positive wallclock floats — random agent shouldn't take
    # more than a small fraction of a second per move on modern hardware,
    # but we keep the upper bound loose to avoid CI flakes.
    for samples in outcome.per_agent_turn_seconds:
        for s in samples:
            assert 0.0 <= s < 5.0


@pytest.mark.skipif(
    not RANDOM_AGENT_DIR.is_dir(),
    reason="baselines/random not present — skipping runtime-store integration test",
)
def test_tournament_persists_runtimes_json(tmp_path: Path):
    """One quick match must write runs/<id>/../runtimes.json with both agents'
    avg_ms populated."""
    cfg = TournamentConfig(
        agents=["baselines/random", "baselines/random"],
        games_per_pair=1,
        mode="fast",
    )
    Tournament(
        config=cfg,
        runs_root=tmp_path,
        zoo_root=REAL_ZOO,
    ).run()

    runtimes_path = tmp_path / "runtimes.json"
    assert runtimes_path.is_file(), "runtimes.json should be written after tournament"
    data = json.loads(runtimes_path.read_text())
    assert data["schema_version"] == 1
    # 'baselines/random' played itself — single key with samples.
    entry = data["runtimes"].get("baselines/random")
    assert entry is not None
    assert entry["total_turns"] > 0
    assert entry["avg_ms"] > 0
    # Sanity: average should be a small fraction of a second in ms units.
    assert entry["avg_ms"] < 5000


@pytest.mark.skipif(
    not RANDOM_AGENT_DIR.is_dir(),
    reason="baselines/random not present — skipping clear-roundtrip integration test",
)
def test_runtime_store_clear_after_tournament(tmp_path: Path):
    cfg = TournamentConfig(
        agents=["baselines/random", "baselines/random"],
        games_per_pair=1,
        mode="fast",
    )
    Tournament(
        config=cfg,
        runs_root=tmp_path,
        zoo_root=REAL_ZOO,
    ).run()

    store = RuntimeStore(tmp_path / "runtimes.json")
    assert store.get("baselines/random") is not None

    assert store.clear("baselines/random") is True
    store.save()

    reloaded = RuntimeStore(tmp_path / "runtimes.json")
    assert reloaded.get("baselines/random") is None
