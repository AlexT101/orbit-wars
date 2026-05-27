"""Tests for orbit_wars_app.agent_extract.ensure_extracted."""
from __future__ import annotations

import io
import os
import tarfile
import time
from pathlib import Path

import pytest

from orbit_wars_app.agent_extract import (
    EXTRACT_DIRNAME,
    MTIME_MARKER,
    TarballError,
    ensure_extracted,
)
from orbit_wars_app.agent_serve import load_agent


def _make_tarball(dest: Path, members: dict[str, bytes]) -> Path:
    """Write a gzip tarball at `dest` with the given name -> content map."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return dest


AGENT_CODE = b"def agent(obs):\n    return ['tar-agent']\n"


def test_ensure_extracted_returns_dir_unchanged_when_no_tarball(tmp_path: Path):
    (tmp_path / "main.py").write_text("def agent(obs): return []\n")
    assert ensure_extracted(tmp_path) == tmp_path


def test_ensure_extracted_prefers_loose_main_py_over_tarball(tmp_path: Path):
    (tmp_path / "main.py").write_text("def agent(obs):\n    return ['loose-agent']\n")
    _make_tarball(tmp_path / "submission.tar.gz", {"main.py": AGENT_CODE})

    assert ensure_extracted(tmp_path) == tmp_path
    assert not (tmp_path / EXTRACT_DIRNAME).exists()


def test_ensure_extracted_unpacks_tarball(tmp_path: Path):
    _make_tarball(tmp_path / "submission.tar.gz", {"main.py": AGENT_CODE})

    extract_dir = ensure_extracted(tmp_path)

    assert extract_dir == tmp_path / EXTRACT_DIRNAME
    assert (extract_dir / "main.py").read_bytes() == AGENT_CODE
    assert (extract_dir / MTIME_MARKER).is_file()


def test_ensure_extracted_caches_on_mtime(tmp_path: Path):
    tarball = _make_tarball(tmp_path / "submission.tar.gz", {"main.py": AGENT_CODE})
    extract_dir = ensure_extracted(tmp_path)
    marker = extract_dir / MTIME_MARKER
    first_marker = marker.read_text()
    assert int(first_marker) == tarball.stat().st_mtime_ns

    # Second call with no tarball change → marker stays put.
    ensure_extracted(tmp_path)
    assert marker.read_text() == first_marker

    # Touch the tarball forward; cache should invalidate and the marker should
    # advance to the tarball's new mtime.
    new_ts = time.time() + 5
    os.utime(tarball, (new_ts, new_ts))
    ensure_extracted(tmp_path)
    assert marker.read_text() != first_marker
    assert int(marker.read_text()) == tarball.stat().st_mtime_ns


def test_ensure_extracted_preserves_aux_files(tmp_path: Path):
    """Native modules and helpers ship alongside main.py — both must land."""
    _make_tarball(
        tmp_path / "submission.tar.gz",
        {"main.py": AGENT_CODE, "weights.bin": b"\x00\x01\x02"},
    )
    extract_dir = ensure_extracted(tmp_path)
    assert (extract_dir / "weights.bin").read_bytes() == b"\x00\x01\x02"


def test_ensure_extracted_rejects_path_traversal(tmp_path: Path):
    _make_tarball(
        tmp_path / "submission.tar.gz",
        {"../evil.py": b"# escapes the extract dir\n"},
    )
    with pytest.raises(TarballError):
        ensure_extracted(tmp_path)


def test_ensure_extracted_drops_removed_files_on_reextract(tmp_path: Path):
    """A new tarball without an old file should erase the old extraction."""
    tarball = _make_tarball(
        tmp_path / "submission.tar.gz",
        {"main.py": AGENT_CODE, "old.txt": b"v1\n"},
    )
    extract_dir = ensure_extracted(tmp_path)
    assert (extract_dir / "old.txt").is_file()

    # Replace tarball — no old.txt this time.
    _make_tarball(tarball, {"main.py": AGENT_CODE})
    # Bump mtime so cache invalidates.
    new_ts = time.time() + 5
    os.utime(tarball, (new_ts, new_ts))

    ensure_extracted(tmp_path)
    assert not (extract_dir / "old.txt").exists()


def test_load_agent_extracts_and_runs_from_tarball(tmp_path: Path):
    """End-to-end: load_agent finds main.py inside submission.tar.gz."""
    agent_dir = tmp_path / "mine" / "tar-bot"
    _make_tarball(agent_dir / "submission.tar.gz", {"main.py": AGENT_CODE})

    agent_fn = load_agent(str(agent_dir))
    assert agent_fn is not None
    assert agent_fn({}) == ["tar-agent"]


def test_load_agent_prefers_loose_main_py_over_tarball(tmp_path: Path):
    """End-to-end: a working main.py wins over a packaged submission."""
    agent_dir = tmp_path / "mine" / "dual-bot"
    agent_dir.mkdir(parents=True)
    (agent_dir / "main.py").write_text(
        "def agent(obs):\n    return ['loose-agent']\n"
    )
    _make_tarball(agent_dir / "submission.tar.gz", {"main.py": AGENT_CODE})

    agent_fn = load_agent(str(agent_dir))

    assert agent_fn is not None
    assert agent_fn({}) == ["loose-agent"]


def test_fast_match_resolves_tarball_agents(tmp_path: Path):
    """Regression: fast mode hands main.py paths to kaggle-envs, which reads
    them directly. Without extracting first, a tarball agent would fail with
    'Could not find: <dir>/main.py'. ensure_extracted must be applied before
    paths are handed off."""
    from orbit_wars_app.match import run_match_fast

    # Materialise two tarball agents — minimal valid orbit_wars agents that
    # just return [] (no moves). kaggle-envs reads them via their main.py.
    a = tmp_path / "tar-a"
    b = tmp_path / "tar-b"
    code = b"def agent(obs):\n    return []\n"
    _make_tarball(a / "submission.tar.gz", {"main.py": code})
    _make_tarball(b / "submission.tar.gz", {"main.py": code})

    out = run_match_fast(["tar-a", "tar-b"], [a, b], seed=0)

    # Before the fix this returned status='crashed' with
    # "Could not find: <tmp>/tar-a/main.py".
    assert out.status in ("ok", "draw"), (out.status, out.replay.get("error"))
    assert out.turns > 0
