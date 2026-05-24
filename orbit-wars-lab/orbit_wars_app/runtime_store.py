"""Persistent per-agent turn-runtime stats.

Mirrors the TrueSkillStore pattern: a single JSON file under `runs/` is the
canonical source. The store accumulates per-call wallclock samples from both
match modes (fast wraps the agent callable; faithful wraps the HTTP handler)
and exposes a rolling mean (ms per turn) for the agents tab.

Schema (v1):
    {
      "schema_version": 1,
      "last_updated": "ISO-8601",
      "runtimes": {
        "<agent_id>": {
          "total_turns": int,
          "total_seconds": float,
          "avg_ms": float,            # derived = total_seconds / total_turns * 1000
          "last_updated": "ISO-8601"
        }
      }
    }
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class RuntimeStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self._runtimes: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return
        if data.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported runtimes schema_version {data.get('schema_version')};"
                f" expected {self.SCHEMA_VERSION}"
            )
        self._runtimes = data.get("runtimes", {})

    def record(self, agent_id: str, turn_seconds: list[float]) -> None:
        """Fold a batch of per-turn wallclock samples into the agent's running mean.

        Empty list is a no-op (e.g. a match that crashed before the agent took
        a single turn shouldn't bias the stats).
        """
        if not turn_seconds:
            return
        entry = self._runtimes.setdefault(
            agent_id,
            {"total_turns": 0, "total_seconds": 0.0, "avg_ms": 0.0, "last_updated": ""},
        )
        entry["total_turns"] = int(entry["total_turns"]) + len(turn_seconds)
        entry["total_seconds"] = float(entry["total_seconds"]) + sum(turn_seconds)
        entry["avg_ms"] = (
            entry["total_seconds"] / entry["total_turns"] * 1000.0
            if entry["total_turns"] > 0
            else 0.0
        )
        entry["last_updated"] = datetime.now(timezone.utc).isoformat()

    def get(self, agent_id: str) -> Optional[dict]:
        return self._runtimes.get(agent_id)

    def list_all(self) -> list[dict]:
        """Return one row per agent for the agents tab."""
        out: list[dict] = []
        for aid, e in self._runtimes.items():
            out.append({
                "agent_id": aid,
                "total_turns": int(e.get("total_turns", 0)),
                "total_seconds": float(e.get("total_seconds", 0.0)),
                "avg_ms": float(e.get("avg_ms", 0.0)),
                "last_updated": e.get("last_updated", ""),
            })
        return out

    def clear(self, agent_id: str) -> bool:
        """Remove one agent's runtime stats. Returns True if there was an entry."""
        return self._runtimes.pop(agent_id, None) is not None

    def clear_all(self) -> int:
        n = len(self._runtimes)
        self._runtimes = {}
        return n

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "runtimes": self._runtimes,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)
