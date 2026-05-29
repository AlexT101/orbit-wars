"""Shared resolver for the real agent zoo used by integration tests.

Prefers `orbit-wars-lab/agents` (the legacy local convention some setups use),
and otherwise falls back to the repo-root `bots/` tree — which already has the
same `baselines/`/`external/`/`mine/` bucket layout `scan_zoo` expects. This
lets the suite run against real agents wherever either tree is present.
"""
from __future__ import annotations

from pathlib import Path

_LAB_ROOT = Path(__file__).parent.parent  # orbit-wars-lab/
_BUCKETS = ("baselines", "external", "mine")


def resolve_real_zoo() -> Path:
    candidates = [_LAB_ROOT / "agents", _LAB_ROOT.parent / "bots"]
    for c in candidates:
        if c.is_dir() and any((c / b).is_dir() for b in _BUCKETS):
            return c
    # Neither present — return the legacy path so failures point somewhere clear.
    return _LAB_ROOT / "agents"


REAL_ZOO = resolve_real_zoo()
