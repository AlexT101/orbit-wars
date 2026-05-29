"""Lightweight JSONL helper.

The original repo had a full HTML training-report generator here; this
experimental_arch variant logs to wandb instead, so all that code was
removed. The JSONL writer is retained because the PPO trainer's resume
logic reads from metrics.jsonl / eval.jsonl / phase_events.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
