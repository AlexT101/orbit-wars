"""Write a trimmed manifest keeping only the last N chunks (most recent days).

Non-destructive: only emits a new manifest JSON next to the original; the .npz
chunks are untouched. Point IL_DATASET_PATH at the output to train on a recency
slice. Chunk paths stay relative ("chunks/..."), so the output must live in the
same data dir as the source manifest.

Usage:
    python trim_manifest.py <source_manifest.json> <last_n_chunks>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src = Path(sys.argv[1]).expanduser().resolve()
    last_n = int(sys.argv[2])
    payload = json.loads(src.read_text(encoding="utf-8"))
    chunks = list(payload.get("chunks") or [])
    if last_n <= 0 or last_n > len(chunks):
        print(f"last_n must be in 1..{len(chunks)}")
        return 2
    kept = chunks[-last_n:]
    payload["chunks"] = kept
    out = src.with_name(f"manifest_last{last_n}.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    kept_rows = sum(int(c["rows"]) for c in kept)
    total_rows = sum(int(c["rows"]) for c in chunks)
    print(f"wrote {out}")
    print(f"kept {len(kept)}/{len(chunks)} chunks  rows {kept_rows:,}/{total_rows:,} "
          f"({kept_rows / max(1, total_rows):.1%})")
    print(f"first kept chunk path: {kept[0]['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
