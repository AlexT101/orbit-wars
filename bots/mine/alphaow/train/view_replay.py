"""Render a saved Kaggle orbit_wars episode JSON to a standalone HTML
player you can open in a browser.

Usage:
    python view_replay.py <replay.json> [out.html]

Run with a Python that has kaggle_environments + the orbit_wars env
installed (e.g. ~/Downloads/orbit-wars/.venv/bin/python).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.utils import structify


def main():
    if len(sys.argv) < 2:
        print("usage: view_replay.py <replay.json> [out.html]", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else path.with_suffix(".html")

    d = json.loads(path.read_bytes())
    env = make("orbit_wars", configuration=d.get("configuration", {}), debug=True)
    env.steps = structify(d["steps"])
    # Surface player names in the title if available.
    info = d.get("info") or {}
    agents = info.get("Agents") or []
    names = [a.get("Name", f"p{i}") for i, a in enumerate(agents)]
    if names:
        print(f"players: {names}  rewards={d.get('rewards')}")

    html = env.render(mode="html", width=1000, height=760)
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
