#!/usr/bin/env python
"""Dump apollo's impossible-bucket aim failures for investigation.

An "impossible" sample is one the benchmark marks `meta['reachable'] == False`:
no launch angle can hit the target, so the correct answer is to decline
(`aim_angle` returns None). A failure here is a sample where apollo instead
returned an angle — it thought the path was clear when the engine says nothing
connects.

This re-runs `apollo_native.aim_angle` over the dataset, collects every such
case, and writes them (with the full obs, so each is replayable without the
npz) to `impossible_failures.json`.

Usage (from repo root, with the project venv's python):
    venv\\Scripts\\python.exe aim_benchmark\\dump_impossible_failures.py
    venv\\Scripts\\python.exe aim_benchmark\\dump_impossible_failures.py --out other.json --limit 1000
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent          # aim_benchmark/
REPO_ROOT = BENCH_DIR.parent


def _warm_import_engine() -> None:
    logging.disable(logging.WARNING)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import kaggle_environments  # noqa: F401
    logging.disable(logging.NOTSET)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "impossible_failures.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="only scan the first N samples")
    ap.add_argument("--npz", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(BENCH_DIR))
    _warm_import_engine()

    import aim_benchmark as ab
    import apollo_native

    samples = list(ab.iter_samples(args.npz))
    if args.limit is not None:
        samples = samples[: args.limit]

    failures = []
    n_impossible = 0
    for i, s in enumerate(samples):
        if s.meta.get("reachable", True):
            continue
        n_impossible += 1
        angle = apollo_native.aim_angle(s.obs, s.source, s.target, s.fleet_size)
        if angle is None:
            continue  # correctly declined
        # apollo shot at an impossible target. Sanity-check what the engine says
        # the shot actually hits (None = sun / off-board / nothing).
        hit = ab._hit_planet(s, float(angle))
        failures.append({
            "sample_index": i,
            "source": int(s.source),
            "target": int(s.target),
            "fleet_size": int(s.fleet_size),
            "apollo_angle": float(angle),
            "engine_hit": (None if hit is None else int(hit)),
            "meta": {k: (None if v is None else (bool(v) if isinstance(v, bool) else v))
                     for k, v in s.meta.items()},
            "obs": s.obs,
        })

    args.out.write_text(json.dumps(failures, indent=2))
    print(f"impossible samples scanned: {n_impossible}")
    print(f"impossible-bucket failures (apollo did not decline): {len(failures)}")
    if n_impossible:
        print(f"decline rate: {(n_impossible - len(failures)) / n_impossible:.2%}")
    print(f"wrote {args.out} ({len(failures)} records)")

    # Quick tally of what the engine says those non-declined shots hit.
    if failures:
        from collections import Counter
        hits = Counter("nothing/sun/oob" if f["engine_hit"] is None
                       else ("hit-target" if f["engine_hit"] == f["target"]
                             else "hit-other-planet")
                       for f in failures)
        print("engine outcome of non-declined shots:")
        for k, v in hits.most_common():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
