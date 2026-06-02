#!/usr/bin/env python
"""Run apollo's aimer against the local Orbit Wars aim benchmark.

A command-line equivalent of `benchmark-for-aiming-implementation.ipynb` that
does not depend on the notebook or on Kaggle. It:

  1. builds the apollo native extension (`maturin develop --release`),
  2. loads the local benchmark dataset from `aim_benchmark/aim_samples.npz`,
  3. aims every sample with `apollo_native.aim_angle`,
  4. scores the angles with the real kaggle engine (`aim_benchmark.validate`).

The `aim_angle` call here is byte-for-byte what the Kaggle notebook runs, so
the accuracy reported locally matches what you see on Kaggle.

Usage (from the repo root, inside the project venv):

    python aim_benchmark/test_apollo_aim.py             # build apollo, run all samples
    python aim_benchmark/test_apollo_aim.py --no-build  # skip the maturin rebuild
    python aim_benchmark/test_apollo_aim.py --limit 500 # quick subset while iterating

Requires the project venv (numpy + kaggle_environments + maturin). Run it with
that venv's python, e.g. `venv\\Scripts\\python.exe aim_benchmark\\test_apollo_aim.py`,
so the freshly built module installs into the same interpreter that imports it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent          # aim_benchmark/
REPO_ROOT = BENCH_DIR.parent
APOLLO_DIR = REPO_ROOT / "bots" / "mine" / "apollo"


def _maturin_cmd() -> list[str]:
    """Prefer the maturin sitting next to the running interpreter (so the build
    installs into *this* venv), falling back to whatever is on PATH."""
    bindir = Path(sys.executable).parent
    for name in ("maturin.exe", "maturin"):
        cand = bindir / name
        if cand.exists():
            return [str(cand)]
    return ["maturin"]


def build_apollo() -> None:
    cmd = _maturin_cmd() + ["develop", "--release"]
    print(f"building apollo: {' '.join(cmd)} (cwd={APOLLO_DIR})")
    subprocess.run(cmd, cwd=APOLLO_DIR, check=True)


def _warm_import_engine() -> None:
    """Import the kaggle engine once, silencing its env-discovery chatter
    (open_spiel logging + 'Loading environment ... failed' prints)."""
    logging.disable(logging.WARNING)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import kaggle_environments  # noqa: F401
    logging.disable(logging.NOTSET)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build", action="store_true",
                    help="skip `maturin develop --release` and use the already-installed module")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N samples (for quick iteration)")
    ap.add_argument("--npz", type=Path, default=None,
                    help="path to aim_samples.npz (defaults to aim_benchmark/aim_samples.npz)")
    args = ap.parse_args()

    if not args.no_build:
        build_apollo()

    # Resolve the benchmark module + engine before importing the native module
    # so a build/import failure is reported clearly.
    sys.path.insert(0, str(BENCH_DIR))
    _warm_import_engine()

    import aim_benchmark as ab

    try:
        import apollo_native
    except ImportError as e:
        print(f"failed to import apollo_native: {e}\n"
              "Build it first (drop --no-build) with the project venv's python.",
              file=sys.stderr)
        return 1

    samples = list(ab.iter_samples(args.npz))
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"benchmark samples: {len(samples)}")

    # Aim every launch — identical to the notebook's `aim` wrapper.
    t0 = time.perf_counter()
    angles = [apollo_native.aim_angle(s.obs, s.source, s.target, s.fleet_size)
              for s in samples]
    aim_secs = time.perf_counter() - t0
    print(f"aimed {len(angles)} launches with apollo_native.aim_angle in {aim_secs:.2f}s "
          f"({1000 * aim_secs / max(len(angles), 1):.3f} ms/shot)")

    # Score with the real kaggle engine. `ab.validate` reloads and re-scores the
    # full dataset; when limiting we score our subset directly (same predicate)
    # so the counts line up and we don't re-decompress the npz.
    print("scoring against the kaggle engine...")
    t0 = time.perf_counter()
    results: list[bool] = []
    for i, (s, a) in enumerate(zip(samples, angles), 1):
        results.append(ab._validate_one(s, a))
        if i % 500 == 0 or i == len(samples):
            print(f"  scored {i}/{len(samples)}", end="\r", flush=True)
    print()
    score_secs = time.perf_counter() - t0

    correct = sum(results)
    n = len(results)
    print(f"\napollo aimer accuracy: {correct}/{n} = {correct / n:.2%} "
          f"(scored in {score_secs:.1f}s)")

    # Breakdown: reachable targets (must hit) vs impossible (must decline).
    reach = [(s, r) for s, r in zip(samples, results) if s.meta.get("reachable", True)]
    impossible = [(s, r) for s, r in zip(samples, results) if not s.meta.get("reachable", True)]
    if reach:
        rc = sum(r for _, r in reach)
        print(f"  reachable:   {rc}/{len(reach)} = {rc / len(reach):.2%} hit the target")
    if impossible:
        ic = sum(r for _, r in impossible)
        print(f"  impossible:  {ic}/{len(impossible)} = {ic / len(impossible):.2%} correctly declined")

    # Where apollo improves on the example aimer (samples it got wrong), if the
    # dataset carries that flag.
    ex_fail = [(s, r) for s, r in zip(samples, results)
               if s.meta.get("example_aimer_fail", False)]
    if ex_fail:
        rec = sum(r for _, r in ex_fail)
        print(f"  of {len(ex_fail)} example-aimer failures, apollo gets {rec} "
              f"= {rec / len(ex_fail):.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
