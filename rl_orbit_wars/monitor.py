from __future__ import annotations

import argparse
import time
from pathlib import Path

from orbit_wars_rl.visualization import write_training_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or refresh the Orbit Wars RL HTML report.")
    parser.add_argument("--log-dir", default="rl_orbit_wars/checkpoints")
    parser.add_argument("--watch", action="store_true", help="Refresh until interrupted.")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    while True:
        out = write_training_report(log_dir)
        print(f"wrote {out}", flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

