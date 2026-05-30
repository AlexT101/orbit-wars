from __future__ import annotations

import sys

from curriculum_train import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--checkpoint-dir",
        "rl_orbit_wars/checkpoints_curriculum",
        "--init-checkpoint",
        "rl_orbit_wars/checkpoints/bc_hellburner_transformer.pt",
        "--total-budget-steps",
        "300000",
        "--chunk-steps",
        "10000",
        "--start-gate-threshold",
        "0.98",
        "--end-gate-threshold",
        "0.65",
        "--gate-games",
        "16",
        "--eval-workers",
        "4",
        "--inner-eval-every-updates",
        "0",
        "--pretrain-if-missing",
        *sys.argv[1:],
    ]
    raise SystemExit(main())
