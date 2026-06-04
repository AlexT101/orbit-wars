# aphrodite Findings

The active bot is DUCT + Apollo candidates + fixed-extrapolation XGB leaf eval.

What remains important:

- `main.py` pins `APHRODITE_EXTRAP_FIX=1`.
- `build_submission.py` packages only `main.py`, the Rust `aphrodite` binary,
  and `xgb_top10_d6_fixed.json`.
- `train/data/fixed/combined_top10_fixed.npz` and
  `train/filter_top10_and_train_xgb.py` are the retained path for rebuilding
  the deployed model.

Removed experiments:

- beam planner
- focused candidate planner
- sequential MCTS planner
- rollout policy variants
- Exp3 selection
- AOWV/MLP/GNN/linear/first-owned model families and weights
