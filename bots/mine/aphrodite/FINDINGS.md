# aphrodite Findings

The active bot is DUCT + Apollo candidates + fixed-extrapolation XGB leaf eval.

What remains important:

- Corrected fleet extrapolation is the runtime default.
- `build_submission.py` packages only `main.py`, the Rust `aphrodite` binary,
  and `xgb_2p_old_top10.json`.
- `train/data/2p/old_top10.npz` and
  `train/filter_top10_and_train_xgb.py` are the retained path for rebuilding
  the deployed model.

Removed experiments:

- beam planner
- focused candidate planner
- sequential MCTS planner
- rollout policy variants
- Exp3 selection
- AOWV/MLP/GNN/linear/first-owned model families and weights
