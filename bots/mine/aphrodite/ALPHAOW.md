# aphrodite Current Notes

Current aphrodite is intentionally narrow:

- DUCT simultaneous-move search
- Apollo candidate generation
- Apollo final redirect pass
- no rollout simulation
- fixed-extrapolation SummaryV2 XGBoost value net

The deployed model is `train/weights/xgb_top10_d6_fixed.json`, and runtime
feature extraction uses the corrected extrapolation by default.

Historical beam search, focused candidates, sequential MCTS, AOWV/MLP value
nets, rollout policy variants, and extra-feature experiments have been removed.
