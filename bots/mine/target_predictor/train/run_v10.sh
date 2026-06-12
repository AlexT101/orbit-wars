#!/usr/bin/env bash
# Train v10 alphaduck model: policy CE + value MSE + auxiliary noop/pair BCE.
#
# Variants:
#   v10_base      : baseline with all features, max-step 200
#   v10_no_turn   : zero turn-related globals (handles games past training horizon)
#   v10_step300   : extend max-step to 300
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-$HERE/data/targets_v10.npz}"
WEIGHTS_DIR="$HERE/weights"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-25}"
BATCH="${BATCH:-256}"
TAG="${1:-base}"

EXTRA_FLAGS=()
case "$TAG" in
  base)
    OUT="$WEIGHTS_DIR/transformer_pair_v10.pt"
    ;;
  no_turn)
    EXTRA_FLAGS=(--drop-turn-features)
    OUT="$WEIGHTS_DIR/transformer_pair_v10_noturn.pt"
    ;;
  step300)
    EXTRA_FLAGS=(--max-step 300)
    OUT="$WEIGHTS_DIR/transformer_pair_v10_step300.pt"
    ;;
  *)
    echo "Unknown tag: $TAG (choices: base, no_turn, step300)"
    exit 1
    ;;
esac

echo "Training $TAG -> $OUT  (device=$DEVICE, epochs=$EPOCHS, max_rows=${MAX_ROWS:-default})"
MAX_ROWS_FLAG=()
if [ -n "${MAX_ROWS:-}" ]; then
  MAX_ROWS_FLAG=(--max-rows "$MAX_ROWS")
fi
python3 "$HERE/pair_net.py" \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --device "$DEVICE" \
  --policy-loss-weight 1.0 \
  --value-loss-weight 1.0 \
  --pair-loss-weight 0.0 \
  --noop-loss-weight 0.0 \
  --select-by policy_ce \
  --out "$OUT" \
  ${MAX_ROWS_FLAG[@]+"${MAX_ROWS_FLAG[@]}"} \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
