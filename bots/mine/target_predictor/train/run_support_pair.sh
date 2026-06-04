#!/usr/bin/env bash
# Train two transformers on support-launch labels (target and source).
# Same architecture as the winning offensive model. Run after v4 dataset built.
set -e
cd "$(dirname "$0")/.."

DATA=train/data/targets_2k_v4.npz
LOGDIR=train/sweep_logs
mkdir -p "$LOGDIR" train/weights

run() {
  local tag=$1; shift
  local flags="$@"
  echo "=== $tag : $flags ==="
  PYTHONUNBUFFERED=1 python3 train/set_net.py \
    --arch transformer --epochs 15 --batch-size 256 \
    --opening-weight smooth \
    --data $DATA \
    --max-step 200 \
    --out train/weights/transformer_${tag}.pt \
    $flags 2>&1 | tee "$LOGDIR/transformer_${tag}.log" \
    | grep -E "epoch|saved|step filter|ship-ratio|label key|opening weight|pos rate|model="
}

run "support_target"   --label-key support_target_labels
run "support_source"   --label-key support_source_labels
run "offensive_source" --label-key offensive_source_labels
