#!/usr/bin/env bash
# MPS version: train support_source + offensive_source on Apple GPU.
# support_target already trained on CPU; preserved as-is.
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
    --arch transformer --epochs 15 --batch-size 512 \
    --opening-weight smooth \
    --device mps \
    --data $DATA \
    --max-step 200 \
    --out train/weights/transformer_${tag}.pt \
    $flags 2>&1 | tee "$LOGDIR/transformer_${tag}.log" \
    | grep -E "epoch|saved|step filter|ship-ratio|label key|opening weight|pos rate|model="
}

run "support_source_mps"   --label-key support_source_labels
run "offensive_source_mps" --label-key offensive_source_labels
