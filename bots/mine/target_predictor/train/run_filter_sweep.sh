#!/usr/bin/env bash
# Train transformer under different label/filter regimes; compare to the
# s200/owsmooth baseline trained in the arch sweep.
#
# All runs: 15 epochs, opening-weight=smooth, select-by=recall@8.
set -e
cd "$(dirname "$0")/.."

LOGDIR=train/sweep_logs
mkdir -p "$LOGDIR" train/weights

# Each row: tag flags
run() {
  local tag=$1; shift
  local flags="$@"
  echo "=== $tag : $flags ==="
  PYTHONUNBUFFERED=1 python3 train/set_net.py \
    --arch transformer --epochs 15 --batch-size 256 \
    --opening-weight smooth \
    --out train/weights/transformer_${tag}.pt \
    $flags 2>&1 | tee "$LOGDIR/transformer_${tag}.log" \
    | grep -E "epoch|saved|step filter|ship-ratio|using offensive|opening weight|pos rate|model="
}

# 1. offensive-only labels on v3 (max-step 200)
run "v3_off_s200"      --data train/data/targets_2k_v3.npz --max-step 200 --use-offensive-labels

# 2. max-step 150 on v2 baseline
run "v2_s150"          --data train/data/targets_2k_v2.npz --max-step 150

# 3. ship-ratio 2 on v2 baseline (max-step 200)
run "v2_s200_sr2"      --data train/data/targets_2k_v2.npz --max-step 200 --max-ship-ratio 2

# 4. offensive + ship-ratio 2 on v3 (max-step 200)
run "v3_off_s200_sr2"  --data train/data/targets_2k_v3.npz --max-step 200 --use-offensive-labels --max-ship-ratio 2

echo
echo "=== FILTER SWEEP DONE ==="
