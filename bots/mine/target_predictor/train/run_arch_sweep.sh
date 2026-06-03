#!/usr/bin/env bash
# Train transformer + 3 GNN variants on the v2 dataset with the new training
# setup (max-step=200, opening-weight=smooth, select-by=recall@8, 15 epochs).
# Writes weights and a summary log to train/sweep_logs/.

set -e
cd "$(dirname "$0")/.."

DATA=train/data/targets_2k_v2.npz
LOGDIR=train/sweep_logs
mkdir -p "$LOGDIR" train/weights

EPOCHS=${EPOCHS:-15}
MAXSTEP=${MAXSTEP:-200}
OW=${OW:-smooth}
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

SUFFIX="_s${MAXSTEP}_ow${OW}"
[[ -n "$EXTRA_FLAGS" ]] && SUFFIX="${SUFFIX}_custom"

for arch in transformer distbias graphsage gat; do
  out=train/weights/${arch}${SUFFIX}.pt
  log=$LOGDIR/${arch}${SUFFIX}.log
  echo "=== training $arch -> $out (log $log) ==="
  python3 train/set_net.py \
    --data $DATA --arch $arch --epochs $EPOCHS \
    --max-step $MAXSTEP --opening-weight $OW \
    --batch-size 256 --out "$out" $EXTRA_FLAGS \
    2>&1 | tee "$log" | grep -E "epoch|model=|pos rate|step filter|ship-ratio|using offensive|saved"
done

echo
echo "=== SWEEP DONE ==="
echo "summary (best epoch per arch):"
for f in $LOGDIR/*${SUFFIX}.log; do
  arch=$(basename $f .log)
  best=$(grep "saved (best" $f | tail -1)
  printf "  %-30s %s\n" "$arch" "$best"
done
