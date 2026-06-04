#!/usr/bin/env bash
# Convenience wrapper: launch aphrodite with the default XGB weights against an opponent.
#
# Usage:
#   ./run.sh <opponent> [run_match.py args]
#
# Examples:
#   ./run.sh apollo_fast --seed 42
#   APHRODITE_VALUE_NET_PATH=train/weights/xgb_4p.json ./run.sh heuristic --seed 42

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WEIGHTS="${APHRODITE_VALUE_NET_PATH:-$HERE/train/weights/xgb_top10_d6_fixed.json}"

if [ ! -f "$WEIGHTS" ]; then
    echo "warning: no XGB weights at $WEIGHTS - running without value net"
    unset APHRODITE_VALUE_NET_PATH
else
    export APHRODITE_VALUE_NET_PATH="$WEIGHTS"
fi

# Make sure binary is built.
if [ ! -x "$HERE/target/release/aphrodite" ]; then
    echo "building aphrodite..."
    (cd "$HERE" && RUSTFLAGS="-C target-cpu=native" cargo build --release)
fi

if [ $# -lt 1 ]; then
    echo "usage: $0 <opponent> [run_match.py args]"
    exit 2
fi

OPP="$1"
shift
cd "$REPO"
source .venv/bin/activate 2>/dev/null || true
exec python3 run_match.py aphrodite "$OPP" "$@"
