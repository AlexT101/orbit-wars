#!/usr/bin/env bash
# Convenience wrapper: launch alphaow with current best weights against any opponent.
#
# Usage:
#   ./run.sh <opponent> [--budget-ms 500] [--seed N]
#
# Examples:
#   ./run.sh apollo_fast --seed 42
#   ./run.sh heuristic --budget-ms 1000

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WEIGHTS="$HERE/train/weights/current.bin"

if [ ! -f "$WEIGHTS" ]; then
    echo "warning: no current weights symlink at $WEIGHTS — running without value net"
    unset ALPHAOW_VALUE_NET_PATH
else
    export ALPHAOW_VALUE_NET_PATH="$WEIGHTS"
fi

# Make sure binary is built.
if [ ! -x "$HERE/target/release/alphaow-bot" ]; then
    echo "building alphaow-bot..."
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
exec python3 run_match.py alphaow "$OPP" "$@"
