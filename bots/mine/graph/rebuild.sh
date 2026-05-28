#!/usr/bin/env bash
# Rebuild the graph_native Rust module (and its env_model dependency) and drop
# the resulting .so next to main.py so `import graph_native` picks up the fresh
# build. No virtualenv needed (unlike `maturin develop`).
#
# Rebuild after editing this crate OR experimental_arch/env_model — env_model is
# compiled in as a path dependency, so Cargo picks up its changes here too.
set -euo pipefail
cd "$(dirname "$0")"
maturin build --release
unzip -o -j target/wheels/graph_native-*.whl '*.so' -d .
echo "graph_native rebuilt."
