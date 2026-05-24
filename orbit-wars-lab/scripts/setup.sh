#!/usr/bin/env bash
# Tworzy .venv oraz instaluje brakujace zaleznosci backendu i viewer'a.
set -euo pipefail

cd "$(dirname "$0")/.."

WITH_RL=0
for arg in "$@"; do
    case "$arg" in
        --with-rl)
            WITH_RL=1
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash scripts/setup.sh [--with-rl]"
            exit 1
            ;;
    esac
done

has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

prepare_temp_dir() {
    if [ -n "${USERPROFILE:-}" ]; then
        TEMP_DIR="${USERPROFILE//\\//}/.owltmp"
    else
        TEMP_DIR="$HOME/.owltmp"
    fi

    mkdir -p "$TEMP_DIR"
    export TMP="$TEMP_DIR"
    export TEMP="$TEMP_DIR"
    export TMPDIR="$TEMP_DIR"
}

create_venv() {
    if [ -d ".venv" ]; then
        return
    fi

    echo "Creating Python virtual environment in .venv ..."
    if has_cmd python3.12; then
        python3.12 -m venv .venv
    elif has_cmd py && py -3.12 -c "import sys" >/dev/null 2>&1; then
        py -3.12 -m venv .venv
    elif has_cmd python; then
        python -m venv .venv
    else
        echo "Python 3.12+ is required. Install it, then rerun bash scripts/setup.sh"
        exit 1
    fi
}

resolve_venv_python() {
    if [ -x ".venv/Scripts/python.exe" ]; then
        VENV_PY=".venv/Scripts/python.exe"
    elif [ -x ".venv/bin/python" ]; then
        VENV_PY=".venv/bin/python"
    else
        echo "No virtualenv Python found in .venv. Remove .venv and rerun bash scripts/setup.sh"
        exit 1
    fi
}

ensure_python_version() {
    if ! "$VENV_PY" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"; then
        echo ".venv is not using Python 3.12+. Remove .venv and rerun bash scripts/setup.sh"
        exit 1
    fi
}

ensure_python_deps() {
    if "$VENV_PY" -c "import fastapi, uvicorn, orbit_wars_app" >/dev/null 2>&1; then
        if [ "$WITH_RL" -eq 0 ] || "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
            return
        fi
    fi

    echo "Installing Python dependencies ..."
    "$VENV_PY" -m pip install -e .
    "$VENV_PY" -m pip install -e ".[dev]"

    if [ "$WITH_RL" -eq 1 ]; then
        echo "Installing optional RL dependencies ..."
        "$VENV_PY" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[rl]"
    fi
}

ensure_frontend_deps() {
    if [ -d "node_modules" ] && [ -d "viewer/node_modules" ]; then
        return
    fi

    if ! has_cmd pnpm; then
        echo "pnpm is required for the viewer. Install it with: npm install -g pnpm"
        exit 1
    fi

    echo "Installing JavaScript dependencies with pnpm ..."
    pnpm install
}

prepare_temp_dir
create_venv
resolve_venv_python
ensure_python_version
ensure_python_deps
ensure_frontend_deps

echo "Setup complete."
