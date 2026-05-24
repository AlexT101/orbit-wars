#!/usr/bin/env bash
# Uruchamia backend FastAPI (:8000) + Vite dev (:5173) jednoczesnie.
# Ctrl+C zabija oba procesy.
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/setup.sh

resolve_venv_python() {
    if [ -x ".venv/Scripts/python.exe" ]; then
        VENV_PY=".venv/Scripts/python.exe"
    elif [ -x ".venv/bin/python" ]; then
        VENV_PY=".venv/bin/python"
    else
        echo "No virtualenv Python found in .venv. Remove .venv and rerun bash scripts/dev.sh"
        exit 1
    fi
}

cleanup() {
    if [ -n "${BACKEND_PID:-}" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
}

resolve_venv_python
trap cleanup EXIT SIGINT SIGTERM

"$VENV_PY" -m uvicorn orbit_wars_app.main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

(cd viewer && pnpm dev -- --port 5173 --strictPort) &
FRONTEND_PID=$!

echo ""
echo "================================"
echo " Backend : http://localhost:8000"
echo " Viewer  : http://localhost:5173"
echo " Ctrl+C  : stop both"
echo "================================"

wait "$BACKEND_PID" "$FRONTEND_PID"
