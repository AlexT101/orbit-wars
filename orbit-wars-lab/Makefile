# Orbit Wars Lab - common developer commands.
# Run `make` or `make help` for a list.

VENV_PY := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)

.DEFAULT_GOAL := help
.PHONY: help setup install install-rl install-dev dev backend viewer test lint build docker-build docker-up docker-down sync-public clean

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## Create .venv + install all deps (full dev setup)
	@bash scripts/setup.sh --with-rl
	@echo "setup complete. Run: make dev"

.venv:
	@bash scripts/setup.sh

install: .venv  ## Install core Python deps (skips torch + pytest)
	"$(VENV_PY)" -m pip install -q --upgrade pip
	"$(VENV_PY)" -m pip install -q -e .

install-rl: .venv  ## Install RL extras (torch CPU for kashiwaba-rl agent)
	"$(VENV_PY)" -m pip install -q --extra-index-url https://download.pytorch.org/whl/cpu -e ".[rl]"

install-dev: .venv  ## Install dev extras (pytest, pytest-asyncio)
	"$(VENV_PY)" -m pip install -q -e ".[dev]"

backend:  ## Run FastAPI backend only (port 8000)
	@bash scripts/setup.sh
	"$(VENV_PY)" -m uvicorn orbit_wars_app.main:app --host 127.0.0.1 --port 8000 --reload

viewer:  ## Run Vite dev server only (port 5173)
	pnpm --filter @orbit-wars-lab/viewer dev -- --port 5173 --strictPort

dev:  ## Run backend + viewer in parallel (Ctrl+C stops both)
	@bash scripts/dev.sh

test:  ## Run pytest test suite
	@bash scripts/setup.sh
	"$(VENV_PY)" -m pytest tests/ -v

lint:  ## Run ruff on Python code
	@bash scripts/setup.sh
	"$(VENV_PY)" -m ruff check orbit_wars_app/ tests/

build:  ## Build viewer production bundle (viewer/dist/)
	pnpm --filter @orbit-wars-lab/viewer build

docker-build:  ## Build Docker image (uses public repo's Dockerfile)
	cd ../orbit-wars-lab && docker compose build

docker-up:  ## Start full app in Docker (http://localhost:6001)
	cd ../orbit-wars-lab && docker compose up -d

docker-down:  ## Stop Docker container
	cd ../orbit-wars-lab && docker compose down

sync-public:  ## Sync relevant files -> ../orbit-wars-lab/ (requires commit msg: make sync-public MSG="...")
	@test -n "$(MSG)" || (echo 'Usage: make sync-public MSG="commit message"' && exit 1)
	@bash scripts/sync-public.sh "$(MSG)"

clean:  ## Remove .venv + build artifacts
	rm -rf .venv viewer/dist viewer/node_modules web/core/node_modules .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
