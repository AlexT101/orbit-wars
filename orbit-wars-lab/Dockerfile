# syntax=docker/dockerfile:1.7

# ==============================================================================
# Stage 1 — Viewer build (Vite / pnpm).
# ==============================================================================
FROM node:20-alpine AS viewer-build

RUN corepack enable
WORKDIR /app

# Workspace manifests first so the layer cache works for repeated rebuilds.
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY viewer/package.json viewer/
COPY web/core/package.json web/core/

RUN pnpm install --frozen-lockfile

# Source for viewer + the vendored Kaggle core that viewer imports.
COPY viewer/ viewer/
COPY web/ web/

# Build all workspace packages → viewer/dist (served as static by the Python
# backend). Recursive build is needed because the viewer imports
# @kaggle-environments/core/dist/style.css, and web/core/dist/ is gitignored
# so a fresh clone has nothing there until web/core builds first. Recursive
# build resolves the dependency order automatically.
RUN pnpm -r build

# ==============================================================================
# Stage 2 — Python dep builder.
#
# Compiles every Python dependency into a wheel, then the runtime stage copies
# just the wheels + installs them without ever seeing a C compiler. Saves
# ~300 MB off the final image vs. installing build-essential into runtime.
# ==============================================================================
FROM python:3.12-slim AS py-build

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# ==============================================================================
# Stage 2b — Rust engine wheel.
#
# Builds the PyO3-backed `orbit_wars_rust` crate into a Linux wheel via
# maturin so the runtime stage can `pip install` it like any other dep.
# Powers ultrafast tournament mode; without this, ultrafast matches would
# fail to load the native module inside the container.
# ==============================================================================
FROM rust:1-slim-bookworm AS rust-build

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so maturin's deps don't get blocked by Debian's PEP 668 mark.
RUN python3 -m venv /opt/maturin-venv \
    && /opt/maturin-venv/bin/pip install --no-cache-dir 'maturin>=1.7,<2.0'

WORKDIR /build/rust_engine
# Build context for this stage is the rust_engine/ dir (see compose.yml's
# `additional_contexts: rust=../rust_engine`).
COPY --from=rust . .
RUN /opt/maturin-venv/bin/maturin build --release --out /wheels-rust

# ==============================================================================
# Stage 3 — Python runtime (FastAPI + tournament runner).
# ==============================================================================
FROM python:3.12-slim

# `git` is still needed at runtime ONLY so `pip install -e .` below can parse
# the kaggle-environments VCS ref embedded in pyproject.toml's `dependencies`.
# No build-essential — wheels are already built.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=py-build /wheels /wheels
COPY --from=rust-build /wheels-rust /wheels-rust
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && pip install --no-cache-dir /wheels-rust/*.whl \
    && rm -rf /wheels /wheels-rust

# Application code + zoo + leaderboard seed.
COPY orbit_wars_app/ orbit_wars_app/
COPY --from=agents . agents/
COPY runs/ runs/

# Register the package + console_scripts entry points from pyproject.toml
# (`orbit-wars-tournament` CLI). --no-deps because deps are already installed.
RUN pip install --no-cache-dir --no-deps -e .

# Prebuilt viewer bundle from stage 1 — backend mounts this on '/'.
COPY --from=viewer-build /app/viewer/dist viewer/dist

# Non-root user so files written to the mounted ./agents and ./runs on the
# host match the host's user instead of coming out root-owned. compose.yml
# overrides UID/GID to the host user's (defaulting to 1000 which matches
# most Linux installs; macOS users set UID=501 in a .env file).
RUN useradd --create-home --uid 1000 --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health').read()" || exit 1
CMD ["uvicorn", "orbit_wars_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--log-level", "warning"]
