"""FastAPI app for Orbit Wars Lab web UI.

Routes:
  GET /api/*         — API endpoints (see orbit_wars_app.api)
  /*                 — Static viewer (built by `pnpm build` in viewer/)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__, api


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """App lifespan — print a friendly ready line on startup (uvicorn's own
    banner is muted via `--log-level warning`), then cleanly shut down the
    scrape executor + match scheduler (kills any in-flight worker processes)
    on exit."""
    # Host port comes from compose's PORT (mapped to the container's 8000);
    # falls back to the documented default if not injected.
    port = os.environ.get("PORT", "6001")
    print(
        f"Running at http://localhost:{port}/#/tournaments",
        flush=True,
    )
    yield
    api._executor.shutdown(wait=False, cancel_futures=True)
    api._shutdown_scheduler()


app = FastAPI(
    title="Orbit Wars Lab",
    description="Local tournament runner + replay viewer",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(api.router)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


# Static viewer (served in production). In dev, Vite serves on :5173 via proxy.
VIEWER_DIST = Path(__file__).parent.parent / "viewer" / "dist"
if VIEWER_DIST.is_dir():
    app.mount("/", StaticFiles(directory=VIEWER_DIST, html=True), name="viewer")
