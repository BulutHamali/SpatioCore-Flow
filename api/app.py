"""
api/app.py — SpatioCore Flow v2.0 FastAPI application factory.

Usage
-----
    # Development server
    uvicorn api.app:app --reload --port 8080

    # From main.py --ui  (Streamlit + FastAPI running concurrently)
    uvicorn api.app:app --port 8080 &
    streamlit run ui/app.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.state import JobStore
from api.routes.health import router as health_router
from api.routes.pipeline import router as pipeline_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared application state on startup; clean up on shutdown."""
    app.state.job_store = JobStore()
    yield
    # Future: flush any pending DB writes, close connection pools, etc.


def create_app() -> FastAPI:
    """
    Application factory — returns a fully configured FastAPI instance.

    Separating construction from the module-level ``app`` object makes the
    app straightforward to instantiate in tests with a fresh state each time.
    """
    application = FastAPI(
        title="SpatioCore Flow API",
        version="2.0.0",
        description=(
            "Gated multi-agent REST interface for single-cell "
            "and spatial transcriptomics analysis."
        ),
        lifespan=lifespan,
    )

    # ── CORS (permissive for local dev — tighten in production) ──────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    application.include_router(health_router, tags=["health"])
    application.include_router(pipeline_router, tags=["pipeline"])

    return application


# Module-level instance used by uvicorn / main.py
app = create_app()
