"""
Apollo Clinical Pipeline — FastAPI Application Entry Point
==========================================================
This module is the root of the FastAPI application.  It handles:
  - Application lifespan management (startup / shutdown hooks)
  - CORS middleware (for browser-based UI integration)
  - Request timing middleware (adds X-Process-Time header to every response)
  - Exception handlers (converts unhandled exceptions to structured JSON)
  - Router registration

Running:
    conda activate nlp
    uvicorn final_implementation.api.main:app --reload --host 0.0.0.0 --port 8000

Or directly:
    python final_implementation/api/main.py
"""

from __future__ import annotations

import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "final_implementation"))

from api.routes import router
from config.settings import (
    API_DESCRIPTION,
    API_HOST,
    API_PORT,
    API_RELOAD,
    API_TITLE,
    API_VERSION,
    API_WORKERS,
)
from retrieval.lookup_engine import get_lookup_engine
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load all heavy resources (FAISS index, metadata, embedding model) ONCE
    at startup so that every subsequent request is served without cold-start
    latency.

    In production, this lifespan context ensures graceful startup sequencing
    and clean resource release on shutdown.
    """
    log.info("application_startup_begin", extra={"title": API_TITLE, "version": API_VERSION})

    # Load the lookup engine (FAISS + metadata + embedding model)
    engine = get_lookup_engine()
    try:
        engine.load()
        log.info(
            "application_startup_complete",
            extra={"faiss_vectors": engine._index.ntotal if engine._index else 0},
        )
    except FileNotFoundError as exc:
        log.error(
            "startup_failed_missing_artifacts",
            extra={"error": str(exc)},
        )
        # We still yield so the /health endpoint reports degraded status
        # rather than crashing the entire process on startup.

    yield  # Application is running

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    log.info("application_shutdown")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ------------------------------------------------------------------
    # CORS (allow all in dev; restrict in production)
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Request timing middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time-Ms"] = str(round(elapsed * 1000, 2))
        return response

    # ------------------------------------------------------------------
    # Global exception handler
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error(
            "unhandled_exception",
            extra={
                "path": request.url.path,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please check server logs.",
            },
        )

    # ------------------------------------------------------------------
    # Register routes
    # ------------------------------------------------------------------
    app.include_router(router)

    # ------------------------------------------------------------------
    # Serve the frontend as static files at /app
    # ------------------------------------------------------------------
    frontend_dir = _HERE.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

        @app.get("/", include_in_schema=False)
        async def root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/app/")
    else:
        log.warning("frontend_dir_not_found", extra={"path": str(frontend_dir)})

    return app


app = create_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "final_implementation.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        workers=API_WORKERS,
    )
