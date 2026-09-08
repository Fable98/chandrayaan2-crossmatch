"""
Lunar Crossmatch -- FastAPI backend.

Provides an API for the frontend to trigger the ingest-and-prepare pipeline
and monitor its progress.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes import ingest

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Lunar Crossmatch API",
    version="0.1.0",
    description="Backend for the Chandrayaan-2 cross-sensor matching pipeline",
)

# -- CORS (allow the Vite dev server) --------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Routers ----------------------------------------------------------------
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])


# -- Health check -----------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


# -- Serve processed_triplets as static files (for result previews) ---------
_TRIPLETS_DIR = Path(
    os.getenv(
        "PROCESSED_TRIPLETS_DIR",
        str(
            Path(__file__).resolve().parents[3]
            / "data_preprocessing_pipeline"
            / "processed_triplets"
        ),
    )
)
if _TRIPLETS_DIR.exists():
    app.mount(
        "/static/triplets",
        StaticFiles(directory=str(_TRIPLETS_DIR)),
        name="triplets",
    )
