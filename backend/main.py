"""
main.py — FastAPI app for the SIH26166 Lunar Image Correspondence backend.

This backend serves pre-computed data from the data/ML teams over HTTP.
It does NOT perform image processing, feature matching, or geometry
computation. Its job is: load JSON from disk → validate with Pydantic →
serve over HTTP with correct CORS headers.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure backend directory is on sys.path for direct imports (data, routers, schemas)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from data import loader
from routers import triplets, footprint, matches, images
from schemas import HealthResponse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORS_ORIGIN: str = os.environ.get("CORS_ORIGIN", "http://localhost:3000")
ALLOWED_ORIGINS: str = os.environ.get(
    "ALLOWED_ORIGINS",
    f"{CORS_ORIGIN},http://localhost:3000,http://127.0.0.1:3000,*",
)

allowed_origins_list: list[str] = [
    origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()
]
if not allowed_origins_list:
    allowed_origins_list = ["*"]


# ---------------------------------------------------------------------------
# Lifespan — load data once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all data into memory before the app starts accepting requests."""
    loader.load_all()
    yield
    # No cleanup needed — data is read-only in-memory dicts


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SIH26166 — Lunar Image Correspondence API",
    description=(
        "Serves Chandrayaan-2 OHRC/TMC/IIRS triplet metadata, footprints, "
        "and LoFTR match correspondences for a Next.js/Leaflet frontend."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow origins from ALLOWED_ORIGINS environment variable + all Vercel previews
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(triplets.router)
app.include_router(footprint.router)
app.include_router(matches.router)
app.include_router(images.router)


# ---------------------------------------------------------------------------
# Static image mount — fallback for direct file paths
# ---------------------------------------------------------------------------

images_dir = os.path.join(loader.DATA_DIR, "images")
if os.path.isdir(images_dir):
    app.mount("/static_images", StaticFiles(directory=images_dir), name="static_images")


# ---------------------------------------------------------------------------
# Health & refresh endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """Check that the backend is running and report how many triplets are loaded."""
    return HealthResponse(
        status="ok",
        triplets_loaded=loader.triplet_count(),
    )


@app.get("/refresh", response_model=HealthResponse, tags=["system"])
def refresh_data():
    """
    Reload all data from disk.

    Use this during the hackathon when the data/ML teams update their output
    files and you want the backend to pick up the changes without a restart.
    """
    loader.load_all()
    return HealthResponse(
        status="refreshed",
        triplets_loaded=loader.triplet_count(),
    )
