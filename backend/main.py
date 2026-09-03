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

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from data import loader
from routers import triplets, footprint, matches, images
from schemas import HealthResponse, RegisterResponse

import shutil
import uuid
import cv2


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
# Static image mount — /images/{sensor}/{tile_id}
# ---------------------------------------------------------------------------

images_dir = os.path.join(loader.DATA_DIR, "images")
if os.path.isdir(images_dir):
    app.mount("/images", StaticFiles(directory=images_dir), name="images")
else:
    print(f"[main] WARNING: images directory not found at {images_dir}")

# Mount for dynamically generated registration outputs
dynamic_runs_dir = os.path.join(loader.DATA_DIR, "dynamic_runs")
os.makedirs(dynamic_runs_dir, exist_ok=True)
app.mount("/dynamic_runs", StaticFiles(directory=dynamic_runs_dir), name="dynamic_runs")


@app.post("/register", response_model=RegisterResponse, tags=["registration"])
async def register_images(
    source_file: UploadFile = File(...),
    reference_file: UploadFile = File(...),
    source_sensor: str = "OHRC",
    reference_sensor: str = "TMC",
):
    """
    Dynamically register an uploaded source image against an uploaded reference image.

    Safety features:
    - Rejects unsupported file types.
    - Rejects files larger than 20MB.
    - Automatically downsizes very large images to prevent CPU/RAM exhaustion.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
    from matcher import match_images

    run_id = str(uuid.uuid4())
    run_dir = Path(loader.DATA_DIR) / "dynamic_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_path = run_dir / "source_img.jpg"
    ref_path = run_dir / "reference_img.jpg"
    output_dir = run_dir / "output"
    output_dir.mkdir(exist_ok=True)

    try:
        with source_path.open("wb") as buffer:
            shutil.copyfileobj(source_file.file, buffer)
        with ref_path.open("wb") as buffer:
            shutil.copyfileobj(reference_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploads: {str(e)}")

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    MAX_FILE_SIZE = 20 * 1024 * 1024
    MAX_DIM = 1500

    for file_obj, path in [(source_file, source_path), (reference_file, ref_path)]:
        filename = file_obj.filename or "uploaded_image"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Allowed: .jpg, .jpeg, .png, .tif, .tiff",
            )

        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File '{filename}' exceeds the 20MB limit.",
            )

        img = cv2.imread(str(path))
        if img is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read image file: {filename}",
            )

        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(path), img)

    try:
        result = match_images(str(source_path), str(ref_path), output_dir=str(output_dir))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching pipeline failed: {str(e)}")

    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=400, detail=str(result["error"]))

    return RegisterResponse(
        status="success",
        metrics=result.get("metrics"),
        homography=result.get("homography"),
        visual_url=f"/dynamic_runs/{run_id}/output/registered_checkerboard.jpg",
        warped_url=f"/dynamic_runs/{run_id}/output/warped_source.jpg",
        matches_url=f"/dynamic_runs/{run_id}/output/matches.json",
    )


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
