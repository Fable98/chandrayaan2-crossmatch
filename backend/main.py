"""
main.py — FastAPI app for the SIH26166 Lunar Image Correspondence backend.

This backend serves pre-computed data from the data/ML teams over HTTP.
It does NOT perform image processing, feature matching, or geometry
computation. Its job is: load JSON from disk → validate with Pydantic →
serve over HTTP with correct CORS headers.
"""

import os
import sys
import logging
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure repository root and backend directory are on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(BACKEND_DIR) in sys.path:
    sys.path.remove(str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR))
if "config" in sys.modules and not hasattr(sys.modules["config"], "settings"):
    sys.modules.pop("config", None)
if "data" in sys.modules and not hasattr(sys.modules["data"], "loader"):
    sys.modules.pop("data", None)

from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger("backend.main")

from fastapi import Depends, FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from data import loader
from routers import triplets, footprint, matches, images, auth, ingest
from routers.auth import get_current_user
from rate_limit import get_limiter
from uploads import (
    ALLOWED_UPLOAD_EXTENSIONS,
    max_upload_bytes,
    purge_expired_runs,
    sanitize_upload_filename,
    save_upload_capped,
)

try:
    from routers import registration as registration_router
except Exception as _reg_exc:
    registration_router = None
    logger.warning("Registration router unavailable: %s", _reg_exc)
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
    f"{CORS_ORIGIN},http://localhost:3000,http://127.0.0.1:3000",
)

allowed_origins_list: list[str] = [
    origin.strip()
    for origin in ALLOWED_ORIGINS.split(",")
    if origin.strip() and origin.strip() != "*"
]
if not allowed_origins_list:
    allowed_origins_list = [
        CORS_ORIGIN,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


# ---------------------------------------------------------------------------
# Lifespan — load data once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all data into memory before the app starts accepting requests."""
    loader.load_all()
    # Step 12: fail-closed secret check + purge stale compute runs on boot.
    try:
        settings.require_jwt_secret()
    except RuntimeError as exc:
        logger.error("SECURITY: %s Auth endpoints will refuse to issue tokens.", exc)
    try:
        purge_expired_runs(loader.DATA_DIR)
    except Exception as exc:
        logger.warning("Startup run purge skipped: %s", exc)
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

# CORS — use explicit origins with credentials and allow Vercel preview deployments.
# A wildcard origin is invalid for credentialed browser requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 12: rate-limit state + handler (limits are declared on /auth/* routes).
limiter = get_limiter()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def block_path_traversal(request: Request, call_next):
    """Step 12: reject any request path that smuggles '..' segments (404).

    Defense-in-depth in front of StaticFiles / FileResponse routers; the
    per-file allowlist gates in uploads.py remain authoritative.
    """
    try:
        raw_path = request.scope.get("path", "") or ""
        if ".." in raw_path.split("/"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
    except Exception:
        pass
    return await call_next(request)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(triplets.router)
app.include_router(footprint.router)
app.include_router(matches.router)
app.include_router(images.router)
app.include_router(auth.router)
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])

if registration_router is not None:
    app.include_router(registration_router.router)


# ---------------------------------------------------------------------------
# Static image mount — /images/{sensor}/{tile_id}
# ---------------------------------------------------------------------------

images_dir = os.path.join(loader.DATA_DIR, "images")
if os.path.isdir(images_dir):
    app.mount("/images", StaticFiles(directory=images_dir), name="images")
else:
    logger.warning("Images directory not found at %s", images_dir)

# Mount for dynamically generated registration outputs
dynamic_runs_dir = os.path.join(loader.DATA_DIR, "dynamic_runs")
os.makedirs(dynamic_runs_dir, exist_ok=True)
app.mount("/dynamic_runs", StaticFiles(directory=dynamic_runs_dir), name="dynamic_runs")


@app.post("/register", response_model=RegisterResponse, tags=["registration"])
async def register_images(
    source_file: UploadFile = File(...),
    reference_file: UploadFile = File(...),
    dem_file: Optional[UploadFile] = File(None),
    source_sensor: str = Form("OHRC"),
    reference_sensor: str = Form("TMC"),
    method: str = Form("cfog"),
    current_user: dict = Depends(get_current_user),
):
    """
    Dynamically register an uploaded source image against an uploaded reference image.
    Requires a valid Bearer token (Step 12).

    Features:
    - 2D Phase Congruency & CFOG structural matching (illumination-robust structural representation).
    - DEM-based relief displacement compensation (when DEM elevation is provided).
    - Multi-scale patch Phase Correlation with empirically benchmarked sub-pixel refinement.
    - Common physical-GSD normalization across multi-resolution sensor pairs.
    - Safety checks: auth, extension allowlist, streamed size cap, traversal-safe
      names, and robust multi-band reading — for source, reference, AND DEM.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
    from matcher_cfog import match_images_cfog, load_as_float_and_color

    # Validate extensions BEFORE touching disk: rejected uploads must never
    # be written (security: no stray .php/.exe payloads in dynamic_runs).
    # DEM uploads go through the exact same gate (Step 12).
    uploads_to_check: list = [source_file, reference_file]
    if dem_file is not None and dem_file.filename:
        uploads_to_check.append(dem_file)
    for file_obj in uploads_to_check:
        sanitize_upload_filename(file_obj.filename, ALLOWED_UPLOAD_EXTENSIONS)

    # Opportunistically purge stale runs (TTL) so the disk cannot fill.
    try:
        purge_expired_runs(loader.DATA_DIR)
    except Exception:
        pass

    run_id = str(uuid.uuid4())
    run_dir = Path(loader.DATA_DIR) / "dynamic_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Fixed server-side stems: only the allowlisted extension is reused, so
    # client filenames (and any '..') never reach the filesystem.
    def _fixed_name(upload: UploadFile, stem: str) -> Path:
        safe = sanitize_upload_filename(upload.filename, ALLOWED_UPLOAD_EXTENSIONS)
        return run_dir / f"{stem}{os.path.splitext(safe)[1].lower()}"

    source_path = _fixed_name(source_file, "source_img")
    ref_path = _fixed_name(reference_file, "reference_img")
    dem_path = None
    output_dir = run_dir / "output"
    output_dir.mkdir(exist_ok=True)

    cap_bytes = max_upload_bytes()
    try:
        await save_upload_capped(source_file, source_path, cap_bytes)
        await save_upload_capped(reference_file, ref_path, cap_bytes)
        if dem_file is not None and dem_file.filename:
            dem_path = _fixed_name(dem_file, "dem_img")
            await save_upload_capped(dem_file, dem_path, cap_bytes)
    except HTTPException:
        # Rejected uploads leave no residue (empty run dirs are removed too).
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    for file_obj, path in [(source_file, source_path), (reference_file, ref_path)]:
        filename = file_obj.filename or "uploaded_image"

        # Validate that image can be parsed by multi-band loader
        try:
            await run_in_threadpool(load_as_float_and_color, path)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read image file: {filename}",
            )
    if dem_path is not None:
        try:
            await run_in_threadpool(load_as_float_and_color, dem_path)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read DEM file: {dem_file.filename if dem_file else 'dem'}",
            )

    try:
        if method.lower() == "loftr":
            from matcher import match_images
            # Heavy CV runs in a threadpool so the async event loop stays
            # responsive (Render free tier serves concurrent dashboard polls).
            result = await run_in_threadpool(
                match_images, str(source_path), str(ref_path), output_dir=str(output_dir)
            )
        else:
            # Default to primary Phase Congruency & CFOG multi-scale matching
            result = await run_in_threadpool(
                match_images_cfog,
                str(source_path),
                str(ref_path),
                dem_path=str(dem_path) if dem_path else None,
                output_dir=str(output_dir),
                source_sensor=source_sensor,
                reference_sensor=reference_sensor,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching pipeline failed: {str(e)}")

    status = result.get("status", "success")
    if status != "success":
        return RegisterResponse(
            status=status,
            message=result.get("message", "Registration failed to verify geometric correspondence."),
            metrics=result.get("metrics"),
            homography=None,
            visual_url=None,
            warped_url=None,
            matches_url=None,
            raster_url=None,
            quiver_url=None,
            metadata=result.get("metadata"),
        )

    # Generate 8-bit web-compatible PNG previews for source and reference so browsers can display any uploaded GeoTIFF / TIFF.
    # All cv2 work runs in a threadpool so the async event loop stays responsive.
    def _make_previews() -> tuple:
        import cv2 as _cv2

        _, src_color, _ = load_as_float_and_color(source_path)
        _, ref_color, _ = load_as_float_and_color(ref_path)
        src_png_path = output_dir / "source_preview.png"
        ref_png_path = output_dir / "reference_preview.png"
        _cv2.imwrite(str(src_png_path), src_color)
        _cv2.imwrite(str(ref_png_path), ref_color)
        return (
            f"/dynamic_runs/{run_id}/output/source_preview.png",
            f"/dynamic_runs/{run_id}/output/reference_preview.png",
        )

    source_url = None
    reference_url = None
    try:
        source_url, reference_url = await run_in_threadpool(_make_previews)
    except Exception as e:
        logger.warning("Could not generate source/reference web previews: %s", e)

    # Generate ISRO PDF report if report_generator is available
    pdf_path = None
    report_url = None
    def _make_report() -> str | None:
        try:
            from report_generator import ISROReportGenerator
        except ImportError:
            try:
                from ML_model.report_generator import ISROReportGenerator
            except Exception:
                return None
        try:
            os.makedirs("reports", exist_ok=True)
            generator = ISROReportGenerator(output_dir="reports/")
            metrics_dict = result.get("metrics") or {}
            out_pdf = generator.generate_report(
                metadata={
                    "run_id": run_id,
                    "source_sensor": source_sensor,
                    "reference_sensor": reference_sensor,
                    "method": method,
                },
                metrics={
                    "rmse": metrics_dict.get("fit_rmse_px", 0.0),
                    "inliers": metrics_dict.get("inlier_count", len(result.get("filtered_ref_pts", []))),
                },
                phases={
                    "phases_executed": result.get("phases_executed", ["CFOG", "SubPixel", "Distribution"]),
                    "phases_failed": result.get("phases_failed", []),
                },
                grid_occupancy=None,
                coverage=metrics_dict.get("combined_coverage_score", 0.7),
                balance=metrics_dict.get("spatial_uniformity", 0.7),
                src_img_path=str(source_path),
                ref_img_path=str(ref_path),
                src_pts=result.get("filtered_src_pts"),
                ref_pts=result.get("filtered_ref_pts"),
                team_name="Team Fable98",
            )
            if isinstance(out_pdf, str) and not out_pdf.startswith("ERROR:"):
                return str(out_pdf)
        except Exception as _rep_err:
            logger.warning("Could not generate ISRO PDF report: %s", _rep_err)
        return None

    try:
        pdf_path = await run_in_threadpool(_make_report)
        if pdf_path:
            report_url = f"/api/registration/report/{run_id}"
    except Exception as e:
        logger.warning("Report generation failed: %s", e)

    # Store run in registration_router.job_manager for /status, /report, and /moon-points
    if registration_router is not None:
        try:
            clean_res = registration_router._result_to_jsonable(dict(result))
            if pdf_path:
                clean_res["pdf_path"] = str(pdf_path)
            clean_res["src_image_path"] = str(source_path)
            clean_res["ref_image_path"] = str(ref_path)
            registration_router.job_manager.create_job(run_id, "registration")
            registration_router.job_manager.update_job(
                run_id,
                status=registration_router.JobStatus.SUCCESS if status == "success" else registration_router.JobStatus.FAILED,
                progress=100.0,
                current_phase="Completed" if status == "success" else "Failed",
                result=clean_res,
            )
            registration_router.job_manager.append_log(
                run_id, f"Run {run_id} completed: status={status}"
            )
        except Exception as _jm_exc:
            logger.warning("Could not store run in job_manager: %s", _jm_exc)

    return RegisterResponse(
        status="success",
        message="Registration verified successfully.",
        metrics=result.get("metrics"),
        homography=result.get("homography"),
        visual_url=f"/dynamic_runs/{run_id}/output/registered_checkerboard.png",
        warped_url=f"/dynamic_runs/{run_id}/output/registered_preview.png",
        source_url=source_url,
        reference_url=reference_url,
        matches_url=f"/dynamic_runs/{run_id}/output/matches.json",
        raster_url=f"/dynamic_runs/{run_id}/output/registered_source.tif",
        quiver_url=(
            f"/dynamic_runs/{run_id}/output/registered_quiver.png"
            if (output_dir / "registered_quiver.png").is_file()
            else None
        ),
        metadata=result.get("metadata"),
        job_id=run_id,
        report_url=report_url,
    )


# ---------------------------------------------------------------------------
# Health & refresh endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["system"])
@app.head("/", tags=["system"])
def root():
    """Root endpoint providing service status, metadata, and docs links."""
    return {
        "status": "online",
        "service": "Chandrayaan-2 Cross-Sensor Image Correspondence API (SIH26166)",
        "version": "0.1.0",
        "triplets_loaded": loader.triplet_count(),
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
@app.head("/health", tags=["system"])
def health_check():
    """Check that the backend is running and report how many triplets are loaded."""
    return HealthResponse(
        status="ok",
        triplets_loaded=loader.triplet_count(),
    )


@app.get("/refresh", response_model=HealthResponse, tags=["system"])
def refresh_data(current_user: dict = Depends(get_current_user)):
    """
    Reload all data from disk. Requires a valid Bearer token (Step 12):
    an unauthenticated refresh lets anyone flush the in-memory cache and
    hammer disk I/O on shared tiers.
    """
    loader.load_all()
    return HealthResponse(
        status="refreshed",
        triplets_loaded=loader.triplet_count(),
    )
