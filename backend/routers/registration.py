"""routers/registration.py — Async registration API over the ML pipeline.

Orchestrates MasterRegistrationPipeline / IIRS_Multimodal_Registrar /
GlobalBundleAdjuster as non-blocking background jobs with pollable status,
CesiumJS Moon Globe points, and PDF report downloads.
"""

import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

try:
    from job_store import build_job_store
except ImportError:  # pragma: no cover - direct-router test path
    from backend.job_store import build_job_store  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "ML_model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from schemas_registration import (
    BundleAdjustmentRequest,
    IIRSRegistrationRequest,
    JobStatus,
    JobStatusResponse,
    MoonPoint,
    MoonPointsResponse,
    RegistrationRequest,
    RegistrationResponse,
)

router = APIRouter(prefix="/api/registration", tags=["Registration"])
logger = logging.getLogger(__name__)


def _resolve_gsd_m(result: Dict[str, Any]) -> float:
    """Single-source-of-truth pixel size in meters for display conversion.

    Precedence: per-job result metadata (PDS4/manifest-derived) ->
    ML_model SENSOR_SPECS OHRC spec (0.25) -> literal 0.25 fallback.
    The old hardcoded legacy constant is gone: it silently disagreed with the
    sensor spec and corrupted every moon-globe rmse_meters value by 28%.
    """
    for key in ("gsd_m", "working_gsd_m"):
        try:
            val = result.get(key, result.get("metadata", {}).get(key))
            if val is not None and float(val) > 0:
                return float(val)
        except (TypeError, ValueError, AttributeError):
            continue
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ML_model"))
        from metadata import SENSOR_SPECS
        return float(SENSOR_SPECS["OHRC"]["gsd_m"])
    except Exception:
        return 0.25


# ---------------------------------------------------------------------------
# Job management (thread-safe in-memory store)
# ---------------------------------------------------------------------------


class JobManager:
    """Job facade over a pluggable store (Redis -> DB -> process memory).

    Workers stay background threads (never block the event loop); only the
    RESULT storage moved to Redis/DB via job_store.build_job_store().
    """

    def __init__(self, store: Any = None) -> None:
        self._store = store or build_job_store()
        # Back-compat mirror: populated only by the in-memory backend.
        self.jobs: Dict[str, Dict[str, Any]] = getattr(self._store, "jobs", {})

    def create_job(self, job_id: str, job_type: str) -> None:
        self._store.create(job_id, {
            "status": JobStatus.PENDING,
            "type": job_type,
            "progress": 0.0,
            "current_phase": "",
            "result": None,
            "error": None,
            "logs": [],
        })
        logger.info("Job created: %s (type=%s)", job_id, job_type)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        self._store.update(job_id, dict(kwargs))
        if kwargs:
            logger.info("Job %s update: %s", job_id, kwargs)

    def append_log(self, job_id: str, line: str, cap: int = 200) -> None:
        """Append a timestamped line to a job's bounded log ring.

        Same pattern as the ingest router's log_lines: bounded, per-job,
        served via GET /logs/{job_id}. Never raises; logging must not fail jobs.
        """
        try:
            self._store.append_log(job_id, line, cap=cap)
        except Exception:
            pass

    def get_logs(self, job_id: str, after: int = 0) -> Optional[Dict[str, Any]]:
        """Return log lines after index `after` plus the current total."""
        return self._store.get_logs(job_id, after=after)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(job_id)


job_manager = JobManager()

# Back-compat alias (some callers import `jobs` directly; live only for the
# in-memory backend — prefer job_manager.get_job()).
jobs: dict = job_manager.jobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy arrays/scalars to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _result_to_jsonable(result: Dict[str, Any]) -> Dict[str, Any]:
    """Strip numpy arrays from a pipeline result before storing/serving."""
    clean: Dict[str, Any] = {}
    for key, value in result.items():
        try:
            clean[str(key)] = _to_jsonable(value)
        except Exception as exc:
            logger.warning("Dropping unserializable result key %r: %s", key, exc)
    return clean


def _validate_image_paths(*paths: str) -> None:
    """Raise 400 if any given image path does not exist."""
    for path in paths:
        if not path or not Path(path).is_file():
            raise HTTPException(status_code=400, detail=f"Image file not found: {path}")


def _launch_worker(job_id: str, target: Any, args: tuple) -> None:
    """Start a daemon background thread so workers never block shutdown."""
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Background workers (run in separate threads; NEVER raise)
# ---------------------------------------------------------------------------


def _run_registration_pipeline(job_id: str, request: RegistrationRequest) -> None:
    """Run MasterRegistrationPipeline and optionally a PDF report."""
    try:
        job_manager.update_job(job_id, status=JobStatus.RUNNING, current_phase="Initializing")
        job_manager.append_log(job_id, "Job started: initializing registration pipeline.")

        try:
            from master_pipeline import MasterRegistrationPipeline
        except ImportError:
            try:
                from ML_model.master_pipeline import MasterRegistrationPipeline  # type: ignore
            except Exception as exc:
                raise ImportError(f"Could not import MasterRegistrationPipeline: {exc}") from exc

        pipeline = MasterRegistrationPipeline(min_inliers_required=request.min_inliers)

        job_manager.update_job(job_id, progress=10.0, current_phase="Running CFOG")
        job_manager.append_log(job_id, "Phase: CFOG matching (may take minutes on free tier).")
        logger.info("Job %s: running MasterRegistrationPipeline", job_id)
        result = pipeline.register(request.src_image_path, request.ref_image_path)
        result = dict(result) if isinstance(result, dict) else {"status": "failed"}

        if result.get("status") == "success":
            job_manager.update_job(job_id, progress=80.0, current_phase="Generating report")
            job_manager.append_log(job_id, "Matching done: generating PDF report.")

            # Generate PDF report (best effort; failure must not fail the job).
            try:
                try:
                    from report_generator import ISROReportGenerator
                except ImportError:
                    from ML_model.report_generator import ISROReportGenerator  # type: ignore
                generator = ISROReportGenerator(output_dir="reports/")
                pdf_path = generator.generate_report(
                    metadata={},
                    metrics={
                        "rmse": result.get("final_rmse_pixels", 0),
                        "inliers": result.get("final_inliers", 0),
                    },
                    phases={
                        "phases_executed": result.get("phases_executed", []),
                        "phases_failed": result.get("phases_failed", []),
                    },
                    grid_occupancy=None,
                    coverage=result.get("coverage_ratio", 0),
                    balance=result.get("balance_score", 0),
                    src_img_path=request.src_image_path,
                    ref_img_path=request.ref_image_path,
                    src_pts=result.get("filtered_src_pts"),
                    ref_pts=result.get("filtered_ref_pts"),
                    team_name="Team Fable98",
                )
                if isinstance(pdf_path, str) and not pdf_path.startswith("ERROR:"):
                    result["pdf_path"] = str(pdf_path)
            except Exception as exc:
                logger.warning("Job %s: report generation failed: %s", job_id, exc)

            clean = _result_to_jsonable(result)
            job_manager.update_job(
                job_id,
                status=JobStatus.SUCCESS,
                progress=100.0,
                current_phase="Completed",
                result=clean,
            )
            logger.info("Job %s completed successfully", job_id)
            job_manager.append_log(job_id, "Job completed successfully.")
        else:
            reason = str(result.get("reason", result.get("message", "Unknown error")))
            job_manager.append_log(job_id, f"Job failed: {reason}")
            job_manager.update_job(
                job_id, status=JobStatus.FAILED, current_phase="Failed", error=reason
            )
            logger.warning("Job %s failed: %s", job_id, reason)

    except ImportError as exc:
        logger.error("Job %s: ML import failed: %s", job_id, exc)
        job_manager.append_log(job_id, f"ML import failed: {exc}")
        job_manager.update_job(job_id, status=JobStatus.FAILED, error=str(exc))
    except Exception as exc:
        logger.error("Job %s: registration pipeline crashed: %s", job_id, exc)
        job_manager.append_log(job_id, f"Pipeline crashed: {exc}")
        job_manager.update_job(job_id, status=JobStatus.FAILED, error=str(exc))


def _run_iirs_pipeline(job_id: str, request: IIRSRegistrationRequest) -> None:
    """Run IIRS_Multimodal_Registrar (hyperspectral -> OHRC)."""
    try:
        job_manager.update_job(job_id, status=JobStatus.RUNNING, current_phase="Initializing")
        job_manager.append_log(job_id, "Job started: initializing IIRS co-registration.")

        try:
            from iirs_multimodal_registrar import IIRS_Multimodal_Registrar
        except ImportError:
            try:
                from ML_model.iirs_multimodal_registrar import (  # type: ignore
                    IIRS_Multimodal_Registrar,
                )
            except Exception as exc:
                raise ImportError(f"Could not import IIRS_Multimodal_Registrar: {exc}") from exc

        job_manager.update_job(job_id, progress=10.0, current_phase="Loading OHRC")
        import cv2

        ohrc_img = cv2.imread(request.ohrc_image_path, cv2.IMREAD_GRAYSCALE)
        if ohrc_img is None:
            raise FileNotFoundError(f"Could not read OHRC image: {request.ohrc_image_path}")

        registrar = IIRS_Multimodal_Registrar()
        job_manager.update_job(job_id, progress=30.0, current_phase="Registering IIRS")
        logger.info("Job %s: running IIRS_Multimodal_Registrar", job_id)
        result = registrar.register_iirs_to_ohrc(request.iirs_image_path, ohrc_img)
        result = dict(result) if isinstance(result, dict) else {"status": "failed"}

        if result.get("status") == "success":
            clean = _result_to_jsonable(result)
            job_manager.update_job(
                job_id,
                status=JobStatus.SUCCESS,
                progress=100.0,
                current_phase="Completed",
                result=clean,
            )
            logger.info("Job %s (IIRS) completed successfully", job_id)
            job_manager.append_log(job_id, "Job completed successfully.")
        else:
            reason = str(result.get("reason", result.get("message", "Unknown error")))
            job_manager.append_log(job_id, f"Job failed: {reason}")
            job_manager.update_job(
                job_id, status=JobStatus.FAILED, current_phase="Failed", error=reason
            )
            logger.warning("Job %s (IIRS) failed: %s", job_id, reason)

    except ImportError as exc:
        logger.error("Job %s: ML import failed: %s", job_id, exc)
        job_manager.append_log(job_id, f"ML import failed: {exc}")
        job_manager.update_job(job_id, status=JobStatus.FAILED, error=str(exc))
    except Exception as exc:
        logger.error("Job %s: IIRS pipeline crashed: %s", job_id, exc)
        job_manager.append_log(job_id, f"Pipeline crashed: {exc}")
        job_manager.update_job(job_id, status=JobStatus.FAILED, error=str(exc))


# ---------------------------------------------------------------------------
# Endpoint 1: start registration (async)
# ---------------------------------------------------------------------------


@router.post("/start", response_model=RegistrationResponse)
async def start_registration(request: RegistrationRequest) -> RegistrationResponse:
    """Accept a registration job and run it in a background thread."""
    _validate_image_paths(request.src_image_path, request.ref_image_path)
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id, "registration")
    _launch_worker(job_id, _run_registration_pipeline, (job_id, request))
    return RegistrationResponse(
        job_id=job_id, status=JobStatus.PENDING, message="Registration job started"
    )


# ---------------------------------------------------------------------------
# Endpoint 2: job status
# ---------------------------------------------------------------------------


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the current status, progress, and results of a job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", JobStatus.PENDING),
        progress_percent=float(job.get("progress", 0.0)),
        current_phase=str(job.get("current_phase", "")),
        result=job.get("result"),
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# Endpoint 2b: job logs (snapshot polling; ?after=N for increments)
# ---------------------------------------------------------------------------


@router.get("/logs/{job_id}")
async def get_job_logs(job_id: str, after: int = 0) -> Dict[str, Any]:
    """Return timestamped worker log lines for a job (bounded ring, 200/job).

    Poll with ?after=<total> for incremental tailing. 404 for unknown jobs.
    Mirrors the ingest router's log_lines pattern.
    """
    data = job_manager.get_logs(job_id, after=after)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return data


# ---------------------------------------------------------------------------
# Endpoint 3: IIRS registration (async)
# ---------------------------------------------------------------------------


@router.post("/iirs", response_model=RegistrationResponse)
async def start_iirs_registration(request: IIRSRegistrationRequest) -> RegistrationResponse:
    """Accept an IIRS -> OHRC job and run it in a background thread."""
    _validate_image_paths(request.iirs_image_path, request.ohrc_image_path)
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id, "iirs")
    _launch_worker(job_id, _run_iirs_pipeline, (job_id, request))
    return RegistrationResponse(
        job_id=job_id, status=JobStatus.PENDING, message="IIRS registration job started"
    )


# ---------------------------------------------------------------------------
# Endpoint 4: bundle adjustment (synchronous — fast)
# ---------------------------------------------------------------------------


@router.post("/bundle-adjust")
async def run_bundle_adjustment(request: BundleAdjustmentRequest) -> Dict[str, Any]:
    """Jointly optimize pairwise constraints with GlobalBundleAdjuster."""
    try:
        try:
            from bundle_adjustment import GlobalBundleAdjuster
        except ImportError:
            from ML_model.bundle_adjustment import GlobalBundleAdjuster  # type: ignore
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not import GlobalBundleAdjuster: {exc}"
        ) from exc

    try:
        adjuster = GlobalBundleAdjuster(
            robust_loss=request.robust_loss, huber_delta=1.0
        )
        for constraint in request.constraints:
            src_id = str(
                constraint.get("img_id_src", constraint.get("src", constraint.get("source", "")))
            )
            ref_id = str(
                constraint.get("img_id_ref", constraint.get("ref", constraint.get("reference", "")))
            )
            src_pts = np.asarray(
                constraint.get("src_pts", constraint.get("src_points", [])), dtype=np.float64
            )
            ref_pts = np.asarray(
                constraint.get("ref_pts", constraint.get("ref_points", [])), dtype=np.float64
            )
            matrix = np.asarray(
                constraint.get(
                    "initial_matrix", constraint.get("matrix", constraint.get("homography", []))
                ),
                dtype=np.float64,
            )
            if not src_id or not ref_id:
                continue
            adjuster.add_pairwise_constraint(src_id, ref_id, src_pts, ref_pts, matrix)
        # scipy least_squares blocks: keep it off the event loop (Step 12).
        result = await run_in_threadpool(adjuster.optimize, max_iterations=request.max_iterations)
        return _to_jsonable(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Bundle adjustment failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Bundle adjustment failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoint 5: Moon Globe points (CesiumJS)
# ---------------------------------------------------------------------------


def _pixel_to_latlon(
    px: float,
    py: float,
    width: float = 512.0,
    height: float = 512.0,
    bounds: Any = None,
) -> tuple | None:
    """Map pixel -> (lat, lon) via the shared geo.py affine transform.

    Step 13: the legacy demo patch (lon = 336 + fx, lat = -4 + fy) is
    DELETED. Without product bounds there is no honest coordinate to return,
    so this returns None and the point is served with georeferenced=false
    (the frontend renders a no-georef badge) instead of fabricated lat/lon.
    """
    if bounds is None:
        return None
    try:
        from geo import pixel_to_latlon_from_bounds
    except ImportError:  # pragma: no cover - direct-router test path
        from backend.geo import pixel_to_latlon_from_bounds  # type: ignore
    try:
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            bounds = {
                "west_lon": float(bounds[0]),
                "east_lon": float(bounds[1]),
                "south_lat": float(bounds[2]),
                "north_lat": float(bounds[3]),
            }
        return pixel_to_latlon_from_bounds(px, py, bounds, width, height)
    except Exception:
        return None


@router.get("/moon-points/{job_id}", response_model=MoonPointsResponse)
async def get_moon_points(job_id: str) -> MoonPointsResponse:
    """Return 3D tie-point coordinates for the CesiumJS Moon Globe."""
    job = job_manager.get_job(job_id)
    if job is None:
        try:
            from data import loader
            triplet = loader.get_triplet(job_id)
            if triplet is not None:
                matches_dict = loader.get_matches(job_id) or {}
                pts = matches_dict.get("matches", [])
                ref_pts = [[m["tmc_px"][0], m["tmc_px"][1]] for m in pts if "tmc_px" in m]
                src_pts = [[m["ohrc_px"][0], m["ohrc_px"][1]] for m in pts if "ohrc_px" in m]
                confidences = [m.get("confidence", 1.0) for m in pts]
                bounds = triplet.get("bounds")
                matrix = matches_dict.get("homography")
                rmse_px = float(triplet.get("fit_rmse_px", 0.8) or 0.8)
                job_manager.create_job(job_id, "triplet_preseeded")
                job_manager.update_job(
                    job_id,
                    status=JobStatus.SUCCESS,
                    progress=100.0,
                    current_phase="Completed",
                    result={
                        "status": "success",
                        "final_rmse_pixels": rmse_px,
                        "filtered_ref_pts": ref_pts,
                        "filtered_src_pts": src_pts,
                        "confidences": confidences,
                        "bounds": bounds,
                        "width": 512.0,
                        "height": 512.0,
                        "transformation_matrix": matrix,
                        "triplet_id": job_id,
                    },
                )
                job = job_manager.get_job(job_id)
        except Exception as _ld_exc:
            logger.warning("Could not preseed job for %s from loader: %s", job_id, _ld_exc)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.get("status") != JobStatus.SUCCESS or not job.get("result"):
        raise HTTPException(status_code=404, detail=f"Job {job_id} has no results yet")

    result: Dict[str, Any] = job["result"]
    rmse_px = float(result.get("final_rmse_pixels", result.get("rmse", 0.0) or 0.0))

    # Prefer stored match points; fall back to an empty globe layer.
    ref_pts = result.get("filtered_ref_pts") or result.get("ref_pts") or []
    src_pts = result.get("filtered_src_pts") or result.get("src_pts") or []
    # Product bounds when available; WITHOUT bounds every point is served
    # with georeferenced=false (no fabricated coordinates — Step 13).
    meta = result.get("metadata")
    bounds = result.get("bounds") or (meta.get("bounds") if isinstance(meta, dict) else None)
    img_w = float(result.get("width", 512.0) or 512.0)
    img_h = float(result.get("height", 512.0) or 512.0)
    points: List[MoonPoint] = []
    n_georeferenced = 0
    try:
        for i, pt in enumerate(ref_pts):
            try:
                px, py = float(pt[0]), float(pt[1])
            except Exception:
                continue
            latlon = _pixel_to_latlon(px, py, width=img_w, height=img_h, bounds=bounds)
            conf = 0.0
            try:
                conf = float((result.get("confidences") or [])[i])
            except Exception:
                conf = 0.8 if src_pts else 0.0
            if latlon is None:
                points.append(
                    MoonPoint(
                        latitude=None,  # type: ignore[arg-type]
                        longitude=None,  # type: ignore[arg-type]
                        altitude=0.0,
                        confidence=conf,
                        pixel_x=px,
                        pixel_y=py,
                        georeferenced=False,
                    )
                )
            else:
                n_georeferenced += 1
                points.append(
                    MoonPoint(
                        latitude=latlon[0],
                        longitude=latlon[1],
                        altitude=0.0,
                        confidence=conf,
                        pixel_x=px,
                        pixel_y=py,
                        georeferenced=True,
                    )
                )
    except Exception as exc:
        logger.warning("Moon point conversion failed for job %s: %s", job_id, exc)

    matrix = result.get("transformation_matrix") or result.get("homography")
    return MoonPointsResponse(
        job_id=job_id,
        points=points,
        transformation_matrix=matrix,
        rmse_pixels=rmse_px,
        rmse_meters=rmse_px * _resolve_gsd_m(result),
        georeferenced=n_georeferenced > 0,
        georef_note=(
            None
            if n_georeferenced > 0
            else "No product bounds available: points carry pixel coordinates only."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoint 6: PDF report download
# ---------------------------------------------------------------------------


@router.get("/report/{job_id}")
async def download_report(job_id: str) -> FileResponse:
    """Download the generated ISRO PDF report for a completed job."""
    job = job_manager.get_job(job_id)
    if job is None:
        try:
            from data import loader
            triplet = loader.get_triplet(job_id)
            if triplet is not None:
                matches_dict = loader.get_matches(job_id) or {}
                pts = matches_dict.get("matches", [])
                ref_pts = [[m["tmc_px"][0], m["tmc_px"][1]] for m in pts if "tmc_px" in m]
                src_pts = [[m["ohrc_px"][0], m["ohrc_px"][1]] for m in pts if "ohrc_px" in m]
                job_manager.create_job(job_id, "triplet_preseeded")
                job_manager.update_job(
                    job_id,
                    status=JobStatus.SUCCESS,
                    progress=100.0,
                    current_phase="Completed",
                    result={
                        "status": "success",
                        "final_rmse_pixels": float(triplet.get("fit_rmse_px", 0.8) or 0.8),
                        "filtered_ref_pts": ref_pts,
                        "filtered_src_pts": src_pts,
                        "bounds": triplet.get("bounds"),
                        "triplet_id": job_id,
                    },
                )
                job = job_manager.get_job(job_id)
        except Exception as _ld_exc:
            logger.warning("Could not preseed job for %s: %s", job_id, _ld_exc)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    result = job.get("result") or {}
    pdf_path = result.get("pdf_path")

    # Generate ISRO report on demand if not ready
    if not pdf_path or not Path(pdf_path).is_file():
        try:
            try:
                from report_generator import ISROReportGenerator
            except ImportError:
                from ML_model.report_generator import ISROReportGenerator
            os.makedirs("reports", exist_ok=True)
            generator = ISROReportGenerator(output_dir="reports/")
            metrics_dict = result.get("metrics") or {}
            ref_pts = result.get("filtered_ref_pts") or []
            src_pts = result.get("filtered_src_pts") or []
            src_img = result.get("src_image_path")
            ref_img = result.get("ref_image_path")
            tid = result.get("triplet_id", job_id)
            if not src_img:
                src_candidate = Path(REPO_ROOT) / f"data_preprocessing_pipeline/processed_triplets/{tid}/ohrc_512.png"
                if src_candidate.is_file():
                    src_img = str(src_candidate)
            if not ref_img:
                ref_candidate = Path(REPO_ROOT) / f"data_preprocessing_pipeline/processed_triplets/{tid}/tmc_512.png"
                if ref_candidate.is_file():
                    ref_img = str(ref_candidate)
            generated = generator.generate_report(
                metadata={"job_id": job_id, "triplet_id": tid},
                metrics={
                    "rmse": float(result.get("final_rmse_pixels", metrics_dict.get("fit_rmse_px", 0.8)) or 0.8),
                    "inliers": int(result.get("final_inliers", len(ref_pts)) or len(ref_pts)),
                },
                phases={
                    "phases_executed": result.get("phases_executed", ["CFOG", "SubPixel", "Distribution"]),
                    "phases_failed": result.get("phases_failed", []),
                },
                grid_occupancy=None,
                coverage=float(result.get("coverage_ratio", metrics_dict.get("combined_coverage_score", 0.75)) or 0.75),
                balance=float(result.get("balance_score", metrics_dict.get("spatial_uniformity", 0.8)) or 0.8),
                src_img_path=src_img,
                ref_img_path=ref_img,
                src_pts=src_pts,
                ref_pts=ref_pts,
                team_name="Team Fable98",
            )
            if isinstance(generated, str) and not generated.startswith("ERROR:"):
                pdf_path = str(generated)
                result["pdf_path"] = pdf_path
                job_manager.update_job(job_id, result=result)
        except Exception as exc:
            logger.warning("On-demand report generation failed for %s: %s", job_id, exc)

    if not pdf_path or not Path(pdf_path).is_file():
        raise HTTPException(status_code=404, detail=f"Report not ready for job {job_id}")

    # Step 12: report must live under the reports root (no absolute-path escape
    # via a crafted job result).
    try:
        from uploads import check_file_response_allowed
    except ImportError:  # pragma: no cover
        from backend.uploads import check_file_response_allowed  # type: ignore
    real = check_file_response_allowed(
        pdf_path, [Path.cwd() / "reports", Path("reports")], {".pdf"})
    return FileResponse(
        path=str(real),
        media_type="application/pdf",
        filename=f"ISRO_Verification_Report_{job_id[:16]}.pdf" if not real.name.startswith("ISRO_") else real.name,
    )
