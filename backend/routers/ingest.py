"""
ingest.py — API routes for the Chandrayaan-2 ingest and preparation pipeline.

Handles zip file uploads, initiates the ingest_and_prepare pipeline as an
asynchronous background subprocess, and exposes status / results endpoints.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

try:
    from routers.auth import get_current_user
except ImportError:  # pragma: no cover - direct-router test path
    from backend.routers.auth import get_current_user  # type: ignore

try:
    from uploads import (
        ALLOWED_INGEST_EXTENSIONS,
        max_ingest_file_bytes,
        max_ingest_total_bytes,
        sanitize_upload_filename,
    )
except ImportError:  # pragma: no cover - direct-router test path
    from backend.uploads import (  # type: ignore
        ALLOWED_INGEST_EXTENSIONS,
        max_ingest_file_bytes,
        max_ingest_total_bytes,
        sanitize_upload_filename,
    )

LOG = logging.getLogger("ingest_router")

router = APIRouter()

# Step 12: hard caps so one upload cannot fill the disk or exhaust memory.
# Sized for real PRADAN triplets (OHRC+TMC-2+IIRS ~1.7GB); tunable via
# MAX_INGEST_FILE_MB / MAX_INGEST_TOTAL_MB env without redeploying code.
MAX_INGEST_FILES = 10
# Legacy module constant kept for import compatibility; the live value is
# max_ingest_total_bytes() (env-driven). Do not read this directly.
MAX_INGEST_TOTAL_BYTES = 3072 * 1024 * 1024

# ---------------------------------------------------------------------------
# In-memory job store (suitable for single-server local/preview tool)
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}

# Resolve paths relative to repository root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PIPELINE_ROOT = _REPO_ROOT / "data_preprocessing_pipeline"
_INGEST_SCRIPT = _PIPELINE_ROOT / "scripts" / "ingest_and_prepare.py"
_PROCESSED_TRIPLETS = _PIPELINE_ROOT / "processed_triplets"
_UPLOAD_ROOT = _PIPELINE_ROOT / ".uploads"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class IngestConfig(BaseModel):
    containment: float = 0.8
    tile_size: int = 512
    no_large_aoi: bool = False
    no_invariants: bool = False
    no_matching: bool = False
    no_registration: bool = False
    max_time_gap_days: float | None = None
    require_dates: bool = False


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    stage: str
    progress_pct: float
    started_at: str | None
    completed_at: str | None
    log_lines: list[str]
    error: str | None = None


class JobResult(BaseModel):
    job_id: str
    status: str
    triplets: list[dict[str, Any]]
    summary: str
    output_dir: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_stage_from_log(line: str) -> tuple[str, float]:
    """Extract current stage info from a log line."""
    stages = {
        "Stage 1/7": ("Unzipping & discovering files...", 8.0),
        "Stage 2/7": ("Parsing PDS4 metadata...", 20.0),
        "Stage 3/7": ("Matching triplets...", 33.0),
        "Stage 4/7": ("Processing crops & tiles...", 55.0),
        "Stage 5/7": ("Cross-sensor matching + registration QA...", 72.0),
        "Stage 6/7": ("Updating manifest...", 86.0),
        "Stage 7/7": ("Generating summary...", 95.0),
        # Legacy 6-stage pipeline logs (pre matching/registration stage).
        "Stage 1/6": ("Unzipping & discovering files...", 10.0),
        "Stage 2/6": ("Parsing PDS4 metadata...", 25.0),
        "Stage 3/6": ("Matching triplets...", 40.0),
        "Stage 4/6": ("Processing crops & tiles...", 65.0),
        "Stage 5/6": ("Updating manifest...", 85.0),
        "Stage 6/6": ("Generating summary...", 95.0),
    }
    for marker, (desc, pct) in stages.items():
        if marker in line:
            return desc, pct
    # Sub-step markers inside the matching/registration stage.
    if "cross-sensor matching" in line:
        return "Cross-sensor matching (linked-cursor dots)...", 68.0
    if "registration QA" in line:
        return "Registration QA (blend/checkerboard/quiver)...", 76.0
    return "", -1.0


def _enrich_ingest_triplet(entry: dict[str, Any]) -> dict[str, Any]:
    """Attach data-region parity assets to one ingest result triplet.

    The pipeline's per-run record already carries most of these; this fills
    gaps for legacy/partial runs so every ingested triplet exposes the same
    bundle as curated regions: 512 image URLs, matches + footprint URLs for
    the Linked Cursor view, and registered blend/checkerboard/quiver URLs
    for the cross-grid QA views.
    """
    if not isinstance(entry, dict):
        return entry
    out = dict(entry)
    rid = out.get("region_id")
    if not rid:
        return out
    out.setdefault("images", {
        "ohrc": f"/images/ohrc/{rid}",
        "tmc": f"/images/tmc/{rid}",
        "iirs": f"/images/iirs/{rid}",
        "dem": f"/images/dem/{rid}",
    })
    out.setdefault("matches_url", f"/triplets/{rid}/matches")
    out.setdefault("footprint_url", f"/triplets/{rid}/footprint")
    out.setdefault("registered", {
        "warped": f"/images/registered/{rid}/registered_ohrc.png",
        "blend": f"/images/registered/{rid}/blend_overlay.png",
        "checkerboard": f"/images/registered/{rid}/checkerboard_qa.png",
        "quiver": f"/images/registered/{rid}/displacement_quiver.png",
    })
    return out


async def _run_ingest_job(job_id: str, input_dir: Path, config: IngestConfig):
    """Run ingest_and_prepare.py as a subprocess, capturing output."""
    job = _jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.now(timezone.utc).isoformat()

    cmd = [
        sys.executable,
        str(_INGEST_SCRIPT),
        str(input_dir),
        "--output-dir", str(_PROCESSED_TRIPLETS),
        "--containment", str(config.containment),
        "--tile-size", str(config.tile_size),
        "--verbose",
    ]
    if config.no_large_aoi:
        cmd.append("--no-large-aoi")
    if config.no_invariants:
        cmd.append("--no-invariants")
    if config.no_matching:
        cmd.append("--no-matching")
    if config.no_registration:
        cmd.append("--no-registration")
    if config.max_time_gap_days is not None:
        cmd.extend(["--max-time-gap-days", str(config.max_time_gap_days)])
    if config.require_dates:
        cmd.append("--require-dates")

    LOG.info("Starting ingest job %s: %s", job_id, " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_PIPELINE_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            job["log_lines"].append(line)
            # Keep only last 500 lines in memory
            if len(job["log_lines"]) > 500:
                job["log_lines"] = job["log_lines"][-500:]

            stage_desc, pct = _parse_stage_from_log(line)
            if pct >= 0:
                job["stage"] = stage_desc
                job["progress_pct"] = pct

        retcode = await proc.wait()

        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        if retcode == 0:
            job["status"] = "completed"
            job["progress_pct"] = 100.0
            job["stage"] = "Done!"
            summary_lines = []
            capture = False
            for ln in job["log_lines"]:
                if "INGEST & PREPARE -- SUMMARY" in ln:
                    capture = True
                if capture:
                    summary_lines.append(ln)
            job["summary"] = "\n".join(summary_lines) if summary_lines else "Pipeline completed."

            manifest_path = _PIPELINE_ROOT / "user_triplets.json"
            per_run_path = _PROCESSED_TRIPLETS / ".last_run_triplets.json"
            triplets: list[dict[str, Any]] = []
            # Prefer the per-run triplet list written by this exact job
            # (Stage 5 of ingest_and_prepare.py). Falls back to the filtered
            # manifest history when the file is missing/stale.
            if per_run_path.exists():
                try:
                    per_run = json.loads(per_run_path.read_text(encoding="utf-8"))
                    if isinstance(per_run, dict):
                        if str(per_run.get("job_input_dir")) == str(input_dir):
                            triplets = per_run.get("triplets", [])
                    elif isinstance(per_run, list):
                        triplets = per_run
                except (OSError, ValueError):
                    triplets = []
            if not triplets and manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    triplets = [t for t in json.load(f) if _is_true_triplet(t)]
            job["triplets"] = [_enrich_ingest_triplet(t) for t in triplets]
            # Reload the in-memory catalog so the new region_auto_* folders
            # immediately serve /images, /matches, /footprint like the
            # curated data regions (no manual /refresh round-trip).
            try:
                from data import loader as _loader
                _loader.load_all()
            except Exception as exc:
                LOG.warning("Post-ingest catalog reload skipped: %s", exc)
        else:
            job["status"] = "failed"
            job["error"] = f"Process exited with code {retcode}"
            job["stage"] = "Failed"

    except Exception as exc:
        LOG.exception("Ingest job %s crashed", job_id)
        job["status"] = "failed"
        job["error"] = str(exc)
        job["stage"] = "Failed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()


def _is_true_triplet(entry: dict[str, Any]) -> bool:
    """True OHRC+TMC-2+IIRS triplets only — exclude external LRO_NAC rows.

    user_triplets.json is a mixed manifest: fresh triplets have
    tmc2_product_id / iirs_product_id / overlap_triplet_pct, while
    external_LRO_NAC cross-matches have none of those. The results table
    can only render true triplets, so filter here.
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("reference_type") == "external_LRO_NAC":
        return False
    return bool(entry.get("tmc2_product_id") and entry.get("iirs_product_id"))
@router.post("/upload")
async def upload_and_ingest(
    files: list[UploadFile] = File(...),
    containment: float = Form(0.8),
    tile_size: int = Form(512),
    no_large_aoi: bool = Form(False),
    no_invariants: bool = Form(False),
    no_matching: bool = Form(False),
    no_registration: bool = Form(False),
    max_time_gap_days: float | None = Form(None),
    require_dates: bool = Form(False),
    current_user: dict = Depends(get_current_user),
):
    """Accept zip file uploads, save them, and start the ingest pipeline.

    Step 13: requires a valid Bearer token — this endpoint writes to disk
    and spawns the ingest subprocess, so anonymous uploads are refused.
    Status/results/jobs reads stay public.
    """
    if len(files) > MAX_INGEST_FILES:
        raise HTTPException(
            status_code=413, detail=f"Too many files (max {MAX_INGEST_FILES})."
        )
    job_id = str(uuid.uuid4())[:8]
    upload_dir = _UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Extension gate BEFORE any disk write; traversal-safe fixed names after.
    for f in files:
        sanitize_upload_filename(f.filename, ALLOWED_INGEST_EXTENSIONS)

    per_file_cap = max_ingest_file_bytes()
    total_cap = max_ingest_total_bytes()
    saved_files = []
    total_bytes = 0
    try:
        for i, f in enumerate(files):
            if not f.filename:
                continue
            safe = sanitize_upload_filename(f.filename, ALLOWED_INGEST_EXTENSIONS)
            dest = upload_dir / f"upload_{i}{Path(safe).suffix.lower()}"
            size = 0
            with dest.open("wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    total_bytes += len(chunk)
                    if size > per_file_cap:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{safe}' exceeds the per-file limit of "
                            f"{per_file_cap // (1024 * 1024)}MB.",
                        )
                    if total_bytes > total_cap:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload batch exceeds the total limit of "
                            f"{total_cap // (1024 * 1024)}MB.",
                        )
                    out.write(chunk)
            try:
                await f.close()
            except Exception:
                pass
            saved_files.append(str(dest))
            LOG.info("Saved upload: %s (%d bytes)", dest.name, size)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    if not saved_files:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No files uploaded")

    config = IngestConfig(
        containment=containment,
        tile_size=tile_size,
        no_large_aoi=no_large_aoi,
        no_invariants=no_invariants,
        no_matching=no_matching,
        no_registration=no_registration,
        max_time_gap_days=max_time_gap_days,
        require_dates=require_dates,
    )

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "Uploading files...",
        "progress_pct": 5.0,
        "started_at": None,
        "completed_at": None,
        "log_lines": [f"Uploaded {len(saved_files)} file(s)"],
        "error": None,
        "summary": "",
        "triplets": [],
        "upload_dir": str(upload_dir),
        "config": config.model_dump(),
    }

    asyncio.create_task(_run_ingest_job(job_id, upload_dir, config))

    return {"job_id": job_id, "files_uploaded": len(saved_files)}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll the status of a running ingest job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        stage=job["stage"],
        progress_pct=job["progress_pct"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        log_lines=job.get("log_lines", [])[-50:],
        error=job.get("error"),
    )


@router.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get the final results of a completed ingest job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] not in ("completed", "failed"):
        raise HTTPException(status_code=409, detail="Job is still running")

    return JobResult(
        job_id=job["job_id"],
        status=job["status"],
        triplets=[_enrich_ingest_triplet(t) for t in job.get("triplets", [])],
        summary=job.get("summary", ""),
        output_dir=str(_PROCESSED_TRIPLETS),
    )


@router.get("/jobs")
async def list_jobs():
    """List all ingest jobs."""
    return [
        {
            "job_id": j["job_id"],
            "status": j["status"],
            "stage": j["stage"],
            "progress_pct": j["progress_pct"],
            "started_at": j.get("started_at"),
            "completed_at": j.get("completed_at"),
        }
        for j in _jobs.values()
    ]


class JobDeleteBulk(BaseModel):
    job_ids: list[str]


def _delete_job_entry(job_id: str) -> dict[str, Any]:
    """Remove one finished job + its staging upload dir. Raises HTTPException."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.get("status") in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is still {job.get('status')}; wait for completion first",
        )
    upload_dir = job.get("upload_dir")
    if upload_dir:
        real = os.path.realpath(str(upload_dir))
        allowed = [os.path.realpath(str(_UPLOAD_ROOT)), os.path.realpath(str(_PROCESSED_TRIPLETS))]
        if any(real == a or real.startswith(a + os.sep) for a in allowed):
            shutil.rmtree(real, ignore_errors=True)
    del _jobs[job_id]
    return {"job_id": job_id, "deleted": True}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Delete one ingestion history entry (finished jobs only)."""
    return _delete_job_entry(job_id)


@router.delete("/jobs")
async def delete_jobs_bulk(
    body: JobDeleteBulk, current_user: dict = Depends(get_current_user)
):
    """Delete multiple ingestion history entries (finished jobs only)."""
    deleted: list[str] = []
    errors: dict[str, str] = {}
    seen: set[str] = set()
    for job_id in body.job_ids:
        if job_id in seen:
            continue
        seen.add(job_id)
        try:
            _delete_job_entry(job_id)
            deleted.append(job_id)
        except HTTPException as exc:
            errors[job_id] = str(exc.detail)
    return {"deleted": deleted, "errors": errors}
