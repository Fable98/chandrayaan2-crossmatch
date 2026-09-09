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

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

LOG = logging.getLogger("ingest_router")

router = APIRouter()

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
    return "", -1.0


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
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    job["triplets"] = json.load(f)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/upload")
async def upload_and_ingest(
    files: list[UploadFile] = File(...),
    containment: float = Form(0.8),
    tile_size: int = Form(512),
    no_large_aoi: bool = Form(False),
    no_invariants: bool = Form(False),
    max_time_gap_days: float | None = Form(None),
    require_dates: bool = Form(False),
):
    """Accept zip file uploads, save them, and start the ingest pipeline."""
    job_id = str(uuid.uuid4())[:8]
    upload_dir = _UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for f in files:
        if f.filename:
            dest = upload_dir / f.filename
            with dest.open("wb") as out:
                content = await f.read()
                out.write(content)
            saved_files.append(str(dest))
            LOG.info("Saved upload: %s (%d bytes)", dest.name, len(content))

    if not saved_files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    config = IngestConfig(
        containment=containment,
        tile_size=tile_size,
        no_large_aoi=no_large_aoi,
        no_invariants=no_invariants,
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
        triplets=job.get("triplets", []),
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
