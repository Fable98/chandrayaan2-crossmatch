"""schemas_registration.py — Pydantic models for the registration API.

Covers async registration jobs, IIRS multimodal jobs, bundle adjustment,
CesiumJS Moon Globe points, and PDF report downloads.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states for a background registration job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class RegistrationRequest(BaseModel):
    """Request a source -> reference registration job."""

    src_image_path: str = Field(..., description="Path to source (moving) image")
    ref_image_path: str = Field(..., description="Path to reference (fixed) image")
    sensor_type: str = Field(default="OHRC", description="Sensor type: OHRC, TMC, or IIRS")
    min_inliers: int = Field(default=50, description="Minimum inliers required")
    run_bundle_adjustment: bool = Field(
        default=False, description="Whether to run global bundle adjustment"
    )


class IIRSRegistrationRequest(BaseModel):
    """Request an IIRS hyperspectral -> OHRC registration job."""

    iirs_image_path: str = Field(..., description="Path to IIRS hyperspectral file")
    ohrc_image_path: str = Field(..., description="Path to OHRC panchromatic image")
    grid_size: int = Field(default=10, description="Grid size for tie point generation")


class BundleAdjustmentRequest(BaseModel):
    """Jointly optimize pairwise constraints with GlobalBundleAdjuster."""

    constraints: List[Dict] = Field(
        ..., description="List of pairwise constraints with img_ids, points, and matrices"
    )
    robust_loss: str = Field(default="huber", description="Robust loss function: huber or cauchy")
    max_iterations: int = Field(default=200, description="Maximum optimization iterations")


class RegistrationResponse(BaseModel):
    """Immediate acknowledgement of an accepted background job."""

    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Pollable status snapshot for a background job."""

    job_id: str
    status: JobStatus
    progress_percent: float = 0.0
    current_phase: str = ""
    result: Optional[Dict] = None
    error: Optional[str] = None


class MoonPoint(BaseModel):
    """One tie point expressed in lunar geographic coordinates."""

    latitude: float
    longitude: float
    altitude: float = 0.0
    confidence: float = 0.0
    pixel_x: float = 0.0
    pixel_y: float = 0.0


class MoonPointsResponse(BaseModel):
    """Tie points for the CesiumJS Moon Globe plus fit quality."""

    job_id: str
    points: List[MoonPoint]
    transformation_matrix: Optional[List[List[float]]] = None
    rmse_pixels: float = 0.0
    rmse_meters: float = 0.0
