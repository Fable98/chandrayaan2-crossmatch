"""
schemas.py — Pydantic response models for the SIH26166 backend.

COORDINATE & BOUNDS CONVENTION:
  All sensors (OHRC, TMC-2, IIRS, and stereo-derived DEM) in a triplet share
  the exact same bounding box (TripletBounds) and 512×512 pixel grid by design.
  OHRC extent defines the common crop boundary; TMC-2, IIRS, and DEM are spatially
  cropped and reprojected into this identical bounding box during preprocessing.

  Longitude uses the standard lunar planetocentric 0–360° convention (e.g. 336.48°, 179.92°).
  Latitude uses standard planetocentric degrees (e.g. -3.41°, -45.09°).
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bounds & coordinate models
# ---------------------------------------------------------------------------

class TripletBounds(BaseModel):
    """
    Shared bounding box for all sensors in a triplet.

    In the real data pipeline, OHRC, TMC-2, IIRS, and DEM are all cropped and
    resampled to this exact same extent and a 512×512 pixel grid.
    """
    west_lon: float
    east_lon: float
    south_lat: float
    north_lat: float


class Coordinate(BaseModel):
    """Legacy single lat/lon point (kept for backwards compatibility)."""
    lat: float
    lon: float


class Footprint(BaseModel):
    """Legacy 4-corner footprint (deprecated in favor of TripletBounds)."""
    top_left: Coordinate
    top_right: Coordinate
    bottom_right: Coordinate
    bottom_left: Coordinate


# ---------------------------------------------------------------------------
# Sensor & triplet models
# ---------------------------------------------------------------------------

class SensorMeta(BaseModel):
    """Per-sensor metadata within a triplet."""
    sensor: str  # "ohrc", "tmc", "iirs", "dem"
    gsd_m: float
    sun_elevation_deg: float | None = None
    sun_azimuth_deg: float | None = None
    incidence_angle_deg: float | None = None


class TripletSummary(BaseModel):
    """
    Full metadata for one region (triplet of OHRC + TMC + IIRS tiles, plus DEM if available).
    """
    id: str
    bounds: TripletBounds
    sensors: list[SensorMeta]
    ohrc_product_id: str | None = None
    tmc2_product_id: str | None = None
    iirs_product_id: str | None = None
    dem_available: bool = False
    dem_url: str | None = None


class TripletListResponse(BaseModel):
    """Response for GET /triplets."""
    triplets: list[TripletSummary]


# ---------------------------------------------------------------------------
# Footprint & overlay responses
# ---------------------------------------------------------------------------

class FootprintResponse(BaseModel):
    """
    Response for GET /triplets/{id}/footprint.

    Returns the shared bounding box common to all sensors in this triplet.
    """
    triplet_id: str
    bounds: TripletBounds


class IIRSOverlay(BaseModel):
    """
    Response for GET /triplets/{id}/iirs-overlay.

    Returns the shared bounding box and image asset URL for the IIRS tile.
    """
    triplet_id: str
    image_url: str
    bounds: TripletBounds
    opacity_hint: float = 0.6


# ---------------------------------------------------------------------------
# Match models
# ---------------------------------------------------------------------------

class MatchPoint(BaseModel):
    """
    A single OHRC ↔ TMC-2 pixel correspondence (post-RANSAC from LoFTR).

    Pixel coordinates come directly from the ML team's matches.json
    (image1_x/y → ohrc_px, image2_x/y → tmc_px).
    Geographic coordinates are computed at load time by the backend
    using the shared affine transform derived from TripletBounds.
    """
    ohrc_px: tuple[float, float]      # (x, y) in OHRC 512×512 pixel space
    tmc_px: tuple[float, float]       # (x, y) in TMC 512×512 pixel space
    ohrc_latlon: tuple[float, float]  # (lat, lon) — computed from shared TripletBounds
    tmc_latlon: tuple[float, float]   # (lat, lon) — computed from shared TripletBounds
    confidence: float


class MatchMetrics(BaseModel):
    """Evaluation metrics for registration accuracy and distribution."""
    num_inliers: int
    num_raw_matches: int = 0
    inlier_ratio: float = 0.0
    rmse_px: float = 0.0
    mean_reprojection_error_px: float = 0.0
    median_reprojection_error_px: float = 0.0
    max_reprojection_error_px: float = 0.0
    sub_pixel_accurate: bool = False
    fraction_below_1px: float = 0.0
    source_coverage_ratio: float = 0.0
    destination_coverage_ratio: float = 0.0
    combined_coverage_score: float = 0.0


class MatchesResponse(BaseModel):
    """Response for GET /triplets/{id}/matches."""
    triplet_id: str
    num_matches: int
    homography: list[list[float]] | None  # 3×3 matrix, re-derived from inlier points; null if < 4 points
    metrics: MatchMetrics | None = None
    matches: list[MatchPoint]


# ---------------------------------------------------------------------------
# Health / refresh
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str
    triplets_loaded: int


class RegisterRequest(BaseModel):
    source_sensor: str
    reference_sensor: str

class RegisterResponse(BaseModel):
    status: str
    metrics: dict | None = None
    homography: list[list[float]] | None = None
    visual_url: str | None = None
    warped_url: str | None = None
    matches_url: str | None = None
