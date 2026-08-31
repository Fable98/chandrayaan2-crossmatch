"""
schemas.py — Pydantic response models for the SIH26166 backend.

COORDINATE CONVENTION (used across all endpoints):
  All geographic coordinates use {"lat": float, "lon": float} objects.
  Footprint corners are always named (top_left, top_right, bottom_right,
  bottom_left) in CLOCKWISE winding order — directly consumable by Leaflet's
  L.polygon() without reordering.

  NEVER derive a bounding box via min()/max() on coordinate lists
  for ANY sensor — including IIRS. Verified via scripts/check_iirs_rotation.py:
  both region_001 and region_002 IIRS footprints are rotated quads (2.4-2.7%
  area loss from bbox derivation). All sensors use the same 4-corner Footprint
  model throughout.
"""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Coordinate primitives
# ---------------------------------------------------------------------------

class Coordinate(BaseModel):
    """A single lat/lon point on the lunar surface."""
    lat: float
    lon: float


class Footprint(BaseModel):
    """
    Four named corners of an image footprint, clockwise from top-left.

    IMPORTANT: These corners come verbatim from the manifest. They represent a
    real rotated quad on the lunar surface — NOT an axis-aligned bounding box.
    Do NOT replace this with min/max corners; that introduced a visible
    misalignment bug in a previous iteration.
    """
    top_left: Coordinate
    top_right: Coordinate
    bottom_right: Coordinate
    bottom_left: Coordinate


class IIRSOverlay(BaseModel):
    """
    Response for GET /triplets/{id}/iirs-overlay.

    IIRS footprint is returned as a proper 4-corner quad (same Footprint
    model as OHRC/TMC-2), NOT an axis-aligned bounding box. Verified via
    scripts/check_iirs_rotation.py: both region_001 and region_002 have
    rotated IIRS quads (bottom lons differ from top lons). Frontend should
    use leaflet-distortableImage or equivalent for rendering, not
    L.imageOverlay (which only supports axis-aligned bounds).
    """
    triplet_id: str
    image_url: str
    corners: Footprint      # 4 named corners, clockwise TL→TR→BR→BL (same as OHRC/TMC)
    opacity_hint: float = 0.6


# ---------------------------------------------------------------------------
# Sensor & triplet models
# ---------------------------------------------------------------------------

class SensorMeta(BaseModel):
    """Per-sensor metadata within a triplet."""
    sensor: str
    gsd_m: float
    sun_elevation_deg: float
    sun_azimuth_deg: float
    incidence_angle_deg: float
    footprint: Footprint


class TripletSummary(BaseModel):
    """Full metadata for one region (triplet of OHRC + TMC + IIRS tiles)."""
    id: str
    sensors: list[SensorMeta]
    intersection_footprint: Footprint


class TripletListResponse(BaseModel):
    """Response for GET /triplets."""
    triplets: list[TripletSummary]


# ---------------------------------------------------------------------------
# Footprint response
# ---------------------------------------------------------------------------

class FootprintResponse(BaseModel):
    """
    Per-sensor footprints plus the combined intersection, for one triplet.

    CORRECTNESS NOTE: each value is the verbatim 4-corner quad from the
    manifest — never a computed bounding box. See the Footprint docstring.
    """
    ohrc: Footprint
    tmc: Footprint
    iirs: Footprint
    intersection: Footprint


# ---------------------------------------------------------------------------
# Match models
# ---------------------------------------------------------------------------

class MatchPoint(BaseModel):
    """
    A single OHRC ↔ TMC-2 pixel correspondence (post-RANSAC from LoFTR).

    Pixel coordinates come directly from the ML team's matches.json
    (image1_x/y → ohrc_px, image2_x/y → tmc_px).
    Geographic coordinates are computed at load time by the backend
    using a perspective transform from the sensor footprint corners.
    """
    ohrc_px: tuple[float, float]      # (x, y) in OHRC 512×512 pixel space
    tmc_px: tuple[float, float]       # (x, y) in TMC 512×512 pixel space
    ohrc_latlon: tuple[float, float]  # (lat, lon) — computed from OHRC footprint corners
    tmc_latlon: tuple[float, float]   # (lat, lon) — computed from TMC footprint corners
    confidence: float


class MatchesResponse(BaseModel):
    """Response for GET /triplets/{id}/matches."""
    triplet_id: str
    num_matches: int
    homography: list[list[float]] | None  # 3×3 matrix, re-derived from inlier points; null if < 4 points
    matches: list[MatchPoint]


# ---------------------------------------------------------------------------
# Health / refresh
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str
    triplets_loaded: int
