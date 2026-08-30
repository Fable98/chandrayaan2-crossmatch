"""
schemas.py — Pydantic response models for the SIH26166 backend.

COORDINATE CONVENTION (used across all endpoints):
  All geographic coordinates use {"lat": float, "lon": float} objects.
  Footprint corners are always named (top_left, top_right, bottom_right,
  bottom_left) in CLOCKWISE winding order — directly consumable by Leaflet's
  L.polygon() without reordering.

  NEVER derive a bounding box via min()/max() on coordinate lists
  for OHRC or TMC-2. See the footprint router for the full rationale.

  EXCEPTION — IIRS only: the BBox model below uses axis-aligned bounds
  intentionally. At 80 m/px (275× coarser than OHRC), rotation error is
  invisible and Leaflet's L.imageOverlay needs an axis-aligned LatLngBounds.
  See loader.py get_iirs_bbox() for the scoped derivation.
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


class BBox(BaseModel):
    """
    Axis-aligned bounding box — used for IIRS overlay bounds ONLY.

    Do NOT use this for OHRC or TMC-2 footprints. Those must remain
    4-corner quads (see Footprint above).
    """
    west_lon: float
    east_lon: float
    south_lat: float
    north_lat: float


class IIRSOverlay(BaseModel):
    """Response for GET /triplets/{id}/iirs-overlay."""
    triplet_id: str
    image_url: str
    bounds: BBox            # axis-aligned — intentional for IIRS only, see loader.py
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
    """A single OHRC ↔ TMC pixel correspondence (post-RANSAC)."""
    ohrc_pixel: list[float]   # [x, y] in OHRC image coordinates
    tmc_pixel: list[float]    # [x, y] in TMC image coordinates
    confidence: float


class MatchesResponse(BaseModel):
    """Response for GET /triplets/{id}/matches."""
    triplet_id: str
    num_matches: int
    homography: list[list[float]]   # 3×3 matrix, pre-computed by ML team
    matches: list[MatchPoint]


# ---------------------------------------------------------------------------
# Health / refresh
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str
    triplets_loaded: int
