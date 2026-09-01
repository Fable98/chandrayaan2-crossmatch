"""
loader.py — Loads user_triplets.json, processed_triplets, and match files into memory at startup.

The backend reads manifest data and ML match output, performs pixel-to-geo
conversion once at load time using shared TripletBounds (via geo.py), detects
DEM (Digital Elevation Model) availability, and caches everything in memory.
It NEVER writes to the data directory or re-runs any ML computation.
"""

import json
import os
from pathlib import Path

from geo import (
    pixel_to_latlon_from_bounds_batch,
    compute_homography_from_points,
)


# ---------------------------------------------------------------------------
# Data directories — override via env vars
# ---------------------------------------------------------------------------

DATA_DIR: str = os.environ.get(
    "DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "processed_user"),
)

PROCESSED_TRIPLETS_DIR: str = os.environ.get(
    "PROCESSED_TRIPLETS_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "processed_triplets"),
)

# ML team's output directory — searched as a secondary source for match files
ML_OUTPUT_DIR: str = os.environ.get(
    "ML_OUTPUT_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "ML_model"),
)

# Image size used by the ML pipeline (matcher.py resizes to this)
IMAGE_SIZE: int = 512


# ---------------------------------------------------------------------------
# In-memory stores (populated by load_all, read by routers)
# ---------------------------------------------------------------------------

_triplets: dict[str, dict] = {}       # keyed by triplet id
_triplet_list: list[dict] = []        # ordered list for GET /triplets
_matches: dict[str, dict] = {}        # keyed by triplet id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ml_matches(
    raw_matches: list[dict],
    bounds: dict,
) -> tuple[list[dict], list[list[float]] | None]:
    """
    Transform raw ML match output into the backend's MatchPoint shape.

    Input format (from ML team):
        [{image1_x, image1_y, image2_x, image2_y, confidence}, ...]

    Output:
        - List of MatchPoint-shaped dicts (with ohrc_px, tmc_px,
          ohrc_latlon, tmc_latlon, confidence)
        - Re-derived 3×3 homography matrix (or None if < 4 points)
    """
    if not raw_matches:
        return [], None

    # Collect pixel pairs for batch geo conversion
    ohrc_pixels = [(m["image1_x"], m["image1_y"]) for m in raw_matches]
    tmc_pixels = [(m["image2_x"], m["image2_y"]) for m in raw_matches]

    # Convert pixels to lat/lon using the shared affine transform from bounds
    ohrc_latlons = pixel_to_latlon_from_bounds_batch(
        ohrc_pixels, bounds, IMAGE_SIZE, IMAGE_SIZE,
    )
    tmc_latlons = pixel_to_latlon_from_bounds_batch(
        tmc_pixels, bounds, IMAGE_SIZE, IMAGE_SIZE,
    )

    # Build the MatchPoint-shaped dicts
    points = []
    for i, m in enumerate(raw_matches):
        points.append({
            "ohrc_px": (m["image1_x"], m["image1_y"]),
            "tmc_px": (m["image2_x"], m["image2_y"]),
            "ohrc_latlon": ohrc_latlons[i],
            "tmc_latlon": tmc_latlons[i],
            "confidence": m["confidence"],
        })

    # Re-derive homography from the inlier match points
    homography = compute_homography_from_points(ohrc_pixels, tmc_pixels)

    return points, homography


def _load_match_file(filepath: str) -> list[dict]:
    """
    Load a match file, handling both the ML team's bare-list format
    and the old wrapper format for backwards compatibility.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # ML team format: bare JSON array
    if isinstance(data, list):
        return data

    # Legacy wrapper format: {triplet_id, homography, matches: [...]}
    if isinstance(data, dict) and "matches" in data:
        return data["matches"]

    return []


def _normalize_triplet(data: dict, default_id: str | None = None, region_dir: str | None = None) -> dict:
    """Ensure triplet has an id, bounds, sensors list, and DEM availability."""
    triplet = dict(data)
    if "id" not in triplet:
        triplet["id"] = default_id or triplet.get("region_id") or triplet.get("ohrc_product_id") or "triplet_01"

    # Build sensors list if not present
    if "sensors" not in triplet:
        sensors = []
        if "ohrc_product_id" in triplet:
            sensors.append({
                "sensor": "ohrc",
                "gsd_m": triplet.get("ohrc_gsd_m", 0.25),
                "sun_elevation_deg": triplet.get("ohrc_sun_elevation_deg"),
                "sun_azimuth_deg": triplet.get("ohrc_sun_azimuth_deg"),
                "incidence_angle_deg": triplet.get("ohrc_incidence_deg"),
            })
        if "tmc2_product_id" in triplet or "tmc_product_id" in triplet:
            sensors.append({
                "sensor": "tmc",
                "gsd_m": triplet.get("tmc2_gsd_m", 5.0),
                "sun_elevation_deg": triplet.get("tmc2_sun_elevation_deg"),
                "sun_azimuth_deg": triplet.get("tmc2_sun_azimuth_deg"),
                "incidence_angle_deg": triplet.get("tmc2_incidence_deg"),
            })
        if "iirs_product_id" in triplet:
            sensors.append({
                "sensor": "iirs",
                "tile_id": "iirs_overlay.png",
                "gsd_m": triplet.get("iirs_gsd_m", 80.0),
                "sun_elevation_deg": triplet.get("iirs_sun_elevation_deg"),
                "sun_azimuth_deg": triplet.get("iirs_sun_azimuth_deg"),
                "incidence_angle_deg": triplet.get("iirs_incidence_deg"),
            })
        triplet["sensors"] = sensors

    # Check DEM (Digital Elevation Model) presence
    has_dem = False
    if region_dir and os.path.isdir(region_dir):
        if os.path.isfile(os.path.join(region_dir, "dem_512.png")) or os.path.isfile(os.path.join(region_dir, "dem_overlay.png")):
            has_dem = True

    # Fallback check in central images/dem
    dem_img_path = os.path.join(DATA_DIR, "images", "dem", "dem_512.png")
    if os.path.isfile(dem_img_path) or triplet.get("dem_available"):
        has_dem = True

    triplet["dem_available"] = has_dem
    if has_dem:
        reg_dem_filename = f"{triplet['id']}_dem_512.png"
        if os.path.isfile(os.path.join(DATA_DIR, "images", "dem", reg_dem_filename)):
            triplet["dem_url"] = f"/images/dem/{reg_dem_filename}"
        else:
            triplet["dem_url"] = "/images/dem/dem_512.png"
    else:
        triplet["dem_url"] = None

    if has_dem and not any(s.get("sensor") == "dem" for s in triplet.get("sensors", [])):
        triplet["sensors"].append({
            "sensor": "dem",
            "gsd_m": 5.0,
            "sun_elevation_deg": None,
            "sun_azimuth_deg": None,
            "incidence_angle_deg": None,
        })

    return triplet


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all() -> None:
    """
    Read user_triplets.json, processed_triplets, manifest.json, or batch triplet
    directories into memory. Called once at startup and again on GET /refresh.

    For each triplet, match points are enriched with lat/lon coordinates
    computed from the shared TripletBounds (see geo.py).
    """
    global _triplets, _triplet_list, _matches

    triplets_raw = []

    # 1. Check processed_triplets directory for the 6 real validated regions
    if os.path.isdir(PROCESSED_TRIPLETS_DIR):
        for entry in sorted(os.listdir(PROCESSED_TRIPLETS_DIR)):
            sub_dir = os.path.join(PROCESSED_TRIPLETS_DIR, entry)
            if os.path.isdir(sub_dir):
                sub_manifest = os.path.join(sub_dir, "manifest.json")
                if os.path.isfile(sub_manifest):
                    with open(sub_manifest, "r") as f:
                        sub_data = json.load(f)
                    if isinstance(sub_data, dict):
                        triplets_raw.append(_normalize_triplet(sub_data, default_id=entry, region_dir=sub_dir))

    # 2. Check user_triplets.json in DATA_DIR
    manifest_path = os.path.join(DATA_DIR, "user_triplets.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if isinstance(manifest, dict):
            for t in manifest.get("triplets", []):
                triplets_raw.append(_normalize_triplet(t))
        elif isinstance(manifest, list):
            for t in manifest:
                triplets_raw.append(_normalize_triplet(t))

    # 3. Check standalone manifest.json in DATA_DIR
    standalone_manifest = os.path.join(DATA_DIR, "manifest.json")
    if os.path.isfile(standalone_manifest):
        with open(standalone_manifest, "r") as f:
            m_data = json.load(f)
        if isinstance(m_data, dict):
            triplets_raw.append(_normalize_triplet(m_data))

    # 4. Check subdirectories in DATA_DIR for manifest.json
    if os.path.isdir(DATA_DIR):
        for entry in os.listdir(DATA_DIR):
            sub_dir = os.path.join(DATA_DIR, entry)
            if os.path.isdir(sub_dir) and sub_dir != PROCESSED_TRIPLETS_DIR:
                sub_manifest = os.path.join(sub_dir, "manifest.json")
                if os.path.isfile(sub_manifest):
                    with open(sub_manifest, "r") as f:
                        sub_data = json.load(f)
                    if isinstance(sub_data, dict):
                        triplets_raw.append(_normalize_triplet(sub_data, default_id=entry, region_dir=sub_dir))

    # Build lookup dict and ordered list (deduplicating by id)
    _triplets = {}
    _triplet_list = []
    for t in triplets_raw:
        tid = t.get("id")
        if tid and tid not in _triplets:
            _triplets[tid] = t
            _triplet_list.append(t)

    # -------------------------------------------------------------------
    # Load match files from two sources:
    #   1. processed_user/matches/{triplet_id}_matches.json
    #   2. ML_model/matches.json (mapped to region_001)
    # -------------------------------------------------------------------
    _matches = {}

    # Source 1: processed_user/matches/ — one file per triplet
    matches_dir = os.path.join(DATA_DIR, "matches")
    if os.path.isdir(matches_dir):
        for filename in os.listdir(matches_dir):
            if filename.endswith("_matches.json"):
                filepath = os.path.join(matches_dir, filename)
                triplet_id = filename.replace("_matches.json", "")
                raw = _load_match_file(filepath)
                _matches[triplet_id] = raw

    # Source 2: ML_model/matches.json — mapped to region_001
    ml_matches_path = os.path.join(ML_OUTPUT_DIR, "matches.json")
    if os.path.isfile(ml_matches_path):
        raw = _load_match_file(ml_matches_path)
        if raw:
            _matches["region_001"] = raw

    # -------------------------------------------------------------------
    # Enrich all match data with lat/lon + homography
    # -------------------------------------------------------------------
    enriched: dict[str, dict] = {}
    for triplet_id, raw_points in _matches.items():
        triplet = _triplets.get(triplet_id)
        if triplet is None:
            continue

        bounds = triplet.get("bounds")
        if bounds is None:
            enriched[triplet_id] = {
                "triplet_id": triplet_id,
                "matches": [],
                "homography": None,
            }
            continue

        points, homography = _parse_ml_matches(raw_points, bounds)

        enriched[triplet_id] = {
            "triplet_id": triplet_id,
            "matches": points,
            "homography": homography,
        }

    _matches = enriched

    print(
        f"[loader] Loaded {len(_triplets)} triplet(s) and "
        f"{len(_matches)} match file(s)"
    )


# ---------------------------------------------------------------------------
# Accessors (used by routers — never modify data)
# ---------------------------------------------------------------------------

def get_triplets() -> list[dict]:
    """Return the full ordered list of triplet dicts."""
    return _triplet_list


def get_triplet(triplet_id: str) -> dict | None:
    """Return a single triplet dict by ID, or None if not found."""
    return _triplets.get(triplet_id)


def get_matches(triplet_id: str) -> dict | None:
    """Return enriched match data for a triplet, or None if not available."""
    return _matches.get(triplet_id)


def triplet_count() -> int:
    """Return the number of loaded triplets (for health check)."""
    return len(_triplets)
