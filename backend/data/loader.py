"""
loader.py — Loads user_triplets.json and match files into memory at startup.

The backend reads manifest data and ML match output, performs pixel-to-geo
conversion once at load time (using geo.py), and caches everything in memory.
It NEVER writes to the data directory or re-runs any ML computation.
"""

import json
import os
from pathlib import Path

from geo import (
    pixel_to_latlon_batch,
    compute_homography_from_points,
)


# ---------------------------------------------------------------------------
# Data directories — override via env vars
# ---------------------------------------------------------------------------

DATA_DIR: str = os.environ.get(
    "DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "processed_user"),
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

def _get_sensor_footprint(triplet: dict, sensor_name: str) -> dict | None:
    """Extract the footprint dict for a given sensor from a triplet."""
    for s in triplet["sensors"]:
        if s["sensor"] == sensor_name:
            return s["footprint"]
    return None


def _parse_ml_matches(
    raw_matches: list[dict],
    ohrc_corners: dict,
    tmc_corners: dict,
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

    # Convert pixels to lat/lon using perspective transform from corners
    ohrc_latlons = pixel_to_latlon_batch(
        ohrc_pixels, ohrc_corners, IMAGE_SIZE, IMAGE_SIZE,
    )
    tmc_latlons = pixel_to_latlon_batch(
        tmc_pixels, tmc_corners, IMAGE_SIZE, IMAGE_SIZE,
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


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all() -> None:
    """
    Read user_triplets.json and all match files into memory.
    Called once at startup and again on GET /refresh.

    For each triplet, match points are enriched with lat/lon coordinates
    computed from the sensor footprint corners (see geo.py).
    """
    global _triplets, _triplet_list, _matches

    manifest_path = os.path.join(DATA_DIR, "user_triplets.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    triplets_raw = manifest.get("triplets", [])

    # Build lookup dict and ordered list
    _triplets = {t["id"]: t for t in triplets_raw}
    _triplet_list = triplets_raw

    # -------------------------------------------------------------------
    # Load match files from two sources:
    #   1. processed_user/matches/{triplet_id}_matches.json
    #   2. ML_model/matches.json (mapped to the first triplet)
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

    # Source 2: ML_model/matches.json — mapped to region_001 (the primary
    # triplet the ML team ran LoFTR on). This is a temporary convention
    # until the ML team adds per-triplet naming.
    ml_matches_path = os.path.join(ML_OUTPUT_DIR, "matches.json")
    if os.path.isfile(ml_matches_path):
        raw = _load_match_file(ml_matches_path)
        if raw:
            # ML output takes precedence over mock data for region_001
            _matches["region_001"] = raw

    # -------------------------------------------------------------------
    # Enrich all match data with lat/lon + homography
    # -------------------------------------------------------------------
    enriched: dict[str, dict] = {}
    for triplet_id, raw_points in _matches.items():
        triplet = _triplets.get(triplet_id)
        if triplet is None:
            # Match file references a triplet not in the manifest — skip
            print(f"[loader] WARNING: match file for '{triplet_id}' has no manifest entry, skipping")
            continue

        ohrc_corners = _get_sensor_footprint(triplet, "ohrc")
        tmc_corners = _get_sensor_footprint(triplet, "tmc")

        if ohrc_corners is None or tmc_corners is None:
            print(f"[loader] WARNING: missing OHRC/TMC footprint for '{triplet_id}', skipping geo conversion")
            enriched[triplet_id] = {
                "triplet_id": triplet_id,
                "matches": [],
                "homography": None,
            }
            continue

        points, homography = _parse_ml_matches(raw_points, ohrc_corners, tmc_corners)

        enriched[triplet_id] = {
            "triplet_id": triplet_id,
            "matches": points,
            "homography": homography,
        }

    _matches = enriched

    print(
        f"[loader] Loaded {len(_triplets)} triplet(s) and "
        f"{len(_matches)} match file(s) from {DATA_DIR}"
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

