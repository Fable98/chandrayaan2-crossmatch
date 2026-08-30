"""
loader.py — Loads user_triplets.json and match files into memory at startup.

The backend NEVER writes to the data directory or computes derived data.
It only reads the manifest and match files that the data/ML teams produce.
"""

import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Data directory — override via DATA_DIR env var
# ---------------------------------------------------------------------------

DATA_DIR: str = os.environ.get(
    "DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "processed_user"),
)

# ---------------------------------------------------------------------------
# In-memory stores (populated by load_all, read by routers)
# ---------------------------------------------------------------------------

_triplets: dict[str, dict] = {}       # keyed by triplet id
_triplet_list: list[dict] = []        # ordered list for GET /triplets
_matches: dict[str, dict] = {}        # keyed by triplet id


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_all() -> None:
    """
    Read user_triplets.json and all match files into memory.
    Called once at startup and again on GET /refresh.
    """
    global _triplets, _triplet_list, _matches

    manifest_path = os.path.join(DATA_DIR, "user_triplets.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    triplets_raw = manifest.get("triplets", [])

    # Build lookup dict and ordered list
    _triplets = {t["id"]: t for t in triplets_raw}
    _triplet_list = triplets_raw

    # Load match files — convention: matches/{triplet_id}_matches.json
    matches_dir = os.path.join(DATA_DIR, "matches")
    _matches = {}

    if os.path.isdir(matches_dir):
        for filename in os.listdir(matches_dir):
            if filename.endswith("_matches.json"):
                filepath = os.path.join(matches_dir, filename)
                with open(filepath, "r") as f:
                    match_data = json.load(f)
                # Use the triplet_id from inside the file if present,
                # otherwise derive from filename
                tid = match_data.get(
                    "triplet_id",
                    filename.replace("_matches.json", ""),
                )
                _matches[tid] = match_data

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
    """Return match data for a triplet, or None if not available."""
    return _matches.get(triplet_id)


def triplet_count() -> int:
    """Return the number of loaded triplets (for health check)."""
    return len(_triplets)


# ---------------------------------------------------------------------------
# IIRS bbox derivation — the ONLY place min()/max() on coordinates is allowed
# ---------------------------------------------------------------------------

def get_iirs_bbox(triplet: dict) -> dict:
    """
    IIRS-only exception: unlike OHRC/TMC-2, IIRS is intentionally treated
    as an axis-aligned overlay (per ML team decision — the 275× scale gap
    between OHRC 0.25 m/px and IIRS 80 m/px makes rotation error invisible
    at this resolution). Leaflet's L.imageOverlay requires an axis-aligned
    LatLngBounds anyway.

    DO NOT reuse this pattern for OHRC/TMC-2 footprints, which must stay
    4-corner quads. See schemas.py and routers/footprint.py.
    """
    # Find the IIRS sensor entry in the sensors list
    iirs_entry = None
    for s in triplet["sensors"]:
        if s["sensor"] == "iirs":
            iirs_entry = s
            break

    if iirs_entry is None:
        return None

    fp = iirs_entry["footprint"]
    corners = [fp["top_left"], fp["top_right"], fp["bottom_right"], fp["bottom_left"]]
    lons = [c["lon"] for c in corners]
    lats = [c["lat"] for c in corners]
    return {
        "west_lon": min(lons),
        "east_lon": max(lons),
        "south_lat": min(lats),
        "north_lat": max(lats),
    }

