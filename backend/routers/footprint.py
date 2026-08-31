"""
footprint.py — GET /triplets/{triplet_id}/footprint

Returns the 4 named corner coordinates for each sensor (OHRC, TMC, IIRS)
plus the combined intersection footprint.

╔══════════════════════════════════════════════════════════════════════════╗
║  CRITICAL CORRECTNESS REQUIREMENT                                      ║
║                                                                        ║
║  Each footprint is a ROTATED QUAD with 4 explicitly named corners      ║
║  (top_left, top_right, bottom_right, bottom_left) pulled VERBATIM      ║
║  from user_triplets.json.                                              ║
║                                                                        ║
║  DO NOT call min()/max() on coordinate lists anywhere in this file     ║
║  or anywhere else in this codebase. A previous implementation used a   ║
║  min/max bounding box which caused visible misalignment on the         ║
║  Leaflet map. That bug must never return.                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from fastapi import APIRouter, HTTPException

from data import loader
from schemas import FootprintResponse, IIRSOverlay

router = APIRouter(tags=["footprint"])


@router.get("/triplets/{triplet_id}/footprint", response_model=FootprintResponse)
def get_footprint(triplet_id: str):
    """
    Return per-sensor footprints and intersection footprint for one triplet.

    Every coordinate is the verbatim named corner from the manifest.
    No bounding-box derivation. No min()/max(). No geometry computation.
    """
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")

    # Build a lookup of sensor name → footprint from the triplet's sensor list
    sensor_footprints: dict = {}
    for sensor_meta in triplet["sensors"]:
        sensor_footprints[sensor_meta["sensor"]] = sensor_meta["footprint"]

    # All three sensors must be present
    for required in ("ohrc", "tmc", "iirs"):
        if required not in sensor_footprints:
            raise HTTPException(
                status_code=500,
                detail=f"Sensor '{required}' missing from triplet '{triplet_id}'",
            )

    return FootprintResponse(
        ohrc=sensor_footprints["ohrc"],
        tmc=sensor_footprints["tmc"],
        iirs=sensor_footprints["iirs"],
        intersection=triplet["intersection_footprint"],
    )


@router.get("/triplets/{triplet_id}/iirs-overlay", response_model=IIRSOverlay)
def get_iirs_overlay(triplet_id: str):
    """
    Return IIRS overlay metadata with full 4-corner footprint.

    Returns the IIRS footprint as a proper rotated quad — the same Footprint
    model used by OHRC/TMC-2 — NOT an axis-aligned bounding box.

    VERIFIED: scripts/check_iirs_rotation.py confirmed both region_001 and
    region_002 IIRS footprints are rotated quads (bottom lons differ from
    top lons, ~2.5% area loss from bbox). Frontend should use
    leaflet-distortableImage or equivalent, not L.imageOverlay.
    """
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")

    # Find the IIRS sensor entry
    iirs_entry = None
    for s in triplet["sensors"]:
        if s["sensor"] == "iirs":
            iirs_entry = s
            break

    if iirs_entry is None:
        raise HTTPException(
            status_code=500,
            detail=f"IIRS sensor missing from triplet '{triplet_id}'",
        )

    tile_id = iirs_entry.get("tile_id", "iirs_overlay.png")

    # Return the 4-corner footprint verbatim from the manifest — no bbox
    # derivation, no min()/max(). Same pattern as the OHRC/TMC footprint.
    return IIRSOverlay(
        triplet_id=triplet_id,
        image_url=f"/images/iirs/{tile_id}",
        corners=iirs_entry["footprint"],
    )


