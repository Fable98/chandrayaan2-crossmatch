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
from schemas import FootprintResponse, IIRSOverlay, BBox

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
    Return IIRS overlay metadata for Leaflet's L.imageOverlay.

    Unlike OHRC/TMC-2 footprints (which are 4-corner quads), the IIRS bounds
    are intentionally axis-aligned. At 80 m/px the rotation error is invisible,
    and Leaflet's imageOverlay requires a LatLngBounds (axis-aligned rectangle).
    See loader.get_iirs_bbox() for the scoped derivation.
    """
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")

    # Find the IIRS sensor entry to get its tile_id
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

    # Use flat bbox fields if the manifest already has them, otherwise derive
    if "west_lon" in iirs_entry:
        bounds_dict = {
            "west_lon": iirs_entry["west_lon"],
            "east_lon": iirs_entry["east_lon"],
            "south_lat": iirs_entry["south_lat"],
            "north_lat": iirs_entry["north_lat"],
        }
    else:
        bounds_dict = loader.get_iirs_bbox(triplet)

    tile_id = iirs_entry.get("tile_id", "iirs_overlay.png")

    return IIRSOverlay(
        triplet_id=triplet_id,
        image_url=f"/images/iirs/{tile_id}",
        bounds=BBox(**bounds_dict),
    )

