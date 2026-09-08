"""
footprint.py — GET /triplets/{triplet_id}/footprint and /triplets/{triplet_id}/iirs-overlay

Returns the shared bounding box (TripletBounds) for the triplet.
All three sensors (OHRC, TMC, IIRS) share the identical extent and 512×512 grid.
"""

from fastapi import APIRouter, HTTPException

from data import loader
from schemas import FootprintResponse, IIRSOverlay

router = APIRouter(tags=["footprint"])


@router.get("/triplets/{triplet_id}/footprint", response_model=FootprintResponse)
def get_footprint(triplet_id: str):
    """
    Return shared bounding box for the triplet.

    OHRC extent is the common crop boundary; TMC-2 and IIRS are cropped
    and reprojected to this identical bounding box during preprocessing.
    """
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")

    bounds = triplet.get("bounds")
    if bounds is None:
        raise HTTPException(
            status_code=500,
            detail=f"Bounds missing from triplet '{triplet_id}'",
        )

    return FootprintResponse(
        triplet_id=triplet_id,
        bounds=bounds,
    )


@router.get("/triplets/{triplet_id}/iirs-overlay", response_model=IIRSOverlay)
def get_iirs_overlay(triplet_id: str):
    """
    Return IIRS overlay metadata with shared TripletBounds.

    Returns the identical bounding box as the footprint endpoint.
    Frontend can render via standard L.imageOverlay or equivalent bounding box layer.
    """
    triplet = loader.get_triplet(triplet_id)
    if triplet is None:
        raise HTTPException(status_code=404, detail=f"Triplet '{triplet_id}' not found")

    bounds = triplet.get("bounds")
    if bounds is None:
        raise HTTPException(
            status_code=500,
            detail=f"Bounds missing from triplet '{triplet_id}'",
        )

    # Find the IIRS sensor entry
    iirs_entry = None
    for s in triplet.get("sensors", []):
        if s.get("sensor") == "iirs":
            iirs_entry = s
            break

    tile_id = iirs_entry.get("tile_id", "iirs_overlay.png") if iirs_entry else "iirs_overlay.png"

    return IIRSOverlay(
        triplet_id=triplet_id,
        image_url=f"/images/iirs/{tile_id}",
        bounds=bounds,
        opacity_hint=0.6,
    )
