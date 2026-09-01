"""
images.py — Dynamic static image serving for Chandrayaan-2 lunar tiles.

Handles requests like:
  - GET /images/ohrc/region_001
  - GET /images/tmc/region_002
  - GET /images/iirs/iirs_overlay.png
  - GET /images/dem/dem_512.png
  - GET /images/dem/region_001_dem_512.png
"""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from data import loader

router = APIRouter(tags=["images"])


@router.get("/images/{sensor}/{identifier:path}")
def get_image(sensor: str, identifier: str):
    """
    Serve lunar imagery for any sensor (ohrc, tmc, iirs, dem) by region ID or filename.
    """
    clean_sensor = sensor.lower()
    clean_id = identifier.replace(".png", "")

    # Candidates list in priority order
    candidates = []

    # 1. If identifier is a known region or subpath, check in processed_triplets/{region_id}/{sensor}_512.png
    triplets_dir = getattr(loader, "PROCESSED_TRIPLETS_DIR", None)
    if triplets_dir:
        reg_dir = os.path.join(triplets_dir, clean_id)
        if os.path.isdir(reg_dir):
            candidates.append(os.path.join(reg_dir, f"{clean_sensor}_512.png"))
            candidates.append(os.path.join(reg_dir, identifier))

    # 2. Check in processed_user/images/{sensor}/...
    images_dir = os.path.join(loader.DATA_DIR, "images")
    candidates.append(os.path.join(images_dir, clean_sensor, identifier))
    candidates.append(os.path.join(images_dir, clean_sensor, f"{identifier}.png"))
    candidates.append(os.path.join(images_dir, clean_sensor, f"{clean_id}_{clean_sensor}_512.png"))

    for path in candidates:
        if os.path.isfile(path):
            return FileResponse(path, media_type="image/png")

    raise HTTPException(
        status_code=404,
        detail=f"Image for sensor '{sensor}' and identifier '{identifier}' not found",
    )
