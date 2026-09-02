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

    candidates = []

    triplets_dir = getattr(loader, "PROCESSED_TRIPLETS_DIR", None)
    if triplets_dir:
        reg_dir = os.path.join(triplets_dir, clean_id)
        if os.path.isdir(reg_dir):
            candidates.append(os.path.join(reg_dir, f"{clean_sensor}_512.png"))
            candidates.append(os.path.join(reg_dir, "dem_512.png" if clean_sensor == "dem" else f"{clean_sensor}_512.png"))
            candidates.append(os.path.join(reg_dir, identifier))

        # Canonical sensor asset names and overlay aliases.
        if clean_sensor == "iirs" and identifier.endswith("iirs_overlay.png"):
            candidates.append(os.path.join(triplets_dir, "iirs_512.png"))
            for region in sorted(os.listdir(triplets_dir)):
                region_dir = os.path.join(triplets_dir, region)
                if os.path.isdir(region_dir):
                    candidates.append(os.path.join(region_dir, "iirs_512.png"))

        if clean_id in {"ohrc", "tmc", "iirs", "dem"} or os.path.isdir(os.path.join(triplets_dir, clean_id)):
            candidates.append(os.path.join(triplets_dir, f"{clean_sensor}_512.png"))
            candidates.append(os.path.join(triplets_dir, f"{clean_id}_{clean_sensor}_512.png"))

        if clean_sensor == "dem" and clean_id in {"dem", "dem_512"}:
            candidates.append(os.path.join(triplets_dir, "dem_512.png"))
            for region in sorted(os.listdir(triplets_dir)):
                region_dir = os.path.join(triplets_dir, region)
                if os.path.isdir(region_dir):
                    candidates.append(os.path.join(region_dir, "dem_512.png"))

    # Check common static-asset locations used by the pipeline and demo data.
    data_dir = getattr(loader, "DATA_DIR", None)
    repo_root = getattr(loader, "REPO_ROOT", Path(__file__).resolve().parent.parent.parent)

    # Registered products
    if clean_sensor in {"registered", "registration"}:
        reg_out_dir = Path(repo_root) / "registration_output" / clean_id
        candidates.append(os.path.join(reg_out_dir, "registered_ohrc.png"))
        candidates.append(os.path.join(reg_out_dir, "blend_overlay.png"))
        candidates.append(os.path.join(reg_out_dir, "checkerboard_qa.png"))
        candidates.append(os.path.join(reg_out_dir, identifier))
        candidates.append(os.path.join(reg_out_dir, f"{identifier}.png"))

    if data_dir:
        if clean_id in {"ohrc", "tmc", "iirs", "dem"} or os.path.isdir(os.path.join(data_dir, clean_id)):
            candidates.append(os.path.join(data_dir, f"{clean_sensor}_512.png"))
        candidates.append(os.path.join(data_dir, identifier))
        candidates.append(os.path.join(data_dir, f"{identifier}.png"))
        candidates.append(os.path.join(data_dir, f"{clean_id}_{clean_sensor}_512.png"))

        images_dir = os.path.join(data_dir, "images")
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
