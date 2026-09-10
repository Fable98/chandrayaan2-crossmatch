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

try:
    from uploads import check_file_response_allowed
except ImportError:  # pragma: no cover - direct-router test path
    from backend.uploads import check_file_response_allowed  # type: ignore

router = APIRouter(tags=["images"])

# Step 12: every FileResponse below must resolve inside one of these roots
# AND carry an image extension — otherwise 404 (no traversal, no /etc/passwd).
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _allowed_roots(repo_root: Path, triplets_dir: str | None, data_dir: str | None) -> list[Path]:
    roots = [
        Path(repo_root) / "registration_output",
        Path(repo_root) / "lunar-frontend" / "public" / "images" / "registered",
        Path(repo_root) / "data_preprocessing_pipeline",
    ]
    if triplets_dir:
        roots.append(Path(triplets_dir))
    if data_dir:
        roots.append(Path(data_dir))
        roots.append(Path(data_dir) / "images")
    return roots


@router.get("/images/{sensor}/{identifier:path}")
def get_image(sensor: str, identifier: str):
    """
    Serve lunar imagery for any sensor (ohrc, tmc, iirs, dem) by region ID or filename.
    """
    clean_sensor = sensor.lower()
    # Reject traversal attempts up front (the allowlist gate re-checks).
    if ".." in identifier.split("/") or ".." in clean_sensor.split("/"):
        raise HTTPException(status_code=404, detail="Image not found")
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

    # LRO NAC external reference imagery
    if clean_sensor in {"lro", "lro_nac", "nac"}:
        lro_pair_dir = Path(repo_root) / "data_preprocessing_pipeline" / "lro_nac_pairs" / clean_id
        candidates.append(os.path.join(lro_pair_dir, "lro_nac_reference_512.png"))
        candidates.append(os.path.join(lro_pair_dir, identifier))
        candidates.append(os.path.join(lro_pair_dir, f"{identifier}.png"))
        reg_lro_dir = Path(repo_root) / "registration_output" / "lro_nac" / clean_id
        candidates.append(os.path.join(reg_lro_dir, "blend_overlay.png"))
        candidates.append(os.path.join(reg_lro_dir, "registered_source.png"))

    # Registered products
    if clean_sensor in {"registered", "registration"}:
        for base_reg in [
            Path(repo_root) / "registration_output",
            Path(repo_root) / "lunar-frontend" / "public" / "images" / "registered",
        ]:
            reg_out_dir = base_reg / clean_id
            candidates.append(os.path.join(reg_out_dir, "registered_ohrc.png"))
            candidates.append(os.path.join(reg_out_dir, "registered_source.png"))
            candidates.append(os.path.join(reg_out_dir, "blend_overlay.png"))
            candidates.append(os.path.join(reg_out_dir, "checkerboard_qa.png"))
            candidates.append(os.path.join(reg_out_dir, identifier))
            candidates.append(os.path.join(reg_out_dir, f"{identifier}.png"))
            candidates.append(str(base_reg / identifier))
            candidates.append(str(base_reg / f"{identifier}.png"))

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

    roots = _allowed_roots(repo_root, triplets_dir, data_dir)
    for path in candidates:
        if os.path.isfile(path):
            try:
                real = check_file_response_allowed(path, roots, ALLOWED_IMAGE_EXTS)
            except HTTPException:
                continue
            return FileResponse(str(real), media_type="image/png")

    raise HTTPException(
        status_code=404,
        detail=f"Image for sensor '{sensor}' and identifier '{identifier}' not found",
    )
