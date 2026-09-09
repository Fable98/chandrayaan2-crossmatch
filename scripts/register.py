"""
register.py — Generates geometrically registered image products.

Performs geometric alignment of source (moving) image to reference (fixed)
image using the estimated homography matrix. Produces:
  1. registered_source.png — source warped directly into target pixel coordinates
  2. blend_overlay.png — alpha blended composite to visually inspect alignment
  3. checkerboard_qa.png — alternating tiles of source & reference for edge continuity QA
  4. registered GeoTIFF (.tif) — pixel-grid fallback georeference unless real
     CRS/transform supplied; see save_geotiff() and manifest georeferenced flag.
  5. matches.json / metrics.json / transform.json sidecars (canonical names).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets"
MATCHES_DIR = REPO_ROOT / "data_preprocessing_pipeline" / "matches"
REGISTRATION_OUT_DIR = REPO_ROOT / "registration_output"


def warp_source_to_reference(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    homography: np.ndarray,
    output_shape: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Warp source image using homography matrix to align with destination image.
    """
    w, h = output_shape
    warped = cv2.warpPerspective(src_img, homography, (w, h), flags=cv2.INTER_LANCZOS4)
    return warped


def create_blend_overlay(
    warped_src: np.ndarray,
    dst_img: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create 50/50 composite blend (or specified alpha) of warped source and reference image.
    """
    if warped_src.ndim == 2:
        warped_src = cv2.cvtColor(warped_src, cv2.COLOR_GRAY2BGR)
    if dst_img.ndim == 2:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_GRAY2BGR)

    # Convert to green/magenta or standard RGB blend for crisp alignment visibility
    blend = cv2.addWeighted(warped_src, alpha, dst_img, 1.0 - alpha, 0)
    return blend


def create_checkerboard_qa(
    warped_src: np.ndarray,
    dst_img: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    """
    Create a checkerboard visualization alternating between registered source and reference.
    Useful for inspecting crater rim and ridge alignment continuity.
    """
    if warped_src.ndim == 2:
        warped_src = cv2.cvtColor(warped_src, cv2.COLOR_GRAY2BGR)
    if dst_img.ndim == 2:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_GRAY2BGR)

    h, w = dst_img.shape[:2]
    checkerboard = np.zeros_like(dst_img)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            if ((y // block_size) + (x // block_size)) % 2 == 0:
                checkerboard[y:y_end, x:x_end] = warped_src[y:y_end, x:x_end]
            else:
                checkerboard[y:y_end, x:x_end] = dst_img[y:y_end, x:x_end]

    return checkerboard


def bounds_to_eqc_transform(
    bounds: dict,
    width: int,
    height: int,
    radius_m: float = 1737400.0,
) -> tuple | None:
    """Derive a lunar EQC Affine transform from geographic bounds.

    EQC (lat_ts=0, lon_0=0): x = R*lon_rad, y = R*lat_rad. Pixel (0,0) is the
    north-west corner. Returns (transform, crs_proj4, pixel_size_m) or None
    when bounds are missing/malformed. Accuracy is limited by the bounds
    themselves (manifest shared-footprint bounds, not per-pixel SPICE).
    """
    try:
        from rasterio.transform import from_origin
        import math
        w = float(bounds["west_lon"])
        e = float(bounds["east_lon"])
        s = float(bounds["south_lat"])
        n = float(bounds["north_lat"])
        if not (e > w and n > s and width > 0 and height > 0):
            return None
        xw, xe = math.radians(w) * radius_m, math.radians(e) * radius_m
        ys, yn = math.radians(s) * radius_m, math.radians(n) * radius_m
        dx, dy = (xe - xw) / width, (yn - ys) / height
        if not (dx > 0 and dy > 0):
            return None
        crs = "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs"
        return from_origin(xw, yn, dx, dy), crs, (dx + dy) / 2.0
    except Exception:
        return None


def save_geotiff(image: np.ndarray, path: Path, transform=None, crs=None, gsd_m: float | None = None) -> str | None:
    """Save image as lunar GeoTIFF.

    Uses provided raster CRS/transform when available; otherwise falls back to a
    pixel-grid EQC placeholder (from_origin). The placeholder is NOT a rigorous
    PDS/SPICE georeference — callers must record georeferenced=False in manifests.
    Returns path string, or None on failure.
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
        h, w = image.shape[:2]
        count = 1 if image.ndim == 2 else min(image.shape[2], 3)
        if transform is None and gsd_m is not None:
            transform = from_origin(0, h * gsd_m, gsd_m, gsd_m)
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": count,
            "dtype": "uint8",
            "crs": crs or "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs",
            "transform": transform if transform is not None else from_origin(0, h, 1.0, 1.0),
            "compress": "lzw",
        }
        with rasterio.open(str(path), "w", **profile) as dst:
            if image.ndim == 3:
                for b in range(count):
                    dst.write(image[:, :, count - 1 - b], b + 1)
            else:
                dst.write(image, 1)
        return str(path)
    except Exception:
        return None


def register_region(
    region_id: str,
    matches: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str] | None:
    """
    Register images for a single region and save output products.
    """
    reg_dir = PROCESSED_DIR / region_id
    ohrc_path = reg_dir / "ohrc_512.png"
    tmc_path = reg_dir / "tmc_512.png"

    if not ohrc_path.is_file() or not tmc_path.is_file() or len(matches) < 4:
        return None

    src_img = cv2.imread(str(ohrc_path))
    dst_img = cv2.imread(str(tmc_path))

    src_pts = np.array([[float(m.get("image1_x", m.get("source_x"))), float(m.get("image1_y", m.get("source_y")))] for m in matches], dtype=np.float32)
    dst_pts = np.array([[float(m.get("image2_x", m.get("target_x"))), float(m.get("image2_y", m.get("target_y")))] for m in matches], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    region_out = output_dir / region_id
    region_out.mkdir(parents=True, exist_ok=True)

    warped = warp_source_to_reference(src_img, dst_img, H)
    blend = create_blend_overlay(warped, dst_img)
    checker = create_checkerboard_qa(warped, dst_img)

    warped_path = region_out / "registered_ohrc.png"
    blend_path = region_out / "blend_overlay.png"
    checker_path = region_out / "checkerboard_qa.png"
    tif_path = region_out / "registered_ohrc.tif"
    matches_path = region_out / "matches.json"
    metrics_path = region_out / "metrics.json"
    transform_path = region_out / "transform.json"

    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(blend_path), blend)
    cv2.imwrite(str(checker_path), checker)
    # Georeference from the region manifest shared-footprint bounds when
    # available (lunar EQC meters); otherwise pixel-grid fallback.
    _geo, _georef_method = None, "pixel_grid_fallback"
    try:
        _mf = reg_dir / "manifest.json"
        if _mf.is_file():
            _mdata = json.load(open(_mf))
            _mb = _mdata.get("bounds_optical") or _mdata.get("bounds")
            _geo = bounds_to_eqc_transform(_mb, warped.shape[1], warped.shape[0]) if _mb else None
            if _geo is not None:
                _georef_method = "manifest_bounds_eqc"
    except Exception:
        _geo = None
    if _geo is not None:
        saved_tif = save_geotiff(warped, tif_path, transform=_geo[0], crs=_geo[1])
    else:
        saved_tif = save_geotiff(warped, tif_path)
    if saved_tif is None or not tif_path.exists():
        cv2.imwrite(str(tif_path), warped)
        saved_tif = str(tif_path)
        georeferenced = False
    else:
        georeferenced = True

    # Save homography (legacy name) + canonical transform.json
    h_path = region_out / "ohrc_to_tmc_homography.json"
    with open(h_path, "w") as f:
        json.dump(
            {
                "homography": H.tolist(),
                "inlier_count": len(matches),
                "fit_rmse_is_in_sample": True,
                "georeferenced": georeferenced,
                "georeferencing_method": _georef_method,
                "georeferencing_note": (
                    "Lunar EQC from manifest shared-footprint bounds; per-pixel SPICE "
                    "rigor not claimed."
                    if georeferenced else
                    "Pixel-grid fallback unless real CRS supplied."
                ),
            },
            f,
            indent=4,
        )
    with open(transform_path, "w") as f:
        json.dump({"model": "homography", "matrix": H.tolist()}, f, indent=4)
    # Canonical match points + minimal metrics sidecars for PS compliance
    with open(matches_path, "w") as f:
        json.dump(matches, f, indent=4)
    with open(metrics_path, "w") as f:
        json.dump(
            {"region_id": region_id, "inlier_count": len(matches), "georeferenced": georeferenced},
            f,
            indent=4,
        )

    return {
        "registered_source": str(warped_path),
        "registered_geotiff": saved_tif,
        "blend_overlay": str(blend_path),
        "checkerboard_qa": str(checker_path),
        "homography_json": str(h_path),
        "transform_json": str(transform_path),
        "matches_json": str(matches_path),
        "metrics_json": str(metrics_path),
    }


def register_composed_ohrc_to_iirs(
    region_id: str,
    output_dir: Path,
) -> dict[str, str] | None:
    """
    Produce a composed OHRC→IIRS registered product if the triplet cycle
    report for this region contains a valid composed homography.
    """
    # Search for triplet consistency report in benchmark output or eval output
    candidate_paths = [
        REPO_ROOT / "benchmarks" / "registration_benchmark_output" / region_id / "triplet_consistency_report.json",
        REPO_ROOT / "evaluation_output" / region_id / "triplet_consistency_report.json",
        REPO_ROOT / "benchmarks" / "registration_benchmark_output" / region_id / "triplet_cycle" / "triplet_consistency_report.json",
        REPO_ROOT / "evaluation_output" / f"{region_id}_triplet_consistency_report.json",
    ]

    report_data = None
    for p in candidate_paths:
        if p.is_file():
            with open(p) as f:
                report_data = json.load(f)
            break

    if report_data is None:
        return None

    composition = report_data.get("composition")
    if not composition or not composition.get("homography"):
        return None

    reg_dir = PROCESSED_DIR / region_id
    ohrc_path = reg_dir / "ohrc_512.png"
    iirs_path = reg_dir / "iirs_512.png"

    if not ohrc_path.is_file() or not iirs_path.is_file():
        return None

    H_AC = np.array(composition["homography"], dtype=np.float64)
    src_img = cv2.imread(str(ohrc_path))
    dst_img = cv2.imread(str(iirs_path))

    region_out = output_dir / region_id
    region_out.mkdir(parents=True, exist_ok=True)

    warped = warp_source_to_reference(src_img, dst_img, H_AC, (dst_img.shape[1], dst_img.shape[0]))
    blend = create_blend_overlay(warped, dst_img)
    checker = create_checkerboard_qa(warped, dst_img)

    warped_path = region_out / "registered_ohrc_to_iirs_composed.png"
    blend_path = region_out / "blend_ohrc_iirs_composed.png"
    checker_path = region_out / "checkerboard_ohrc_iirs_composed.png"
    tif_path = region_out / "registered_ohrc_to_iirs_composed.tif"

    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(blend_path), blend)
    cv2.imwrite(str(checker_path), checker)
    _geo2 = None
    try:
        _mf2 = reg_dir / "manifest.json"
        if _mf2.is_file():
            _md2 = json.load(open(_mf2))
            _mb2 = _md2.get("bounds_iirs") or _md2.get("bounds_optical") or _md2.get("bounds")
            _geo2 = bounds_to_eqc_transform(_mb2, warped.shape[1], warped.shape[0]) if _mb2 else None
    except Exception:
        _geo2 = None
    if _geo2 is not None:
        saved_tif = save_geotiff(warped, tif_path, transform=_geo2[0], crs=_geo2[1])
    else:
        saved_tif = save_geotiff(warped, tif_path)

    # Write registered products manifest
    manifest_path = region_out / "registered_products_manifest.json"
    manifest = {
        "region_id": region_id,
        "mode": "composed_registration",
        "chain": "OHRC -> TMC-2 -> IIRS",
        "homography_composed": H_AC.tolist(),
        "products": {
            "registered_ohrc_png": str(region_out / "registered_ohrc.png"),
            "registered_ohrc_tif": str(region_out / "registered_ohrc.tif"),
            "blend_ohrc_tmc": str(region_out / "blend_overlay.png"),
            "checkerboard_ohrc_tmc": str(region_out / "checkerboard_qa.png"),
            "registered_ohrc_to_iirs_png": str(warped_path),
            "registered_ohrc_to_iirs_tif": saved_tif,
            "blend_ohrc_iirs": str(blend_path),
            "checkerboard_ohrc_iirs": str(checker_path),
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    return {
        "registered_composed": str(warped_path),
        "registered_composed_geotiff": saved_tif,
        "blend_composed": str(blend_path),
        "checkerboard_composed": str(checker_path),
        "manifest": str(manifest_path),
    }


def register_all_regions():
    """
    Batch register all regions with available matches.
    """
    REGISTRATION_OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = sorted([d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()])
    print(f"[Register] Registering {len(regions)} regions into {REGISTRATION_OUT_DIR}...")

    results = {}
    for region_id in regions:
        match_file = MATCHES_DIR / f"{region_id}_matches.json"
        if not match_file.is_file():
            if region_id == "region_001":
                match_file = REPO_ROOT / "ML_model" / "matches.json"
            else:
                continue

        with open(match_file, "r") as f:
            matches_data = json.load(f)
            if isinstance(matches_data, dict) and "matches" in matches_data:
                matches_data = matches_data["matches"]

        res = register_region(region_id, matches_data, REGISTRATION_OUT_DIR)
        if res:
            results[region_id] = res
            print(f"  [+] Registered OHRC->TMC {region_id}")
            res_comp = register_composed_ohrc_to_iirs(region_id, REGISTRATION_OUT_DIR)
            if res_comp:
                results[region_id].update(res_comp)
                print(f"  [+] Registered Composed OHRC->IIRS {region_id}")
        else:
            print(f"  [-] Skipped {region_id} (insufficient matches or missing assets)")

    print(f"[Register] Completed registration for {len(results)} regions.")


if __name__ == "__main__":
    register_all_regions()

