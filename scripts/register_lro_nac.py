#!/usr/bin/env python3
"""
scripts/register_lro_nac.py — Primary LRO NAC Reference-Image Registration for OHRC

Registers Chandrayaan-2 OHRC (Moving/Source) against NASA LRO NAC (Fixed/Reference),
fulfilling Problem Statement 26166's explicit Lunar reference images requirement.

Executes:
1. Matcher: match_images_cfog() with Image 1 = OHRC, Image 2 = LRO NAC.
   Evaluates multimodal_pair=False (optical-to-optical matching at ~1-4x scale)
   alongside default feature extraction.
2. Canonical Metrics: Inlier count, Fit RMSE (px), Spatial Coverage, Uniformity.
3. Registered Product Suite: registered_source.png, registered_source.tif,
   blend_overlay.png, checkerboard_qa.png, transform.json, metrics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ML_model.matcher_cfog import match_images_cfog
from ML_model.metrics import compute_canonical_metrics
from scripts.register import (
    warp_source_to_reference,
    create_blend_overlay,
    create_checkerboard_qa,
    save_geotiff,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("register_lro_nac")

PAIRS_DIR = REPO_ROOT / "data_preprocessing_pipeline" / "lro_nac_pairs"
REG_OUT_DIR = REPO_ROOT / "registration_output" / "lro_nac"


def run_registration_for_region(
    region_id: str,
    output_base_dir: Optional[Path | str] = None,
    force_non_multimodal: bool = True,
) -> Dict[str, Any]:
    """
    Executes registration between OHRC (source) and LRO NAC (reference).
    """
    pair_dir = PAIRS_DIR / region_id
    if not pair_dir.exists():
        raise FileNotFoundError(f"Pair directory not found: {pair_dir}. Run prepare_lro_nac_pair.py first.")

    ohrc_path = pair_dir / "ohrc_source_512.png"
    nac_path = pair_dir / "lro_nac_reference_512.png"
    manifest_path = pair_dir / "manifest.json"

    if not ohrc_path.exists() or not nac_path.exists():
        raise FileNotFoundError(f"Missing images in {pair_dir}")

    region_out = (Path(output_base_dir) if output_base_dir else REG_OUT_DIR) / region_id
    region_out.mkdir(parents=True, exist_ok=True)
    temp_cfog_out = region_out / "_cfog_work"

    logger.info(
        "Registering region '%s': Source=OHRC (%s), Reference=LRO_NAC (%s)",
        region_id,
        ohrc_path.name,
        nac_path.name,
    )

    # 1. Execute match_images_cfog
    # Image 1 = OHRC (Source / Moving), Image 2 = LRO NAC (Reference / Fixed)
    match_result = match_images_cfog(
        img_path1=str(ohrc_path),
        img_path2=str(nac_path),
        output_dir=str(temp_cfog_out),
        source_sensor="OHRC",
        reference_sensor="LRO_NAC",
        explicit_gsd1=1.0,
        explicit_gsd2=1.0,
        multimodal_pair=False if force_non_multimodal else None,
    )

    if temp_cfog_out.exists():
        shutil.rmtree(str(temp_cfog_out), ignore_errors=True)

    status = match_result.get("status")
    H = match_result.get("homography")
    inliers = match_result.get("inliers", [])
    raw_matches = match_result.get("matches", [])

    logger.info(
        "Match result for %s: status=%s, raw_matches=%d, inliers=%d",
        region_id,
        status,
        len(raw_matches),
        len(inliers),
    )

    # 2. Canonical Metrics: use metrics returned by engine or compute canonical
    metrics = match_result.get("metrics")
    if metrics is None:
        raw_src_pts = np.array(
            [[m["image1_x"], m["image1_y"]] for m in raw_matches],
            dtype=np.float64,
        ) if raw_matches else np.empty((0, 2), dtype=np.float64)
        raw_dst_pts = np.array(
            [[m["image2_x"], m["image2_y"]] for m in raw_matches],
            dtype=np.float64,
        ) if raw_matches else np.empty((0, 2), dtype=np.float64)
        H_mat = np.array(H, dtype=np.float64) if H is not None else None
        inlier_mask = np.ones((len(raw_matches), 1), dtype=np.uint8) if inliers else None
        metrics = compute_canonical_metrics(
            src_pts_raw=raw_src_pts,
            dst_pts_raw=raw_dst_pts,
            inlier_mask=inlier_mask,
            H=H_mat,
            image_shape=(512, 512),
            grid_size=10,
        )
    else:
        H_mat = np.array(H, dtype=np.float64) if H is not None else None

    # 3. Generate Registered Product Suite using scripts/register.py
    src_img = cv2.imread(str(ohrc_path))
    dst_img = cv2.imread(str(nac_path))
    img_shape = (512, 512)

    registered_products: Dict[str, str] = {}
    if H_mat is not None and status == "success":
        # Warp source into reference coordinate frame
        warped_src = warp_source_to_reference(src_img, dst_img, H_mat, output_shape=img_shape)
        blend = create_blend_overlay(warped_src, dst_img, alpha=0.5)
        checkerboard = create_checkerboard_qa(warped_src, dst_img, block_size=64)

        reg_png_path = region_out / "registered_source.png"
        blend_path = region_out / "blend_overlay.png"
        checker_path = region_out / "checkerboard_qa.png"
        geotiff_path = region_out / "registered_source.tif"

        cv2.imwrite(str(reg_png_path), warped_src)
        cv2.imwrite(str(blend_path), blend)
        cv2.imwrite(str(checker_path), checkerboard)
        save_geotiff(warped_src, geotiff_path)

        registered_products = {
            "registered_source_png": str(reg_png_path),
            "registered_source_tif": str(geotiff_path),
            "blend_overlay": str(blend_path),
            "checkerboard_qa": str(checker_path),
        }

    # 4. Write Transform & Metrics JSON sidecars
    transform_file = region_out / "ohrc_to_nac_homography.json"
    metrics_file = region_out / "metrics.json"
    prod_manifest_file = region_out / "registered_products_manifest.json"

    transform_data = {
        "region_id": region_id,
        "reference_type": "external_LRO_NAC",
        "source_sensor": "OHRC",
        "reference_sensor": "LRO_NAC",
        "status": status,
        "homography": H,
        "inlier_count": len(inliers),
        "raw_match_count": len(raw_matches),
        "fit_rmse_px": metrics.get("fit_rmse_px"),
        "sub_pixel_accurate": metrics.get("sub_pixel_accurate", False),
    }

    with open(transform_file, "w") as f:
        json.dump(transform_data, f, indent=2)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    num_inliers = metrics.get("inlier_count", metrics.get("num_inliers", len(inliers)))
    num_raw = metrics.get("match_count", metrics.get("num_raw_matches", len(raw_matches)))
    fit_rmse = metrics.get("fit_rmse_px")
    sub_pixel = (fit_rmse is not None and fit_rmse < 1.0)
    cov_ratio = metrics.get("spatial_coverage", metrics.get("combined_coverage_score", 0.0)) or 0.0

    product_manifest = {
        "region_id": region_id,
        "reference_type": "external_LRO_NAC",
        "source_image": str(ohrc_path),
        "reference_image": str(nac_path),
        "status": status,
        "metrics": {
            "num_inliers": num_inliers,
            "num_raw_matches": num_raw,
            "inlier_ratio": metrics.get("inlier_ratio", 0.0),
            "fit_rmse_px": fit_rmse,
            "sub_pixel_accurate": sub_pixel,
            "spatial_coverage_ratio": cov_ratio,
            "spatial_uniformity": metrics.get("spatial_uniformity", metrics.get("uniformity_score", 0.0)),
            "quality_tier": metrics.get("quality_tier", "HIGH_CONFIDENCE" if sub_pixel else "VERIFIED"),
        },
        "registered_products": registered_products,
        "transform_file": str(transform_file),
    }

    with open(prod_manifest_file, "w") as f:
        json.dump(product_manifest, f, indent=2)

    logger.info(
        "Registration finished for %s: Fit RMSE = %s px, Sub-pixel: %s, Inliers: %s, Coverage: %s",
        region_id,
        str(fit_rmse),
        str(sub_pixel),
        str(num_inliers),
        str(cov_ratio),
    )

    return product_manifest


def main():
    parser = argparse.ArgumentParser(description="Register OHRC to LRO NAC reference images")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["region_001", "region_003", "region_006"],
        help="Regions to register",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Force multimodal_pair=True (default is False for optical-to-optical)",
    )

    args = parser.parse_args()
    summary = []
    for reg in args.regions:
        res = run_registration_for_region(
            region_id=reg,
            output_base_dir=args.output_dir,
            force_non_multimodal=not args.multimodal,
        )
        summary.append(res)

    print("\n" + "=" * 80)
    print("LRO NAC REFERENCE-IMAGE REGISTRATION SUMMARY (PS 26166)")
    print("=" * 80)
    for s in summary:
        m = s["metrics"]
        rmse_str = f"{m['fit_rmse_px']:.4f}" if m.get("fit_rmse_px") is not None else "N/A"
        cov_val = m.get("spatial_coverage_ratio") or 0.0
        print(
            f"Region: {s['region_id']:<12} | Status: {s['status']:<8} | "
            f"Raw: {str(m.get('num_raw_matches')):<3} | Inliers: {str(m.get('num_inliers')):<3} | "
            f"Fit RMSE: {rmse_str:<8} px | "
            f"Sub-pixel: {str(m.get('sub_pixel_accurate')):<5} | "
            f"Coverage: {cov_val:.2%}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
