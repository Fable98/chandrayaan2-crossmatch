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
    create_displacement_quiver,
    save_geotiff,
    bounds_to_eqc_transform,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("register_lro_nac")

PAIRS_DIR = (
    REPO_ROOT / "data_preprocessing_pipeline" / "lro_nac_real"
    if (REPO_ROOT / "data_preprocessing_pipeline" / "lro_nac_real").exists()
    else REPO_ROOT / "data_preprocessing_pipeline" / "lro_nac_pairs"
)
REG_OUT_DIR = REPO_ROOT / "registration_output" / "lro_nac"


def run_registration_for_region(
    region_id: str,
    output_base_dir: Optional[Path | str] = None,
    force_non_multimodal: bool = False,
) -> Dict[str, Any]:
    """
    Executes registration between OHRC (source) and LRO NAC (reference).
    Defaults to the multimodal MI path (required on real CDRs); pass
    force_non_multimodal=True only for synthetic-proxy NCC runs.
    """
    pair_dir = PAIRS_DIR / region_id
    if not pair_dir.exists():
        raise FileNotFoundError(f"Pair directory not found: {pair_dir}. Run prepare_lro_nac_pair.py first.")

    ohrc_path = pair_dir / "ohrc_source_512.png"
    nac_path = pair_dir / "lro_nac_reference_512.png"
    manifest_path = pair_dir / "manifest.json"

    if not ohrc_path.exists() or not nac_path.exists():
        raise FileNotFoundError(f"Missing images in {pair_dir}")

    # Resolve native GSDs + shared-footprint bounds from the pair manifest.
    # Falls back to sensor specs only when manifest is absent.
    explicit_gsd1 = 0.25
    explicit_gsd2 = 0.914
    reference_provenance = "unknown"
    pair_bounds = None
    try:
        if manifest_path.exists():
            with open(manifest_path) as _mf:
                _mdata = json.load(_mf)
            explicit_gsd1 = float(_mdata.get("ohrc_native_gsd_m", explicit_gsd1))
            explicit_gsd2 = float(_mdata.get("lro_nac_native_gsd_m", explicit_gsd2))
            reference_provenance = _mdata.get(
                "reference_provenance", _mdata.get("provenance", "unknown")
            )
            pair_bounds = (_mdata.get("bounds_optical") or _mdata.get("bounds"))
    except Exception as exc:
        logger.warning("Could not parse pair manifest %s: %s; using defaults", manifest_path, exc)

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
    # Use manifest native GSDs so the engine normalizes by the true physical
    # scale ratio (~0.25 vs ~0.9-1.1 => ~3.6-4.5x), not a forced 1.0/1.0.
    logger.info(
        "Native GSDs from manifest: OHRC=%.3fm, LRO_NAC=%.3fm (provenance=%s)",
        explicit_gsd1, explicit_gsd2, reference_provenance,
    )
    match_result = match_images_cfog(
        img_path1=str(ohrc_path),
        img_path2=str(nac_path),
        output_dir=str(temp_cfog_out),
        source_sensor="OHRC",
        reference_sensor="LRO_NAC",
        explicit_gsd1=explicit_gsd1,
        explicit_gsd2=explicit_gsd2,
        multimodal_pair=False if force_non_multimodal else True,
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
    # Native reference shape (tiles need not be square, e.g. 317x512 west-crop).
    img_shape = (int(dst_img.shape[1]), int(dst_img.shape[0]))

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
        matches_path = region_out / "matches.json"
        transform_path = region_out / "transform.json"

        cv2.imwrite(str(reg_png_path), warped_src)
        cv2.imwrite(str(blend_path), blend)
        cv2.imwrite(str(checker_path), checkerboard)
        quiver_lro_path = region_out / "displacement_quiver.png"
        try:
            _qin = np.array([[m.get("image1_x", m.get("source_x", 0)),
                              m.get("image1_y", m.get("source_y", 0))] for m in inliers],
                            dtype=np.float64)
            _qout = np.array([[m.get("image2_x", m.get("target_x", 0)),
                               m.get("image2_y", m.get("target_y", 0))] for m in inliers],
                             dtype=np.float64)
            if len(_qin) < 3 or H_mat is None:
                raise ValueError("insufficient inliers for quiver")
            if create_displacement_quiver(_qin, _qout, np.asarray(H_mat, dtype=np.float64),
                                          (dst_img.shape[0], dst_img.shape[1]),
                                          quiver_lro_path) is None:
                quiver_lro_path = None
        except Exception as exc:
            logger.warning("Quiver QA skipped for %s: %s", region_id, exc)
            quiver_lro_path = None
        # Georeference from pair-manifest shared-footprint bounds (lunar EQC).
        _lro_geo = bounds_to_eqc_transform(
            pair_bounds, warped_src.shape[1], warped_src.shape[0]) if pair_bounds else None
        if _lro_geo is not None:
            saved_tif = save_geotiff(warped_src, geotiff_path,
                                     transform=_lro_geo[0], crs=_lro_geo[1])
        else:
            saved_tif = save_geotiff(warped_src, geotiff_path)
        if saved_tif is None or not geotiff_path.exists():
            # Fallback: plain TIFF via OpenCV so the product always exists;
            # record that it is not georeferenced in the manifest.
            cv2.imwrite(str(geotiff_path), warped_src)
            saved_tif = str(geotiff_path)
            geotiff_georeferenced = False
        else:
            geotiff_georeferenced = True

        # Persist canonical match points (measured inliers) alongside products.
        try:
            with open(matches_path, "w") as _mf:
                json.dump(
                    [
                        {
                            "image1_x": float(m.get("image1_x", m.get("source_x", 0))),
                            "image1_y": float(m.get("image1_y", m.get("source_y", 0))),
                            "image2_x": float(m.get("image2_x", m.get("target_x", 0))),
                            "image2_y": float(m.get("image2_y", m.get("target_y", 0))),
                            "confidence": float(m.get("confidence", 1.0)),
                            "is_inlier": True,
                        }
                        for m in inliers
                    ],
                    _mf,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Failed to write matches.json for %s: %s", region_id, exc)
            matches_path = None

        # Canonical transform.json (model + matrix) in addition to legacy name.
        try:
            with open(transform_path, "w") as _tf:
                json.dump({"model": "homography", "matrix": H_mat.tolist()}, _tf, indent=2)
        except Exception as exc:
            logger.warning("Failed to write transform.json for %s: %s", region_id, exc)

        registered_products = {
            "registered_source_png": str(reg_png_path),
            "registered_source_tif": str(geotiff_path),
            "blend_overlay": str(blend_path),
            "checkerboard_qa": str(checker_path),
            "displacement_quiver": str(quiver_lro_path) if quiver_lro_path is not None else None,
        }

    # 4. Write Transform & Metrics JSON sidecars
    transform_file = region_out / "ohrc_to_nac_homography.json"
    metrics_file = region_out / "metrics.json"
    prod_manifest_file = region_out / "registered_products_manifest.json"

    transform_data = {
        "region_id": region_id,
        "reference_type": "external_LRO_NAC",
        "reference_provenance": reference_provenance,
        "provenance_warning": (
            "SYNTHETIC OHRC-derived proxy; not a real LRO CDR. Replace with real CDR for flight validation."
            if reference_provenance == "synthetic_ohrc_derived_proxy" else None
        ),
        "source_sensor": "OHRC",
        "reference_sensor": "LRO_NAC",
        "ohrc_native_gsd_m": explicit_gsd1,
        "lro_nac_native_gsd_m": explicit_gsd2,
        "scale_ratio": round(explicit_gsd2 / max(explicit_gsd1, 1e-9), 2),
        "status": status,
        "homography": H,
        "inlier_count": len(inliers),
        "raw_match_count": len(raw_matches),
        "fit_rmse_px": metrics.get("fit_rmse_px"),
        "fit_rmse_is_in_sample": True,
        "validation_rmse_px": metrics.get("held_out_validation_rmse_px", metrics.get("validation_rmse_px")),
        "sub_pixel_accurate": metrics.get("sub_pixel_accurate", False),
    }

    with open(transform_file, "w") as f:
        json.dump(transform_data, f, indent=2)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    num_inliers = metrics.get("inlier_count", metrics.get("num_inliers", len(inliers)))
    num_raw = metrics.get("match_count", metrics.get("num_raw_matches", len(raw_matches)))
    fit_rmse = metrics.get("fit_rmse_px")
    val_rmse = metrics.get("held_out_validation_rmse_px", metrics.get("validation_rmse_px"))
    sub_pixel = (fit_rmse is not None and fit_rmse < 1.0)
    cov_ratio = metrics.get("spatial_coverage", metrics.get("combined_coverage_score", 0.0)) or 0.0

    product_manifest = {
        "region_id": region_id,
        "reference_type": "external_LRO_NAC",
        "reference_provenance": reference_provenance,
        "ohrc_native_gsd_m": explicit_gsd1,
        "lro_nac_native_gsd_m": explicit_gsd2,
        "geotiff_georeferenced": registered_products.get("registered_source_tif") is not None and locals().get("geotiff_georeferenced", True),
        "georeferencing_method": "manifest_bounds_eqc" if pair_bounds else "pixel_grid_fallback",
        "georeferencing_note": ("Lunar EQC from pair-manifest shared-footprint bounds "
                                "(partial-overlap caveat in manifest); per-pixel SPICE rigor not claimed."
                                if pair_bounds else
                                "GeoTIFF uses reference CRS/transform when available; otherwise pixel-grid fallback from_origin(). Not a PDS/SPICE rigorous georeference."),
        "source_image": str(ohrc_path),
        "reference_image": str(nac_path),
        "status": status,
        "metrics": {
            "num_inliers": num_inliers,
            "num_raw_matches": num_raw,
            "inlier_ratio": metrics.get("inlier_ratio", 0.0),
            "fit_rmse_px": fit_rmse,
            "fit_rmse_insample_px": metrics.get("fit_rmse_insample_px", fit_rmse),
            "validation_rmse_px": val_rmse,
            "sub_pixel_accurate": sub_pixel,
            "spatial_coverage_ratio": cov_ratio,
            "spatial_uniformity": metrics.get("spatial_uniformity", metrics.get("uniformity_score", 0.0)),
            "quality_tier": metrics.get("quality_tier") if status == "success" else "FAILED",
        },
        "registered_products": registered_products,
        "transform_file": str(transform_file),
    }

    # On failure the engine metrics may be None: record an explicit FAILED stub
    # instead of writing nulls that downstream readers mistake for data.
    if metrics is None:
        with open(metrics_file, "w") as f:
            json.dump({"status": status, "inlier_count": 0, "match_count": len(raw_matches),
                       "quality_tier": "FAILED",
                       "note": "No metrics: registration did not succeed; see transform_file status."},
                      f, indent=2)

    with open(prod_manifest_file, "w") as f:
        json.dump(product_manifest, f, indent=2)

    logger.info(
        "Registration finished for %s: Fit RMSE = %s px, Val RMSE = %s px, Sub-pixel: %s, Inliers: %s, Coverage: %s",
        region_id,
        str(fit_rmse),
        str(val_rmse),
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
    # Measured 2026-09-10 on real CDRs: the optical-NCC path finds 0 candidates
    # under true ~104-132deg sun gaps; the multimodal MI path is REQUIRED
    # (001: 32/6@0.50, 003: 27/5@0.60, 006: 24/5@0.18 with MI vs 0/NCC).
    # MI is therefore the default; --optical-ncc restores the proxy-era path.
    parser.add_argument(
        "--optical-ncc",
        action="store_true",
        help="Force multimodal_pair=False (optical NCC; only works on synthetic proxies)",
    )

    args = parser.parse_args()
    regions = list(args.regions)
    # Step 14: --regions all-real expands from the CDR registry (validated
    # real_downloaded_cdr entries only; pending slots are never scheduled).
    if len(regions) == 1 and regions[0] == "all-real":
        registry_path = REPO_ROOT / "data_preprocessing_pipeline" / "lro_cdr_registry.json"
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            regions = [e["region_id"] for e in registry.get("real_cdrs", [])
                       if e.get("status") == "real_downloaded_cdr" and e.get("region_id")]
            logger.info("Registry expansion: %d real CDR region(s): %s", len(regions), regions)
        except Exception as exc:
            logger.error("Could not read CDR registry (%s); aborting.", exc)
            return
        if not regions:
            logger.error("Registry lists zero real CDRs; nothing to run.")
            return
    summary = []
    for reg in regions:
        res = run_registration_for_region(
            region_id=reg,
            output_base_dir=args.output_dir,
            force_non_multimodal=args.optical_ncc,
        )
        summary.append(res)

    print("\n" + "=" * 98)
    print("LRO NAC REFERENCE-IMAGE REGISTRATION SUMMARY (PS 26166)")
    print("=" * 98)
    for s in summary:
        m = s["metrics"]
        fit_rmse_str = f"{m['fit_rmse_px']:.4f}" if m.get("fit_rmse_px") is not None else "N/A"
        val_rmse = m.get("validation_rmse_px")
        val_rmse_str = f"{val_rmse:.4f}" if val_rmse is not None else "N/A"
        cov_val = m.get("spatial_coverage_ratio") or 0.0
        print(
            f"Region: {s['region_id']:<10} | Status: {s['status']:<7} | "
            f"Raw: {str(m.get('num_raw_matches')):<2} | Inliers: {str(m.get('num_inliers')):<2} | "
            f"Fit RMSE: {fit_rmse_str:<7} px | Val RMSE: {val_rmse_str:<7} px | "
            f"Sub-pixel: {str(m.get('sub_pixel_accurate')):<5} | "
            f"Coverage: {cov_val:.1%}"
        )
    print("=" * 98)


if __name__ == "__main__":
    main()
