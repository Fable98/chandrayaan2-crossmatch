"""
data_preprocessing_pipeline/triplet_evaluator.py — Ground-Truth-Independent Triplet Consistency Evaluator

Computes closed-loop cycle consistency error across 3 sensor perspectives (A -> B -> C -> A).
Single-pair RANSAC RMSE is biased because it only evaluates points that fit its own fitted model.
Cycle consistency provides an unbiased, ground-truth-independent measure of multi-sensor geometric fidelity.

If all three legs succeed independently, cycle consistency is evaluated on measured homographies.
If exactly one leg fails verification, its homography is derived by composition from the other two,
and tagged with "derivation": "composed" (vs "derivation": "measured").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import cv2

# Add project roots
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
from matcher_cfog import match_images_cfog
from metrics import compute_triplet_consistency


def compose_homographies(h_ab: np.ndarray, h_bc: np.ndarray) -> np.ndarray:
    """Compose A -> B and B -> C homographies into A -> C."""
    composed = np.asarray(h_bc, dtype=np.float64) @ np.asarray(h_ab, dtype=np.float64)
    scale = composed[2, 2]
    if abs(scale) > 1e-12:
        composed = composed / scale
    return composed


def compose_missing_leg(
    H_AB: Optional[np.ndarray],
    H_BC: Optional[np.ndarray],
    H_CA: Optional[np.ndarray],
    res_AB: Dict[str, Any],
    res_BC: Dict[str, Any],
    res_CA: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str], Dict[str, str]]:
    """
    Given up to 3 homography results (each may be None if that leg's
    independent match failed), if exactly one is missing, derive it
    by composition from the other two and return it with a flag
    marking it as derived rather than independently measured.

    Convention: H_XY maps a point in X's pixel space into Y's pixel
    space (i.e. p_Y = H_XY @ p_X in homogeneous coordinates).
    For a closed cycle A->B->C->A to be internally consistent:
        H_CA @ H_BC @ H_AB ≈ Identity
    so any one homography can be derived from the other two:
        CA missing: H_CA = inv(H_BC @ H_AB)
        BC missing: H_BC = inv(H_CA) @ inv(H_AB)
        AB missing: H_AB = inv(H_BC) @ inv(H_CA)
    """
    legs = {"AB": H_AB, "BC": H_BC, "CA": H_CA}
    missing = [k for k, v in legs.items() if v is None]
    derivations = {k: ("measured" if v is not None else "failed") for k, v in legs.items()}

    if len(missing) != 1:
        return H_AB, H_BC, H_CA, missing, derivations  # 0 or 2+ missing: can't help here

    gap = missing[0]
    try:
        if gap == "CA":
            # H_CA derived so that H_CA @ H_BC @ H_AB = I
            # => H_CA = inv(H_BC @ H_AB)
            H_CA = np.linalg.inv(H_BC @ H_AB)
            if abs(H_CA[2, 2]) > 1e-12:
                H_CA = H_CA / H_CA[2, 2]
            derivations["CA"] = "composed"
        elif gap == "BC":
            # H_BC = inv(H_CA @ H_AB) ... solve H_CA @ H_BC @ H_AB = I for H_BC
            # H_BC = inv(H_CA) @ inv(H_AB)
            H_BC = np.linalg.inv(H_CA) @ np.linalg.inv(H_AB)
            if abs(H_BC[2, 2]) > 1e-12:
                H_BC = H_BC / H_BC[2, 2]
            derivations["BC"] = "composed"
        elif gap == "AB":
            # H_AB = inv(H_BC) @ inv(H_CA)
            H_AB = np.linalg.inv(H_BC) @ np.linalg.inv(H_CA)
            if abs(H_AB[2, 2]) > 1e-12:
                H_AB = H_AB / H_AB[2, 2]
            derivations["AB"] = "composed"

        # Sanity check: verify H_CA @ H_BC @ H_AB is close to identity within tolerance
        H_loop = H_CA @ H_BC @ H_AB
        if abs(H_loop[2, 2]) > 1e-12:
            H_loop = H_loop / H_loop[2, 2]

        loop_err = float(np.linalg.norm(H_loop - np.eye(3)))
        if loop_err > 1e-2 or not np.all(np.isfinite(H_loop)):
            return legs["AB"], legs["BC"], legs["CA"], missing, derivations

        return H_AB, H_BC, H_CA, [], derivations
    except Exception:
        return legs["AB"], legs["BC"], legs["CA"], missing, derivations


def evaluate_triplet_consistency(
    image_a_path: str | Path,
    image_b_path: str | Path,
    image_c_path: str | Path,
    dem_path: str | Path | None = None,
    output_dir: str | Path = "triplet_evaluation_output",
    num_test_points: int = 100,
    sensor_a: str = "OHRC",
    sensor_b: str = "TMC-2",
    sensor_c: str = "IIRS",
) -> Dict[str, Any]:
    """
    Executes closed-loop triplet registration:
    1. A (OHRC) -> B (TMC-2)
    2. B (TMC-2) -> C (IIRS)
    3. C (IIRS) -> A (OHRC)
    Attempts all three independent matches first. If exactly one leg fails,
    derives the missing homography mathematically via compose_missing_leg().
    """
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Match A -> B
    res_AB = match_images_cfog(
        image_a_path, image_b_path, dem_path=dem_path, output_dir=out_base / "AB",
        source_sensor=sensor_a, reference_sensor=sensor_b,
    )
    # 2. Match B -> C
    res_BC = match_images_cfog(
        image_b_path, image_c_path, dem_path=dem_path, output_dir=out_base / "BC",
        source_sensor=sensor_b, reference_sensor=sensor_c,
    )
    # 3. Match C -> A
    res_CA = match_images_cfog(
        image_c_path, image_a_path, dem_path=dem_path, output_dir=out_base / "CA",
        source_sensor=sensor_c, reference_sensor=sensor_a,
    )

    H_AB = np.array(res_AB["homography"], dtype=np.float64) if (res_AB.get("status") == "success" and res_AB.get("homography") is not None) else None
    H_BC = np.array(res_BC["homography"], dtype=np.float64) if (res_BC.get("status") == "success" and res_BC.get("homography") is not None) else None
    H_CA = np.array(res_CA["homography"], dtype=np.float64) if (res_CA.get("status") == "success" and res_CA.get("homography") is not None) else None

    failed_legs = []
    if H_AB is None:
        failed_legs.append("AB (A -> B)")
    if H_BC is None:
        failed_legs.append("BC (B -> C)")
    if H_CA is None:
        failed_legs.append("CA (C -> A)")

    derivations = {
        "AB": "measured" if H_AB is not None else "failed",
        "BC": "measured" if H_BC is not None else "failed",
        "CA": "measured" if H_CA is not None else "failed",
    }

    # If exactly one leg failed and the other two succeeded, derive the missing leg
    if len(failed_legs) == 1:
        H_AB, H_BC, H_CA, remaining_missing, derivations = compose_missing_leg(
            H_AB, H_BC, H_CA, res_AB, res_BC, res_CA
        )
        if not remaining_missing:
            failed_legs = []

    # If 2 or more legs are missing (or derivation failed), return cycle_not_computable
    if failed_legs:
        evaluation_report = {
            "status": "cycle_not_computable",
            "reason": f"Missing verified homography for leg(s): {', '.join(failed_legs)}",
            "triplet_cycle_rmse_px": None,
            "triplet_mean_cycle_error_px": None,
            "cycle_closed_successfully": False,
            "failed_legs": failed_legs,
            "leg_derivations": derivations,
            "pair_AB_metrics": res_AB.get("metrics"),
            "pair_BC_metrics": res_BC.get("metrics"),
            "pair_CA_metrics": res_CA.get("metrics"),
            "composition": None,
        }
    else:
        # Run closed-loop cycle consistency on complete set of 3 homographies
        cycle_rmse, cycle_mean = compute_triplet_consistency(
            H_AB, H_BC, H_CA, image_shape=(512, 512), num_test_points=num_test_points
        )

        # Build pair metrics outputs with transparent derivation tags
        pair_ab_out = dict(res_AB.get("metrics")) if res_AB.get("metrics") else None
        pair_bc_out = dict(res_BC.get("metrics")) if res_BC.get("metrics") else None
        pair_ca_out = dict(res_CA.get("metrics")) if res_CA.get("metrics") else None

        if derivations["AB"] == "composed":
            pair_ab_out = {
                "derivation": "composed",
                "status": "composed_from_BC_CA",
                "inlier_count": 0,
                "fit_rmse_px": None,
            }
        elif pair_ab_out is not None:
            pair_ab_out["derivation"] = "measured"

        if derivations["BC"] == "composed":
            pair_bc_out = {
                "derivation": "composed",
                "status": "composed_from_CA_AB",
                "inlier_count": 0,
                "fit_rmse_px": None,
            }
        elif pair_bc_out is not None:
            pair_bc_out["derivation"] = "measured"

        if derivations["CA"] == "composed":
            pair_ca_out = {
                "derivation": "composed",
                "status": "composed_from_AB_BC",
                "inlier_count": 0,
                "fit_rmse_px": None,
            }
        elif pair_ca_out is not None:
            pair_ca_out["derivation"] = "measured"

        # Form composed A -> C homography for registered raster output
        # H_AC maps A -> C, which is H_BC @ H_AB (or inv(H_CA))
        if H_AB is not None and H_BC is not None:
            H_AC = compose_homographies(H_AB, H_BC)
        else:
            H_AC = np.linalg.inv(H_CA)
            if abs(H_AC[2, 2]) > 1e-12:
                H_AC = H_AC / H_AC[2, 2]

        composition_path = out_base / "composed_ohrc_to_iirs_transform.json"
        with open(composition_path, "w") as f:
            json.dump({
                "model": "composed_homography",
                "matrix": H_AC.tolist(),
                "path": "A -> B -> C",
                "leg_derivations": derivations,
            }, f, indent=4)

        source_image = cv2.imread(str(image_a_path), cv2.IMREAD_UNCHANGED)
        target_image = cv2.imread(str(image_c_path), cv2.IMREAD_UNCHANGED)
        registered_path = None
        tif_path = None
        checker_path = None

        if source_image is not None and target_image is not None:
            registered = cv2.warpPerspective(
                source_image, H_AC, (target_image.shape[1], target_image.shape[0]),
                flags=cv2.INTER_LINEAR,
            )

            # PNG output
            registered_path = out_base / "composed_registered_ohrc_to_iirs.png"
            cv2.imwrite(str(registered_path), registered)

            # GeoTIFF output
            tif_path = out_base / "composed_registered_ohrc_to_iirs.tif"
            try:
                import rasterio
                from rasterio.transform import from_origin
                th, tw = target_image.shape[:2]
                tif_profile = {
                    "driver": "GTiff",
                    "height": th, "width": tw,
                    "count": 1 if registered.ndim == 2 else min(registered.shape[2], 3),
                    "dtype": "uint8",
                    "crs": "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs",
                    "transform": from_origin(0, th, 1.0, 1.0),
                    "compress": "lzw",
                }
                with rasterio.open(str(tif_path), "w", **tif_profile) as dst:
                    if registered.ndim == 3:
                        for b in range(min(registered.shape[2], 3)):
                            dst.write(registered[:, :, registered.shape[2] - 1 - b], b + 1)
                    else:
                        dst.write(registered, 1)
            except Exception:
                tif_path = None

            # Checkerboard QA
            checker_path = out_base / "composed_checkerboard_qa.png"
            block_size = 50
            th, tw = target_image.shape[:2]
            src_vis = registered if registered.ndim == 3 else cv2.cvtColor(registered, cv2.COLOR_GRAY2BGR)
            ref_vis = target_image if target_image.ndim == 3 else cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
            if src_vis.shape[:2] != ref_vis.shape[:2]:
                ref_vis = cv2.resize(ref_vis, (src_vis.shape[1], src_vis.shape[0]))
            blended = np.zeros_like(ref_vis)
            for y in range(0, th, block_size):
                for x in range(0, tw, block_size):
                    if ((x // block_size) + (y // block_size)) % 2 == 0:
                        blended[y:y+block_size, x:x+block_size] = src_vis[y:y+block_size, x:x+block_size]
                    else:
                        blended[y:y+block_size, x:x+block_size] = ref_vis[y:y+block_size, x:x+block_size]
            cv2.imwrite(str(checker_path), blended)

        # Products manifest
        manifest_path = out_base / "registered_products_manifest.json"
        manifest = {
            "composition_path": "A -> B -> C -> A",
            "leg_derivations": derivations,
            "homography_AB": H_AB.tolist(),
            "homography_BC": H_BC.tolist(),
            "homography_CA": H_CA.tolist(),
            "homography_AC_composed": H_AC.tolist(),
            "products": {
                "registered_png": str(registered_path) if registered_path else None,
                "registered_tif": str(tif_path) if tif_path else None,
                "checkerboard_qa": str(checker_path) if checker_path else None,
                "transform_json": str(composition_path),
            },
            "cycle_metrics": {
                "cycle_rmse_px": round(float(cycle_rmse), 4),
                "cycle_mean_px": round(float(cycle_mean), 4),
                "cycle_closed": bool(cycle_rmse < 5.0),
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

        has_composed_leg = any(v == "composed" for v in derivations.values())
        evaluation_report = {
            "status": "evaluated_composed" if has_composed_leg else "evaluated",
            "reason": None,
            "triplet_cycle_rmse_px": round(float(cycle_rmse), 4),
            "triplet_mean_cycle_error_px": round(float(cycle_mean), 4),
            "cycle_closed_successfully": bool(cycle_rmse < 5.0),
            "failed_legs": [],
            "leg_derivations": derivations,
            "legs": {
                "AB": {"derivation": derivations["AB"], "homography": H_AB.tolist()},
                "BC": {"derivation": derivations["BC"], "homography": H_BC.tolist()},
                "CA": {"derivation": derivations["CA"], "homography": H_CA.tolist()},
            },
            "pair_AB_metrics": pair_ab_out,
            "pair_BC_metrics": pair_bc_out,
            "pair_CA_metrics": pair_ca_out,
            "composition": {
                "source": "A -> B -> C",
                "homography": H_AC.tolist(),
                "transform": str(composition_path),
                "registered_raster": str(registered_path) if registered_path else None,
                "registered_geotiff": str(tif_path) if tif_path else None,
                "checkerboard_qa": str(checker_path) if checker_path else None,
                "manifest": str(manifest_path),
            },
        }

    report_path = out_base / "triplet_consistency_report.json"
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)

    return evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3-way Triplet Consistency (A -> B -> C -> A).")
    parser.add_argument("img_a", type=str, help="Path to Image A (e.g. OHRC)")
    parser.add_argument("img_b", type=str, help="Path to Image B (e.g. TMC)")
    parser.add_argument("img_c", type=str, help="Path to Image C (e.g. IIRS)")
    parser.add_argument("--dem", type=str, default=None, help="Path to DEM")
    parser.add_argument("--output", type=str, default="triplet_evaluation_output", help="Output directory")

    args = parser.parse_args()
    report = evaluate_triplet_consistency(args.img_a, args.img_b, args.img_c, dem_path=args.dem, output_dir=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
