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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matcher_cfog import match_images_cfog, propagate_composed_covariance_monte_carlo
from metrics import compute_triplet_consistency, calculate_absolute_rmse_meters

try:
    from bundle_adjustment import GlobalBundleAdjuster
except Exception:
    try:
        from ML_model.bundle_adjustment import GlobalBundleAdjuster
    except Exception:
        GlobalBundleAdjuster = None  # type: ignore[assignment]


def _extract_inlier_points(res: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Pull (src, dst) inlier arrays from a matcher result dict. Never raises."""
    try:
        matches = res.get("matches") or []
        s: List[List[float]] = []
        d: List[List[float]] = []
        for m in matches:
            try:
                s.append([float(m.get("source_x", m.get("image1_x"))),
                          float(m.get("source_y", m.get("image1_y")))])
                d.append([float(m.get("target_x", m.get("image2_x"))),
                          float(m.get("target_y", m.get("image2_y")))])
            except Exception:
                continue
        if s and d:
            return np.asarray(s, dtype=np.float64), np.asarray(d, dtype=np.float64)
    except Exception:
        pass
    return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)


def _extract_cov(res: Dict[str, Any]) -> Optional[np.ndarray]:
    """Pull 9x9 homography covariance from a matcher result. None if absent."""
    try:
        for key in ("H_cov", "bootstrap"):
            pass
        cov = res.get("H_cov")
        if cov is None and isinstance(res.get("bootstrap"), dict):
            cov = res["bootstrap"].get("H_cov")
        if cov is None and isinstance(res.get("metrics"), dict):
            cov = res["metrics"].get("H_cov")
        if cov is None:
            return None
        m = np.asarray(cov, dtype=np.float64).reshape(9, 9)
        if np.all(np.isfinite(m)):
            return m
    except Exception:
        pass
    return None


def run_triplet_bundle_adjustment(
    res_AB: Dict[str, Any],
    res_BC: Dict[str, Any],
    res_CA: Dict[str, Any],
    H_AB: Optional[np.ndarray],
    H_BC: Optional[np.ndarray],
    H_CA: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Jointly optimize triplet poses with Huber robust loss. Never raises."""
    try:
        if GlobalBundleAdjuster is None:
            return {"status": "skipped", "reason": "bundle_adjuster_unavailable"}
        s_ab, d_ab = _extract_inlier_points(res_AB)
        s_bc, d_bc = _extract_inlier_points(res_BC)
        s_ca, d_ca = _extract_inlier_points(res_CA)
        adjuster = GlobalBundleAdjuster(robust_loss="huber", huber_delta=1.0)
        if H_AB is not None and len(s_ab) >= 4:
            adjuster.add_pairwise_constraint("A", "B", s_ab, d_ab, np.asarray(H_AB, dtype=np.float64))
        if H_BC is not None and len(s_bc) >= 4:
            adjuster.add_pairwise_constraint("B", "C", s_bc, d_bc, np.asarray(H_BC, dtype=np.float64))
        if H_CA is not None and len(s_ca) >= 4:
            adjuster.add_pairwise_constraint("C", "A", s_ca, d_ca, np.asarray(H_CA, dtype=np.float64))
        result = adjuster.optimize(max_iterations=200)
        if not isinstance(result, dict):
            return {"status": "failed", "reason": "optimizer_returned_none"}
        # JSON-safe matrices.
        try:
            mats = result.get("optimized_matrices") or result.get("matrices") or {}
            result["optimized_matrices"] = {k: np.asarray(v, dtype=np.float64).tolist() for k, v in mats.items()}
            if "matrices" in result and isinstance(result["matrices"], dict):
                result["matrices"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in result["matrices"].items()}
        except Exception:
            pass
        return result
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

try:
    from data.ingestion.lro_basemap import fetch_lro_basemap
except Exception:
    fetch_lro_basemap = None


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
    lro_basemap: str | Path | None = None,
    lro_bbox: Optional[Tuple[float, float, float, float]] = None,
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
        try:
            _bundle_fail = run_triplet_bundle_adjustment(
                res_AB, res_BC, res_CA, H_AB, H_BC, H_CA
            )
        except Exception as exc:
            _bundle_fail = {"status": "failed", "reason": str(exc)}
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
            "bundle_adjustment": _bundle_fail,
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

        # Joint bundle adjustment over the triplet loop (Step 9.4).
        try:
            bundle_result = run_triplet_bundle_adjustment(
                res_AB, res_BC, res_CA, H_AB, H_BC, H_CA
            )
        except Exception as exc:
            bundle_result = {"status": "failed", "reason": str(exc)}

        # Covariance propagation Sigma_AC for the composed leg (Step 10.1).
        try:
            _cov_ab = _extract_cov(res_AB)
            _cov_bc = _extract_cov(res_BC)
            _gsd_ac = None
            try:
                for _r in (res_AB, res_BC):
                    _m = (_r.get("metrics") or {})
                    if _m.get("absolute_rmse_m") is not None:
                        pass
                _ws = (res_BC.get("working_scale") or {})
                _gsd_ac = _ws.get("gsd_m")
            except Exception:
                _gsd_ac = None
            if _gsd_ac is None:
                _gsd_ac = 5.0
            composed_cov = propagate_composed_covariance_monte_carlo(
                np.asarray(H_AB, dtype=np.float64),
                _cov_ab,
                np.asarray(H_BC, dtype=np.float64),
                _cov_bc,
                n_samples=500,
                gsd_m=_gsd_ac,
            )
        except Exception as exc:
            composed_cov = {"status": "failed", "reason": str(exc),
                            "Sigma_AC": None, "uncertainty_m": None}
        try:
            _sig_ac = composed_cov.get("Sigma_AC")
            _unc_ac = composed_cov.get("uncertainty_m")
        except Exception:
            _sig_ac, _unc_ac = None, None

        composition_path = out_base / "composed_ohrc_to_iirs_transform.json"
        with open(composition_path, "w") as f:
            json.dump({
                "model": "composed_homography",
                "matrix": H_AC.tolist(),
                "Sigma_AC": _sig_ac,
                "spatial_uncertainty_m": _unc_ac,
                "path": "A -> B -> C",
                "leg_derivations": derivations,
            }, f, indent=4)

        # Honest derived-leg accounting: composed OHRC->IIRS is never measured.
        try:
            _n_ab = int((res_AB.get("metrics") or {}).get("inlier_count", 0) or 0)
            _n_bc = int((res_BC.get("metrics") or {}).get("inlier_count", 0) or 0)
            _n_derived = int(_n_ab + _n_bc)
        except Exception:
            _n_derived = 0
        composed_metrics = {
            "num_measured_matches": 0,
            "num_derived_matches": _n_derived,
            "derivation": "composed_via_triplet",
            "uncertainty_m": _unc_ac,
        }
        try:
            with open(out_base / "composed_ohrc_to_iirs_metrics.json", "w") as f:
                json.dump(composed_metrics, f, indent=4)
        except Exception:
            pass

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
                # A derived leg voids closure validation (tautological ~0.0).
                "cycle_rmse_px": (round(float(cycle_rmse), 4)
                                  if not any(v == "composed" for v in derivations.values()) else None),
                "cycle_mean_px": (round(float(cycle_mean), 4)
                                  if not any(v == "composed" for v in derivations.values()) else None),
                "cycle_closed": (bool(cycle_rmse < 5.0)
                                 if not any(v == "composed" for v in derivations.values()) else False),
                "cycle_validation": ("measured_loop"
                                      if not any(v == "composed" for v in derivations.values())
                                      else "not_applicable_derived_leg"),
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

        has_composed_leg = any(v == "composed" for v in derivations.values())
        # A closure error measured WITH a derived leg is tautological (the leg
        # was built from the other two, so ~0.0px "closure" validates nothing).
        # Report numbers only for fully-measured loops; otherwise null + reason.
        if has_composed_leg:
            cycle_rmse_out, cycle_mean_out, closed_out = None, None, False
            cycle_note = ("cycle_validation_not_applicable: a derived leg voids "
                          "closure; H_AC product + covariance above remain usable")
        else:
            cycle_rmse_out = round(float(cycle_rmse), 4)
            cycle_mean_out = round(float(cycle_mean), 4)
            closed_out = bool(cycle_rmse < 5.0)
            cycle_note = None
        evaluation_report = {
            "status": "evaluated_composed" if has_composed_leg else "evaluated",
            "reason": None,
            "triplet_cycle_rmse_px": cycle_rmse_out,
            "triplet_mean_cycle_error_px": cycle_mean_out,
            "cycle_closed_successfully": closed_out,
            "cycle_validation_note": cycle_note,
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
            "bundle_adjustment": bundle_result,
            "composed_covariance": {
                "Sigma_AC": _sig_ac,
                "uncertainty_m": _unc_ac,
            },
            "composed_metrics": composed_metrics,
            "composition": {
                "source": "A -> B -> C",
                "homography": H_AC.tolist(),
                "Sigma_AC": _sig_ac,
                "uncertainty_m": _unc_ac,
                "transform": str(composition_path),
                "registered_raster": str(registered_path) if registered_path else None,
                "registered_geotiff": str(tif_path) if tif_path else None,
                "checkerboard_qa": str(checker_path) if checker_path else None,
            },
        }

    # 4. External Reference Basemap Registration Stage (Master CH2 -> LRO Basemap)
    res_basemap = None
    lro_path_to_use = lro_basemap
    if lro_path_to_use is None and fetch_lro_basemap is not None:
        try:
            bbox = lro_bbox or (-10.0, -9.0, 30.0, 31.0)
            _, lro_meta = fetch_lro_basemap(bbox, out_dir=out_base / "LRO_BASEMAP_CACHE")
            lro_path_to_use = lro_meta.get("image_path")
        except Exception:
            lro_path_to_use = None

    if lro_path_to_use and Path(lro_path_to_use).exists():
        try:
            master_img = image_b_path if Path(image_b_path).exists() else image_a_path
            master_sensor = sensor_b if Path(image_b_path).exists() else sensor_a
            res_basemap = match_images_cfog(
                master_img,
                lro_path_to_use,
                dem_path=dem_path,
                output_dir=out_base / "LRO_BASEMAP",
                source_sensor=master_sensor,
                reference_sensor="LRO_WAC",
            )
        except Exception as e:
            res_basemap = {"status": "failed", "message": str(e), "metrics": None}

    # Extract required evaluation output metrics
    intra_ch2_pixel_rmse = round(float(cycle_rmse), 4) if (not failed_legs and 'cycle_rmse' in locals() and cycle_rmse is not None) else None

    basemap_pixel_rmse = None
    abs_rmse_meters = None
    if res_basemap and res_basemap.get("status") == "success" and res_basemap.get("metrics"):
        basemap_pixel_rmse = res_basemap["metrics"].get("fit_rmse_px")
        abs_rmse_meters = res_basemap["metrics"].get("absolute_rmse_m")

    # If basemap registration was not run or failed, compute absolute RMSE for intra-CH2 using GSD and DEM
    if abs_rmse_meters is None and intra_ch2_pixel_rmse is not None:
        dem_arr = None
        if dem_path and Path(dem_path).exists():
            try:
                dem_arr = cv2.imread(str(dem_path), cv2.IMREAD_UNCHANGED)
            except Exception:
                pass
        abs_rmse_meters = calculate_absolute_rmse_meters(
            intra_ch2_pixel_rmse, gsd=5.0, dem_data=dem_arr
        )

    evaluation_report["Intra-CH2 Pixel RMSE"] = intra_ch2_pixel_rmse
    evaluation_report["Basemap Pixel RMSE"] = basemap_pixel_rmse
    evaluation_report["Absolute RMSE (Meters)"] = abs_rmse_meters
    evaluation_report["basemap_registration"] = {
        "status": res_basemap.get("status") if res_basemap else "skipped",
        "basemap_pixel_rmse": basemap_pixel_rmse,
        "absolute_rmse_m": abs_rmse_meters,
        "inlier_count": res_basemap.get("metrics", {}).get("inlier_count") if (res_basemap and res_basemap.get("metrics")) else 0,
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
