"""
evaluation/baselines/ncc_matcher.py — Standard Normalized Cross-Correlation (NCC) Baseline
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
from metrics import calculate_reprojection_errors, calculate_absolute_rmse_meters


def match_ncc(
    img1: np.ndarray | str | Path,
    img2: np.ndarray | str | Path,
    gsd_m: float = 5.0,
    dem: Optional[np.ndarray] = None,
    grid_size: int = 12,
    patch_radius: int = 24,
    search_radius: int = 64,
    ncc_thresh: float = 0.50,
    ransac_thresh: float = 5.0,
) -> Dict[str, Any]:
    """
    Standard area-based Normalized Cross-Correlation (NCC) intensity template matcher baseline.
    """
    start_time = time.time()

    def to_u8_gray(item):
        if isinstance(item, (str, Path)):
            raw = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise ValueError(f"Could not read image: {item}")
            return raw
        arr = np.asarray(item)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr / 255.0 if arr.max() > 1.0 else arr, 0, 1) * 255.0).astype(np.uint8)
        return arr

    u8_1 = to_u8_gray(img1)
    u8_2 = to_u8_gray(img2)

    h1, w1 = u8_1.shape[:2]
    h2, w2 = u8_2.shape[:2]

    step_x = w1 // grid_size
    step_y = h1 // grid_size

    pts1 = []
    pts2 = []

    for gy in range(1, grid_size - 1):
        for gx in range(1, grid_size - 1):
            cx = gx * step_x
            cy = gy * step_y

            # Source patch
            if (cy - patch_radius < 0 or cy + patch_radius >= h1 or
                cx - patch_radius < 0 or cx + patch_radius >= w1):
                continue

            tmpl = u8_1[cy - patch_radius : cy + patch_radius, cx - patch_radius : cx + patch_radius]
            if float(np.std(tmpl)) < 5.0:
                continue

            # Expected target center (scaled)
            cx2 = int(cx * (w2 / float(w1)))
            cy2 = int(cy * (h2 / float(h1)))

            s_min_x = max(0, cx2 - search_radius)
            s_max_x = min(w2, cx2 + search_radius)
            s_min_y = max(0, cy2 - search_radius)
            s_max_y = min(h2, cy2 + search_radius)

            search_region = u8_2[s_min_y:s_max_y, s_min_x:s_max_x]
            if search_region.shape[0] <= tmpl.shape[0] or search_region.shape[1] <= tmpl.shape[1]:
                continue

            res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val >= ncc_thresh:
                best_x2 = s_min_x + max_loc[0] + patch_radius
                best_y2 = s_min_y + max_loc[1] + patch_radius
                pts1.append([cx, cy])
                pts2.append([best_x2, best_y2])

    if len(pts1) < 4:
        return {
            "method": "Standard NCC",
            "status": "insufficient_matches",
            "match_count": len(pts1),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "fit_rmse_px": None,
            "absolute_rmse_m": None,
            "runtime_s": round(time.time() - start_time, 4),
            "matches": [],
        }

    pts1_arr = np.array(pts1, dtype=np.float32)
    pts2_arr = np.array(pts2, dtype=np.float32)

    H, mask = cv2.findHomography(pts1_arr, pts2_arr, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)

    if H is None or mask is None:
        inlier_count = 0
        inlier_ratio = 0.0
        fit_rmse = None
        abs_rmse = None
        status = "ransac_failed"
    else:
        inlier_mask = mask.ravel() == 1
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count / max(1, len(pts1)))

        if inlier_count >= 4:
            inliers_src = pts1_arr[inlier_mask]
            inliers_dst = pts2_arr[inlier_mask]
            errors = calculate_reprojection_errors(inliers_src, inliers_dst, H)
            fit_rmse = float(np.sqrt(np.mean(errors**2)))
            abs_rmse = calculate_absolute_rmse_meters((inliers_src, inliers_dst, H), gsd=gsd_m, dem_data=dem)
            status = "success"
        else:
            fit_rmse = None
            abs_rmse = None
            status = "low_inliers"

    elapsed = time.time() - start_time

    return {
        "method": "Standard NCC",
        "status": status,
        "match_count": len(pts1),
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "fit_rmse_px": round(fit_rmse, 4) if fit_rmse is not None else None,
        "absolute_rmse_m": round(abs_rmse, 4) if abs_rmse is not None else None,
        "runtime_s": round(elapsed, 4),
        "homography": H.tolist() if H is not None else None,
    }
