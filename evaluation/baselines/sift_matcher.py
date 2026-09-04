"""
evaluation/baselines/sift_matcher.py — Standard SIFT/ASIFT Feature Matching Baseline
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


def match_sift(
    img1: np.ndarray | str | Path,
    img2: np.ndarray | str | Path,
    gsd_m: float = 5.0,
    dem: Optional[np.ndarray] = None,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0,
) -> Dict[str, Any]:
    """
    Standard SIFT baseline matching wrapper using OpenCV SIFT_create().
    """
    start_time = time.time()

    # Load images
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

    # 1. SIFT Feature Extraction
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(u8_1, None)
    kp2, des2 = sift.detectAndCompute(u8_2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {
            "method": "Pure SIFT",
            "status": "insufficient_features",
            "match_count": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "fit_rmse_px": None,
            "absolute_rmse_m": None,
            "runtime_s": round(time.time() - start_time, 4),
            "matches": [],
        }

    # 2. BFMatcher with Lowe's Ratio Test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        return {
            "method": "Pure SIFT",
            "status": "insufficient_matches",
            "match_count": len(good_matches),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "fit_rmse_px": None,
            "absolute_rmse_m": None,
            "runtime_s": round(time.time() - start_time, 4),
            "matches": [],
        }

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # 3. RANSAC Homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)

    if H is None or mask is None:
        inlier_count = 0
        inlier_ratio = 0.0
        fit_rmse = None
        abs_rmse = None
        status = "ransac_failed"
    else:
        inlier_mask = mask.ravel() == 1
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count / max(1, len(good_matches)))

        if inlier_count >= 4:
            inliers_src = pts1[inlier_mask]
            inliers_dst = pts2[inlier_mask]
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
        "method": "Pure SIFT",
        "status": status,
        "match_count": len(good_matches),
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "fit_rmse_px": round(fit_rmse, 4) if fit_rmse is not None else None,
        "absolute_rmse_m": round(abs_rmse, 4) if abs_rmse is not None else None,
        "runtime_s": round(elapsed, 4),
        "homography": H.tolist() if H is not None else None,
    }
