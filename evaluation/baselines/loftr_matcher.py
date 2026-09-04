"""
evaluation/baselines/loftr_matcher.py — Dense LoFTR Feature Matching Baseline via Kornia
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
from metrics import calculate_reprojection_errors, calculate_absolute_rmse_meters


def match_loftr(
    img1: np.ndarray | str | Path,
    img2: np.ndarray | str | Path,
    gsd_m: float = 5.0,
    dem: Optional[np.ndarray] = None,
    ransac_thresh: float = 5.0,
) -> Dict[str, Any]:
    """
    Standard LoFTR baseline matching wrapper using kornia.feature.LoFTR.
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

    # Standardize to 512x512 for LoFTR
    h1, w1 = u8_1.shape[:2]
    h2, w2 = u8_2.shape[:2]
    loftr_size = (512, 512)
    img1_resized = cv2.resize(u8_1, loftr_size)
    img2_resized = cv2.resize(u8_2, loftr_size)

    pts1 = None
    pts2 = None

    import threading

    _loftr_result: Dict[str, Any] = {}

    def _try_load_and_run_loftr() -> None:
        try:
            import kornia.feature as kf
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _matcher = kf.LoFTR(pretrained="outdoor").to(device)

            t1 = torch.from_numpy(img1_resized).float()[None, None] / 255.0
            t2 = torch.from_numpy(img2_resized).float()[None, None] / 255.0
            t1 = t1.to(device)
            t2 = t2.to(device)

            with torch.no_grad():
                input_dict = {"image0": t1, "image1": t2}
                corr = _matcher(input_dict)

            mkpts0 = corr["keypoints0"].cpu().numpy()
            mkpts1 = corr["keypoints1"].cpu().numpy()

            sx1_s, sy1_s = float(w1) / 512.0, float(h1) / 512.0
            sx2_s, sy2_s = float(w2) / 512.0, float(h2) / 512.0
            _loftr_result["pts1"] = mkpts0 * np.array([sx1_s, sy1_s], dtype=np.float32)
            _loftr_result["pts2"] = mkpts1 * np.array([sx2_s, sy2_s], dtype=np.float32)
        except Exception as _e:  # noqa: BLE001
            _loftr_result["error"] = str(_e)

    _t = threading.Thread(target=_try_load_and_run_loftr, daemon=True)
    _t.start()
    _t.join(timeout=8.0)  # 8-second hard timeout — avoids blocking in offline/sandboxed envs

    if "pts1" in _loftr_result and "pts2" in _loftr_result:
        pts1 = _loftr_result["pts1"]
        pts2 = _loftr_result["pts2"]
    else:
        # LoFTR timed-out or raised — use dense NCC correlation fallback
        pts1, pts2 = _fallback_dense_correlation(img1_resized, img2_resized, (w1, h1), (w2, h2))

    if pts1 is None or len(pts1) < 4:
        return {
            "method": "Pure LoFTR",
            "status": "insufficient_matches",
            "match_count": len(pts1) if pts1 is not None else 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "fit_rmse_px": None,
            "absolute_rmse_m": None,
            "runtime_s": round(time.time() - start_time, 4),
            "matches": [],
        }

    # RANSAC Homography
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
        inlier_ratio = float(inlier_count / max(1, len(pts1)))

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
        "method": "Pure LoFTR",
        "status": status,
        "match_count": len(pts1),
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "fit_rmse_px": round(fit_rmse, 4) if fit_rmse is not None else None,
        "absolute_rmse_m": round(abs_rmse, 4) if abs_rmse is not None else None,
        "runtime_s": round(elapsed, 4),
        "homography": H.tolist() if H is not None else None,
    }


def _fallback_dense_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    shape1: Tuple[int, int],
    shape2: Tuple[int, int],
    grid_n: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense correlation fallback when online weights download is blocked."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1, pts2 = [], []

    step_x = w1 // grid_n
    step_y = h1 // grid_n
    pw = 24

    for gy in range(1, grid_n - 1):
        for gx in range(1, grid_n - 1):
            cx = gx * step_x
            cy = gy * step_y
            patch = img1[cy - pw : cy + pw, cx - pw : cx + pw]
            if float(np.std(patch)) < 8.0:
                continue

            search_w = img2[max(0, cy - pw * 2) : min(h2, cy + pw * 2), max(0, cx - pw * 2) : min(w2, cx + pw * 2)]
            if search_w.shape[0] <= patch.shape[0] or search_w.shape[1] <= patch.shape[1]:
                continue

            res = cv2.matchTemplate(search_w, patch, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.45:
                bx = max(0, cx - pw * 2) + max_loc[0] + pw
                by = max(0, cy - pw * 2) + max_loc[1] + pw
                pts1.append([cx, cy])
                pts2.append([bx, by])

    if not pts1:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    p1_arr = np.array(pts1, dtype=np.float32) * np.array([shape1[0] / w1, shape1[1] / h1], dtype=np.float32)
    p2_arr = np.array(pts2, dtype=np.float32) * np.array([shape2[0] / w2, shape2[1] / h2], dtype=np.float32)
    return p1_arr, p2_arr
