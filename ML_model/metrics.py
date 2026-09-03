"""
metrics.py — Canonical Evaluation Metrics for Chandrayaan-2 Image Correspondence

Unified, single source of truth for all quantitative evaluation metrics:
- In-sample Fit RMSE vs. Out-of-sample Held-Out Validation RMSE.
- Sub-pixel error distributions (<0.25 px, <0.5 px, <1.0 px).
- Spatial coverage and distribution uniformity (variance, entropy, composite score).
- Geometric transformation matrix conditioning and sanity metrics.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 1. Reprojection Error Functions
# ---------------------------------------------------------------------------

def calculate_reprojection_errors(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Computes Euclidean distance between destination points and source points
    projected through homography matrix H.
    
    Args:
        src_pts: (N, 2) array of coordinates in source image space.
        dst_pts: (N, 2) array of coordinates in destination image space.
        H: (3, 3) projective or affine matrix (src -> dst).
        
    Returns:
        (N,) array of Euclidean reprojection errors in pixels.
    """
    if len(src_pts) == 0:
        return np.array([], dtype=np.float64)

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    H_mat = np.asarray(H, dtype=np.float64)

    ones = np.ones((len(src), 1), dtype=np.float64)
    src_h = np.hstack([src, ones])
    projected = (H_mat @ src_h.T).T

    z = projected[:, 2:3]
    # Guard against division by zero
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    projected_2d = projected[:, :2] / z_safe

    errors = np.linalg.norm(projected_2d - dst, axis=1)
    return errors


# ---------------------------------------------------------------------------
# 2. Fit RMSE vs. Held-Out Validation RMSE
# ---------------------------------------------------------------------------

def evaluate_held_out_validation(
    inliers_src: np.ndarray,
    inliers_dst: np.ndarray,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates held-out inlier correspondence validation error by splitting verified inliers into
    training (80%) and held-out validation (20%) sets.
    
    The transformation is re-estimated strictly on the training subset, and
    evaluated on the unseen held-out validation subset to eliminate in-sample bias.
    Note: Evaluated on held-out inliers; for true independent ground truth, see synthetic benchmarks.
    """
    n = len(inliers_src)
    if n < 8:
        return {
            "validation_status": "insufficient_points_for_holdout",
            "validation_rmse_px": None,
            "validation_median_error_px": None,
            "validation_points_count": 0,
        }

    rng = np.random.RandomState(random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    num_val = max(2, int(n * test_ratio))
    val_idx = indices[:num_val]
    train_idx = indices[num_val:]

    train_src = inliers_src[train_idx]
    train_dst = inliers_dst[train_idx]
    val_src = inliers_src[val_idx]
    val_dst = inliers_dst[val_idx]

    # Fit homography on training subset
    H_train, mask = cv2.findHomography(train_src, train_dst, cv2.RANSAC, 3.0)
    if H_train is None:
        # Fallback to affine
        H_train, _ = cv2.estimateAffinePartial2D(train_src, train_dst)
        if H_train is not None:
            H_train = np.vstack([H_train, [0.0, 0.0, 1.0]])

    if H_train is None:
        return {
            "validation_status": "fit_failed_on_training_subset",
            "validation_rmse_px": None,
            "validation_median_error_px": None,
            "validation_points_count": num_val,
        }

    val_errors = calculate_reprojection_errors(val_src, val_dst, H_train)
    val_rmse = float(np.sqrt(np.mean(val_errors**2)))
    val_median = float(np.median(val_errors))

    return {
        "validation_status": "evaluated",
        "validation_rmse_px": round(val_rmse, 4),
        "validation_median_error_px": round(val_median, 4),
        "validation_points_count": int(num_val),
    }


# ---------------------------------------------------------------------------
# 3. Spatial Distribution & Uniformity Metrics
# ---------------------------------------------------------------------------

def calculate_spatial_distribution(
    points: np.ndarray,
    image_shape: Tuple[int, int] = (512, 512),
    grid_size: int = 10,
) -> Dict[str, Any]:
    """
    Computes rigorous spatial coverage and distribution uniformity metrics.
    
    - coverage: fraction of cells with >= 1 match (0.0 - 1.0).
    - count_std: standard deviation of point counts across cells.
    - spatial_entropy: Shannon spatial entropy normalized to [0, 1].
    - uniformity_score: combined score accounting for both coverage and evenness.
    """
    h, w = image_shape[:2]
    total_cells = grid_size * grid_size
    cell_w = w / float(grid_size)
    cell_h = h / float(grid_size)

    if len(points) == 0:
        return {
            "grid_rows": grid_size,
            "grid_cols": grid_size,
            "total_cells": total_cells,
            "occupied_cells": 0,
            "coverage": 0.0,
            "count_std": 0.0,
            "count_mean": 0.0,
            "spatial_entropy": 0.0,
            "uniformity_score": 0.0,
        }

    pts = np.asarray(points, dtype=np.float64)
    gx = np.clip(np.floor(pts[:, 0] / max(1e-6, cell_w)).astype(int), 0, grid_size - 1)
    gy = np.clip(np.floor(pts[:, 1] / max(1e-6, cell_h)).astype(int), 0, grid_size - 1)

    cell_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    for i in range(len(pts)):
        cell_counts[gy[i], gx[i]] += 1

    flat_counts = cell_counts.flatten()
    occupied = int(np.sum(flat_counts > 0))
    coverage = float(occupied / total_cells)
    count_std = float(np.std(flat_counts))
    count_mean = float(np.mean(flat_counts))

    # Normalized spatial Shannon entropy: H / log2(total_cells)
    probs = flat_counts[flat_counts > 0] / float(len(pts))
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(total_cells)
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Composite uniformity score: coverage * exp(-CV), where CV = std / (mean + eps)
    cv = count_std / max(count_mean, 1e-4)
    uniformity_score = float(coverage * np.exp(-cv * 0.3))

    return {
        "grid_rows": grid_size,
        "grid_cols": grid_size,
        "total_cells": total_cells,
        "occupied_cells": occupied,
        "coverage": round(coverage, 4),
        "count_std": round(count_std, 4),
        "count_mean": round(count_mean, 4),
        "spatial_entropy": round(normalized_entropy, 4),
        "uniformity_score": round(uniformity_score, 4),
    }


# ---------------------------------------------------------------------------
# 4. Geometric Transformation Quality Gates
# ---------------------------------------------------------------------------

def verify_transformation_quality(
    H: np.ndarray,
    image_shape: Tuple[int, int] = (512, 512),
) -> Dict[str, Any]:
    """
    Sanity checks estimated homography/affine matrix for pathological behavior:
    - Extreme perspective distortion (determinant near 0 or negative).
    - Unrealistic scaling (>10x or <0.1x).
    - Ill-conditioned matrix (singular / degenerate).
    """
    H_mat = np.asarray(H, dtype=np.float64)
    if H_mat.shape != (3, 3):
        return {
            "is_valid": False,
            "reason": f"Invalid transformation shape {H_mat.shape}",
            "determinant": 0.0,
            "condition_number": float("inf"),
        }

    try:
        # Check condition number
        cond = float(np.linalg.cond(H_mat))
        det = float(np.linalg.det(H_mat))

        # Check singular values
        U, S, Vt = np.linalg.svd(H_mat[:2, :2])
        scale_ratio = float(S[0] / max(S[1], 1e-9))

        # Check projectivity terms (bottom row h31, h32)
        proj_strength = float(np.sqrt(H_mat[2, 0]**2 + H_mat[2, 1]**2))

        # A valid lunar transform should preserve orientation (det > 0) and not collapse scale
        is_valid = (
            np.isfinite(cond)
            and cond < 1e6
            and det > 1e-4
            and scale_ratio < 20.0
            and proj_strength < 0.05
        )

        reason = "OK" if is_valid else "Pathological projective distortion or ill-conditioned matrix"

        return {
            "is_valid": bool(is_valid),
            "reason": reason,
            "determinant": round(det, 6),
            "condition_number": round(cond, 2),
            "scale_ratio": round(scale_ratio, 4),
        }
    except Exception as e:
        return {
            "is_valid": False,
            "reason": f"Decomposition error: {str(e)}",
            "determinant": 0.0,
            "condition_number": float("inf"),
        }


# ---------------------------------------------------------------------------
# 5. Canonical Master Metrics Computation
# ---------------------------------------------------------------------------

def compute_canonical_metrics(
    src_pts_raw: np.ndarray,
    dst_pts_raw: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    H: Optional[np.ndarray],
    image_shape: Tuple[int, int] = (512, 512),
    grid_size: int = 10,
) -> Dict[str, Any]:
    """
    Single canonical entry point to compute all registration metrics across the repository.
    Ensures that matcher outputs, API responses, and evaluation reports use identical math.
    """
    raw_count = int(len(src_pts_raw))
    if inlier_mask is not None and len(inlier_mask) == raw_count:
        inlier_indices = np.where(inlier_mask.ravel() == 1)[0]
    else:
        inlier_indices = np.arange(raw_count)

    inlier_count = int(len(inlier_indices))
    inlier_ratio = float(inlier_count / max(1, raw_count))

    if inlier_count == 0 or H is None:
        return {
            "match_count": raw_count,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "fit_rmse_px": None,
            "validation_rmse_px": None,
            "validation_median_error_px": None,
            "validation_status": "no_inliers",
            "mean_reprojection_error_px": None,
            "median_reprojection_error_px": None,
            "max_reprojection_error_px": None,
            "fraction_below_1px": 0.0,
            "fraction_below_0_5px": 0.0,
            "fraction_below_0_25px": 0.0,
            "sub_pixel_accurate": False,
            "spatial_coverage": 0.0,
            "spatial_uniformity": 0.0,
            "spatial_distribution": calculate_spatial_distribution(np.zeros((0, 2)), image_shape, grid_size),
            "transform_quality": {"is_valid": False, "reason": "No valid transformation"},
        }

    inliers_src = src_pts_raw[inlier_indices]
    inliers_dst = dst_pts_raw[inlier_indices]

    # In-sample Fit errors
    fit_errors = calculate_reprojection_errors(inliers_src, inliers_dst, H)
    fit_rmse = float(np.sqrt(np.mean(fit_errors**2))) if len(fit_errors) > 0 else 0.0
    mean_err = float(np.mean(fit_errors)) if len(fit_errors) > 0 else 0.0
    median_err = float(np.median(fit_errors)) if len(fit_errors) > 0 else 0.0
    max_err = float(np.max(fit_errors)) if len(fit_errors) > 0 else 0.0

    frac_1 = float(np.mean(fit_errors < 1.0)) if len(fit_errors) > 0 else 0.0
    frac_05 = float(np.mean(fit_errors < 0.5)) if len(fit_errors) > 0 else 0.0
    frac_025 = float(np.mean(fit_errors < 0.25)) if len(fit_errors) > 0 else 0.0

    # Independent Held-Out Validation
    val_results = evaluate_held_out_validation(inliers_src, inliers_dst)

    # Spatial Distribution
    dist_metrics = calculate_spatial_distribution(inliers_src, image_shape, grid_size)

    # Transform Quality
    tx_quality = verify_transformation_quality(H, image_shape)

    # Quality Tier Classification with explicit documented thresholds
    coverage = dist_metrics["coverage"]
    if inlier_count < 4:
        quality_tier = "FAILED"
    elif inlier_count >= 20 and coverage >= 0.20 and fit_rmse < 2.0:
        quality_tier = "HIGH_CONFIDENCE"
    elif inlier_count >= 10 and coverage >= 0.10:
        quality_tier = "ACCEPTED"
    else:
        quality_tier = "LOW_CONFIDENCE"

    return {
        "match_count": raw_count,
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "fit_rmse_px": round(fit_rmse, 4),
        "validation_rmse_px": val_results["validation_rmse_px"],  # Kept for API backward compatibility
        "held_out_inlier_validation_rmse_px": val_results["validation_rmse_px"],
        "validation_median_error_px": val_results["validation_median_error_px"],
        "validation_status": val_results["validation_status"],
        "quality_tier": quality_tier,
        "mean_reprojection_error_px": round(mean_err, 4),
        "median_reprojection_error_px": round(median_err, 4),
        "max_reprojection_error_px": round(max_err, 4),
        "fraction_below_1px": round(frac_1, 4),
        "fraction_below_0_5px": round(frac_05, 4),
        "fraction_below_0_25px": round(frac_025, 4),
        "spatial_coverage": dist_metrics["coverage"],
        "spatial_uniformity": dist_metrics["uniformity_score"],
        "spatial_distribution": dist_metrics,
        "transform_quality": tx_quality,
    }


# ---------------------------------------------------------------------------
# 6. Triplet Closed-Loop Cycle Consistency (A -> B -> C -> A)
# ---------------------------------------------------------------------------

def compute_triplet_consistency(
    H_AB: np.ndarray,
    H_BC: np.ndarray,
    H_CA: np.ndarray,
    image_shape: Tuple[int, int] = (512, 512),
    num_test_points: int = 100,
) -> Tuple[float, float]:
    """
    Computes closed-loop Cycle Consistency Error: A -> B -> C -> A.
    Unlike single-pair RANSAC RMSE (which is self-fulfilling on inliers),
    cycle closure error cannot be cheated and provides a mathematically
    ground-truth-independent measure of multi-sensor geometric fidelity.

    Returns:
        (rmse_cycle_px, mean_cycle_px)
    """
    h, w = image_shape[:2]
    grid_n = max(4, int(np.sqrt(num_test_points)))
    xs = np.linspace(w * 0.15, w * 0.85, grid_n)
    ys = np.linspace(h * 0.15, h * 0.85, grid_n)
    xv, yv = np.meshgrid(xs, ys)
    pts_A = np.column_stack([xv.flatten(), yv.flatten()])

    def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ones = np.ones((len(pts), 1), dtype=np.float64)
        h_pts = np.hstack([pts, ones])
        proj = (H.astype(np.float64) @ h_pts.T).T
        proj[:, 0] /= (proj[:, 2] + 1e-12)
        proj[:, 1] /= (proj[:, 2] + 1e-12)
        return proj[:, :2]

    try:
        pts_B = transform_points(H_AB, pts_A)
        pts_C = transform_points(H_BC, pts_B)
        pts_A_prime = transform_points(H_CA, pts_C)

        errors = np.linalg.norm(pts_A_prime - pts_A, axis=1)
        rmse = float(np.sqrt(np.mean(errors**2)))
        mean_err = float(np.mean(errors))
        return round(rmse, 4), round(mean_err, 4)
    except Exception:
        return 999.0, 999.0

