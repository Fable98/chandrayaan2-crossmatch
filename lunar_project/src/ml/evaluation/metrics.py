"""
metrics.py — Evaluation metrics for Chandrayaan-2 image correspondence.

Provides all the metrics required by the problem statement:
  - Reprojection error (per-point and mean)
  - RMSE (Root Mean Square Error)
  - Inlier count and inlier ratio
  - Spatial coverage (uniform distribution metric)
  - Sub-pixel accuracy verification

All functions operate on numpy arrays of matched point pairs and
can be called standalone or via compute_all_metrics().
"""

from __future__ import annotations

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Per-point reprojection error
# ---------------------------------------------------------------------------

def reprojection_errors(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Compute per-point reprojection error.

    Given source points, destination points, and the homography H that maps
    source → destination, project each source point via H and measure the
    Euclidean distance to the corresponding destination point.

    Args:
        src_pts: (N, 2) array of source pixel coordinates (e.g. OHRC).
        dst_pts: (N, 2) array of destination pixel coordinates (e.g. TMC).
        H: (3, 3) homography matrix (src → dst).

    Returns:
        (N,) array of Euclidean reprojection errors in pixels.
    """
    if len(src_pts) == 0:
        return np.array([], dtype=np.float64)

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    H_mat = np.asarray(H, dtype=np.float64)

    # Project source points through homography
    ones = np.ones((len(src), 1), dtype=np.float64)
    src_h = np.hstack([src, ones])  # (N, 3)
    projected = (H_mat @ src_h.T).T  # (N, 3)

    # Normalize homogeneous coordinates
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    projected_2d = projected[:, :2]

    # Euclidean distance to actual destination points
    errors = np.linalg.norm(projected_2d - dst, axis=1)
    return errors


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def rmse(errors: np.ndarray) -> float:
    """Root Mean Square Error from an array of per-point errors."""
    if len(errors) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(errors ** 2)))


def mean_error(errors: np.ndarray) -> float:
    """Mean reprojection error from an array of per-point errors."""
    if len(errors) == 0:
        return float("inf")
    return float(np.mean(errors))


def max_error(errors: np.ndarray) -> float:
    """Maximum reprojection error from an array of per-point errors."""
    if len(errors) == 0:
        return float("inf")
    return float(np.max(errors))


def median_error(errors: np.ndarray) -> float:
    """Median reprojection error from an array of per-point errors."""
    if len(errors) == 0:
        return float("inf")
    return float(np.median(errors))


# ---------------------------------------------------------------------------
# Inlier metrics
# ---------------------------------------------------------------------------

def inlier_count(num_inliers: int) -> int:
    """Number of geometrically verified (post-RANSAC) matches."""
    return num_inliers


def inlier_ratio(num_inliers: int, num_raw: int) -> float:
    """Ratio of RANSAC inliers to total raw matches (0.0–1.0)."""
    if num_raw == 0:
        return 0.0
    return round(num_inliers / num_raw, 4)


# ---------------------------------------------------------------------------
# Spatial distribution metrics
# ---------------------------------------------------------------------------

def spatial_coverage(
    points: np.ndarray,
    image_size: int = 512,
    grid_size: int = 8,
) -> dict:
    """
    Measure spatial uniformity of match points across the image.

    Divides the image into a grid_size × grid_size grid and reports:
      - coverage_ratio: fraction of cells with ≥1 match point
      - occupied_cells: number of cells with ≥1 match
      - total_cells: total number of grid cells
      - points_per_cell: list of match counts per cell (row-major)
      - std_per_cell: standard deviation of points per cell
        (lower = more uniform distribution)

    Args:
        points: (N, 2) array of pixel coordinates.
        image_size: Image dimension (default 512 for the pipeline's tiles).
        grid_size: Number of grid divisions per axis (default 8 → 64 cells).

    Returns:
        Dict of coverage statistics.
    """
    total_cells = grid_size * grid_size
    cell_size = image_size / grid_size

    if len(points) == 0:
        return {
            "coverage_ratio": 0.0,
            "occupied_cells": 0,
            "total_cells": total_cells,
            "points_per_cell": [0] * total_cells,
            "std_per_cell": 0.0,
        }

    pts = np.asarray(points, dtype=np.float64)
    # Clamp to image bounds
    pts = np.clip(pts, 0, image_size - 1e-6)

    # Compute cell indices
    col_idx = np.floor(pts[:, 0] / cell_size).astype(int)
    row_idx = np.floor(pts[:, 1] / cell_size).astype(int)
    col_idx = np.clip(col_idx, 0, grid_size - 1)
    row_idx = np.clip(row_idx, 0, grid_size - 1)
    cell_ids = row_idx * grid_size + col_idx

    # Count points per cell
    counts = np.zeros(total_cells, dtype=int)
    for cid in cell_ids:
        counts[cid] += 1

    occupied = int(np.sum(counts > 0))

    return {
        "coverage_ratio": round(occupied / total_cells, 4),
        "occupied_cells": occupied,
        "total_cells": total_cells,
        "points_per_cell": counts.tolist(),
        "std_per_cell": round(float(np.std(counts)), 4),
    }


def match_distribution_score(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    image_size: int = 512,
    grid_size: int = 8,
) -> dict:
    """
    Combined distribution score across both source and destination images.

    Returns coverage metrics for both images, plus a combined score
    (geometric mean of the two coverage ratios).
    """
    src_cov = spatial_coverage(src_points, image_size, grid_size)
    dst_cov = spatial_coverage(dst_points, image_size, grid_size)

    src_ratio = src_cov["coverage_ratio"]
    dst_ratio = dst_cov["coverage_ratio"]
    combined = round(float(np.sqrt(src_ratio * dst_ratio)), 4) if src_ratio > 0 and dst_ratio > 0 else 0.0

    return {
        "source_coverage": src_cov,
        "destination_coverage": dst_cov,
        "combined_coverage_score": combined,
    }


# ---------------------------------------------------------------------------
# Sub-pixel accuracy check
# ---------------------------------------------------------------------------

def sub_pixel_accuracy(errors: np.ndarray) -> dict:
    """
    Check whether the registration achieves sub-pixel accuracy.

    Sub-pixel accuracy means the mean reprojection error < 1.0 pixel.
    Also reports what fraction of individual points are sub-pixel.

    Returns:
        Dict with sub_pixel_accurate flag, mean error, and fraction below 1 px.
    """
    if len(errors) == 0:
        return {
            "sub_pixel_accurate": False,
            "mean_reprojection_error_px": float("inf"),
            "fraction_below_1px": 0.0,
        }

    mean_err = float(np.mean(errors))
    frac_below_1 = float(np.mean(errors < 1.0))

    return {
        "sub_pixel_accurate": mean_err < 1.0,
        "mean_reprojection_error_px": round(mean_err, 4),
        "fraction_below_1px": round(frac_below_1, 4),
    }


# ---------------------------------------------------------------------------
# Unified metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    num_raw_matches: int | None = None,
    image_size: int = 512,
    grid_size: int = 8,
) -> dict:
    """
    Compute all evaluation metrics for a set of matched point pairs.

    This is the primary entry point. Given RANSAC-filtered source and
    destination match points, it computes and returns every metric
    required by the problem statement.

    Args:
        src_pts: (N, 2) post-RANSAC source pixel coordinates.
        dst_pts: (N, 2) post-RANSAC destination pixel coordinates.
        num_raw_matches: Total raw matches before RANSAC (for inlier ratio).
            If None, inlier ratio is set to 1.0 (all matches treated as inliers).
        image_size: Image dimension (512 for the pipeline's standard tiles).
        grid_size: Grid divisions for spatial coverage analysis.

    Returns:
        Dict containing all metrics:
        {
            "num_inliers": int,
            "num_raw_matches": int,
            "inlier_ratio": float,
            "rmse_px": float,
            "mean_reprojection_error_px": float,
            "median_reprojection_error_px": float,
            "max_reprojection_error_px": float,
            "sub_pixel_accurate": bool,
            "fraction_below_1px": float,
            "source_coverage_ratio": float,
            "destination_coverage_ratio": float,
            "combined_coverage_score": float,
            "source_occupied_cells": int,
            "destination_occupied_cells": int,
            "total_cells": int,
            "per_point_errors": list[float],
        }
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = len(src)

    if num_raw_matches is None:
        num_raw_matches = n

    result: dict = {
        "num_inliers": n,
        "num_raw_matches": num_raw_matches,
        "inlier_ratio": inlier_ratio(n, num_raw_matches),
    }

    # Compute homography and reprojection errors
    if n >= 4:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            errors = reprojection_errors(src, dst, H)
        else:
            errors = np.zeros(n, dtype=np.float64)
    else:
        errors = np.zeros(n, dtype=np.float64)

    result["rmse_px"] = round(rmse(errors), 4)
    result["mean_reprojection_error_px"] = round(mean_error(errors), 4)
    result["median_reprojection_error_px"] = round(median_error(errors), 4)
    result["max_reprojection_error_px"] = round(max_error(errors), 4)

    # Sub-pixel accuracy
    spa = sub_pixel_accuracy(errors)
    result["sub_pixel_accurate"] = spa["sub_pixel_accurate"]
    result["fraction_below_1px"] = spa["fraction_below_1px"]

    # Spatial distribution
    dist = match_distribution_score(src, dst, image_size, grid_size)
    result["source_coverage_ratio"] = dist["source_coverage"]["coverage_ratio"]
    result["destination_coverage_ratio"] = dist["destination_coverage"]["coverage_ratio"]
    result["combined_coverage_score"] = dist["combined_coverage_score"]
    result["source_occupied_cells"] = dist["source_coverage"]["occupied_cells"]
    result["destination_occupied_cells"] = dist["destination_coverage"]["occupied_cells"]
    result["total_cells"] = dist["source_coverage"]["total_cells"]

    # Per-point errors for downstream visualization
    result["per_point_errors"] = [round(float(e), 4) for e in errors]

    return result
