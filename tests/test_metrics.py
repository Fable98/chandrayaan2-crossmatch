"""
tests/test_metrics.py — Unit tests for Absolute RMSE (meters) and registration metrics
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from metrics import calculate_absolute_rmse_meters, compute_canonical_metrics


def test_absolute_rmse_flat_dem_scalar():
    """
    Validates that with a flat DEM (zero elevation delta) and a known GSD (2.5 m/px),
    the absolute RMSE in meters equals exactly (pixel RMSE * GSD).
    """
    # 4 points with identical 2.0 px Euclidean error: (dx=1.2, dy=1.6) -> sqrt(1.2^2 + 1.6^2) = 2.0
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    pts2 = np.array([[11.2, 11.6], [21.2, 21.6], [31.2, 31.6], [41.2, 41.6]])
    gsd = 2.5  # meters per pixel

    # Flat DEM (constant 500m elevation)
    flat_dem = np.full((100, 100), 500.0, dtype=np.float32)

    rmse_meters = calculate_absolute_rmse_meters((pts1, pts2), gsd=gsd, dem_data=flat_dem)
    expected_rmse = 2.0 * 2.5  # 5.0 meters

    assert abs(rmse_meters - expected_rmse) < 1e-3, f"Expected {expected_rmse}, got {rmse_meters}"


def test_absolute_rmse_match_dicts_and_flat_dem():
    """
    Validates calculate_absolute_rmse_meters with a list of correspondence dicts.
    """
    # Point 1: dx=3, dy=4 -> dr=5 px -> 5 * 2.5 = 12.5 m
    # Point 2: dx=0, dy=0 -> dr=0 px -> 0 m
    # RMS = sqrt((12.5^2 + 0^2)/2) = sqrt(156.25 / 2) = sqrt(78.125) ≈ 8.8388 m
    match_records = [
        {"source_x": 10.0, "source_y": 20.0, "target_x": 13.0, "target_y": 24.0},
        {"source_x": 50.0, "source_y": 50.0, "target_x": 50.0, "target_y": 50.0},
    ]
    gsd = 2.5
    flat_dem = np.zeros((100, 100), dtype=np.float32)

    rmse_meters = calculate_absolute_rmse_meters(match_records, gsd=gsd, dem_data=flat_dem)
    expected_rmse = float(np.sqrt((12.5**2) / 2.0))

    assert abs(rmse_meters - round(expected_rmse, 4)) < 1e-3, f"Expected {expected_rmse}, got {rmse_meters}"


def test_absolute_rmse_topographic_relief():
    """
    Validates that a non-flat DEM properly incorporates 3D elevation delta (dz).
    """
    # Point from (10, 10) to (13, 14): dx=3 px, dy=4 px -> planar = 5 px * 2.5 = 12.5 m
    # Elevation: dem[10, 10] = 100.0 m, dem[14, 13] = 105.0 m -> dz = 5.0 m
    # 3D distance = sqrt(12.5^2 + 5.0^2) = sqrt(156.25 + 25) = sqrt(181.25) ≈ 13.4629 m
    pts1 = np.array([[10.0, 10.0]])
    pts2 = np.array([[13.0, 14.0]])
    gsd = 2.5

    dem = np.full((100, 100), 100.0, dtype=np.float32)
    dem[14, 13] = 105.0  # target location has +5m elevation

    rmse_meters = calculate_absolute_rmse_meters((pts1, pts2), gsd=gsd, dem_data=dem)
    expected_3d = float(np.sqrt(12.5**2 + 5.0**2))

    assert abs(rmse_meters - round(expected_3d, 4)) < 1e-3, f"Expected {expected_3d}, got {rmse_meters}"


def test_canonical_metrics_includes_absolute_rmse():
    """
    Validates that compute_canonical_metrics returns 'absolute_rmse_m' when gsd_m is provided.
    """
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    pts2 = np.array([[12.0, 10.0], [22.0, 20.0], [32.0, 30.0], [42.0, 40.0]])  # pure dx=2.0 px
    H = np.eye(3, dtype=np.float64)  # Identity homography -> error is 2.0 px
    mask = np.ones(4, dtype=np.uint8)

    metrics = compute_canonical_metrics(
        pts1, pts2, mask, H, image_shape=(100, 100), gsd_m=2.5
    )

    assert "absolute_rmse_m" in metrics
    assert metrics["absolute_rmse_m"] is not None
    assert abs(metrics["absolute_rmse_m"] - 5.0) < 1e-3
