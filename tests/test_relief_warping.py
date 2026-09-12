"""
tests/test_relief_warping.py — Validation of Non-Planar Lunar Relief & Topographic Warping.

Tests:
1. TPS boundary accuracy: verify corners are pinned to projected homography instead of canvas corners.
2. DEM-aware RANSAC: verify retention of steep crater-wall inliers under off-nadir parallax.
3. Topographic relief strain estimation: detect non-planar parallax strain without DEM.
4. Pipeline integration: verify metrics["topographic_relief"] audit trail.
"""

import math
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from geometry import (
    warp_thin_plate_splines,
    ransac_dem_aware_fit,
    estimate_topographic_relief_strain,
    dem_ray_intersection,
)
from matcher_cfog import match_images_cfog


def test_tps_preserves_projected_boundary():
    """Verify TPS boundary corners follow global_H and avoid unnatural border shearing."""
    h, w = 300, 300
    out_h, out_w = 400, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 200

    # Transformation: translate by (+50, +50)
    H = np.array([
        [1.0, 0.0, 50.0],
        [0.0, 1.0, 50.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Control points inside
    src_pts = np.array([[50.0, 50.0], [200.0, 50.0], [50.0, 200.0], [200.0, 200.0]], dtype=np.float32)
    dst_pts = src_pts + 50.0  # matches the translation

    warped = warp_thin_plate_splines(img, src_pts, dst_pts, (out_h, out_w), global_H=H)

    assert warped.shape == (out_h, out_w, 3)
    # The pixel at (20, 20) in output should be outside the translated image (i.e. black 0)
    # If corners were pinned to (0,0), it would be stretched and non-zero!
    assert np.all(warped[10, 10] == 0), "Corner (10, 10) should be background black, not sheared to canvas edge."
    # The center of the translated image (200, 200) should be filled
    assert np.mean(warped[200, 200]) > 100


def test_dem_aware_ransac_preserves_crater_wall_inliers():
    """Verify DEM-aware RANSAC retains crater-wall correspondences under off-nadir viewing."""
    h, w = 512, 512
    cx, cy = 256.0, 256.0
    crater_r = 120.0
    crater_depth_m = 1500.0
    gsd_m = 5.0
    emission_deg = 20.0
    azimuth_deg = 45.0

    # Build synthetic DEM: flat terrain with parabolic crater
    y_grid, x_grid = np.indices((h, w), dtype=np.float32)
    dist = np.hypot(x_grid - cx, y_grid - cy)
    dem = np.zeros((h, w), dtype=np.float32)
    crater_mask = dist < crater_r
    dem[crater_mask] = -crater_depth_m * (1.0 - (dist[crater_mask] / crater_r) ** 2)

    # Sample points heavily on crater walls
    wall_radii = np.linspace(crater_r * 0.3, crater_r * 0.9, 6)
    wall_angles = np.linspace(0, 2 * math.pi, 20, endpoint=False)
    wall_pts = []
    for r in wall_radii:
        for theta in wall_angles:
            wall_pts.append([cx + r * math.cos(theta), cy + r * math.sin(theta)])
    src_pts = np.array(wall_pts, dtype=np.float32)

    # Ground truth relief displacement under 20-deg off-nadir viewing
    _, relief_dxdy = dem_ray_intersection(
        src_pts, dem, emission_deg=emission_deg, azimuth_deg=azimuth_deg, gsd_m=gsd_m
    )
    # Destination points are displaced by topography (relief parallax)
    dst_pts = src_pts + relief_dxdy

    # 1. Standard planar RANSAC finds 0 or very few inliers due to 35-degree wall parallax
    _, planar_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    planar_inliers = int(np.sum(planar_mask)) if planar_mask is not None else 0

    # 2. DEM-aware RANSAC corrects for relief and identifies crater-wall inliers
    H_dem, dem_mask, info = ransac_dem_aware_fit(
        src_pts, dst_pts, dem=dem, emission_deg=emission_deg, azimuth_deg=azimuth_deg, gsd_m=gsd_m, reproj_thresh_px=4.0
    )
    dem_inliers = int(np.sum(dem_mask)) if dem_mask is not None else 0

    assert dem_inliers >= len(src_pts) * 0.85, f"DEM-aware RANSAC should preserve >= 85% inliers, got {dem_inliers}/{len(src_pts)}"
    assert dem_inliers > planar_inliers, f"DEM-aware inliers ({dem_inliers}) must exceed planar inliers ({planar_inliers})"


def test_topographic_relief_strain_estimation():
    """Verify estimate_topographic_relief_strain distinguishes flat mare from crater relief."""
    np.random.seed(42)
    pts1 = np.random.uniform(50, 450, (24, 2))
    
    # Flat terrain: purely planar homography with 0.25px sensor noise
    H_true = np.array([[1.02, 0.01, 10.0], [-0.01, 1.01, -5.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_flat = (H_true @ pts1_h.T).T[:, :2] + np.random.normal(0, 0.25, pts1.shape)

    res_flat = estimate_topographic_relief_strain(pts1, pts2_flat, H_true)
    assert res_flat["strain_detected"] is False
    assert res_flat["strain_ratio"] < 1.35

    # Non-planar crater terrain: inject 3.0px differential parallax into center crater points
    pts2_relief = pts2_flat.copy()
    center_dist = np.hypot(pts1[:, 0] - 250, pts1[:, 1] - 250)
    crater_pts = center_dist < 130
    pts2_relief[crater_pts, 0] += 3.5  # Relief displacement along X

    H_est, _ = cv2.findHomography(pts1, pts2_relief, 0)
    res_relief = estimate_topographic_relief_strain(pts1, pts2_relief, H_est)
    assert res_relief["strain_detected"] is True
    assert res_relief["strain_ratio"] >= 1.35


def test_pipeline_topographic_relief_metrics(tmp_path):
    """Verify registration pipeline computes metrics['topographic_relief']."""
    src_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png"
    ref_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/tmc_512.png"

    if not src_path.exists() or not ref_path.exists():
        pytest.skip("Region 001 sample image not found.")

    out_dir = tmp_path / "relief_out"
    res = match_images_cfog(
        src_path,
        ref_path,
        source_sensor="OHRC",
        reference_sensor="TMC",
        output_dir=out_dir,
    )

    assert res["status"] == "success"
    metrics = res["metrics"]
    assert metrics is not None
    assert "topographic_relief" in metrics
    relief_info = metrics["topographic_relief"]
    assert "model" in relief_info
    assert "relief_strain_detected" in relief_info
    assert "relief_strain_ratio" in relief_info
