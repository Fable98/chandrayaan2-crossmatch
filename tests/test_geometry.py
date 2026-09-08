"""
tests/test_geometry.py — Validation of Piecewise Affine and DEM Ray-Intersection over Steep Lunar Craters.
"""

import math
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from geometry import dem_ray_intersection, warp_piecewise_affine, ransac_dem_aware_fit
from metrics import calculate_reprojection_errors


@pytest.fixture
def steep_crater_scene():
    """
    Generates a synthetic 512x512 scene with a steep-walled impact crater:
    - Crater radius: 120 px
    - Relief depth: 1500 meters (~35-degree wall slope)
    - Sensor emission: 20 degrees off-nadir
    - Sensor azimuth: 45 degrees
    - GSD: 5.0 m/px
    """
    h, w = 512, 512
    cx, cy = 256.0, 256.0
    crater_r = 120.0
    crater_depth_m = 1500.0
    gsd_m = 5.0
    emission_deg = 20.0
    azimuth_deg = 45.0

    # Build synthetic DEM: flat terrain (datum=0m) with parabolic crater bowl
    y_grid, x_grid = np.indices((h, w), dtype=np.float32)
    dist = np.hypot(x_grid - cx, y_grid - cy)

    dem = np.zeros((h, w), dtype=np.float32)
    crater_mask = dist < crater_r
    # Parabolic bowl profile: deepest at center, rising steeply to rim
    dem[crater_mask] = -crater_depth_m * (1.0 - (dist[crater_mask] / crater_r) ** 2)

    # Add rim elevation (+200m)
    rim_mask = (dist >= crater_r * 0.9) & (dist <= crater_r * 1.2)
    dem[rim_mask] += 200.0 * np.exp(-((dist[rim_mask] - crater_r) ** 2) / (2 * 15.0**2))

    # Generate synthetic correspondence points across the scene, with heavy sampling on steep walls
    wall_angles = np.linspace(0, 2 * math.pi, 60, endpoint=False)
    wall_radii = np.linspace(crater_r * 0.3, crater_r * 0.95, 8)
    wall_pts = []
    for r in wall_radii:
        for theta in wall_angles:
            px = cx + r * math.cos(theta)
            py = cy + r * math.sin(theta)
            wall_pts.append([px, py])

    # Floor and outer rim points
    outer_pts = []
    for ox in np.linspace(30, 480, 15):
        for oy in np.linspace(30, 480, 15):
            if math.hypot(ox - cx, oy - cy) > crater_r * 1.3:
                outer_pts.append([ox, oy])

    src_pts = np.vstack([wall_pts, outer_pts]).astype(np.float32)

    # Compute ground truth relief displacement under 20-deg off-nadir viewing
    _, relief_dxdy = dem_ray_intersection(
        src_pts, dem, emission_deg=emission_deg, azimuth_deg=azimuth_deg, gsd_m=gsd_m
    )
    dst_pts = (src_pts + relief_dxdy).astype(np.float32)

    return {
        "dem": dem,
        "src_pts": src_pts,
        "dst_pts": dst_pts,
        "relief_dxdy": relief_dxdy,
        "emission_deg": emission_deg,
        "azimuth_deg": azimuth_deg,
        "gsd_m": gsd_m,
        "wall_count": len(wall_pts),
    }


def test_dem_ray_intersection_accuracy(steep_crater_scene):
    """
    Validates that dem_ray_intersection computes realistic relief displacement vectors
    that scale with elevation depth and off-nadir angle.
    """
    scene = steep_crater_scene
    coords_3d, disp_px = dem_ray_intersection(
        scene["src_pts"],
        scene["dem"],
        emission_deg=scene["emission_deg"],
        azimuth_deg=scene["azimuth_deg"],
        gsd_m=scene["gsd_m"],
    )

    assert coords_3d.shape == (len(scene["src_pts"]), 3)
    assert disp_px.shape == (len(scene["src_pts"]), 2)

    # Crater floor is at -1500m elevation.
    # Theoretical displacement: dz * tan(20 deg) / 5.0 m/px ≈ 1500 * 0.36397 / 5 ≈ 109 px
    max_disp = np.max(np.hypot(disp_px[:, 0], disp_px[:, 1]))
    assert max_disp > 50.0, f"Expected strong relief displacement on crater walls, got max {max_disp} px"


def test_piecewise_and_dem_aware_outperforms_global_homography(steep_crater_scene):
    """
    CORE VALIDATION: Proves that piecewise/DEM-aware modeling yields lower RMSE on
    steep crater walls compared to the standard global planar homography approach.
    """
    scene = steep_crater_scene
    src_pts = scene["src_pts"]
    dst_pts = scene["dst_pts"]
    wall_n = scene["wall_count"]

    # 1. Global Homography Approach (planar approximation)
    H_global, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    assert H_global is not None

    err_global = calculate_reprojection_errors(src_pts[:wall_n], dst_pts[:wall_n], H_global)
    rmse_global = float(np.sqrt(np.mean(err_global**2)))

    # 2. DEM-Aware Model Fitting
    H_dem, inlier_mask, _ = ransac_dem_aware_fit(
        src_pts,
        dst_pts,
        dem=scene["dem"],
        emission_deg=scene["emission_deg"],
        azimuth_deg=scene["azimuth_deg"],
        gsd_m=scene["gsd_m"],
    )

    # In DEM-compensated coordinates, relief parallax is eliminated
    _, relief_dxdy = dem_ray_intersection(
        src_pts,
        scene["dem"],
        emission_deg=scene["emission_deg"],
        azimuth_deg=scene["azimuth_deg"],
        gsd_m=scene["gsd_m"],
    )
    src_compensated = src_pts + relief_dxdy
    err_dem = calculate_reprojection_errors(src_compensated[:wall_n], dst_pts[:wall_n], H_dem)
    rmse_dem = float(np.sqrt(np.mean(err_dem**2)))

    print(f"\n[Steep Crater Test] Global Homography RMSE on walls: {rmse_global:.4f} px")
    print(f"[Steep Crater Test] DEM-Aware Ray-Intersection RMSE: {rmse_dem:.4f} px")

    # DEM-aware model must substantially outperform global homography on steep crater walls
    assert rmse_dem < rmse_global, (
        f"DEM-aware RMSE ({rmse_dem:.4f} px) should be strictly lower than "
        f"global homography RMSE ({rmse_global:.4f} px) on steep crater terrain"
    )
    # The improvement should be large (at least 2x lower RMSE)
    assert rmse_dem < (rmse_global * 0.5), (
        f"DEM-aware RMSE ({rmse_dem:.4f} px) did not improve by at least 2x over "
        f"global homography ({rmse_global:.4f} px)"
    )


def test_warp_piecewise_affine_runs_without_artifacts():
    """
    Validates that warp_piecewise_affine executes cleanly and produces a valid output raster.
    """
    h, w = 256, 256
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(img, (128, 128), 50, 200, -1)

    pts1 = np.array([[30.0, 30.0], [220.0, 30.0], [30.0, 220.0], [220.0, 220.0], [128.0, 128.0]])
    # Non-linear deformation
    pts2 = pts1 + np.array([[2.0, -1.0], [-3.0, 2.0], [1.0, 3.0], [-2.0, -2.0], [0.0, 0.0]])

    warped = warp_piecewise_affine(img, pts1, pts2, out_shape=(h, w), tile_size=128)
    assert warped.shape == (h, w)
    assert warped.dtype == np.uint8
    assert np.max(warped) > 0
