"""
Tests for TMC-2 Triplet Stereo Photogrammetry Engine.

Validates:
    - Along-track Base-to-Height (B/H) calculations
    - Closed-form disparity-elevation inverse mappings
    - Synthetic Fore/Aft along-track parallax generation
    - Dense SGBM disparity estimation
    - End-to-end DEM derivation on simulated and flight lunar scenes
"""

import math
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "ML_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ML_model.tmc_stereo import (
    compute_tmc_base_to_height_ratio,
    disparity_to_elevation,
    elevation_to_disparity,
    generate_synthetic_stereo_views,
    compute_tmc_stereo_disparity,
    derive_dem_from_tmc_stereo,
)


def test_base_to_height_ratio():
    """Verify B/H calculation for Chandrayaan-2 TMC-2 camera angles."""
    # Fore (+26 deg) and Aft (-26 deg)
    bh_fore_aft = compute_tmc_base_to_height_ratio(26.0, -26.0)
    expected_fore_aft = 2.0 * math.tan(math.radians(26.0))
    assert math.isclose(bh_fore_aft, expected_fore_aft, rel_tol=1e-5)
    assert math.isclose(bh_fore_aft, 0.975525, rel_tol=1e-4)

    # Fore (+26 deg) and Nadir (0 deg)
    bh_fore_nadir = compute_tmc_base_to_height_ratio(26.0, 0.0)
    assert math.isclose(bh_fore_nadir, math.tan(math.radians(26.0)), rel_tol=1e-5)
    assert math.isclose(bh_fore_nadir, 0.487733, rel_tol=1e-4)


def test_disparity_to_elevation_math():
    """Verify exact algebraic relationship between disparity and physical elevation."""
    gsd = 5.0
    b_over_h = 0.975525
    known_z = 1000.0  # 1000 meters elevation

    # d = Z * (B/H) / GSD
    expected_d = known_z * b_over_h / gsd
    calculated_d = elevation_to_disparity(np.array([known_z]), gsd_m=gsd, b_over_h=b_over_h)
    assert np.isclose(calculated_d[0], expected_d, rtol=1e-5)

    # Inverse: Z = d * GSD / (B/H)
    recovered_z = disparity_to_elevation(calculated_d, gsd_m=gsd, b_over_h=b_over_h)
    assert np.isclose(recovered_z[0], known_z, rtol=1e-5)


def test_synthetic_stereo_views_generation():
    """Verify backward mapping of Nadir image into Fore and Aft views."""
    h, w = 256, 256
    # Create textured lunar surface with a central circular crater depression
    np.random.seed(42)
    nadir = np.random.randint(80, 180, (h, w), dtype=np.uint8)
    nadir = cv2.GaussianBlur(nadir, (7, 7), 2.0)

    # Synthetic DEM: 400m elevation mound in the center
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - w // 2) ** 2 + (yy - h // 2) ** 2)
    dem = np.maximum(0.0, 400.0 * (1.0 - r / 60.0)).astype(np.float32)

    fore, aft = generate_synthetic_stereo_views(nadir, dem, gsd_m=5.0, fore_angle_deg=26.0, aft_angle_deg=-26.0)

    assert fore.shape == (h, w)
    assert aft.shape == (h, w)
    # The parallax shift must produce measurable differences between fore and aft views
    diff = np.abs(fore.astype(np.float32) - aft.astype(np.float32))
    assert np.mean(diff) > 0.5


def test_derive_dem_from_tmc_stereo_pipeline():
    """Verify end-to-end stereo DEM reconstruction produces valid physical ranges."""
    h, w = 256, 256
    # Generate textured terrain with distinct crater rims
    grid_y, grid_x = np.ogrid[:h, :w]
    pattern = ((np.sin(grid_x / 10.0) + np.cos(grid_y / 10.0)) * 60.0 + 128.0).astype(np.uint8)
    noise = np.random.normal(0, 5, (h, w)).astype(np.int16)
    textured_nadir = np.clip(pattern.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 300m elevation gradient across the scene
    true_dem = (grid_x.astype(np.float32) / w) * 300.0

    fore, aft = generate_synthetic_stereo_views(textured_nadir, true_dem, gsd_m=5.0)

    res = derive_dem_from_tmc_stereo(fore, aft, img_nadir=textured_nadir, gsd_m=5.0, num_disparities=48)

    assert "dem_meters" in res
    assert "dem_u8" in res
    assert "disparity_map" in res
    assert "valid_mask" in res
    assert "metrics" in res

    assert res["dem_meters"].shape == (h, w)
    assert res["dem_u8"].dtype == np.uint8
    assert res["dem_u8"].shape == (h, w)
    assert math.isclose(res["b_over_h"], 0.975525, rel_tol=1e-4)

    metrics = res["metrics"]
    assert metrics["relief_range_m"] >= 0.0
    assert 0.0 <= metrics["valid_match_ratio"] <= 1.0


def test_flight_scene_stereo_integration():
    """Verify stereo processing on an actual flight triplet crop if present."""
    sample_path = Path("data_preprocessing_pipeline/processed_triplets/region_001/tmc_512.png")
    if not sample_path.exists():
        pytest.skip("Flight triplet sample region_001 not present.")

    tmc_nadir = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    assert tmc_nadir is not None

    # Synthesize along-track parallax from terrain texture gradient
    grad_x = cv2.Sobel(tmc_nadir, cv2.CV_32F, 1, 0, ksize=3)
    grad_norm = grad_x / (np.max(np.abs(grad_x)) + 1e-6)
    relief_init = -cv2.GaussianBlur(grad_norm, (9, 9), 2.0) * 150.0

    fore, aft = generate_synthetic_stereo_views(tmc_nadir, relief_init, gsd_m=5.0)
    stereo_res = derive_dem_from_tmc_stereo(fore, aft, img_nadir=tmc_nadir, gsd_m=5.0)

    assert stereo_res["dem_u8"].shape == tmc_nadir.shape
    assert stereo_res["metrics"]["valid_match_ratio"] > 0.3
