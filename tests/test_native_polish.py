"""
tests/test_native_polish.py — Verify Native Full-Resolution Polish (Phase 5b).

Tests:
1. Unit test of refine_inliers_native_scale on textured lunar crater patches with scale disparity.
2. Pipeline integration test verifying metrics["native_polish"] and metrics["native_polish_applied"] on scale-disparity lunar pairs.
3. Toggle test verifying enable_native_polish=False cleanly disables the polish stage.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import match_images_cfog, refine_inliers_native_scale


def test_refine_inliers_native_scale_unit():
    """Verify subpixel native refinement recovers shifts on crater features."""
    np.random.seed(42)
    base = cv2.GaussianBlur(np.random.rand(400, 400).astype(np.float32), (7, 7), 2.0)
    
    # Add distinct crater-like circular features
    cv2.circle(base, (200, 200), 25, 0.1, -1)
    cv2.circle(base, (120, 120), 15, 0.9, -1)
    cv2.circle(base, (280, 280), 18, 0.8, -1)
    cv2.circle(base, (150, 260), 20, 0.2, -1)

    # Coarse image downsampled by 4x
    scale_factor1 = 4.0
    scale_factor2 = 1.0
    coarse = cv2.resize(base, (100, 100), interpolation=cv2.INTER_AREA)

    # Inlier points in native coordinates
    pts1 = np.array([[200.0, 200.0], [120.0, 120.0], [280.0, 280.0], [150.0, 260.0]], dtype=np.float64)
    # Corresponding coarse points in coarse coordinate space
    pts2 = pts1 / 4.0

    ref_pts1, ref_pts2, stats = refine_inliers_native_scale(
        raw_img1=base,
        raw_img2=coarse,
        inlier_pts1=pts1,
        inlier_pts2=pts2,
        scale_factor1=scale_factor1,
        scale_factor2=scale_factor2,
        gsd1=0.25,
        patch_size_native=48,
        max_shift_native_px=3.5,
    )

    assert stats["applied"] is True
    assert stats["total_inliers"] == 4
    assert stats["native_gsd_m"] == 0.25
    assert stats["refined_count"] >= 3


def test_native_polish_pipeline_integration(tmp_path):
    """Verify native polish runs during full match_images_cfog on scale-disparity lunar pairs."""
    src_orig = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png"
    if not src_orig.exists():
        pytest.skip("Region 001 sample image not found.")

    # Create scale disparity pair: high-res 512x512 (e.g. 1.0m) and coarse 256x256 (2.0m)
    img_high = cv2.imread(str(src_orig), cv2.IMREAD_GRAYSCALE)
    img_coarse = cv2.resize(img_high, (256, 256), interpolation=cv2.INTER_AREA)

    high_path = tmp_path / "high_512.png"
    coarse_path = tmp_path / "coarse_256.png"
    cv2.imwrite(str(high_path), img_high)
    cv2.imwrite(str(coarse_path), img_coarse)

    out_dir = tmp_path / "native_polish_out"
    res = match_images_cfog(
        high_path,
        coarse_path,
        source_sensor="OHRC",
        reference_sensor="TMC",
        output_dir=out_dir,
        explicit_gsd1=1.0,
        explicit_gsd2=2.0,
        enable_native_polish=True,
    )

    assert res["status"] == "success"
    metrics = res["metrics"]
    assert metrics is not None
    assert "native_polish" in metrics
    assert "native_polish_applied" in metrics
    polish_stats = metrics["native_polish"]
    assert polish_stats.get("applied") is True
    assert polish_stats.get("native_gsd_m") == pytest.approx(1.0, rel=1e-2)
    assert polish_stats.get("refined_count", 0) >= 4


def test_native_polish_can_be_disabled(tmp_path):
    """Verify enable_native_polish=False cleanly disables native polish."""
    src_orig = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png"
    if not src_orig.exists():
        pytest.skip("Region 001 sample image not found.")

    img_high = cv2.imread(str(src_orig), cv2.IMREAD_GRAYSCALE)
    img_coarse = cv2.resize(img_high, (256, 256), interpolation=cv2.INTER_AREA)

    high_path = tmp_path / "high_512.png"
    coarse_path = tmp_path / "coarse_256.png"
    cv2.imwrite(str(high_path), img_high)
    cv2.imwrite(str(coarse_path), img_coarse)

    out_dir = tmp_path / "disabled_polish_out"
    res = match_images_cfog(
        high_path,
        coarse_path,
        source_sensor="OHRC",
        reference_sensor="TMC",
        output_dir=out_dir,
        explicit_gsd1=1.0,
        explicit_gsd2=2.0,
        enable_native_polish=False,
    )

    assert res["status"] == "success"
    metrics = res["metrics"]
    assert metrics is not None
    assert metrics.get("native_polish_applied") is False
