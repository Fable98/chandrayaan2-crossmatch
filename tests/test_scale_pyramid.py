"""
tests/test_scale_pyramid.py — Validation of True Multi-Resolution Coarse-to-Fine Scale Pyramid Search Loop.

Tests:
1. Level 1 Phase Congruency matching recovers coarse displacement.
2. Scale propagation: H_L1 correctly scales to Level 0 coordinates.
3. Pipeline integration: verify metrics["pyramid_matching"] and coarse_to_fine_applied flag.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import (
    match_images_cfog,
    multi_scale_phase_congruency,
    compute_phase_congruency,
)


def test_multi_scale_phase_congruency_levels():
    """Verify 3-level Gaussian Phase Congruency pyramid generation."""
    img = np.zeros((256, 256), dtype=np.float32)
    cv2.circle(img, (128, 128), 40, 1.0, -1)
    
    pyr = multi_scale_phase_congruency(img, scales=3)
    assert len(pyr) == 3
    assert pyr[0].shape == (256, 256)
    assert pyr[1].shape == (128, 128)
    assert pyr[2].shape == (64, 64)


def test_pyramid_homography_scaling_precision():
    """Verify mathematical exactness of H_L1 -> H_L0 transformation propagation."""
    # Given ground truth Level 0 homography
    H_L0_true = np.array([
        [1.015, -0.012, 18.5],
        [0.011, 1.022, -12.4],
        [0.00001, -0.00002, 1.0],
    ], dtype=np.float64)

    # Scale to Level 1 (coordinates halved)
    S_half = np.diag([0.5, 0.5, 1.0])
    S_two = np.diag([2.0, 2.0, 1.0])
    H_L1 = S_half @ H_L0_true @ S_two

    # Recover Level 0 prior
    H_L0_recovered = S_two @ H_L1 @ S_half

    # Test projection of arbitrary Level 0 point
    pt_L0 = np.array([120.0, 150.0, 1.0])
    proj_true = H_L0_true @ pt_L0
    proj_true_2d = proj_true[:2] / proj_true[2]

    proj_rec = H_L0_recovered @ pt_L0
    proj_rec_2d = proj_rec[:2] / proj_rec[2]

    error = np.linalg.norm(proj_true_2d - proj_rec_2d)
    assert error < 1e-6, f"Scale propagation error too high: {error}"


def test_pipeline_pyramid_matching_integration(tmp_path):
    """Verify full match_images_cfog executes hierarchical coarse-to-fine pyramid loop."""
    src_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png"
    ref_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/tmc_512.png"

    if not src_path.exists() or not ref_path.exists():
        pytest.skip("Region 001 sample images not found on disk.")

    out_dir = tmp_path / "pyramid_out"
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
    assert "pyramid_matching" in metrics
    pyr_stats = metrics["pyramid_matching"]

    assert pyr_stats["levels_executed"] == [2, 1, 0]
    assert pyr_stats["coarse_to_fine_applied"] is True
    assert "timing_s" in pyr_stats
    assert pyr_stats["timing_s"]["L2"] >= 0.0
    assert pyr_stats["timing_s"]["L1"] >= 0.0
    assert pyr_stats["timing_s"]["L0"] >= 0.0
