"""
tests/test_illumination_metrics.py — Unit tests for illumination-robust registration quality metrics:
SSIM, PSNR, Normalized Mutual Information (NMI), and Composite Quality Score.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from metrics import (
    calculate_overlap_mask,
    calculate_ssim_over_overlap,
    calculate_psnr_over_overlap,
    calculate_normalized_mutual_information,
    calculate_composite_quality_score,
    compute_canonical_metrics,
)


def _generate_synthetic_lunar_patch(shape=(128, 128), seed=42) -> np.ndarray:
    """Generates a synthetic lunar-like texture with craters and varied relief."""
    rng = np.random.RandomState(seed)
    base = rng.uniform(50.0, 180.0, size=shape).astype(np.float32)
    # Smooth background terrain
    base = cv2.GaussianBlur(base, (15, 15), 3.0)

    # Add synthetic circular craters
    for _ in range(5):
        cx = rng.randint(20, shape[1] - 20)
        cy = rng.randint(20, shape[0] - 20)
        radius = rng.randint(8, 25)
        y, x = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        crater_mask = dist <= radius
        # Crater floor is shadowed/darker, rim is brighter
        rim_mask = (dist > radius - 2) & (dist <= radius + 2)
        base[crater_mask] *= 0.65
        base[rim_mask] = np.clip(base[rim_mask] * 1.35, 0, 255)

    return np.clip(base, 10.0, 245.0)


def test_identity_pair_ssim_psnr_nmi():
    """
    Validates that an identical image pair yields SSIM ≈ 1.0, PSNR = inf/high, and NMI ≈ 1.0.
    """
    img = _generate_synthetic_lunar_patch()

    mask = calculate_overlap_mask(img, img)
    assert np.all(mask), "Overlap mask for identical non-zero images should be 100% valid"

    psnr = calculate_psnr_over_overlap(img, img, mask=mask)
    assert psnr == float("inf") or psnr >= 99.0, f"Expected infinite or high PSNR for identity, got {psnr}"

    ssim = calculate_ssim_over_overlap(img, img, mask=mask)
    assert ssim is not None and ssim >= 0.99, f"Expected SSIM ≈ 1.0 for identity, got {ssim}"

    nmi = calculate_normalized_mutual_information(img, img, mask=mask)
    assert nmi is not None and nmi >= 0.99, f"Expected NMI ≈ 1.0 for identity, got {nmi}"


def test_illumination_robustness_nmi_under_contrast_and_bias():
    """
    Validates that Normalized Mutual Information (NMI) remains high under
    drastic illumination/contrast shifts (simulating differing solar angles),
    while linear MSE-dependent metrics (PSNR) degrade.
    """
    img1 = _generate_synthetic_lunar_patch()
    # Apply non-linear photometric perturbation (power-law gamma curve + bias)
    img2 = np.clip(255.0 * ((img1 / 255.0) ** 1.8) + 15.0, 0, 255)

    mask = calculate_overlap_mask(img1, img2)
    nmi = calculate_normalized_mutual_information(img1, img2, mask=mask)
    psnr = calculate_psnr_over_overlap(img1, img2, mask=mask)

    # NMI preserves shared information under monotonic brightness transforms
    assert nmi is not None and nmi >= 0.75, f"Expected high NMI under monotonic illumination change, got {nmi}"
    # PSNR should be distinctly degraded due to intensity shift
    assert psnr is not None and psnr < 35.0, f"Expected lower PSNR due to photometric offset, got {psnr}"


def test_composite_quality_score_derivation_and_weighting():
    """
    Validates calculate_composite_quality_score formula, weighting, and explicit
    provenance tagging (derived/synthetic, not measured).
    """
    # 1. With image alignment metrics
    comp_full = calculate_composite_quality_score(
        inlier_ratio=0.8,
        fit_rmse_px=0.5,
        spatial_uniformity=0.7,
        nmi=0.9,
        ssim=0.85,
    )
    assert comp_full["composite_quality_score_is_derived"] is True
    assert "composite_quality_score" in comp_full
    score = comp_full["composite_quality_score"]
    assert 0.0 <= score <= 1.0

    components = comp_full["composite_quality_score_components"]
    assert components["inlier_ratio_term"] == 0.8
    assert components["alignment_term"] is not None

    # 2. Without image alignment metrics (feature-only 3-way split)
    comp_geom = calculate_composite_quality_score(
        inlier_ratio=0.6,
        fit_rmse_px=1.0,
        spatial_uniformity=0.5,
        nmi=None,
        ssim=None,
    )
    assert comp_geom["composite_quality_score_is_derived"] is True
    assert comp_geom["composite_quality_score_components"]["alignment_term"] is None
    expected_geom = (0.6 + np.exp(-1.0 / 2.0) + 0.5) / 3.0
    assert abs(comp_geom["composite_quality_score"] - round(expected_geom, 4)) < 1e-3


def test_compute_canonical_metrics_with_images_and_identity_warp():
    """
    Validates that compute_canonical_metrics correctly warps and computes
    SSIM, PSNR, NMI, and composite quality score when image arrays are provided.
    """
    img = _generate_synthetic_lunar_patch((128, 128))
    pts = np.array([[20.0, 20.0], [100.0, 20.0], [20.0, 100.0], [100.0, 100.0]], dtype=np.float32)
    mask = np.ones((4, 1), dtype=np.uint8)
    H_eye = np.eye(3, dtype=np.float32)

    metrics = compute_canonical_metrics(
        src_pts_raw=pts,
        dst_pts_raw=pts,
        inlier_mask=mask,
        H=H_eye,
        image_shape=(128, 128),
        grid_size=4,
        source_img=img,
        ref_img=img,
    )

    assert "ssim" in metrics and metrics["ssim"] is not None
    assert metrics["ssim"] >= 0.99
    assert "psnr" in metrics and metrics["psnr"] is not None
    assert "nmi" in metrics and metrics["nmi"] is not None
    assert metrics["nmi"] >= 0.99
    assert "composite_quality_score" in metrics
    assert metrics["composite_quality_score_is_derived"] is True
    assert metrics["composite_quality_score"] >= 0.75

    # Also verify that with fully distributed inliers across cells, composite score reaches > 0.95
    grid_pts = []
    for gy in range(4):
        for gx in range(4):
            grid_pts.append([gx * 32.0 + 16.0, gy * 32.0 + 16.0])
    pts_dense = np.array(grid_pts, dtype=np.float32)
    mask_dense = np.ones((len(pts_dense), 1), dtype=np.uint8)

    metrics_dense = compute_canonical_metrics(
        src_pts_raw=pts_dense,
        dst_pts_raw=pts_dense,
        inlier_mask=mask_dense,
        H=H_eye,
        image_shape=(128, 128),
        grid_size=4,
        source_img=img,
        ref_img=img,
    )
    assert metrics_dense["composite_quality_score"] >= 0.95


def test_compute_canonical_metrics_zero_inliers_fallback():
    """
    Validates that compute_canonical_metrics reports null/safe metrics when inliers=0.
    """
    pts = np.zeros((0, 2), dtype=np.float32)
    metrics = compute_canonical_metrics(
        src_pts_raw=pts,
        dst_pts_raw=pts,
        inlier_mask=None,
        H=None,
    )

    assert metrics["inlier_count"] == 0
    assert metrics["ssim"] is None
    assert metrics["psnr"] is None
    assert metrics["nmi"] is None
    assert metrics["composite_quality_score"] == 0.0
    assert metrics["composite_quality_score_is_derived"] is True
