"""
tests/test_subpixel_uncertainty.py — Test Suite for Sub-Pixel Uncertainty Estimation.

Verifies:
1. Local correlation peak Hessian / curvature estimation and conversion to a 2x2 covariance matrix.
2. Accurate 1-sigma major/minor axes and orientation angle extraction.
3. Proper handling, rejection, or downgrading of:
   - Sharp peaks (well-conditioned, sub-pixel accurate, status="valid")
   - Flat peaks (low curvature / ridge / aperture problem, status="flat_peak")
   - Multimodal peaks (competing secondary peaks, status="multimodal")
4. Mandatory preservation of uncertainty fields (covariance_xy, sigma_major_px, sigma_minor_px,
   ellipse_angle_deg, uncertainty_status) across all match records.
5. Uncertainty propagation into weighted geometric fitting.
"""

from pathlib import Path
import sys
import math
import numpy as np
import pytest
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT))

from subpixel_uncertainty import (
    SubpixelUncertaintyResult,
    estimate_peak_hessian_and_covariance,
    subpixel_phase_correlation_with_uncertainty,
    covariance_to_ellipse,
    build_fallback_uncertainty,
)
from matcher_cfog import (
    subpixel_phase_correlation,
    make_match_record,
    estimate_weighted_homography,
    match_images_cfog,
)


# ===========================================================================
# 1. Synthetic Peak Generators
# ===========================================================================

def create_synthetic_sharp_peak(shape=(32, 32), center=(16, 16), sigma=1.5, peak_val=0.85):
    """Creates an isotropic, sharp single Gaussian peak."""
    h, w = shape
    cy, cx = center
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    corr = peak_val * np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float32)
    return corr


def create_synthetic_flat_peak(shape=(32, 32), center=(16, 16), sigma=12.0, peak_val=0.40):
    """Creates a very broad, low-curvature (flat) Gaussian peak."""
    h, w = shape
    cy, cx = center
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    corr = peak_val * np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float32)
    return corr


def create_synthetic_multimodal_peak(shape=(32, 32), p1=(16, 16), p2=(16, 24), val1=0.85, val2=0.80):
    """Creates a correlation surface with two competing prominent peaks."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    d1 = (x - p1[1]) ** 2 + (y - p1[0]) ** 2
    d2 = (x - p2[1]) ** 2 + (y - p2[0]) ** 2
    corr = val1 * np.exp(-d1 / (2.0 * 1.5 ** 2)) + val2 * np.exp(-d2 / (2.0 * 1.5 ** 2))
    return np.clip(corr, 0.0, 1.0).astype(np.float32)


def create_synthetic_ridge_peak(shape=(32, 32), center=(16, 16), angle_deg=45.0, sigma_along=8.0, sigma_across=1.2):
    """Creates an anisotropic ridge peak oriented at angle_deg."""
    h, w = shape
    cy, cx = center
    y, x = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)
    theta = math.radians(angle_deg)
    # Coordinate rotation
    x_rot = x * math.cos(theta) + y * math.sin(theta)
    y_rot = -x * math.sin(theta) + y * math.cos(theta)
    corr = 0.85 * np.exp(-0.5 * ((x_rot / sigma_along) ** 2 + (y_rot / sigma_across) ** 2))
    return corr.astype(np.float32)


# ===========================================================================
# 2. Unit Tests
# ===========================================================================

def test_sharp_correlation_peak_covariance():
    """Requirement 1, 2, 7: Sharp peak yields positive-definite, sub-pixel covariance."""
    corr = create_synthetic_sharp_peak(shape=(32, 32), center=(16, 16), sigma=1.5, peak_val=0.90)
    res = estimate_peak_hessian_and_covariance(corr, peak_y=16, peak_x=16)

    assert res.is_valid is True
    assert res.uncertainty_status == "valid"
    assert res.sigma_major_px < 0.5, f"Expected sharp peak sigma_major < 0.5px, got {res.sigma_major_px}"
    assert res.sigma_minor_px < 0.5, f"Expected sharp peak sigma_minor < 0.5px, got {res.sigma_minor_px}"
    assert res.secondary_peak_ratio < 0.2

    cov = np.array(res.covariance_xy)
    assert cov.shape == (2, 2)
    assert cov[0, 0] > 0.0
    assert cov[1, 1] > 0.0
    # Covariance determinant must be strictly positive (positive definite)
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    assert det > 0.0


def test_flat_correlation_peak_rejection_or_downgrade():
    """Requirement 3, 7: Low-curvature / flat peak is rejected or downgraded with status='flat_peak'."""
    corr = create_synthetic_flat_peak(shape=(32, 32), center=(16, 16), sigma=15.0, peak_val=0.30)
    res = estimate_peak_hessian_and_covariance(corr, peak_y=16, peak_x=16)

    assert res.is_valid is False
    assert res.uncertainty_status == "flat_peak"
    # Flat peak must assign conservative large uncertainty
    assert res.sigma_major_px >= 1.0


def test_multimodal_correlation_peak_detection():
    """Requirement 3, 7: Dual competing peaks trigger multimodal status and are downgraded."""
    corr = create_synthetic_multimodal_peak(shape=(32, 32), p1=(16, 16), p2=(16, 25), val1=0.85, val2=0.82)
    res = estimate_peak_hessian_and_covariance(corr, peak_y=16, peak_x=16)

    assert res.is_valid is False
    assert res.uncertainty_status == "multimodal"
    assert res.secondary_peak_ratio > 0.80


def test_anisotropic_edge_ridge_peak_orientation():
    """Requirement 1, 2, 4: Anisotropic ridge peak reflects directional uncertainty."""
    corr = create_synthetic_ridge_peak(shape=(32, 32), center=(16, 16), angle_deg=45.0, sigma_along=8.0, sigma_across=1.2)
    res = estimate_peak_hessian_and_covariance(corr, peak_y=16, peak_x=16)

    # Uncertainty along the ridge must be significantly larger than across the ridge
    assert res.sigma_major_px > res.sigma_minor_px
    assert res.curvature_condition_number > 2.0


def test_no_subpixel_coordinate_without_uncertainty():
    """Requirement 8: Never report a sub-pixel coordinate without its confidence or uncertainty."""
    # Test 1: make_match_record without explicit uncertainty populates full fallback
    rec = make_match_record(10.5, 20.3, 11.2, 21.0, confidence=0.92)
    required_keys = [
        "confidence",
        "covariance_xy",
        "sigma_major_px",
        "sigma_minor_px",
        "ellipse_angle_deg",
        "uncertainty_status",
    ]
    for k in required_keys:
        assert k in rec, f"Missing mandatory uncertainty key: {k}"
        assert rec[k] is not None, f"Key {k} must not be None"

    assert isinstance(rec["covariance_xy"], list)
    assert len(rec["covariance_xy"]) == 2
    assert len(rec["covariance_xy"][0]) == 2
    assert isinstance(rec["sigma_major_px"], float)
    assert isinstance(rec["sigma_minor_px"], float)
    assert isinstance(rec["ellipse_angle_deg"], float)
    assert isinstance(rec["uncertainty_status"], str)


def test_subpixel_phase_correlation_returns_uncertainty():
    """Requirement 1, 4: subpixel_phase_correlation provides full uncertainty metadata when requested."""
    # Create two identical patches with known small sub-pixel translation
    rng = np.random.RandomState(42)
    base = rng.uniform(50, 200, (32, 32)).astype(np.float32)
    base = cv2.GaussianBlur(base, (5, 5), 1.0)
    M = np.float32([[1, 0, 0.4], [0, 1, -0.3]])
    shifted = cv2.warpAffine(base, M, (32, 32))

    # Standard call (4-tuple backward compatibility)
    dx, dy, peak, valid = subpixel_phase_correlation(base, shifted)
    assert isinstance(dx, float)
    assert isinstance(dy, float)
    assert isinstance(peak, float)
    assert isinstance(valid, bool)

    # Full uncertainty call
    dx_u, dy_u, peak_u, valid_u, unc = subpixel_phase_correlation(base, shifted, return_uncertainty=True)
    assert valid_u is True
    assert "covariance_xy" in unc
    assert "sigma_major_px" in unc
    assert "sigma_minor_px" in unc
    assert "ellipse_angle_deg" in unc
    assert "uncertainty_status" in unc
    assert unc["uncertainty_status"] == "valid"


def test_uncertainty_weighted_homography_propagation():
    """Requirement 5: Uncertainty propagation downweights high-uncertainty and multimodal matches."""
    # Generate 10 correspondences
    rng = np.random.RandomState(101)
    pts1 = rng.uniform(50, 450, (10, 2)).astype(np.float32)
    H_gt = np.array([[1.02, 0.01, 15.0],
                     [-0.01, 1.01, 10.0],
                     [0.00001, 0.00002, 1.0]], dtype=np.float64)

    pts2 = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H_gt).reshape(-1, 2)

    # First 6 matches have low uncertainty (sharp peaks, sigma=0.15)
    # Remaining 4 matches have high uncertainty (flat / multimodal peaks, sigma=3.0)
    matches_meta = []
    for i in range(10):
        if i < 6:
            rec = make_match_record(
                pts1[i, 0], pts1[i, 1], pts2[i, 0], pts2[i, 1], confidence=0.9,
                sigma_major_px=0.15, sigma_minor_px=0.15, uncertainty_status="valid"
            )
        else:
            # Perturb noisy/uncertain points
            pts2[i] += rng.normal(0, 4.0, 2)
            rec = make_match_record(
                pts1[i, 0], pts1[i, 1], pts2[i, 0], pts2[i, 1], confidence=0.8,
                sigma_major_px=3.5, sigma_minor_px=2.5, uncertainty_status="multimodal"
            )
        matches_meta.append(rec)

    # Compute weights incorporating uncertainty
    raw_weights = []
    for m in matches_meta:
        prior = float(m["confidence"])
        sig_maj = float(m["sigma_major_px"])
        sig_min = float(m["sigma_minor_px"])
        status = m["uncertainty_status"]
        prec = 1.0 / (sig_maj ** 2 + sig_min ** 2 + 1e-4)
        factor = 0.05 if status in ("flat_peak", "multimodal") else 1.0
        raw_weights.append(prior * prec * factor)

    weights = np.array(raw_weights, dtype=np.float64)
    weights /= np.max(weights)

    # The high-precision matches should have dramatically higher weights than multimodal points
    assert weights[0] > 10.0 * weights[8]

    H_est, mask, tag = estimate_weighted_homography(
        pts1, pts2, weights,
        estimator_method=cv2.RANSAC,
        ransac_reproj_threshold=5.0,
        image_shape=(512, 512),
        rng_seed=42,
    )
    assert H_est is not None
    assert mask is not None
    # Sharp inliers must be preserved
    assert np.all(mask[:6].ravel() == 1)


def test_end_to_end_matcher_stores_uncertainty_in_all_matches(tmp_path):
    """Requirement 4, 8: End-to-end match_images_cfog produces matches.json with uncertainty fields."""
    # Synthetic pair
    rng = np.random.RandomState(42)
    img1 = rng.uniform(80, 180, (256, 256)).astype(np.uint8)
    img1 = cv2.GaussianBlur(img1, (7, 7), 2.0)
    # Add high-contrast textured crater-like spots
    for cx, cy, r in [(64, 64, 20), (180, 180, 25), (70, 190, 18), (190, 70, 22)]:
        cv2.circle(img1, (cx, cy), r, (30,), -1)
        cv2.circle(img1, (cx, cy), r, (220,), 3)

    M = np.float32([[1, 0, 3.0], [0, 1, 2.0]])
    img2 = cv2.warpAffine(img1, M, (256, 256))

    out_dir = tmp_path / "test_out_unc"
    res = match_images_cfog(
        img_path1=img1,
        img_path2=img2,
        output_dir=out_dir,
        explicit_gsd1=1.0,
        explicit_gsd2=1.0,
    )

    all_matches = res.get("all_matches", [])
    assert len(all_matches) > 0

    for m in all_matches:
        assert "covariance_xy" in m
        assert "sigma_major_px" in m
        assert "sigma_minor_px" in m
        assert "ellipse_angle_deg" in m
        assert "uncertainty_status" in m
        assert m["covariance_xy"] is not None
        assert isinstance(m["sigma_major_px"], float)
        assert isinstance(m["sigma_minor_px"], float)
        assert isinstance(m["ellipse_angle_deg"], float)
        assert isinstance(m["uncertainty_status"], str)
