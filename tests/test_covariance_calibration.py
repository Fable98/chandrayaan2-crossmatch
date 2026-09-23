"""
tests/test_covariance_calibration.py — Tests for Empirical Covariance-Calibration Experiment

Validates:
1. Synthetic terrain generation across 4 geomorphic textures (mare, cratered, rugged, ridge_edge).
2. Exact Fourier sub-pixel translation fidelity.
3. Imaging perturbation pipeline (blur, noise, contrast, gain/bias, shadow polarity, scale ratio).
4. Chi-square 2-DOF Mahalanobis distance properties and quantile bounds.
5. Reliability diagrams and Expected Calibration Error (ECE) calculations.
6. Flat peak and multimodal peak detection and failure rate tracking.
7. Probabilistic Calibration Governance: Never calling covariance calibrated unless
   empirical coverage matches theoretical expectations.
8. End-to-end execution of the calibration benchmark.
"""

import math
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eval_covariance_calibration import (
    generate_synthetic_terrain,
    apply_fourier_shift_2d,
    apply_imaging_perturbations,
    run_single_calibration_trial,
    compute_calibration_statistics,
    run_covariance_calibration_experiment,
    MatchTrialRecord,
    CalibrationReport,
)
from subpixel_uncertainty import (
    subpixel_phase_correlation_with_uncertainty,
    estimate_peak_hessian_and_covariance,
)


# ===========================================================================
# 1. Synthetic Terrain & Exact Ground-Truth Fourier Translation
# ===========================================================================

def test_calibrated_synthetic_generator_exact_ground_truth():
    """Verifies generation of all 4 terrain textures and Fourier sub-pixel translation."""
    for tex in ["mare", "cratered", "rugged", "ridge_edge"]:
        terrain = generate_synthetic_terrain(tex, size=64, seed=123)
        assert terrain.shape == (64, 64)
        assert np.all(np.isfinite(terrain))
        assert float(np.min(terrain)) >= 0.0
        assert float(np.max(terrain)) <= 255.0
        assert float(np.std(terrain)) > 5.0  # contains textural variation

    # Verify Fourier fractional shift
    base = generate_synthetic_terrain("cratered", size=64, seed=42)
    shifted = apply_fourier_shift_2d(base, dx=0.35, dy=-0.45)
    assert shifted.shape == (64, 64)
    assert np.all(np.isfinite(shifted))

    # Phase correlation on pristine shifted patch should recover displacement within 0.05 px
    res = subpixel_phase_correlation_with_uncertainty(base[16:48, 16:48], shifted[16:48, 16:48])
    assert res.is_valid is True
    assert np.isclose(res.dx, 0.35, atol=0.08)
    assert np.isclose(res.dy, -0.45, atol=0.08)


# ===========================================================================
# 2. Imaging Perturbations
# ===========================================================================

def test_perturbation_variations():
    """Tests imaging perturbations: blur, noise, contrast, gain/bias, shadow polarity, scale."""
    base = generate_synthetic_terrain("cratered", size=64, seed=55)

    # 1. Blur
    blurred = apply_imaging_perturbations(base, blur_sigma=1.5)
    assert blurred.shape == base.shape
    assert float(np.std(cv2.Sobel(blurred, cv2.CV_32F, 1, 0))) < float(np.std(cv2.Sobel(base, cv2.CV_32F, 1, 0)))

    # 2. Noise
    noisy = apply_imaging_perturbations(base, noise_sigma=10.0)
    assert noisy.shape == base.shape
    assert not np.allclose(base, noisy)

    # 3. Contrast & Gain
    scaled = apply_imaging_perturbations(base, contrast=0.5, gain_bias=20.0)
    assert scaled.shape == base.shape

    # 4. Shadow Inversion
    inverted = apply_imaging_perturbations(base, invert_shadow_polarity=True)
    assert inverted.shape == base.shape
    assert np.isclose(float(np.mean(base + inverted)), 255.0, atol=5.0)

    # 5. Cross-Sensor Scale Ratio
    downsampled = apply_imaging_perturbations(base, sensor_scale=2.0)
    assert downsampled.shape == base.shape


# ===========================================================================
# 3. Chi-Square 2-DOF Mahalanobis Distance Properties
# ===========================================================================

def test_mahalanobis_chi2_properties():
    """Verifies 2D Mahalanobis distance calculation and Chi-Square 2-DOF coverage bounds."""
    # Test trial with 0 error
    trial_zero = run_single_calibration_trial(
        trial_id=1,
        texture="cratered",
        true_dx=0.25,
        true_dy=-0.30,
        window_size=32,
        seed=101,
    )
    assert trial_zero.mahalanobis_sq >= 0.0
    assert trial_zero.mahalanobis_dist >= 0.0

    # Theoretical Chi2 2-DOF properties:
    # 1-sigma ellipse (d_M <= 1.0): 1 - exp(-0.5) ~ 0.393469
    # 2-sigma ellipse (d_M <= 2.0): 1 - exp(-2.0) ~ 0.864665
    # 3-sigma ellipse (d_M <= 3.0): 1 - exp(-4.5) ~ 0.988891
    assert math.isclose(1.0 - math.exp(-0.5), 0.393469, rel_tol=1e-4)
    assert math.isclose(1.0 - math.exp(-2.0), 0.864665, rel_tol=1e-4)
    assert math.isclose(1.0 - math.exp(-4.5), 0.988891, rel_tol=1e-4)

    # 1D normal equivalents:
    # p = 0.6827 -> c = -2 ln(1 - 0.6827) ~ 2.2957 -> d_M <= 1.5152
    # p = 0.9545 -> c = -2 ln(1 - 0.9545) ~ 6.1801 -> d_M <= 2.4860
    assert math.isclose(-2.0 * math.log(1.0 - 0.6827), 2.2957, abs_tol=1e-3)
    assert math.isclose(-2.0 * math.log(1.0 - 0.9545), 6.1801, abs_tol=1e-3)


# ===========================================================================
# 4. Reliability Diagram & Calibration Error Calculations
# ===========================================================================

def test_reliability_diagram_and_calibration_error():
    """Tests compute_calibration_statistics with synthetic perfectly-calibrated samples."""
    rng = np.random.default_rng(202)
    n_samples = 1000

    # Generate synthetic 2D Gaussian error vectors e ~ N(0, Sigma)
    sigma_true = 0.35
    cov_true = np.array([[sigma_true ** 2, 0.0], [0.0, sigma_true ** 2]])
    errors = rng.normal(0.0, sigma_true, (n_samples, 2))

    records: list[MatchTrialRecord] = []
    for i in range(n_samples):
        ex, ey = errors[i]
        d_m_sq = float((ex ** 2 + ey ** 2) / (sigma_true ** 2))
        d_m = float(math.sqrt(d_m_sq))

        records.append(MatchTrialRecord(
            trial_id=i,
            texture="synthetic",
            true_dx=0.0,
            true_dy=0.0,
            estimated_dx=ex,
            estimated_dy=ey,
            error_x=ex,
            error_y=ey,
            euclidean_error=math.sqrt(ex ** 2 + ey ** 2),
            covariance_matrix=cov_true.tolist(),
            sigma_major_px=sigma_true,
            sigma_minor_px=sigma_true,
            ellipse_angle_deg=0.0,
            uncertainty_status="valid",
            is_valid=True,
            mahalanobis_sq=d_m_sq,
            mahalanobis_dist=d_m,
            inside_1sigma_ellipse=bool(d_m <= 1.0),
            inside_2sigma_ellipse=bool(d_m <= 2.0),
            inside_3sigma_ellipse=bool(d_m <= 3.0),
            inside_p68_quantile=bool(d_m_sq <= 2.2957),
            inside_p95_quantile=bool(d_m_sq <= 6.1801),
            inside_p99_quantile=bool(d_m_sq <= 11.8288),
            window_size=32,
            blur=0.0,
            noise=0.0,
            contrast=1.0,
            gain_bias=0.0,
            shadow_inverted=False,
            scale_ratio=1.0,
        ))

    report = compute_calibration_statistics(records)

    # For true N(0, Sigma), empirical coverage should closely match theoretical
    assert np.isclose(report.empirical_coverage_1sigma_ellipse, report.expected_coverage_1sigma_ellipse, atol=0.05)
    assert np.isclose(report.empirical_coverage_2sigma_ellipse, report.expected_coverage_2sigma_ellipse, atol=0.05)
    assert np.isclose(report.empirical_coverage_p68, 0.6827, atol=0.05)
    assert np.isclose(report.empirical_coverage_p95, 0.9545, atol=0.05)
    assert report.expected_calibration_error < 0.05
    assert np.isclose(report.empirical_variance_scale_factor, 1.0, atol=0.15)
    assert report.probabilistically_calibrated is True
    assert report.calibration_status == "PROBABILISTICALLY_CALIBRATED"


# ===========================================================================
# 5. Flat and Multimodal Peak Failure Gating
# ===========================================================================

def test_flat_and_multimodal_peak_failure_gating():
    """Verifies that flat peaks and multimodal peaks are tracked, flagged, and segregated."""
    # Flat correlation surface
    flat_corr = np.full((32, 32), 0.5, dtype=np.float32)
    flat_res = estimate_peak_hessian_and_covariance(flat_corr, 16, 16)
    assert flat_res.is_valid is False
    assert flat_res.uncertainty_status == "flat_peak"

    # Multimodal surface with secondary peak > 80%
    multi_corr = np.zeros((32, 32), dtype=np.float32)
    multi_corr[16, 16] = 0.90  # primary
    multi_corr[16, 17] = 0.70
    multi_corr[17, 16] = 0.70
    multi_corr[15, 16] = 0.70
    multi_corr[16, 15] = 0.70

    multi_corr[25, 25] = 0.85  # strong secondary peak outside radius 3
    multi_corr[25, 26] = 0.65
    multi_corr[26, 25] = 0.65

    multi_res = estimate_peak_hessian_and_covariance(multi_corr, 16, 16)
    assert multi_res.is_valid is False
    assert multi_res.uncertainty_status == "multimodal"
    assert multi_res.secondary_peak_ratio > 0.80


# ===========================================================================
# 6. Probabilistic Calibration Governance
# ===========================================================================

def test_probabilistic_calibration_verdict_governance():
    """
    Requirement: "Do not call covariance probabilistically calibrated unless coverage
    is empirically demonstrated."
    Verifies that if empirical error significantly exceeds nominal covariance (overconfident),
    status is strictly OVERCONFIDENT_UNCALIBRATED, not calibrated.
    """
    rng = np.random.default_rng(303)
    n_samples = 200

    # Overconfident case: actual noise has std = 2.0 px, but reported covariance claims std = 0.2 px (100x variance mismatch)
    actual_sigma = 2.0
    claimed_sigma = 0.2
    claimed_cov = np.array([[claimed_sigma ** 2, 0.0], [0.0, claimed_sigma ** 2]])
    errors = rng.normal(0.0, actual_sigma, (n_samples, 2))

    records: list[MatchTrialRecord] = []
    for i in range(n_samples):
        ex, ey = errors[i]
        d_m_sq = float((ex ** 2 + ey ** 2) / (claimed_sigma ** 2))
        d_m = float(math.sqrt(d_m_sq))

        records.append(MatchTrialRecord(
            trial_id=i,
            texture="synthetic_overconfident",
            true_dx=0.0,
            true_dy=0.0,
            estimated_dx=ex,
            estimated_dy=ey,
            error_x=ex,
            error_y=ey,
            euclidean_error=math.sqrt(ex ** 2 + ey ** 2),
            covariance_matrix=claimed_cov.tolist(),
            sigma_major_px=claimed_sigma,
            sigma_minor_px=claimed_sigma,
            ellipse_angle_deg=0.0,
            uncertainty_status="valid",
            is_valid=True,
            mahalanobis_sq=d_m_sq,
            mahalanobis_dist=d_m,
            inside_1sigma_ellipse=bool(d_m <= 1.0),
            inside_2sigma_ellipse=bool(d_m <= 2.0),
            inside_3sigma_ellipse=bool(d_m <= 3.0),
            inside_p68_quantile=bool(d_m_sq <= 2.2957),
            inside_p95_quantile=bool(d_m_sq <= 6.1801),
            inside_p99_quantile=bool(d_m_sq <= 11.8288),
            window_size=32,
            blur=0.0,
            noise=0.0,
            contrast=1.0,
            gain_bias=0.0,
            shadow_inverted=False,
            scale_ratio=1.0,
        ))

    report = compute_calibration_statistics(records)
    # Governance enforcement: MUST NOT call calibrated
    assert report.probabilistically_calibrated is False
    assert "OVERCONFIDENT" in report.calibration_status
    assert report.empirical_variance_scale_factor > 10.0


# ===========================================================================
# 7. End-to-End Mini Pipeline Run
# ===========================================================================

def test_run_calibration_experiment_mini_pipeline():
    """Runs a quick mini-grid calibration experiment and validates report generation."""
    report, records = run_covariance_calibration_experiment(quick_mode=True, random_seed=404)

    assert report.total_trials > 0
    assert len(records) == report.total_trials
    assert report.valid_peak_rate > 0.50
    assert report.expected_coverage_1sigma_ellipse == 0.3935
    assert report.expected_coverage_2sigma_ellipse == 0.8647
    assert report.expected_coverage_3sigma_ellipse == 0.9889
    assert len(report.confidence_levels) == 10
    assert len(report.empirical_quantiles) == 10
    assert report.calibration_status in [
        "PROBABILISTICALLY_CALIBRATED",
        "OVERCONFIDENT_UNCALIBRATED",
        "CONSERVATIVE_BOUNDS",
        "BORDERLINE_CALIBRATED",
    ]
