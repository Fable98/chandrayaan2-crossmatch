"""
tests/test_covariance_geometry.py — Verification of Covariance-Weighted Geometric Estimation.

Requirements tested:
1. Combine source and reference covariance into residual-space covariance.
2. Whiten residuals using Cholesky or eigen decomposition.
3. Use covariance-weighted affine/homography refinement after robust inlier selection.
4. Keep robust RANSAC/MAGSAC for outlier rejection (low-uncertainty outlier rejection).
5. Report both unweighted and covariance-weighted RMSE.
6. Prevent singular covariance matrices using bounded eigenvalue floors.
7. Add synthetic tests where:
   - high-confidence isotropic points dominate appropriately,
   - elongated uncertain points receive lower directional weight,
   - a low-uncertainty outlier is still rejected by robust estimation.
8. Do not use uncertainty to hide bad residuals; report raw residuals too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from covariance_geometry import (
    combine_covariances,
    compute_dual_rmse_metrics,
    compute_homography_jacobian,
    compute_whitening_matrix,
    condition_covariance,
    refine_affine_covariance_weighted,
    refine_homography_covariance_weighted,
    whiten_residuals,
)
from matcher_cfog import estimate_weighted_homography, match_images_cfog
from metrics import compute_canonical_metrics


def test_condition_covariance_and_eigenvalue_flooring():
    """Requirement 6: Singular covariance matrices are prevented using bounded eigenvalue floors."""
    # Test 1: Rank-deficient singular covariance (zero eigenvalue)
    singular_cov = np.array([[4.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    cond_cov = condition_covariance(singular_cov, min_eig=1e-4, max_eig=100.0)

    w, _ = np.linalg.eigh(cond_cov)
    assert w[0] >= 1e-4, f"Zero eigenvalue was not floored: {w[0]}"
    assert np.all(np.isfinite(cond_cov))

    # Test 2: Negative eigenvalue (non-positive definite)
    non_pos_cov = np.array([[2.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    cond_pos = condition_covariance(non_pos_cov, min_eig=1e-4)
    w_pos, _ = np.linalg.eigh(cond_pos)
    assert np.all(w_pos >= 1e-4)

    # Test 3: Extremely ill-conditioned matrix
    ill_cov = np.array([[100.0, 0.0], [0.0, 1e-8]], dtype=np.float64)
    cond_ill = condition_covariance(ill_cov, min_eig=1e-4, max_cond=1e3)
    w_ill, _ = np.linalg.eigh(cond_ill)
    assert w_ill[1] / w_ill[0] <= 1e3 + 1e-5


def test_residual_space_covariance_combination():
    """Requirement 1: Combine source and reference covariance into residual-space covariance."""
    # Source covariance: isotropic sigma = 0.5 px
    cov_src = (0.5**2) * np.eye(2)
    # Destination covariance: anisotropic sigma_x = 1.0 px, sigma_y = 0.2 px
    cov_dst = np.diag([1.0**2, 0.2**2])

    # 2D Affine Jacobian: scale by 2.0 along x, 1.5 along y
    J_T = np.diag([2.0, 1.5])

    # Theoretical residual covariance:
    # J_T @ cov_src @ J_T.T + cov_dst = diag(4 * 0.25 + 1.0, 2.25 * 0.25 + 0.04) = diag(2.0, 0.6025)
    cov_res = combine_covariances(cov_src, cov_dst, J_T=J_T)

    assert abs(cov_res[0, 0] - 2.0) < 1e-4
    assert abs(cov_res[1, 1] - 0.6025) < 1e-4
    assert abs(cov_res[0, 1]) < 1e-6


def test_residual_whitening_cholesky_and_eigen():
    """Requirement 2: Whiten residuals using Cholesky or eigen decomposition."""
    cov = np.array([[4.0, 1.2], [1.2, 2.0]], dtype=np.float64)

    # Cholesky whitening matrix W
    W_chol = compute_whitening_matrix(cov, method="cholesky")
    ident_chol = W_chol @ cov @ W_chol.T
    np.testing.assert_allclose(ident_chol, np.eye(2), atol=1e-5)

    # Eigen whitening matrix W
    W_eigen = compute_whitening_matrix(cov, method="eigen")
    ident_eigen = W_eigen @ cov @ W_eigen.T
    np.testing.assert_allclose(ident_eigen, np.eye(2), atol=1e-5)

    # Whitened residuals and Mahalanobis distance
    residuals = np.array([[2.0, -1.0], [0.5, 0.5]], dtype=np.float64)
    whitened, dists = whiten_residuals(residuals, [cov, cov])

    for i in range(len(residuals)):
        d_m_direct = np.sqrt(residuals[i].T @ np.linalg.inv(cov) @ residuals[i])
        assert abs(dists[i] - d_m_direct) < 1e-5
        assert abs(np.linalg.norm(whitened[i]) - d_m_direct) < 1e-5


def test_synthetic_high_confidence_isotropic_points_dominate():
    """Requirement 3, 7: High-confidence isotropic points dominate geometric refinement appropriately."""
    # True transformation: 2D affine (scale 1.1, rotation 5 deg, translation [12, -8])
    theta = np.radians(5.0)
    s = 1.1
    A_true = np.array([
        [s * np.cos(theta), -s * np.sin(theta), 12.0],
        [s * np.sin(theta), s * np.cos(theta), -8.0],
        [0.0, 0.0, 1.0],
    ])

    rng = np.random.RandomState(42)
    # 4 high-confidence points (sigma = 0.05 px)
    pts1_high = np.array([[50.0, 50.0], [450.0, 50.0], [450.0, 450.0], [50.0, 450.0]], dtype=np.float64)
    pts2_high_clean = (A_true @ np.hstack([pts1_high, np.ones((4, 1))]).T).T[:, :2]
    pts2_high = pts2_high_clean + rng.normal(0.0, 0.02, pts2_high_clean.shape)
    covs_high = [(0.05**2) * np.eye(2) for _ in range(4)]

    # 10 low-confidence points with substantial noise (sigma = 4.0 px)
    pts1_low = rng.uniform(80.0, 400.0, size=(10, 2))
    pts2_low_clean = (A_true @ np.hstack([pts1_low, np.ones((10, 1))]).T).T[:, :2]
    pts2_low = pts2_low_clean + rng.normal(0.0, 3.5, pts2_low_clean.shape)
    covs_low = [(4.0**2) * np.eye(2) for _ in range(10)]

    all_pts1 = np.vstack([pts1_high, pts1_low])
    all_pts2 = np.vstack([pts2_high, pts2_low])
    all_covs = covs_high + covs_low

    # 1. Unweighted Least Squares Affine
    M_unweighted, _ = cv2.estimateAffine2D(all_pts1, all_pts2)
    H_unweighted = np.vstack([M_unweighted, [0.0, 0.0, 1.0]])

    # 2. Covariance-Weighted Affine Refinement
    H_cov_affine = refine_affine_covariance_weighted(all_pts1, all_pts2, all_covs)
    assert H_cov_affine is not None

    # Compare error on the high-confidence points
    err_unweighted = np.linalg.norm((H_unweighted @ np.hstack([pts1_high, np.ones((4, 1))]).T).T[:, :2] - pts2_high_clean, axis=1)
    err_cov = np.linalg.norm((H_cov_affine @ np.hstack([pts1_high, np.ones((4, 1))]).T).T[:, :2] - pts2_high_clean, axis=1)

    rmse_unweighted = float(np.sqrt(np.mean(err_unweighted**2)))
    rmse_cov = float(np.sqrt(np.mean(err_cov**2)))

    # Covariance-weighted fit must achieve significantly better accuracy on high-confidence anchors
    assert rmse_cov < rmse_unweighted
    assert rmse_cov < 0.20, f"Expected covariance-weighted RMSE < 0.20px, got {rmse_cov:.4f}"

    # Also test homography refinement
    H_cov_homo = refine_homography_covariance_weighted(all_pts1, all_pts2, all_covs, H_init=H_unweighted)
    assert H_cov_homo is not None
    err_homo = np.linalg.norm((H_cov_homo @ np.hstack([pts1_high, np.ones((4, 1))]).T).T[:, :2] - pts2_high_clean, axis=1)
    rmse_homo = float(np.sqrt(np.mean(err_homo**2)))
    assert rmse_homo < rmse_unweighted


def test_synthetic_elongated_uncertain_points_receive_lower_directional_weight():
    """Requirement 7: Elongated uncertain points (ridge aperture problem) receive lower directional weight."""
    # True translation: dx = 15.0, dy = -10.0
    H_true = np.array([
        [1.0, 0.0, 15.0],
        [0.0, 1.0, -10.0],
        [0.0, 0.0, 1.0],
    ])

    rng = np.random.RandomState(101)
    # 6 points along horizontal crater rims: very uncertain along x (sigma_x = 6.0 px), but tight across y (sigma_y = 0.15 px)
    pts1 = np.array([
        [100.0, 100.0],
        [200.0, 100.0],
        [300.0, 100.0],
        [100.0, 300.0],
        [200.0, 300.0],
        [300.0, 300.0],
    ], dtype=np.float64)

    pts2_clean = pts1 + np.array([15.0, -10.0])
    # Add large noise ONLY along x (simulating ridge sliding)
    pts2_noisy = pts2_clean.copy()
    pts2_noisy[:, 0] += rng.normal(0.0, 4.0, size=len(pts1))
    pts2_noisy[:, 1] += rng.normal(0.0, 0.05, size=len(pts1))

    # Anisotropic covariances
    anisotropic_covs = [np.diag([6.0**2, 0.15**2]) for _ in range(len(pts1))]

    # 1. Unweighted Affine
    M_unw, _ = cv2.estimateAffine2D(pts1, pts2_noisy)
    H_unw = np.vstack([M_unw, [0.0, 0.0, 1.0]])

    # 2. Covariance-Weighted Affine
    H_cov = refine_affine_covariance_weighted(pts1, pts2_noisy, anisotropic_covs)
    assert H_cov is not None

    # Measure across-ridge error (y-direction):
    proj_unw = (H_unw @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T[:, :2]
    proj_cov = (H_cov @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T[:, :2]

    y_err_unw = np.abs(proj_unw[:, 1] - pts2_clean[:, 1])
    y_err_cov = np.abs(proj_cov[:, 1] - pts2_clean[:, 1])

    # Across-ridge error must be tiny in covariance-weighted fit
    assert np.mean(y_err_cov) <= np.mean(y_err_unw) + 1e-4
    assert np.mean(y_err_cov) < 0.20, f"Expected y-error < 0.20px, got {np.mean(y_err_cov):.4f}"


def test_synthetic_low_uncertainty_outlier_rejected_by_robust_estimation():
    """Requirement 4, 7: Low-uncertainty outlier is still rejected by robust estimation before refinement."""
    H_true = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 5.0],
        [0.0, 0.0, 1.0],
    ])

    # 8 consistent inlier points
    pts1 = np.array([
        [50.0, 50.0], [150.0, 60.0], [250.0, 50.0], [350.0, 70.0],
        [60.0, 250.0], [160.0, 260.0], [260.0, 250.0], [360.0, 270.0],
    ], dtype=np.float32)
    pts2 = (pts1 + np.array([10.0, 5.0])).astype(np.float32)

    # Add 1 gross outlier (dx = +80, dy = -50) with artificially tiny covariance (false confident match)
    outlier_pt1 = np.array([[200.0, 150.0]], dtype=np.float32)
    outlier_pt2 = np.array([[280.0, 100.0]], dtype=np.float32)

    all_pts1 = np.vstack([pts1, outlier_pt1])
    all_pts2 = np.vstack([pts2, outlier_pt2])

    # 8 normal weights/covariances, 1 outlier with high weight (tiny covariance)
    weights = np.array([0.5] * 8 + [1.0], dtype=np.float64)
    covs = [np.eye(2) * (0.5**2)] * 8 + [np.eye(2) * (0.01**2)]

    H_est, inlier_mask, tag = estimate_weighted_homography(
        all_pts1, all_pts2, weights,
        estimator_method=cv2.RANSAC,
        ransac_reproj_threshold=5.0,
        covariances=covs,
    )

    assert H_est is not None
    assert inlier_mask is not None

    # The outlier must be REJECTED by robust RANSAC despite its tiny covariance
    mask_flat = inlier_mask.ravel()
    assert mask_flat[-1] == 0, "Low-uncertainty outlier was falsely accepted as an inlier!"
    # The 8 genuine inliers must be accepted
    assert np.all(mask_flat[:8] == 1)


def test_dual_rmse_reporting_and_raw_residuals_not_hidden():
    """Requirement 5, 8: Report both unweighted and covariance-weighted RMSE; do not hide raw residuals."""
    H_true = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, -5.0],
        [0.0, 0.0, 1.0],
    ])

    src = np.array([[100.0, 100.0], [200.0, 100.0], [100.0, 200.0], [200.0, 200.0]], dtype=np.float64)
    # Known residuals: [1.0, 0.0], [0.0, 2.0], [-1.0, -1.0], [0.0, 0.0]
    dst = (src + np.array([10.0, -5.0])) - np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [-1.0, -1.0],
        [0.0, 0.0],
    ])

    # Unweighted RMSE calculation
    raw_res_vecs = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0], [0.0, 0.0]])
    expected_unweighted_rmse = np.sqrt(np.mean(np.sum(raw_res_vecs**2, axis=1)))

    covs = [np.eye(2) for _ in range(4)]
    res = compute_dual_rmse_metrics(src, dst, H_true, covariances_dst=covs)

    assert "unweighted_rmse_px" in res
    assert "covariance_weighted_rmse" in res
    assert "raw_residuals_px" in res
    assert "raw_residual_vectors_px" in res
    assert "mahalanobis_distances" in res

    assert abs(res["unweighted_rmse_px"] - round(float(expected_unweighted_rmse), 4)) < 1e-4

    # Requirement 8: Artificial inflation of covariance must NOT alter unweighted RMSE or raw residuals
    inflated_covs = [np.eye(2) * 100.0 for _ in range(4)]
    res_inflated = compute_dual_rmse_metrics(src, dst, H_true, covariances_dst=inflated_covs)

    assert res_inflated["unweighted_rmse_px"] == res["unweighted_rmse_px"]
    assert res_inflated["raw_residuals_px"] == res["raw_residuals_px"]
    # Mahalanobis distance scales down with inflated covariance, but raw residuals stay untouched
    assert res_inflated["covariance_weighted_rmse"] < res["covariance_weighted_rmse"]


def test_canonical_metrics_populates_dual_rmse():
    """Requirement 5: compute_canonical_metrics outputs dual RMSE and residual fields."""
    src = np.array([[50.0, 50.0], [150.0, 50.0], [50.0, 150.0], [150.0, 150.0]], dtype=np.float32)
    dst = src + np.array([2.0, -1.0], dtype=np.float32)
    mask = np.ones(4, dtype=np.uint8)
    H = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    covs = [np.eye(2) * 0.25 for _ in range(4)]
    metrics = compute_canonical_metrics(src, dst, mask, H, match_covariances=covs)

    assert "unweighted_rmse_px" in metrics
    assert "covariance_weighted_rmse" in metrics
    assert "raw_residuals_px" in metrics
    assert "mahalanobis_residuals" in metrics
    assert metrics["unweighted_rmse_px"] is not None
    assert metrics["covariance_weighted_rmse"] is not None


def test_end_to_end_matcher_with_covariance_refinement(tmp_path):
    """Requirement 3, 5: End-to-end match_images_cfog runs covariance refinement and reports dual RMSE."""
    rng = np.random.RandomState(42)
    img1 = rng.uniform(80, 180, (256, 256)).astype(np.uint8)
    img1 = cv2.GaussianBlur(img1, (7, 7), 2.0)
    # Add high-contrast textured crater spots
    for cx, cy, r in [
        (50, 50, 16), (120, 60, 14), (200, 70, 18),
        (60, 130, 15), (140, 140, 20), (210, 130, 16),
        (70, 200, 18), (150, 210, 15), (210, 200, 17),
    ]:
        cv2.circle(img1, (cx, cy), r, 30, -1)
        cv2.circle(img1, (cx, cy), r // 2, 220, -1)

    M = np.float32([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]])
    img2 = cv2.warpAffine(img1, M, (256, 256))

    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)

    res = match_images_cfog(
        p1, p2, output_dir=tmp_path / "out",
        explicit_gsd1=1.0,
        explicit_gsd2=1.0,
        enable_covariance_refinement=True,
        propagate_uncertainty_to_weights=True,
    )

    assert res.get("status") == "success"
    metrics = res.get("metrics") or {}
    assert "unweighted_rmse_px" in metrics
    assert "covariance_weighted_rmse" in metrics
    assert "raw_residuals_px" in metrics
    assert metrics.get("unweighted_rmse_px") is not None
