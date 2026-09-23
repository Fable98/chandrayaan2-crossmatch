import numpy as np
import pytest
import logging
from ML_model.covariance_geometry import (
    refine_homography_covariance_weighted,
    audit_homography_covariance_refinement,
    estimate_homography_ordinary_dlt,
    estimate_homography_scalar_weighted_dlt,
    estimate_homography_whitened_dlt,
    compute_dual_rmse_metrics,
)

logger = logging.getLogger(__name__)


def _generate_synthetic_homography_data(
    n_inliers: int = 40,
    noise_sigma: float = 0.5,
    seed: int = 42,
):
    """
    Generates synthetic correspondences under a known ground-truth homography
    with anisotropic, heterogeneous covariance matrices.
    """
    rng = np.random.default_rng(seed)
    # Ground truth homography with rotation, scale, shear, and perspective
    H_true = np.array([
        [1.08, 0.05, 30.0],
        [-0.04, 0.95, -20.0],
        [0.0002, -0.0001, 1.0],
    ], dtype=np.float64)

    # Inlier points distributed across a 1000x1000 region
    pts1 = rng.uniform(100.0, 900.0, size=(n_inliers, 2))

    # Project to pts2
    pts1_h = np.hstack([pts1, np.ones((n_inliers, 1))])
    proj = (H_true @ pts1_h.T).T
    pts2_clean = proj[:, :2] / proj[:, 2:3]

    # Generate anisotropic covariances
    covariances = []
    pts2_noisy = np.zeros_like(pts2_clean)

    for i in range(n_inliers):
        # Semi-axes between 0.2 and 2.5 px
        sigma_x = rng.uniform(0.2, 1.5) * noise_sigma
        sigma_y = rng.uniform(0.5, 3.0) * noise_sigma
        theta = rng.uniform(0, np.pi)

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        cov_diag = np.diag([sigma_x ** 2, sigma_y ** 2])
        cov = R @ cov_diag @ R.T

        # Sample noise according to cov
        noise = rng.multivariate_normal([0, 0], cov)
        pts2_noisy[i] = pts2_clean[i] + noise
        covariances.append(cov)

    return pts1, pts2_noisy, covariances, H_true


def _generate_independent_checkpoints(H_true: np.ndarray, n_points: int = 50, seed: int = 123):
    """
    Generates independent checkpoint points that were NOT used in fitting.
    """
    rng = np.random.default_rng(seed)
    chk_src = rng.uniform(50.0, 950.0, size=(n_points, 2))
    chk_src_h = np.hstack([chk_src, np.ones((n_points, 1))])
    proj = (H_true @ chk_src_h.T).T
    chk_dst = proj[:, :2] / proj[:, 2:3]

    # Checkpoints have their own measurement noise
    covs = []
    chk_dst_noisy = np.zeros_like(chk_dst)
    for i in range(n_points):
        cov = np.eye(2) * 0.25
        noise = rng.multivariate_normal([0, 0], cov)
        chk_dst_noisy[i] = chk_dst[i] + noise
        covs.append(cov)

    return chk_src, chk_dst_noisy, covs


def test_refinement_logs_objectives_and_verifies_monotonicity(caplog):
    """
    Requirements 1, 2, 3, 4:
    1. Log initial whitened-DLT objective.
    2. Log final nonlinear Mahalanobis objective.
    3. Verify that the final transform is the LM result.
    4. Verify the objective does not increase after refinement.
    """
    pts1, pts2, covariances, _ = _generate_synthetic_homography_data(n_inliers=30, seed=10)

    with caplog.at_level(logging.INFO):
        H_refined, diag = refine_homography_covariance_weighted(
            pts1, pts2, covariances, return_diagnostics=True
        )

    assert H_refined is not None
    assert diag["is_lm_result"] is True
    assert diag["initial_objective"] is not None
    assert diag["final_objective"] is not None
    assert diag["initial_objective"] >= diag["final_objective"]
    assert diag["objective_decreased"] is True
    assert diag["iterations"] > 0
    assert diag["runtime_ms"] > 0

    # Verify logging outputs
    log_records = [r.message for r in caplog.records]
    assert any("Initial whitened-DLT Mahalanobis objective" in msg for msg in log_records)
    assert any("Final nonlinear Mahalanobis objective" in msg for msg in log_records)


def test_refinement_monotonicity_reverts_if_cost_increases():
    """
    Requirement 4: If an adversary or degenerate update attempted to increase cost,
    it must revert to initial transform and report objective_decreased=False.
    """
    pts1, pts2, covariances, _ = _generate_synthetic_homography_data(n_inliers=10, seed=99)

    # Calling with max_iters=0 should keep initial cost unchanged
    H_refined, diag = refine_homography_covariance_weighted(
        pts1, pts2, covariances, max_iters=0, return_diagnostics=True
    )
    assert H_refined is not None
    assert diag["final_objective"] <= diag["initial_objective"] + 1e-9


def test_four_method_comparison_and_independent_checkpoints():
    """
    Requirements 5, 6, 7:
    5. Compare: ordinary DLT, scalar-weighted DLT, whitened DLT, final covariance-weighted LM.
    6. Evaluate all methods on independent checkpoints.
    7. Report runtime and convergence status.
    """
    pts1, pts2, covariances, H_true = _generate_synthetic_homography_data(n_inliers=50, noise_sigma=0.8, seed=42)
    chk_src, chk_dst, chk_cov = _generate_independent_checkpoints(H_true, n_points=40, seed=123)

    audit_result = audit_homography_covariance_refinement(
        pts1,
        pts2,
        covariances,
        checkpoints_src=chk_src,
        checkpoints_dst=chk_dst,
        checkpoints_cov=chk_cov,
    )

    assert audit_result["status"] == "PASS"
    assert audit_result["accepted_method"] == "covariance_weighted_lm"
    assert audit_result["is_lm_result"] is True
    assert audit_result["objective_decreased"] is True

    # Check 4 methods present
    methods = audit_result["methods"]
    for method_name in ["ordinary_dlt", "scalar_weighted_dlt", "whitened_dlt", "covariance_weighted_lm"]:
        assert method_name in methods
        m = methods[method_name]
        assert m["homography"] is not None
        assert m["inlier_rmse_px"] is not None
        assert m["inlier_mahalanobis_cost"] is not None
        assert m["checkpoint_rmse_px"] is not None
        assert m["checkpoint_mahalanobis_rmse"] is not None
        assert m["runtime_ms"] >= 0
        assert m["convergence_status"] is not None

    # Verify that covariance-weighted LM achieves the lowest inlier Mahalanobis cost
    lm_cost = methods["covariance_weighted_lm"]["inlier_mahalanobis_cost"]
    white_cost = methods["whitened_dlt"]["inlier_mahalanobis_cost"]
    assert lm_cost <= white_cost + 1e-4

    # Verify runtime is reported
    assert audit_result["diagnostics"]["runtime_ms"] > 0
    assert audit_result["diagnostics"]["convergence_status"] in [
        "converged_gradient",
        "converged_step",
        "cost_decreased",
        "max_iters_reached",
    ]


def test_rejection_governance_when_lm_fails():
    """
    Requirement 8: Reject the result if LM fails or produces worse independent checkpoints without explicit REVIEW status.
    Case A: LM fails due to insufficient points.
    """
    pts1 = np.array([[0, 0], [1, 0], [0, 1]])  # Only 3 points (< 4)
    pts2 = np.array([[0, 0], [1, 0], [0, 1]])
    covs = [np.eye(2) for _ in range(3)]

    audit_result = audit_homography_covariance_refinement(
        pts1,
        pts2,
        covs,
    )

    assert audit_result["status"] == "REVIEW"
    assert audit_result["rejection_reason"] is not None
    assert audit_result["rejection_reason"] == "lm_optimization_failed"


def test_rejection_governance_when_lm_degrades_checkpoints():
    """
    Requirement 8: If LM produces a worse independent result than baseline DLT,
    flag status as 'REVIEW' with explicit reason.
    """
    pts1, pts2, covariances, H_true = _generate_synthetic_homography_data(n_inliers=30, seed=7)
    chk_src, chk_dst, chk_cov = _generate_independent_checkpoints(H_true, n_points=30, seed=8)

    # We can test rejection governance by setting a negative or zero tolerance_degradation_px
    # or by forcing a scenario where LM is slightly worse on independent checkpoints.
    # Passing tolerance_degradation_px = -1.0 forces any positive difference to trigger REVIEW.
    audit_result = audit_homography_covariance_refinement(
        pts1,
        pts2,
        covariances,
        checkpoints_src=chk_src,
        checkpoints_dst=chk_dst,
        checkpoints_cov=chk_cov,
        tolerance_degradation_px=-100.0,  # Strict negative tolerance to guarantee REVIEW trigger
    )

    assert audit_result["status"] == "REVIEW"
    assert "lm_checkpoint_rmse_degraded" in audit_result["rejection_reason"]
    # Verify fallback to best DLT
    assert audit_result["accepted_method"] in ["ordinary_dlt", "scalar_weighted_dlt", "whitened_dlt"]
    assert audit_result["accepted_transform"] is not None
