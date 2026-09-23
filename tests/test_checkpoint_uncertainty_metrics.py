import numpy as np
import pytest
from ML_model.metrics import (
    evaluate_independent_checkpoints,
    compute_canonical_metrics,
)


def _generate_synthetic_checkpoint_data(
    n_points: int = 20,
    noise_sigma: float = 0.4,
    seed: int = 42,
):
    """
    Generates synthetic independent checkpoints with known true homography
    and independent observation covariances.
    """
    rng = np.random.default_rng(seed)
    H_true = np.array([
        [1.05, -0.02, 12.0],
        [0.03, 0.98, -8.0],
        [0.0001, -0.00005, 1.0],
    ], dtype=np.float64)

    # Independent surveyed points in image domain
    chk_src = rng.uniform(50.0, 450.0, size=(n_points, 2))
    chk_src_h = np.hstack([chk_src, np.ones((n_points, 1))])
    proj = (H_true @ chk_src_h.T).T
    chk_dst_clean = proj[:, :2] / proj[:, 2:3]

    covs_dst = []
    chk_dst_noisy = np.zeros_like(chk_dst_clean)
    for i in range(n_points):
        # Semi-axes around noise_sigma
        s1 = float(rng.uniform(0.2, 0.6))
        s2 = float(rng.uniform(0.2, 0.6))
        cov = np.diag([s1**2, s2**2])
        noise = rng.multivariate_normal([0, 0], cov)
        chk_dst_noisy[i] = chk_dst_clean[i] + noise
        covs_dst.append(cov)

    return chk_src, chk_dst_noisy, covs_dst, H_true


def test_checkpoint_evaluation_with_independent_covariance_and_meters():
    """
    Verifies that evaluate_independent_checkpoints reports:
    - raw Euclidean residual in pixels,
    - residual in metres,
    - checkpoint covariance-weighted Mahalanobis error,
    - covariance ellipse containment,
    - median and p95 values,
    - validation mode,
    - covariance provenance.
    """
    chk_src, chk_dst, chk_cov, H = _generate_synthetic_checkpoint_data(n_points=25, seed=10)
    pixel_res_m = 0.5  # e.g., 0.5 m/pixel (Chandrayaan-2 OHRC scale)

    res = evaluate_independent_checkpoints(
        checkpoints_src=chk_src,
        checkpoints_dst=chk_dst,
        H=H,
        checkpoints_cov_dst=chk_cov,
        pixel_resolution_m=pixel_res_m,
    )

    # 1. Validation mode & provenance
    assert res["validation_mode"] == "INDEPENDENT_CHECKPOINTS"
    assert res["is_independent_validation"] is True
    assert res["validation_status"] == "evaluated"
    assert res["checkpoint_covariance_provenance"] == "independent_observations"
    assert res["checkpoint_points_count"] == 25

    # 2. Raw Euclidean residuals in pixels
    assert res["checkpoint_rmse_px"] is not None
    assert res["checkpoint_unweighted_rmse_px"] == res["checkpoint_rmse_px"]
    assert res["checkpoint_mean_error_px"] is not None
    assert res["checkpoint_median_error_px"] is not None
    assert res["checkpoint_p95_error_px"] is not None
    assert res["checkpoint_max_error_px"] is not None
    assert len(res["checkpoint_raw_residuals_px"]) == 25
    assert len(res["checkpoint_raw_residual_vectors_px"]) == 25

    # 3. Residuals in metres
    assert res["pixel_resolution_m"] == 0.5
    assert res["checkpoint_rmse_m"] is not None
    assert pytest.approx(res["checkpoint_rmse_m"], rel=1e-2) == res["checkpoint_rmse_px"] * pixel_res_m
    assert pytest.approx(res["checkpoint_median_error_m"], rel=1e-2) == res["checkpoint_median_error_px"] * pixel_res_m
    assert pytest.approx(res["checkpoint_p95_error_m"], rel=1e-2) == res["checkpoint_p95_error_px"] * pixel_res_m
    assert len(res["checkpoint_raw_residuals_m"]) == 25

    # 4. Checkpoint covariance-weighted Mahalanobis error
    assert res["checkpoint_covariance_weighted_rmse"] is not None
    assert len(res["checkpoint_mahalanobis_distances"]) == 25
    assert res["checkpoint_median_mahalanobis"] is not None
    assert res["checkpoint_p95_mahalanobis"] is not None
    assert res["checkpoint_mean_mahalanobis"] is not None
    assert res["checkpoint_max_mahalanobis"] is not None

    # 5. Covariance ellipse containment
    containment = res["covariance_ellipse_containment"]
    assert containment is not None
    assert 0.0 <= containment["containment_1sigma_fraction"] <= 1.0
    assert 0.0 <= containment["containment_2sigma_fraction"] <= 1.0
    assert 0.0 <= containment["containment_3sigma_fraction"] <= 1.0
    assert 0.0 <= containment["containment_p95_fraction"] <= 1.0
    # Higher sigmas must have equal or higher containment
    assert containment["containment_3sigma_fraction"] >= containment["containment_2sigma_fraction"]
    assert containment["containment_2sigma_fraction"] >= containment["containment_1sigma_fraction"]
    assert containment["total_checkpoints"] == 25

    # Top-level aliases for containment fractions
    assert res["containment_1sigma_fraction"] == containment["containment_1sigma_fraction"]
    assert res["containment_2sigma_fraction"] == containment["containment_2sigma_fraction"]
    assert res["containment_3sigma_fraction"] == containment["containment_3sigma_fraction"]
    assert res["containment_p95_fraction"] == containment["containment_p95_fraction"]


def test_checkpoint_evaluation_when_covariance_is_unavailable():
    """
    Requirement:
    If checkpoint covariance is unavailable:
    - report raw metrics,
    - set Mahalanobis checkpoint metrics to null,
    - do not borrow training-match covariance.
    """
    chk_src, chk_dst, _, H = _generate_synthetic_checkpoint_data(n_points=15, seed=77)
    # Training match covariances (which MUST NOT be borrowed)
    training_inlier_covs = [np.eye(2) * 0.01 for _ in range(15)]

    res = evaluate_independent_checkpoints(
        checkpoints_src=chk_src,
        checkpoints_dst=chk_dst,
        H=H,
        checkpoints_cov_dst=None,  # Unavailable
        inlier_covariances=training_inlier_covs,  # Training match covariance
        allow_inherited_inlier_covariance=False,
    )

    # 1. Provenance reports unweighted euclidean
    assert res["checkpoint_covariance_provenance"] == "unweighted_euclidean"
    assert res["validation_mode"] == "INDEPENDENT_CHECKPOINTS"
    assert res["is_independent_validation"] is True

    # 2. Raw metrics are reported
    assert res["checkpoint_rmse_px"] is not None
    assert res["checkpoint_median_error_px"] is not None
    assert res["checkpoint_p95_error_px"] is not None
    assert len(res["checkpoint_raw_residuals_px"]) == 15

    # 3. All Mahalanobis metrics are strictly None / null
    assert res["checkpoint_covariance_weighted_rmse"] is None
    assert res["checkpoint_mahalanobis_distances"] is None
    assert res["checkpoint_mean_mahalanobis"] is None
    assert res["checkpoint_median_mahalanobis"] is None
    assert res["checkpoint_p95_mahalanobis"] is None
    assert res["checkpoint_max_mahalanobis"] is None

    # 4. Ellipse containment is strictly None / null
    assert res["covariance_ellipse_containment"] is None
    assert res["containment_1sigma_fraction"] is None
    assert res["containment_2sigma_fraction"] is None
    assert res["containment_3sigma_fraction"] is None
    assert res["containment_p95_fraction"] is None


def test_circularity_rejection_when_borrowing_training_covariance():
    """
    Requirement:
    Do not borrow training-match covariance. Passing inlier_covariances
    as checkpoints_cov_dst raises ValueError unless explicitly allowed.
    """
    chk_src, chk_dst, _, H = _generate_synthetic_checkpoint_data(n_points=10)
    fake_training_covs = [np.eye(2) * 0.05 for _ in range(10)]

    # Attempting to borrow inlier covariances directly
    with pytest.raises(ValueError, match="Circularity detected"):
        evaluate_independent_checkpoints(
            checkpoints_src=chk_src,
            checkpoints_dst=chk_dst,
            H=H,
            checkpoints_cov_dst=fake_training_covs,
            inlier_covariances=fake_training_covs,
            allow_inherited_inlier_covariance=False,
        )


def test_canonical_metrics_integration_with_uncertainty_checkpoints():
    """
    Verifies that compute_canonical_metrics seamlessly incorporates all extended
    uncertainty-aware checkpoint metrics, including meter resolution scaling.
    """
    chk_src, chk_dst, chk_cov, H = _generate_synthetic_checkpoint_data(n_points=20, seed=55)

    # Inliers for the fit
    rng = np.random.default_rng(123)
    inliers_src = rng.uniform(100, 400, size=(12, 2))
    inliers_dst = inliers_src + np.array([12.0, -8.0]) + rng.normal(0, 0.1, size=(12, 2))

    metrics = compute_canonical_metrics(
        src_pts_raw=inliers_src,
        dst_pts_raw=inliers_dst,
        inlier_mask=np.ones(12, dtype=int),
        H=H,
        gsd_m=0.5,  # 0.5 meters per pixel
        checkpoints_src=chk_src,
        checkpoints_dst=chk_dst,
        checkpoints_cov_dst=chk_cov,
    )

    assert metrics["validation_mode"] == "INDEPENDENT_CHECKPOINTS"
    assert metrics["is_independent_validation"] is True
    assert metrics["checkpoint_rmse_px"] is not None
    assert metrics["checkpoint_rmse_m"] is not None
    assert metrics["pixel_resolution_m"] == 0.5
    assert metrics["checkpoint_covariance_weighted_rmse"] is not None
    assert metrics["checkpoint_covariance_provenance"] == "independent_observations"
    assert metrics["checkpoint_median_mahalanobis"] is not None
    assert metrics["checkpoint_p95_mahalanobis"] is not None
    assert metrics["covariance_ellipse_containment"] is not None
    assert metrics["containment_1sigma_fraction"] is not None
    assert metrics["containment_2sigma_fraction"] is not None
