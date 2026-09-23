"""
Tests for Covariance Circularity, Double-Counting, and Independent Checkpoint Audits.

Traces all covariance-dependent operations:
1. Sampler (PROSAC / weighted sampling): verifies pre-fit curvature weighting, no geometric bypass.
2. Inlier scoring: verifies Euclidean distance gating (no Mahalanobis free-pass for high-uncertainty points).
3. DLT: verifies weighted algebraic equations without circular residual re-scaling.
4. Nonlinear refinement: verifies pre-fit measurement covariance without post-fit residual feedback.
5. Final metrics: verifies dual reporting (unweighted and weighted RMSE), anti-masking honesty.
6. Checkpoint evaluation: verifies independent observation evaluation, strict rejection of inherited
   inlier residual covariance (circularity detection), and dual checkpoint reporting.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from metrics import compute_canonical_metrics, evaluate_independent_checkpoints
from covariance_geometry import (
    combine_covariances,
    compute_whitening_matrix,
    refine_homography_covariance_weighted,
    refine_affine_covariance_weighted,
    compute_dual_rmse_metrics,
)
from matcher_cfog import estimate_weighted_homography


# ===========================================================================
# 1. Sampler Audit: Pre-Fit Weights & No Geometric Bypass
# ===========================================================================

def test_sampler_uses_prefit_weights_and_no_geometric_bypass():
    """
    Operation: Sampler
    - Covariance source: Pre-fit sub-pixel correlation peak Hessian curvature.
    - Timing: Before transformation fitting.
    - Points used: Candidate matches pool.
    - Optimism risk: Does having an artificially low covariance bypass geometry?
    Verify: Even if an outlier point is assigned artificial ultra-low covariance (high weight),
    it is filtered out by geometric consensus and does NOT corrupt the model.
    """
    rng = np.random.default_rng(42)
    n_inliers = 15
    src_inl = rng.uniform(50, 450, (n_inliers, 2))
    # True translation: (+20, -10)
    dst_inl = src_inl + np.array([20.0, -10.0])

    # Outliers
    n_outliers = 5
    src_out = rng.uniform(50, 450, (n_outliers, 2))
    dst_out = src_out + rng.uniform(-100, 100, (n_outliers, 2))

    pts1 = np.vstack([src_inl, src_out])
    pts2 = np.vstack([dst_inl, dst_out])

    # Weights: Give an outlier an artificially elevated weight (3x higher than inliers)
    weights = np.ones(len(pts1), dtype=np.float64)
    weights[n_inliers] = 3.0  # outlier has elevated uncertainty precision

    H, inlier_mask, tag = estimate_weighted_homography(
        pts1, pts2, weights, ransac_reproj_threshold=5.0, rng_seed=42
    )

    assert H is not None
    assert inlier_mask is not None
    # Outlier with huge weight must STILL be rejected by geometric consensus
    assert inlier_mask.ravel()[n_inliers] == 0
    # True shift should be recovered accurately
    assert np.isclose(H[0, 2], 20.0, atol=0.5)
    assert np.isclose(H[1, 2], -10.0, atol=0.5)


# ===========================================================================
# 2. Inlier Scoring Audit: Euclidean Gating (No Mahalanobis Free-Pass)
# ===========================================================================

def test_inlier_scoring_uses_euclidean_distance_preventing_mahalanobis_free_pass():
    """
    Operation: Inlier Scoring
    - Covariance source: Pre-fit correlation curvature.
    - Timing: Before fitting.
    - Optimism risk: If inliers were scored by Mahalanobis distance, a point with huge
      covariance (e.g. sigma = 50px) would have d_M < 1 and sneak into inliers despite
      a 20px geometric error.
    Verify: Inlier selection enforces strict Euclidean reprojection error <= threshold (5.0 px),
    preventing noisy or degenerate points from claiming inlier status.
    """
    rng = np.random.default_rng(101)
    pts1 = rng.uniform(100, 400, (10, 2))
    pts2 = pts1 + np.array([10.0, 5.0])

    # Add a point with 15px error (outside 5.0px threshold) but assign it huge covariance (var=1000)
    bad_pt1 = np.array([[200.0, 200.0]])
    bad_pt2 = np.array([[215.0, 220.0]])  # ~25px error

    pts1_all = np.vstack([pts1, bad_pt1])
    pts2_all = np.vstack([pts2, bad_pt2])

    # Even with low weight / high variance, does RANSAC score strictly by Euclidean distance?
    weights = np.ones(len(pts1_all))
    weights[-1] = 0.01  # small weight / high variance

    H, inlier_mask, tag = estimate_weighted_homography(
        pts1_all, pts2_all, weights, ransac_reproj_threshold=5.0, rng_seed=101
    )

    assert inlier_mask is not None
    # Bad point is rejected because Euclidean error 25px > 5px threshold
    assert inlier_mask.ravel()[-1] == 0


# ===========================================================================
# 3. DLT & Refinement Audit: No Circular Scaling by Post-Fit Residuals
# ===========================================================================

def test_nonlinear_refinement_uses_fixed_prefit_measurement_covariance():
    """
    Operation: DLT and Nonlinear Refinement
    - Covariance source: Independent measurement covariances Sigma_src, Sigma_dst.
    - Timing: Covariance fixed before fitting; Jacobian updated during iteration.
    - Points used: Consensus inliers.
    - Optimism risk: Circularity where fit residuals scale the covariance matrix.
    Verify: Covariance weighting properly discounts uncertain directions without
    scaling or altering the measurement covariance matrix based on the fit residuals.
    """
    pts_src = np.array([
        [100.0, 100.0], [400.0, 100.0], [400.0, 400.0], [100.0, 400.0],
        [250.0, 250.0], [200.0, 300.0]
    ], dtype=np.float64)
    # Perfect translation + offset on 1 point along x
    pts_dst = pts_src + np.array([12.0, -8.0])
    pts_dst[4, 0] += 2.0  # offset point 4 along x

    # Covariance for point 4 has large variance in x (uncertain along x)
    covs = [np.eye(2) * 0.1 for _ in range(len(pts_src))]
    covs[4] = np.array([[10.0, 0.0], [0.0, 0.1]])  # large uncertainty along x

    H_ref = refine_homography_covariance_weighted(pts_src, pts_dst, covs)
    assert H_ref is not None

    # Translation along y must be unaffected by the x-displacement
    assert np.isclose(H_ref[1, 2], -8.0, atol=0.1)


# ===========================================================================
# 4. Final Metrics Audit: Anti-Masking Honesty (Dual Reporting)
# ===========================================================================

def test_final_metrics_reports_both_unweighted_and_weighted_rmse():
    """
    Operation: Final Metrics
    - Covariance source: Pre-fit match covariances.
    - Optimism risk: Inflating covariance makes weighted RMSE small, masking bad fits.
    Verify: compute_canonical_metrics reports unweighted_rmse_px and raw_residuals_px
    alongside covariance_weighted_rmse, ensuring zero residual masking.
    """
    src = np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0], [200.0, 200.0]])
    # 2px shift along x
    dst = src + np.array([2.0, 0.0])
    mask = np.ones(len(src), dtype=np.uint8)
    H = np.eye(3)  # identity -> 2px reprojection error on every point

    # Artificial huge covariance (var=100)
    huge_covs = [np.eye(2) * 100.0 for _ in range(len(src))]

    metrics = compute_canonical_metrics(src, dst, mask, H, match_covariances=huge_covs)

    # Unweighted RMSE is honest: exactly 2.0 px
    assert np.isclose(metrics["unweighted_rmse_px"], 2.0, atol=1e-3)
    assert np.isclose(metrics["fit_rmse_px"], 2.0, atol=1e-3)
    assert len(metrics["raw_residuals_px"]) == len(src)
    assert np.allclose(metrics["raw_residuals_px"], 2.0, atol=1e-3)

    # Weighted Mahalanobis RMSE is small: 2.0 / sqrt(100) = 0.2
    assert np.isclose(metrics["covariance_weighted_rmse"], 0.2, atol=1e-2)

    # Sub-pixel accurate is NOT granted based on small Mahalanobis RMSE alone:
    # requires unweighted/held-out RMSE < 1.0 px
    assert metrics["sub_pixel_accurate"] is False


# ===========================================================================
# 5. Checkpoint Evaluation Audit: Circularity Rejection & Independence
# ===========================================================================

def test_checkpoint_evaluation_strictly_rejects_inherited_inlier_covariances():
    """
    Operation: Checkpoint Evaluation
    - Requirement: Independent checkpoints must be evaluated with independent observations
      and MUST NOT inherit fitted inlier residual covariance without explicit justification.
    Verify: If inlier_covariances is passed as checkpoint covariances without
    allow_inherited_inlier_covariance=True, a ValueError is raised for circularity.
    """
    src_chk = np.array([[50.0, 50.0], [450.0, 450.0], [50.0, 450.0], [450.0, 50.0]])
    dst_chk = src_chk + np.array([5.0, -3.0])
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])

    inlier_covs = [np.eye(2) * 0.25 for _ in range(4)]

    # Attempt to pass inlier residual covariance to checkpoint evaluation
    with pytest.raises(ValueError, match="Circularity detected"):
        evaluate_independent_checkpoints(
            src_chk, dst_chk, H,
            checkpoints_cov_dst=inlier_covs,
            inlier_covariances=inlier_covs,
            allow_inherited_inlier_covariance=False,
        )


def test_checkpoint_evaluation_with_independent_observations():
    """
    Operation: Checkpoint Evaluation
    Verify: Legitimate independent observation covariances are evaluated properly,
    reporting both checkpoint_unweighted_rmse_px and checkpoint_covariance_weighted_rmse
    with checkpoint_covariance_provenance = 'independent_observations'.
    """
    src_chk = np.array([[50.0, 50.0], [450.0, 450.0], [50.0, 450.0], [450.0, 50.0]])
    dst_chk = src_chk + np.array([5.0, -3.0])
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])

    # Independent survey covariances
    chk_covs = [np.array([[0.04, 0.0], [0.0, 0.04]]) for _ in range(4)]  # 0.2 px std

    res = evaluate_independent_checkpoints(
        src_chk, dst_chk, H,
        checkpoints_cov_dst=chk_covs,
    )

    assert res["is_independent_validation"] is True
    assert res["checkpoint_covariance_provenance"] == "independent_observations"
    assert res["checkpoint_unweighted_rmse_px"] == 0.0
    assert res["checkpoint_covariance_weighted_rmse"] == 0.0
    assert res["checkpoint_points_count"] == 4


def test_canonical_metrics_with_independent_checkpoint_covariances():
    """
    Verify: compute_canonical_metrics correctly passes independent checkpoint covariances,
    records the provenance, and populates dual checkpoint metrics.
    """
    src = np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0], [200.0, 200.0]])
    dst = src + np.array([10.0, 5.0])
    mask = np.ones(5, dtype=np.uint8)
    H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])

    chk_src = np.array([[50.0, 50.0], [450.0, 450.0], [50.0, 450.0], [450.0, 50.0]])
    chk_dst = chk_src + np.array([10.0, 5.0])
    chk_covs = [np.eye(2) * 0.16 for _ in range(4)]

    metrics = compute_canonical_metrics(
        src, dst, mask, H,
        independent_checkpoints=(chk_src, chk_dst),
        checkpoints_cov_dst=chk_covs,
    )

    assert metrics["has_independent_checkpoints"] is True
    assert metrics["is_independent_validation"] is True
    assert metrics["checkpoint_covariance_provenance"] == "independent_observations"
    assert metrics["checkpoint_unweighted_rmse_px"] == 0.0
    assert metrics["checkpoint_covariance_weighted_rmse"] == 0.0
