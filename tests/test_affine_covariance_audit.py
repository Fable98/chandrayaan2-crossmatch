import numpy as np
import pytest
import cv2
from ML_model.covariance_geometry import (
    refine_affine_covariance_weighted,
    audit_affine_covariance_dependence,
    compare_affine_gls_solvers,
    combine_covariances,
    compute_whitening_matrix,
    compute_dual_rmse_metrics,
)


def _generate_synthetic_affine_data(
    n_points: int = 40,
    anisotropic_ratio: float = 15.0,
    seed: int = 42,
):
    """
    Generates synthetic correspondences under a known ground-truth affine transform
    with significant rotation (40 deg), scale, and shear, where source points
    have strongly anisotropic measurement uncertainty.
    """
    rng = np.random.default_rng(seed)

    # True affine transformation: rotation (40 deg) + non-uniform scale + shear + translation
    theta = np.deg2rad(40.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    S = np.array([[1.18, 0.15], [0.0, 0.88]], dtype=np.float64)
    A_true = R @ S
    t_true = np.array([35.0, -22.0], dtype=np.float64)
    H_true = np.array([
        [A_true[0, 0], A_true[0, 1], t_true[0]],
        [A_true[1, 0], A_true[1, 1], t_true[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # True clean source points
    pts1_clean = rng.uniform(100.0, 900.0, size=(n_points, 2))

    # Clean destination points
    pts2_clean = (A_true @ pts1_clean.T).T + t_true

    # Source covariances: strongly anisotropic (elongated along x-axis)
    # sigma_x = 3.0 px, sigma_y = 0.2 px (ratio 15:1)
    covs_src = []
    pts1_noisy = np.zeros_like(pts1_clean)
    for i in range(n_points):
        # Varying orientation of source uncertainty
        sigma_major = 2.5
        sigma_minor = sigma_major / anisotropic_ratio
        cov_diag = np.diag([sigma_major ** 2, sigma_minor ** 2])
        # Random slight rotation for source ellipse
        phi = rng.uniform(-0.2, 0.2)
        c_p, s_p = np.cos(phi), np.sin(phi)
        R_p = np.array([[c_p, -s_p], [s_p, c_p]])
        cov_s = R_p @ cov_diag @ R_p.T
        noise_s = rng.multivariate_normal([0, 0], cov_s)
        pts1_noisy[i] = pts1_clean[i] + noise_s
        covs_src.append(cov_s)

    # Destination covariances: moderate isotropic noise
    covs_dst = []
    pts2_noisy = np.zeros_like(pts2_clean)
    for i in range(n_points):
        cov_d = np.eye(2) * (0.25 ** 2)
        noise_d = rng.multivariate_normal([0, 0], cov_d)
        pts2_noisy[i] = pts2_clean[i] + noise_d
        covs_dst.append(cov_d)

    return pts1_noisy, pts2_noisy, covs_src, covs_dst, H_true


def test_affine_covariance_dependence_audit():
    """
    Determine whether residual covariance is:
    1. fixed from an initial transform,
    2. recomputed during iterations,
    3. dependent on the affine parameters.
    """
    pts1, pts2, covs_src, covs_dst, H_true = _generate_synthetic_affine_data(n_points=10)

    # Case A: When source covariance is present
    audit_iter = audit_affine_covariance_dependence(
        covariances_src=covs_src,
        covariances_dst=covs_dst,
        mode="iterative",
    )
    assert audit_iter["is_dependent_on_affine_parameters"] is True
    assert audit_iter["is_recomputed_during_iterations"] is True
    assert audit_iter["is_fixed_from_initial_transform"] is False
    assert "Cov(r_i) = A Sigma_{src, i} A^T + Sigma_{dst, i}" in audit_iter["rationale"]

    # Case B: When in one-pass mode
    audit_one = audit_affine_covariance_dependence(
        covariances_src=covs_src,
        covariances_dst=covs_dst,
        mode="one_pass",
    )
    assert audit_one["is_dependent_on_affine_parameters"] is True
    assert audit_one["is_recomputed_during_iterations"] is False
    assert audit_one["is_fixed_from_initial_transform"] is True

    # Case C: When source covariance is zero/absent (only destination uncertainty)
    audit_no_src = audit_affine_covariance_dependence(
        covariances_src=None,
        covariances_dst=covs_dst,
        mode="iterative",
    )
    assert audit_no_src["is_dependent_on_affine_parameters"] is False
    assert audit_no_src["is_recomputed_during_iterations"] is False
    assert audit_no_src["is_fixed_from_initial_transform"] is True


def test_irgls_recomputes_covariance_and_converges_with_tolerances():
    """
    Verify that iterative GLS:
    - recomputes residual-space covariance after each update,
    - stops using parameter and objective tolerances,
    - caps iterations,
    - retains the best valid solution.
    """
    pts1, pts2, covs_src, covs_dst, H_true = _generate_synthetic_affine_data(n_points=35, seed=77)

    H_aff, diag = refine_affine_covariance_weighted(
        pts1,
        pts2,
        covariances_src=covs_src,
        covariances_dst=covs_dst,
        max_iters=15,
        tol_param=1e-5,
        tol_obj=1e-4,
        mode="iterative",
        return_diagnostics=True,
    )

    assert H_aff is not None
    assert diag["is_source_covariance_dependent"] is True
    assert diag["iterations"] > 1
    assert diag["iterations"] <= 15
    assert diag["convergence_status"] in ["converged_tolerances", "max_iters_reached"]
    assert diag["final_objective"] <= diag["initial_objective"]
    assert diag["best_iteration"] >= 1

    # Verify history captures decreasing or stabilizing objective
    history = diag["history"]
    assert len(history) == diag["iterations"]
    assert history[0]["cost"] >= history[-1]["cost"] - 1e-4


def test_iteration_capping_and_best_solution_retention():
    """
    Verify that IRGLS strictly caps iterations at max_iters and retains the best valid solution.
    """
    pts1, pts2, covs_src, covs_dst, _ = _generate_synthetic_affine_data(n_points=25, seed=12)

    # Set very tight tolerances so it runs exactly max_iters=3
    H_aff, diag = refine_affine_covariance_weighted(
        pts1,
        pts2,
        covariances_src=covs_src,
        covariances_dst=covs_dst,
        max_iters=3,
        tol_param=1e-12,
        tol_obj=1e-12,
        mode="iterative",
        return_diagnostics=True,
    )

    assert H_aff is not None
    assert diag["iterations"] == 3
    assert diag["convergence_status"] == "max_iters_reached"
    assert diag["best_iteration"] in [1, 2, 3]


def test_compare_one_pass_and_iterative_gls_on_anisotropic_source():
    """
    Compare one-pass GLS and iterative GLS on synthetic affine transformations
    with anisotropic source covariance.
    Document the approximation if one-pass mode remains available.
    """
    pts1, pts2, covs_src, covs_dst, H_true = _generate_synthetic_affine_data(
        n_points=50,
        anisotropic_ratio=20.0,
        seed=101,
    )

    comp = compare_affine_gls_solvers(
        pts1,
        pts2,
        covariances_src=covs_src,
        covariances_dst=covs_dst,
        H_true=H_true,
    )

    one_pass = comp["one_pass_gls"]
    iterative = comp["iterative_gls"]

    # Iterative GLS must run more than 1 iteration
    assert one_pass["iterations"] == 1
    assert iterative["iterations"] > 1

    # Iterative GLS must achieve lower or equal Mahalanobis cost than one-pass GLS
    assert iterative["mahalanobis_cost"] <= one_pass["mahalanobis_cost"] + 1e-4
    assert comp["iterative_improves_mahalanobis"] is True

    # Iterative GLS provides superior parameter recovery under anisotropic source covariance
    # because it rotates the covariance metric tensor to match the true affine transformation
    assert iterative["param_error_frobenius"] < one_pass["param_error_frobenius"]

    # Verify dependence audit is attached
    assert comp["dependence_audit"]["is_dependent_on_affine_parameters"] is True


def test_backward_compatibility_with_single_covariance_sequence():
    """
    Verifies that calling refine_affine_covariance_weighted(pts1, pts2, covariances)
    retains full backward compatibility with existing codebase callers.
    """
    rng = np.random.default_rng(5)
    pts1 = rng.uniform(50, 450, size=(15, 2))
    pts2 = pts1 + np.array([12.0, -8.0]) + rng.normal(0, 0.1, size=(15, 2))
    covs = [np.eye(2) * 0.2 for _ in range(15)]

    # Standard call as in existing tests
    H_aff = refine_affine_covariance_weighted(pts1, pts2, covs)
    assert H_aff is not None
    assert H_aff.shape == (3, 3)
    assert np.all(np.isfinite(H_aff))
