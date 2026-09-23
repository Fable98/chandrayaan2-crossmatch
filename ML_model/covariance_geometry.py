"""
ML_model/covariance_geometry.py — Covariance-Weighted Geometric Estimation & Residual Whitening.

Photogrammetric module implementing:
1. Residual-space covariance combination combining source and reference covariance with transformation Jacobians.
2. Positive-definite covariance conditioning with bounded eigenvalue floors/ceilings.
3. Residual whitening via Cholesky and eigen decomposition.
4. Exact Generalized Least Squares (GLS) covariance-weighted 2D affine refinement.
5. Covariance-whitened DLT and Gauss-Newton non-linear homography refinement on consensus inliers.
6. Dual metric reporting: unweighted Euclidean RMSE, covariance-weighted Mahalanobis RMSE, and raw residuals.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Bounded eigenvalue parameters preventing singular covariance
DEFAULT_MIN_EIGENVALUE = 1e-4  # (0.01 px)^2
DEFAULT_MAX_EIGENVALUE = 100.0  # (10.0 px)^2
DEFAULT_MAX_CONDITION = 1e4


def condition_covariance(
    cov: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    min_eig: float = DEFAULT_MIN_EIGENVALUE,
    max_eig: float = DEFAULT_MAX_EIGENVALUE,
    max_cond: float = DEFAULT_MAX_CONDITION,
    default_sigma: float = 0.5,
) -> np.ndarray:
    """
    Conditions a 2x2 covariance matrix to guarantee strict symmetry,
    positive-definiteness, and bounded condition number.
    
    Args:
        cov: 2x2 covariance matrix, or None.
        min_eig: Minimum allowed eigenvalue floor (px^2).
        max_eig: Maximum allowed eigenvalue ceiling (px^2).
        max_cond: Maximum allowed condition number kappa = lambda_max / lambda_min.
        default_sigma: Standard deviation to use if cov is None or non-finite.
        
    Returns:
        Conditioned 2x2 symmetric positive-definite covariance matrix.
    """
    if cov is None:
        return (default_sigma ** 2) * np.eye(2, dtype=np.float64)

    cov_arr = np.asarray(cov, dtype=np.float64)
    if cov_arr.shape != (2, 2) or not np.all(np.isfinite(cov_arr)):
        return (default_sigma ** 2) * np.eye(2, dtype=np.float64)

    # 1. Enforce exact symmetry
    sym = 0.5 * (cov_arr + cov_arr.T)

    # 2. Spectral decomposition
    try:
        w, v = np.linalg.eigh(sym)
    except np.linalg.LinAlgError:
        return (default_sigma ** 2) * np.eye(2, dtype=np.float64)

    # 3. Floor and ceiling eigenvalues
    w = np.clip(w, float(min_eig), float(max_eig))

    # 4. Cap condition number
    if w[1] > w[0] * max_cond:
        w[0] = max(w[0], w[1] / max_cond)

    return v @ np.diag(w) @ v.T


def compute_whitening_matrix(
    cov: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    min_eig: float = DEFAULT_MIN_EIGENVALUE,
    max_eig: float = DEFAULT_MAX_EIGENVALUE,
    method: str = "cholesky",
) -> np.ndarray:
    """
    Computes the 2x2 residual whitening matrix W such that W @ Sigma @ W.T = I.
    
    Args:
        cov: 2x2 covariance matrix.
        min_eig: Floor on eigenvalues.
        max_eig: Ceiling on eigenvalues.
        method: "cholesky" (lower triangular inverse) or "eigen" (symmetric inverse square root).
        
    Returns:
        2x2 whitening matrix W.
    """
    c = condition_covariance(cov, min_eig=min_eig, max_eig=max_eig)

    if method == "cholesky":
        try:
            L = np.linalg.cholesky(c)
            return np.linalg.inv(L)
        except np.linalg.LinAlgError:
            pass  # Fall back to eigen

    # Eigen-decomposition method: W = V @ diag(1 / sqrt(w)) @ V.T
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, min_eig)
    inv_sqrt = 1.0 / np.sqrt(w)
    return v @ np.diag(inv_sqrt) @ v.T


def compute_homography_jacobian(H: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """
    Computes the 2x2 Jacobian J = d(x')/d(x) for homography mapping x' = H * x.
    
    Args:
        H: 3x3 homography matrix.
        pt: 2D point (x, y) in source image coordinates.
        
    Returns:
        2x2 Jacobian matrix.
    """
    x, y = float(pt[0]), float(pt[1])
    h = np.asarray(H, dtype=np.float64)

    w1 = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    w2 = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    w3 = h[2, 0] * x + h[2, 1] * y + h[2, 2]

    if abs(w3) < 1e-12:
        w3 = 1e-12 if w3 >= 0 else -1e-12

    u = w1 / w3
    v = w2 / w3

    # Derivatives du/dx, du/dy, dv/dx, dv/dy
    j00 = (h[0, 0] - u * h[2, 0]) / w3
    j01 = (h[0, 1] - u * h[2, 1]) / w3
    j10 = (h[1, 0] - v * h[2, 0]) / w3
    j11 = (h[1, 1] - v * h[2, 1]) / w3

    return np.array([[j00, j01], [j10, j11]], dtype=np.float64)


def compute_numerical_jacobian(
    transform_fn: Any,
    pt: np.ndarray,
    step: float = 1e-5,
    order: int = 4,
) -> np.ndarray:
    """
    Computes numerical finite-difference Jacobian of a 2D->2D transform at point pt.
    
    Supports 2nd-order (central) or 4th-order central differences.
    
    Args:
        transform_fn: Callable taking (2,) array [x, y] and returning (2,) array [u, v].
        pt: (2,) evaluation point [x, y].
        step: Finite-difference step size h.
        order: Difference order (2 or 4).
        
    Returns:
        2x2 numerical Jacobian [[du/dx, du/dy], [dv/dx, dv/dy]].
    """
    x, y = float(pt[0]), float(pt[1])
    h = float(step)

    if order == 4:
        # 4th-order central difference: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        fx_p2 = np.asarray(transform_fn(np.array([x + 2 * h, y])), dtype=np.float64)
        fx_p1 = np.asarray(transform_fn(np.array([x + h, y])), dtype=np.float64)
        fx_m1 = np.asarray(transform_fn(np.array([x - h, y])), dtype=np.float64)
        fx_m2 = np.asarray(transform_fn(np.array([x - 2 * h, y])), dtype=np.float64)
        d_dx = (-fx_p2 + 8.0 * fx_p1 - 8.0 * fx_m1 + fx_m2) / (12.0 * h)

        fy_p2 = np.asarray(transform_fn(np.array([x, y + 2 * h])), dtype=np.float64)
        fy_p1 = np.asarray(transform_fn(np.array([x, y + h])), dtype=np.float64)
        fy_m1 = np.asarray(transform_fn(np.array([x, y - h])), dtype=np.float64)
        fy_m2 = np.asarray(transform_fn(np.array([x, y - 2 * h])), dtype=np.float64)
        d_dy = (-fy_p2 + 8.0 * fy_p1 - 8.0 * fy_m1 + fy_m2) / (12.0 * h)
    else:
        # 2nd-order central difference: (f(x+h) - f(x-h)) / (2h)
        fx_p = np.asarray(transform_fn(np.array([x + h, y])), dtype=np.float64)
        fx_m = np.asarray(transform_fn(np.array([x - h, y])), dtype=np.float64)
        d_dx = (fx_p - fx_m) / (2.0 * h)

        fy_p = np.asarray(transform_fn(np.array([x, y + h])), dtype=np.float64)
        fy_m = np.asarray(transform_fn(np.array([x, y - h])), dtype=np.float64)
        d_dy = (fy_p - fy_m) / (2.0 * h)

    return np.column_stack([d_dx, d_dy])


def compute_transformation_jacobian(
    H: np.ndarray,
    pt: np.ndarray,
    convention: str = "src_to_dst",
) -> np.ndarray:
    """
    Computes analytic 2x2 Jacobian for affine or homography transformation.
    
    Supports both:
    - "src_to_dst": d(x_dst)/d(x_src) evaluated at point pt (in source frame).
    - "dst_to_src": d(x_src)/d(x_dst) evaluated at point pt (in destination frame).
    
    Args:
        H: 3x3 projective or affine matrix (src -> dst).
        pt: Evaluation point (x, y) in the domain corresponding to convention.
        convention: "src_to_dst" or "dst_to_src".
        
    Returns:
        2x2 analytic Jacobian matrix.
    """
    H_mat = np.asarray(H, dtype=np.float64)
    if convention == "src_to_dst":
        return compute_homography_jacobian(H_mat, pt)
    elif convention == "dst_to_src":
        try:
            H_inv = np.linalg.inv(H_mat)
        except np.linalg.LinAlgError:
            raise ValueError("Transformation matrix is singular; inverse does not exist.")
        return compute_homography_jacobian(H_inv, pt)
    else:
        raise ValueError(f"Unknown convention: '{convention}'. Must be 'src_to_dst' or 'dst_to_src'.")



def combine_covariances(
    cov_src: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    cov_dst: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    J_T: Optional[np.ndarray] = None,
    default_var: float = 0.25**2,
) -> np.ndarray:
    """
    Combines source and destination covariance into residual-space covariance:
        Sigma_r = J_T @ Sigma_src @ J_T.T + Sigma_dst
        
    Args:
        cov_src: 2x2 source point localization covariance (or None).
        cov_dst: 2x2 destination point localization covariance (or None).
        J_T: 2x2 transformation Jacobian d(x')/d(x) (identity if None).
        default_var: Fallback variance if covariance is absent.
        
    Returns:
        Conditioned 2x2 residual covariance matrix.
    """
    c_src = condition_covariance(cov_src, default_sigma=float(np.sqrt(default_var)))
    c_dst = condition_covariance(cov_dst, default_sigma=float(np.sqrt(default_var)))

    if J_T is not None:
        J = np.asarray(J_T, dtype=np.float64)
        if J.shape == (2, 2) and np.all(np.isfinite(J)):
            sigma_r = J @ c_src @ J.T + c_dst
        else:
            sigma_r = c_src + c_dst
    else:
        sigma_r = c_src + c_dst

    return condition_covariance(sigma_r)


def whiten_residuals(
    residuals: np.ndarray,
    residual_covariances: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Whitens residual vectors using per-point residual covariances.
    
    Args:
        residuals: (N, 2) array of raw residual vectors r_i = T(x_i) - x'_i.
        residual_covariances: (N, 2, 2) sequence of residual covariances.
        
    Returns:
        whitened_residuals: (N, 2) array where r_tilde_i = W_i @ r_i.
        mahalanobis_distances: (N,) array of Mahalanobis distances d_M = ||r_tilde_i||.
    """
    res = np.asarray(residuals, dtype=np.float64)
    n = len(res)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)

    whitened = np.zeros_like(res)
    dists = np.zeros(n, dtype=np.float64)

    for i in range(n):
        cov = residual_covariances[i] if i < len(residual_covariances) else None
        W = compute_whitening_matrix(cov)
        r_w = W @ res[i]
        whitened[i] = r_w
        dists[i] = float(np.linalg.norm(r_w))

    return whitened, dists


def audit_affine_covariance_dependence(
    covariances_src: Optional[Sequence[Optional[np.ndarray]]] = None,
    covariances_dst: Optional[Sequence[Optional[np.ndarray]]] = None,
    H_init: Optional[np.ndarray] = None,
    mode: str = "iterative",
) -> Dict[str, Any]:
    """
    Audits whether affine residual covariance is:
    1. Fixed from an initial transform.
    2. Recomputed during iterations.
    3. Dependent on the affine parameters.
    
    Mathematical Derivation:
        Residual vector for correspondence i: r_i = A x_i + t - x'_i.
        Since x_i ~ N(mu_i, Sigma_{src, i}) and x'_i ~ N(mu'_i, Sigma_{dst, i}):
        Cov(r_i) = A Sigma_{src, i} A^T + Sigma_{dst, i}.
        
    Conclusion:
        - If Sigma_{src, i} != 0 for any correspondence, the residual-space covariance
          Sigma_{r, i}(A) strictly depends on the affine linear parameters A.
        - In one-pass GLS, Sigma_r is evaluated once at H_init (or neutral unweighted fit) and held fixed.
          This is an approximation whenever A differs from H_init and Sigma_{src} != 0.
        - In iterative GLS, Sigma_r(A^{(k)}) is recomputed after each update until
          parameter and objective tolerances are met, capping iterations, and retaining the best valid solution.
    """
    has_source_cov = False
    if covariances_src is not None:
        for c in covariances_src:
            if c is not None and np.any(np.asarray(c) != 0):
                has_source_cov = True
                break

    is_fixed = (mode == "one_pass") or (not has_source_cov)
    is_recomputed = (mode == "iterative") and has_source_cov
    is_dependent = has_source_cov

    rationale = (
        "Residual vector r_i = A x_i + t - x'_i gives Cov(r_i) = A Sigma_{src, i} A^T + Sigma_{dst, i}. "
        "When source measurement uncertainty is present (Sigma_{src} != 0), residual-space covariance "
        "is strictly dependent on the linear transformation parameters A. "
        "In one-pass GLS, Sigma_r is held fixed from an initial transform H_init (or neutral fit), which is an "
        "approximation when A differs from H_init. Iteratively reweighted GLS recomputes Sigma_r(A^{(k)}) "
        "after each update, stopping when parameter and objective tolerances are met, capping iterations, "
        "and retaining the best valid solution."
    )

    return {
        "is_fixed_from_initial_transform": is_fixed,
        "is_recomputed_during_iterations": is_recomputed,
        "is_dependent_on_affine_parameters": is_dependent,
        "has_source_covariance": has_source_cov,
        "mode": mode,
        "rationale": rationale,
    }


def refine_affine_covariance_weighted(
    pts1: np.ndarray,
    pts2: np.ndarray,
    covariances: Optional[Sequence[Optional[np.ndarray]]] = None,
    covariances_src: Optional[Sequence[Optional[np.ndarray]]] = None,
    covariances_dst: Optional[Sequence[Optional[np.ndarray]]] = None,
    H_init: Optional[np.ndarray] = None,
    max_iters: int = 15,
    tol_param: float = 1e-6,
    tol_obj: float = 1e-6,
    mode: str = "iterative",
    return_diagnostics: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Dict[str, Any]]]:
    """
    Computes Generalized Least Squares (GLS) 2D affine refinement:
        x' = A * x + t
    weighted by per-match 2x2 residual covariances.
    
    Residual Covariance Dependence:
        Residual vector r_i = A x_i + t - x'_i has covariance:
            Sigma_{r, i}(A) = A Sigma_{src, i} A^T + Sigma_{dst, i}.
        - If Sigma_{src} is present and non-zero, residual covariance depends directly
          on the affine parameters A.
        - Mode "iterative" implements Iteratively Reweighted GLS (IRGLS), recomputing
          Sigma_{r, i}(A^{(k)}) after each parameter update, stopping when parameter
          and objective tolerances are satisfied (or max_iters reached), and retaining the
          best valid solution.
        - Mode "one_pass" evaluates Sigma_{r, i} once at H_init (or unweighted least squares)
          and solves in a single linear pass. This is an approximation if Sigma_{src} != 0 and A != A_init.
          
    Args:
        pts1: (N, 2) source points.
        pts2: (N, 2) destination points.
        covariances: (N, 2, 2) sequence of per-match residual covariances (backward compatibility).
        covariances_src: (N, 2, 2) sequence of per-match source covariances.
        covariances_dst: (N, 2, 2) sequence of per-match destination covariances.
        H_init: 3x3 initial affine matrix (optional).
        max_iters: Maximum IRGLS iterations (capped at this value).
        tol_param: Parameter convergence tolerance (||p_{k+1} - p_k|| < tol_param).
        tol_obj: Objective convergence tolerance (|Phi_{k+1} - Phi_k| < tol_obj).
        mode: "iterative" (IRGLS) or "one_pass" (single-pass approximation).
        return_diagnostics: If True, returns (H_refined, diagnostics_dict).
        
    Returns:
        3x3 homogeneous affine transformation matrix, or (H, diagnostics) if return_diagnostics=True.
    """
    t_start = time.perf_counter()
    pts1_arr = np.asarray(pts1, dtype=np.float64)
    pts2_arr = np.asarray(pts2, dtype=np.float64)
    n = len(pts1_arr)

    default_fail_diag = {
        "mode": mode,
        "iterations": 0,
        "best_iteration": 0,
        "convergence_status": "insufficient_points",
        "initial_objective": None,
        "final_objective": None,
        "delta_objective": None,
        "delta_param": None,
        "is_source_covariance_dependent": False,
        "history": [],
        "runtime_ms": 0.0,
    }

    if n < 3 or len(pts2_arr) != n:
        if return_diagnostics:
            return None, default_fail_diag
        return None

    # Handle covariance inputs
    if covariances_dst is None and covariances is not None:
        covariances_dst = covariances

    has_source_cov = False
    if covariances_src is not None:
        for c in covariances_src:
            if c is not None and np.any(np.asarray(c) != 0):
                has_source_cov = True
                break

    # Determine initial affine transform A_curr, t_curr
    if H_init is not None and np.all(np.isfinite(H_init)) and abs(H_init[2, 2]) > 1e-9:
        H_norm = H_init / H_init[2, 2]
        A_curr = H_norm[:2, :2].astype(np.float64)
        t_curr = H_norm[:2, 2].astype(np.float64)
    else:
        # Compute ordinary unweighted least squares affine as neutral initial condition
        C_stack = []
        y_stack = []
        for i in range(n):
            x, y = pts1_arr[i, 0], pts1_arr[i, 1]
            xp, yp = pts2_arr[i, 0], pts2_arr[i, 1]
            C_stack.append([x, y, 1.0, 0.0, 0.0, 0.0])
            C_stack.append([0.0, 0.0, 0.0, x, y, 1.0])
            y_stack.append(xp)
            y_stack.append(yp)
        try:
            p_unw, _, _, _ = np.linalg.lstsq(
                np.asarray(C_stack, dtype=np.float64),
                np.asarray(y_stack, dtype=np.float64),
                rcond=1e-7,
            )
            A_curr = np.array([[p_unw[0], p_unw[1]], [p_unw[3], p_unw[4]]], dtype=np.float64)
            t_curr = np.array([p_unw[2], p_unw[5]], dtype=np.float64)
        except Exception:
            A_curr = np.eye(2, dtype=np.float64)
            t_curr = np.zeros(2, dtype=np.float64)

    p_curr = np.array([A_curr[0, 0], A_curr[0, 1], t_curr[0], A_curr[1, 0], A_curr[1, 1], t_curr[1]], dtype=np.float64)

    def _eval_cost(params: np.ndarray) -> float:
        A_m = np.array([[params[0], params[1]], [params[3], params[4]]], dtype=np.float64)
        t_v = np.array([params[2], params[5]], dtype=np.float64)
        total = 0.0
        for i in range(n):
            r_i = A_m @ pts1_arr[i] + t_v - pts2_arr[i]
            c_s = covariances_src[i] if covariances_src is not None and i < len(covariances_src) else None
            c_d = covariances_dst[i] if covariances_dst is not None and i < len(covariances_dst) else None
            c_r = combine_covariances(c_s, c_d, J_T=A_m)
            W_i = compute_whitening_matrix(c_r)
            r_w = W_i @ r_i
            total += float(r_w @ r_w)
        return total

    initial_cost = _eval_cost(p_curr)
    best_cost = float("inf")
    best_p = p_curr.copy()
    best_iter = 0

    max_steps = 1 if mode == "one_pass" else max(1, max_iters)
    convergence_status = "one_pass" if mode == "one_pass" else "max_iters_reached"
    history = []
    iters_taken = 0
    delta_p_last = None
    delta_obj_last = None

    for it in range(max_steps):
        iters_taken += 1
        # 1. Recompute residual-space covariance using current A_curr
        # 2. Form whitened GLS normal equations
        A_gls = np.zeros((6, 6), dtype=np.float64)
        b_gls = np.zeros(6, dtype=np.float64)

        for i in range(n):
            x, y = pts1_arr[i, 0], pts1_arr[i, 1]
            xp, yp = pts2_arr[i, 0], pts2_arr[i, 1]

            c_s = covariances_src[i] if covariances_src is not None and i < len(covariances_src) else None
            c_d = covariances_dst[i] if covariances_dst is not None and i < len(covariances_dst) else None
            c_r = combine_covariances(c_s, c_d, J_T=A_curr)
            W_i = compute_whitening_matrix(c_r)

            C_i = np.array([
                [x, y, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, x, y, 1.0],
            ], dtype=np.float64)
            y_i = np.array([xp, yp], dtype=np.float64)

            C_w = W_i @ C_i
            y_w = W_i @ y_i

            A_gls += C_w.T @ C_w
            b_gls += C_w.T @ y_w

        try:
            p_cand = np.linalg.solve(A_gls, b_gls)
        except np.linalg.LinAlgError:
            try:
                p_cand, _, _, _ = np.linalg.lstsq(A_gls, b_gls, rcond=1e-7)
            except Exception:
                convergence_status = "singular_system"
                break

        if not np.all(np.isfinite(p_cand)):
            convergence_status = "non_finite_parameter"
            break

        cand_cost = _eval_cost(p_cand)
        delta_p = float(np.linalg.norm(p_cand - p_curr))
        delta_obj = abs(cand_cost - (initial_cost if iters_taken == 1 else history[-1]["cost"]))
        delta_p_last = delta_p
        delta_obj_last = delta_obj

        # Retain best valid solution
        if cand_cost < best_cost:
            best_cost = cand_cost
            best_p = p_cand.copy()
            best_iter = iters_taken

        history.append({
            "iter": iters_taken,
            "cost": round(float(cand_cost), 6),
            "delta_p": round(float(delta_p), 8),
            "delta_obj": round(float(delta_obj), 6),
        })

        if mode == "one_pass":
            convergence_status = "one_pass"
            p_curr = p_cand
            break

        # If covariance does NOT depend on transform, 1 pass gives the exact solution
        if not has_source_cov:
            convergence_status = "exact_single_pass"
            p_curr = p_cand
            break

        # Check stopping criteria using parameter and objective tolerances
        if delta_p < tol_param and delta_obj < tol_obj:
            convergence_status = "converged_tolerances"
            p_curr = p_cand
            break

        # Update current parameters for next iteration
        p_curr = p_cand
        A_curr = np.array([[p_curr[0], p_curr[1]], [p_curr[3], p_curr[4]]], dtype=np.float64)
        t_curr = np.array([p_curr[2], p_curr[5]], dtype=np.float64)

    # Retain the best valid solution
    p_final = best_p if mode != "one_pass" else p_curr
    if not np.all(np.isfinite(p_final)):
        p_final = p_curr

    H_aff = np.array([
        [p_final[0], p_final[1], p_final[2]],
        [p_final[3], p_final[4], p_final[5]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    t_end = time.perf_counter()
    runtime_ms = (t_end - t_start) * 1000.0

    final_cost = _eval_cost(p_final)
    delta_obj_total = initial_cost - final_cost

    diagnostics = {
        "mode": mode,
        "iterations": iters_taken,
        "best_iteration": best_iter,
        "convergence_status": convergence_status,
        "initial_objective": round(float(initial_cost), 6),
        "final_objective": round(float(final_cost), 6),
        "delta_objective": round(float(delta_obj_total), 6),
        "delta_param": round(float(delta_p_last), 8) if delta_p_last is not None else 0.0,
        "is_source_covariance_dependent": has_source_cov,
        "history": history,
        "runtime_ms": round(float(runtime_ms), 3),
    }

    if return_diagnostics:
        return H_aff, diagnostics
    return H_aff


def compare_affine_gls_solvers(
    pts1: np.ndarray,
    pts2: np.ndarray,
    covariances_src: Optional[Sequence[Optional[np.ndarray]]] = None,
    covariances_dst: Optional[Sequence[Optional[np.ndarray]]] = None,
    H_true: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    max_iters: int = 15,
    tol_param: float = 1e-6,
    tol_obj: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compares one-pass GLS and iterative GLS on 2D affine refinement.
    
    Evaluates:
    - Ground-truth parameter error ||H - H_true||_F (if H_true provided).
    - Inlier Euclidean RMSE.
    - True inlier Mahalanobis objective Phi(A) = sum r_i^T (Sigma_{r, i}(A))^{-1} r_i.
    - Runtime and iteration count.
    - Convergence status.
    """
    H_one, diag_one = refine_affine_covariance_weighted(
        pts1,
        pts2,
        covariances_src=covariances_src,
        covariances_dst=covariances_dst,
        H_init=H_init,
        mode="one_pass",
        return_diagnostics=True,
    )

    H_iter, diag_iter = refine_affine_covariance_weighted(
        pts1,
        pts2,
        covariances_src=covariances_src,
        covariances_dst=covariances_dst,
        H_init=H_init,
        max_iters=max_iters,
        tol_param=tol_param,
        tol_obj=tol_obj,
        mode="iterative",
        return_diagnostics=True,
    )

    def _eval_model(H: Optional[np.ndarray]) -> Dict[str, Any]:
        if H is None:
            return {"euclidean_rmse_px": None, "mahalanobis_cost": None, "param_error_frobenius": None}
        d = compute_dual_rmse_metrics(pts1, pts2, H, covariances_src=covariances_src, covariances_dst=covariances_dst)
        m_cost = None
        if d.get("mahalanobis_distances"):
            m_cost = float(np.sum(np.array(d["mahalanobis_distances"]) ** 2))
        f_err = None
        if H_true is not None:
            f_err = float(np.linalg.norm(H - H_true, ord="fro"))
        return {
            "homography": H,
            "euclidean_rmse_px": d.get("unweighted_rmse_px"),
            "mahalanobis_cost": round(m_cost, 4) if m_cost is not None else None,
            "param_error_frobenius": round(f_err, 6) if f_err is not None else None,
        }

    res_one = _eval_model(H_one)
    res_one.update({
        "iterations": diag_one["iterations"],
        "runtime_ms": diag_one["runtime_ms"],
        "convergence_status": diag_one["convergence_status"],
    })

    res_iter = _eval_model(H_iter)
    res_iter.update({
        "iterations": diag_iter["iterations"],
        "runtime_ms": diag_iter["runtime_ms"],
        "convergence_status": diag_iter["convergence_status"],
        "best_iteration": diag_iter["best_iteration"],
        "delta_objective": diag_iter["delta_objective"],
    })

    dependence_audit = audit_affine_covariance_dependence(
        covariances_src=covariances_src,
        covariances_dst=covariances_dst,
        H_init=H_init,
        mode="iterative",
    )

    return {
        "one_pass_gls": res_one,
        "iterative_gls": res_iter,
        "dependence_audit": dependence_audit,
        "iterative_improves_mahalanobis": (
            res_iter["mahalanobis_cost"] <= res_one["mahalanobis_cost"] + 1e-4
            if (res_iter["mahalanobis_cost"] is not None and res_one["mahalanobis_cost"] is not None)
            else None
        ),
    }



def estimate_homography_ordinary_dlt(
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Standard unweighted Direct Linear Transformation (DLT) for 3x3 homography.
    Solves A h = 0 via SVD.
    """
    pts1_arr = np.asarray(pts1, dtype=np.float64)
    pts2_arr = np.asarray(pts2, dtype=np.float64)
    n = len(pts1_arr)
    if n < 4 or len(pts2_arr) != n:
        return None
    A = []
    for i in range(n):
        x, y = pts1_arr[i, 0], pts1_arr[i, 1]
        xp, yp = pts2_arr[i, 0], pts2_arr[i, 1]
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp])
    try:
        _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
        H = vt[-1].reshape(3, 3)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
            return H if np.all(np.isfinite(H)) else None
    except Exception:
        pass
    return None


def estimate_homography_scalar_weighted_dlt(
    pts1: np.ndarray,
    pts2: np.ndarray,
    covariances: Optional[Sequence[np.ndarray]] = None,
    weights: Optional[Sequence[float]] = None,
) -> Optional[np.ndarray]:
    """
    Scalar-weighted DLT row-scaled by sqrt(w_i) where w_i derives from inverse trace variance.
    Solves sum w_i ||A_i h||^2 = 0 via SVD.
    """
    pts1_arr = np.asarray(pts1, dtype=np.float64)
    pts2_arr = np.asarray(pts2, dtype=np.float64)
    n = len(pts1_arr)
    if n < 4 or len(pts2_arr) != n:
        return None

    if weights is not None and len(weights) == n:
        sw = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None))
    elif covariances is not None and len(covariances) == n:
        sw_list = []
        for i in range(n):
            cov_i = condition_covariance(covariances[i])
            tr = float(cov_i[0, 0] + cov_i[1, 1])
            w_i = 1.0 / max(tr / 2.0, 1e-4)
            sw_list.append(math.sqrt(w_i))
        sw = np.array(sw_list, dtype=np.float64)
    else:
        sw = np.ones(n, dtype=np.float64)

    A = []
    for i in range(n):
        x, y = pts1_arr[i, 0], pts1_arr[i, 1]
        xp, yp = pts2_arr[i, 0], pts2_arr[i, 1]
        s = float(sw[i])
        A.append([-x * s, -y * s, -s, 0.0, 0.0, 0.0, xp * x * s, xp * y * s, xp * s])
        A.append([0.0, 0.0, 0.0, -x * s, -y * s, -s, yp * x * s, yp * y * s, yp * s])
    try:
        _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
        H = vt[-1].reshape(3, 3)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
            return H if np.all(np.isfinite(H)) else None
    except Exception:
        pass
    return None


def estimate_homography_whitened_dlt(
    pts1: np.ndarray,
    pts2: np.ndarray,
    covariances: Sequence[np.ndarray],
) -> Optional[np.ndarray]:
    """
    2x2 matrix-whitened block DLT solving W_i A_i h = 0 where W_i = Sigma_i^{-1/2}.
    """
    pts1_arr = np.asarray(pts1, dtype=np.float64)
    pts2_arr = np.asarray(pts2, dtype=np.float64)
    n = len(pts1_arr)
    if n < 4 or len(pts2_arr) != n:
        return None
    A_rows = []
    for i in range(n):
        x, y = pts1_arr[i, 0], pts1_arr[i, 1]
        xp, yp = pts2_arr[i, 0], pts2_arr[i, 1]
        cov_i = covariances[i] if i < len(covariances) else None
        W_i = compute_whitening_matrix(cov_i)

        block = np.array([
            [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp],
            [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp],
        ], dtype=np.float64)

        w_block = W_i @ block
        A_rows.append(w_block[0])
        A_rows.append(w_block[1])

    try:
        _, _, vt = np.linalg.svd(np.asarray(A_rows, dtype=np.float64))
        H_dlt = vt[-1].reshape(3, 3)
        if abs(H_dlt[2, 2]) > 1e-12:
            H = H_dlt / H_dlt[2, 2]
            return H if np.all(np.isfinite(H)) else None
    except Exception:
        pass
    return None


def refine_homography_covariance_weighted(
    pts1: np.ndarray,
    pts2: np.ndarray,
    covariances: Sequence[np.ndarray],
    H_init: Optional[np.ndarray] = None,
    max_iters: int = 15,
    return_diagnostics: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Dict[str, Any]]]:
    """
    Refines a 3x3 homography using per-match 2x2 residual covariances:
    1. Whitened DLT algebraic solution.
    2. Non-linear Levenberg-Marquardt refinement minimizing the total Mahalanobis error.
    3. Guarantees objective monotonicity: final_objective <= initial_objective.
    4. Explicitly logs initial whitened DLT and final nonlinear Mahalanobis objectives.
       
    Args:
        pts1: (N, 2) inlier source points.
        pts2: (N, 2) inlier destination points.
        covariances: (N, 2, 2) per-inlier residual covariances.
        H_init: Initial 3x3 homography (from RANSAC consensus). If None, whitened DLT is used.
        max_iters: Maximum Levenberg-Marquardt iterations.
        return_diagnostics: If True, returns (H_refined, diagnostics_dict).
        
    Returns:
        Refined 3x3 homography (or tuple of (H, diagnostics) if return_diagnostics=True).
    """
    t_start = time.perf_counter()
    pts1_arr = np.asarray(pts1, dtype=np.float64)
    pts2_arr = np.asarray(pts2, dtype=np.float64)
    n = len(pts1_arr)

    default_fail_diag = {
        "initial_objective": None,
        "final_objective": None,
        "delta_objective": None,
        "objective_decreased": False,
        "is_lm_result": False,
        "iterations": 0,
        "convergence_status": "insufficient_points",
        "lambda_param": None,
        "runtime_ms": 0.0,
    }

    if n < 4 or len(pts2_arr) != n:
        if return_diagnostics:
            return None, default_fail_diag
        return None

    # Step 1: Initialize H via Whitened DLT if H_init is None or non-finite
    H = None
    if H_init is not None and np.all(np.isfinite(H_init)) and abs(H_init[2, 2]) > 1e-9:
        H = (H_init / H_init[2, 2]).copy()
    else:
        H = estimate_homography_whitened_dlt(pts1_arr, pts2_arr, covariances)

    if H is None or not np.all(np.isfinite(H)):
        default_fail_diag["convergence_status"] = "whitened_dlt_failed"
        if return_diagnostics:
            return None, default_fail_diag
        return None

    H_initial = H.copy()

    # Step 2: Non-linear Levenberg-Marquardt refinement on Mahalanobis residuals
    # Parametrize H with 8 free parameters: p = [h00, h01, h02, h10, h11, h12, h20, h21] with h22 = 1.0
    p = np.array([
        H[0, 0], H[0, 1], H[0, 2],
        H[1, 0], H[1, 1], H[1, 2],
        H[2, 0], H[2, 1],
    ], dtype=np.float64)

    def _param_to_H(params: np.ndarray) -> np.ndarray:
        return np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ], dtype=np.float64)

    def _cost(params: np.ndarray) -> float:
        H_curr = _param_to_H(params)
        total_sq = 0.0
        for i in range(n):
            x, y = pts1_arr[i, 0], pts1_arr[i, 1]
            xp, yp = pts2_arr[i, 0], pts2_arr[i, 1]
            w3 = H_curr[2, 0] * x + H_curr[2, 1] * y + 1.0
            if abs(w3) < 1e-9:
                return 1e12
            u = (H_curr[0, 0] * x + H_curr[0, 1] * y + H_curr[0, 2]) / w3
            v = (H_curr[1, 0] * x + H_curr[1, 1] * y + H_curr[1, 2]) / w3
            r = np.array([u - xp, v - yp], dtype=np.float64)
            W_i = compute_whitening_matrix(covariances[i] if i < len(covariances) else None)
            r_w = W_i @ r
            total_sq += float(r_w @ r_w)
        return total_sq

    # Requirement 1: Log initial whitened-DLT objective
    initial_objective = _cost(p)
    logger.info("Initial whitened-DLT Mahalanobis objective: %.6f (N=%d points)", initial_objective, n)

    current_cost = initial_objective
    lam = 1e-3
    convergence_status = "max_iters_reached"
    iters_taken = 0

    for it in range(max_iters):
        iters_taken += 1
        H_curr = _param_to_H(p)
        J_stack = []
        r_stack = []

        for i in range(n):
            x, y = pts1_arr[i, 0], pts1_arr[i, 1]
            xp, yp = pts2_arr[i, 0], pts2_arr[i, 1]
            w3 = H_curr[2, 0] * x + H_curr[2, 1] * y + 1.0
            if abs(w3) < 1e-9:
                w3 = 1e-9 if w3 >= 0 else -1e-9

            w1 = H_curr[0, 0] * x + H_curr[0, 1] * y + H_curr[0, 2]
            w2 = H_curr[1, 0] * x + H_curr[1, 1] * y + H_curr[1, 2]
            u = w1 / w3
            v = w2 / w3

            # Raw residual
            r = np.array([u - xp, v - yp], dtype=np.float64)
            W_i = compute_whitening_matrix(covariances[i] if i < len(covariances) else None)

            # Jacobian of (u, v) w.r.t [h00..h21] (2x8 matrix)
            J_raw = np.array([
                [x / w3, y / w3, 1.0 / w3, 0.0, 0.0, 0.0, -u * x / w3, -u * y / w3],
                [0.0, 0.0, 0.0, x / w3, y / w3, 1.0 / w3, -v * x / w3, -v * y / w3],
            ], dtype=np.float64)

            J_w = W_i @ J_raw
            r_w = W_i @ r

            J_stack.append(J_w[0])
            J_stack.append(J_w[1])
            r_stack.append(r_w[0])
            r_stack.append(r_w[1])

        J_all = np.asarray(J_stack, dtype=np.float64)
        r_all = np.asarray(r_stack, dtype=np.float64)

        Hess = J_all.T @ J_all
        grad = J_all.T @ r_all

        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-6:
            convergence_status = "converged_gradient"
            break

        # Levenberg-Marquardt step
        Hess_damped = Hess + lam * np.diag(np.diagonal(Hess) + 1e-6)

        try:
            delta = np.linalg.solve(Hess_damped, -grad)
        except np.linalg.LinAlgError:
            convergence_status = "singular_hessian"
            break

        p_candidate = p + delta
        candidate_cost = _cost(p_candidate)

        # Requirement 4: Verify step does not increase objective
        if candidate_cost < current_cost:
            p = p_candidate
            current_cost = candidate_cost
            lam = max(lam * 0.2, 1e-7)
            if np.linalg.norm(delta) < 1e-6:
                convergence_status = "converged_step"
                break
        else:
            lam = min(lam * 10.0, 1e4)

    # Requirement 3: Final transform is the LM result derived from parameter vector p
    H_refined = _param_to_H(p)
    final_objective = _cost(p)

    # Requirement 4: Verification that objective does not increase after refinement
    delta_obj = initial_objective - final_objective
    if final_objective > initial_objective + 1e-9:
        logger.warning(
            "LM objective increased (%.6f -> %.6f); reverting to initial transform.",
            initial_objective, final_objective
        )
        H_refined = H_initial
        final_objective = initial_objective
        delta_obj = 0.0
        objective_decreased = False
        is_lm_result = False
        convergence_status = "reverted_cost_increase"
    else:
        objective_decreased = True
        is_lm_result = True
        if convergence_status == "max_iters_reached" and delta_obj > 0:
            convergence_status = "cost_decreased"

    # Requirement 2: Log final nonlinear Mahalanobis objective
    logger.info(
        "Final nonlinear Mahalanobis objective: %.6f (delta: %.6f, status=%s, iters=%d)",
        final_objective, delta_obj, convergence_status, iters_taken
    )

    t_end = time.perf_counter()
    runtime_ms = (t_end - t_start) * 1000.0

    diagnostics = {
        "initial_objective": round(float(initial_objective), 6),
        "final_objective": round(float(final_objective), 6),
        "delta_objective": round(float(delta_obj), 6),
        "objective_decreased": objective_decreased,
        "is_lm_result": is_lm_result,
        "iterations": iters_taken,
        "convergence_status": convergence_status,
        "lambda_param": float(lam),
        "runtime_ms": round(float(runtime_ms), 3),
    }

    if not np.all(np.isfinite(H_refined)):
        if return_diagnostics:
            return None, diagnostics
        return None

    if return_diagnostics:
        return H_refined, diagnostics
    return H_refined


def audit_homography_covariance_refinement(
    pts1: np.ndarray,
    pts2: np.ndarray,
    covariances: Sequence[np.ndarray],
    checkpoints_src: Optional[np.ndarray] = None,
    checkpoints_dst: Optional[np.ndarray] = None,
    checkpoints_cov: Optional[Sequence[np.ndarray]] = None,
    H_init: Optional[np.ndarray] = None,
    tolerance_degradation_px: float = 0.05,
) -> Dict[str, Any]:
    """
    Performs a comprehensive audit of covariance-weighted homography refinement (Requirements 1-8):
    1. Logs initial whitened-DLT objective.
    2. Logs final nonlinear Mahalanobis objective.
    3. Verifies that the final transform is the LM result.
    4. Verifies the objective does not increase after refinement.
    5. Compares 4 estimation methods:
       - ordinary DLT
       - scalar-weighted DLT
       - whitened DLT
       - final covariance-weighted LM
    6. Evaluates all methods on independent checkpoints.
    7. Reports runtime and convergence status.
    8. Rejects the result if LM fails or degrades independent checkpoint error without explicit REVIEW status.
    """
    pts1_arr = np.asarray(pts1, dtype=np.float64)
    pts2_arr = np.asarray(pts2, dtype=np.float64)
    n_inliers = len(pts1_arr)

    has_checkpoints = (
        checkpoints_src is not None
        and checkpoints_dst is not None
        and len(checkpoints_src) > 0
        and len(checkpoints_src) == len(checkpoints_dst)
    )

    def _eval_model(H_cand: Optional[np.ndarray]) -> Dict[str, Any]:
        if H_cand is None or not np.all(np.isfinite(H_cand)):
            return {
                "homography": None,
                "inlier_rmse_px": None,
                "inlier_mahalanobis_cost": None,
                "checkpoint_rmse_px": None,
                "checkpoint_mahalanobis_rmse": None,
            }
        # Inlier evaluation
        d_inl = compute_dual_rmse_metrics(pts1_arr, pts2_arr, H_cand, covariances_dst=covariances)
        inl_rmse = d_inl.get("unweighted_rmse_px")
        # Inlier Mahalanobis cost sum ||W_i r_i||^2
        raw_m = d_inl.get("mahalanobis_distances", [])
        m_cost = float(np.sum(np.array(raw_m) ** 2)) if raw_m else None

        chk_rmse = None
        chk_m_rmse = None
        if has_checkpoints:
            d_chk = compute_dual_rmse_metrics(
                checkpoints_src, checkpoints_dst, H_cand, covariances_dst=checkpoints_cov
            )
            chk_rmse = d_chk.get("unweighted_rmse_px")
            chk_m_rmse = d_chk.get("covariance_weighted_rmse")

        return {
            "homography": H_cand,
            "inlier_rmse_px": round(float(inl_rmse), 4) if inl_rmse is not None else None,
            "inlier_mahalanobis_cost": round(float(m_cost), 4) if m_cost is not None else None,
            "checkpoint_rmse_px": round(float(chk_rmse), 4) if chk_rmse is not None else None,
            "checkpoint_mahalanobis_rmse": round(float(chk_m_rmse), 4) if chk_m_rmse is not None else None,
        }

    # 1. Ordinary DLT
    t0 = time.perf_counter()
    H_ord = estimate_homography_ordinary_dlt(pts1_arr, pts2_arr)
    t_ord = (time.perf_counter() - t0) * 1000.0
    m_ord = _eval_model(H_ord)
    m_ord["runtime_ms"] = round(float(t_ord), 3)
    m_ord["convergence_status"] = "svd_exact"

    # 2. Scalar-weighted DLT
    t0 = time.perf_counter()
    H_scalar = estimate_homography_scalar_weighted_dlt(pts1_arr, pts2_arr, covariances)
    t_scalar = (time.perf_counter() - t0) * 1000.0
    m_scalar = _eval_model(H_scalar)
    m_scalar["runtime_ms"] = round(float(t_scalar), 3)
    m_scalar["convergence_status"] = "svd_exact"

    # 3. Whitened DLT
    t0 = time.perf_counter()
    H_white = estimate_homography_whitened_dlt(pts1_arr, pts2_arr, covariances)
    t_white = (time.perf_counter() - t0) * 1000.0
    m_white = _eval_model(H_white)
    m_white["runtime_ms"] = round(float(t_white), 3)
    m_white["convergence_status"] = "svd_exact"

    # 4. Covariance-weighted Levenberg-Marquardt
    H_init_use = H_init if H_init is not None else H_white
    H_lm, lm_diag = refine_homography_covariance_weighted(
        pts1_arr, pts2_arr, covariances, H_init=H_init_use, return_diagnostics=True
    )
    m_lm = _eval_model(H_lm)
    m_lm["runtime_ms"] = lm_diag.get("runtime_ms", 0.0)
    m_lm["convergence_status"] = lm_diag.get("convergence_status", "failed")
    m_lm["iterations"] = lm_diag.get("iterations", 0)
    m_lm["lambda_param"] = lm_diag.get("lambda_param")

    methods_dict = {
        "ordinary_dlt": m_ord,
        "scalar_weighted_dlt": m_scalar,
        "whitened_dlt": m_white,
        "covariance_weighted_lm": m_lm,
    }

    # Requirement 8: Reject the result if LM fails or produces worse independent checkpoints without explicit REVIEW
    status = "PASS"
    rejection_reason = None
    accepted_method = "covariance_weighted_lm"
    accepted_transform = H_lm

    if H_lm is None:
        status = "REVIEW"
        rejection_reason = "lm_optimization_failed"
        accepted_transform = H_white if H_white is not None else H_scalar
        accepted_method = "whitened_dlt" if H_white is not None else "scalar_weighted_dlt"
    elif has_checkpoints:
        lm_chk = m_lm.get("checkpoint_rmse_px")
        # Check against best baseline DLT
        dlt_checkpoints = {
            name: methods_dict[name]["checkpoint_rmse_px"]
            for name in ["ordinary_dlt", "scalar_weighted_dlt", "whitened_dlt"]
            if methods_dict[name]["checkpoint_rmse_px"] is not None
        }
        if dlt_checkpoints and lm_chk is not None:
            best_dlt_name = min(dlt_checkpoints, key=dlt_checkpoints.get)
            best_dlt_val = dlt_checkpoints[best_dlt_name]
            if lm_chk > best_dlt_val + tolerance_degradation_px:
                status = "REVIEW"
                rejection_reason = (
                    f"lm_checkpoint_rmse_degraded: LM checkpoint RMSE ({lm_chk:.4f}px) "
                    f"exceeds best DLT ({best_dlt_name}: {best_dlt_val:.4f}px) by >{tolerance_degradation_px:.2f}px"
                )
                accepted_transform = methods_dict[best_dlt_name]["homography"]
                accepted_method = best_dlt_name

    return {
        "status": status,
        "accepted_method": accepted_method,
        "accepted_transform": accepted_transform,
        "rejection_reason": rejection_reason,
        "initial_whitened_dlt_objective": lm_diag.get("initial_objective"),
        "final_nonlinear_objective": lm_diag.get("final_objective"),
        "delta_objective": lm_diag.get("delta_objective"),
        "objective_decreased": lm_diag.get("objective_decreased", False),
        "is_lm_result": lm_diag.get("is_lm_result", False),
        "methods": methods_dict,
        "diagnostics": lm_diag,
    }


def compute_dual_rmse_metrics(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    H: np.ndarray,
    covariances_src: Optional[Sequence[np.ndarray]] = None,
    covariances_dst: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Calculates both unweighted Euclidean RMSE and covariance-weighted Mahalanobis RMSE.
    Does NOT use uncertainty to hide bad residuals; reports raw residuals explicitly.
    
    Args:
        src_pts: (N, 2) source points.
        dst_pts: (N, 2) destination points.
        H: 3x3 homography or affine matrix.
        covariances_src: (N, 2, 2) source localization covariances (optional).
        covariances_dst: (N, 2, 2) destination localization covariances (optional).
        
    Returns:
        Dict containing:
            unweighted_rmse_px: Standard Euclidean RMSE in pixels.
            covariance_weighted_rmse: Mahalanobis whitened RMSE (dimensionless).
            mean_raw_error_px: Mean Euclidean error in pixels.
            median_raw_error_px: Median Euclidean error in pixels.
            max_raw_error_px: Maximum Euclidean error in pixels.
            raw_residuals_px: List of Euclidean error magnitudes per correspondence.
            raw_residual_vectors_px: List of [dx, dy] residual vectors.
            mahalanobis_distances: List of Mahalanobis error magnitudes.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = len(src)

    if n == 0 or H is None or not np.all(np.isfinite(H)):
        return {
            "unweighted_rmse_px": None,
            "covariance_weighted_rmse": None,
            "mean_raw_error_px": None,
            "median_raw_error_px": None,
            "max_raw_error_px": None,
            "raw_residuals_px": [],
            "raw_residual_vectors_px": [],
            "mahalanobis_distances": [],
        }

    # Project source points through H
    ones = np.ones((n, 1), dtype=np.float64)
    src_h = np.hstack([src, ones])
    proj_h = (H @ src_h.T).T

    z = proj_h[:, 2:3]
    eps = 1e-12
    z_safe = np.where(np.abs(z) < eps, np.copysign(eps, z), z)
    proj_2d = proj_h[:, :2] / z_safe

    # Raw residual vectors: r_i = proj_2d_i - dst_i
    raw_vecs = proj_2d - dst
    raw_errors = np.linalg.norm(raw_vecs, axis=1)

    # Compute residual covariances and whitened residuals
    res_covs = []
    for i in range(n):
        cov_s = covariances_src[i] if covariances_src is not None and i < len(covariances_src) else None
        cov_d = covariances_dst[i] if covariances_dst is not None and i < len(covariances_dst) else None
        J_i = compute_homography_jacobian(H, src[i])
        c_r = combine_covariances(cov_s, cov_d, J_T=J_i)
        res_covs.append(c_r)

    whitened, m_dists = whiten_residuals(raw_vecs, res_covs)

    unweighted_rmse = float(np.sqrt(np.mean(raw_errors ** 2)))
    weighted_rmse = float(np.sqrt(np.mean(m_dists ** 2)))

    return {
        "unweighted_rmse_px": round(unweighted_rmse, 4),
        "covariance_weighted_rmse": round(weighted_rmse, 4),
        "mean_raw_error_px": round(float(np.mean(raw_errors)), 4),
        "median_raw_error_px": round(float(np.median(raw_errors)), 4),
        "max_raw_error_px": round(float(np.max(raw_errors)), 4),
        "raw_residuals_px": [round(float(e), 4) for e in raw_errors],
        "raw_residual_vectors_px": [[round(float(v[0]), 4), round(float(v[1]), 4)] for v in raw_vecs],
        "mahalanobis_distances": [round(float(d), 4) for d in m_dists],
    }
