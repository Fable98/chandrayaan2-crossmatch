"""
tests/test_jacobian_audit.py — Finite-Difference Jacobian Audit for Coordinate Transformations.

Requirements:
1. Compute analytic source-coordinate Jacobians for affine and homography transforms.
2. Compute numerical finite-difference Jacobians (2nd and 4th order central differences).
3. Compare them across:
   - identity,
   - translation,
   - scale,
   - rotation,
   - affine shear,
   - projective perspective,
   - points near the image boundary (corners, edges).
4. Test both source-to-destination and destination-to-source conventions.
5. Fail if relative or absolute error exceeds configured tolerance (rtol=1e-5, atol=1e-6).
6. Verify Inverse Function Theorem consistency (J_src_to_dst * J_dst_to_src == I_2).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from covariance_geometry import (
    compute_homography_jacobian,
    compute_numerical_jacobian,
    compute_transformation_jacobian,
)

# Configured tolerance thresholds
RTOL = 1e-5
ATOL = 1e-6


def _forward_project(H: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a callable f(x) = H * x mapping 2D point to 2D projected point."""
    H_mat = np.asarray(H, dtype=np.float64)

    def _proj(pt: np.ndarray) -> np.ndarray:
        x, y = float(pt[0]), float(pt[1])
        w1 = H_mat[0, 0] * x + H_mat[0, 1] * y + H_mat[0, 2]
        w2 = H_mat[1, 0] * x + H_mat[1, 1] * y + H_mat[1, 2]
        w3 = H_mat[2, 0] * x + H_mat[2, 1] * y + H_mat[2, 2]
        if abs(w3) < 1e-12:
            w3 = 1e-12 if w3 >= 0 else -1e-12
        return np.array([w1 / w3, w2 / w3], dtype=np.float64)

    return _proj


def _assert_jacobians_match(
    J_analytic: np.ndarray,
    J_numeric: np.ndarray,
    label: str,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    """Verifies that analytic and numerical Jacobians match within configured tolerance."""
    assert J_analytic.shape == (2, 2), f"{label}: J_analytic shape is {J_analytic.shape}"
    assert J_numeric.shape == (2, 2), f"{label}: J_numeric shape is {J_numeric.shape}"
    assert np.all(np.isfinite(J_analytic)), f"{label}: J_analytic has non-finite values"
    assert np.all(np.isfinite(J_numeric)), f"{label}: J_numeric has non-finite values"

    abs_err = np.abs(J_analytic - J_numeric)
    max_abs = float(np.max(abs_err))

    denom = np.maximum(np.abs(J_analytic), 1e-6)
    rel_err = abs_err / denom
    max_rel = float(np.max(rel_err))

    # Must satisfy either absolute or relative tolerance
    passed = np.all((abs_err <= atol) | (rel_err <= rtol))
    assert passed, (
        f"{label} Jacobian mismatch:\n"
        f"Analytic:\n{J_analytic}\n"
        f"Numeric:\n{J_numeric}\n"
        f"Max absolute error: {max_abs:.2e} (tol={atol})\n"
        f"Max relative error: {max_rel:.2e} (tol={rtol})"
    )


# ---------------------------------------------------------------------------
# Test Transformation Matrices
# ---------------------------------------------------------------------------

TRANSFORMS: Dict[str, np.ndarray] = {
    # 1. Identity
    "identity": np.eye(3, dtype=np.float64),

    # 2. Pure Translation
    "translation_pos": np.array([
        [1.0, 0.0, 25.4],
        [0.0, 1.0, 18.7],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),
    "translation_neg": np.array([
        [1.0, 0.0, -32.8],
        [0.0, 1.0, -44.1],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),

    # 3. Scale (isotropic and anisotropic)
    "scale_isotropic": np.array([
        [1.25, 0.0, 0.0],
        [0.0, 1.25, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),
    "scale_anisotropic": np.array([
        [1.40, 0.0, 10.0],
        [0.0, 0.75, -15.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),

    # 4. Rotation
    "rotation_30deg": np.array([
        [np.cos(np.radians(30)), -np.sin(np.radians(30)), 15.0],
        [np.sin(np.radians(30)), np.cos(np.radians(30)), -20.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),
    "rotation_160deg": np.array([
        [np.cos(np.radians(160)), -np.sin(np.radians(160)), 50.0],
        [np.sin(np.radians(160)), np.cos(np.radians(160)), 30.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),

    # 5. Affine Shear
    "affine_shear": np.array([
        [1.1, 0.35, 12.0],
        [-0.2, 0.95, -8.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),

    # 6. Projective Perspective (moderate and strong tilt)
    "perspective_mild": np.array([
        [1.05, 0.02, 10.0],
        [-0.01, 0.98, 15.0],
        [0.0001, -0.00015, 1.0],
    ], dtype=np.float64),
    "perspective_strong": np.array([
        [1.15, -0.08, 45.0],
        [0.06, 1.12, -30.0],
        [0.0004, 0.0003, 1.0],
    ], dtype=np.float64),
}

# Test Evaluation Points
TEST_POINTS: List[Tuple[str, np.ndarray]] = [
    ("interior_center", np.array([256.0, 256.0])),
    ("interior_quadrant1", np.array([120.0, 80.0])),
    ("interior_quadrant4", np.array([400.0, 420.0])),
    # Points near image boundaries (Requirement 3: points near image boundary)
    ("boundary_top_left_corner", np.array([0.5, 0.5])),
    ("boundary_top_right_corner", np.array([511.5, 0.5])),
    ("boundary_bottom_left_corner", np.array([0.5, 511.5])),
    ("boundary_bottom_right_corner", np.array([511.5, 511.5])),
    ("boundary_top_edge", np.array([256.0, 0.2])),
    ("boundary_bottom_edge", np.array([256.0, 511.8])),
    ("boundary_left_edge", np.array([0.2, 256.0])),
    ("boundary_right_edge", np.array([511.8, 256.0])),
]


@pytest.mark.parametrize("transform_name", list(TRANSFORMS.keys()))
@pytest.mark.parametrize("point_name,pt", TEST_POINTS)
def test_forward_jacobian_finite_difference_audit(transform_name: str, point_name: str, pt: np.ndarray):
    """
    Requirement 1, 2, 3, 5:
    Compare analytic and numerical finite-difference Jacobians for forward
    source-to-destination mapping across identity, translation, scale, rotation,
    shear, perspective, and boundary points.
    """
    H = TRANSFORMS[transform_name]
    proj_fn = _forward_project(H)

    # 1. Analytic source-coordinate Jacobian
    J_analytic = compute_transformation_jacobian(H, pt, convention="src_to_dst")

    # 2. Numerical finite-difference Jacobian (4th-order central difference)
    J_numeric = compute_numerical_jacobian(proj_fn, pt, step=1e-5, order=4)

    # 3. Compare within configured tolerance
    label = f"[src_to_dst | {transform_name} @ {point_name}]"
    _assert_jacobians_match(J_analytic, J_numeric, label=label, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("transform_name", list(TRANSFORMS.keys()))
@pytest.mark.parametrize("point_name,pt", TEST_POINTS)
def test_inverse_jacobian_finite_difference_audit(transform_name: str, point_name: str, pt: np.ndarray):
    """
    Requirement 4, 5:
    Compare analytic and numerical finite-difference Jacobians for inverse
    destination-to-source convention across all transformation modes.
    """
    H = TRANSFORMS[transform_name]
    H_inv = np.linalg.inv(H)
    inv_proj_fn = _forward_project(H_inv)

    # Evaluate at point in destination space
    pt_dst = _forward_project(H)(pt)

    # 1. Analytic destination-to-source Jacobian
    J_analytic = compute_transformation_jacobian(H, pt_dst, convention="dst_to_src")

    # 2. Numerical finite-difference Jacobian of the inverse mapping
    J_numeric = compute_numerical_jacobian(inv_proj_fn, pt_dst, step=1e-5, order=4)

    # 3. Compare within configured tolerance
    label = f"[dst_to_src | {transform_name} @ {point_name}]"
    _assert_jacobians_match(J_analytic, J_numeric, label=label, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("transform_name", list(TRANSFORMS.keys()))
@pytest.mark.parametrize("point_name,pt", TEST_POINTS)
def test_inverse_function_theorem_duality(transform_name: str, point_name: str, pt: np.ndarray):
    """
    Requirement 4:
    Verifies that the forward and inverse Jacobians are exact matrix inverses:
        J_src_to_dst(x) @ J_dst_to_src(x') == I_2
    """
    H = TRANSFORMS[transform_name]
    pt_dst = _forward_project(H)(pt)

    J_fwd = compute_transformation_jacobian(H, pt, convention="src_to_dst")
    J_inv = compute_transformation_jacobian(H, pt_dst, convention="dst_to_src")

    product = J_fwd @ J_inv
    np.testing.assert_allclose(
        product, np.eye(2),
        rtol=1e-5, atol=1e-6,
        err_msg=f"Inverse Function Theorem duality failed for {transform_name} @ {point_name}",
    )


def test_failure_on_exceeded_tolerance():
    """Requirement 5: Fail if relative or absolute error exceeds configured tolerance."""
    H = TRANSFORMS["perspective_mild"]
    pt = np.array([100.0, 100.0])
    J_analytic = compute_transformation_jacobian(H, pt, convention="src_to_dst")

    # Artificially corrupted Jacobian
    J_corrupted = J_analytic.copy()
    J_corrupted[0, 1] += 0.05

    with pytest.raises(AssertionError, match="Jacobian mismatch"):
        _assert_jacobians_match(J_analytic, J_corrupted, label="corrupted_test", rtol=1e-5, atol=1e-6)
