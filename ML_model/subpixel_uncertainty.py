"""
subpixel_uncertainty.py — Local Correlation Peak Uncertainty Estimation for Sub-Pixel Registration

Implements rigorous physical uncertainty quantification for Fourier Phase Correlation
and template matching:
1. 2D paraboloid surface fitting to the local 3x3 correlation peak neighborhood.
2. Direct derivation of the peak Hessian matrix H and negative Hessian (curvature) J = -H.
3. Conversion of curvature into a 2x2 spatial displacement covariance matrix Sigma = s^2 * J^{-1}.
4. Eigen-decomposition of Sigma into 1-sigma major and minor semi-axes (sigma_major_px, sigma_minor_px)
   and orientation angle (ellipse_angle_deg).
5. Flat peak detection via curvature eigenvalues, det(J), and condition number.
6. Multimodal peak detection via distinct secondary local maxima identification.
7. Rejection / downgrading of ambiguous peaks to protect geometric fitting.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import cv2

logger = logging.getLogger("ML_model.subpixel_uncertainty")


@dataclass
class SubpixelUncertaintyResult:
    """Sub-pixel displacement estimate and its localized spatial covariance."""
    dx: float
    dy: float
    peak_val: float
    is_valid: bool
    covariance_xy: List[List[float]]  # 2x2 symmetric positive-definite covariance [[Cxx, Cxy], [Cyx, Cyy]]
    sigma_major_px: float             # 1-sigma semi-major axis in pixels
    sigma_minor_px: float             # 1-sigma semi-minor axis in pixels
    ellipse_angle_deg: float          # Angle of the major axis relative to X-axis in degrees [-180, 180]
    uncertainty_status: str           # "valid", "flat_peak", "multimodal", "prior_only", "failed"
    secondary_peak_ratio: float = 0.0 # R2 / R1
    curvature_condition_number: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def covariance_to_ellipse(
    cov: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Computes 1-sigma ellipse parameters from a 2x2 covariance matrix.

    Returns:
        sigma_major_px: 1-sigma semi-major axis length.
        sigma_minor_px: 1-sigma semi-minor axis length.
        ellipse_angle_deg: Angle of major axis in degrees [-180, 180].
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return 1.0, 1.0, 0.0

    # Ensure symmetry
    c_xx = float(cov[0, 0])
    c_yy = float(cov[1, 1])
    c_xy = float(0.5 * (cov[0, 1] + cov[1, 0]))

    # Analytical eigenvalues of 2x2 symmetric matrix
    delta = math.sqrt(max(0.0, (c_xx - c_yy) ** 2 + 4.0 * (c_xy ** 2)))
    lambda1 = 0.5 * (c_xx + c_yy + delta)
    lambda2 = 0.5 * (c_xx + c_yy - delta)

    # Major and minor axes (1-sigma)
    sigma_major = math.sqrt(max(1e-6, lambda1))
    sigma_minor = math.sqrt(max(1e-6, lambda2))

    # Angle of major eigenvector
    if abs(c_xy) > 1e-9 or abs(c_xx - c_yy) > 1e-9:
        angle_rad = 0.5 * math.atan2(2.0 * c_xy, c_xx - c_yy)
        angle_deg = math.degrees(angle_rad)
    else:
        angle_deg = 0.0

    return round(sigma_major, 4), round(sigma_minor, 4), round(angle_deg, 2)


def estimate_peak_hessian_and_covariance(
    corr: np.ndarray,
    peak_y: int,
    peak_x: int,
    min_peak_val: float = 0.15,
    secondary_peak_radius: int = 3,
    multimodal_ratio_thresh: float = 0.80,
    flat_cond_thresh: float = 40.0,
    flat_curvature_min: float = 0.02,
) -> SubpixelUncertaintyResult:
    """
    Fits a 2D paraboloid z(x,y) = ax^2 + by^2 + c_cross*xy + dx + ey + f to the
    3x3 neighborhood of (peak_y, peak_x) on the correlation surface.

    Derives the negative Hessian (curvature matrix), detects flat and multimodal
    responses, and computes the 2x2 spatial localization covariance matrix.
    """
    h, w = corr.shape[:2]
    c_val = float(corr[peak_y, peak_x])

    if c_val < min_peak_val or peak_y <= 0 or peak_y >= h - 1 or peak_x <= 0 or peak_x >= w - 1:
        # Invalid or boundary peak
        return SubpixelUncertaintyResult(
            dx=0.0,
            dy=0.0,
            peak_val=c_val,
            is_valid=False,
            covariance_xy=[[1.0, 0.0], [0.0, 1.0]],
            sigma_major_px=1.0,
            sigma_minor_px=1.0,
            ellipse_angle_deg=0.0,
            uncertainty_status="failed",
            secondary_peak_ratio=0.0,
            curvature_condition_number=1.0,
        )

    # 1. Multimodal Peak Analysis: Identify distinct secondary local maxima
    # Outside the primary peak basin (radius secondary_peak_radius)
    dilated = cv2.dilate(corr, np.ones((3, 3), dtype=np.uint8))
    local_maxima = (corr == dilated) & (corr > min_peak_val)

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2
    outside_mask = dist_sq > (secondary_peak_radius ** 2)

    sec_candidates = local_maxima & outside_mask
    if np.any(sec_candidates):
        secondary_peak_val = float(np.max(corr[sec_candidates]))
        sec_ratio = max(0.0, secondary_peak_val / max(c_val, 1e-6))
    else:
        secondary_peak_val = 0.0
        sec_ratio = 0.0

    is_multimodal = sec_ratio > multimodal_ratio_thresh

    # 2. Extract 3x3 neighborhood
    c = float(corr[peak_y, peak_x])
    c_l = float(corr[peak_y, peak_x - 1])
    c_r = float(corr[peak_y, peak_x + 1])
    c_u = float(corr[peak_y - 1, peak_x])
    c_d = float(corr[peak_y + 1, peak_x])
    c_ul = float(corr[peak_y - 1, peak_x - 1])
    c_ur = float(corr[peak_y - 1, peak_x + 1])
    c_dl = float(corr[peak_y + 1, peak_x - 1])
    c_dr = float(corr[peak_y + 1, peak_x + 1])

    # 3. 2D Paraboloid Coefficients
    # z(x,y) = a*x^2 + b*y^2 + c_cross*x*y + d*x + e*y + f
    a = 0.5 * (c_r + c_l - 2.0 * c)
    b = 0.5 * (c_d + c_u - 2.0 * c)
    c_cross = 0.25 * (c_dr + c_ul - c_dl - c_ur)
    d = 0.5 * (c_r - c_l)
    e = 0.5 * (c_d - c_u)

    # 4. Negative Hessian J = -H
    # H = [[2a, c_cross], [c_cross, 2b]] -> J = [[-2a, -c_cross], [-c_cross, -2b]]
    j_xx = -2.0 * a
    j_yy = -2.0 * b
    j_xy = -c_cross

    det_j = j_xx * j_yy - j_xy * j_xy
    trace_j = j_xx + j_yy

    # Check concave downward (local maximum): j_xx > 0, j_yy > 0, det_j > 0
    disc = max(0.0, trace_j ** 2 - 4.0 * det_j)
    eig1 = 0.5 * (trace_j + math.sqrt(disc))
    eig2 = 0.5 * (trace_j - math.sqrt(disc))
    min_eig = min(eig1, eig2)
    max_eig = max(eig1, eig2)

    cond_number = max_eig / max(min_eig, 1e-9) if min_eig > 0 else float("inf")
    is_flat = (
        det_j <= 1e-7
        or min_eig < flat_curvature_min
        or j_xx <= 0.0
        or j_yy <= 0.0
        or cond_number > flat_cond_thresh
    )

    # 5. Sub-pixel shift solve
    denom = 4.0 * a * b - c_cross * c_cross
    if abs(denom) > 1e-9:
        dx = float((c_cross * e - 2.0 * b * d) / denom)
        dy = float((c_cross * d - 2.0 * a * e) / denom)
        dx = float(np.clip(dx, -0.9, 0.9))
        dy = float(np.clip(dy, -0.9, 0.9))
    else:
        dx, dy = 0.0, 0.0
        is_flat = True

    # 6. Covariance Computation
    # Inverse curvature scaled by Phase Correlation residual variance
    noise_var = max(0.01, (1.0 - min(c_val, 0.999) ** 2) / (4.0 * max(c_val, 0.1) ** 2))

    if not is_flat and det_j > 1e-7:
        inv_j_xx = j_yy / det_j
        inv_j_yy = j_xx / det_j
        inv_j_xy = -j_xy / det_j

        c_xx = noise_var * inv_j_xx
        c_yy = noise_var * inv_j_yy
        c_xy = noise_var * inv_j_xy

        # Enforce realistic bounds on physical pixel uncertainty
        c_xx = float(np.clip(c_xx, 1e-4, 4.0))
        c_yy = float(np.clip(c_yy, 1e-4, 4.0))
        c_xy = float(np.clip(c_xy, -math.sqrt(c_xx * c_yy) * 0.99, math.sqrt(c_xx * c_yy) * 0.99))
    else:
        # Flat / ill-conditioned peak -> assign conservative large variance
        c_xx, c_yy, c_xy = 2.25, 2.25, 0.0  # 1.5 px 1-sigma

    cov_mat = np.array([[c_xx, c_xy], [c_xy, c_yy]], dtype=np.float64)

    # If multimodal, inflate covariance to reflect ambiguity
    if is_multimodal:
        inflation = 1.0 + 2.0 * (sec_ratio - multimodal_ratio_thresh)
        cov_mat *= (inflation ** 2)

    sigma_maj, sigma_min, angle_deg = covariance_to_ellipse(cov_mat)

    # 7. Uncertainty Status & Validity Gate
    # Hierarchy: Flat peaks are rejected for low curvature; multimodal are rejected for ambiguity
    if is_flat:
        uncertainty_status = "flat_peak"
        is_valid = False
    elif is_multimodal:
        uncertainty_status = "multimodal"
        is_valid = False
    else:
        uncertainty_status = "valid"
        is_valid = True

    return SubpixelUncertaintyResult(
        dx=dx,
        dy=dy,
        peak_val=c_val,
        is_valid=is_valid,
        covariance_xy=[
            [round(float(cov_mat[0, 0]), 6), round(float(cov_mat[0, 1]), 6)],
            [round(float(cov_mat[1, 0]), 6), round(float(cov_mat[1, 1]), 6)],
        ],
        sigma_major_px=sigma_maj,
        sigma_minor_px=sigma_min,
        ellipse_angle_deg=angle_deg,
        uncertainty_status=uncertainty_status,
        secondary_peak_ratio=round(sec_ratio, 4),
        curvature_condition_number=round(cond_number, 2) if math.isfinite(cond_number) else 999.0,
    )


def subpixel_phase_correlation_with_uncertainty(
    patch1: np.ndarray,
    patch2: np.ndarray,
    min_peak_val: float = 0.15,
) -> SubpixelUncertaintyResult:
    """
    Fourier Phase Correlation with rigorous local peak uncertainty estimation.
    Returns full SubpixelUncertaintyResult.
    """
    h, w = patch1.shape[:2]
    if h < 16 or w < 16:
        return build_fallback_uncertainty(0.0, 0.0, 0.0, is_valid=False, status="failed")

    p1 = patch1.astype(np.float32)
    p2 = patch2.astype(np.float32)

    if np.var(p1) < 1e-6 or np.var(p2) < 1e-6:
        return build_fallback_uncertainty(0.0, 0.0, 0.0, is_valid=False, status="failed")

    win_y = np.hanning(h).astype(np.float32)
    win_x = np.hanning(w).astype(np.float32)
    window = np.outer(win_y, win_x)

    p1 = (p1 - float(np.mean(p1))) * window
    p2 = (p2 - float(np.mean(p2))) * window

    F1 = np.fft.fft2(p1)
    F2 = np.fft.fft2(p2)

    denom = np.abs(F2 * np.conj(F1)) + 1e-9
    cross_power = (F2 * np.conj(F1)) / denom
    corr = np.fft.fftshift(np.real(np.fft.ifft2(cross_power)).astype(np.float32))

    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[peak_y, peak_x])

    if peak_val < min_peak_val:
        return build_fallback_uncertainty(0.0, 0.0, peak_val, is_valid=False, status="flat_peak")

    cy, cx = h // 2, w // 2
    res = estimate_peak_hessian_and_covariance(corr, peak_y, peak_x, min_peak_val=min_peak_val)

    # Shift relative to patch center
    shift_x = float(float(peak_x) + res.dx - cx)
    shift_y = float(float(peak_y) + res.dy - cy)

    return SubpixelUncertaintyResult(
        dx=shift_x,
        dy=shift_y,
        peak_val=peak_val,
        is_valid=res.is_valid,
        covariance_xy=res.covariance_xy,
        sigma_major_px=res.sigma_major_px,
        sigma_minor_px=res.sigma_minor_px,
        ellipse_angle_deg=res.ellipse_angle_deg,
        uncertainty_status=res.uncertainty_status,
        secondary_peak_ratio=res.secondary_peak_ratio,
        curvature_condition_number=res.curvature_condition_number,
    )


def build_fallback_uncertainty(
    dx: float = 0.0,
    dy: float = 0.0,
    peak_val: float = 0.5,
    is_valid: bool = False,
    status: str = "prior_only",
    default_sigma: float = 1.0,
) -> SubpixelUncertaintyResult:
    """Builds a standardized conservative uncertainty record when subpixel refinement is not performed."""
    var = default_sigma ** 2
    return SubpixelUncertaintyResult(
        dx=float(dx),
        dy=float(dy),
        peak_val=float(peak_val),
        is_valid=bool(is_valid),
        covariance_xy=[[round(var, 6), 0.0], [0.0, round(var, 6)]],
        sigma_major_px=round(default_sigma, 4),
        sigma_minor_px=round(default_sigma, 4),
        ellipse_angle_deg=0.0,
        uncertainty_status=str(status),
        secondary_peak_ratio=0.0,
        curvature_condition_number=1.0,
    )
