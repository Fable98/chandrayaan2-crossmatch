"""
ML_model/overlap_recovery.py — Content-Based Image Overlap Recovery Pre-Matching

Recovers true physical and pixel image overlap via 1D row/column profile cross-correlation
followed by 2D Fourier Phase Correlation. Used when PDS label-derived bounds (bounds_optical)
are imprecise or misaligned due to orbital ephemeris/attitude jitter in lunar orbiter labels.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

logger = logging.getLogger("ML_model.overlap_recovery")

MOON_RADIUS_METERS = 1737400.0


def _compute_1d_profiles(gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 1D normalized intensity and gradient energy profiles along rows (Y) and columns (X).
    """
    arr = gray_img.astype(np.float32)
    # Subtract local mean to center energy around zero
    norm = arr - float(np.mean(arr))
    std = float(np.std(norm))
    if std > 1e-6:
        norm = norm / std

    # Gradient magnitude to emphasize craters and topographic ridges
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag - float(np.mean(grad_mag))
    g_std = float(np.std(grad_norm))
    if g_std > 1e-6:
        grad_norm = grad_norm / g_std

    # Blended profile: 60% intensity + 40% structural gradient
    blended = 0.6 * norm + 0.4 * grad_norm

    row_profile = np.mean(blended, axis=1)  # (H,)
    col_profile = np.mean(blended, axis=0)  # (W,)
    return row_profile, col_profile


def _correlate_1d(profile_ref: np.ndarray, profile_src: np.ndarray, max_shift: int) -> Tuple[int, float]:
    """
    Finds best integer 1D offset maximizing cross-correlation between source and reference.
    Returns (best_shift, peak_correlation_score).
    """
    n_ref = len(profile_ref)
    n_src = len(profile_src)
    if n_ref == 0 or n_src == 0:
        return 0, 0.0

    # Cross-correlation via FFT
    n_fft = n_ref + n_src - 1
    fft_ref = np.fft.rfft(profile_ref, n_fft)
    fft_src = np.fft.rfft(profile_src[::-1], n_fft)
    corr = np.fft.irfft(fft_ref * fft_src, n_fft)

    # Shift index where offset = 0
    zero_idx = n_src - 1
    min_idx = max(0, zero_idx - max_shift)
    max_idx = min(len(corr), zero_idx + max_shift + 1)

    window = corr[min_idx:max_idx]
    if len(window) == 0:
        return 0, 0.0

    best_local_idx = int(np.argmax(window))
    best_shift = (min_idx + best_local_idx) - zero_idx

    # Normalize correlation peak into approximate [-1, 1]
    denom = np.linalg.norm(profile_ref) * np.linalg.norm(profile_src) + 1e-9
    norm_score = float(window[best_local_idx] / denom)
    return best_shift, norm_score


def recover_content_overlap(
    source_img: np.ndarray,
    ref_img: np.ndarray,
    initial_bounds: Optional[Dict[str, float]] = None,
    gsd_m: Optional[float] = None,
    max_shift_fraction: float = 0.25,
) -> Dict[str, Any]:
    """
    Recovers true image overlap using 1D row/column profile cross-correlation
    refined by 2D Fourier Phase Correlation.

    Args:
        source_img: (H, W) or (H, W, C) source image array.
        ref_img: (H, W) or (H, W, C) reference image array.
        initial_bounds: Optional dict with 'west_lon', 'east_lon', 'south_lat', 'north_lat'.
        gsd_m: Ground sampling distance in meters per pixel.
        max_shift_fraction: Maximum allowed shift as fraction of image size (default 25%).

    Returns:
        Dict with dx_px, dy_px, confidence, initial_bounds, recovered_bounds, overlap_recovered.
    """
    def _to_gray_f32(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        return arr.astype(np.float32)

    src_gray = _to_gray_f32(source_img)
    ref_gray = _to_gray_f32(ref_img)

    h_src, w_src = src_gray.shape[:2]
    h_ref, w_ref = ref_gray.shape[:2]

    # Resize to common dimensions if canvas shapes differ
    target_h = min(h_src, h_ref)
    target_w = min(w_src, w_ref)

    if (h_src, w_src) != (target_h, target_w):
        src_work = cv2.resize(src_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        src_work = src_gray

    if (h_ref, w_ref) != (target_h, target_w):
        ref_work = cv2.resize(ref_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        ref_work = ref_gray

    max_shift_x = int(target_w * max_shift_fraction)
    max_shift_y = int(target_h * max_shift_fraction)

    # 1. 1D Profile Cross-Correlation
    src_rows, src_cols = _compute_1d_profiles(src_work)
    ref_rows, ref_cols = _compute_1d_profiles(ref_work)

    shift_y_1d, score_y = _correlate_1d(ref_rows, src_rows, max_shift_y)
    shift_x_1d, score_x = _correlate_1d(ref_cols, src_cols, max_shift_x)
    profile_confidence = float(np.clip((score_x + score_y) / 2.0, 0.0, 1.0))

    # 2. 2D Fourier Phase Correlation refinement
    win_h, win_w = target_h, target_w
    hanning = cv2.createHanningWindow((win_w, win_h), cv2.CV_32F)
    try:
        (phase_dx, phase_dy), phase_response = cv2.phaseCorrelate(src_work, ref_work, hanning)
    except Exception as e:
        logger.warning("2D Phase correlation failed during overlap recovery: %s", e)
        phase_dx, phase_dy, phase_response = 0.0, 0.0, 0.0

    # 3. Decision & Fusion:
    # If phase correlation yields a sensible shift within max boundaries with high response, prefer it;
    # otherwise fall back to 1D profile correlation.
    if abs(phase_dx) <= max_shift_x and abs(phase_dy) <= max_shift_y and phase_response > 0.05:
        final_dx = float(phase_dx)
        final_dy = float(phase_dy)
        confidence = float(np.clip(phase_response * 2.0, 0.0, 1.0))
        method = "2d_phase_correlation"
    elif profile_confidence > 0.10:
        final_dx = float(shift_x_1d)
        final_dy = float(shift_y_1d)
        confidence = profile_confidence
        method = "1d_profile_cross_correlation"
    else:
        final_dx = 0.0
        final_dy = 0.0
        confidence = 0.0
        method = "none_fallback_zero"

    # Clamp shifts to safeguard against wild divergence
    final_dx = float(np.clip(final_dx, -max_shift_x, max_shift_x))
    final_dy = float(np.clip(final_dy, -max_shift_y, max_shift_y))

    overlap_recovered = bool(confidence >= 0.05 and (abs(final_dx) > 0.1 or abs(final_dy) > 0.1))

    # 4. Bounds Adjustment (if initial_bounds provided)
    recovered_bounds = None
    bounds_shift_meters = None
    if initial_bounds is not None:
        recovered_bounds = dict(initial_bounds)
        if gsd_m is not None and gsd_m > 0 and overlap_recovered:
            dx_m = final_dx * float(gsd_m)
            dy_m = final_dy * float(gsd_m)
            bounds_shift_meters = {"dx_m": round(dx_m, 2), "dy_m": round(dy_m, 2)}

            mid_lat = (initial_bounds["south_lat"] + initial_bounds["north_lat"]) / 2.0
            lat_rad = np.radians(mid_lat)
            m_per_deg_lat = (np.pi * MOON_RADIUS_METERS) / 180.0
            m_per_deg_lon = m_per_deg_lat * max(0.01, float(np.cos(lat_rad)))

            delta_lon = float(dx_m / m_per_deg_lon)
            # In image space, positive dy is down (southward), so latitude shifts opposite
            delta_lat = float(-dy_m / m_per_deg_lat)

            recovered_bounds["west_lon"] = round(initial_bounds["west_lon"] + delta_lon, 6)
            recovered_bounds["east_lon"] = round(initial_bounds["east_lon"] + delta_lon, 6)
            recovered_bounds["south_lat"] = round(initial_bounds["south_lat"] + delta_lat, 6)
            recovered_bounds["north_lat"] = round(initial_bounds["north_lat"] + delta_lat, 6)

    return {
        "dx_px": round(final_dx, 4),
        "dy_px": round(final_dy, 4),
        "confidence": round(confidence, 4),
        "method": method,
        "overlap_recovered": overlap_recovered,
        "initial_bounds": initial_bounds,
        "recovered_bounds": recovered_bounds if recovered_bounds else initial_bounds,
        "bounds_shift_meters": bounds_shift_meters,
    }
