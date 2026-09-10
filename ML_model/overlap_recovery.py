"""
ML_model/overlap_recovery.py — Content-Based Image Overlap Recovery Pre-Matching

Recovers true physical and pixel image overlap via scale-first estimation
followed by 1D row/column profile cross-correlation and 2D Fourier Phase
Correlation. Used when PDS label-derived bounds (bounds_optical) are
imprecise or misaligned due to orbital ephemeris/attitude jitter in lunar
orbiter labels.

Pipeline order (scale FIRST, then translation):
  1. Relative scale ratio S is estimated with a translation-invariant
     Fourier-Mellin log-polar estimator on ratio-preserving thumbnails.
     Neither input is resized to min(H, W) — that operation erases the very
     scale gap being measured and injects up-to-20x systematic error on
     OHRC<->TMC-2 style pairs.
  2. The smaller-scale canvas is resampled to the common scale; translation
     is then measured on the scale-normalized pair (1D profiles at native
     lengths, 2D phase correlation on zero-padded — never resized —
     canvases) and mapped back to reference pixels.

Documented capability caps:
  * SCALE_CAP (default 10.0x): log-polar scale estimates are reliable in
    roughly 1-10x. Raw estimates above the cap are clamped and reported via
    ``scale_capped=True``; translation then proceeds scale-uncertain with
    reduced confidence. Gaps around ~20x (OHRC<->TMC-2 natives) MUST go
    through the metadata-GSD common-scale path in matcher_cfog first — this
    module alone cannot bridge them reliably.
  * SHIFT_CAP: translation is clamped to ``max_shift_fraction`` of the
    reference dims (default 25%); larger true offsets are reported clamped
    with ``shift_capped=True``.
  * MIN_DIM: canvases below ~32 px carry no stable correlation cue; the
    module returns overlap_recovered=False instead of a hallucinated shift.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

logger = logging.getLogger("ML_model.overlap_recovery")

MOON_RADIUS_METERS = 1737400.0

# Documented capability caps (see module docstring).
SCALE_CAP = 10.0
SCALE_TOLERANCE = 0.15
MIN_WORKING_DIM = 32
LP_MAX_DIM = 256
LP_RESPONSE_THRESHOLD = 0.03


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
    Handles differing profile lengths natively (no resizing). Returns (best_shift, peak_correlation_score).
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


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    return arr.astype(np.float32)


def estimate_scale_ratio_logpolar(
    source_img: np.ndarray,
    ref_img: np.ndarray,
    lp_max_dim: int = LP_MAX_DIM,
    response_threshold: float = LP_RESPONSE_THRESHOLD,
) -> Tuple[Optional[float], float]:
    """Translation-invariant relative scale estimate (Fourier-Mellin).

    Both inputs are downsampled by the SAME ratio-preserving factor (the
    scale gap survives); the 2D Fourier magnitude spectra are log-polar
    warped about DC and phase-correlated. Returns (S, response) with S >= 1
    the magnitude of the scale gap (larger-canvas / smaller-canvas pixel
    count direction is resolved by the caller), or (None, response) when the
    correlation is too weak to trust.
    """
    try:
        src = _to_gray_f32(source_img)
        ref = _to_gray_f32(ref_img)
        h1, w1 = src.shape[:2]
        h2, w2 = ref.shape[:2]
        if min(h1, w1, h2, w2) < 16:
            return None, 0.0
        # Ratio-preserving joint downsample (SAME factor keeps S intact).
        down = max(1.0, float(max(h1, w1, h2, w2)) / float(lp_max_dim))
        d1 = cv2.resize(src, (max(8, int(round(w1 / down))), max(8, int(round(h1 / down)))),
                        interpolation=cv2.INTER_AREA)
        d2 = cv2.resize(ref, (max(8, int(round(w2 / down))), max(8, int(round(h2 / down)))),
                        interpolation=cv2.INTER_AREA)
        if float(np.std(d1)) < 1e-9 or float(np.std(d2)) < 1e-9:
            return None, 0.0
        n = max(d1.shape[0], d1.shape[1], d2.shape[0], d2.shape[1])
        if n % 2 == 1:
            n += 1

        def _pad(a: np.ndarray) -> np.ndarray:
            h, w = a.shape[:2]
            return cv2.copyMakeBorder(a, (n - h) // 2, n - h - (n - h) // 2,
                                      (n - w) // 2, n - w - (n - w) // 2,
                                      cv2.BORDER_CONSTANT, value=float(np.mean(a)))

        p1, p2 = _pad(d1.astype(np.float64)), _pad(d2.astype(np.float64))
        win = cv2.createHanningWindow((n, n), cv2.CV_64F)
        mag1 = np.abs(np.fft.fftshift(np.fft.fft2((p1 - p1.mean()) * win)))
        mag2 = np.abs(np.fft.fftshift(np.fft.fft2((p2 - p2.mean()) * win)))
        mag1 = np.log1p(np.maximum(0.0, cv2.GaussianBlur(mag1, (31, 31), 5.0) - cv2.GaussianBlur(mag1, (3, 3), 1.0)))
        mag2 = np.log1p(np.maximum(0.0, cv2.GaussianBlur(mag2, (31, 31), 5.0) - cv2.GaussianBlur(mag2, (3, 3), 1.0)))
        cx = cy = float(n) / 2.0
        rmax = float(n) / 2.0
        m_gain = float(n) / float(math.log(max(rmax, 2.0)))
        lp1 = cv2.warpPolar(mag1.astype(np.float32), (n, n), (cx, cy), rmax,
                            cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG)
        lp2 = cv2.warpPolar(mag2.astype(np.float32), (n, n), (cx, cy), rmax,
                            cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG)
        lp1 = lp1.astype(np.float64) - float(np.mean(lp1))
        lp2 = lp2.astype(np.float64) - float(np.mean(lp2))
        (dx, _dy), response = cv2.phaseCorrelate(lp1, lp2, win)
        if not np.isfinite(dx) or float(response) < float(response_threshold):
            return None, float(response) if np.isfinite(response) else 0.0
        s_ratio = float(math.exp(abs(float(dx)) / m_gain))
        if not np.isfinite(s_ratio) or s_ratio < 1.0:
            return None, float(response)
        return s_ratio, float(response)
    except Exception as exc:
        logger.warning("Log-polar scale estimation failed: %s", exc)
        return None, 0.0


def recover_content_overlap(
    source_img: np.ndarray,
    ref_img: np.ndarray,
    initial_bounds: Optional[Dict[str, float]] = None,
    gsd_m: Optional[float] = None,
    max_shift_fraction: float = 0.25,
    scale_ratio: Optional[float] = None,
    estimate_scale: bool = True,
    max_scale_ratio: float = SCALE_CAP,
) -> Dict[str, Any]:
    """
    Recovers true image overlap: scale estimation FIRST, then translation.

    Neither input is resized to min(H, W); scale is estimated on
    ratio-preserving thumbnails and the smaller-scale canvas is resampled to
    the common scale before translation measurement.

    Args:
        source_img: (H, W) or (H, W, C) source image array.
        ref_img: (H, W) or (H, W, C) reference image array.
        initial_bounds: Optional dict with 'west_lon', 'east_lon', 'south_lat', 'north_lat'.
        gsd_m: Ground sampling distance in meters per pixel (reference frame).
        max_shift_fraction: Maximum allowed shift as fraction of reference
            size (translation cap; default 25%).
        scale_ratio: Known relative scale (larger-canvas px / smaller-canvas
            px for the same footprint). When given and finite, it is used
            directly and no estimation runs.
        estimate_scale: When False, skip estimation (translation-only on
            zero-padded native canvases).
        max_scale_ratio: Documented scale cap (default 10.0x). Raw estimates
            above it are clamped with scale_capped=True.

    Returns:
        Dict with dx_px, dy_px (reference pixels), confidence,
        scale_ratio, scale_confidence, scale_capped, shift_capped,
        frame="reference_pixels", method, overlap_recovered, bounds fields.
    """
    src_gray = _to_gray_f32(source_img)
    ref_gray = _to_gray_f32(ref_img)

    h_src, w_src = src_gray.shape[:2]
    h_ref, w_ref = ref_gray.shape[:2]

    if min(h_src, w_src, h_ref, w_ref) < 8:
        return {
            "dx_px": 0.0, "dy_px": 0.0, "confidence": 0.0,
            "scale_ratio": 1.0, "scale_confidence": 0.0, "scale_capped": False,
            "shift_capped": False, "frame": "reference_pixels",
            "method": "none_canvas_too_small", "overlap_recovered": False,
            "initial_bounds": initial_bounds,
            "recovered_bounds": initial_bounds, "bounds_shift_meters": None,
        }

    # ---- Step 1: scale FIRST (never min(H, W) resize) ----
    scale_capped = False
    scale_confidence = 0.0
    s_est: Optional[float] = None
    if scale_ratio is not None and np.isfinite(float(scale_ratio)) and float(scale_ratio) >= 1.0:
        s_est = float(scale_ratio)
        scale_confidence = 1.0
    elif estimate_scale:
        s_raw, resp = estimate_scale_ratio_logpolar(src_gray, ref_gray)
        scale_confidence = round(float(resp), 4)
        if s_raw is not None:
            if s_raw > float(max_scale_ratio):
                s_est = float(max_scale_ratio)
                scale_capped = True
                logger.warning(
                    "Scale estimate %.2fx exceeds cap %.1fx; clamped (translation proceeds scale-uncertain).",
                    s_raw, float(max_scale_ratio),
                )
            else:
                s_est = float(s_raw)
    if s_est is None:
        s_est = 1.0

    src_px = h_src * w_src
    ref_px = h_ref * w_ref
    tiny_side = min(h_src, w_src, h_ref, w_ref) < MIN_WORKING_DIM

    def _phase_on_pair(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
        """Windowed, mean-centered 2D phase correlation on equal canvases."""
        try:
            ah, aw = a.shape[:2]
            hann = cv2.createHanningWindow((aw, ah), cv2.CV_64F)
            aa = (a.astype(np.float64) - float(np.mean(a))) * hann
            bb = (b.astype(np.float64) - float(np.mean(b))) * hann
            (pdx, pdy), resp = cv2.phaseCorrelate(aa, bb, hann)
            if not (np.isfinite(pdx) and np.isfinite(pdy) and np.isfinite(resp)):
                return 0.0, 0.0, 0.0
            return float(pdx), float(pdy), float(resp)
        except Exception as exc:
            logger.warning("2D Phase correlation failed during overlap recovery: %s", exc)
            return 0.0, 0.0, 0.0

    def _pad_to(img: np.ndarray, th: int, tw: int) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.copyMakeBorder(img, 0, max(0, th - h), 0, max(0, tw - w),
                                  cv2.BORDER_CONSTANT, value=float(np.mean(img)))

    def _resample(img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        if (h, w) == (th, tw):
            return img
        if th * tw <= h * w:
            return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_CUBIC)

    # ---- Step 2: translation by evidence-based model selection ----
    # Hypothesis A (same scale): zero-pad to a common canvas (scale-preserving).
    common_h, common_w = max(h_src, h_ref), max(w_src, w_ref)
    A_src, A_ref = _pad_to(src_gray, common_h, common_w), _pad_to(ref_gray, common_h, common_w)
    A_dx, A_dy, A_resp = _phase_on_pair(A_src, A_ref)

    # Hypothesis B (scale gap): resample the coarser canvas to a common scale.
    # The gap magnitude comes from metadata when provided, else the log-polar
    # estimate, else the canvas-size ratio under an explicit same-footprint
    # assumption (flagged; the metadata-GSD path stays authoritative).
    B_applicable = False
    B_dx, B_dy, B_resp = 0.0, 0.0, 0.0
    B_s = 1.0
    scale_note = "translation_only"
    scale_assumed_from_canvas = False
    explicit_scale = (scale_ratio is not None and np.isfinite(float(scale_ratio))
                      and float(scale_ratio) >= 1.0)
    if explicit_scale:
        B_s = float(scale_ratio)  # type: ignore[arg-type]
        B_applicable = B_s > 1.0 + SCALE_TOLERANCE
    elif estimate_scale and s_est > 1.0 + SCALE_TOLERANCE and scale_confidence >= 0.10:
        B_s, B_applicable = float(s_est), True
    elif not tiny_side:
        lin_ratio = math.sqrt(max(src_px, ref_px) / max(min(src_px, ref_px), 1))
        if lin_ratio > 1.0 + SCALE_TOLERANCE:
            B_s, B_applicable = float(lin_ratio), True
            scale_assumed_from_canvas = True
    if B_s > float(max_scale_ratio):
        B_s = float(max_scale_ratio)
        scale_capped = True
        logger.warning(
            "Scale gap %.2fx exceeds cap %.1fx; clamped (translation proceeds scale-uncertain).",
            float(s_est if not explicit_scale else scale_ratio), float(max_scale_ratio),
        )
    B_fx_ref = 1.0
    if B_applicable:
        # Compare at the COARSER scale: downsampling the finer canvas with
        # area averaging matches frequency content (upscaling the blurry side
        # instead yields confident-but-wrong peaks). Shift is measured in
        # coarse pixels and mapped back to reference pixels below.
        try:
            if src_px >= ref_px:
                tw, th = max(8, int(round(w_src / B_s))), max(8, int(round(h_src / B_s)))
                B_src = _resample(src_gray, tw, th)
                B_ref = ref_gray
            else:
                tw, th = max(8, int(round(w_ref / B_s))), max(8, int(round(h_ref / B_s)))
                B_ref = _resample(ref_gray, tw, th)
                B_src = src_gray
            ch, cw = max(B_src.shape[0], B_ref.shape[0]), max(B_src.shape[1], B_ref.shape[1])
            B_dx, B_dy, B_resp = _phase_on_pair(_pad_to(B_src, ch, cw), _pad_to(B_ref, ch, cw))
            B_fx_ref = B_ref.shape[1] / float(w_ref)
            scale_note = "compared_at_coarse_scale"
        except Exception as exc:
            logger.warning("Scaled hypothesis failed (%s); using translation-only.", exc)
            B_applicable = False

    # 1D profiles at native lengths (size-robust, no resizing) — fallback cue.
    src_rows, src_cols = _compute_1d_profiles(src_gray)
    ref_rows, ref_cols = _compute_1d_profiles(ref_gray)
    shift_y_1d, score_y = _correlate_1d(ref_rows, src_rows, int(h_ref * max_shift_fraction))
    shift_x_1d, score_x = _correlate_1d(ref_cols, src_cols, int(w_ref * max_shift_fraction))
    profile_confidence = float(np.clip((score_x + score_y) / 2.0, 0.0, 1.0))

    max_shift_x = int(w_ref * max_shift_fraction)
    max_shift_y = int(h_ref * max_shift_fraction)

    def _fuse(pdx: float, pdy: float, presp: float,
              cap_cx: int, cap_cy: int) -> Tuple[float, float, float, str]:
        sane = (abs(pdx) <= cap_cx and abs(pdy) <= cap_cy and presp > 0.05)
        if sane:
            return pdx, pdy, float(np.clip(presp * 2.0, 0.0, 1.0)), "2d_phase_correlation"
        if profile_confidence > 0.10:
            return float(shift_x_1d), float(shift_y_1d), profile_confidence, "1d_profile_cross_correlation"
        return 0.0, 0.0, 0.0, "none_fallback_zero"

    # Winner by evidence; ties and weak margins keep translation-only (A).
    A_res = _fuse(A_dx, A_dy, A_resp, int(common_w * max_shift_fraction), int(common_h * max_shift_fraction))
    hypotheses = [("A", 1.0, 1.0, A_res)]
    if B_applicable:
        B_cap = (int(B_src.shape[1] * max_shift_fraction), int(B_src.shape[0] * max_shift_fraction))
        hypotheses.append(("B", B_s, B_fx_ref, _fuse(B_dx, B_dy, B_resp, B_cap[0], B_cap[1])))
    best = hypotheses[0]
    for hyp in hypotheses[1:]:
        if hyp[3][2] > best[3][2] + 0.05:
            best = hyp
    _tag, S_win, fx_ref_win, (dx_c, dy_c, confidence, method) = best

    # ---- Conjugate-peak guard: phase correlation can lock onto the negated
    # shift when padding edges dominate. Verify {d, -d, 0} by masked NCC in
    # the winner's frame and keep the best-supported shift. ----
    try:
        if _tag == "A":
            _v_src, _v_ref = A_src, A_ref
            _vs = np.zeros_like(_v_src, dtype=np.float32); _vs[:h_src, :w_src] = 1.0
            _vr = np.zeros_like(_v_ref, dtype=np.float32); _vr[:h_ref, :w_ref] = 1.0
        else:
            _v_src, _v_ref = B_src, B_ref
            _sh, _sw = _v_src.shape[:2]
            _rh, _rw = _v_ref.shape[:2]
            _vs = np.zeros((_sh, _sw), dtype=np.float32)
            _vs[:min(_sh, B_src.shape[0]), :min(_sw, B_src.shape[1])] = 1.0
            _vr = np.zeros((_sh, _sw), dtype=np.float32)
            _vr[:min(_sh, _rh), :min(_sw, _rw)] = 1.0

        def _masked_ncc(dxx: float, dyy: float) -> float:
            M = np.float32([[1, 0, dxx], [0, 1, dyy]])
            h, w = _v_ref.shape[:2]
            warped = cv2.warpAffine(_v_src, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=float(np.mean(_v_src)))
            wvalid = cv2.warpAffine(_vs, M, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            ov = (wvalid > 0.5) & (_vr > 0.5)
            if int(np.sum(ov)) < max(64, int(0.10 * max(1, int(np.sum(_vr > 0.5))))):
                return float("-inf")
            a = warped[ov].astype(np.float64)
            b = _v_ref[ov].astype(np.float64)
            a -= a.mean()
            b -= b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom < 1e-9 or float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
                return float("-inf")
            return float(np.dot(a, b) / denom)

        _cands = [(dx_c, dy_c), (-dx_c, -dy_c), (0.0, 0.0)]
        _scores = [_masked_ncc(_x, _y) for _x, _y in _cands]
        _bi = int(np.argmax(_scores))
        if _scores[_bi] == float("-inf"):
            dx_c, dy_c, confidence, method = 0.0, 0.0, 0.0, "none_fallback_zero"
        elif _bi == 1:
            dx_c, dy_c = -dx_c, -dy_c
            method = f"{method}_sign_corrected"
    except Exception as exc:
        logger.warning("Shift verification skipped (%s).", exc)
    if _tag == "B":
        s_est = float(S_win)
        scale_note = "compared_at_coarse_scale"
    else:
        fx_ref_win = 1.0
    if scale_assumed_from_canvas and _tag == "B":
        method = f"{method}_scale_assumed_from_canvas"
        confidence = float(confidence * 0.5)
    if scale_capped and method != "none_fallback_zero":
        method = f"{method}_scale_capped"
        confidence = float(confidence * 0.5)

    final_dx = dx_c / max(fx_ref_win, 1e-9)
    final_dy = dy_c / max(fx_ref_win, 1e-9)

    # Clamp shifts to the documented translation cap.
    shift_capped = False
    if abs(final_dx) > max_shift_x or abs(final_dy) > max_shift_y:
        shift_capped = True
    final_dx = float(np.clip(final_dx, -max_shift_x, max_shift_x))
    final_dy = float(np.clip(final_dy, -max_shift_y, max_shift_y))

    # Tiny canvases (<32 px) cannot support correlation claims on large gaps:
    # demand strong evidence instead of a hallucinated shift.
    _min_conf = 0.30 if (tiny_side and (scale_capped or float(s_est) > 2.0)) else 0.05
    overlap_recovered = bool(confidence >= _min_conf and (abs(final_dx) > 0.1 or abs(final_dy) > 0.1))

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
        "scale_ratio": round(float(s_est), 4),
        "scale_confidence": scale_confidence,
        "scale_capped": bool(scale_capped),
        "shift_capped": bool(shift_capped),
        "scale_note": scale_note,
        "frame": "reference_pixels",
        "method": method,
        "overlap_recovered": overlap_recovered,
        "initial_bounds": initial_bounds,
        "recovered_bounds": recovered_bounds if recovered_bounds else initial_bounds,
        "bounds_shift_meters": bounds_shift_meters,
    }
