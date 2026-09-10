"""
matcher_cfog.py — Primary Cross-Sensor Registration Engine for Chandrayaan-2

Implements scientifically defensible cross-sensor alignment:
1. Common physical-GSD normalization resolving the ~16–20x scale gap between OHRC and TMC-2
   via area-averaged resampling to the coarser GSD (not a scale-invariant descriptor;
   ~275x OHRC->IIRS is overlay/composition only, not direct matching).
2. Illumination-robust (moderate) single-channel Phase Congruency structural representation.
   NOTE: multi-channel CFOG tensor F(x,y,k) is not implemented; matching uses
   normalized single-channel PC + NCC/MI. Robust to gain/bias, not to full
   shadow-reversal (see 162deg triplet_new_2022 failure).
3. Spatially distributed correspondence selection across configurable grid cells.
4. Local patch-level Fourier Phase Correlation sub-pixel refinement at matched physical ground scales.
5. Robust geometric estimation (RANSAC with transformation quality sanity gates).
6. ZERO synthetic fallbacks in this engine (never fabricates corner points or identity matrices).
   IIRS chained composition (0-inlier legs) is reported as composed, not measured.
7. Complete output package: registered GeoTIFF raster, checkerboard QA, matches JSON, canonical metrics JSON, and metadata JSON.
   Canonical reporting grid is fixed 10x10; dynamic grids are matching-internal only.
"""

from __future__ import annotations

import os
import json
import math
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import logging

import numpy as np
import cv2

from metadata import extract_sensor_metadata, SensorMetadata
from metrics import compute_canonical_metrics, verify_transformation_quality, calculate_reprojection_errors
from geometry import warp_piecewise_affine, warp_thin_plate_splines, dem_ray_intersection, ransac_dem_aware_fit
from spectral import enhance_iirs_structural_features, quantify_iirs_residuals
from spatial_suppression import (
    detect_salient_keypoints,
    suppression_via_square_covering,
    apply_grid_density_budgeting,
)

logger = logging.getLogger("ML_model.matcher_cfog")


# ---------------------------------------------------------------------------
# Sub-pixel-preserving JSON serialization helpers
# ---------------------------------------------------------------------------

class SubpixelJSONEncoder(json.JSONEncoder):
    """JSONEncoder that converts NumPy scalars/arrays to native Python types
    without truncating sub-pixel precision.

    Coordinates are passed through as full-precision ``float()`` — never
    ``int()``, ``round()`` or ``astype(int)`` — so values like 123.4567
    survive a dump/load round-trip with error < 1e-4.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return sanitize_for_json(o.tolist())
        if isinstance(o, (set, tuple)):
            return sanitize_for_json(list(o))
        return super().default(o)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy types to JSON-serializable native types.

    Floats use plain ``float()`` (full repr precision, >= 4 decimals
    retained). Only the ``confidence`` score may be rounded by the caller;
    coordinates must never be rounded here.
    """
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def make_match_record(
    source_x: Any,
    source_y: Any,
    target_x: Any,
    target_y: Any,
    confidence: Any,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a single match dict with full sub-pixel coordinate precision.

    Coordinates are coerced with ``float()`` only (no ``round()``/``int()``).
    ``confidence`` is rounded to 2 decimals for readability.
    """
    return {
        "source_x": float(source_x),
        "source_y": float(source_y),
        "target_x": float(target_x),
        "target_y": float(target_y),
        "image1_x": float(source_x),
        "image1_y": float(source_y),
        "image2_x": float(target_x),
        "image2_y": float(target_y),
        "confidence": round(float(confidence), 2),
        **{k: sanitize_for_json(v) for k, v in extra.items()},
    }


def dumps_matches_json(records: Any, indent: int = 2) -> str:
    """Serialize match records to a human-readable JSON string (indent=2)."""
    return json.dumps(sanitize_for_json(records), indent=indent, cls=SubpixelJSONEncoder)


def dump_matches_json(records: Any, path: str | Path, indent: int = 2) -> Path:
    """Write match records to *path* preserving sub-pixel precision."""
    path = Path(path)
    path.write_text(dumps_matches_json(records, indent=indent), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Robust Multi-Band Image Loader
# ---------------------------------------------------------------------------

def load_as_float_and_color(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads an image from path. Supports GeoTIFF, multi-band hyperspectral cubes,
    PNG, and JPEG. Returns normalized 2D grayscale float32 [0, 1], uint8 BGR color,
    and embedded raster metadata.
    """
    path_str = str(path)
    raster_meta: Dict[str, Any] = {"driver": None, "crs": None, "transform": None, "count": 1}

    # Attempt rasterio first for multi-band / GeoTIFF
    try:
        import rasterio
        with rasterio.open(path_str) as src:
            raster_meta["driver"] = src.driver
            raster_meta["crs"] = str(src.crs) if src.crs else None
            raster_meta["transform"] = list(src.transform) if src.transform else None
            raster_meta["count"] = src.count

            if src.count > 3:
                # Hyperspectral cube (e.g. IIRS): Spectral feature engineering (PC1 + band ratio)
                bands = src.read().astype(np.float32)
                gray = enhance_iirs_structural_features(bands)
            elif src.count >= 3:
                rgb = np.dstack([src.read(i) for i in (1, 2, 3)]).astype(np.float32)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = src.read(1).astype(np.float32)

            g_min, g_max = float(np.nanmin(gray)), float(np.nanmax(gray))
            if g_max > g_min:
                gray = (gray - g_min) / (g_max - g_min)
            else:
                gray = np.zeros_like(gray)

            u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
            color = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            return gray.astype(np.float32), color, raster_meta
    except Exception:
        pass

    # Fallback to OpenCV
    raw = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not read image: {path_str}")

    if raw.ndim == 3 and raw.shape[2] > 3:
        gray = enhance_iirs_structural_features(raw)
        color = raw[:, :, :3].copy()
    elif raw.ndim == 3 and raw.shape[2] == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
        color = raw.copy()
    else:
        gray = raw.astype(np.float32)
        u8 = np.clip(gray, 0, 255).astype(np.uint8) if gray.max() > 1.0 else (gray * 255).astype(np.uint8)
        color = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    g_min, g_max = float(np.min(gray)), float(np.max(gray))
    if g_max > g_min:
        gray = (gray - g_min) / (g_max - g_min)
    else:
        gray = np.zeros_like(gray)

    return gray.astype(np.float32), color, raster_meta


# ---------------------------------------------------------------------------
# 2. Phase 1: Adaptive Illumination Normalization
# ---------------------------------------------------------------------------

def adaptive_illumination_normalization(img_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase 1: Adaptive Illumination Normalization
    Purpose: Mitigate extreme sun angle variations and shadow artifacts
             before Phase Congruency feature extraction.
    Method: CLAHE for local contrast equalization + Shadow Masking.
    Returns: (normalized_image, valid_mask)
    """
    # 1. CLAHE to normalize local contrast (sun angle invariance)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_uint8 = np.clip(img_gray * 255.0, 0, 255).astype(np.uint8)
    clahe_img = clahe.apply(img_uint8).astype(np.float32) / 255.0

    # 2. Shadow & Saturation Mask Generation
    # Shadows typically fall in the bottom percentile of pixel intensities.
    # We create a mask where 1.0 = valid texture, 0.0 = deep shadow or pure white saturation.
    shadow_threshold = np.percentile(img_gray, 5.0) 
    saturation_threshold = np.percentile(img_gray, 99.5)

    valid_mask = ((img_gray >= shadow_threshold) & (img_gray <= saturation_threshold)).astype(np.float32)

    # Apply a slight Gaussian blur to the mask to avoid harsh edge artifacts in the FFT
    valid_mask = cv2.GaussianBlur(valid_mask, (5, 5), 0)

    # 3. Multiply CLAHE image by the mask so shadows become neutral (0.0)
    # This prevents Phase Congruency from generating false edges in pitch-black shadows
    normalized_img = clahe_img * valid_mask

    return normalized_img, valid_mask


# ---------------------------------------------------------------------------
# 3. Phase Congruency (Illumination-Robust Structural Features)
# ---------------------------------------------------------------------------

def compute_phase_congruency(
    img: np.ndarray,
    num_orientations: int = 4,
    num_scales: int = 3,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
) -> np.ndarray:
    """
    Computes 2D Phase Congruency via Log-Gabor filter banks in frequency domain.
    Phase Congruency detects structural features based on frequency-phase agreement,
    providing moderate robustness to gain/bias and mild illumination change.
    It does NOT guarantee invariance to diametric shadow reversal (e.g. ~162deg
    sun-azimuth flip in triplet_new_2022) or to full contrast inversion.
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)
    img_f = img_f - float(np.mean(img_f))

    y_idx = np.fft.fftfreq(h).astype(np.float32)
    x_idx = np.fft.fftfreq(w).astype(np.float32)
    xv, yv = np.meshgrid(x_idx, y_idx)
    radius = np.sqrt(xv**2 + yv**2).astype(np.float32)
    radius[0, 0] = 1.0  # avoid log(0)
    theta = np.arctan2(-yv, xv).astype(np.float32)

    F = np.fft.fft2(img_f)

    energy_total = np.zeros((h, w), dtype=np.float32)
    amplitude_total = np.zeros((h, w), dtype=np.float32)

    d_theta = np.pi / float(num_orientations)
    theta_sigma = 1.2 / float(num_orientations)

    for o in range(num_orientations):
        angl = o * d_theta
        diff_theta = np.abs(np.arctan2(np.sin(theta - angl), np.cos(theta - angl)))
        ang_filter = np.exp(-(diff_theta**2) / (2.0 * theta_sigma**2)).astype(np.float32)

        sum_e = np.zeros((h, w), dtype=np.float32)
        sum_o = np.zeros((h, w), dtype=np.float32)

        wavelength = min_wavelength
        for s in range(num_scales):
            fo = 1.0 / wavelength
            log_gabor = np.exp(
                -((np.log(radius / fo)) ** 2) / (2.0 * (np.log(sigma_on_f)) ** 2)
            ).astype(np.float32)
            log_gabor[0, 0] = 0.0

            filter_2d = log_gabor * ang_filter
            resp = np.fft.ifft2(F * filter_2d)

            re = np.real(resp).astype(np.float32)
            im = np.imag(resp).astype(np.float32)
            amp = np.sqrt(re**2 + im**2)

            sum_e += re
            sum_o += im
            amplitude_total += amp
            wavelength *= mult

        energy_o = np.sqrt(sum_e**2 + sum_o**2)
        energy_total += energy_o

    pc = energy_total / (amplitude_total + 1e-4)
    return np.clip(pc, 0.0, 1.0).astype(np.float32)


def multi_scale_phase_congruency(img: np.ndarray, scales: int = 3) -> List[np.ndarray]:
    """
    Phase 2: Multi-Scale Feature Extraction
    Purpose: Handle the massive scale gap between OHRC and TMC-2.
    Method: Compute Phase Congruency at 3 Gaussian pyramid levels.
    Returns: List of Phase Congruency maps [Level 0 (Original), Level 1 (1/2), Level 2 (1/4)]
    """
    pyramid = [img]
    current_img = img
    for _ in range(scales - 1):
        # Downsample by 2 for the next scale
        current_img = cv2.pyrDown(current_img)
        pyramid.append(current_img)

    pc_pyramid = []
    for level_img in pyramid:
        pc = compute_phase_congruency(level_img, num_orientations=4, num_scales=3)
        pc_pyramid.append(pc)

    return pc_pyramid


# ---------------------------------------------------------------------------
# 3b. CV Scale-Ratio Fallback (Log-Polar Phase Congruency, no metadata)
# ---------------------------------------------------------------------------

def phase_symmetry_center(pc: np.ndarray) -> Tuple[float, float]:
    """
    Robust log-polar center from a Phase Congruency map.

    Returns the intensity-weighted centroid of phase-symmetric structure
    (bright PC pixels = structural symmetry axes such as crater rims/ridges),
    blended toward the geometric center for stability. Falls back to the
    geometric center when the map is flat or the centroid is pushed to the
    image border (e.g. by shadow-edge bias).

    Pure geometric-center log-polar transforms mis-estimate scale whenever
    the scene's structural mass is off-center, so the centroid must lead.
    """
    h, w = pc.shape[:2]
    gx, gy = w / 2.0, h / 2.0
    try:
        pc_n = np.clip(np.asarray(pc, dtype=np.float64), 0.0, None)
        if not np.all(np.isfinite(pc_n)) or float(np.std(pc_n)) < 1e-12:
            return gx, gy
        # Winsorize at p99 so a single bright rim cannot hijack the centroid.
        cap = float(np.percentile(pc_n, 99.0))
        if cap > 1e-12:
            pc_n = np.minimum(pc_n, cap)
        moments = cv2.moments(pc_n.astype(np.float32))
        if abs(float(moments["m00"])) < 1e-9:
            return gx, gy
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        if not (np.isfinite(cx) and np.isfinite(cy)):
            return gx, gy
        if abs(cx - gx) > 0.25 * w or abs(cy - gy) > 0.25 * h:
            return gx, gy
        return 0.75 * cx + 0.25 * gx, 0.75 * cy + 0.25 * gy
    except Exception:
        return gx, gy


def estimate_scale_ratio_cv(
    img1: np.ndarray,
    img2: np.ndarray,
    lp_max_dim: int = 1024,
    response_threshold: float = 0.03,
    max_ratio: float = 300.0,
) -> float:
    """
    Pure computer-vision estimate of the relative scale ratio between two
    images of overlapping terrain, requiring no PDS4/sensor metadata.

    Algorithm (Fourier-Mellin style, translation assumed small):
      1. Structural representation: Phase Congruency maps of both images
         (gain/bias robust; reuses :func:`compute_phase_congruency`).
      2. Common downsampling (same factor for both, ratio-preserving) so the
         joint canvas fits ``lp_max_dim``, then centered zero-padding to a
         common N x N canvas WITHOUT resampling either image to the other's
         size (which would erase the very ratio being measured).
      3. Shared log-polar center = mean of the two phase-symmetry centroids
         (see :func:`phase_symmetry_center`), shared radial gain
         ``M = N / ln(Rmax)`` so both maps share one log-radius axis:
         ``rho = M * ln(r)``.
      4. ``cv2.phaseCorrelate`` on the mean-removed log-polar maps with a
         Hanning window. A zoom by ``s`` is a shift ``d_rho = M * ln(s)``,
         hence ``s = exp(d_rho / M)``.

    Returns:
        ``S >= 1``: magnitude of the scale gap. Direction is intentionally
        NOT signed: the caller resolves it from pixel dimensions (the larger
        image is assumed finer for the same footprint) and upscales the
        smaller image by ``S``.

    Raises:
        ValueError: blank/uniform inputs, correlation response below
            ``response_threshold`` (unrelated scenes or gaps beyond ~8-10x
            where log-polar correlation decorrelates), or ``S > max_ratio``.

    Validated on synthetic crater fields: true 1.0/1.6/2.0/4.0/8.0 ->
    estimated 1.00/1.59/1.98/3.92/7.69 with responses 1.0..0.05.
    """
    def _as_gray_float(a: np.ndarray, name: str) -> np.ndarray:
        g = np.asarray(a)
        if g.ndim == 3:
            if g.shape[2] == 1:
                g = g[:, :, 0]
            elif g.shape[2] in (3, 4):
                g = g[:, :, :3].mean(axis=2)
            else:
                raise ValueError(f"{name}: unsupported channel count {g.shape[2]}")
        elif g.ndim != 2:
            raise ValueError(f"{name}: expected 2D grayscale, got shape {g.shape}")
        g = g.astype(np.float64)
        if not np.all(np.isfinite(g)):
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        g -= float(g.min())
        mx = float(g.max())
        if mx < 1e-12 or float(np.std(g)) < 1e-12:
            raise ValueError(f"{name}: blank or uniform image carries no structural scale cue")
        return (g / mx).astype(np.float32)

    g1 = _as_gray_float(img1, "img1")
    g2 = _as_gray_float(img2, "img2")
    h1, w1 = g1.shape[:2]
    h2, w2 = g2.shape[:2]
    if min(h1, w1, h2, w2) < 16:
        raise ValueError("images too small for log-polar scale estimation (min dim < 16 px)")

    # Ratio-preserving joint downsample (-same- factor keeps S intact).
    down = max(1.0, float(max(h1, w1, h2, w2)) / float(lp_max_dim))
    nw1, nh1 = max(8, int(round(w1 / down))), max(8, int(round(h1 / down)))
    nw2, nh2 = max(8, int(round(w2 / down))), max(8, int(round(h2 / down)))
    d1 = cv2.resize(g1, (nw1, nh1), interpolation=cv2.INTER_AREA)
    d2 = cv2.resize(g2, (nw2, nh2), interpolation=cv2.INTER_AREA)

    pc1 = compute_phase_congruency(d1)
    pc2 = compute_phase_congruency(d2)
    if float(np.std(pc1)) < 1e-9 or float(np.std(pc2)) < 1e-9:
        raise ValueError("phase congruency maps are flat; no structural scale cue")

    n = max(pc1.shape[0], pc1.shape[1], pc2.shape[0], pc2.shape[1])
    if n % 2 == 1:
        n += 1

    def _center_pad(pc: np.ndarray) -> np.ndarray:
        h, w = pc.shape[:2]
        top = (n - h) // 2
        left = (n - w) // 2
        return cv2.copyMakeBorder(
            pc, top, n - h - top, left, n - w - left,
            cv2.BORDER_CONSTANT, value=0.0,
        )

    p1 = _center_pad(pc1)
    p2 = _center_pad(pc2)

    c1 = phase_symmetry_center(p1)
    c2 = phase_symmetry_center(p2)
    cx, cy = (c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0

    corners = np.array([[0, 0], [n, 0], [0, n], [n, n]], dtype=np.float64)
    rmax = float(np.max(np.sqrt((corners[:, 0] - cx) ** 2 + (corners[:, 1] - cy) ** 2)))
    m_gain = float(n) / float(np.log(max(rmax, 2.0)))

    lp1 = cv2.logPolar(p1.astype(np.float32), (float(cx), float(cy)), m_gain,
                       cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    lp2 = cv2.logPolar(p2.astype(np.float32), (float(cx), float(cy)), m_gain,
                       cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    lp1 = lp1.astype(np.float64) - float(np.mean(lp1))
    lp2 = lp2.astype(np.float64) - float(np.mean(lp2))
    window = cv2.createHanningWindow((n, n), cv2.CV_64F)
    (dx, _dy), response = cv2.phaseCorrelate(lp1, lp2, window)

    if not np.isfinite(dx) or float(response) < float(response_threshold):
        raise ValueError(
            f"log-polar phase correlation too weak (response={float(response):.4f} "
            f"< {float(response_threshold)}); scenes may not overlap or the scale "
            "gap exceeds the reliable ~8-10x range"
        )
    s_ratio = float(np.exp(abs(float(dx)) / m_gain))
    if not np.isfinite(s_ratio) or s_ratio > float(max_ratio):
        raise ValueError(f"estimated scale ratio {s_ratio:.1f}x exceeds plausible max {float(max_ratio)}x")

    logger.info(
        "CV log-polar scale estimate: S=%.3f (d_rho=%.2f px, response=%.3f, center=(%.1f, %.1f))",
        s_ratio, dx, float(response), cx, cy,
    )
    return max(1.0, s_ratio)


# ---------------------------------------------------------------------------
# 3. DEM Relief Displacement Compensation
# ---------------------------------------------------------------------------

def apply_dem_relief_compensation(
    img: np.ndarray,
    dem: Optional[np.ndarray],
    emission_deg: Optional[float],
    azimuth_deg: Optional[float],
    gsd_m: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies simplified local DEM-based relief displacement compensation.
    Corrects parallax displacement caused by terrain elevation under off-nadir viewing.
    Note: Labeled honestly as relief displacement compensation, NOT full sensor-model ray-tracing.
    """
    if dem is None or emission_deg is None or abs(emission_deg) < 0.5:
        return img.copy(), {"enabled": False, "method": None, "reason": "No DEM or nadir viewing"}

    if azimuth_deg is None:
        # Sensor line-of-sight azimuth is genuinely unknown here: applying the
        # relief shift in a guessed direction (the old 45.0 default) moves
        # every pixel the wrong way on 3 of 4 compass quadrants. Disabled is
        # honest; callers log sun azimuth as provenance instead.
        return img.copy(), {"enabled": False, "method": None,
                            "reason": "Sensor LOS azimuth unavailable; relief compensation disabled to prevent hallucination."}

    h, w = img.shape[:2]
    if dem.shape[:2] != (h, w):
        dem_res = cv2.resize(dem.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        dem_res = dem.astype(np.float32)

    dem_rel = (dem_res - float(np.mean(dem_res))).astype(np.float32)
    e_rad = np.radians(emission_deg)
    psi_rad = np.radians(azimuth_deg)

    scale = float(np.tan(e_rad) / max(gsd_m, 1e-3))
    dx = (dem_rel * (scale * np.cos(psi_rad))).astype(np.float32)
    dy = (dem_rel * (scale * np.sin(psi_rad))).astype(np.float32)

    x_coords, y_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)

    compensated = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return compensated, {
        "enabled": True,
        "method": "dem_relief_displacement_compensation",
        "emission_deg": emission_deg,
        "azimuth_deg": azimuth_deg,
        "limitations": "Simplified local relief displacement; not rigorous photogrammetric ray-intersection.",
    }


# ---------------------------------------------------------------------------
# 3b. Synthetic DEM Hillshade (Sun-Angle-Invariant Reference Projection)
# ---------------------------------------------------------------------------

def sun_azimuth_delta_deg(az1: Optional[float], az2: Optional[float]) -> Optional[float]:
    """Circular absolute sun-azimuth difference in [0, 180]. None if unknown."""
    if az1 is None or az2 is None:
        return None
    try:
        d = abs(float(az1) - float(az2)) % 360.0
        return float(d if d <= 180.0 else 360.0 - d)
    except Exception:
        return None


def resolve_sun_elevation_deg(meta) -> float:
    """Resolve sun elevation from metadata; incidence -> elevation fallback."""
    try:
        el = getattr(meta, "sun_elevation_deg", None)
        if el is not None and np.isfinite(float(el)):
            return float(np.clip(float(el), 5.0, 85.0))
        inc = getattr(meta, "incidence_angle_deg", None)
        if inc is not None and np.isfinite(float(inc)):
            return float(np.clip(90.0 - float(inc), 5.0, 85.0))
    except Exception:
        pass
    return 45.0


def render_synthetic_shaded_relief(
    dem_array: np.ndarray,
    target_azimuth_deg: float,
    target_elevation_deg: float,
    working_gsd_m: float = 5.0,
    z_factor: float = 1.0,
    blur_ksize: int = 3,
) -> np.ndarray:
    """
    Pure NumPy/OpenCV GIS hillshade (Esri-style) rendered from a DEM.

    Args:
        dem_array: (H, W) elevation in meters (e.g. 1000..5000). NaNs allowed.
        target_azimuth_deg: Sun azimuth, degrees clockwise from north.
        target_elevation_deg: Sun elevation above horizon, degrees.
        working_gsd_m: Meters per pixel — divides the Horn gradients so shadow
            lengths stay physically accurate after common-GSD resampling.
        z_factor: Vertical exaggeration (1.0 = true scale).
        blur_ksize: Gaussian blur kernel (odd, >=3) applied to suppress
            high-frequency DEM noise before Phase Congruency.

    Returns:
        (H, W) float32 shaded relief normalized to [0, 1].
    """
    dem = np.asarray(dem_array, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError(f"dem_array must be 2D, got shape {dem.shape}")
    # Neutralize voids: fill NaN/Inf with local mean so gradients stay finite.
    if not np.all(np.isfinite(dem)):
        fill = float(np.nanmean(dem)) if np.any(np.isfinite(dem)) else 0.0
        dem = np.where(np.isfinite(dem), dem, fill)
    gsd = max(float(working_gsd_m), 1e-3)

    # Horn (1981) 3x3 gradient in elevation-units per meter.
    padded = np.pad(dem, 1, mode="reflect")
    dzdx = (
        (padded[0:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[0:-2, 0:-2] + 2.0 * padded[1:-1, 0:-2] + padded[2:, 0:-2])
    ) / (8.0 * gsd)
    dzdy = (
        (padded[2:, 0:-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[0:-2, 0:-2] + 2.0 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
    ) / (8.0 * gsd)
    dzdx *= float(z_factor)
    dzdy *= float(z_factor)

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect_rad = np.arctan2(dzdx, -dzdy)
    aspect_rad = np.where(aspect_rad < 0.0, aspect_rad + 2.0 * np.pi, aspect_rad)

    az = float(target_azimuth_deg) % 360.0
    el = float(np.clip(float(target_elevation_deg), 5.0, 85.0))
    zenith_rad = np.radians(90.0 - el)
    azimuth_math = np.radians((360.0 - az + 90.0) % 360.0)

    shade = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_math - aspect_rad)
    )
    shade = np.clip(shade, 0.0, 1.0).astype(np.float32)

    # Suppress high-frequency DEM noise / fake micro-craters before PC.
    k = int(blur_ksize) if int(blur_ksize) >= 3 else 3
    if k % 2 == 0:
        k += 1
    try:
        shade = cv2.GaussianBlur(shade, (k, k), 0)
    except Exception:
        pass
    shade = np.clip(shade, 0.0, 1.0).astype(np.float32)
    return shade


# ---------------------------------------------------------------------------
# 4. Patch-Level Fourier Phase Correlation Sub-Pixel Refinement
# ---------------------------------------------------------------------------

def subpixel_phase_correlation(
    patch1: np.ndarray,
    patch2: np.ndarray,
) -> Tuple[float, float, float, bool]:
    """
    Fourier Phase Correlation with 2D quadratic peak surface fitting.
    Returns (delta_x, delta_y, peak_correlation, is_valid).
    Rejects patches with low texture or ambiguous peak responses.
    """
    h, w = patch1.shape[:2]
    if h < 16 or w < 16:
        return 0.0, 0.0, 0.0, False

    p1 = patch1.astype(np.float32)
    p2 = patch2.astype(np.float32)

    # Check texture variance
    if np.var(p1) < 1e-6 or np.var(p2) < 1e-6:
        return 0.0, 0.0, 0.0, False

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

    if peak_val < 0.15:
        # Ambiguous peak
        return 0.0, 0.0, peak_val, False

    cy, cx = h // 2, w // 2
    sub_y, sub_x = float(peak_y), float(peak_x)

    # --- TRUE 2D ALGEBRAIC PARABOLOID FIT ---
    # Fits z(x,y) = ax^2 + by^2 + cxy + dx + ey + f to the 3x3 neighborhood
    if 0 < peak_y < h - 1 and 0 < peak_x < w - 1:
        # Extract 3x3 neighborhood around the peak
        c = float(corr[peak_y, peak_x])
        c_l = float(corr[peak_y, peak_x - 1])
        c_r = float(corr[peak_y, peak_x + 1])
        c_u = float(corr[peak_y - 1, peak_x])
        c_d = float(corr[peak_y + 1, peak_x])
        c_ul = float(corr[peak_y - 1, peak_x - 1])
        c_ur = float(corr[peak_y - 1, peak_x + 1])
        c_dl = float(corr[peak_y + 1, peak_x - 1])
        c_dr = float(corr[peak_y + 1, peak_x + 1])

        # Compute coefficients for 2D paraboloid
        a = 0.5 * (c_r + c_l - 2 * c)
        b = 0.5 * (c_d + c_u - 2 * c)
        c_cross = 0.25 * (c_dr + c_ul - c_dl - c_ur)
        d = 0.5 * (c_r - c_l)
        e = 0.5 * (c_d - c_u)

        denom = 4 * a * b - c_cross * c_cross
        if abs(denom) > 1e-9:
            dx = (c_cross * e - 2 * b * d) / denom
            dy = (c_cross * d - 2 * a * e) / denom

            # Clip to prevent crazy jumps
            dx = np.clip(dx, -0.9, 0.9)
            dy = np.clip(dy, -0.9, 0.9)

            sub_x += float(dx)
            sub_y += float(dy)

    shift_x = float(sub_x - cx)
    shift_y = float(sub_y - cy)
    return shift_x, shift_y, peak_val, True


def mutual_information_score(image_a: np.ndarray, image_b: np.ndarray, bins: int = 32) -> float:
    """Estimate normalized mutual information for two equally shaped patches."""
    a = np.asarray(image_a, dtype=np.float32).ravel()
    b = np.asarray(image_b, dtype=np.float32).ravel()
    if a.size == 0 or a.size != b.size or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    a_edges = np.linspace(float(a.min()), float(a.max()) + 1e-6, bins + 1)
    b_edges = np.linspace(float(b.min()), float(b.max()) + 1e-6, bins + 1)
    joint, _, _ = np.histogram2d(a, b, bins=(a_edges, b_edges))
    joint = joint / max(float(joint.sum()), 1.0)
    marginal_a = joint.sum(axis=1, keepdims=True)
    marginal_b = joint.sum(axis=0, keepdims=True)
    expected = marginal_a @ marginal_b
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log((joint[mask] + 1e-12) / (expected[mask] + 1e-12))))
    entropy = -float(np.sum(joint[mask] * np.log(joint[mask] + 1e-12)))
    return mi / max(entropy, 1e-12)


def detect_blob_centroids(image: np.ndarray, min_area: int = 3) -> np.ndarray:
    """Detect coarse structural blobs and return intensity-weighted centroids."""
    image_f = np.asarray(image, dtype=np.float32)
    if image_f.ndim != 2 or np.std(image_f) < 1e-6:
        return np.empty((0, 2), dtype=np.float32)
    threshold = float(np.percentile(image_f, 75.0))
    binary = (image_f >= threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    found: List[List[float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = binary == label
        weights = np.maximum(image_f[mask] - threshold, 0.0)
        ys, xs = np.nonzero(mask)
        weight_sum = float(weights.sum())
        if weight_sum > 1e-6:
            found.append([float(np.dot(xs, weights) / weight_sum), float(np.dot(ys, weights) / weight_sum)])
        else:
            found.append([float(centroids[label, 0]), float(centroids[label, 1])])
    return np.asarray(found, dtype=np.float32).reshape(-1, 2)


def apply_grid_nms(
    matches: List[Dict[str, Any]] | np.ndarray,
    image_shape: Tuple[int, int] = (512, 512),
    grid_dims: Tuple[int, int] = (10, 10),
    max_per_cell: int = 4,
) -> List[Dict[str, Any]] | np.ndarray:
    """
    Applies Grid-based Non-Maximum Suppression (Grid NMS) and density budgeting to candidate correspondences.
    
    Divides the image area into grid_dims[0] x grid_dims[1] cells (e.g. 10x10).
    Enforces grid density budgeting: actively prioritizes matches from under-represented
    cells before dense cells receive additional candidate allocations (up to max_per_cell).
    Guarantees uniform spatial distribution across the entire scene and prevents
    clustering exclusively on prominent crater rims.
    """
    if matches is None or len(matches) == 0:
        return matches

    is_dict_list = isinstance(matches, list) and len(matches) > 0 and isinstance(matches[0], dict)
    if is_dict_list:
        return apply_grid_density_budgeting(
            matches,
            image_shape=image_shape,
            grid_dims=grid_dims,
            max_per_cell=max_per_cell,
        )

    # Array of points or pairs: tiered round-robin density budgeting
    h, w = image_shape[:2]
    gw, gh = grid_dims
    cell_w = max(1.0, float(w) / float(gw))
    cell_h = max(1.0, float(h) / float(gh))

    grid_bins: Dict[Tuple[int, int], List[Any]] = {}
    arr = np.asarray(matches)
    for i in range(len(arr)):
        pt = arr[i]
        x = float(pt[0])
        y = float(pt[1])
        gx = min(gw - 1, max(0, int(x / cell_w)))
        gy = min(gh - 1, max(0, int(y / cell_h)))
        grid_bins.setdefault((gx, gy), []).append((i, pt))

    selected_indices = []
    occupied_cells = sorted(grid_bins.keys())
    for round_idx in range(max_per_cell):
        for cell_key in occupied_cells:
            items = grid_bins[cell_key]
            if round_idx < len(items):
                selected_indices.append(items[round_idx][0])
    selected_indices.sort()
    return arr[selected_indices]


def verify_spatial_quality_gate(
    inlier_cells: List[Tuple[int, int]],
    min_distinct_cells: int = 3,
    max_single_cell_concentration: float = 0.60,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    QUALITY GATE 4: Spatial Support & Concentration Check.
    Verifies that verified inliers have genuine spatial support:
    1. Inliers must span >= min_distinct_cells distinct grid cells.
    2. No single cell may contain > max_single_cell_concentration (e.g. 60%) of all inliers.

    Returns:
        (is_valid, failure_reason, details_dict)
    """
    total_inliers = len(inlier_cells)
    if total_inliers < 4:
        return True, "Fewer than 4 inliers; minimum inlier threshold gate applies", {
            "distinct_cells": len(set(inlier_cells)),
            "concentration_ratio": 1.0 if inlier_cells else 0.0,
            "max_in_single_cell": len(inlier_cells),
            "total_inliers": total_inliers,
        }

    from collections import Counter
    distinct_cells = len(set(inlier_cells))
    cell_counts = Counter(inlier_cells)
    max_in_single_cell = max(cell_counts.values()) if cell_counts else 0
    concentration_ratio = max_in_single_cell / max(1, total_inliers)

    details = {
        "distinct_cells": distinct_cells,
        "concentration_ratio": concentration_ratio,
        "max_in_single_cell": max_in_single_cell,
        "total_inliers": total_inliers,
    }

    if distinct_cells < min_distinct_cells:
        reason = (
            f"inliers are excessively concentrated ({distinct_cells} distinct cells occupied, "
            f"required >= {min_distinct_cells})"
        )
        return False, reason, details

    if concentration_ratio > max_single_cell_concentration:
        reason = (
            f"inliers are excessively concentrated (single cell concentration {concentration_ratio*100:.1f}%, "
            f"maximum allowed is {max_single_cell_concentration*100:.1f}%)"
        )
        return False, reason, details

    return True, "Spatial distribution gate passed", details


def refine_inliers_lucas_kanade(
    work_feat1: np.ndarray,
    work_feat2: np.ndarray,
    inlier_src: np.ndarray,
    inlier_dst: np.ndarray,
    scale_factor1: float,
    scale_factor2: float,
    win_size: int = 15,
    max_shift_px: float = 3.0,
    fb_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Post-RANSAC Lucas-Kanade optical flow refinement for sub-pixel accuracy.
    Refines each inlier match position via iterative LK optical flow with
    forward-backward consistency verification on structural feature maps.

    Args:
        work_feat1: Working-scale source structural feature map (e.g. Phase Congruency) [0, 1].
        work_feat2: Working-scale reference structural feature map [0, 1].
        inlier_src: (N, 2) native-space source points.
        inlier_dst: (N, 2) native-space destination points.
        scale_factor1: Conversion from native px to working px for source.
        scale_factor2: Conversion from native px to working px for reference.
        win_size: LK window size in working pixels.
        max_shift_px: Maximum allowed refinement shift in working pixels.
        fb_threshold: Forward-backward round-trip error threshold in working pixels.

    Returns:
        Refined (src, dst) arrays in native space and a stats dict.
    """
    n = len(inlier_src)
    if n == 0:
        return inlier_src.copy(), inlier_dst.copy(), {"refined_count": 0, "total": 0}

    # Convert normalized structural feature representations to uint8 for OpenCV LK
    img1_u8 = (np.clip(work_feat1, 0.0, 1.0) * 255.0).astype(np.uint8)
    img2_u8 = (np.clip(work_feat2, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Map native-space dst points into working-scale space
    work_dst = inlier_dst.copy()
    work_dst[:, 0] /= scale_factor2
    work_dst[:, 1] /= scale_factor2
    pts_fwd = work_dst.reshape(-1, 1, 2).astype(np.float32)

    work_src_pts = inlier_src.copy()
    work_src_pts[:, 0] /= scale_factor1
    work_src_pts[:, 1] /= scale_factor1
    pts_src_lk = work_src_pts.reshape(-1, 1, 2).astype(np.float32)

    # CRITICAL: Preserve copies of initial target and source points before LK
    # because cv2.calcOpticalFlowPyrLK mutates the nextPts array in place!
    initial_target = pts_fwd.copy()
    initial_source = pts_src_lk.copy()

    lk_params = dict(
        winSize=(win_size, win_size),
        maxLevel=0,  # Pure native-resolution sub-pixel refinement around initial flow guess
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,  # Track starting from coarse match position
    )

    # Forward: source -> target starting from initial coarse match position
    fwd_guess = pts_fwd.copy()
    refined_fwd, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        img1_u8, img2_u8, pts_src_lk, fwd_guess, **lk_params
    )

    # Backward: refined target -> source starting from initial source position
    bwd_guess = pts_src_lk.copy()
    refined_bwd, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        img2_u8, img1_u8, refined_fwd, bwd_guess, **lk_params
    )

    refined_dst = inlier_dst.copy()
    refined_count = 0
    debug_points: List[Dict[str, Any]] = []

    for i in range(n):
        s_fwd = int(status_fwd[i, 0]) if status_fwd is not None and i < len(status_fwd) else 0
        s_bwd = int(status_bwd[i, 0]) if status_bwd is not None and i < len(status_bwd) else 0

        fb_err: Optional[float] = None
        if refined_bwd is not None and i < len(refined_bwd):
            fb_err = float(np.linalg.norm(refined_bwd[i, 0] - initial_source[i, 0]))

        shift_mag: Optional[float] = None
        if refined_fwd is not None and i < len(refined_fwd):
            shift = refined_fwd[i, 0] - initial_target[i, 0]
            shift_mag = float(np.linalg.norm(shift))

        pt_record: Dict[str, Any] = {
            "point_index": i,
            "src_pt": [float(inlier_src[i, 0]), float(inlier_src[i, 1])],
            "dst_pt": [float(inlier_dst[i, 0]), float(inlier_dst[i, 1])],
            "status_fwd": s_fwd,
            "status_bwd": s_bwd,
            "fb_err": fb_err,
            "shift_mag": shift_mag,
            "passed": False,
            "failure_gate": None,
        }

        if s_fwd != 1:
            pt_record["failure_gate"] = "fwd_non_convergence"
        elif s_bwd != 1:
            pt_record["failure_gate"] = "bwd_non_convergence"
        elif fb_err is not None and fb_err > fb_threshold:
            pt_record["failure_gate"] = f"fb_threshold_exceeded (fb_err={fb_err:.4f} > {fb_threshold})"
        elif shift_mag is not None and shift_mag > max_shift_px:
            pt_record["failure_gate"] = f"max_shift_exceeded (shift_mag={shift_mag:.4f} > {max_shift_px})"
        else:
            pt_record["passed"] = True
            # Accept refinement — convert back to native space
            refined_dst[i, 0] = float(refined_fwd[i, 0, 0]) * scale_factor2
            refined_dst[i, 1] = float(refined_fwd[i, 0, 1]) * scale_factor2
            refined_count += 1

        debug_points.append(pt_record)

    stats = {
        "refined_count": refined_count,
        "total": n,
        "fb_threshold": fb_threshold,
        "max_shift_px": max_shift_px,
        "debug_points": debug_points,
    }
    return inlier_src.copy(), refined_dst, stats


def _guided_refill_matches(
    kps1_ssc,
    selected_matches,
    pc1,
    pc2,
    H_native,
    scale_factor1: float,
    scale_factor2: float,
    work_w1: int,
    work_h1: int,
    work_w2: int,
    work_h2: int,
    half_patch_c: int,
    multimodal_pair: bool,
    grid_size: int,
    cell_w: float,
    cell_h: float,
    max_add: int = 80,
    radius: int = 14,
):
    """H-guided second-pass matching (honest densification, no synthesis).

    Projects unused pre-match SSC keypoints through the RANSAC homography
    (native scale) and correlates a tight local window around the prediction
    with the SAME NCC/MI thresholds as coarse matching. Every returned point
    is a measured correlation, re-verified by a second RANSAC + quality gates
    by the caller. Returns [] when H is None/degenerate.
    """
    try:
        Hm = np.asarray(H_native, dtype=np.float64)
        if Hm.shape != (3, 3) or not np.all(np.isfinite(Hm)):
            return []
        used = set()
        for m in selected_matches:
            try:
                ux = int(round(float(m["work_x1"])))
                uy = int(round(float(m["work_y1"])))
                used.add((ux, uy))
            except Exception:
                continue
        added = []
        thresh = 0.05 if multimodal_pair else 0.35
        for kx, ky, _ in kps1_ssc:
            if len(added) >= max_add:
                break
            # Integer pixels for slicing only; full floats preserved for coords.
            kx_f, ky_f = float(kx), float(ky)
            cx, cy = int(round(kx_f)), int(round(ky_f))
            if (cx, cy) in used:
                continue
            if cy < half_patch_c or cy >= work_h1 - half_patch_c:
                continue
            if cx < half_patch_c or cx >= work_w1 - half_patch_c:
                continue
            tmpl = pc1[cy - half_patch_c:cy + half_patch_c, cx - half_patch_c:cx + half_patch_c]
            if float(np.std(tmpl)) < 1e-4:
                continue
            # native -> H -> work2 prediction (full-precision floats)
            p1 = np.array([[[float(kx_f * scale_factor1), float(ky_f * scale_factor1)]]], dtype=np.float64)
            try:
                p2 = cv2.perspectiveTransform(p1, Hm).reshape(-1)
            except Exception:
                continue
            px, py = float(p2[0] / max(scale_factor2, 1e-9)), float(p2[1] / max(scale_factor2, 1e-9))
            s_min_x = max(0, int(px) - radius)
            s_max_x = min(work_w2, int(px) + radius)
            s_min_y = max(0, int(py) - radius)
            s_max_y = min(work_h2, int(py) + radius)
            if s_max_x - s_min_x <= tmpl.shape[1] or s_max_y - s_min_y <= tmpl.shape[0]:
                continue
            search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
            if float(np.std(search_region)) < 1e-4:
                continue
            res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if multimodal_pair:
                cand = search_region[max_loc[1]:max_loc[1] + tmpl.shape[0],
                                     max_loc[0]:max_loc[0] + tmpl.shape[1]]
                if cand.shape == tmpl.shape:
                    max_val = mutual_information_score(tmpl, cand)
            if max_val > thresh:
                bx = s_min_x + max_loc[0] + half_patch_c
                by = s_min_y + max_loc[1] + half_patch_c
                added.append({
                    "work_x1": float(kx_f), "work_y1": float(ky_f),
                    "work_x2": float(bx), "work_y2": float(by),
                    "score": float(max_val),
                    "cell": (min(grid_size - 1, int(kx_f / max(cell_w, 1e-6))),
                             min(grid_size - 1, int(ky_f / max(cell_h, 1e-6)))),
                    "method": "guided_refill",
                })
                used.add((cx, cy))
        return added
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 5. Primary Registration Pipeline
# ---------------------------------------------------------------------------

def match_images_cfog(
    img_path1: str | Path,
    img_path2: str | Path,
    dem_path: Optional[str | Path] = None,
    output_dir: str | Path = "output",
    source_sensor: Optional[str] = None,
    reference_sensor: Optional[str] = None,
    explicit_gsd1: Optional[float] = None,
    explicit_gsd2: Optional[float] = None,
    explicit_emission1: Optional[float] = None,
    explicit_emission2: Optional[float] = None,
    grid_size: int = 10,
    max_matches_per_cell: int = 4,
    patch_size_m: float = 160.0,  # Physical patch width in meters
    multimodal_pair: Optional[bool] = None,
    _is_inverted_call: bool = False,
    _cv_scale_ratio: Optional[float] = None,
    experimental_stack: bool = False,
) -> Dict[str, Any]:
    """
    Executes the primary cross-sensor registration pipeline:
    1. Metadata extraction & provenance (non-fatal: CV log-polar fallback).
    2. Common physical-GSD normalization (or relative CV scale normalization).
    3. DEM relief displacement compensation.
    4. Illumination-robust Phase Congruency structural feature extraction.
    5. Spatially distributed coarse matching.
    6. Physically scaled local Fourier Phase Correlation sub-pixel refinement.
    7. RANSAC verification with transformation quality gates.
    8. Complete output raster and metadata package.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Deterministic RANSAC sampling: with 5-7 inliers, unseeded RANSAC swings
    # fit RMSE by ~+-0.5px run to run (measured 003: 0.60/1.29 across runs).
    # Seeding changes nothing about expected quality; it makes published
    # numbers reproducible for evaluators re-running the pipeline.
    try:
        cv2.setRNGSeed(42)
    except Exception:
        pass

    # 1. Ingest metadata (NON-FATAL: missing GSD triggers the CV fallback below,
    # never a hard crash, so the pipeline stays generic per the SIH requirement).
    logger.info("Initializing registration pipeline: source='%s', reference='%s'", img_path1, img_path2)
    meta1: Optional[SensorMetadata] = None
    meta2: Optional[SensorMetadata] = None
    try:
        meta1 = extract_sensor_metadata(img_path1, source_sensor, explicit_gsd1, explicit_emission1)
    except Exception as e:
        logger.warning("Source metadata unavailable (%s); will attempt CV scale fallback.", e)
    try:
        meta2 = extract_sensor_metadata(img_path2, reference_sensor, explicit_gsd2, explicit_emission2)
    except Exception as e:
        logger.warning("Reference metadata unavailable (%s); will attempt CV scale fallback.", e)

    def _finite_gsd(m: Optional[SensorMetadata]) -> Optional[float]:
        g = getattr(m, "gsd_m", None)
        return float(g) if g is not None and np.isfinite(g) and float(g) > 0 else None

    def _placeholder_meta(declared: Optional[str], side: str) -> SensorMetadata:
        name = str(declared).strip().upper() if declared else "UNKNOWN"
        return SensorMetadata(
            sensor=name,
            gsd_m=None,  # type: ignore[assignment] -- unknown; _finite_gsd() treats as missing
            provenance={"sensor": "request" if declared else "unknown",
                        "gsd_m": "unavailable",
                        "note": f"{side} GSD missing; relative CV scale in use"},
        )

    # 2. Load images (needed by the CV fallback when metadata is missing)
    raw1_gray, raw1_color, raster_meta1 = load_as_float_and_color(img_path1)
    raw2_gray, raw2_color, raster_meta2 = load_as_float_and_color(img_path2)

    dem_arr = None
    if dem_path and Path(dem_path).exists():
        try:
            raw_dem = cv2.imread(str(dem_path), cv2.IMREAD_UNCHANGED)
            if raw_dem is not None:
                dem_arr = raw_dem.astype(np.float32)
                logger.info("DEM loaded successfully from '%s' (shape: %s)", dem_path, dem_arr.shape)
        except Exception as e:
            logger.warning("Failed to load DEM from '%s': %s", dem_path, e)
            dem_arr = None

    orig_h1, orig_w1 = raw1_gray.shape[:2]
    orig_h2, orig_w2 = raw2_gray.shape[:2]

    # 3. Common Physical GSD Normalization (metadata path) or relative CV
    # normalization (fallback path when GSD metadata is missing).
    gsd1 = _finite_gsd(meta1)
    gsd2 = _finite_gsd(meta2)
    scale_estimation_method = "pds4_metadata"
    estimated_scale_ratio: Optional[float] = None
    working_scale_note = "common_physical_gsd_normalization"

    if gsd1 is not None and gsd2 is not None:
        # Bring both images to the working physical scale (e.g. TMC-2 ~5.0 m/px)
        working_gsd = max(gsd1, gsd2)
        scale_factor1 = float(working_gsd / gsd1)  # e.g. 5.0 / 0.25 = 20.0
        scale_factor2 = float(working_gsd / gsd2)  # e.g. 5.0 / 5.0 = 1.0
        logger.info(
            "Sensor metadata: %s (%.2fm GSD) -> %s (%.2fm GSD). Target working scale: %.2fm/px (scales: %.1fx, %.1fx)",
            meta1.sensor, gsd1, meta2.sensor, gsd2, working_gsd, scale_factor1, scale_factor2
        )
    else:
        # --- CV LOG-POLAR FALLBACK ---
        # No absolute GSD: keep placeholder metas (gsd None, UNKNOWN sensor) so
        # downstream provenance/reporting code keeps working, and build the
        # working canvas relatively: the larger image stays fixed (scale 1.0)
        # while the smaller image is upscaled by the estimated ratio S, so
        # both canvases depict comparable ground sampling.
        if meta1 is None:
            meta1 = _placeholder_meta(source_sensor, "source")
        if meta2 is None:
            meta2 = _placeholder_meta(reference_sensor, "reference")
        scale_estimation_method = "cv_log_polar_fallback"
        working_gsd = 1.0  # nominal relative unit; absolute meters unknown
        working_scale_note = "cv_log_polar_relative_normalization"
        try:
            if _cv_scale_ratio is not None and np.isfinite(_cv_scale_ratio) \
                    and float(_cv_scale_ratio) >= 1.0:
                s_est = float(_cv_scale_ratio)
                logger.info("Reusing threaded CV scale ratio S=%.3f (inverted call).", s_est)
            else:
                s_est = estimate_scale_ratio_cv(raw1_gray, raw2_gray)
            estimated_scale_ratio = float(np.clip(s_est, 1.0, 300.0))
            if orig_h1 * orig_w1 >= orig_h2 * orig_w2:
                scale_factor1 = 1.0
                scale_factor2 = 1.0 / estimated_scale_ratio
            else:
                scale_factor1 = 1.0 / estimated_scale_ratio
                scale_factor2 = 1.0
            logger.info(
                "CV fallback scale: S=%.3f (larger image fixed, smaller upscaled %.3fx). "
                "Absolute GSD unknown; metric distances are in relative units.",
                estimated_scale_ratio, estimated_scale_ratio
            )
        except Exception as e:
            logger.error("CV log-polar scale fallback failed: %s", e)
            return {
                "status": "scale_estimation_failed",
                "message": (
                    "Unable to determine the inter-image scale ratio: PDS4/sensor "
                    "GSD metadata is missing and the CV log-polar fallback failed "
                    f"({e}). Provide 'explicit_gsd1'/'explicit_gsd2', attach PDS4 "
                    "XML labels, or use overlapping scenes with structural texture."
                ),
                "match_count": 0,
                "inlier_count": 0,
                "metrics": None,
                "homography": None,
                "metadata": {
                    "source": meta1.to_dict(),
                    "reference": meta2.to_dict(),
                    "scale_estimation_method": "failed",
                    "working_scale": {"working_gsd_m": None,
                                      "method": working_scale_note},
                },
            }

    # Metric GSD for absolute-RMSE reporting: defined once here so BOTH the
    # inverted-direction block below and the main-path metrics call see it.
    # (Fixes UnboundLocalError on every inverted/multimodal call.)
    metric_gsd: Optional[float] = working_gsd if scale_estimation_method == "pds4_metadata" else None

    work_w1 = int(round(orig_w1 / scale_factor1))
    work_h1 = int(round(orig_h1 / scale_factor1))
    work_w2 = int(round(orig_w2 / scale_factor2))
    work_h2 = int(round(orig_h2 / scale_factor2))

    # Reject if physical footprint is too small for multi-scale matching without altering pixel scale
    MIN_WORKING_DIM = 16
    if work_w1 < MIN_WORKING_DIM or work_h1 < MIN_WORKING_DIM or work_w2 < MIN_WORKING_DIM or work_h2 < MIN_WORKING_DIM:
        return {
            "status": "dimension_error",
            "message": (
                f"Normalized physical image dimensions too small for multi-scale matching "
                f"({work_w1}x{work_h1} px for source, {work_w2}x{work_h2} px for reference "
                f"at {working_gsd}m/px [{scale_estimation_method}]; "
                f"minimum required footprint is {MIN_WORKING_DIM}x{MIN_WORKING_DIM} px)."
            ),
            "match_count": 0,
            "inlier_count": 0,
            "metrics": None,
            "homography": None,
            "metadata": {
                "source": meta1.to_dict(),
                "reference": meta2.to_dict(),
                "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
            },
        }

    # Resample to working scale with area averaging
    work1_gray = cv2.resize(raw1_gray, (work_w1, work_h1), interpolation=cv2.INTER_AREA)
    work2_gray = cv2.resize(raw2_gray, (work_w2, work_h2), interpolation=cv2.INTER_AREA)

    # --- PHASE 1: ADAPTIVE ILLUMINATION NORMALIZATION (OPT-IN ONLY) ---
    # Measured 2026-09-10, full 8-region primary benchmark with the stack ON:
    # region_003 6@0.99 SUCCESS -> 0 inliers FAIL; triplet_new_2022 honest FAIL
    # -> 5@0.17 "success" (selection-bias consensus); all other fits shifted
    # unpredictably. LRO: 001 32/6@0.63, 003 Gate2-FAIL, 006 degraded.
    # Global on/off is unjustifiable either way: Phase 1 + RF stay behind
    # experimental_stack=True until validated per pair. Default path is the
    # committed classical behavior so published numbers reproduce exactly.
    try:
        from metadata import normalize_sensor_name as _raw_norm_s

        def _norm_s(x):
            return _raw_norm_s(x) if x else ""
    except Exception:
        def _norm_s(x):
            return str(x or "").strip().upper()
    _pair_sensors = {_norm_s(getattr(meta1, "sensor", "")),
                     _norm_s(getattr(meta2, "sensor", "")),
                     _norm_s(source_sensor), _norm_s(reference_sensor)}
    _classical_ohrc_tmc = not ({"IIRS", "LRO_NAC"} & _pair_sensors)
    _use_new_stack = bool(experimental_stack)
    if _use_new_stack:
        work1_gray, mask1 = adaptive_illumination_normalization(work1_gray)
        work2_gray, mask2 = adaptive_illumination_normalization(work2_gray)
    else:
        mask1 = mask2 = None
        logger.info("Phase 1 off (default classical path; opt in via experimental_stack=True).")

    # --- DYNAMIC SPATIAL GRID SCALING ---
    # Calculate grid size based on the smallest working dimension for INTERNAL
    # matching only. Canonical REPORTING always uses a fixed 10x10 grid via
    # compute_canonical_metrics(grid_size=10) so Before/After coverage numbers
    # are directly comparable. The dynamic grid is stored as
    # matching_grid_size in metrics for diagnostics.
    min_working_dim = min(work_w1, work_h1)
    dynamic_grid_size = max(4, min(20, int(min_working_dim / 256)))

    # Macro grid should be roughly half the density of the main grid (e.g., 2x2 for 4x4, 5x5 for 10x10)
    dynamic_macro_grid = max(2, dynamic_grid_size // 2)

    # Preserve canonical 10x10 for reporting; use dynamic grid only for matching.
    canonical_grid_size = 10
    matching_grid_size = dynamic_grid_size
    # Override the function arguments with dynamic values for internal processing
    # (We keep the function signature intact for API compatibility, but adapt internally)
    if grid_size == 10:  # Only override if it's the default value
        grid_size = dynamic_grid_size
    macro_grid = dynamic_macro_grid

    logger.info(
        "Dynamic Spatial Scaling: Image dim=%dx%d. Set grid_size=%dx%d, macro_grid=%dx%d (canonical reporting grid=10x10)",
        work_w1, work_h1, grid_size, grid_size, macro_grid, macro_grid
    )

    # Direction-Invariance Check for Multimodal (IIRS-involving) Pairs:
    # Always match using the LARGER working-scale canvas as Image 1 (template source).
    # If the caller requested the reverse direction, run in the better-conditioned
    # direction and return the inverted homography.
    sensors = {str(meta1.sensor).upper(), str(meta2.sensor).upper(), str(source_sensor).upper(), str(reference_sensor).upper()}
    if multimodal_pair is None:
        multimodal_pair = "IIRS" in sensors
    else:
        multimodal_pair = bool(multimodal_pair)

    area1 = work_w1 * work_h1
    area2 = work_w2 * work_h2

    if multimodal_pair and not _is_inverted_call and area2 > area1:
        inv_temp_dir = Path(output_dir) / "_inv_temp" if output_dir else None
        res_ba = match_images_cfog(
            img_path1=img_path2,
            img_path2=img_path1,
            dem_path=dem_path,
            output_dir=inv_temp_dir or output_dir,
            source_sensor=meta2.sensor,
            reference_sensor=meta1.sensor,
            explicit_gsd1=_finite_gsd(meta2),
            explicit_gsd2=_finite_gsd(meta1),
            explicit_emission1=meta2.emission_angle_deg,
            explicit_emission2=meta1.emission_angle_deg,
            grid_size=grid_size,
            max_matches_per_cell=max_matches_per_cell,
            patch_size_m=patch_size_m,
            multimodal_pair=multimodal_pair,
            _cv_scale_ratio=estimated_scale_ratio,
            _is_inverted_call=True,
            experimental_stack=experimental_stack,
        )

        if inv_temp_dir and inv_temp_dir.exists():
            shutil.rmtree(str(inv_temp_dir), ignore_errors=True)

        if res_ba.get("status") != "success" or res_ba.get("homography") is None:
            fail_meta = {
                "source": meta1.to_dict(),
                "reference": meta2.to_dict(),
                "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
                "direction": "inverted_from_BA",
                "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
            }
            return {
                "status": res_ba.get("status", "failed"),
                "message": f"Optimal-direction match ({meta2.sensor} -> {meta1.sensor}) produced: {res_ba.get('message')}",
                "direction": "inverted_from_BA",
                "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                "match_count": res_ba.get("match_count", 0),
                "inlier_count": res_ba.get("inlier_count", 0),
                "metrics": None,
                "homography": None,
                "metadata": fail_meta,
                "source": {"sensor": meta1.sensor, "width": orig_w1, "height": orig_h1, "gsd_m": _finite_gsd(meta1)},
                "reference": {"sensor": meta2.sensor, "width": orig_w2, "height": orig_h2, "gsd_m": _finite_gsd(meta2)},
                "working_scale": {"gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
                "matches": [],
                "all_matches": [],
                "outputs": {},
            }

        # Invert measured homography H_ba -> H_ab
        H_ba = np.array(res_ba["homography"], dtype=np.float64)
        try:
            H_ab = np.linalg.inv(H_ba)
            if abs(H_ab[2, 2]) > 1e-12:
                H_ab = H_ab / H_ab[2, 2]
        except np.linalg.LinAlgError:
            return {
                "status": "degenerate_matrix",
                "message": "Inversion of measured BA homography failed (singular matrix).",
                "direction": "inverted_from_BA",
                "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                "match_count": 0,
                "inlier_count": 0,
                "metrics": None,
                "homography": None,
                "metadata": {
                    "source": meta1.to_dict(),
                    "reference": meta2.to_dict(),
                    "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                    "scale_estimation_method": scale_estimation_method,
                    "estimated_scale_ratio": estimated_scale_ratio,
                    "direction": "inverted_from_BA",
                    "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                },
                "source": {"sensor": meta1.sensor, "width": orig_w1, "height": orig_h1, "gsd_m": _finite_gsd(meta1)},
                "reference": {"sensor": meta2.sensor, "width": orig_w2, "height": orig_h2, "gsd_m": _finite_gsd(meta2)},
                "working_scale": {"gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
                "matches": [],
                "all_matches": [],
                "outputs": {},
            }

        tx_check = verify_transformation_quality(H_ab, (orig_h2, orig_w2))
        if not tx_check["is_valid"]:
            return {
                "status": "geometric_verification_failed",
                "message": f"Inverted transformation rejected by quality gate: {tx_check['reason']}",
                "direction": "inverted_from_BA",
                "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                "match_count": res_ba.get("match_count", 0),
                "inlier_count": res_ba.get("inlier_count", 0),
                "metrics": None,
                "homography": None,
                "metadata": {
                    "source": meta1.to_dict(),
                    "reference": meta2.to_dict(),
                    "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                    "scale_estimation_method": scale_estimation_method,
                    "estimated_scale_ratio": estimated_scale_ratio,
                    "direction": "inverted_from_BA",
                    "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                },
                "source": {"sensor": meta1.sensor, "width": orig_w1, "height": orig_h1, "gsd_m": _finite_gsd(meta1)},
                "reference": {"sensor": meta2.sensor, "width": orig_w2, "height": orig_h2, "gsd_m": _finite_gsd(meta2)},
                "working_scale": {"gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
                "matches": [],
                "all_matches": [],
                "outputs": {},
            }

        # Invert correspondences: target of BA becomes source of AB, source of BA becomes target of AB
        inverted_all_matches = []
        pts1_list = []
        pts2_list = []
        for m in res_ba.get("all_matches", []):
            x1 = float(m["target_x"])
            y1 = float(m["target_y"])
            x2 = float(m["source_x"])
            y2 = float(m["source_y"])
            inv_m = {
                "source_x": x1,
                "source_y": y1,
                "target_x": x2,
                "target_y": y2,
                "image1_x": x1,
                "image1_y": y1,
                "image2_x": x2,
                "image2_y": y2,
                "confidence": float(m.get("confidence", 0.0)),
                "is_inlier": bool(m.get("is_inlier", False)),
                "is_refined": bool(m.get("is_refined", False)),
                "lk_refined": bool(m.get("lk_refined", False)),
            }
            inverted_all_matches.append(inv_m)
            pts1_list.append([x1, y1])
            pts2_list.append([x2, y2])

        pts1_arr = np.array(pts1_list, dtype=np.float32) if pts1_list else np.empty((0, 2), dtype=np.float32)
        pts2_arr = np.array(pts2_list, dtype=np.float32) if pts2_list else np.empty((0, 2), dtype=np.float32)
        inlier_mask_arr = np.array([1 if m.get("is_inlier") else 0 for m in res_ba.get("all_matches", [])], dtype=np.uint8)

        inlier_count = int(np.sum(inlier_mask_arr))
        if inlier_count < 4:
            return {
                "status": "geometric_verification_failed",
                "message": f"Optimal-direction match has fewer than 4 verified inliers ({inlier_count} found).",
                "direction": "inverted_from_BA",
                "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                "match_count": len(pts1_arr),
                "inlier_count": inlier_count,
                "metrics": None,
                "homography": None,
                "metadata": {
                    "source": meta1.to_dict(),
                    "reference": meta2.to_dict(),
                    "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                    "scale_estimation_method": scale_estimation_method,
                    "estimated_scale_ratio": estimated_scale_ratio,
                    "direction": "inverted_from_BA",
                    "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                },
                "source": {"sensor": meta1.sensor, "width": orig_w1, "height": orig_h1, "gsd_m": _finite_gsd(meta1)},
                "reference": {"sensor": meta2.sensor, "width": orig_w2, "height": orig_h2, "gsd_m": _finite_gsd(meta2)},
                "working_scale": {"gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
                "matches": [],
                "all_matches": inverted_all_matches,
                "outputs": {},
            }

        metrics = compute_canonical_metrics(
            pts1_arr, pts2_arr, inlier_mask_arr, H_ab, (orig_h2, orig_w2), canonical_grid_size,
            gsd_m=metric_gsd, dem_data=dem_arr
        )
        metrics["direction"] = "inverted_from_BA"
        metrics["measured_direction"] = f"{meta2.sensor} -> {meta1.sensor}"
        metrics["matching_grid_size"] = matching_grid_size
        metrics["canonical_grid_size"] = canonical_grid_size
        # Propagate illumination-compensation flags from the measured BA leg
        # so the required metrics.json keys survive homography inversion.
        try:
            _inner = (res_ba.get("metrics") or {})
            metrics["synthetic_reference_used"] = bool(_inner.get("synthetic_reference_used", False))
            metrics["illumination_compensation"] = _inner.get("illumination_compensation", "none")
            if "illumination_detail" in _inner:
                metrics["illumination_detail"] = _inner["illumination_detail"]
        except Exception:
            metrics.setdefault("synthetic_reference_used", False)
            metrics.setdefault("illumination_compensation", "none")

        fit_rmse = metrics.get("fit_rmse_px")
        tx_check = verify_transformation_quality(
            H_ab, (orig_h2, orig_w2), fit_rmse_px=fit_rmse
        )
        if not tx_check["is_valid"]:
            return {
                "status": "geometric_verification_failed",
                "message": f"Inverted transformation rejected by quality gate: {tx_check['reason']}",
                "direction": "inverted_from_BA",
                "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                "match_count": len(pts1_arr),
                "inlier_count": inlier_count,
                "metrics": None,
                "homography": None,
                "metadata": {
                    "source": meta1.to_dict(),
                    "reference": meta2.to_dict(),
                    "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                    "scale_estimation_method": scale_estimation_method,
                    "estimated_scale_ratio": estimated_scale_ratio,
                    "direction": "inverted_from_BA",
                    "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
                },
                "source": {"sensor": meta1.sensor, "width": orig_w1, "height": orig_h1, "gsd_m": _finite_gsd(meta1)},
                "reference": {"sensor": meta2.sensor, "width": orig_w2, "height": orig_h2, "gsd_m": _finite_gsd(meta2)},
                "working_scale": {"gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
                "matches": [],
                "all_matches": inverted_all_matches,
                "outputs": {},
            }

        if len(pts1_arr) >= 4:
            warped_source = warp_piecewise_affine(
                raw1_color, pts1_arr, pts2_arr, (orig_h2, orig_w2),
                tile_size=256, global_H=H_ab
            )
        else:
            warped_source = cv2.warpPerspective(
                raw1_color, H_ab, (orig_w2, orig_h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )

        tif_path = out_path / "registered_source.tif"
        written_tif = False
        try:
            import rasterio
            from rasterio.transform import from_origin
            profile = {
                "driver": "GTiff",
                "height": orig_h2,
                "width": orig_w2,
                "count": 3 if warped_source.ndim == 3 else 1,
                "dtype": "uint8",
                "nodata": 0,
                "crs": raster_meta2.get("crs") or "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs",
                "transform": raster_meta2.get("transform") or from_origin(0, orig_h2, tag_gsd, tag_gsd),
            }
            with rasterio.open(str(tif_path), "w", **profile) as dst:
                if warped_source.ndim == 3:
                    for b in range(3):
                        dst.write(warped_source[:, :, 2 - b], b + 1)
                else:
                    dst.write(warped_source, 1)
            written_tif = True
        except Exception:
            pass
        if not written_tif:
            cv2.imwrite(str(tif_path), warped_source)

        preview_path = out_path / "registered_preview.png"
        cv2.imwrite(str(preview_path), warped_source)

        block_size = 50
        # Vectorized checkerboard (identical output to the old nested loops).
        yy, xx = np.mgrid[0:orig_h2, 0:orig_w2]
        mask = ((xx // block_size) + (yy // block_size)) % 2 == 0
        blended = np.where(mask[..., None], warped_source, raw2_color)
        checker_path = out_path / "registered_checkerboard.png"
        cv2.imwrite(str(checker_path), blended)

        # Displacement-vector quiver QA (best effort; never fails the run).
        quiver_path = out_path / "registered_quiver.png"
        try:
            from quiver import create_displacement_quiver
            _qm = np.where(inlier_mask_arr.ravel() == 1)[0]
            if len(_qm) >= 3 and H_ab is not None:
                create_displacement_quiver(
                    pts1_arr[_qm], pts2_arr[_qm], np.asarray(H_ab, dtype=np.float64),
                    (orig_h2, orig_w2), quiver_path)
            else:
                quiver_path = None
        except Exception:
            quiver_path = None

        matches_path = out_path / "matches.json"
        dump_matches_json(
            [m for m in inverted_all_matches if m.get("is_inlier", False)],
            matches_path,
            indent=2,
        )

        metrics_path = out_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(sanitize_for_json(metrics), f, indent=2, cls=SubpixelJSONEncoder)

        transform_path = out_path / "transform.json"
        transform_data = {
            "model": "homography",
            "matrix": H_ab.tolist(),
            "quality": tx_check,
            "direction": "inverted_from_BA",
            "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
        }
        with open(transform_path, "w") as f:
            json.dump(sanitize_for_json(transform_data), f, indent=2, cls=SubpixelJSONEncoder)

        metadata_path = out_path / "metadata.json"
        full_metadata = {
            "source": meta1.to_dict(),
            "reference": meta2.to_dict(),
            "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
            "scale_estimation_method": scale_estimation_method,
            "estimated_scale_ratio": estimated_scale_ratio,
            "terrain_correction": {"source": None, "reference": None},
            "direction": "inverted_from_BA",
            "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
            "provenance": {
                "source_path": str(img_path1),
                "reference_path": str(img_path2),
                "dem_path": str(dem_path) if dem_path else None,
                "matcher": "CFOG_PhaseCongruency_v2.0",
                "direction": "inverted_from_BA",
                "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
            },
        }
        with open(metadata_path, "w") as f:
            json.dump(sanitize_for_json(full_metadata), f, indent=2, cls=SubpixelJSONEncoder)

        return {
            "status": "success",
            "direction": "inverted_from_BA",
            "measured_direction": f"{meta2.sensor} -> {meta1.sensor}",
            "source": {"sensor": meta1.sensor, "width": orig_w1, "height": orig_h1, "gsd_m": _finite_gsd(meta1)},
            "reference": {"sensor": meta2.sensor, "width": orig_w2, "height": orig_h2, "gsd_m": _finite_gsd(meta2)},
            "working_scale": {"gsd_m": working_gsd, "method": working_scale_note},
            "scale_estimation_method": scale_estimation_method,
            "estimated_scale_ratio": estimated_scale_ratio,
            "metrics": metrics,
            "homography": H_ab.tolist(),
            "terrain_correction": full_metadata["terrain_correction"],
            "spatial_attempts": res_ba.get("spatial_attempts", 0),
            "matches": [m for m in inverted_all_matches if m.get("is_inlier", False)],
            "all_matches": inverted_all_matches,
            "outputs": {
                "registered_raster": str(tif_path),
                "preview": str(preview_path),
                "checkerboard": str(checker_path),
                "quiver": str(quiver_path) if quiver_path is not None else None,
                "matches": str(matches_path),
                "metrics": str(metrics_path),
                "transform": str(transform_path),
                "metadata": str(metadata_path),
            },
        }

    # 4. DEM Relief Compensation
    # NOTE: dem_arr was already raw-loaded above via cv2.imread (elevation meters
    # preserved). Only (re)load here if still None, preferring raw meters so the
    # downstream absolute_rmse_m stays physically meaningful. Null-safe: stays None
    # when no DEM is supplied.
    if dem_arr is None and dem_path and os.path.exists(str(dem_path)):
        try:
            raw_dem_reload = cv2.imread(str(dem_path), cv2.IMREAD_UNCHANGED)
            if raw_dem_reload is not None:
                dem_arr = raw_dem_reload.astype(np.float32)
            else:
                dem_arr, _, _ = load_as_float_and_color(dem_path)
        except Exception:
            dem_arr = None

    # NOTE: sun_azimuth_deg is solar illumination geometry, NOT sensor line-of-sight
    # azimuth. DEM relief parallax must use emission (off-nadir) geometry. We do
    # not have per-product sensor LOS azimuth in metadata, so pass None here
    # (defaults to 45deg in apply_dem_relief_compensation) and keep sun angles
    # only as provenance for illumination-robustness auditing, not for DEM shifts.
    comp1_gray, terrain_info1 = apply_dem_relief_compensation(
        work1_gray, dem_arr, meta1.emission_angle_deg, None, working_gsd
    )
    comp2_gray, terrain_info2 = apply_dem_relief_compensation(
        work2_gray, dem_arr, meta2.emission_angle_deg, None, working_gsd
    )
    for _ti, _meta in ((terrain_info1, meta1), (terrain_info2, meta2)):
        _ti["sun_azimuth_deg_not_used_for_dem"] = _meta.sun_azimuth_deg
        _ti["sun_azimuth_provenance"] = _meta.provenance.get("sun_azimuth_deg")

    # 4b. Sun-Angle-Invariant Intercept (DEM hillshade projection).
    # Quality Gate 3 correctly rejects severe sun-angle mismatches (e.g. 162deg
    # azimuth flip inverts crater-rim shadows and breaks Phase Congruency).
    # When delta_azimuth > 90deg, render a synthetic shaded relief from the DEM
    # under the SOURCE sun geometry and match against its Phase Congruency
    # instead of the raw reference image.
    delta_azimuth = sun_azimuth_delta_deg(meta1.sun_azimuth_deg, meta2.sun_azimuth_deg)
    synthetic_reference_used = False
    illumination_compensation = "none"
    illumination_detail: Dict[str, Any] = {
        "delta_azimuth_deg": delta_azimuth,
        "threshold_deg": 90.0,
        "source_azimuth_deg": meta1.sun_azimuth_deg,
        "reference_azimuth_deg": meta2.sun_azimuth_deg,
    }
    # Reference-domain image actually fed to PC / sub-pixel refinement.
    # Defaults to the real reference; swapped for synthetic hillshade on trigger.
    match_ref_gray = comp2_gray
    if delta_azimuth is not None and delta_azimuth > 90.0:
        illumination_detail["triggered"] = True
        if dem_arr is not None and meta1.sun_azimuth_deg is not None:
            try:
                src_az = float(meta1.sun_azimuth_deg)
                src_el = resolve_sun_elevation_deg(meta1)
                dem_work2 = cv2.resize(
                    np.asarray(dem_arr, dtype=np.float32),
                    (work_w2, work_h2),
                    interpolation=cv2.INTER_LINEAR,
                )
                synth_gray = render_synthetic_shaded_relief(
                    dem_work2, src_az, src_el, working_gsd_m=working_gsd
                )
                match_ref_gray = np.clip(synth_gray, 0.0, 1.0).astype(np.float32)
                synthetic_reference_used = True
                illumination_compensation = "dem_hillshade_projection"
                illumination_detail.update({
                    "synthetic_azimuth_deg": src_az,
                    "synthetic_elevation_deg": src_el,
                    "dem_resampled_to": [work_h2, work_w2],
                })
                logger.info(
                    "Illumination intercept: delta_azimuth=%.1fdeg > 90deg; "
                    "using synthetic DEM hillshade (src az=%.1f, el=%.1f) as PC reference.",
                    delta_azimuth, src_az, src_el,
                )
            except Exception as e:
                logger.warning("Synthetic hillshade projection failed (%s); using raw reference.", e)
                illumination_detail["synthetic_error"] = str(e)
        else:
            illumination_detail["skipped_reason"] = (
                "DEM or source sun-azimuth unavailable; cannot synthesize reference"
            )
            logger.warning(
                "Illumination intercept triggered (delta_azimuth=%.1fdeg) but %s.",
                delta_azimuth, illumination_detail["skipped_reason"],
            )
    else:
        illumination_detail["triggered"] = False

    # 5. Phase Congruency (Illumination-Robust Structural Features)
    # NOTE (compute saving): multi_scale_phase_congruency() builds a 3-level
    # pyramid but only Level 0 was ever consumed — identical output at ~3x the
    # FFT cost. Call the base resolution directly (Render free tier: 512MB);
    # the pyramid helper stays for future coarse-to-fine work (a coarse-seeded
    # guided pass was trialed 2026-09-10 with zero gain).
    # In synthetic mode pc2 comes from the hillshade map, not the raw reference.
    pc1 = compute_phase_congruency(comp1_gray, num_orientations=4, num_scales=3)
    pc2 = compute_phase_congruency(match_ref_gray, num_orientations=4, num_scales=3)

    # 6. Spatially Distributed Coarse Matching (Symmetric Scale-Aware Sizing)
    min_work_w = min(work_w1, work_w2)
    min_work_h = min(work_h1, work_h2)

    cell_w = work_w1 / float(grid_size)
    cell_h = work_h1 / float(grid_size)

    coarse_matches = []
    attempted_cells = set()
    sensors = {str(source_sensor).upper(), str(reference_sensor).upper()}
    if multimodal_pair is None:
        multimodal_pair = "IIRS" in sensors
    # Key half-patch size to preserve Phase Congruency Log-Gabor support
    half_patch_c = max(4 if multimodal_pair else 8, int(round((patch_size_m / working_gsd) / 4.0)))

    # Search window in image 2 — wider for multimodal to compensate for IIRS's coarse resolution
    if multimodal_pair:
        search_half_w = max(16, work_w2 // 3)
        search_half_h = max(16, work_h2 // 3)
    else:
        search_half_w = max(16, work_w2 // 6)
        search_half_h = max(16, work_h2 // 6)

    # --- 6a. Pre-match Spatial Suppression (ANMS / SSC, Bailo et al. 2018) ---
    # Detect candidate salient keypoints in BOTH images and apply Suppression via Square Covering (SSC)
    # so keypoints are selected for maximum spatial spread across the scene rather than clustering on single crater rims.
    kps1_raw = detect_salient_keypoints(pc1, max_corners=500, quality_level=0.01)
    kps2_raw = detect_salient_keypoints(pc2, max_corners=500, quality_level=0.01)

    target_ssc = max(36, min(100, grid_size * grid_size * 2))
    kps1_ssc = suppression_via_square_covering(
        kps1_raw, num_ret_points=target_ssc, tolerance=0.15, cols=work_w1, rows=work_h1
    )
    kps2_ssc = suppression_via_square_covering(
        kps2_raw, num_ret_points=target_ssc, tolerance=0.15, cols=work_w2, rows=work_h2
    )

    logger.info(
        "Pre-match SSC keypoint selection: Image 1: %d -> %d; Image 2: %d -> %d",
        len(kps1_raw), len(kps1_ssc), len(kps2_raw), len(kps2_ssc),
    )

    # Correlation matching on spatially uniform pre-match SSC keypoints
    # NOTE: kx/ky may be sub-pixel (goodFeaturesToTrack floats). Keep the
    # full float for reported work_x1/work_y1; integers are slicing-only.
    for kx, ky, _ in kps1_ssc:
        kx_f, ky_f = float(kx), float(ky)
        cx = int(round(kx_f))
        cy = int(round(ky_f))

        if (
            cy < half_patch_c
            or cy >= work_h1 - half_patch_c
            or cx < half_patch_c
            or cx >= work_w1 - half_patch_c
        ):
            continue

        tmpl = pc1[cy - half_patch_c : cy + half_patch_c, cx - half_patch_c : cx + half_patch_c]
        if float(np.std(tmpl)) < 1e-4:
            continue

        cx2 = int(cx * (work_w2 / float(work_w1)))
        cy2 = int(cy * (work_h2 / float(work_h1)))

        s_min_x = max(0, cx2 - search_half_w)
        s_max_x = min(work_w2, cx2 + search_half_w)
        s_min_y = max(0, cy2 - search_half_h)
        s_max_y = min(work_h2, cy2 + search_half_h)

        search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
        if (
            search_region.shape[0] <= tmpl.shape[0]
            or search_region.shape[1] <= tmpl.shape[1]
            or float(np.std(search_region)) < 1e-4
        ):
            continue

        res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if multimodal_pair:
            candidate = search_region[
                max_loc[1] : max_loc[1] + tmpl.shape[0],
                max_loc[0] : max_loc[0] + tmpl.shape[1],
            ]
            if candidate.shape == tmpl.shape:
                max_val = mutual_information_score(tmpl, candidate)

        if max_val > (0.05 if multimodal_pair else 0.35):
            best_x2 = s_min_x + max_loc[0] + half_patch_c
            best_y2 = s_min_y + max_loc[1] + half_patch_c
            gx = min(grid_size - 1, int(kx_f / max(cell_w, 1e-6)))
            gy = min(grid_size - 1, int(ky_f / max(cell_h, 1e-6)))
            coarse_matches.append({
                "work_x1": float(kx_f),
                "work_y1": float(ky_f),
                "work_x2": float(best_x2),
                "work_y2": float(best_y2),
                "score": float(max_val),
                "cell": (gx, gy),
                "method": "ssc_patch",
            })

    # --- 2b. For multimodal pairs, run centroid matching (with SSC on centroids) ---
    if multimodal_pair:
        centroids1 = detect_blob_centroids(pc1, min_area=2)
        centroids2 = detect_blob_centroids(pc2, min_area=2)
        if len(centroids1) > 24:
            c1_tuples = [(float(c[0]), float(c[1]), float(pc1[min(work_h1 - 1, int(c[1])), min(work_w1 - 1, int(c[0]))])) for c in centroids1]
            c1_ssc = suppression_via_square_covering(c1_tuples, num_ret_points=min(40, len(centroids1)), tolerance=0.1, cols=work_w1, rows=work_h1)
            centroids1 = np.array([[c[0], c[1]] for c in c1_ssc], dtype=np.float32)
        if len(centroids2) > 24:
            c2_tuples = [(float(c[0]), float(c[1]), float(pc2[min(work_h2 - 1, int(c[1])), min(work_w2 - 1, int(c[0]))])) for c in centroids2]
            c2_ssc = suppression_via_square_covering(c2_tuples, num_ret_points=min(40, len(centroids2)), tolerance=0.1, cols=work_w2, rows=work_h2)
            centroids2 = np.array([[c[0], c[1]] for c in c2_ssc], dtype=np.float32)

        max_distance = max(search_half_w, search_half_h)
        used_c2 = set()
        for x1, y1 in centroids1:
            expected = np.array([x1 * work_w2 / float(work_w1), y1 * work_h2 / float(work_h1)])
            if len(centroids2) == 0:
                break
            distances = np.linalg.norm(centroids2 - expected, axis=1)
            order = np.argsort(distances)
            for nearest in order:
                if int(nearest) in used_c2:
                    continue
                if float(distances[nearest]) > max_distance:
                    break
                x2, y2 = centroids2[nearest]
                # Validate with MI on local patches around centroids
                hpc = min(half_patch_c, min(int(x1), int(y1), int(x2), int(y2),
                          work_w1 - int(x1) - 1, work_h1 - int(y1) - 1,
                          work_w2 - int(x2) - 1, work_h2 - int(y2) - 1))
                if hpc >= 3:
                    p1 = pc1[int(y1) - hpc:int(y1) + hpc, int(x1) - hpc:int(x1) + hpc]
                    p2 = pc2[int(y2) - hpc:int(y2) + hpc, int(x2) - hpc:int(x2) + hpc]
                    if p1.size > 0 and p2.size > 0 and p1.shape == p2.shape:
                        mi_score = mutual_information_score(p1, p2)
                    else:
                        mi_score = 0.0
                else:
                    mi_score = float(1.0 / (1.0 + distances[nearest]))
                if mi_score > 0.03:
                    cell = (
                        min(grid_size - 1, int(x1 / max(cell_w, 1e-6))),
                        min(grid_size - 1, int(y1 / max(cell_h, 1e-6))),
                    )
                    coarse_matches.append({
                        "work_x1": float(x1), "work_y1": float(y1),
                        "work_x2": float(x2), "work_y2": float(y2),
                        "score": float(mi_score),
                        "cell": cell,
                        "method": "centroid",
                    })
                    used_c2.add(int(nearest))
                    break

    # --- Standard grid-based patch matching (primary for same-sensor, augments centroids for multimodal) ---
    for gy in range(grid_size):
        for gx in range(grid_size):
            attempted_cells.add((gx, gy))
            cx = int((gx + 0.5) * cell_w)
            cy = int((gy + 0.5) * cell_h)

            if (
                cy < half_patch_c
                or cy >= work_h1 - half_patch_c
                or cx < half_patch_c
                or cx >= work_w1 - half_patch_c
            ):
                continue

            tmpl = pc1[cy - half_patch_c : cy + half_patch_c, cx - half_patch_c : cx + half_patch_c]
            if float(np.std(tmpl)) < 1e-4:
                continue

            cx2 = int(cx * (work_w2 / float(work_w1)))
            cy2 = int(cy * (work_h2 / float(work_h1)))

            s_min_x = max(0, cx2 - search_half_w)
            s_max_x = min(work_w2, cx2 + search_half_w)
            s_min_y = max(0, cy2 - search_half_h)
            s_max_y = min(work_h2, cy2 + search_half_h)

            search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
            if (
                search_region.shape[0] <= tmpl.shape[0]
                or search_region.shape[1] <= tmpl.shape[1]
                or float(np.std(search_region)) < 1e-4
            ):
                continue

            res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # 2c. For multimodal pairs, use MI as the primary ranking criterion
            if multimodal_pair:
                candidate = search_region[
                    max_loc[1] : max_loc[1] + tmpl.shape[0],
                    max_loc[0] : max_loc[0] + tmpl.shape[1],
                ]
                if candidate.shape == tmpl.shape:
                    max_val = mutual_information_score(tmpl, candidate)

            if max_val > (0.05 if multimodal_pair else 0.35):
                best_x2 = s_min_x + max_loc[0] + half_patch_c
                best_y2 = s_min_y + max_loc[1] + half_patch_c
                coarse_matches.append({
                    "work_x1": float(cx),
                    "work_y1": float(cy),
                    "work_x2": float(best_x2),
                    "work_y2": float(best_y2),
                    "score": float(max_val),
                    "cell": (gx, gy),
                    "method": "patch",
                })

    # --- Step 2: Post-match Grid Density Budgeting (NxN grid, 10x10) ---
    # Prioritizes keeping candidates from under-represented cells before dense cells
    # receive additional candidate allocations (tiered round-robin).
    selected_matches = apply_grid_density_budgeting(
        coarse_matches,
        image_shape=(work_h1, work_w1),
        grid_dims=(10, 10),
        max_per_cell=max_matches_per_cell,
    )

    # --- Item 4: 4x4 Mandatory Macro-Cell Coverage Enforcement ---
    # Divide source image into 4x4 macro-cells and fill gaps with relaxed-threshold searches.
    macro_cell_w = work_w1 / float(macro_grid)
    macro_cell_h = work_h1 / float(macro_grid)
    occupied_macro: set = set()
    for m in selected_matches:
        mc_x = min(macro_grid - 1, int(m["work_x1"] / max(macro_cell_w, 1e-6)))
        mc_y = min(macro_grid - 1, int(m["work_y1"] / max(macro_cell_h, 1e-6)))
        occupied_macro.add((mc_x, mc_y))

    mandatory_fill_count = 0
    relaxed_ncc = 0.20 if not multimodal_pair else 0.03
    for mc_y in range(macro_grid):
        for mc_x in range(macro_grid):
            if (mc_x, mc_y) in occupied_macro:
                continue
            # Attempt a match at the macro-cell center with relaxed threshold
            cx = int((mc_x + 0.5) * macro_cell_w)
            cy = int((mc_y + 0.5) * macro_cell_h)
            if cy < half_patch_c or cy >= work_h1 - half_patch_c or cx < half_patch_c or cx >= work_w1 - half_patch_c:
                continue
            tmpl = pc1[cy - half_patch_c : cy + half_patch_c, cx - half_patch_c : cx + half_patch_c]
            if float(np.std(tmpl)) < 1e-5:
                continue
            cx2 = int(cx * (work_w2 / float(work_w1)))
            cy2 = int(cy * (work_h2 / float(work_h1)))
            s_min_x = max(0, cx2 - search_half_w)
            s_max_x = min(work_w2, cx2 + search_half_w)
            s_min_y = max(0, cy2 - search_half_h)
            s_max_y = min(work_h2, cy2 + search_half_h)
            search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
            if search_region.shape[0] <= tmpl.shape[0] or search_region.shape[1] <= tmpl.shape[1]:
                continue
            res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if multimodal_pair:
                candidate = search_region[max_loc[1]:max_loc[1]+tmpl.shape[0], max_loc[0]:max_loc[0]+tmpl.shape[1]]
                if candidate.shape == tmpl.shape:
                    max_val = mutual_information_score(tmpl, candidate)
            if max_val > relaxed_ncc:
                best_x2 = s_min_x + max_loc[0] + half_patch_c
                best_y2 = s_min_y + max_loc[1] + half_patch_c
                fine_cell = (min(grid_size - 1, int(cx / max(cell_w, 1e-6))), min(grid_size - 1, int(cy / max(cell_h, 1e-6))))
                selected_matches.append({
                    "work_x1": float(cx), "work_y1": float(cy),
                    "work_x2": float(best_x2), "work_y2": float(best_y2),
                    "score": float(max_val),
                    "cell": fine_cell,
                    "method": "mandatory_fill",
                })
                occupied_macro.add((mc_x, mc_y))
                mandatory_fill_count += 1

    spatial_attempts = {
        "grid_size": grid_size,
        "attempted_cells": len(attempted_cells),
        "total_cells": grid_size * grid_size,
        "attempted_coverage": len(attempted_cells) / float(grid_size * grid_size),
        "matched_cells": len({m["cell"] for m in selected_matches if "cell" in m}),
        "macro_grid": macro_grid,
        "macro_cells_occupied": len(occupied_macro),
        "mandatory_fill_count": mandatory_fill_count,
    }

    # Enforce Spatial Uniformity via Grid-based Non-Maximum Suppression & Density Budgeting (10x10)
    selected_matches = apply_grid_nms(
        selected_matches,
        image_shape=(work_h1, work_w1),
        grid_dims=(10, 10),
        max_per_cell=max_matches_per_cell,
    )

    # QUALITY GATE 1: Insufficient Genuine Matches
    # ZERO FAKE CORRESPONDENCES ALLOWED. Fail cleanly if real matches < 4.
    if len(selected_matches) < 4:
        logger.warning(
            "Quality Gate 1 Rejected: Insufficient verified correspondences (%d < 4). Aborting without synthetic fallback.",
            len(selected_matches),
        )
        return {
            "status": "insufficient_correspondences",
            "message": f"Insufficient verified correspondences for geometric registration (found {len(selected_matches)}, minimum required is 4).",
            "match_count": len(selected_matches),
            "inlier_count": 0,
            "metrics": None,
            "homography": None,
            "spatial_attempts": spatial_attempts,
            "metadata": {
                "source": meta1.to_dict(),
                "reference": meta2.to_dict(),
                "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
            },
        }

    logger.info("Quality Gate 1 Passed: %d candidate matches retained across grid.", len(selected_matches))

    # 7. Local Fourier Phase Correlation Sub-Pixel Refinement
    patch_size_work = max(16, int(round(patch_size_m / working_gsd)))
    half_p = patch_size_work // 2

    native_pts1 = []
    native_pts2 = []
    refinement_records = []

    for m in selected_matches:
        # Preserve sub-pixel precision: keep working-space coords as float
        # for all native-space math. Integer pixels are used ONLY for array
        # slicing, never for the reported coordinates.
        wx1_f, wy1_f = float(m["work_x1"]), float(m["work_y1"])
        wx2_f, wy2_f = float(m["work_x2"]), float(m["work_y2"])
        wx1, wy1 = int(round(wx1_f)), int(round(wy1_f))
        wx2, wy2 = int(round(wx2_f)), int(round(wy2_f))

        ref_dx, ref_dy = 0.0, 0.0
        refined = False

        if (
            wy1 >= half_p
            and wy1 < work_h1 - half_p
            and wx1 >= half_p
            and wx1 < work_w1 - half_p
            and wy2 >= half_p
            and wy2 < work_h2 - half_p
            and wx2 >= half_p
            and wx2 < work_w2 - half_p
        ):
            p1 = comp1_gray[wy1 - half_p : wy1 + half_p, wx1 - half_p : wx1 + half_p]
            # In synthetic mode match_ref_gray IS the DEM hillshade rendered
            # under the source sun angles, keeping refinement in the same
            # domain as the coarse PC matching.
            p2 = match_ref_gray[wy2 - half_p : wy2 + half_p, wx2 - half_p : wx2 + half_p]

            dx, dy, peak, valid = subpixel_phase_correlation(p1, p2)
            if valid:
                ref_dx, ref_dy = float(dx), float(dy)
                refined = True

        # Map working-scale coordinates back to NATIVE sensor pixel spaces
        # using the full-precision floats (never the truncated ints).
        nat_x1 = float(wx1_f * scale_factor1)
        nat_y1 = float(wy1_f * scale_factor1)
        nat_x2 = float((wx2_f + ref_dx) * scale_factor2)
        nat_y2 = float((wy2_f + ref_dy) * scale_factor2)

        native_pts1.append([nat_x1, nat_y1])
        native_pts2.append([nat_x2, nat_y2])

        # Coordinates: full float() precision (no round/int). Only
        # confidence may be rounded (2 decimals for readability).
        refinement_records.append(make_match_record(
            nat_x1, nat_y1, nat_x2, nat_y2, m["score"],
            refinement_dx=float(ref_dx),
            refinement_dy=float(ref_dy),
            is_refined=bool(refined),
        ))

    # Stable per-record identity (Fix P0-2): the AI verifier may drop records,
    # so NOTHING downstream may assume refinement_records[i] corresponds to
    # RANSAC position i. match_id + direct "cell" attach (kept as tuples for
    # Counter hashing; NOT routed through sanitize_for_json) make the mapping
    # explicit. Verified entries are the same dict objects, so identity-based
    # updates below always land on the right records.
    for _i, (_rec, _m) in enumerate(zip(refinement_records, selected_matches)):
        _rec.setdefault("match_id", _i)
        _rec["cell"] = _m.get("cell")

    # --- PHASE 4: AI MATCH VERIFICATION (OPT-IN ONLY) ---
    # Measured 2026-09-10: with the stack ON, all 3 real-CDR LRO pairs
    # Gate2-FAIL (RF vetoes true low-MI matches pre-RANSAC) and the primary
    # benchmark flips outcomes both ways (003 success->fail, triplet_new_2022
    # honest-fail->5@0.17 pseudo-success). Same opt-in rule as Phase 1:
    # default path passes refinement records straight to RANSAC.
    from ai_verifier import AIMatchVerifier
    try:
        from metadata import normalize_sensor_name as _raw_norm_s4

        def _norm_s4(x):
            return _raw_norm_s4(x) if x else ""
    except Exception:
        def _norm_s4(x):
            return str(x or "").strip().upper()
    _s4 = {_norm_s4(getattr(meta1, "sensor", "")),
           _norm_s4(getattr(meta2, "sensor", "")),
           _norm_s4(source_sensor), _norm_s4(reference_sensor)}
    _use_verifier = bool(experimental_stack)
    if _use_verifier:
        verifier = AIMatchVerifier()
        verified_matches, rejected_matches = verifier.filter_matches(refinement_records, threshold=0.5)
    else:
        verified_matches, rejected_matches = list(refinement_records), []
        logger.info("AI Verifier off (default classical path; opt in via experimental_stack=True).")

    # Update the points array based on verified matches
    if len(verified_matches) >= 4:
        native_pts1 = [[m["source_x"], m["source_y"]] for m in verified_matches]
        native_pts2 = [[m["target_x"], m["target_y"]] for m in verified_matches]
        logger.info(f"AI Verifier passed {len(verified_matches)} matches to RANSAC.")
    else:
        logger.warning("AI Verifier rejected too many matches. Falling back to original points.")

    pts1_arr = np.array(native_pts1, dtype=np.float32)
    pts2_arr = np.array(native_pts2, dtype=np.float32)

    # 8. Robust Geometric Estimation (RANSAC)
    # NOTE: azimuth for DEM-aware fitting must be sensor line-of-sight azimuth,
    # never sun azimuth (see relief-compensation fix above). LOS azimuth is
    # currently unavailable, so pass None and let the helper use its default.
    em1 = meta1.emission_deg if meta1.emission_deg is not None else 0.0
    if dem_arr is not None and abs(em1) > 1e-2:
        H_final, inlier_mask, _ = ransac_dem_aware_fit(
            pts1_arr, pts2_arr, dem=dem_arr, emission_deg=em1,
            azimuth_deg=None, gsd_m=working_gsd
        )
    else:
        # --- PHASE 7: WEIGHTED (PROSAC-style) RANSAC ---
        # High-confidence matches are sampled with higher probability and pull
        # the final refinement more strongly via weighted DLT. Forward-compatible:
        # tries native cv2 weights kwarg first, else manual weighted sampling.
        # NOTE: verified/refinement records carry "confidence" (not "score").
        try:
            if len(verified_matches) >= 4 and len(verified_matches) == len(pts1_arr):
                _w_src = verified_matches
            elif len(refinement_records) == len(pts1_arr):
                _w_src = refinement_records
            else:
                _w_src = []
            if len(_w_src) == len(pts1_arr) and len(pts1_arr) >= 4:
                weights = np.array(
                    [float(m.get("confidence", m.get("score", 0.5))) for m in _w_src],
                    dtype=np.float64,
                )
                weights = np.nan_to_num(weights, nan=0.5, posinf=1.0, neginf=0.0)
                weights = np.clip(weights, 0.0, None)
                if float(np.max(weights)) > 0:
                    weights = weights / float(np.max(weights))
                else:
                    weights = np.full_like(weights, 0.5)
            else:
                weights = np.full(len(pts1_arr), 0.5, dtype=np.float64)

            try:
                # Forward-compat: newer OpenCV builds accept per-point weights.
                H_final, inlier_mask = cv2.findHomography(
                    pts1_arr, pts2_arr, cv2.RANSAC, ransacReprojThreshold=5.0,
                    weights=weights.astype(np.float32),
                )
                logger.info("Weighted RANSAC applied successfully (native weights).")
            except TypeError:
                # OpenCV 4.x has no weights kwarg: manual confidence-weighted sampling.
                logger.info("Native weighted RANSAC unavailable; using confidence-weighted sampling RANSAC.")
                _rng = np.random.default_rng(42)
                _p = weights / max(float(np.sum(weights)), 1e-12)
                _n = len(pts1_arr)
                _best_H, _best_inliers, _best_score = None, None, -1.0
                for _ in range(2000):
                    _idx = _rng.choice(_n, size=4, replace=False, p=_p)
                    _H_cand, _ = cv2.findHomography(pts1_arr[_idx], pts2_arr[_idx], 0)
                    if _H_cand is None or not np.all(np.isfinite(_H_cand)):
                        continue
                    _proj = cv2.perspectiveTransform(pts1_arr.reshape(-1, 1, 2), _H_cand).reshape(-1, 2)
                    _err = np.linalg.norm(_proj - pts2_arr, axis=1)
                    _inl = _err <= 5.0
                    if int(np.sum(_inl)) < 4:
                        continue
                    _score = float(np.sum(weights[_inl]))
                    if _score > _best_score:
                        _best_score, _best_H, _best_inliers = _score, _H_cand, _inl
                if _best_H is not None:
                    # Weighted DLT refinement on inliers (sqrt(w) row scaling + SVD).
                    _ii = np.where(_best_inliers)[0]
                    _sw = np.sqrt(weights[_ii])
                    _A = []
                    for _k, _i in enumerate(_ii):
                        _x, _y = float(pts1_arr[_i, 0]), float(pts1_arr[_i, 1])
                        _xp, _yp = float(pts2_arr[_i, 0]), float(pts2_arr[_i, 1])
                        _s = float(_sw[_k])
                        _A.append([-_x * _s, -_y * _s, -_s, 0, 0, 0, _xp * _x * _s, _xp * _y * _s, _xp * _s])
                        _A.append([0, 0, 0, -_x * _s, -_y * _s, -_s, _yp * _x * _s, _yp * _y * _s, _yp * _s])
                    _, _, _Vt = np.linalg.svd(np.asarray(_A, dtype=np.float64))
                    _H_ref = _Vt[-1].reshape(3, 3)
                    if abs(_H_ref[2, 2]) > 1e-12:
                        _H_ref = _H_ref / _H_ref[2, 2]
                    H_final = _H_ref
                    inlier_mask = _best_inliers.reshape(-1, 1).astype(np.uint8)
                    logger.info("Weighted RANSAC applied successfully (sampling + weighted DLT).")
                else:
                    H_final, inlier_mask = cv2.findHomography(
                        pts1_arr, pts2_arr, cv2.RANSAC, ransacReprojThreshold=5.0
                    )
        except Exception as e:
            logger.warning("Weighted RANSAC failed (%s). Falling back to standard RANSAC.", e)
            H_final, inlier_mask = cv2.findHomography(
                pts1_arr, pts2_arr, cv2.RANSAC, ransacReprojThreshold=5.0
            )

    if H_final is not None and inlier_mask is not None and np.sum(inlier_mask) >= 4:
        try:
            det = float(np.linalg.det(H_final))
            if det <= 1e-4:
                H_final = None
        except Exception:
            H_final = None

    if H_final is None or inlier_mask is None or np.sum(inlier_mask) < 4:
        # Try affine transformation if perspective fails or is reflective
        H_aff, inlier_mask = cv2.estimateAffinePartial2D(pts1_arr, pts2_arr)
        if H_aff is not None and inlier_mask is not None and np.sum(inlier_mask) >= 4:
            H_final = np.vstack([H_aff, [0.0, 0.0, 1.0]])
        else:
            # QUALITY GATE 2: Geometric Verification Failed
            # ZERO IDENTITY MATRIX FALLBACKS ALLOWED. Report failure cleanly.
            logger.error("Quality Gate 2 Rejected: Robust geometric estimation failed to find consistent transformation.")
            return {
                "status": "geometric_verification_failed",
                "message": "Robust geometric verification failed to estimate a valid transformation from verified correspondences.",
                "match_count": len(pts1_arr),
                "inlier_count": int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
                "metrics": None,
                "homography": None,
            "metadata": {
                "source": meta1.to_dict(),
                "reference": meta2.to_dict(),
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
                "working_scale": {"working_gsd_m": working_gsd,
                                  "method": working_scale_note},
            },
        }

    # Effective GSD for metric reporting: None in CV-fallback mode so that
    # absolute RMSE in meters is reported as unavailable (relative units only).
    metric_gsd: Optional[float] = working_gsd if scale_estimation_method == "pds4_metadata" else None
    # NaN-safe GSD for GeoTIFF geotransform tags (relative grid when unknown).
    tag_gsd = working_gsd if np.isfinite(working_gsd) else 1.0

    # QUALITY GATE 3: Sanity Check Transformation Conditioning
    # Include inlier fit RMSE so excessive residuals fail here, not silently.
    try:
        _gate3_idx = np.where(inlier_mask.ravel() == 1)[0]
        _gate3_err = calculate_reprojection_errors(
            pts1_arr[_gate3_idx], pts2_arr[_gate3_idx], H_final) if len(_gate3_idx) else np.array([])
        _gate3_rmse = float(np.sqrt(np.mean(_gate3_err ** 2))) if len(_gate3_err) else None
    except Exception:
        _gate3_rmse = None
    tx_check = verify_transformation_quality(H_final, (orig_h2, orig_w2), fit_rmse_px=_gate3_rmse)
    if not tx_check["is_valid"]:
        logger.warning("Quality Gate 3 Rejected: Transformation conditioning invalid: %s", tx_check["reason"])
        return {
            "status": "geometric_verification_failed",
            "message": f"Estimated transformation rejected by quality gate: {tx_check['reason']}",
            "match_count": len(pts1_arr),
            "inlier_count": int(np.sum(inlier_mask)),
            "metrics": None,
            "homography": None,
            "metadata": {
                "source": meta1.to_dict(),
                "reference": meta2.to_dict(),
                "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
            },
        }

    # QUALITY GATE 4: Spatial Support & Concentration Check
    # Cells read from verified_matches positions (Fix P0-2): inlier_mask
    # indexes the verified subset, NOT selected_matches, so positional lookup
    # into selected_matches would score the wrong cells after verifier drops.
    inlier_indices = np.where(inlier_mask.ravel() == 1)[0]
    inlier_cells = [verified_matches[int(i)].get("cell", selected_matches[int(i)].get("cell"))
                    for i in inlier_indices if int(i) < len(verified_matches)]
    is_spatial_valid, spatial_reason, spatial_info = verify_spatial_quality_gate(inlier_cells)

    if not is_spatial_valid:
        logger.warning("Quality Gate 4 Rejected: Spatial support check failed: %s", spatial_reason)
        return {
            "status": "geometric_verification_failed",
            "message": f"Transformation rejected by spatial quality gate: {spatial_reason}.",
            "match_count": len(pts1_arr),
            "inlier_count": len(inlier_indices),
            "metrics": None,
            "homography": None,
            "metadata": {
                "source": meta1.to_dict(),
                "reference": meta2.to_dict(),
                "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
            },
        }

    # Mark inliers in records — IDENTITY-BASED (Fix P0-2). inlier_mask positions
    # index verified_matches (same dict objects as the surviving subset of
    # refinement_records), never the full list: positional indexing would flag
    # the wrong records whenever the verifier drops candidates.
    for _rec in refinement_records:
        _rec["is_inlier"] = False
    for _j in np.where(inlier_mask.ravel() == 1)[0].tolist():
        if 0 <= _j < len(verified_matches):
            verified_matches[_j]["is_inlier"] = True
    n_inliers_pre_refill = int(np.sum(inlier_mask))

    # --- Guided refill: H-constrained second pass over unused SSC keypoints ---
    # Honest densification: same NCC/MI thresholds, re-RANSAC + gates. Adopt the
    # refilled solution only on strict improvement (more inliers) with valid
    # conditioning and spatial support; otherwise keep the original solution.
    try:
        guided = _guided_refill_matches(
            kps1_ssc, selected_matches, pc1, pc2, H_final,
            scale_factor1, scale_factor2, work_w1, work_h1, work_w2, work_h2,
            half_patch_c, bool(multimodal_pair), grid_size, cell_w, cell_h,
        )
    except Exception:
        guided = []
    if guided:
        _g1 = [float(g["work_x1"]) * scale_factor1 for g in guided]
        _g1y = [float(g["work_y1"]) * scale_factor1 for g in guided]
        _g2 = [float(g["work_x2"]) * scale_factor2 for g in guided]
        _g2y = [float(g["work_y2"]) * scale_factor2 for g in guided]
        aug1 = np.vstack([pts1_arr, np.column_stack([_g1, _g1y]).astype(np.float32)])
        aug2 = np.vstack([pts2_arr, np.column_stack([_g2, _g2y]).astype(np.float32)])
        H_g, mask_g = cv2.findHomography(aug1, aug2, cv2.RANSAC, ransacReprojThreshold=5.0)
        if H_g is not None and mask_g is not None and int(np.sum(mask_g)) > n_inliers_pre_refill:
            _g_err = calculate_reprojection_errors(
                aug1[np.where(mask_g.ravel() == 1)[0]], aug2[np.where(mask_g.ravel() == 1)[0]], H_g)
            _g_rmse = float(np.sqrt(np.mean(_g_err ** 2))) if len(_g_err) else None
            _g_tx = verify_transformation_quality(H_g, (orig_h2, orig_w2), fit_rmse_px=_g_rmse)
            if _g_tx.get("is_valid"):
                for g in guided:
                    selected_matches.append(g)
                    refinement_records.append(make_match_record(
                        float(g["work_x1"]) * scale_factor1,
                        float(g["work_y1"]) * scale_factor1,
                        float(g["work_x2"]) * scale_factor2,
                        float(g["work_y2"]) * scale_factor2,
                        float(g["score"]),
                        refinement_dx=0.0,
                        refinement_dy=0.0,
                        is_refined=False,
                        method="guided_refill",
                        match_id=len(refinement_records),
                    ))
                    refinement_records[-1]["cell"] = g.get("cell")
                pts1_arr, pts2_arr, H_final, inlier_mask = aug1, aug2, H_g, mask_g
                inlier_flat = inlier_mask.ravel()
                # Identity-based marking (Fix P0-2): aug order == records order
                # here (appended consistently), lengths match by construction.
                for i, rec in enumerate(refinement_records):
                    rec["is_inlier"] = bool(i < len(inlier_flat) and inlier_flat[i] == 1)
                logger.info("Guided refill: inliers %d -> %d (+%d measured)",
                            n_inliers_pre_refill, int(np.sum(mask_g)), len(guided))
            else:
                logger.info("Guided refill rejected by quality gate (%s); keeping original solution.",
                            _g_tx.get("reason"))
        else:
            logger.info("Guided refill: no strict inlier gain (%d candidates); keeping original solution.",
                        len(guided))

    # --- Item 3: Post-RANSAC Lucas-Kanade Sub-Pixel Refinement (OHRC↔TMC-2 only) ---
    lk_stats: Optional[Dict[str, Any]] = None
    if not multimodal_pair and comp1_gray.shape == match_ref_gray.shape and int(np.sum(inlier_mask)) >= 4:
        inlier_idx_lk = np.where(inlier_mask.ravel() == 1)[0]
        lk_src = pts1_arr[inlier_idx_lk]
        lk_dst = pts2_arr[inlier_idx_lk]

        refined_src, refined_dst, lk_stats = refine_inliers_lucas_kanade(
            pc1, pc2, lk_src, lk_dst, scale_factor1, scale_factor2
        )

        if lk_stats["refined_count"] > 0:
            # (a) ORIGINAL baseline: pre-refinement point set and homography
            err_a = calculate_reprojection_errors(lk_src, lk_dst, H_final)
            rmse_a = float(np.sqrt(np.mean(err_a**2)))

            # (b) MIXED refined+unrefined point set with a homography re-fit on all of them
            H_b, mask_b = cv2.findHomography(
                refined_src, refined_dst, cv2.RANSAC, ransacReprojThreshold=5.0
            )
            rmse_b = None
            err_b = None
            if H_b is not None:
                err_b = calculate_reprojection_errors(refined_src, refined_dst, H_b)
                rmse_b = float(np.sqrt(np.mean(err_b**2)))

            # (c) ONLY successfully-refined points
            debug_pts = lk_stats.get("debug_points", [])
            passed_mask = np.array([pt.get("passed", False) for pt in debug_pts], dtype=bool)
            n_passed = int(np.sum(passed_mask))
            rmse_c = None
            err_c = None
            if n_passed >= 4:
                src_c = refined_src[passed_mask]
                dst_c = refined_dst[passed_mask]
                H_c, _ = cv2.findHomography(src_c, dst_c, cv2.RANSAC, ransacReprojThreshold=5.0)
                if H_c is None:
                    H_c, _ = cv2.findHomography(src_c, dst_c, 0)
                if H_c is not None:
                    err_c = calculate_reprojection_errors(src_c, dst_c, H_c)
                    rmse_c = float(np.sqrt(np.mean(err_c**2)))

            # Record Step 1 comparison
            lk_stats["step1_comparison"] = {
                "fit_rmse_a_orig": round(rmse_a, 4),
                "fit_rmse_b_mixed": round(rmse_b, 4) if rmse_b is not None else None,
                "fit_rmse_c_refined_only": round(rmse_c, 4) if rmse_c is not None else None,
                "refined_count": n_passed,
                "total_inliers": len(lk_src),
                "err_a_per_point": [round(float(e), 4) for e in err_a],
                "err_b_per_point": [round(float(e), 4) for e in err_b] if err_b is not None else [],
                "err_c_per_point": [round(float(e), 4) for e in err_c] if err_c is not None else [],
            }

            # Step 3 Fix (Option B):
            # Only attempt re-fit if fraction of refined points is >= 50% and count >= 4.
            # Re-fitting across mixed refined+coarse points or low-refined subsets destabilizes the model.
            refined_fraction = n_passed / max(1, len(lk_src))
            re_fit_accepted = False

            if n_passed >= 4 and refined_fraction >= 0.50 and H_c is not None:
                tx_check_c = verify_transformation_quality(H_c, (orig_h2, orig_w2))
                if tx_check_c["is_valid"] and rmse_c is not None and rmse_c < rmse_a:
                    # Verified improvement: adopt refined homography and update inliers to refined subset
                    H_final = H_c
                    re_fit_accepted = True

                    refined_inlier_indices = inlier_idx_lk[passed_mask]
                    new_inlier_mask = np.zeros_like(inlier_mask)
                    new_inlier_mask[refined_inlier_indices] = 1
                    inlier_mask = new_inlier_mask

                    pts1_arr[refined_inlier_indices] = src_c
                    pts2_arr[refined_inlier_indices] = dst_c

            # Update match records with refined coordinates for downstream use
            # Full float() precision — never round() coordinates.
            for j, idx in enumerate(inlier_idx_lk):
                if idx < len(refinement_records):
                    if debug_pts and j < len(debug_pts) and debug_pts[j]["passed"]:
                        refinement_records[idx]["target_x"] = float(refined_dst[j, 0])
                        refinement_records[idx]["target_y"] = float(refined_dst[j, 1])
                        refinement_records[idx]["image2_x"] = float(refined_dst[j, 0])
                        refinement_records[idx]["image2_y"] = float(refined_dst[j, 1])
                        refinement_records[idx]["lk_refined"] = True

    # 9. Compute Canonical Master Metrics (fixed 10x10 reporting grid)
    metrics = compute_canonical_metrics(
        pts1_arr, pts2_arr, inlier_mask, H_final, (orig_h2, orig_w2), canonical_grid_size,
        gsd_m=metric_gsd, dem_data=dem_arr
    )
    metrics["matching_grid_size"] = matching_grid_size
    metrics["canonical_grid_size"] = canonical_grid_size
    # SIH illumination-invariance audit trail (required keys).
    metrics["synthetic_reference_used"] = bool(synthetic_reference_used)
    metrics["illumination_compensation"] = illumination_compensation
    metrics["illumination_detail"] = illumination_detail
    if lk_stats:
        metrics["lk_refinement"] = lk_stats

    fit_rmse = metrics.get("fit_rmse_px")
    tx_check = verify_transformation_quality(
        H_final, (orig_h2, orig_w2), fit_rmse_px=fit_rmse
    )
    if not tx_check["is_valid"]:
        return {
            "status": "geometric_verification_failed",
            "message": f"Transformation rejected by quality gate: {tx_check['reason']}",
            "direction": "native",
            "match_count": len(pts1_arr),
            "inlier_count": int(np.sum(inlier_mask)),
            "metrics": None,
            "homography": None,
            "metadata": {
                "source": meta1.to_dict(),
                "reference": meta2.to_dict(),
                "working_scale": {"working_gsd_m": working_gsd, "method": working_scale_note},
                "scale_estimation_method": scale_estimation_method,
                "estimated_scale_ratio": estimated_scale_ratio,
            },
        }

    # 10. Generate Output Products
    # A. Warped source image into reference space via Piecewise Affine / TPS
    curr_inliers = np.where(inlier_mask.ravel() == 1)[0] if inlier_mask is not None else []
    if len(curr_inliers) >= 4:
        warped_source = warp_piecewise_affine(
            raw1_color, pts1_arr[curr_inliers], pts2_arr[curr_inliers],
            (orig_h2, orig_w2), tile_size=256, global_H=H_final
        )
    else:
        warped_source = cv2.warpPerspective(
            raw1_color, H_final, (orig_w2, orig_h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

    # B. Export Registered GeoTIFF Raster
    tif_path = out_path / "registered_source.tif"
    written_tif = False
    try:
        import rasterio
        from rasterio.transform import from_origin

        profile = {
            "driver": "GTiff",
            "height": orig_h2,
            "width": orig_w2,
            "count": 3 if warped_source.ndim == 3 else 1,
            "dtype": "uint8",
            "nodata": 0,
            "crs": raster_meta2.get("crs") or "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs",
            "transform": raster_meta2.get("transform") or from_origin(0, orig_h2, tag_gsd, tag_gsd),
        }
        with rasterio.open(str(tif_path), "w", **profile) as dst:
            if warped_source.ndim == 3:
                for b in range(3):
                    dst.write(warped_source[:, :, 2 - b], b + 1)  # BGR to RGB
            else:
                dst.write(warped_source, 1)
        written_tif = True
    except Exception:
        pass

    if not written_tif:
        cv2.imwrite(str(tif_path), warped_source)

    # C. Registered Preview PNG
    preview_path = out_path / "registered_preview.png"
    cv2.imwrite(str(preview_path), warped_source)

    # D. 50px Alternating Checkerboard QA (vectorized; identical to nested loops)
    block_size = 50
    yy, xx = np.mgrid[0:orig_h2, 0:orig_w2]
    mask = ((xx // block_size) + (yy // block_size)) % 2 == 0
    blended = np.where(mask[..., None], warped_source, raw2_color)

    checker_path = out_path / "registered_checkerboard.png"
    cv2.imwrite(str(checker_path), blended)

    # D2. Displacement-vector quiver QA (best effort; never fails the run).
    quiver_path = out_path / "registered_quiver.png"
    try:
        from quiver import create_displacement_quiver
        _qm = np.where(inlier_mask.ravel() == 1)[0]
        if len(_qm) >= 3 and H_final is not None:
            create_displacement_quiver(
                pts1_arr[_qm], pts2_arr[_qm], np.asarray(H_final, dtype=np.float64),
                (orig_h2, orig_w2), quiver_path)
        else:
            quiver_path = None
    except Exception:
        quiver_path = None

    # E. Save matches JSON (sub-pixel precision preserved, indent=2)
    matches_path = out_path / "matches.json"
    dump_matches_json(refinement_records, matches_path, indent=2)

    # F. Save metrics JSON
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(sanitize_for_json(metrics), f, indent=2, cls=SubpixelJSONEncoder)

    # G. Save transform JSON
    transform_path = out_path / "transform.json"
    transform_data = {
        "model": "homography",
        "matrix": H_final.tolist(),
        "quality": tx_check,
        "direction": "native",
    }
    with open(transform_path, "w") as f:
        json.dump(sanitize_for_json(transform_data), f, indent=2, cls=SubpixelJSONEncoder)

    # H. Save metadata JSON
    metadata_path = out_path / "metadata.json"
    full_metadata = {
        "source": meta1.to_dict(),
        "reference": meta2.to_dict(),
        "working_scale": {
            "working_gsd_m": working_gsd,
            "method": working_scale_note,
        },
        "scale_estimation_method": scale_estimation_method,
        "estimated_scale_ratio": estimated_scale_ratio,
        "terrain_correction": {
            "source": terrain_info1,
            "reference": terrain_info2,
        },
        "illumination_compensation": illumination_compensation,
        "synthetic_reference_used": bool(synthetic_reference_used),
        "illumination_detail": illumination_detail,
        "direction": "native",
        "provenance": {
            "source_path": str(img_path1),
            "reference_path": str(img_path2),
            "dem_path": str(dem_path) if dem_path else None,
            "matcher": "CFOG_PhaseCongruency_v2.0",
            "spatial_attempts": spatial_attempts,
            "coarse_matcher": "mutual_information" if multimodal_pair else "normalized_correlation",
            "direction": "native",
            "illumination_compensation": illumination_compensation,
            "synthetic_reference_used": bool(synthetic_reference_used),
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(sanitize_for_json(full_metadata), f, indent=2, cls=SubpixelJSONEncoder)

    abs_str = f"{metrics['absolute_rmse_m']:.2f} m" if metrics.get("absolute_rmse_m") is not None else "N/A"
    logger.info(
        "Registration succeeded: inliers=%d/%d (%.1f%%), fit_rmse=%.4f px, absolute_rmse=%s",
        metrics.get("inlier_count", 0),
        metrics.get("match_count", 0),
        metrics.get("inlier_ratio", 0.0) * 100,
        metrics.get("fit_rmse_px", 0.0),
        abs_str,
    )

    return {
        "status": "success",
        "direction": "native",
        "source": {
            "sensor": meta1.sensor,
            "width": orig_w1,
            "height": orig_h1,
            "gsd_m": _finite_gsd(meta1),
        },
        "reference": {
            "sensor": meta2.sensor,
            "width": orig_w2,
            "height": orig_h2,
            "gsd_m": _finite_gsd(meta2),
        },
        "working_scale": {
            "gsd_m": working_gsd,
            "method": working_scale_note,
        },
        "scale_estimation_method": scale_estimation_method,
        "estimated_scale_ratio": estimated_scale_ratio,
        "metrics": metrics,
        "homography": H_final.tolist(),
        "synthetic_reference_used": bool(synthetic_reference_used),
        "illumination_compensation": illumination_compensation,
        "terrain_correction": full_metadata["terrain_correction"],
        "spatial_attempts": spatial_attempts,
        "matches": [m for m in refinement_records if m.get("is_inlier", False)],
        "all_matches": refinement_records,
        "outputs": {
            "registered_raster": str(tif_path),
            "preview": str(preview_path),
            "checkerboard": str(checker_path),
            "quiver": str(quiver_path) if quiver_path is not None else None,
            "matches": str(matches_path),
            "metrics": str(metrics_path),
            "transform": str(transform_path),
            "metadata": str(metadata_path),
        },
    }
