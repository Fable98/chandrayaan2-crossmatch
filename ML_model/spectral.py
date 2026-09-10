"""
ML_model/spectral.py — Hyperspectral Feature Engineering for Chandrayaan-2 IIRS

Eliminates naive grayscale conversion and simple band averaging for IIRS hyperspectral cubes.
Implements:
1. Principal Component Analysis (PCA) to extract PC1 as high-contrast structural basemap.
2. Lunar mineralogical band ratio indices (R950/R750, continuum depth) highlighting crater rims.
3. Contrast-enhanced structural fusion for 2D Phase Congruency & CFOG extraction.
4. Independent quantification of Spatial Misalignment (pixels) vs. Spectral Variance (SAM/variance).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

logger = logging.getLogger("ML_model.spectral")


def enhance_iirs_structural_features(
    hypercube: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transforms a multi-band/hyperspectral IIRS data cube into a single high-contrast
    structural feature map using Principal Component Analysis (PC1) and lunar band ratios.
    
    Bypasses naive cv2.cvtColor and arithmetic band averaging to retain structural edges
    on crater rims, ejecta blankets, and shadowed morphologies.

    Args:
        hypercube: 3D array of shape (H, W, B) or (B, H, W) where B >= 3 bands.
        wavelengths: Optional 1D array of band wavelengths in nanometers.

    Returns:
        2D float32 array normalized to [0.0, 1.0] with enhanced morphological boundaries.
    """
    arr = np.asarray(hypercube, dtype=np.float32)

    # Standardize dimensions to (H, W, B)
    if arr.ndim == 2:
        return np.clip(arr / 255.0 if arr.max() > 1.0 else arr, 0.0, 1.0)
    elif arr.ndim == 3:
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2] and arr.shape[0] > 1:
            # (B, H, W) -> (H, W, B)
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Invalid hyperspectral array shape: {arr.shape}")

    h, w, b = arr.shape
    if b < 2:
        return arr[:, :, 0]

    # Replace NaNs / Infs with band medians
    for band_idx in range(b):
        band = arr[:, :, band_idx]
        bad_mask = ~np.isfinite(band)
        if np.any(bad_mask):
            valid_val = float(np.nanmedian(band)) if np.any(~bad_mask) else 0.0
            arr[bad_mask, band_idx] = valid_val

    # 1. Principal Component Analysis (PCA)
    logger.info(
        "Enhancing hyperspectral features: %d bands, spatial dims=(%d, %d). Applying PCA PC1 + mineral ratio fusion.",
        b, h, w
    )
    flat = arr.reshape(-1, b)
    mean_vec = np.mean(flat, axis=0, keepdims=True)
    centered = flat - mean_vec

    # Compute covariance matrix (B x B)
    cov = (centered.T @ centered) / max(1, flat.shape[0] - 1)

    # Eigen decomposition (ordered descending)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(eig_vals)[::-1]
    pc1_vec = eig_vecs[:, sort_idx[0]]

    # Project onto PC1
    pc1_flat = centered @ pc1_vec
    pc1 = pc1_flat.reshape(h, w)

    # Ensure consistent positive correlation with overall radiance
    mean_img = np.mean(arr, axis=2)
    corr = np.corrcoef(pc1.ravel(), mean_img.ravel())[0, 1]
    if corr < 0:
        pc1 = -pc1

    # Robust percentile stretch [2nd to 98th percentile]
    p2, p98 = float(np.percentile(pc1, 2)), float(np.percentile(pc1, 98))
    if p98 > p2:
        pc1_norm = np.clip((pc1 - p2) / (p98 - p2), 0.0, 1.0)
    else:
        pc1_norm = np.zeros((h, w), dtype=np.float32)

    # 2. Lunar Spectral Index (physical R950/R750 only when calibrated).
    # Honest radiometry: fabricated proxy bands (b//4, 3b//4) plus synthetic
    # blending and CLAHE distort physical I/F radiometry, so they are disabled
    # unless calibrated wavelengths with true 750 nm and 950 nm coverage exist.
    # Without calibration we return linearly normalized PC1 directly.
    wl = None
    try:
        if wavelengths is not None:
            wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    except Exception:
        wl = None
    has_calibrated_ratio = False
    idx_750: int = -1
    idx_950: int = -1
    if wl is not None and wl.size == b and np.all(np.isfinite(wl)):
        try:
            wmin, wmax = float(np.min(wl)), float(np.max(wl))
            if wmin <= 750.0 <= wmax and wmin <= 950.0 <= wmax:
                c750 = int(np.argmin(np.abs(wl - 750.0)))
                c950 = int(np.argmin(np.abs(wl - 950.0)))
                # Nearest calibrated band must actually sample the feature.
                if abs(float(wl[c750]) - 750.0) <= 150.0 and abs(float(wl[c950]) - 950.0) <= 150.0 and c750 != c950:
                    idx_750, idx_950 = c750, c950
                    has_calibrated_ratio = True
        except Exception:
            has_calibrated_ratio = False

    if not has_calibrated_ratio:
        logger.info(
            "No calibrated 750/950 nm coverage; returning linearly normalized "
            "PC1 without synthetic ratio or CLAHE to preserve radiometry."
        )
        return np.clip(pc1_norm, 0.0, 1.0).astype(np.float32)

    band_750 = arr[:, :, idx_750]
    band_950 = arr[:, :, idx_950]

    # Physical ratio highlighting pyroxene absorption contrast.
    ratio = band_950 / np.maximum(band_750, 1e-4)
    rp2, rp98 = float(np.percentile(ratio, 2)), float(np.percentile(ratio, 98))
    if rp98 > rp2:
        ratio_norm = np.clip((ratio - rp2) / (rp98 - rp2), 0.0, 1.0)
    else:
        ratio_norm = np.zeros((h, w), dtype=np.float32)

    # 3. Structural Fusion (75% PC1 + 25% physical Spectral Index)
    structural_map = 0.75 * pc1_norm + 0.25 * ratio_norm

    # Contrast enhancement for phase congruency (calibrated path only)
    u8 = (np.clip(structural_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_u8 = clahe.apply(u8)
    enhanced_float = enhanced_u8.astype(np.float32) / 255.0

    return enhanced_float


def compute_sam_angle_map(
    hypercube: np.ndarray,
    reference_spectrum: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-pixel Spectral Angle Mapper divergence in radians.

    theta_SAM = arccos(dot(S, R_ref) / (|S| |R_ref| + eps)).

    Args:
        hypercube: (H, W, B) or (B, H, W) spectral cube.
        reference_spectrum: (B,) reference vector; defaults to spatial mean.
        eps: small stabilizer for the denominator.

    Returns:
        (H, W) float32 array of spectral angles in radians in [0, pi].
    """
    arr = np.asarray(hypercube, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"hypercube must be 3D, got shape {arr.shape}")
    h, w, nb = arr.shape
    flat = arr.reshape(-1, nb)
    if reference_spectrum is None:
        ref = np.mean(flat, axis=0)
    else:
        ref = np.asarray(reference_spectrum, dtype=np.float64).ravel()
    norm_ref = float(np.linalg.norm(ref))
    norm_flat = np.linalg.norm(flat, axis=1)
    denom = norm_flat * norm_ref + float(eps)
    cos_a = np.clip(np.sum(flat * ref, axis=1) / np.maximum(denom, 1e-12), -1.0, 1.0)
    theta = np.arccos(cos_a).reshape(h, w)
    return np.clip(theta, 0.0, float(np.pi)).astype(np.float32)


def apply_sam_gate_to_overlay(
    overlay: np.ndarray,
    sam_map: np.ndarray,
    threshold_rad: float = 0.35,
    fill_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Gate a PC1 blend overlay by SAM divergence.

    Pixels with theta_SAM above threshold (saturated pixels, shadow
    artifacts, anomalous spectra) are rejected from the visualization.

    Args:
        overlay: (H, W) blended visualization to gate.
        sam_map: (H, W) SAM angles in radians (see compute_sam_angle_map).
        threshold_rad: rejection threshold in radians.
        fill_value: replacement for rejected pixels; defaults to overlay median.

    Returns:
        (gated_overlay, valid_mask, info) where valid_mask is True for kept
        pixels and info reports rejected_fraction and threshold used.
    """
    ov = np.asarray(overlay, dtype=np.float32)
    sam = np.asarray(sam_map, dtype=np.float32)
    if ov.shape[:2] != sam.shape[:2]:
        raise ValueError(f"shape mismatch overlay {ov.shape} vs sam {sam.shape}")
    thresh = float(threshold_rad)
    valid = np.isfinite(sam) & (sam <= thresh)
    if fill_value is None:
        try:
            fill = float(np.median(ov[np.isfinite(ov)])) if np.any(np.isfinite(ov)) else 0.0
        except Exception:
            fill = 0.0
    else:
        fill = float(fill_value)
    gated = ov.copy()
    gated[~valid] = fill
    info: Dict[str, Any] = {
        "threshold_rad": thresh,
        "rejected_fraction": float(1.0 - np.mean(valid)) if valid.size else 0.0,
        "num_rejected": int(np.sum(~valid)),
        "num_valid": int(np.sum(valid)),
    }
    return gated.astype(np.float32), valid, info


def quantify_iirs_residuals(
    hypercube: np.ndarray,
    reprojection_errors_px: np.ndarray,
) -> Dict[str, Any]:
    """
    Separates geometric Spatial Misalignment from physical Spectral Variance
    for the IIRS hyperspectral leg of registration.

    Returns:
        spatial_misalignment_px: Reprojection RMSE in pixels.
        spectral_variance: Normalized variance across spectral bands.
        mean_spectral_angle_deg: Average spectral angle deviation from mean signature.
    """
    arr = np.asarray(hypercube, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))

    # 1. Spatial Misalignment (pixels)
    if len(reprojection_errors_px) > 0:
        spatial_misalignment = float(np.sqrt(np.mean(reprojection_errors_px**2)))
    else:
        spatial_misalignment = 0.0

    # 2. Spectral Variance
    if arr.ndim == 3 and arr.shape[2] > 1:
        # Variance across spectral bands per pixel, averaged spatially
        band_var = np.var(arr, axis=2)
        mean_var = float(np.mean(band_var))
        
        # Spectral Angle Mapper (SAM) deviation from spatial mean spectrum
        flat = arr.reshape(-1, arr.shape[2])
        mean_spectrum = np.mean(flat, axis=0)
        norm_mean = np.linalg.norm(mean_spectrum)
        norm_flat = np.linalg.norm(flat, axis=1)

        safe_denom = np.maximum(norm_flat * norm_mean, 1e-8)
        cos_angles = np.clip(np.sum(flat * mean_spectrum, axis=1) / safe_denom, -1.0, 1.0)
        sam_deg = float(np.mean(np.degrees(np.arccos(cos_angles))))
    else:
        mean_var = 0.0
        sam_deg = 0.0

    logger.info(
        "IIRS residual analysis: spatial misalignment=%.4f px, spectral variance=%.6f, SAM=%.2f deg",
        spatial_misalignment, mean_var, sam_deg
    )

    return {
        "spatial_misalignment_px": round(spatial_misalignment, 4),
        "spectral_variance": round(mean_var, 6),
        "spectral_angle_mapper_deg": round(sam_deg, 4),
    }
