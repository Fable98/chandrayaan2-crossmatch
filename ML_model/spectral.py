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

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2


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

    # 2. Lunar Spectral Index (e.g. R950 / R750 ratio for pyroxene / crater rim delineation)
    # If wavelengths are provided, locate nearest bands; otherwise pick proxy bands
    if wavelengths is not None and len(wavelengths) == b:
        idx_750 = int(np.argmin(np.abs(wavelengths - 750.0)))
        idx_950 = int(np.argmin(np.abs(wavelengths - 950.0)))
    else:
        # Proxies: early NIR band and later NIR band
        idx_750 = min(b // 4, b - 1)
        idx_950 = min((3 * b) // 4, b - 1)

    band_750 = arr[:, :, idx_750]
    band_950 = arr[:, :, idx_950]

    # Ratio highlighting absorption and composition contrast
    ratio = band_950 / np.maximum(band_750, 1e-4)
    rp2, rp98 = float(np.percentile(ratio, 2)), float(np.percentile(ratio, 98))
    if rp98 > rp2:
        ratio_norm = np.clip((ratio - rp2) / (rp98 - rp2), 0.0, 1.0)
    else:
        ratio_norm = np.zeros((h, w), dtype=np.float32)

    # 3. Structural Fusion (75% PC1 + 25% Spectral Index)
    structural_map = 0.75 * pc1_norm + 0.25 * ratio_norm

    # Contrast enhancement for phase congruency
    u8 = (np.clip(structural_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_u8 = clahe.apply(u8)
    enhanced_float = enhanced_u8.astype(np.float32) / 255.0

    return enhanced_float


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

    return {
        "spatial_misalignment_px": round(spatial_misalignment, 4),
        "spectral_variance": round(mean_var, 6),
        "spectral_angle_mapper_deg": round(sam_deg, 4),
    }
