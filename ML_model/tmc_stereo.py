"""
TMC-2 Triplet Stereo Photogrammetry Engine.

Implements along-track stereoscopic disparity estimation and digital elevation
model (DEM) derivation for Chandrayaan-2 Terrain Mapping Camera-2 (TMC-2).

Orbital Physical Geometry:
    - Fore camera:  +26.0 deg forward tilt along flight track
    - Nadir camera:   0.0 deg nadir view
    - Aft camera:   -26.0 deg backward tilt along flight track

Base-to-Height Ratio (B/H):
    B/H = tan(theta_fore) - tan(theta_aft)
        = tan(+26 deg) - tan(-26 deg)
        = 2 * tan(26 deg) ~= 0.9755

Elevation from Parallax Disparity:
    Z(x, y) = (disparity_px(x, y) * GSD) / (B / H)
    where GSD = 5.0 m (TMC-2 nominal ground sampling distance).

Authors: Chandrayaan-2 SIH 26166 Team
"""

from __future__ import annotations
import math
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple


def compute_tmc_base_to_height_ratio(
    fore_angle_deg: float = 26.0,
    aft_angle_deg: float = -26.0,
) -> float:
    """
    Compute along-track base-to-height ratio (B/H) from camera tilt angles.

    Formula:
        B/H = tan(rad(theta_fore)) - tan(rad(theta_aft))

    For Fore (+26 deg) and Aft (-26 deg):
        B/H = tan(26 deg) - (-tan(26 deg)) = 2 * tan(26 deg) ~= 0.975525.
    For Fore (+26 deg) and Nadir (0 deg):
        B/H = tan(26 deg) ~= 0.487733.
    """
    rad_fore = math.radians(fore_angle_deg)
    rad_aft = math.radians(aft_angle_deg)
    b_over_h = math.tan(rad_fore) - math.tan(rad_aft)
    return float(b_over_h)


def disparity_to_elevation(
    disparity_px: np.ndarray,
    gsd_m: float = 5.0,
    b_over_h: float = 0.975525,
    datum_m: float = 0.0,
) -> np.ndarray:
    """
    Convert stereoscopic along-track disparity (in pixels) to physical elevation (in meters).

    Formula:
        Z(x, y) = datum_m + (disparity(x, y) * GSD) / (B/H)
    """
    if abs(b_over_h) < 1e-6:
        raise ValueError(f"Invalid B/H ratio ({b_over_h}); division by zero.")
    return datum_m + (disparity_px * gsd_m) / b_over_h


def elevation_to_disparity(
    elevation_m: np.ndarray,
    gsd_m: float = 5.0,
    b_over_h: float = 0.975525,
    datum_m: float = 0.0,
) -> np.ndarray:
    """
    Convert physical elevation (in meters) to stereoscopic along-track disparity (in pixels).

    Formula:
        disparity(x, y) = (Z(x, y) - datum_m) * (B/H) / GSD
    """
    if gsd_m <= 0:
        raise ValueError(f"GSD must be positive, got {gsd_m}")
    return (elevation_m - datum_m) * b_over_h / gsd_m


def generate_synthetic_stereo_views(
    nadir_img: np.ndarray,
    dem_meters: np.ndarray,
    gsd_m: float = 5.0,
    fore_angle_deg: float = 26.0,
    aft_angle_deg: float = -26.0,
    axis: str = "x",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Fore and Aft stereo views from a Nadir image and DEM.

    Uses photogrammetric backward warping:
        For axis='x' (horizontal epipolar alignment):
            disp_fore = Z * tan(theta_fore) / GSD
            disp_aft  = Z * tan(theta_aft) / GSD

    Returns:
        (fore_img, aft_img)
    """
    h, w = nadir_img.shape[:2]
    tan_fore = math.tan(math.radians(fore_angle_deg))
    tan_aft = math.tan(math.radians(aft_angle_deg))

    disp_fore = (dem_meters * tan_fore / gsd_m).astype(np.float32)
    disp_aft = (dem_meters * tan_aft / gsd_m).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    if axis == "x":
        map_fore_x = grid_x - disp_fore
        map_fore_y = grid_y
        map_aft_x = grid_x - disp_aft
        map_aft_y = grid_y
    else:
        map_fore_x = grid_x
        map_fore_y = grid_y - disp_fore
        map_aft_x = grid_x
        map_aft_y = grid_y - disp_aft

    fore_img = cv2.remap(nadir_img, map_fore_x, map_fore_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    aft_img = cv2.remap(nadir_img, map_aft_x, map_aft_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return fore_img, aft_img


def compute_tmc_stereo_disparity(
    img_left: np.ndarray,
    img_right: np.ndarray,
    min_disparity: int = -16,
    num_disparities: int = 64,
    block_size: int = 7,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dense disparity between rectified stereo pair using Semi-Global Block Matching (SGBM).

    Args:
        img_left: Left (or Fore) image, uint8 or float32.
        img_right: Right (or Aft) image, uint8 or float32.
        min_disparity: Minimum disparity value (can be negative for relative elevation around datum).
        num_disparities: Range of disparity search (must be divisible by 16).
        block_size: Matched block size (odd integer >= 3).

    Returns:
        (disparity_map, valid_mask):
            - disparity_map: float32 ndarray in pixels.
            - valid_mask: bool ndarray (True where disparity is valid).
    """
    # Ensure uint8
    if img_left.dtype != np.uint8:
        left_u8 = cv2.normalize(img_left, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        left_u8 = img_left

    if img_right.dtype != np.uint8:
        right_u8 = cv2.normalize(img_right, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        right_u8 = img_right

    if left_u8.ndim == 3:
        left_u8 = cv2.cvtColor(left_u8, cv2.COLOR_BGR2GRAY)
    if right_u8.ndim == 3:
        right_u8 = cv2.cvtColor(right_u8, cv2.COLOR_BGR2GRAY)

    # Ensure num_disparities is positive multiple of 16
    num_disp = max(16, int(math.ceil(num_disparities / 16.0) * 16))
    blk_size = max(3, block_size if block_size % 2 == 1 else block_size + 1)

    p1 = 8 * 1 * blk_size * blk_size
    p2 = 32 * 1 * blk_size * blk_size

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disp,
        blockSize=blk_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=mode,
    )

    disp16 = stereo.compute(left_u8, right_u8)
    # SGBM returns 16-bit fixed point with 4 fractional bits (divide by 16)
    disp = disp16.astype(np.float32) / 16.0

    # In OpenCV SGBM, invalid pixels have value (minDisparity - 1)
    valid_mask = disp >= float(min_disparity)

    # Inpaint or smooth invalid regions if partial occlusions exist
    if np.any(~valid_mask) and np.any(valid_mask):
        disp_clean = disp.copy()
        disp_clean[~valid_mask] = 0.0
        mask_inpaint = (~valid_mask).astype(np.uint8)
        disp_filled = cv2.inpaint(
            disp_clean.astype(np.float32),
            mask_inpaint,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA,
        )
        disp_clean[~valid_mask] = disp_filled[~valid_mask]
    else:
        disp_clean = disp

    return disp_clean, valid_mask


def derive_dem_from_tmc_stereo(
    img_fore: np.ndarray,
    img_aft: np.ndarray,
    img_nadir: Optional[np.ndarray] = None,
    gsd_m: float = 5.0,
    fore_angle_deg: float = 26.0,
    aft_angle_deg: float = -26.0,
    min_disparity: int = -16,
    num_disparities: int = 64,
    block_size: int = 7,
    datum_m: float = 0.0,
) -> Dict[str, Any]:
    """
    Derive a photogrammetric Digital Elevation Model (DEM) from TMC-2 Fore and Aft views.

    Args:
        img_fore: Fore camera image (+26 deg view along track).
        img_aft: Aft camera image (-26 deg view along track).
        img_nadir: Optional Nadir camera image.
        gsd_m: Ground sampling distance (5.0 m nominal for TMC-2).
        fore_angle_deg: Fore camera tilt angle (+26.0 deg).
        aft_angle_deg: Aft camera tilt angle (-26.0 deg).
        min_disparity: Minimum disparity in pixels (negative allowed for bidirectional parallax).
        num_disparities: Disparity search range.
        block_size: Matching window size.
        datum_m: Reference zero elevation datum.

    Returns:
        dict containing:
            - 'dem_meters': float32 ndarray of physical elevations in meters.
            - 'dem_u8': uint8 ndarray normalized for display/export.
            - 'disparity_map': float32 ndarray of along-track disparity in pixels.
            - 'valid_mask': bool ndarray.
            - 'b_over_h': float B/H ratio (~0.9755).
            - 'metrics': dictionary of terrain statistics.
    """
    b_over_h = compute_tmc_base_to_height_ratio(fore_angle_deg, aft_angle_deg)
    disp, valid_mask = compute_tmc_stereo_disparity(
        img_fore,
        img_aft,
        min_disparity=min_disparity,
        num_disparities=num_disparities,
        block_size=block_size,
    )

    # Convert disparity to physical elevation in meters
    dem_meters = disparity_to_elevation(disp, gsd_m=gsd_m, b_over_h=b_over_h, datum_m=datum_m)

    # Compute statistics on valid pixels
    valid_elev = dem_meters[valid_mask] if np.any(valid_mask) else dem_meters
    min_elev = float(np.percentile(valid_elev, 1)) if len(valid_elev) > 0 else 0.0
    max_elev = float(np.percentile(valid_elev, 99)) if len(valid_elev) > 0 else 1.0
    mean_elev = float(np.mean(valid_elev)) if len(valid_elev) > 0 else 0.0
    relief_m = max_elev - min_elev

    # Normalize to 8-bit visual DEM
    if relief_m > 1e-4:
        dem_norm = np.clip((dem_meters - min_elev) / relief_m, 0.0, 1.0)
        dem_u8 = (dem_norm * 255.0).astype(np.uint8)
    else:
        dem_u8 = np.full(dem_meters.shape, 128, dtype=np.uint8)

    valid_ratio = float(np.sum(valid_mask)) / float(valid_mask.size) if valid_mask.size > 0 else 0.0

    return {
        "dem_meters": dem_meters.astype(np.float32),
        "dem_u8": dem_u8,
        "disparity_map": disp.astype(np.float32),
        "valid_mask": valid_mask,
        "b_over_h": b_over_h,
        "metrics": {
            "min_elevation_m": min_elev,
            "max_elevation_m": max_elev,
            "mean_elevation_m": mean_elev,
            "relief_range_m": relief_m,
            "valid_match_ratio": valid_ratio,
            "gsd_m": gsd_m,
            "b_over_h": b_over_h,
        },
    }
