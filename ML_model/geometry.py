"""
ML_model/geometry.py — Non-Planar Warping Helpers and Simplified DEM Relief Compensation.

Provides:
1. Piecewise Affine transformation on overlapping tiles with smooth 2D cosine blending.
2. Thin Plate Splines (TPS) non-rigid 2D resampling for non-planar surfaces.
3. Simplified DEM relief-displacement compensation (local vertical-offset shift
   along emission direction). This is NOT rigorous 3D photogrammetric
   sensor-model ray-intersection; see README Limitations. On steep crater walls
   (>30deg) or large off-nadir angles, residual parallax remains and the
   pipeline correctly fails closed via Quality Gates instead of forcing a fit.
4. DEM-aware robust RANSAC model estimation helpers.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

logger = logging.getLogger("ML_model.geometry")


# ---------------------------------------------------------------------------
# 1. Rigorous DEM Ray-Intersection
# ---------------------------------------------------------------------------

def dem_ray_intersection(
    pixel_coords: np.ndarray,
    dem: np.ndarray,
    emission_deg: float = 0.0,
    azimuth_deg: float = 45.0,
    gsd_m: float = 5.0,
    camera_altitude_m: float = 100000.0,
    max_iters: int = 15,
    tolerance_m: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified DEM relief-displacement estimator (closed-form local shift).

    Approximates line-of-sight/DEM intersection as Δz*tan(emission)/GSD along
    the sensor LOS azimuth. This is NOT a rigorous orbital sensor-model
    ray-trace (no intrinsics/extrinsics, no iterative ray-march against a
    geodetically registered DEM). Pass sensor LOS azimuth — never sun azimuth.

    Azimuth convention (unified 2026-09-15 with
    matcher_cfog.compute_dem_ray_shift_correction and
    matcher_cfog.apply_dem_relief_compensation): compass-style degrees
    clockwise from north (0=N, 90=E), matching sun_azimuth_deg /
    sensor_los_azimuth_deg metadata. In y-down pixel coordinates the unit
    shift direction is (sin(psi), -cos(psi)).
    """
    pts = np.asarray(pixel_coords, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))
    if azimuth_deg is None:
        raise ValueError(
            "dem_ray_intersection: azimuth_deg is required (compass deg clockwise "
            "from north). Pass sensor LOS azimuth; never silently default, or the "
            "relief shift is applied in a hallucinated direction."
        )

    h, w = dem.shape[:2]
    e_rad = math.radians(float(emission_deg))
    psi_rad = math.radians(float(azimuth_deg))

    # Center origin of coordinate frame at image center
    cx, cy = w / 2.0, h / 2.0
    mean_dem = float(np.nanmean(dem))
    logger.debug(
        "DEM ray intersection: %d points, emission=%.1f deg, azimuth=%.1f deg, mean DEM elevation=%.1f m",
        len(pts), emission_deg, azimuth_deg, mean_dem
    )

    # Sample DEM with bilinear interpolation
    def sample_elevation(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        x_c = np.clip(x_arr, 0.0, float(w - 1))
        y_c = np.clip(y_arr, 0.0, float(h - 1))
        x0 = np.floor(x_c).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.floor(y_c).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)

        wx = x_c - x0
        wy = y_c - y0

        val00 = dem[y0, x0]
        val10 = dem[y0, x1]
        val01 = dem[y1, x0]
        val11 = dem[y1, x1]

        elev = (
            (1.0 - wx) * (1.0 - wy) * val00
            + wx * (1.0 - wy) * val10
            + (1.0 - wx) * wy * val01
            + wx * wy * val11
        )
        return elev.astype(np.float64)

    # Initial horizontal ground coordinates at nominal mean datum
    x_px = pts[:, 0].copy()
    y_px = pts[:, 1].copy()

    # Iterative ray-marching to intersect terrain surface
    for _ in range(max_iters):
        current_elev = sample_elevation(x_px, y_px)
        delta_elev = current_elev - mean_dem

        # Relief parallax displacement (compass azimuth in y-down pixels):
        # dx = dz * tan(emission) * sin(azimuth) / gsd
        # dy = dz * tan(emission) * (-cos(azimuth)) / gsd
        scale = math.tan(e_rad) / max(gsd_m, 1e-4)
        target_dx = delta_elev * (scale * math.sin(psi_rad))
        target_dy = delta_elev * (scale * -math.cos(psi_rad))

        new_x = pts[:, 0] + target_dx
        new_y = pts[:, 1] + target_dy

        shift = np.hypot(new_x - x_px, new_y - y_px) * gsd_m
        x_px = new_x
        y_px = new_y
        if np.max(shift) < tolerance_m:
            break

    final_elev = sample_elevation(x_px, y_px)
    # NOTE: Y_m is pixel-frame (south-positive), not ENU north — no consumer
    # currently uses coords_3d, but do not interpret column 1 as northing.
    X_m = (x_px - cx) * gsd_m
    Y_m = (y_px - cy) * gsd_m
    Z_m = final_elev

    coords_3d = np.column_stack([X_m, Y_m, Z_m])
    displacements_px = np.column_stack([x_px - pts[:, 0], y_px - pts[:, 1]])

    return coords_3d, displacements_px


# ---------------------------------------------------------------------------
# 2. Piecewise Affine Transformation on Overlapping Tiles
# ---------------------------------------------------------------------------

def warp_piecewise_affine(
    image: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    out_shape: Tuple[int, int],
    tile_size: int = 256,
    overlap_ratio: float = 0.25,
    global_H: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Warps source image to destination space using piecewise affine transformations
    fitted locally over overlapping tiles, blended smoothly with 2D cosine windows.
    
    Eliminates global homography planar distortion over non-planar terrain (e.g. crater walls).
    
    Args:
        image: Source image array (H, W) or (H, W, C).
        src_pts: (N, 2) inlier coordinates in source image space.
        dst_pts: (N, 2) inlier coordinates in destination image space.
        out_shape: (out_h, out_w) shape of target output image.
        tile_size: Tile width and height in destination space (e.g. 256 or 512).
        overlap_ratio: Overlap between adjacent tiles (e.g. 0.25).
        global_H: Fallback global homography if local inliers are insufficient.
    """
    out_h, out_w = out_shape[:2]
    is_color = (image.ndim == 3)
    channels = image.shape[2] if is_color else 1

    # Prepare output accumulator and weight map
    accum = np.zeros((out_h, out_w, channels), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)

    src_arr = np.asarray(src_pts, dtype=np.float32)
    dst_arr = np.asarray(dst_pts, dtype=np.float32)
    logger.debug(
        "Piecewise affine warping: %d tie-points, source shape=%s, target shape=%s, tile_size=%d",
        len(src_arr), image.shape, out_shape, tile_size
    )

    # Compute global fallback affine or homography
    if global_H is None:
        if len(src_arr) >= 4:
            global_H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        if global_H is None and len(src_arr) >= 3:
            M_aff, _ = cv2.estimateAffine2D(src_arr, dst_arr)
            if M_aff is not None:
                global_H = np.vstack([M_aff, [0, 0, 1]])
        if global_H is None:
            global_H = np.eye(3, dtype=np.float32)

    # Step size with overlap
    step = max(32, int(tile_size * (1.0 - overlap_ratio)))

    # Precompute 2D cosine (Hann) window for smooth blending
    def make_hann_window(th: int, tw: int) -> np.ndarray:
        wx = np.hanning(tw + 2)[1:-1]
        wy = np.hanning(th + 2)[1:-1]
        w2d = np.outer(wy, wx).astype(np.float32)
        # Ensure minimum weight so edges never have divide-by-zero
        return np.maximum(w2d, 1e-3)[:, :, np.newaxis]

    x_starts = list(range(0, out_w, step))
    y_starts = list(range(0, out_h, step))

    for y0 in y_starts:
        y1 = min(out_h, y0 + tile_size)
        th = y1 - y0
        for x0 in x_starts:
            x1 = min(out_w, x0 + tile_size)
            tw = x1 - x0

            # Find correspondences whose destination point lies in this tile (with 20% margin)
            margin_x = int(tw * 0.2)
            margin_y = int(th * 0.2)
            tx0_m = max(0, x0 - margin_x)
            tx1_m = min(out_w, x1 + margin_x)
            ty0_m = max(0, y0 - margin_y)
            ty1_m = min(out_h, y1 + margin_y)

            in_tile_mask = (
                (dst_arr[:, 0] >= tx0_m)
                & (dst_arr[:, 0] < tx1_m)
                & (dst_arr[:, 1] >= ty0_m)
                & (dst_arr[:, 1] < ty1_m)
            )
            tile_src = src_arr[in_tile_mask]
            tile_dst = dst_arr[in_tile_mask]

            local_H = None
            if len(tile_src) >= 4:
                H_loc, _ = cv2.findHomography(tile_src, tile_dst, cv2.RANSAC, 4.0)
                if H_loc is not None and abs(np.linalg.det(H_loc)) > 1e-4:
                    local_H = H_loc
            if local_H is None and len(tile_src) >= 3:
                aff_loc, _ = cv2.estimateAffine2D(tile_src, tile_dst)
                if aff_loc is not None:
                    local_H = np.vstack([aff_loc, [0, 0, 1]])

            # Fallback to global H if local support is too sparse
            if local_H is None:
                local_H = global_H

            # Warp full image via local_H or warp tile crop
            try:
                # We need inverse transform to map tile pixels [x0..x1, y0..y1] into source
                inv_H = np.linalg.inv(local_H)
                if abs(inv_H[2, 2]) > 1e-12:
                    inv_H = inv_H / inv_H[2, 2]

                # Generate coordinate grid for this tile
                grid_y, grid_x = np.indices((th, tw), dtype=np.float32)
                grid_x += x0
                grid_y += y0

                # Remap destination tile coords to source image coords
                ones = np.ones_like(grid_x)
                pts_h = np.stack([grid_x, grid_y, ones], axis=-1)  # (th, tw, 3)
                mapped_h = pts_h @ inv_H.T
                # No abs() sign-flip at the plane: behind-plane pixels keep
                # their sign (copysign) so they sample far-away border instead
                # of mirroring into deceptively valid image content.
                z = np.where(
                    np.abs(mapped_h[:, :, 2:3]) < 1e-12,
                    np.copysign(1e-12, mapped_h[:, :, 2:3]),
                    mapped_h[:, :, 2:3],
                )
                map_x = (mapped_h[:, :, 0:1] / z).astype(np.float32)
                map_y = (mapped_h[:, :, 1:2] / z).astype(np.float32)

                # Interpolate from source image
                src_for_remap = image if is_color else image[:, :, np.newaxis]
                warped_tile = cv2.remap(
                    src_for_remap,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                if warped_tile.ndim == 2:
                    warped_tile = warped_tile[:, :, np.newaxis]

                # Blend using 2D Hann window
                hann = make_hann_window(th, tw)
                accum[y0:y1, x0:x1] += warped_tile.astype(np.float32) * hann
                weight_map[y0:y1, x0:x1] += hann
            except Exception:
                continue

    # Normalize accumulator by blended weights
    safe_weights = np.maximum(weight_map, 1e-6)
    blended = accum / safe_weights
    blended = np.clip(blended, 0.0, 255.0 if image.dtype == np.uint8 else 1.0)

    if not is_color:
        blended = blended[:, :, 0]

    return blended.astype(image.dtype)


# ---------------------------------------------------------------------------
# 3. Thin Plate Splines (TPS) Warping
# ---------------------------------------------------------------------------

def warp_thin_plate_splines(
    image: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    out_shape: Tuple[int, int],
    num_ctrl_points: int = 64,
    global_H: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Warps source image to destination space using Thin Plate Splines (TPS).
    Provides smooth non-rigid interpolation that conforms precisely to local crater relief.
    """
    out_h, out_w = out_shape[:2]
    src_arr = np.asarray(src_pts, dtype=np.float32)
    dst_arr = np.asarray(dst_pts, dtype=np.float32)

    if len(src_arr) < 4:
        if global_H is not None:
            return cv2.warpPerspective(image, global_H, (out_w, out_h), flags=cv2.INTER_LINEAR)
        return cv2.resize(image, (out_w, out_h))

    # Subsample control points if too dense (for efficiency)
    if len(src_arr) > num_ctrl_points:
        indices = np.linspace(0, len(src_arr) - 1, num_ctrl_points, dtype=int)
        ctrl_src = src_arr[indices]
        ctrl_dst = dst_arr[indices]
    else:
        ctrl_src = src_arr
        ctrl_dst = dst_arr

    # Add 4 image corner points to pin boundaries accurately in destination space
    ih, iw = image.shape[:2]
    corners_src = np.array([[0, 0], [iw - 1, 0], [0, ih - 1], [iw - 1, ih - 1]], dtype=np.float32)

    # Project corners using global_H or robust fitted transformation
    corners_dst = None
    if global_H is not None:
        try:
            c_proj = cv2.perspectiveTransform(corners_src.reshape(1, -1, 2), global_H.astype(np.float64))
            if c_proj is not None and np.all(np.isfinite(c_proj)):
                corners_dst = c_proj.reshape(-1, 2).astype(np.float32)
        except Exception:
            corners_dst = None

    if corners_dst is None:
        try:
            H_est, _ = cv2.findHomography(src_arr, dst_arr, 0)
            if H_est is not None:
                c_proj = cv2.perspectiveTransform(corners_src.reshape(1, -1, 2), H_est.astype(np.float64))
                if c_proj is not None and np.all(np.isfinite(c_proj)):
                    corners_dst = c_proj.reshape(-1, 2).astype(np.float32)
        except Exception:
            corners_dst = None

    if corners_dst is None:
        try:
            M_aff, _ = cv2.estimateAffine2D(src_arr, dst_arr)
            if M_aff is not None:
                c_h = np.hstack([corners_src, np.ones((4, 1), dtype=np.float32)])
                corners_dst = (c_h @ M_aff.T).astype(np.float32)
        except Exception:
            corners_dst = None

    if corners_dst is None:
        corners_dst = np.array([[0, 0], [out_w - 1, 0], [0, out_h - 1], [out_w - 1, out_h - 1]], dtype=np.float32)

    all_src = np.vstack([ctrl_src, corners_src])
    all_dst = np.vstack([ctrl_dst, corners_dst])

    try:
        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(len(all_src))]
        tps.estimateTransformation(
            all_dst.reshape(1, -1, 2),
            all_src.reshape(1, -1, 2),
            matches,
        )
        warped = tps.warpImage(image)
        if warped.shape[:2] != (out_h, out_w):
            warped = cv2.resize(warped, (out_w, out_h))
        return warped
    except Exception:
        # Fallback to perspective or affine
        if global_H is not None:
            return cv2.warpPerspective(image, global_H, (out_w, out_h), flags=cv2.INTER_LINEAR)
        M, _ = cv2.estimateAffine2D(src_arr, dst_arr)
        if M is not None:
            return cv2.warpAffine(image, M, (out_w, out_h))
        return cv2.resize(image, (out_w, out_h))



# ---------------------------------------------------------------------------
# 3b. Hartley-Normalized Minimal DLT Solver
# ---------------------------------------------------------------------------

def _hartley_normalize(pts: np.ndarray) -> np.ndarray:
    """Similarity T mapping points to zero mean, mean distance sqrt(2)."""
    pts = np.asarray(pts, dtype=np.float64)
    mean = pts.mean(axis=0)
    mean_dist = float(np.mean(np.hypot(*(pts - mean).T)))
    s = math.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array([[s, 0.0, -s * mean[0]], [0.0, s, -s * mean[1]], [0.0, 0.0, 1.0]])


def _dlt_homography_hartley(
    src: np.ndarray,
    dst: np.ndarray,
    min_triangle_area: float = 1e-4,
    sample_reproj_tol_px: float = 1.0,
) -> Optional[np.ndarray]:
    """Exact 4-point homography via Hartley-normalized DLT (SVD).

    Returns None for degenerate samples instead of a garbage matrix. Checks:
    - sample geometry: every omit-one triangle must span nonzero area in
      Hartley-normalized coordinates (rejects collinear/coincident samples).
      NOTE: an SVD null-space ratio was tried as the discriminator and
      REJECTED — for exact 4-point data the design matrix is rank-deficient
      by construction, and s[-1]/s[-2] measured 0.16 on clean similarity
      data vs 0.017 on collinear data (inverted signal).
    - normalized determinant (area collapse after H[2,2] == 1 normalization).
    - self-consistency: the exact fit must reproduce its own 4 samples.
    """
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(s) != 4 or len(d) != 4:
        return None
    try:
        T1, T2 = _hartley_normalize(s), _hartley_normalize(d)
        s_h = np.hstack([s, np.ones((4, 1))])
        sn = (T1 @ s_h.T).T[:, :2]
        dn = (T2 @ np.hstack([d, np.ones((4, 1))]).T).T[:, :2]
        for pts in (sn, dn):
            for skip in range(4):
                (x1, y1), (x2, y2), (x3, y3) = np.delete(pts, skip, axis=0)
                if abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0 <= min_triangle_area:
                    return None  # collinear/coincident sample
        A = []
        for (x, y), (xp, yp) in zip(sn, dn):
            A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp])
            A.append([0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp])
        _, _, Vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
        H = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1
        if abs(H[2, 2]) < 1e-12:
            return None
        H = H / H[2, 2]
        if abs(np.linalg.det(H)) < 1e-8:
            return None
        proj = (H @ s_h.T).T
        if np.any(np.abs(proj[:, 2]) < 1e-12):
            return None
        if float(np.max(np.linalg.norm(proj[:, :2] / proj[:, 2:3] - d, axis=1))) > sample_reproj_tol_px:
            return None
        return H
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. DEM-Aware RANSAC Model Fitting
# ---------------------------------------------------------------------------

def ransac_dem_aware_fit(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    dem: Optional[np.ndarray] = None,
    emission_deg: float = 0.0,
    azimuth_deg: float = 45.0,
    gsd_m: float = 5.0,
    max_iters: int = 500,
    reproj_thresh_px: float = 5.0,
    random_seed: int = 42,
) -> Tuple[Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Robust RANSAC estimator that incorporates terrain relief correction into the error metric.

    Instead of penalizing correspondences on steep crater walls as outliers due to planar
    parallax discrepancy, the error metric projects candidates through the DEM ray-intersection.

    FRAME CONTRACT (wrong-frame fix 2026-09-15): the returned H maps
    RELIEF-CORRECTED source points (raw + info["relief_correction_field_px"])
    to destination points — NOT raw source points. A single matrix cannot
    represent the per-point relief field, so no raw-frame H is returned;
    instead info["frame"] = "relief_corrected_source_to_dst" and
    info["raw_frame_rmse_px"] quantifies the error of misusing H on raw
    points. Downstream consumers must either add the correction field before
    applying H, or treat H as a corrected-frame approximation (see
    info["relief_max_shift_px"] for the approximation bound).
    """
    pts1 = np.asarray(src_pts, dtype=np.float64)
    pts2 = np.asarray(dst_pts, dtype=np.float64)
    n = len(pts1)

    if n < 4:
        return None, np.zeros(n, dtype=np.uint8), {"status": "insufficient_points"}

    rng = np.random.RandomState(random_seed)

    # 1. Precompute DEM relief displacement vectors if DEM is present
    relief_dxdy = np.zeros_like(pts1)
    relief_enabled = False
    relief_reason = "dem_unavailable"
    if dem is None or not isinstance(dem, np.ndarray) or dem.ndim != 2:
        relief_reason = "dem_unavailable"
    elif abs(float(emission_deg) if emission_deg is not None else 0.0) <= 1e-2:
        relief_reason = "nadir_or_emission_unavailable"
    elif azimuth_deg is None:
        # Sensor LOS azimuth genuinely unknown: a guessed direction moves every
        # pixel the wrong way on 3 of 4 quadrants. Disabled is honest (and this
        # previously crashed with TypeError inside dem_ray_intersection).
        relief_reason = "los_azimuth_unavailable"
    else:
        try:
            _, relief_dxdy = dem_ray_intersection(
                pts1, dem, emission_deg=emission_deg, azimuth_deg=azimuth_deg, gsd_m=gsd_m
            )
            relief_enabled = True
            relief_reason = "relief_compensated"
        except Exception as exc:
            logger.warning("DEM relief correction failed (%s); fitting uncorrected.", exc)
            relief_dxdy = np.zeros_like(pts1)
            relief_reason = f"ray_shift_failed: {exc}"
    corrected_pts1 = pts1 + relief_dxdy if relief_enabled else pts1.copy()

    best_inliers = np.zeros(n, dtype=bool)
    best_H = None
    best_count = 0

    # Standard RANSAC on relief-compensated space, with a Hartley-normalized
    # DLT minimal solver: the old unnormalized getPerspectiveTransform +
    # scale-dependent det<1e-5 check admitted near-degenerate samples (e.g.
    # near-collinear crater-wall points) whose garbage H then won consensus.
    for _ in range(max_iters):
        sample_idx = rng.choice(n, 4, replace=False)
        s1 = corrected_pts1[sample_idx]
        s2 = pts2[sample_idx]

        try:
            H_candidate = _dlt_homography_hartley(s1, s2)
            if H_candidate is None:
                continue

            # Project all points (no abs() sign-flip: behind-plane points are
            # rejected with inf error, never mirrored into false inliers).
            ones = np.ones((n, 1), dtype=np.float64)
            p1_h = np.hstack([corrected_pts1, ones])
            proj = (H_candidate @ p1_h.T).T
            _z = proj[:, 2:3]
            _behind = (_z.ravel() <= 1e-12)
            _z_safe = np.where(np.abs(_z) < 1e-12, np.copysign(1e-12, _z), _z)
            proj_2d = proj[:, :2] / _z_safe

            errors = np.linalg.norm(proj_2d - pts2, axis=1)
            if np.any(_behind):
                errors = np.where(_behind, np.inf, errors)
            inliers = errors < reproj_thresh_px
            inliers = errors < reproj_thresh_px
            count = int(np.sum(inliers))

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_H = H_candidate
        except Exception:
            continue

    if best_count >= 4:
        # Refit on all inliers
        inlier_s1 = corrected_pts1[best_inliers]
        inlier_s2 = pts2[best_inliers]
        H_refined, mask = cv2.findHomography(
            inlier_s1, inlier_s2, cv2.RANSAC, ransacReprojThreshold=reproj_thresh_px
        )
        if H_refined is not None:
            best_H = H_refined
            final_inlier_mask = np.zeros(n, dtype=np.uint8)
            inlier_indices = np.where(best_inliers)[0]
            if mask is not None:
                final_inlier_mask[inlier_indices[mask.ravel() == 1]] = 1
            else:
                final_inlier_mask[inlier_indices] = 1
        else:
            final_inlier_mask = best_inliers.astype(np.uint8)
    else:
        final_inlier_mask = np.zeros(n, dtype=np.uint8)
        best_H = None

    final_count = int(np.sum(final_inlier_mask))
    logger.info(
        "DEM-aware RANSAC completed: %d/%d inliers (%.1f%%), relief compensated: %s",
        final_count, n, (final_count / max(1, n)) * 100,
        relief_enabled
    )

    shift_mag = np.hypot(relief_dxdy[:, 0], relief_dxdy[:, 1]) if relief_enabled else np.zeros(n)
    # Honest frame-gap quantification: error of applying the corrected-frame H
    # to RAW source points (what happens if a consumer ignores the frame tag).
    raw_frame_rmse = None
    if best_H is not None:
        try:
            with np.errstate(all="ignore"):
                _raw_h = np.hstack([pts1, np.ones((n, 1))])
                _proj = (np.asarray(best_H, dtype=np.float64) @ _raw_h.T).T
                _rz = _proj[:, 2:3]
                _rbehind = (_rz.ravel() <= 1e-12)
                _rz_safe = np.where(np.abs(_rz) < 1e-12, np.copysign(1e-12, _rz), _rz)
                _rerr = np.sqrt(np.sum(((_proj[:, :2] / _rz_safe) - pts2) ** 2, axis=1))
                if np.any(_rbehind):
                    _rerr = np.where(_rbehind, np.inf, _rerr)
                _rmse = float(np.sqrt(np.mean(_rerr ** 2)))
                raw_frame_rmse = _rmse if np.isfinite(_rmse) else None
        except Exception:
            raw_frame_rmse = None

    return best_H, final_inlier_mask, {
        "inlier_count": final_count,
        "total_points": n,
        "dem_compensated": bool(relief_enabled),
        "relief_reason": relief_reason,
        # Frame contract: H maps relief-corrected source -> dst.
        "frame": "relief_corrected_source_to_dst" if relief_enabled else "raw_source_to_dst",
        "relief_correction_field_px": np.asarray(relief_dxdy, dtype=np.float64),
        "relief_mean_shift_px": float(np.mean(shift_mag)),
        "relief_max_shift_px": float(np.max(shift_mag)) if n else 0.0,
        "raw_frame_rmse_px": raw_frame_rmse,
    }


def estimate_topographic_relief_strain(
    pts1: np.ndarray,
    pts2: np.ndarray,
    H_global: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Estimates non-planar topographic relief strain from correspondence residuals.
    Compares global homography residual variance against local piecewise affine residuals.
    A high strain ratio indicates that elevation relief violates the single-plane homography.
    """
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)
    n = len(p1)
    if n < 6 or H_global is None:
        return {"strain_detected": False, "strain_ratio": 1.0, "reason": "insufficient_points"}

    # 1. Global homography reprojection error (copysign: behind-plane points
    # inflate the error honestly instead of mirroring into small residuals).
    p1_h = np.hstack([p1, np.ones((n, 1))])
    proj = (H_global.astype(np.float64) @ p1_h.T).T
    z = proj[:, 2:3]
    with np.errstate(all="ignore"):
        z = np.where(np.abs(z) < 1e-12, np.copysign(1e-12, z), z)
        global_err = np.linalg.norm(proj[:, :2] / z - p2, axis=1)
        global_err = np.where(np.isfinite(global_err), global_err, 1e6)
    rmse_global = float(np.sqrt(np.mean(global_err ** 2)))

    # 2. Local piecewise affine / k-NN error
    local_errs = []
    for i in range(n):
        dists = np.linalg.norm(p1 - p1[i], axis=1)
        k_indices = np.argsort(dists)[:max(4, min(6, n))]
        M, _ = cv2.estimateAffine2D(p1[k_indices], p2[k_indices])
        if M is not None:
            pt_proj = M @ np.array([p1[i, 0], p1[i, 1], 1.0])
            local_errs.append(float(np.linalg.norm(pt_proj - p2[i])))
        else:
            local_errs.append(float(global_err[i]))

    rmse_local = float(np.sqrt(np.mean(np.array(local_errs) ** 2))) if local_errs else rmse_global
    strain_ratio = float(rmse_global / max(rmse_local, 0.05))

    # If global error is elevated and local models significantly reduce error (> 1.35x),
    # physical relief displacement is present across the terrain.
    strain_detected = bool(rmse_global >= 1.2 and strain_ratio >= 1.35)

    return {
        "strain_detected": strain_detected,
        "rmse_global_px": round(rmse_global, 4),
        "rmse_local_px": round(rmse_local, 4),
        "strain_ratio": round(strain_ratio, 4),
    }

