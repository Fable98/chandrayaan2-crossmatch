"""
ML_model/geometry.py — Non-Planar Transformations, Piecewise Affine/TPS Warping,
and Rigorous DEM Ray-Intersection for Lunar Topography.

Resolves homography/planar approximation breakdown on steep crater walls, central peaks,
and off-nadir emission geometries by:
1. Piecewise Affine transformation on overlapping tiles with smooth 2D cosine blending.
2. Thin Plate Splines (TPS) non-rigid deformation for non-planar surfaces.
3. Rigorous 3D photogrammetric sensor ray-intersection against Lunar DEM height fields.
4. DEM-aware robust RANSAC model estimation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2


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
    Rigorously intersects line-of-sight camera rays with a lunar Digital Elevation Model (DEM).
    
    For steep crater walls and off-nadir geometries, standard planar assumptions fail because
    terrain elevation creates strong relief parallax along the look-direction.
    
    Args:
        pixel_coords: (N, 2) array of [x, y] pixel coordinates in the source sensor space.
        dem: (H, W) 2D array of lunar surface elevation values in meters.
        emission_deg: Sensor emission angle (off-nadir tilt) in degrees.
        azimuth_deg: Sensor line-of-sight azimuth angle in degrees.
        gsd_m: Ground Sample Distance in meters/pixel.
        camera_altitude_m: Spacecraft altitude above lunar reference datum (m).
        max_iters: Maximum iterations for iterative ray-surface intersection.
        tolerance_m: Convergence threshold in meters.
        
    Returns:
        coords_3d: (N, 3) real-world lunar surface coordinates [X_m, Y_m, Z_elev_m].
        displacements_px: (N, 2) terrain-induced relief displacement vectors [dx_px, dy_px].
    """
    pts = np.asarray(pixel_coords, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))

    h, w = dem.shape[:2]
    e_rad = math.radians(float(emission_deg))
    psi_rad = math.radians(float(azimuth_deg))

    # Ray unit direction vector in local topocentric system:
    # Z is pointing upward normal to datum; X east; Y north
    # Off-nadir ray points downward towards surface
    dir_x = math.sin(e_rad) * math.cos(psi_rad)
    dir_y = math.sin(e_rad) * math.sin(psi_rad)
    dir_z = -math.cos(e_rad)

    # Center origin of coordinate frame at image center
    cx, cy = w / 2.0, h / 2.0
    mean_dem = float(np.nanmean(dem))

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

        # Relief parallax displacement: dx = dz * tan(emission) * cos(azimuth) / gsd
        scale = math.tan(e_rad) / max(gsd_m, 1e-4)
        target_dx = delta_elev * (scale * math.cos(psi_rad))
        target_dy = delta_elev * (scale * math.sin(psi_rad))

        new_x = pts[:, 0] + target_dx
        new_y = pts[:, 1] + target_dy

        shift = np.hypot(new_x - x_px, new_y - y_px) * gsd_m
        x_px = new_x
        y_px = new_y
        if np.max(shift) < tolerance_m:
            break

    final_elev = sample_elevation(x_px, y_px)
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
                z = np.maximum(np.abs(mapped_h[:, :, 2:3]), 1e-12)
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
) -> np.ndarray:
    """
    Warps source image to destination space using Thin Plate Splines (TPS).
    Provides smooth non-rigid interpolation that conforms precisely to local crater relief.
    """
    out_h, out_w = out_shape[:2]
    src_arr = np.asarray(src_pts, dtype=np.float32)
    dst_arr = np.asarray(dst_pts, dtype=np.float32)

    if len(src_arr) < 4:
        # Fallback to simple identity or homography
        return cv2.resize(image, (out_w, out_h))

    # Subsample control points if too dense (for efficiency)
    if len(src_arr) > num_ctrl_points:
        indices = np.linspace(0, len(src_arr) - 1, num_ctrl_points, dtype=int)
        ctrl_src = src_arr[indices]
        ctrl_dst = dst_arr[indices]
    else:
        ctrl_src = src_arr
        ctrl_dst = dst_arr

    # Add 4 image corner points to pin boundaries
    ih, iw = image.shape[:2]
    corners_src = np.array([[0, 0], [iw - 1, 0], [0, ih - 1], [iw - 1, ih - 1]], dtype=np.float32)
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
        # Fallback to affine
        M, _ = cv2.estimateAffine2D(src_arr, dst_arr)
        if M is not None:
            return cv2.warpAffine(image, M, (out_w, out_h))
        return cv2.resize(image, (out_w, out_h))


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
    """
    pts1 = np.asarray(src_pts, dtype=np.float64)
    pts2 = np.asarray(dst_pts, dtype=np.float64)
    n = len(pts1)

    if n < 4:
        return None, np.zeros(n, dtype=np.uint8), {"status": "insufficient_points"}

    rng = np.random.RandomState(random_seed)

    # 1. Precompute DEM relief displacement vectors if DEM is present
    if dem is not None and abs(emission_deg) > 1e-2 and dem.ndim == 2:
        _, relief_dxdy = dem_ray_intersection(
            pts1, dem, emission_deg=emission_deg, azimuth_deg=azimuth_deg, gsd_m=gsd_m
        )
        # Correct source points by relief displacement before planar model fitting
        corrected_pts1 = pts1 + relief_dxdy
    else:
        corrected_pts1 = pts1.copy()

    best_inliers = np.zeros(n, dtype=bool)
    best_H = None
    best_count = 0

    # Standard RANSAC on relief-compensated space
    for _ in range(max_iters):
        sample_idx = rng.choice(n, 4, replace=False)
        s1 = corrected_pts1[sample_idx]
        s2 = pts2[sample_idx]

        try:
            H_candidate = cv2.getPerspectiveTransform(s1.astype(np.float32), s2.astype(np.float32))
            if abs(np.linalg.det(H_candidate)) < 1e-5:
                continue

            # Project all points
            ones = np.ones((n, 1), dtype=np.float64)
            p1_h = np.hstack([corrected_pts1, ones])
            proj = (H_candidate.astype(np.float64) @ p1_h.T).T
            z = np.where(np.abs(proj[:, 2:3]) < 1e-12, 1e-12, proj[:, 2:3])
            proj_2d = proj[:, :2] / z

            errors = np.linalg.norm(proj_2d - pts2, axis=1)
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

    return best_H, final_inlier_mask, {
        "inlier_count": int(np.sum(final_inlier_mask)),
        "total_points": n,
        "dem_compensated": bool(dem is not None and abs(emission_deg) > 1e-2),
    }
