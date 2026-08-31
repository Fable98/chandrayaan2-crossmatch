"""
geo.py — Pixel-to-geographic coordinate conversion for Chandrayaan-2 imagery.

INTEGRATION NOTES (do not re-litigate — decisions documented here):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PIXEL-TO-GEO METHOD: Perspective transform from 4 corners.
   The manifest (user_triplets.json) provides only 4 named footprint
   corners per sensor — no GDAL-style affine transform, no RPC model.
   We solve a 3×3 homography mapping the pixel-space rectangle
   [(0,0), (W,0), (W,H), (0,H)] to the 4 geographic corners, then
   apply it to each match point. This is exact for a planar scene
   (valid at lunar-tile scale).

2. ML OUTPUT LOCATION: ML_model/matches.json at repo root.
   Single file per image pair. Bare JSON list of
   {image1_x, image1_y, image2_x, image2_y, confidence}.
   No triplet_id, no homography serialized.

3. IIRS FORMAT: The manifest has IIRS as 4-corner footprint only
   (same structure as OHRC/TMC). Verified via scripts/check_iirs_rotation.py:
   BOTH region_001 and region_002 have ROTATED IIRS quads (bottom lons
   differ from top lons, ~2.5% area loss from a bbox). The /iirs-overlay
   endpoint now returns full 4-corner quads (Footprint model), NOT a bbox.
   Frontend should use leaflet-distortableImage, not L.imageOverlay.

4. IMAGE SIZE: matcher.py resizes all inputs to 512×512. Pixel
   coordinates in matches.json are in this 512×512 space.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Perspective transform: pixel (x, y) → geographic (lat, lon)
# ---------------------------------------------------------------------------

def _solve_perspective_matrix(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    Solve for a 3×3 perspective transform matrix H such that
    dst = H @ src (in homogeneous coordinates).

    src_pts, dst_pts: (4, 2) arrays of corresponding points.
    Returns: (3, 3) matrix H.
    """
    # Build the 8×9 system from the 4 point correspondences
    # Each point pair gives 2 equations
    A = []
    for (sx, sy), (dx, dy) in zip(src_pts, dst_pts):
        A.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
        A.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])

    A = np.array(A, dtype=np.float64)

    # Solve via SVD — H is the last row of V^T (null space of A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize so H[2,2] = 1
    H /= H[2, 2]
    return H


def pixel_to_latlon_from_corners(
    px: float,
    py: float,
    corners: dict,
    image_width: int = 512,
    image_height: int = 512,
) -> tuple[float, float]:
    """
    Convert a pixel coordinate (px, py) to geographic (lat, lon) using
    the 4 named footprint corners from the manifest.

    The mapping is a perspective transform from the pixel rectangle to
    the geographic quadrilateral defined by the corners.

    Args:
        px, py: Pixel coordinates (0-indexed, origin top-left).
        corners: Dict with keys top_left, top_right, bottom_right,
                 bottom_left, each containing {lat, lon}.
        image_width, image_height: Pixel dimensions (default 512×512).

    Returns:
        (lat, lon) tuple.

    Corner-to-pixel mapping:
        top_left     → (0, 0)
        top_right    → (W, 0)
        bottom_right → (W, H)
        bottom_left  → (0, H)
    """
    W, H = float(image_width), float(image_height)

    # Source: pixel-space rectangle corners (x, y order)
    src = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H],
    ], dtype=np.float64)

    # Destination: geographic corners (lon, lat order for x, y mapping)
    # We map pixel-x → lon, pixel-y → lat, then return (lat, lon)
    tl = corners["top_left"]
    tr = corners["top_right"]
    br = corners["bottom_right"]
    bl = corners["bottom_left"]

    dst = np.array([
        [tl["lon"], tl["lat"]],
        [tr["lon"], tr["lat"]],
        [br["lon"], br["lat"]],
        [bl["lon"], bl["lat"]],
    ], dtype=np.float64)

    H_mat = _solve_perspective_matrix(src, dst)

    # Apply H to the query point in homogeneous coordinates
    pt = np.array([px, py, 1.0], dtype=np.float64)
    result = H_mat @ pt
    result /= result[2]  # de-homogenize

    lon, lat = result[0], result[1]
    return (lat, lon)


def pixel_to_latlon_batch(
    points: list[tuple[float, float]],
    corners: dict,
    image_width: int = 512,
    image_height: int = 512,
) -> list[tuple[float, float]]:
    """
    Convert multiple pixel coordinates to (lat, lon) in one call.
    More efficient than calling pixel_to_latlon_from_corners in a loop
    because the perspective matrix is solved only once.
    """
    W, H_dim = float(image_width), float(image_height)

    src = np.array([
        [0, 0], [W, 0], [W, H_dim], [0, H_dim],
    ], dtype=np.float64)

    tl = corners["top_left"]
    tr = corners["top_right"]
    br = corners["bottom_right"]
    bl = corners["bottom_left"]

    dst = np.array([
        [tl["lon"], tl["lat"]],
        [tr["lon"], tr["lat"]],
        [br["lon"], br["lat"]],
        [bl["lon"], bl["lat"]],
    ], dtype=np.float64)

    H_mat = _solve_perspective_matrix(src, dst)

    results = []
    for px, py in points:
        pt = np.array([px, py, 1.0], dtype=np.float64)
        r = H_mat @ pt
        r /= r[2]
        results.append((r[1], r[0]))  # (lat, lon)

    return results


# ---------------------------------------------------------------------------
# Homography re-derivation from match points
# ---------------------------------------------------------------------------

def compute_homography_from_points(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
) -> list[list[float]] | None:
    """
    Re-derive a 3×3 homography matrix from matched point pairs.

    The ML team's matcher.py computes H via cv2.findHomography for RANSAC
    filtering but does not serialize it to matches.json. This function
    re-derives H from the surviving inlier points using a standard
    least-squares solve (no RANSAC needed since these are already inliers).

    Returns the 3×3 matrix as a list of lists, or None if fewer than 4
    point pairs are provided (underdetermined system).
    """
    if len(src_points) < 4:
        return None

    src = np.array(src_points, dtype=np.float64)
    dst = np.array(dst_points, dtype=np.float64)

    # For exactly 4 points, use the exact solve
    if len(src_points) == 4:
        H = _solve_perspective_matrix(src, dst)
    else:
        # Over-determined: build the full system and solve via SVD
        A = []
        for (sx, sy), (dx, dy) in zip(src, dst):
            A.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
            A.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])

        A_mat = np.array(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A_mat)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]

    return [[float(H[i, j]) for j in range(3)] for i in range(3)]
