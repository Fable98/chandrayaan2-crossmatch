"""
geo.py — Pixel-to-geographic coordinate conversion for Chandrayaan-2 imagery.

PIPELINE GEOMETRY ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SHARED BOUNDING BOX (BY DESIGN):
   In the confirmed preprocessing pipeline, OHRC, TMC-2, and IIRS tiles
   are spatially cropped and reprojected to the exact same shared bounding
   box (TripletBounds: west_lon, east_lon, south_lat, north_lat) on a common
   512×512 pixel grid. There are no independent per-sensor rotations.

2. SHARED AFFINE TRANSFORM:
   Because all three sensors share one bounding box over the 512×512 pixel grid,
   pixel-to-latlon conversion uses a single direct affine transform:
       lon = west_lon + (px / 512.0) * (east_lon - west_lon)
       lat = north_lat - (py / 512.0) * (north_lat - south_lat)
   This applies identically to OHRC, TMC-2, and IIRS pixel coordinates.

3. 0–360° LONGITUDE CONVENTION:
   The real data pipeline and PDS4 labels use standard lunar planetocentric
   0–360° longitudes (e.g. 336.48°). Code and bounds comparisons preserve
   this convention without arbitrary ±180° truncation.

4. LEGACY CORNER-BASED CONVERSION (UNUSED):
   The earlier per-sensor corner-based perspective transform functions
   (pixel_to_latlon_from_corners / pixel_to_latlon_batch) were built against
   mock test data with artificial footprint rotation. They are preserved below
   as unused/legacy references.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared-Bounds Affine Transform: pixel (x, y) → geographic (lat, lon)
# ---------------------------------------------------------------------------

def pixel_to_latlon_from_bounds(
    px: float,
    py: float,
    bounds: dict,
    image_width: int = 512,
    image_height: int = 512,
) -> tuple[float, float]:
    """
    Convert a pixel coordinate (px, py) to geographic (lat, lon) using
    the shared TripletBounds (west_lon, east_lon, south_lat, north_lat).

    Args:
        px, py: Pixel coordinates (0-indexed, origin at top-left).
        bounds: Dict or object with keys west_lon, east_lon, south_lat, north_lat.
        image_width, image_height: Dimensions in pixels (default 512×512).

    Returns:
        (lat, lon) tuple in standard lunar degrees (lat in [-90, 90], lon in [0, 360]).
    """
    if hasattr(bounds, "model_dump"):
        b = bounds.model_dump()
    elif hasattr(bounds, "dict"):
        b = bounds.dict()
    else:
        b = bounds

    w_lon = float(b["west_lon"])
    e_lon = float(b["east_lon"])
    s_lat = float(b["south_lat"])
    n_lat = float(b["north_lat"])

    W = float(image_width)
    H = float(image_height)

    # Affine linear mapping from image grid to bounding box
    lon = w_lon + (px / W) * (e_lon - w_lon)
    lat = n_lat - (py / H) * (n_lat - s_lat)

    return (lat, lon)


def pixel_to_latlon_from_bounds_batch(
    points: list[tuple[float, float]],
    bounds: dict,
    image_width: int = 512,
    image_height: int = 512,
) -> list[tuple[float, float]]:
    """
    Convert multiple pixel coordinates to (lat, lon) in one call using shared bounds.

    Args:
        points: List of (px, py) tuples.
        bounds: Dict or object with keys west_lon, east_lon, south_lat, north_lat.
        image_width, image_height: Dimensions in pixels (default 512×512).

    Returns:
        List of (lat, lon) tuples.
    """
    if hasattr(bounds, "model_dump"):
        b = bounds.model_dump()
    elif hasattr(bounds, "dict"):
        b = bounds.dict()
    else:
        b = bounds

    w_lon = float(b["west_lon"])
    e_lon = float(b["east_lon"])
    s_lat = float(b["south_lat"])
    n_lat = float(b["north_lat"])

    W = float(image_width)
    H = float(image_height)

    d_lon = e_lon - w_lon
    d_lat = n_lat - s_lat

    return [
        (n_lat - (py / H) * d_lat, w_lon + (px / W) * d_lon)
        for px, py in points
    ]


# ---------------------------------------------------------------------------
# Homography re-derivation from match points
# ---------------------------------------------------------------------------

def _solve_perspective_matrix(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    Solve for a 3×3 perspective transform matrix H such that
    dst = H @ src (in homogeneous coordinates).
    """
    A = []
    for (sx, sy), (dx, dy) in zip(src_pts, dst_pts):
        A.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
        A.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])

    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    return H


def compute_homography_from_points(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
) -> list[list[float]] | None:
    """
    Re-derive a 3×3 homography matrix from matched point pairs.

    The ML team's matcher.py computes H via cv2.findHomography for RANSAC
    filtering but does not serialize it to matches.json. This function
    re-derives H from the surviving inlier points using a standard
    least-squares solve.

    Returns the 3×3 matrix as a list of lists, or None if fewer than 4
    point pairs are provided (underdetermined system).
    """
    if len(src_points) < 4:
        return None

    src = np.array(src_points, dtype=np.float64)
    dst = np.array(dst_points, dtype=np.float64)

    if len(src_points) == 4:
        H = _solve_perspective_matrix(src, dst)
    else:
        A = []
        for (sx, sy), (dx, dy) in zip(src, dst):
            A.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
            A.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])

        A_mat = np.array(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A_mat)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]

    return [[float(H[i, j]) for j in range(3)] for i in range(3)]


# ---------------------------------------------------------------------------
# Legacy functions (Unused in live path; kept for backward reference)
# ---------------------------------------------------------------------------

def pixel_to_latlon_from_corners(
    px: float,
    py: float,
    corners: dict,
    image_width: int = 512,
    image_height: int = 512,
) -> tuple[float, float]:
    """
    [LEGACY / UNUSED] Convert a pixel coordinate (px, py) to geographic (lat, lon)
    using 4 independent footprint corners.
    """
    W, H = float(image_width), float(image_height)
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
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
    pt = np.array([px, py, 1.0], dtype=np.float64)
    result = H_mat @ pt
    result /= result[2]
    return (result[1], result[0])


def pixel_to_latlon_batch(
    points: list[tuple[float, float]],
    corners: dict,
    image_width: int = 512,
    image_height: int = 512,
) -> list[tuple[float, float]]:
    """
    [LEGACY / UNUSED] Convert multiple pixel coordinates using 4 independent corners.
    """
    W, H_dim = float(image_width), float(image_height)
    src = np.array([[0, 0], [W, 0], [W, H_dim], [0, H_dim]], dtype=np.float64)
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
        results.append((r[1], r[0]))
    return results
