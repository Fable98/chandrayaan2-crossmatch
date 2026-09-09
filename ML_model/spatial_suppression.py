"""
spatial_suppression.py — Pre-match and post-match spatial suppression algorithms.

Implements:
1. Pre-match: Adaptive Non-Maximal Suppression (ANMS) and Suppression via
   Square Covering (SSC, Bailo et al. PRL 2018) for homogeneous spatial keypoint
   distribution across multi-sensor lunar imagery.
2. Post-match: Grid Density Budgeting on an NxN grid (default 10x10), actively
   prioritizing candidate correspondences from under-represented cells before
   dense texture hotspots receive additional allocations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("spatial_suppression")


# ---------------------------------------------------------------------------
# 1. Pre-match Keypoint Detection
# ---------------------------------------------------------------------------

def detect_salient_keypoints(
    image: np.ndarray,
    max_corners: int = 500,
    quality_level: float = 0.01,
    min_distance: float = 2.0,
    use_pc_peaks: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Detects salient structural keypoints across an image.
    Returns list of (x, y, response).
    
    Combines Shi-Tomasi cornerness and Phase Congruency peaks when available.
    """
    img_f = np.asarray(image, dtype=np.float32)
    if img_f.ndim == 3:
        img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)

    h, w = img_f.shape[:2]
    if h < 8 or w < 8 or float(np.std(img_f)) < 1e-6:
        return []

    # Normalize to 0..255 uint8 for standard detectors
    mn, mx = float(np.nanmin(img_f)), float(np.nanmax(img_f))
    denom = max(mx - mn, 1e-6)
    img_u8 = np.clip((img_f - mn) / denom * 255.0, 0, 255).astype(np.uint8)

    keypoints: List[Tuple[float, float, float]] = []

    # 1. Shi-Tomasi Corner Detector (Good Features to Track)
    corners = cv2.goodFeaturesToTrack(
        img_u8,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )
    if corners is not None:
        for c in corners:
            x, y = float(c[0, 0]), float(c[0, 1])
            # Response: sample local gradient magnitude / intensity
            ix, iy = int(round(x)), int(round(y))
            ix = min(w - 1, max(0, ix))
            iy = min(h - 1, max(0, iy))
            resp = float(img_f[iy, ix])
            keypoints.append((x, y, resp))

    # 2. Local Extrema / Peaks (especially for Phase Congruency maps)
    if use_pc_peaks:
        kernel_size = 5
        dilated = cv2.dilate(img_f, np.ones((kernel_size, kernel_size), np.uint8))
        peaks = (img_f == dilated) & (img_f > np.percentile(img_f, 75.0))
        ys, xs = np.nonzero(peaks)
        for x, y in zip(xs, ys):
            keypoints.append((float(x), float(y), float(img_f[y, x])))

    # Deduplicate very close points (within 1 px)
    if not keypoints:
        return []

    # Sort descending by response
    keypoints.sort(key=lambda k: k[2], reverse=True)
    dedup: List[Tuple[float, float, float]] = []
    seen = np.zeros((h, w), dtype=bool)
    for x, y, r in keypoints:
        ix, iy = int(round(x)), int(round(y))
        ix = min(w - 1, max(0, ix))
        iy = min(h - 1, max(0, iy))
        if not seen[iy, ix]:
            seen[max(0, iy - 1) : min(h, iy + 2), max(0, ix - 1) : min(w, ix + 2)] = True
            dedup.append((x, y, r))

    return dedup


# ---------------------------------------------------------------------------
# 2. Pre-match Adaptive Non-Maximal Suppression (ANMS / SSC)
# ---------------------------------------------------------------------------

def suppression_via_square_covering(
    keypoints: Sequence[Tuple[float, float, float]],
    num_ret_points: int,
    tolerance: float = 0.1,
    cols: int = 512,
    rows: int = 512,
) -> List[Tuple[float, float, float]]:
    """
    Suppression via Square Covering (SSC) for homogeneous spatial keypoint distribution.
    
    Reference:
    Bailo, Rameau, Joo, Park, Bogdan, Kweon: "Efficient adaptive non-maximal
    suppression algorithms for homogeneous spatial keypoint distribution."
    Pattern Recognition Letters, 2018.

    Args:
        keypoints: List of (x, y, response) tuples.
        num_ret_points: Desired number of spatially distributed points to retain.
        tolerance: Fractional tolerance on num_ret_points (e.g. 0.1 = +/- 10%).
        cols: Image width in pixels.
        rows: Image height in pixels.

    Returns:
        List of retained (x, y, response) keypoints with maximal spatial dispersion.
    """
    if len(keypoints) <= num_ret_points or num_ret_points <= 0:
        return list(keypoints)

    # Keypoints must be sorted descending by response
    kps = sorted(keypoints, key=lambda k: k[2], reverse=True)

    low = 1
    high = max(cols, rows)
    prev_r = -1
    result: List[Tuple[float, float, float]] = list(kps[:num_ret_points])

    # Binary search over square covering radius r
    while low < high:
        r = (low + high) // 2
        if r == prev_r:
            break
        prev_r = r

        cell_size = max(1, r)
        grid_w = int(np.ceil(cols / cell_size))
        grid_h = int(np.ceil(rows / cell_size))
        grid = np.zeros((grid_h, grid_w), dtype=bool)

        covered: List[Tuple[float, float, float]] = []
        for kp in kps:
            gx = min(grid_w - 1, max(0, int(kp[0] / cell_size)))
            gy = min(grid_h - 1, max(0, int(kp[1] / cell_size)))
            if not grid[gy, gx]:
                grid[gy, gx] = True
                covered.append(kp)

        num_cov = len(covered)
        if num_cov > num_ret_points * (1.0 + tolerance):
            low = r + 1
        elif num_cov < num_ret_points * (1.0 - tolerance):
            high = r - 1
        else:
            result = covered
            break
        result = covered

    # Enforce upper bound if result exceeds num_ret_points
    if len(result) > num_ret_points:
        result = result[:num_ret_points]

    return result


def standard_anms(
    keypoints: Sequence[Tuple[float, float, float]],
    num_ret_points: int,
    c_robust: float = 0.9,
) -> List[Tuple[float, float, float]]:
    """
    Standard Brown et al. (MOPS 2005) Adaptive Non-Maximal Suppression.
    
    For each keypoint i:
        r_i = min_{j: score_j > c_robust * score_i} || p_i - p_j ||
    Points with largest suppression radii r_i are selected.
    """
    if len(keypoints) <= num_ret_points:
        return list(keypoints)

    kps = sorted(keypoints, key=lambda k: k[2], reverse=True)
    n = len(kps)
    pts = np.array([[k[0], k[1]] for k in kps], dtype=np.float32)
    scores = np.array([k[2] for k in kps], dtype=np.float32)

    radii = np.full(n, np.inf, dtype=np.float32)

    for i in range(n):
        # Only compare with points having significantly higher score
        higher_mask = scores[:i] > (c_robust * scores[i])
        if np.any(higher_mask):
            diffs = pts[:i][higher_mask] - pts[i]
            dists_sq = np.sum(diffs**2, axis=1)
            radii[i] = float(np.min(dists_sq))

    order = np.argsort(-radii)
    return [kps[idx] for idx in order[:num_ret_points]]


# ---------------------------------------------------------------------------
# 3. Post-match Grid Density Budgeting
# ---------------------------------------------------------------------------

def apply_grid_density_budgeting(
    matches: List[Dict[str, Any]],
    image_shape: Tuple[int, int] = (512, 512),
    grid_dims: Tuple[int, int] = (10, 10),
    max_per_cell: int = 4,
    total_budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Post-match Grid Density Budgeting.
    
    Partitions the candidate matches into an NxN grid (default 10x10) and
    actively prioritizes matches from under-represented cells before dense
    cells receive additional candidate slots.
    
    Unlike a flat independent per-cell cap (which lets dense texture regions
    monopolize RANSAC's candidate pool while sparse regions are drowned out),
    density budgeting uses tiered round-robin allocation:
    - Round 1: Every occupied cell contributes its top-1 highest confidence match.
      This gives under-represented cells (with only 1 or 2 candidates) immediate
      and equal representation.
    - Round 2..max_per_cell: Cells with remaining candidates contribute their next
      best matches until the per-cell cap or overall budget is reached.

    Args:
        matches: List of match dictionaries (must have work_x1/y1 or source_x/y).
        image_shape: (height, width) of the image space.
        grid_dims: (grid_cols, grid_rows), typically (10, 10).
        max_per_cell: Maximum candidates any single cell may contribute (cap).
        total_budget: Optional overall maximum candidates to return.

    Returns:
        Filtered and spatially budgeted list of match dictionaries.
    """
    if not matches:
        return []

    h, w = image_shape[:2]
    gw, gh = grid_dims
    cell_w = max(1.0, float(w) / float(gw))
    cell_h = max(1.0, float(h) / float(gh))

    # Partition matches into (gx, gy) grid cells
    cell_bins: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for m in matches:
        x = float(m.get("work_x1", m.get("source_x", m.get("image1_x", 0.0))))
        y = float(m.get("work_y1", m.get("source_y", m.get("image1_y", 0.0))))
        gx = min(gw - 1, max(0, int(x / cell_w)))
        gy = min(gh - 1, max(0, int(y / cell_h)))
        cell_bins.setdefault((gx, gy), []).append(m)

    # Sort matches inside each cell descending by confidence/score
    for cell, items in cell_bins.items():
        items.sort(
            key=lambda it: float(it.get("score", it.get("confidence", 1.0))),
            reverse=True,
        )

    # Tiered Round-Robin Selection:
    # Under-represented cells get representation in Round 1 before dense cells get 2nd/3rd slots.
    selected: List[Dict[str, Any]] = []
    occupied_cells = sorted(cell_bins.keys())

    for round_idx in range(max_per_cell):
        for cell in occupied_cells:
            items = cell_bins[cell]
            if round_idx < len(items):
                selected.append(items[round_idx])
                if total_budget is not None and len(selected) >= total_budget:
                    return selected

    return selected
