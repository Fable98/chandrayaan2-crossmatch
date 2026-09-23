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
        if ys.size:
            # Cap peak candidates before the Python loop: argpartition top-K
            # by response instead of materializing every plateau pixel.
            peak_budget = max(max_corners * 4, 2000)
            if ys.size > peak_budget:
                resp_all = img_f[ys, xs]
                top = np.argpartition(-resp_all, peak_budget)[:peak_budget]
                xs, ys = xs[top], ys[top]
            for x, y in zip(xs, ys):
                keypoints.append((float(x), float(y), float(img_f[y, x])))

    # Deduplicate very close points (within 1 px)
    if not keypoints:
        return []

    # Sort descending by response
    keypoints.sort(key=lambda k: k[2], reverse=True)
    # Trim to a bounded candidate pool before grid-hash dedup (argpartition,
    # not a full sort of 100k+ peaks on gigapixel scenes).
    dedup_budget = max(max_corners * 4, 2000)
    if len(keypoints) > dedup_budget:
        resp = np.fromiter((k[2] for k in keypoints), dtype=np.float32, count=len(keypoints))
        keep = np.argpartition(-resp, dedup_budget)[:dedup_budget]
        keep = keep[np.argsort(-resp[keep])]
        keypoints = [keypoints[i] for i in keep]
    # Grid-hash NMS: dict cell -> kept (no HxW bool array, gigapixel-safe).
    dedup: List[Tuple[float, float, float]] = []
    seen_cells = set()
    for x, y, r in keypoints:
        cell = (int(round(x)) // 2, int(round(y)) // 2)
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
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
    # Bound the O(n^2) ANMS pass: keep the top 4x pool by response via
    # argpartition (the winners always come from high-response candidates).
    pool = max(num_ret_points * 4, num_ret_points + 1)
    if len(kps) > pool:
        resp_all = np.fromiter((k[2] for k in kps), dtype=np.float32, count=len(kps))
        top = np.argpartition(-resp_all, pool)[:pool]
        top = top[np.argsort(-resp_all[top])]
        kps = [kps[i] for i in top]
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

    order = np.argpartition(-radii, min(num_ret_points, n - 1))[:num_ret_points]
    order = order[np.argsort(-radii[order])]
    return [kps[idx] for idx in order[:num_ret_points]]


# ---------------------------------------------------------------------------
# 3. Post-match Grid Density Budgeting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Post-match Grid Density Budgeting & Covariance-Aware Suppression
# ---------------------------------------------------------------------------

def extract_match_covariance_properties(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts or computes the 2x2 spatial covariance matrix, principal semi-axes
    (sigma_major, sigma_minor), ellipse orientation, and structural type from a match dictionary.
    """
    sigma_maj = m.get("sigma_major_px")
    sigma_min = m.get("sigma_minor_px")
    angle_deg = m.get("ellipse_angle_deg")

    cov_mat = None
    for key in ("covariance_xy", "cov", "covariance"):
        if key in m and m[key] is not None:
            c_cand = np.asarray(m[key], dtype=np.float64)
            if c_cand.shape == (2, 2) and np.all(np.isfinite(c_cand)):
                cov_mat = c_cand
                break

    if cov_mat is not None and (sigma_maj is None or sigma_min is None):
        # Spectral decomposition to recover semi-axes and orientation
        cov_sym = 0.5 * (cov_mat + cov_mat.T)
        eigvals, eigvecs = np.linalg.eigh(cov_sym)
        eigvals = np.clip(eigvals, 1e-4, 100.0)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        sigma_maj = float(np.sqrt(eigvals[0]))
        sigma_min = float(np.sqrt(eigvals[1]))
        if angle_deg is None:
            angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))) % 180.0

    if sigma_maj is None:
        sigma_maj = 0.5
    if sigma_min is None:
        sigma_min = 0.5
    if angle_deg is None:
        angle_deg = 0.0

    sigma_maj = max(float(sigma_maj), 1e-3)
    sigma_min = max(min(float(sigma_min), sigma_maj), 1e-3)
    angle_deg = float(angle_deg) % 180.0

    if cov_mat is None:
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        cov_mat = R @ np.diag([sigma_maj ** 2, sigma_min ** 2]) @ R.T

    tr = float(cov_mat[0, 0] + cov_mat[1, 1])
    anisotropy = float(sigma_maj / sigma_min)

    is_elongated = (anisotropy >= 2.0) and (sigma_min <= 0.6)
    is_isotropic = (anisotropy < 2.0) and (sigma_maj <= 1.2)
    is_uncertain = (sigma_min > 0.8) or (sigma_maj > 1.8 and not is_elongated)

    return {
        "cov": cov_mat,
        "sigma_major": sigma_maj,
        "sigma_minor": sigma_min,
        "ellipse_angle_deg": angle_deg,
        "trace": tr,
        "anisotropy": anisotropy,
        "is_elongated": is_elongated,
        "is_isotropic": is_isotropic,
        "is_uncertain": is_uncertain,
    }


def compute_covariance_aware_confidence(
    m: Dict[str, Any],
    alpha_pen: float = 0.5,
    beta_constr: float = 0.6,
) -> float:
    """
    Computes a covariance-aware confidence score for a candidate match:
    1. Base score S from correlation peak (e.g. NCC, CFOG, phase congruency).
    2. Penalizes high overall uncertainty: (sigma_major^0.7 * sigma_minor^0.5).
    3. Preserves useful elongated points when their constrained direction (sigma_minor) is valuable.
    4. Does not select only isotropic points.
    
    Formula:
        C_cov = S * [1 / (1 + alpha_pen * sigma_major^0.7 * sigma_minor^0.5)] * [1 + beta_constr / (1 + 2.0 * sigma_minor)]
    """
    score = float(m.get("score", m.get("confidence", 1.0)))
    cov_prop = extract_match_covariance_properties(m)
    sigma_maj = cov_prop["sigma_major"]
    sigma_min = cov_prop["sigma_minor"]

    # Penalty for overall uncertainty
    uncertainty_metric = (sigma_maj ** 0.7) * (sigma_min ** 0.5)
    penalty = 1.0 / (1.0 + alpha_pen * uncertainty_metric)

    # Constrained direction value bonus (higher when sigma_minor is small)
    constraint_bonus = 1.0 + beta_constr / (1.0 + 2.0 * sigma_min)

    cov_confidence = score * penalty * constraint_bonus
    return float(cov_confidence)


def compute_cell_covariance_statistics(
    selected_matches: Sequence[Dict[str, Any]],
    image_shape: Tuple[int, int] = (512, 512),
    grid_dims: Tuple[int, int] = (10, 10),
) -> Dict[str, Any]:
    """
    Computes selected-point covariance statistics per grid cell and globally across the image.
    
    Returns:
        Dict containing per-cell statistics and global summary statistics.
    """
    h, w = image_shape[:2]
    gw, gh = grid_dims
    cell_w = max(1.0, float(w) / float(gw))
    cell_h = max(1.0, float(h) / float(gh))

    cell_bins: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for m in selected_matches:
        x = float(m.get("work_x1", m.get("source_x", m.get("image1_x", 0.0))))
        y = float(m.get("work_y1", m.get("source_y", m.get("image1_y", 0.0))))
        gx = min(gw - 1, max(0, int(x / cell_w)))
        gy = min(gh - 1, max(0, int(y / cell_h)))
        cell_bins.setdefault((gx, gy), []).append(m)

    per_cell_stats: Dict[str, Dict[str, Any]] = {}
    all_majors: List[float] = []
    all_minors: List[float] = []
    all_anisotropies: List[float] = []
    total_iso = 0
    total_elong = 0
    total_unc = 0

    for (gx, gy), items in sorted(cell_bins.items()):
        cell_props = [extract_match_covariance_properties(it) for it in items]
        majors = [p["sigma_major"] for p in cell_props]
        minors = [p["sigma_minor"] for p in cell_props]
        anisotropies = [p["anisotropy"] for p in cell_props]
        traces = [p["trace"] for p in cell_props]
        angles = [p["ellipse_angle_deg"] for p in cell_props]

        all_majors.extend(majors)
        all_minors.extend(minors)
        all_anisotropies.extend(anisotropies)

        n_iso = sum(1 for p in cell_props if p["is_isotropic"])
        n_elong = sum(1 for p in cell_props if p["is_elongated"])
        n_unc = sum(1 for p in cell_props if p["is_uncertain"])

        total_iso += n_iso
        total_elong += n_elong
        total_unc += n_unc

        # Circular mean of angles
        rads = [np.radians(2.0 * a) for a in angles]
        mean_sin = float(np.mean(np.sin(rads))) if rads else 0.0
        mean_cos = float(np.mean(np.cos(rads))) if rads else 1.0
        dominant_deg = float(np.degrees(0.5 * np.arctan2(mean_sin, mean_cos))) % 180.0

        scores = [float(it.get("score", it.get("confidence", 1.0))) for it in items]
        cov_scores = [float(it.get("covariance_aware_confidence", scores[idx])) for idx, it in enumerate(items)]

        per_cell_stats[f"cell_{gx}_{gy}"] = {
            "cell_coord": [int(gx), int(gy)],
            "num_selected": len(items),
            "mean_sigma_major_px": round(float(np.mean(majors)), 4),
            "mean_sigma_minor_px": round(float(np.mean(minors)), 4),
            "mean_anisotropy": round(float(np.mean(anisotropies)), 4),
            "mean_trace_cov": round(float(np.mean(traces)), 4),
            "mean_confidence": round(float(np.mean(scores)), 4),
            "mean_covariance_aware_confidence": round(float(np.mean(cov_scores)), 4),
            "dominant_orientation_deg": round(dominant_deg, 2),
            "counts_by_type": {
                "isotropic": n_iso,
                "elongated": n_elong,
                "uncertain": n_unc,
            },
        }

    total_cells = gw * gh
    occupied_count = len(cell_bins)
    occupancy_rate = float(occupied_count) / float(total_cells) if total_cells > 0 else 0.0

    global_stats = {
        "total_cells": total_cells,
        "occupied_cells_count": occupied_count,
        "occupancy_rate": round(occupancy_rate, 4),
        "total_selected_matches": len(selected_matches),
        "global_mean_sigma_major_px": round(float(np.mean(all_majors)), 4) if all_majors else 0.0,
        "global_mean_sigma_minor_px": round(float(np.mean(all_minors)), 4) if all_minors else 0.0,
        "global_mean_anisotropy": round(float(np.mean(all_anisotropies)), 4) if all_anisotropies else 1.0,
        "total_isotropic": total_iso,
        "total_elongated": total_elong,
        "total_uncertain": total_unc,
        "elongated_fraction": round(float(total_elong) / max(len(selected_matches), 1), 4),
    }

    return {
        "grid_dims": [int(gw), int(gh)],
        "global_statistics": global_stats,
        "per_cell_statistics": per_cell_stats,
    }


def apply_grid_density_budgeting(
    matches: List[Dict[str, Any]],
    image_shape: Tuple[int, int] = (512, 512),
    grid_dims: Tuple[int, int] = (10, 10),
    max_per_cell: int = 4,
    total_budget: Optional[int] = None,
    use_covariance: bool = True,
    alpha_pen: float = 0.5,
    beta_constr: float = 0.6,
    return_cell_stats: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Post-match Grid Density Budgeting with Covariance-Aware Suppression.
    
    Guarantees:
    1. Hard Grid Occupancy Constraint:
       Every occupied cell receives representation in Round 1 before dense cells
       receive additional slots. Dense hotspots cannot starve under-represented cells.
    2. Covariance-Aware Confidence:
       Candidates within each cell are ranked using covariance-aware confidence.
    3. Uncertainty Penalty:
       Penalizes points with high overall uncertainty.
    4. Anisotropic Inclusivity:
       Does NOT select only isotropic points.
    5. Preservation of Useful Elongated Points:
       Preserves elongated points when their constrained direction (sigma_minor) is sharp.
    6. Directional Complementarity:
       In subsequent allocation rounds (Round 2+), prioritizes complementary orientations
       to provide bidirectional constraints within cells.
    7. Cell Covariance Statistics Reporting:
       If return_cell_stats=True, reports per-cell covariance statistics and global summary.

    Args:
        matches: List of match dictionaries.
        image_shape: (height, width) of the image space.
        grid_dims: (grid_cols, grid_rows), typically (10, 10).
        max_per_cell: Maximum candidates any single cell may contribute.
        total_budget: Optional overall maximum candidates to return.
        use_covariance: If True, uses covariance-aware confidence scoring; otherwise scalar score.
        alpha_pen: Uncertainty penalty weighting.
        beta_constr: Constrained direction bonus weighting.
        return_cell_stats: If True, returns (selected_matches, cell_stats_dict).

    Returns:
        Filtered list of match dictionaries, or (selected_matches, cell_stats_dict) if return_cell_stats=True.
    """
    if not matches:
        empty_stats = compute_cell_covariance_statistics([], image_shape=image_shape, grid_dims=grid_dims)
        if return_cell_stats:
            return [], empty_stats
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

        # Compute and annotate covariance properties
        if use_covariance:
            cov_conf = compute_covariance_aware_confidence(m, alpha_pen=alpha_pen, beta_constr=beta_constr)
            cov_props = extract_match_covariance_properties(m)
            m_copy = dict(m)
            m_copy["covariance_aware_confidence"] = round(cov_conf, 6)
            m_copy["sigma_major_px"] = round(cov_props["sigma_major"], 4)
            m_copy["sigma_minor_px"] = round(cov_props["sigma_minor"], 4)
            m_copy["ellipse_angle_deg"] = round(cov_props["ellipse_angle_deg"], 2)
            m_copy["anisotropy"] = round(cov_props["anisotropy"], 2)
        else:
            m_copy = dict(m)
            m_copy["covariance_aware_confidence"] = float(m.get("score", m.get("confidence", 1.0)))

        cell_bins.setdefault((gx, gy), []).append(m_copy)

    # Sort matches inside each cell descending by covariance-aware confidence
    for cell, items in cell_bins.items():
        items.sort(
            key=lambda it: float(it.get("covariance_aware_confidence", it.get("score", it.get("confidence", 1.0)))),
            reverse=True,
        )

    # Tiered Round-Robin Selection:
    # Round 0 guarantees hard grid occupancy constraint (every occupied cell gets 1 slot)
    selected: List[Dict[str, Any]] = []
    selected_by_cell: Dict[Tuple[int, int], List[Dict[str, Any]]] = {c: [] for c in cell_bins}
    occupied_cells = sorted(cell_bins.keys())

    for round_idx in range(max_per_cell):
        for cell in occupied_cells:
            items = cell_bins[cell]
            already_selected = selected_by_cell[cell]
            if len(already_selected) >= len(items):
                continue

            if round_idx == 0:
                # Round 0: Best candidate in cell (hard occupancy constraint)
                chosen = items[0]
            else:
                # Round 1+: Choose remaining candidate that maximizes covariance confidence
                # with a bonus for directional diversity relative to already-chosen candidates
                remaining = [it for it in items if it not in already_selected]
                if not remaining:
                    continue

                def _selection_rank(cand: Dict[str, Any]) -> float:
                    base_conf = float(cand.get("covariance_aware_confidence", 1.0))
                    cand_ang = float(cand.get("ellipse_angle_deg", 0.0))
                    # Compute minimum angle difference to already selected points in cell
                    diffs = [
                        abs(cand_ang - float(prev.get("ellipse_angle_deg", 0.0)))
                        for prev in already_selected
                    ]
                    # Map to [0, 90] deg line orientation difference
                    diffs_folded = [min(d % 180.0, 180.0 - (d % 180.0)) for d in diffs]
                    min_diff = min(diffs_folded) if diffs_folded else 90.0
                    # Orthogonal candidates (|diff| ~ 90 deg) receive up to 25% diversity bonus
                    diversity_factor = 1.0 + 0.25 * float(np.sin(np.radians(min_diff * 2.0)))
                    return base_conf * diversity_factor

                chosen = max(remaining, key=_selection_rank)

            selected_by_cell[cell].append(chosen)
            selected.append(chosen)

            if total_budget is not None and len(selected) >= total_budget:
                break
        if total_budget is not None and len(selected) >= total_budget:
            break

    if return_cell_stats:
        stats = compute_cell_covariance_statistics(selected, image_shape=image_shape, grid_dims=grid_dims)
        return selected, stats

    return selected

