"""
ML_model/spatial_distribution.py — Uniform spatial distribution filter
for Chandrayaan-2 tie-points (OHRC / TMC / IIRS registration).

Coordinate convention (STRICT, never swap):
    (x, y) means (column, row).
    x ranges from 0 to image_width - 1.
    y ranges from 0 to image_height - 1.

Enforces uniform distribution by dividing the image frame into a
grid_rows x grid_cols grid and keeping the top-K highest-confidence
matches per cell. Lightweight: only numpy + logging.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _to_point_array(arr, name: str) -> np.ndarray:
    """Convert arbitrary input to (N, 2) float32 array. Never raises on shape."""
    try:
        a = np.asarray(arr, dtype=np.float32)
    except Exception:
        logger.warning("%s could not be converted to array; treating as empty.", name)
        return np.zeros((0, 2), dtype=np.float32)
    if a.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if a.ndim == 1:
        # Flat list: must contain an even number of values to reshape to (N, 2).
        if a.size % 2 != 0:
            logger.warning(
                "%s has odd flat size %d; cannot reshape to (N, 2). Truncating last value.",
                name, a.size,
            )
            a = a[:-1]
            if a.size == 0:
                return np.zeros((0, 2), dtype=np.float32)
        return a.reshape(-1, 2).astype(np.float32, copy=False)
    if a.ndim == 2:
        if a.shape[1] == 2:
            return a.astype(np.float32, copy=False)
        # Wrong second dim but total size divisible by 2 -> reshape.
        if a.size % 2 == 0:
            logger.warning(
                "%s has shape %s; reshaping to (N, 2). Expected (N, 2) with (x=col, y=row).",
                name, a.shape,
            )
            return a.reshape(-1, 2).astype(np.float32, copy=False)
        logger.warning("%s has incompatible shape %s; treating as empty.", name, a.shape)
        return np.zeros((0, 2), dtype=np.float32)
    # ndim >= 3: try to flatten pairs if possible, else empty.
    if a.size % 2 == 0:
        logger.warning(
            "%s has shape %s; flattening to (N, 2). Expected (N, 2).", name, a.shape
        )
        return a.reshape(-1, 2).astype(np.float32, copy=False)
    logger.warning("%s has incompatible shape %s; treating as empty.", name, a.shape)
    return np.zeros((0, 2), dtype=np.float32)


class UniformDistributionFilter:
    """Enforce uniform spatial distribution of matched tie-points via grid binning."""

    def __init__(self, grid_rows: int = 8, grid_cols: int = 8, points_per_cell: int = 5):
        try:
            grid_rows = int(grid_rows)
        except Exception:
            logger.warning("grid_rows=%r invalid; clamping to 1.", grid_rows)
            grid_rows = 1
        try:
            grid_cols = int(grid_cols)
        except Exception:
            logger.warning("grid_cols=%r invalid; clamping to 1.", grid_cols)
            grid_cols = 1
        try:
            points_per_cell = int(points_per_cell)
        except Exception:
            logger.warning("points_per_cell=%r invalid; clamping to 1.", points_per_cell)
            points_per_cell = 1

        if grid_rows < 1:
            logger.warning("grid_rows=%d < 1; clamping to 1.", grid_rows)
            grid_rows = 1
        if grid_cols < 1:
            logger.warning("grid_cols=%d < 1; clamping to 1.", grid_cols)
            grid_cols = 1
        if points_per_cell < 1:
            logger.warning("points_per_cell=%d < 1; clamping to 1.", points_per_cell)
            points_per_cell = 1

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.points_per_cell = points_per_cell

    def _empty_result(self, original_count: int) -> dict:
        total_cells = int(self.grid_rows * self.grid_cols)
        _ = total_cells  # for clarity
        return {
            "status": "empty",
            "filtered_src_pts": np.zeros((0, 2), dtype=np.float32),
            "filtered_ref_pts": np.zeros((0, 2), dtype=np.float32),
            "filtered_confidence": np.zeros((0,), dtype=np.float32),
            "original_count": int(original_count),
            "filtered_count": 0,
            "coverage_ratio": 0.0,
            "balance_score": 0.0,
            "grid_occupancy": np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32),
        }

    def filter_points(
        self,
        src_pts,
        ref_pts,
        image_width,
        image_height,
        confidence_scores=None,
    ) -> dict:
        # ---- 1. Input validation ----
        src = _to_point_array(src_pts, "src_pts")
        ref = _to_point_array(ref_pts, "ref_pts")

        n_src = int(src.shape[0])
        n_ref = int(ref.shape[0])
        if n_src != n_ref:
            n_min = min(n_src, n_ref)
            logger.warning(
                "src_pts (%d) and ref_pts (%d) lengths differ; truncating both to %d.",
                n_src, n_ref, n_min,
            )
            src = src[:n_min]
            ref = ref[:n_min]
        n = int(src.shape[0])
        original_count = n

        if n == 0:
            return self._empty_result(original_count)

        # Confidence scores
        if confidence_scores is None:
            conf = np.ones((n,), dtype=np.float32)
        else:
            try:
                conf = np.asarray(confidence_scores, dtype=np.float32).ravel()
            except Exception:
                logger.warning("confidence_scores could not be parsed; using uniform 1.0.")
                conf = np.ones((n,), dtype=np.float32)
            if conf.size != n:
                logger.warning(
                    "confidence_scores length (%d) != points (%d); using uniform 1.0.",
                    conf.size, n,
                )
                conf = np.ones((n,), dtype=np.float32)
            else:
                conf = conf.astype(np.float32, copy=False)

        # Image dims guard
        try:
            w = float(image_width)
            h = float(image_height)
        except Exception:
            logger.warning("Invalid image dims (%r, %r); returning empty.", image_width, image_height)
            return self._empty_result(original_count)
        if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
            logger.warning("Invalid image dims (%r, %r); returning empty.", image_width, image_height)
            return self._empty_result(original_count)
        image_width = w
        image_height = h

        # ---- 2. Boundary filtering (x=col, y=row on src frame) ----
        # Keep points with 0 <= x < image_width and 0 <= y < image_height.
        xs = src[:, 0]
        ys = src[:, 1]
        mask = (xs >= 0) & (ys >= 0) & (xs < image_width) & (ys < image_height)
        src = src[mask]
        ref = ref[mask]
        conf = conf[mask]
        if src.shape[0] == 0:
            return self._empty_result(original_count)

        # ---- 3. Grid assignment ----
        cell_width = float(image_width) / float(self.grid_cols)
        cell_height = float(image_height) / float(self.grid_rows)
        # Guard against degenerate zero-size cells (should not happen since dims > 0, grids >= 1).
        if not np.isfinite(cell_width) or cell_width <= 0:
            cell_width = float(image_width)
        if not np.isfinite(cell_height) or cell_height <= 0:
            cell_height = float(image_height)

        # (x, y) = (col, row); cell_x from x, cell_y from y. Never swap.
        cell_x = np.clip(
            (src[:, 0] / cell_width).astype(np.int64),
            0, self.grid_cols - 1,
        )
        cell_y = np.clip(
            (src[:, 1] / cell_height).astype(np.int64),
            0, self.grid_rows - 1,
        )

        # ---- 4. Per-cell selection ----
        cells: dict[tuple[int, int], list[int]] = {}
        for i in range(src.shape[0]):
            key = (int(cell_x[i]), int(cell_y[i]))
            cells.setdefault(key, []).append(i)

        selected_idx: list[int] = []
        for key, idx_list in cells.items():
            # Sort by confidence descending.
            idx_list_sorted = sorted(idx_list, key=lambda j: float(conf[j]), reverse=True)
            selected_idx.extend(idx_list_sorted[: self.points_per_cell])

        selected_idx_arr = np.asarray(selected_idx, dtype=np.int64)
        filtered_src = src[selected_idx_arr].astype(np.float32, copy=False)
        filtered_ref = ref[selected_idx_arr].astype(np.float32, copy=False)
        filtered_conf = conf[selected_idx_arr].astype(np.float32, copy=False)
        # Ensure shapes (M, 2) / (M,)
        if filtered_src.size == 0:
            filtered_src = np.zeros((0, 2), dtype=np.float32)
            filtered_ref = np.zeros((0, 2), dtype=np.float32)
            filtered_conf = np.zeros((0,), dtype=np.float32)
        else:
            filtered_src = np.reshape(filtered_src, (-1, 2)).astype(np.float32, copy=False)
            filtered_ref = np.reshape(filtered_ref, (-1, 2)).astype(np.float32, copy=False)
            filtered_conf = np.reshape(filtered_conf, (-1,)).astype(np.float32, copy=False)

        m = int(filtered_src.shape[0])

        # ---- 5. Distribution metrics ----
        grid_occupancy = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        # cell_y is row index, cell_x is column index -> occupancy[row, col].
        sel_cx = cell_x[selected_idx_arr] if m > 0 else np.zeros((0,), dtype=np.int64)
        sel_cy = cell_y[selected_idx_arr] if m > 0 else np.zeros((0,), dtype=np.int64)
        for cx_i, cy_i in zip(sel_cx.tolist(), sel_cy.tolist()):
            grid_occupancy[int(cy_i), int(cx_i)] += 1

        total_cells = int(self.grid_rows * self.grid_cols)
        occupied_cells = int(np.count_nonzero(grid_occupancy))
        coverage_ratio = float(occupied_cells / total_cells) if total_cells > 0 else 0.0

        counts = grid_occupancy.astype(np.float64).ravel()
        std_dev = float(np.std(counts)) if counts.size > 0 else 0.0
        if m == 0 or total_cells <= 1:
            balance_score = 1.0 if m > 0 and total_cells <= 1 else 0.0
            if m == 0:
                balance_score = 0.0
        else:
            mean = float(m) / float(total_cells)
            # Max std: all M points in a single cell, rest empty.
            var_max = ((float(m) - mean) ** 2 + float(total_cells - 1) * (mean ** 2)) / float(total_cells)
            std_max = float(np.sqrt(max(var_max, 0.0)))
            if std_max <= 1e-12:
                balance_score = 1.0
            else:
                balance_score = 1.0 - (std_dev / std_max)
            balance_score = float(np.clip(balance_score, 0.0, 1.0))

        coverage_ratio = float(np.clip(coverage_ratio, 0.0, 1.0))

        # ---- 6. Return ----
        return {
            "status": "success",
            "filtered_src_pts": filtered_src,
            "filtered_ref_pts": filtered_ref,
            "filtered_confidence": filtered_conf,
            "original_count": int(original_count),
            "filtered_count": int(m),
            "coverage_ratio": float(coverage_ratio),
            "balance_score": float(balance_score),
            "grid_occupancy": grid_occupancy,
        }
