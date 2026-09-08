"""
distribution.py — Spatial uniformity and grid-based match filtering.

Enforces uniform distribution across image pairs by dividing the image
coordinate space into a 2D grid and filtering matches to retain high-confidence
representatives in each spatial cell.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def filter_uniform_matches(
    matches: list[dict[str, Any]],
    image_size: int = 512,
    grid_size: int = 8,
    max_per_cell: int = 2,
    min_confidence: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Filter match correspondences to ensure uniform spatial distribution.

    Args:
        matches: List of match dicts (must contain image1_x, image1_y, confidence).
        image_size: Standard tile dimension in pixels (default 512).
        grid_size: Number of bins along each axis (default 8 -> 64 cells).
        max_per_cell: Maximum number of matches to retain per grid cell.
        min_confidence: Minimum confidence threshold.

    Returns:
        Filtered list of match dictionaries sorted by confidence descending.
    """
    if not matches:
        return []

    cell_size = image_size / grid_size
    grid: dict[tuple[int, int], list[dict[str, Any]]] = {}

    # Sort matches by confidence descending so highest confidence matches come first
    sorted_matches = sorted(
        [m for m in matches if m.get("confidence", 1.0) >= min_confidence],
        key=lambda m: m.get("confidence", 1.0),
        reverse=True,
    )

    filtered: list[dict[str, Any]] = []

    for m in sorted_matches:
        x = float(m["image1_x"])
        y = float(m["image1_y"])

        # Determine cell index
        col = int(min(max(x // cell_size, 0), grid_size - 1))
        row = int(min(max(y // cell_size, 0), grid_size - 1))
        cell_key = (row, col)

        if cell_key not in grid:
            grid[cell_key] = []

        if len(grid[cell_key]) < max_per_cell:
            grid[cell_key].append(m)
            filtered.append(m)

    return filtered


def compute_distribution_uniformity(
    points: list[tuple[float, float]] | np.ndarray,
    image_size: int = 512,
    grid_size: int = 8,
) -> dict[str, float | int]:
    """
    Quantifies spatial coverage and entropy/uniformity of points across the image grid.
    """
    total_cells = grid_size * grid_size
    cell_size = image_size / grid_size

    if len(points) == 0:
        return {
            "occupied_cells": 0,
            "total_cells": total_cells,
            "coverage_ratio": 0.0,
            "uniformity_score": 0.0,
        }

    pts = np.asarray(points, dtype=np.float64)
    cols = np.clip(np.floor(pts[:, 0] / cell_size).astype(int), 0, grid_size - 1)
    rows = np.clip(np.floor(pts[:, 1] / cell_size).astype(int), 0, grid_size - 1)
    cell_indices = rows * grid_size + cols

    counts = np.bincount(cell_indices, minlength=total_cells)
    occupied = int(np.count_nonzero(counts))
    coverage = occupied / total_cells

    # Normalized Shannon Entropy (1.0 = perfectly uniform distribution among occupied cells)
    probs = counts[counts > 0] / len(pts)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(total_cells)
    uniformity = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        "occupied_cells": occupied,
        "total_cells": total_cells,
        "coverage_ratio": round(coverage, 4),
        "uniformity_score": round(uniformity, 4),
    }
