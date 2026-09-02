"""
error_analysis.py — Visual error analysis and distribution plots.

Generates publication-quality visualizations for the evaluation report:
  1. Reprojection error heatmap overlaid on the source image
  2. Match distribution scatter plot (source + destination)
  3. Error histogram
  4. Spatial coverage grid

All plots are saved as PNGs to evaluation_output/.

Usage:
    python -m ml.evaluation.error_analysis
    -- or --
    python error_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure 'src' is on sys.path for standalone script execution
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Support running both as a package and standalone
try:
    from ml.evaluation.metrics import (
        compute_all_metrics,
        reprojection_errors,
        spatial_coverage,
    )
except ImportError:
    from .metrics import (
        compute_all_metrics,
        reprojection_errors,
        spatial_coverage,
    )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for ancestor in [p] + list(p.parents):
        if (ancestor / "data_preprocessing_pipeline").is_dir() and (ancestor / "ML_model").is_dir():
            return ancestor
    return p.parent.parent.parent.parent.parent


REPO_ROOT = _find_repo_root()
PROCESSED_DIR = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets"
OUTPUT_DIR = REPO_ROOT / "evaluation_output"


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def draw_match_distribution(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    src_img: np.ndarray | None,
    dst_img: np.ndarray | None,
    output_path: Path,
    image_size: int = 512,
    grid_size: int = 8,
    title: str = "",
) -> None:
    """
    Draw match distribution scatter plot on source and destination images.
    Shows grid overlay and colors points by confidence (if available).
    """
    cell = image_size // grid_size

    # Create side-by-side canvas
    canvas_w = image_size * 2 + 40
    canvas_h = image_size + 80
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30  # dark bg

    # Place images
    if src_img is not None:
        img_s = cv2.resize(src_img, (image_size, image_size))
        if img_s.ndim == 2:
            img_s = cv2.cvtColor(img_s, cv2.COLOR_GRAY2BGR)
        canvas[40:40 + image_size, 0:image_size] = img_s
    if dst_img is not None:
        img_d = cv2.resize(dst_img, (image_size, image_size))
        if img_d.ndim == 2:
            img_d = cv2.cvtColor(img_d, cv2.COLOR_GRAY2BGR)
        canvas[40:40 + image_size, image_size + 40:] = img_d

    # Draw grid on both
    for side_offset in [0, image_size + 40]:
        for i in range(1, grid_size):
            x = side_offset + i * cell
            cv2.line(canvas, (x, 40), (x, 40 + image_size), (80, 80, 80), 1)
            y = 40 + i * cell
            cv2.line(canvas, (side_offset, y), (side_offset + image_size, y), (80, 80, 80), 1)

    # Draw match points on source (cyan) and destination (green)
    for pt in src_pts:
        x, y = int(pt[0]), int(pt[1]) + 40
        cv2.circle(canvas, (x, y), 5, (255, 255, 0), -1)  # cyan
        cv2.circle(canvas, (x, y), 5, (200, 200, 0), 1)

    for pt in dst_pts:
        x, y = int(pt[0]) + image_size + 40, int(pt[1]) + 40
        cv2.circle(canvas, (x, y), 5, (0, 255, 128), -1)  # green
        cv2.circle(canvas, (x, y), 5, (0, 200, 100), 1)

    # Labels
    cv2.putText(canvas, "Source (OHRC)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(canvas, "Destination (TMC)", (image_size + 50, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    src_cov = spatial_coverage(src_pts, image_size, grid_size)
    dst_cov = spatial_coverage(dst_pts, image_size, grid_size)
    info = f"Src coverage: {src_cov['coverage_ratio']:.0%} ({src_cov['occupied_cells']}/{src_cov['total_cells']})   |   Dst coverage: {dst_cov['coverage_ratio']:.0%} ({dst_cov['occupied_cells']}/{dst_cov['total_cells']})"
    cv2.putText(canvas, info, (10, canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if title:
        cv2.putText(canvas, title, (canvas_w // 2 - 100, canvas_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def draw_reprojection_error_map(
    src_pts: np.ndarray,
    errors: np.ndarray,
    src_img: np.ndarray | None,
    output_path: Path,
    image_size: int = 512,
) -> None:
    """
    Draw per-point reprojection errors as colored circles on the source image.
    Colors: green (low error) → yellow → red (high error).
    """
    if src_img is not None:
        canvas = cv2.resize(src_img, (image_size, image_size))
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        canvas = canvas.copy()
    else:
        canvas = np.ones((image_size, image_size, 3), dtype=np.uint8) * 40

    if len(errors) == 0:
        cv2.imwrite(str(output_path), canvas)
        return

    # Normalize errors for color mapping
    max_err = max(float(np.max(errors)), 1e-6)
    for pt, err in zip(src_pts, errors):
        x, y = int(pt[0]), int(pt[1])
        # Map error to color: green (0) → red (max)
        ratio = min(err / max_err, 1.0)
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        color = (0, g, r)  # BGR
        radius = max(4, int(6 + 8 * ratio))
        cv2.circle(canvas, (x, y), radius, color, -1)
        cv2.circle(canvas, (x, y), radius, (255, 255, 255), 1)
        # Label with error value
        cv2.putText(canvas, f"{err:.1f}", (x + radius + 2, y + 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Legend
    cv2.putText(canvas, f"Max err: {max_err:.2f}px", (10, 20),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(canvas, (10, 30), (30, 45), (0, 255, 0), -1)
    cv2.putText(canvas, "Low", (35, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.rectangle(canvas, (70, 30), (90, 45), (0, 0, 255), -1)
    cv2.putText(canvas, "High", (95, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def draw_error_histogram(
    errors: np.ndarray,
    output_path: Path,
    bins: int = 20,
) -> None:
    """
    Draw a simple error histogram using OpenCV drawing primitives.
    No matplotlib dependency.
    """
    if len(errors) == 0:
        canvas = np.ones((300, 500, 3), dtype=np.uint8) * 40
        cv2.putText(canvas, "No data", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.imwrite(str(output_path), canvas)
        return

    # Compute histogram
    hist_counts, bin_edges = np.histogram(errors, bins=bins)
    max_count = max(int(np.max(hist_counts)), 1)

    # Canvas
    margin_l, margin_b, margin_t, margin_r = 60, 50, 40, 20
    bar_w = 20
    canvas_w = margin_l + bins * bar_w + margin_r
    canvas_h = margin_t + 250 + margin_b
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40

    # Draw bars
    plot_h = 250
    for i, count in enumerate(hist_counts):
        bar_height = int((count / max_count) * plot_h)
        x1 = margin_l + i * bar_w + 1
        x2 = x1 + bar_w - 2
        y2 = margin_t + plot_h
        y1 = y2 - bar_height
        # Color gradient
        ratio = i / max(bins - 1, 1)
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, g, r), -1)

    # Axes
    cv2.line(canvas, (margin_l, margin_t), (margin_l, margin_t + plot_h), (200, 200, 200), 1)
    cv2.line(canvas, (margin_l, margin_t + plot_h), (margin_l + bins * bar_w, margin_t + plot_h), (200, 200, 200), 1)

    # X-axis labels (every 4th bin)
    for i in range(0, bins + 1, max(1, bins // 5)):
        x = margin_l + i * bar_w
        val = bin_edges[min(i, len(bin_edges) - 1)]
        cv2.putText(canvas, f"{val:.1f}", (x - 10, margin_t + plot_h + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # Y-axis labels
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_t + int(plot_h * (1 - frac))
        val = int(max_count * frac)
        cv2.putText(canvas, str(val), (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.line(canvas, (margin_l - 3, y), (margin_l, y), (150, 150, 150), 1)

    # Title and labels
    cv2.putText(canvas, "Reprojection Error Distribution", (margin_l + 20, 25),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Error (px)", (canvas_w // 2 - 30, canvas_h - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def draw_coverage_grid(
    points: np.ndarray,
    output_path: Path,
    image_size: int = 512,
    grid_size: int = 8,
    label: str = "Coverage Grid",
) -> None:
    """
    Draw a grid heatmap showing match density per cell.
    """
    cell = image_size // grid_size
    canvas_size = grid_size * cell
    canvas = np.ones((canvas_size + 50, canvas_size, 3), dtype=np.uint8) * 40

    cov = spatial_coverage(points, image_size, grid_size)
    counts = cov["points_per_cell"]
    max_count = max(max(counts), 1)

    for i, count in enumerate(counts):
        row = i // grid_size
        col = i % grid_size
        x1, y1 = col * cell, row * cell
        x2, y2 = x1 + cell, y1 + cell

        if count > 0:
            intensity = int(50 + 205 * (count / max_count))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, intensity, 0), -1)
            cv2.putText(canvas, str(count), (x1 + cell // 3, y1 + cell // 2 + 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 20, 20), -1)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 80), 1)

    # Info bar
    cv2.putText(canvas, f"{label} | Coverage: {cov['coverage_ratio']:.0%} ({cov['occupied_cells']}/{cov['total_cells']})",
                 (10, canvas_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


# ---------------------------------------------------------------------------
# Full analysis for one region
# ---------------------------------------------------------------------------

def analyze_region(region_id: str, matches: list[dict]) -> None:
    """Generate all visual analysis outputs for one region."""
    if not matches:
        print(f"  [{region_id}] No matches — skipping visualization")
        return

    src_pts = np.array([[m["image1_x"], m["image1_y"]] for m in matches])
    dst_pts = np.array([[m["image2_x"], m["image2_y"]] for m in matches])

    # Try to load images for overlay
    region_dir = PROCESSED_DIR / region_id
    src_img = None
    dst_img = None
    if (region_dir / "ohrc_512.png").is_file():
        src_img = cv2.imread(str(region_dir / "ohrc_512.png"))
    if (region_dir / "tmc_512.png").is_file():
        dst_img = cv2.imread(str(region_dir / "tmc_512.png"))

    out = OUTPUT_DIR / region_id
    out.mkdir(parents=True, exist_ok=True)

    # 1. Match distribution
    draw_match_distribution(
        src_pts, dst_pts, src_img, dst_img,
        out / "match_distribution.png",
        title=f"Region: {region_id}",
    )

    # 2. Reprojection error map
    if len(src_pts) >= 4:
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            errors = reprojection_errors(src_pts, dst_pts, H)
            draw_reprojection_error_map(src_pts, errors, src_img, out / "error_map.png")
            draw_error_histogram(errors, out / "error_histogram.png")
        else:
            print(f"  [{region_id}] Homography computation failed")
    else:
        print(f"  [{region_id}] Too few points for homography ({len(src_pts)})")

    # 3. Coverage grids
    draw_coverage_grid(src_pts, out / "coverage_source.png", label=f"{region_id} Source (OHRC)")
    draw_coverage_grid(dst_pts, out / "coverage_destination.png", label=f"{region_id} Destination (TMC)")

    print(f"  [{region_id}] Saved visualizations to {out}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    """Run visual error analysis on all discovered match files."""
    # Reuse evaluate's discovery logic
    try:
        from ml.evaluation.evaluate import discover_match_files, _load_matches
    except ImportError:
        from .evaluate import discover_match_files, _load_matches

    match_files = discover_match_files()
    if not match_files:
        print("[error_analysis] No match files found")
        return

    print(f"[error_analysis] Analyzing {len(match_files)} region(s)")
    for region_id, filepath in sorted(match_files.items()):
        matches = _load_matches(filepath)
        analyze_region(region_id, matches)

    print(f"\n[error_analysis] All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_analysis()
