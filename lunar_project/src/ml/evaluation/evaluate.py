"""
evaluate.py — Batch evaluation entry point for all processed regions.

Loads match JSON files from the ML pipeline output, computes all evaluation
metrics for each region, and outputs a summary table + JSON report.

Usage:
    python -m ml.evaluation.evaluate
    -- or --
    python evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure 'src' is on sys.path for standalone script execution
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Support running both as a package and standalone
try:
    from ml.evaluation.metrics import compute_all_metrics
except ImportError:
    from .metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root (contains data_preprocessing_pipeline and ML_model)."""
    p = Path(__file__).resolve()
    for ancestor in [p] + list(p.parents):
        if (ancestor / "data_preprocessing_pipeline").is_dir() and (ancestor / "ML_model").is_dir():
            return ancestor
    return p.parent.parent.parent.parent.parent


REPO_ROOT = _find_repo_root()

MATCH_SOURCES = [
    REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets",
    REPO_ROOT / "processed_user" / "matches",
    REPO_ROOT / "ML_model",
]

OUTPUT_DIR = REPO_ROOT / "evaluation_output"


# ---------------------------------------------------------------------------
# Match loading
# ---------------------------------------------------------------------------

def _load_matches(filepath: Path) -> list[dict]:
    """Load a match file (bare list or wrapped format)."""
    with open(filepath, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "matches" in data:
        return data["matches"]
    return []


def discover_match_files() -> dict[str, Path]:
    """Find all match files across known directories."""
    found: dict[str, Path] = {}

    # processed_user/matches/{region_id}_matches.json
    matches_dir = REPO_ROOT / "processed_user" / "matches"
    if matches_dir.is_dir():
        for f in sorted(matches_dir.glob("*_matches.json")):
            region_id = f.stem.replace("_matches", "")
            found[region_id] = f

    # ML_model/matches.json → region_001
    ml_matches = REPO_ROOT / "ML_model" / "matches.json"
    if ml_matches.is_file() and "region_001" not in found:
        found["region_001"] = ml_matches

    return found


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_region(region_id: str, matches: list[dict]) -> dict:
    """
    Evaluate a single region's matches and return all metrics.

    Args:
        region_id: Region identifier (e.g. "region_001").
        matches: List of match dicts with image1_x/y, image2_x/y, confidence.

    Returns:
        Dict containing region_id and all computed metrics.
    """
    if not matches:
        return {
            "region_id": region_id,
            "status": "no_matches",
            "num_inliers": 0,
        }

    src_pts = np.array([[m["image1_x"], m["image1_y"]] for m in matches])
    dst_pts = np.array([[m["image2_x"], m["image2_y"]] for m in matches])

    metrics = compute_all_metrics(
        src_pts=src_pts,
        dst_pts=dst_pts,
        num_raw_matches=None,  # These are already post-RANSAC
        image_size=512,
        grid_size=8,
    )

    # Add confidence stats
    confidences = [m.get("confidence", 0.0) for m in matches]
    metrics["mean_confidence"] = round(float(np.mean(confidences)), 4)
    metrics["min_confidence"] = round(float(np.min(confidences)), 4)
    metrics["max_confidence"] = round(float(np.max(confidences)), 4)

    # Remove per-point errors from summary (too verbose)
    per_point = metrics.pop("per_point_errors", [])

    return {
        "region_id": region_id,
        "status": "evaluated",
        **metrics,
        "_per_point_errors": per_point,  # kept for error_analysis
    }


def run_evaluation() -> list[dict]:
    """
    Run evaluation on all discovered match files.

    Returns list of per-region metric dicts.
    """
    match_files = discover_match_files()

    if not match_files:
        print("[evaluate] No match files found. Check paths:")
        for s in MATCH_SOURCES:
            print(f"  - {s}")
        return []

    print(f"[evaluate] Found {len(match_files)} region(s) with matches")
    print("=" * 72)

    results = []
    for region_id, filepath in sorted(match_files.items()):
        matches = _load_matches(filepath)
        result = evaluate_region(region_id, matches)
        results.append(result)

        # Console output
        if result["status"] == "no_matches":
            print(f"  {region_id}: NO MATCHES")
        else:
            sub_px = "YES" if result.get("sub_pixel_accurate") else "NO"
            print(
                f"  {region_id}: "
                f"inliers={result['num_inliers']:3d}  "
                f"RMSE={result['rmse_px']:.3f}px  "
                f"mean_err={result['mean_reprojection_error_px']:.3f}px  "
                f"sub-pixel={sub_px}  "
                f"coverage={result['combined_coverage_score']:.2f}  "
                f"conf={result['mean_confidence']:.3f}"
            )

    print("=" * 72)

    # Save summary JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "evaluation_summary.json"

    # Strip internal per-point data for the summary file
    summary_data = []
    for r in results:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        summary_data.append(clean)

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[evaluate] Summary saved to {summary_path}")

    # Also save detailed per-region reports
    for r in results:
        if r["status"] == "evaluated":
            detail_path = OUTPUT_DIR / f"{r['region_id']}_metrics.json"
            with open(detail_path, "w") as f:
                detail = {k: v for k, v in r.items() if not k.startswith("_")}
                json.dump(detail, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_evaluation()
    if not results:
        sys.exit(1)

    # Print summary table
    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY TABLE")
    print("=" * 72)
    print(f"{'Region':<14} {'Inliers':>8} {'RMSE':>8} {'MeanErr':>8} {'SubPx':>6} {'Cover':>7} {'Conf':>6}")
    print("-" * 72)
    for r in results:
        if r["status"] == "no_matches":
            print(f"{r['region_id']:<14} {'N/A':>8}")
            continue
        print(
            f"{r['region_id']:<14} "
            f"{r['num_inliers']:>8d} "
            f"{r['rmse_px']:>8.3f} "
            f"{r['mean_reprojection_error_px']:>8.3f} "
            f"{'  YES' if r['sub_pixel_accurate'] else '   NO':>6} "
            f"{r['combined_coverage_score']:>7.3f} "
            f"{r['mean_confidence']:>6.3f}"
        )
    print("=" * 72)
