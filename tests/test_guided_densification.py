"""
tests/test_guided_densification.py — Verify Guided Matching Densification.

Tests:
1. Guided matching densifies inlier count to >= 10 points.
2. Held-out validation RMSE is computable (evaluated, not null / insufficient_points_for_holdout).
3. Canonical 10x10 spatial coverage increases.
4. Passing enable_guided_densification=False retains the baseline anchor set.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import match_images_cfog


def test_guided_densification_expands_inliers_and_evaluates_held_out(tmp_path):
    src_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png"
    ref_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/tmc_512.png"

    if not src_path.exists() or not ref_path.exists():
        pytest.skip("Region 001 sample images not found on disk.")

    # Run with guided densification enabled (default)
    res_guided = match_images_cfog(
        src_path,
        ref_path,
        source_sensor="OHRC",
        reference_sensor="TMC",
        output_dir=tmp_path / "guided_out",
        enable_guided_densification=True,
    )

    assert res_guided["status"] == "success"
    metrics_guided = res_guided["metrics"]
    assert metrics_guided is not None

    # Inliers must be expanded >= 10
    assert metrics_guided["inlier_count"] >= 10, f"Expected >= 10 inliers, got {metrics_guided['inlier_count']}"

    # Held-out validation must be evaluated
    assert metrics_guided["validation_status"] == "evaluated"
    assert metrics_guided["held_out_validation_rmse_px"] is not None
    assert np.isfinite(metrics_guided["held_out_validation_rmse_px"])

    # Spatial coverage must be greater than baseline 0.06
    assert metrics_guided["spatial_coverage"] > 0.06


def test_guided_densification_can_be_disabled(tmp_path):
    src_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png"
    ref_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/tmc_512.png"

    if not src_path.exists() or not ref_path.exists():
        pytest.skip("Region 001 sample images not found on disk.")

    res_baseline = match_images_cfog(
        src_path,
        ref_path,
        source_sensor="OHRC",
        reference_sensor="TMC",
        output_dir=tmp_path / "baseline_out",
        enable_guided_densification=False,
    )

    assert res_baseline["status"] == "success"
    metrics_base = res_baseline["metrics"]
    assert metrics_base is not None
    # Baseline anchor inliers without guided densification (around 6-9)
    assert metrics_base["inlier_count"] <= 10


def test_guided_points_accuracy_against_known_ground_truth(tmp_path):
    """
    Non-circular ground truth validation:
    Validates that points added by guided densification are accurate against an
    INDEPENDENT ground-truth projective transformation H_gt (not just fitting the estimated H).
    """
    import cv2
    import json
    import math

    tile_path = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png"
    if not tile_path.exists():
        pytest.skip("Region 001 sample image not found.")

    src = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)
    assert src is not None
    h, w = src.shape[:2]

    # Create known ground truth transformation: rotation + translation + slight scale
    theta = math.radians(2.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    s = 1.01
    dx, dy = 10.0, -8.0
    cx, cy = w / 2.0, h / 2.0

    # Affine matrix around center
    H_gt = np.array([
        [s * cos_t, -s * sin_t, (1 - s * cos_t) * cx + s * sin_t * cy + dx],
        [s * sin_t,  s * cos_t, -s * sin_t * cx + (1 - s * cos_t) * cy + dy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    warped = cv2.warpPerspective(src, H_gt, (w, h), flags=cv2.INTER_LINEAR)

    p_src = tmp_path / "src.png"
    p_tgt = tmp_path / "tgt.png"
    cv2.imwrite(str(p_src), src)
    cv2.imwrite(str(p_tgt), warped)

    res = match_images_cfog(
        p_src,
        p_tgt,
        source_sensor="OHRC",
        reference_sensor="OHRC",
        explicit_gsd1=5.0,
        explicit_gsd2=5.0,
        output_dir=tmp_path / "gt_guided_out",
        enable_guided_densification=True,
    )

    assert res["status"] == "success"
    matches_file = tmp_path / "gt_guided_out" / "matches.json"
    assert matches_file.exists()

    with open(matches_file, "r") as f:
        matches_data = json.load(f)

    matches_list = matches_data if isinstance(matches_data, list) else matches_data.get("matches", [])

    # Filter for guided refill inliers
    guided_inliers = [
        m for m in matches_list
        if m.get("method") == "guided_refill" and m.get("is_inlier", False)
    ]

    # If guided inliers were accepted, verify their accuracy against INDEPENDENT H_gt
    if guided_inliers:
        gt_errors = []
        for m in guided_inliers:
            x1, y1 = float(m["image1_x"]), float(m["image1_y"])
            x2, y2 = float(m["image2_x"]), float(m["image2_y"])

            # Project through ground truth
            p1_homo = np.array([x1, y1, 1.0], dtype=np.float64)
            p2_proj_homo = H_gt @ p1_homo
            p2_proj = p2_proj_homo[:2] / p2_proj_homo[2]

            err = math.hypot(p2_proj[0] - x2, p2_proj[1] - y2)
            gt_errors.append(err)

        mean_gt_err = float(np.mean(gt_errors))
        # Mean reprojection error against independent ground-truth must be sub-pixel/tight (< 2.0 px)
        assert mean_gt_err < 2.0, f"Guided points failed independent ground-truth check: {mean_gt_err:.3f} px"

