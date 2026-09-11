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
    assert metrics_guided["spatial_coverage"] >= 0.10


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
    # Baseline anchor inliers should be around 6
    assert metrics_base["inlier_count"] <= 8
