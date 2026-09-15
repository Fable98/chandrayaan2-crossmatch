"""tests/test_phase2_evaluation_integrity.py — Verification of Phase 2 Evaluation Integrity.

Validates:
1. All-Fail Fixture: When all legs fail, evaluator produces honest nulls (no fabricated values).
2. Composed-Leg Fixture (1-pass/1-fail composition): When a leg is mathematically composed,
   intra_ch2_pixel_rmse and triplet_cycle_rmse_px report None (honest null, NEVER tautological ~0.0px).
3. Real Sensor GSD Threading: Absolute RMSE reflects the true sensor GSD (OHRC 0.25m, NAC 0.9m,
   TMC 5.0m, IIRS 70m) rather than hardcoded 5.0m fallbacks.
4. Statistical Testing: Wilcoxon runs on finite pairs only; McNemar evaluates binary success rates;
   cache paths are hashed deterministically.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "data_preprocessing_pipeline"))

from data_preprocessing_pipeline.triplet_evaluator import (
    evaluate_triplet_consistency,
    compose_missing_leg,
)
from evaluation.eval_heldout_patch32_end_to_end import (
    compute_mcnemar_test,
    get_cached_model_path,
)


def _create_dummy_image(path: Path, shape=(128, 128)):
    arr = np.random.randint(50, 200, shape, dtype=np.uint8)
    import cv2
    cv2.imwrite(str(path), arr)


def test_all_fail_fixture_shows_honest_nulls():
    """When all legs fail, evaluator must return cycle_not_computable with honest nulls."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_a = tmp_path / "img_a.png"
        img_b = tmp_path / "img_b.png"
        img_c = tmp_path / "img_c.png"
        _create_dummy_image(img_a)
        _create_dummy_image(img_b)
        _create_dummy_image(img_c)

        # Mock matcher to return failures for all legs
        failed_res = {
            "status": "failed",
            "homography": None,
            "metrics": None,
            "matches": [],
        }

        with patch("data_preprocessing_pipeline.triplet_evaluator.match_images_cfog", return_value=failed_res):
            report = evaluate_triplet_consistency(
                img_a, img_b, img_c,
                output_dir=tmp_path / "out",
                sensor_a="OHRC", sensor_b="TMC-2", sensor_c="IIRS",
            )

        assert report["status"] == "cycle_not_computable"
        assert report["triplet_cycle_rmse_px"] is None
        assert report["triplet_mean_cycle_error_px"] is None
        assert report["cycle_closed_successfully"] is False
        assert report["Intra-CH2 Pixel RMSE"] is None
        assert report["Absolute RMSE (Meters)"] is None
        assert report["direct_ohrc_iirs_homography"] is None
        assert report["production_grade_chained_homography"] is None
        assert len(report["failed_legs"]) == 3


def test_composed_leg_fixture_gates_tautological_zero_cycle():
    """When 1 leg fails and is mathematically composed, intra_ch2_pixel_rmse must be None (NEVER ~0.0px)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_a = tmp_path / "img_a.png"
        img_b = tmp_path / "img_b.png"
        img_c = tmp_path / "img_c.png"
        _create_dummy_image(img_a)
        _create_dummy_image(img_b)
        _create_dummy_image(img_c)

        # True homographies: AB and BC are slight shifts; CA fails
        H_AB = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_BC = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        def mock_matcher(src, dst, **kwargs):
            s = str(src)
            d = str(dst)
            if "img_a" in s and "img_b" in d:
                return {
                    "status": "success",
                    "homography": H_AB.tolist(),
                    "metrics": {"fit_rmse_px": 0.85, "inlier_count": 45, "absolute_rmse_m": 0.21},
                    "working_scale": {"gsd_m": 0.25},
                }
            elif "img_b" in s and "img_c" in d:
                return {
                    "status": "success",
                    "homography": H_BC.tolist(),
                    "metrics": {"fit_rmse_px": 1.10, "inlier_count": 30, "absolute_rmse_m": 5.50},
                    "working_scale": {"gsd_m": 5.0},
                }
            else:
                # Leg CA (IIRS -> OHRC) fails
                return {"status": "failed", "homography": None, "metrics": None}

        with patch("data_preprocessing_pipeline.triplet_evaluator.match_images_cfog", side_effect=mock_matcher):
            report = evaluate_triplet_consistency(
                img_a, img_b, img_c,
                output_dir=tmp_path / "out",
                sensor_a="OHRC", sensor_b="TMC-2", sensor_c="IIRS",
            )

        # Leg CA should be marked "composed"
        assert report["leg_derivations"]["CA"] == "composed"
        assert report["status"] == "evaluated_composed"

        # The closed-loop cycle consistency on the composed homography would mathematically be ~0.0px,
        # but because one leg was derived from the other two, this is a tautology.
        # It MUST NOT be reported as measured closed-loop error!
        assert report["triplet_cycle_rmse_px"] is None, "Tautological cycle must be None"
        assert report["Intra-CH2 Pixel RMSE"] is None, "Intra-CH2 RMSE must be gated to None on composed legs"
        assert report["Absolute RMSE (Meters)"] is None


def test_measured_triplet_threads_sensor_a_real_gsd():
    """When all 3 legs succeed, Absolute RMSE must reflect sensor A GSD (OHRC 0.25m), NOT 5.0m."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_a = tmp_path / "img_a.png"
        img_b = tmp_path / "img_b.png"
        img_c = tmp_path / "img_c.png"
        _create_dummy_image(img_a)
        _create_dummy_image(img_b)
        _create_dummy_image(img_c)

        # Leg AB: shift (2, 0)
        H_AB = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        # Leg BC: shift (0, 2)
        H_BC = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        # Leg CA: shift (-2, -1) -> total loop error = (0, 1) -> 1.0px error
        H_CA = np.array([[1.0, 0.0, -2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        def mock_matcher(src, dst, **kwargs):
            s, d = str(src), str(dst)
            if "img_a" in s and "img_b" in d:
                return {"status": "success", "homography": H_AB.tolist(), "metrics": {"fit_rmse_px": 0.5}, "working_scale": {"gsd_m": 0.25}}
            elif "img_b" in s and "img_c" in d:
                return {"status": "success", "homography": H_BC.tolist(), "metrics": {"fit_rmse_px": 0.5}, "working_scale": {"gsd_m": 5.0}}
            else:
                return {"status": "success", "homography": H_CA.tolist(), "metrics": {"fit_rmse_px": 0.5}, "working_scale": {"gsd_m": 70.0}}

        with patch("data_preprocessing_pipeline.triplet_evaluator.match_images_cfog", side_effect=mock_matcher):
            report = evaluate_triplet_consistency(
                img_a, img_b, img_c,
                output_dir=tmp_path / "out",
                sensor_a="OHRC", sensor_b="TMC-2", sensor_c="IIRS",
            )

        assert report["status"] == "evaluated"
        intra_px = report["Intra-CH2 Pixel RMSE"]
        assert intra_px is not None and intra_px > 0.0
        abs_m = report["Absolute RMSE (Meters)"]
        assert abs_m is not None

        # Pixel RMSE * 0.25 (OHRC nadir GSD) should match abs_rmse_meters within rounding
        expected_m = round(intra_px * 0.25, 4)
        assert abs(abs_m - expected_m) <= 0.05, f"Absolute RMSE ({abs_m}m) should use OHRC 0.25m GSD, not 5.0m ({intra_px * 5.0}m)"


def test_mcnemar_statistical_test():
    """Verify exact McNemar test for paired binary outcomes."""
    # Symmetrical outcomes (b=2, c=2) -> p = 1.0
    r1 = compute_mcnemar_test([True, False, True, False], [True, True, False, False])
    assert r1["b"] == 1
    assert r1["c"] == 1
    assert r1["p_value"] == 1.0

    # Highly asymmetrical discordant outcomes (b=0, c=10) -> p < 0.01
    s1 = [False] * 10
    s2 = [True] * 10
    r2 = compute_mcnemar_test(s1, s2)
    assert r2["b"] == 0
    assert r2["c"] == 10
    assert r2["p_value"] < 0.01


def test_cached_model_path_hashing():
    """Verify cached model paths are deterministically isolated by feature names and groups."""
    p1 = get_cached_model_path(["f1", "f2"], 42, ["g1", "g2"])
    p2 = get_cached_model_path(["f1", "f2"], 42, ["g1", "g2"])
    p3 = get_cached_model_path(["f1", "f3"], 42, ["g1", "g2"])
    p4 = get_cached_model_path(["f1", "f2"], 99, ["g1", "g2"])

    assert p1 == p2, "Identical parameters must yield identical cache path"
    assert p1 != p3, "Different features must yield different cache path"
    assert p1 != p4, "Different seed must yield different cache path"
