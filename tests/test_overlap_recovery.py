"""
tests/test_overlap_recovery.py — Regression tests for content-based overlap recovery pre-matching.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from overlap_recovery import recover_content_overlap
from matcher_cfog import match_images_cfog


def test_overlap_recovery_synthetic_known_shift():
    """
    Validates that recover_content_overlap detects known 2D translation
    and updates geographic bounds without wild divergence.
    """
    h, w = 256, 256
    np.random.seed(42)
    base = np.random.uniform(50, 200, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (11, 11), 2.5)

    # Shift by dx=+8, dy=+5
    M = np.float32([[1, 0, 8], [0, 1, 5]])
    shifted = cv2.warpAffine(base, M, (w, h))

    initial_bounds = {
        "west_lon": 336.48,
        "east_lon": 336.58,
        "south_lat": -3.37,
        "north_lat": -3.25,
    }
    gsd_m = 5.0

    res = recover_content_overlap(
        base,
        shifted,
        initial_bounds=initial_bounds,
        gsd_m=gsd_m,
    )

    assert res["overlap_recovered"] is True
    # The recovered shift should align with the applied offset
    assert abs(res["dx_px"] - 8.0) <= 2.0
    assert abs(res["dy_px"] - 5.0) <= 2.0

    rec_b = res["recovered_bounds"]
    assert rec_b is not None
    # Sanity-check: recovered bounds must not diverge wildly from initial bounds (< 0.05 degrees)
    for k in ("west_lon", "east_lon", "south_lat", "north_lat"):
        assert abs(rec_b[k] - initial_bounds[k]) < 0.05, f"Bounds diverged wildly on {k}: {rec_b[k]} vs {initial_bounds[k]}"


def test_overlap_recovery_regression_on_sample_region():
    """
    Regression test comparing recovered bounds against label-derived bounds
    on actual sample Chandrayaan-2 imagery (sample_data/ohrc_sample.png and tmc_sample.png).
    Ensures that content-based overlap recovery does not diverge wildly on real lunar terrain.
    """
    ohrc_path = REPO_ROOT / "sample_data" / "ohrc_sample.png"
    tmc_path = REPO_ROOT / "sample_data" / "tmc_sample.png"

    if not ohrc_path.exists() or not tmc_path.exists():
        pytest.skip("Sample imagery not found in sample_data/")

    ohrc_img = cv2.imread(str(ohrc_path), cv2.IMREAD_GRAYSCALE)
    tmc_img = cv2.imread(str(tmc_path), cv2.IMREAD_GRAYSCALE)

    initial_bounds = {
        "west_lon": 336.484646,
        "east_lon": 336.589455,
        "south_lat": -3.374861,
        "north_lat": -3.248733,
    }

    res = recover_content_overlap(
        ohrc_img,
        tmc_img,
        initial_bounds=initial_bounds,
        gsd_m=5.0,
    )

    assert res is not None
    assert "dx_px" in res and "dy_px" in res
    assert "recovered_bounds" in res
    rec_b = res["recovered_bounds"]

    # Sanity check: recovered bounds must remain in strict physical proximity (< 0.05 deg)
    for k in ("west_lon", "east_lon", "south_lat", "north_lat"):
        delta = abs(rec_b[k] - initial_bounds[k])
        assert delta < 0.05, f"Wild divergence detected on {k}: delta={delta:.6f} deg"


def test_matcher_cfog_recover_overlap_flag():
    """
    Validates that recover_overlap_from_content=True triggers the pre-matching step
    and logs/records it in metrics and metadata without altering registration stability.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        h, w = 256, 256
        img1 = np.zeros((h, w), dtype=np.uint8)
        np.random.seed(42)
        for _ in range(30):
            cx = np.random.randint(30, w - 30)
            cy = np.random.randint(30, h - 30)
            rad = np.random.randint(10, 25)
            val = int(np.random.randint(120, 240))
            cv2.circle(img1, (cx, cy), rad, val, -1)
            cv2.circle(img1, (cx, cy), max(2, rad - 5), int(val * 0.4), -1)
        img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
        M = np.float32([[1, 0, 5], [0, 1, -3]])
        img2 = cv2.warpAffine(img1, M, (w, h))

        p1 = tmp_path / "source.png"
        p2 = tmp_path / "reference.png"
        cv2.imwrite(str(p1), img1)
        cv2.imwrite(str(p2), img2)

        out_dir = tmp_path / "reg_overlap"
        res = match_images_cfog(
            p1,
            p2,
            output_dir=out_dir,
            explicit_gsd1=1.0,
            explicit_gsd2=1.0,
            recover_overlap_from_content=True,
            grid_size=6,
        )

        assert res["status"] == "success"
        assert "content_overlap_recovery" in res
        assert res["content_overlap_recovery"] is not None
        assert "dx_px" in res["content_overlap_recovery"]
        assert "metrics" in res
        assert "content_overlap_recovery" in res["metrics"]
