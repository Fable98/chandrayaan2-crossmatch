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
        assert "shift_applied" in res["content_overlap_recovery"]
        assert "metrics" in res
        assert "content_overlap_recovery" in res["metrics"]


def test_overlap_recovery_estimates_scale_before_translation():
    """
    Step 11: scale must be estimated FIRST — inputs must NOT be resized to
    min(H, W) (which erases the scale gap). A 4x same-footprint pair must
    report scale_ratio ~= 4 and map the shift back to reference pixels.
    """
    h, w = 256, 256
    np.random.seed(7)
    base = np.random.uniform(50, 200, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (11, 11), 2.5)

    # 4x coarser sampling of the same footprint + a known fine-scale shift.
    gt_dx_fine, gt_dy_fine = 20.0, 12.0
    M = np.float32([[1, 0, gt_dx_fine], [0, 1, gt_dy_fine]])
    shifted = cv2.warpAffine(base, M, (w, h))
    coarse = cv2.resize(shifted, (w // 4, h // 4), interpolation=cv2.INTER_AREA)

    res = recover_content_overlap(base, coarse, gsd_m=5.0)

    assert res["frame"] == "reference_pixels"
    assert abs(res["scale_ratio"] - 4.0) <= 1.0, f"scale not estimated first: {res['scale_ratio']}"
    assert res["overlap_recovered"] is True
    # Shift mapped back to (coarse) reference pixels: 20/4, 12/4.
    assert abs(res["dx_px"] - gt_dx_fine / 4.0) <= 1.5, f"dx={res['dx_px']}"
    assert abs(res["dy_px"] - gt_dy_fine / 4.0) <= 1.5, f"dy={res['dy_px']}"


def test_overlap_recovery_20x_scale_gap_capped_and_honest():
    """
    Step 11: ~20x gaps (OHRC<->TMC-2 natives) exceed the documented 10x cap.
    The module must clamp (scale_capped), must NOT hallucinate a confident
    shift, and must leave bounds untouched.
    """
    h, w = 512, 512
    np.random.seed(11)
    base = np.random.uniform(50, 200, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (11, 11), 2.5)
    tiny = cv2.resize(base, (w // 20, h // 20), interpolation=cv2.INTER_AREA)
    assert tiny.shape == (25, 25) or tiny.shape == (26, 26)

    initial_bounds = {
        "west_lon": 336.48,
        "east_lon": 336.58,
        "south_lat": -3.37,
        "north_lat": -3.25,
    }
    res = recover_content_overlap(
        base, tiny, initial_bounds=initial_bounds, gsd_m=5.0,
    )

    # Documented cap behavior: clamped or scale-uncertain, never confident.
    assert res["scale_capped"] is True or res["scale_ratio"] >= 10.0 or res["confidence"] < 0.30
    assert res["overlap_recovered"] is False
    # Bounds must not diverge on a refused recovery.
    assert res["recovered_bounds"] == initial_bounds


def test_overlap_recovery_recenter_large_offset_integration():
    """
    Integration test proving that content-based overlap recovery actually improves
    matching quality by re-centering the search window:
    - When an offset exceeds the unshifted search window radius (> search_half_w),
      recover_overlap_from_content=False fails to find correspondences.
    - When recover_overlap_from_content=True, the recovered offset re-centers the
      coarse search window, enabling successful matching, high inliers, and
      accurate homography recovery.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        h, w = 384, 384
        np.random.seed(99)
        img1 = np.full((h, w), 120, dtype=np.uint8)
        for _ in range(40):
            cx = np.random.randint(60, w - 60)
            cy = np.random.randint(60, h - 60)
            r = np.random.randint(12, 24)
            val = int(np.random.randint(170, 255))
            cv2.circle(img1, (cx, cy), r, val, -1)
            cv2.circle(img1, (cx, cy), max(3, r - 5), int(val * 0.3), -1)
        img1 = cv2.GaussianBlur(img1, (7, 7), 1.5)

        # Apply a large shift (dx=+80, dy=+60) exceeding search_half_w = 384 // 6 = 64
        gt_dx, gt_dy = 80.0, 60.0
        M = np.float32([[1, 0, gt_dx], [0, 1, gt_dy]])
        img2 = cv2.warpAffine(img1, M, (w, h))

        p1 = tmp_path / "source_offset.png"
        p2 = tmp_path / "reference_offset.png"
        cv2.imwrite(str(p1), img1)
        cv2.imwrite(str(p2), img2)

        initial_bounds = {
            "west_lon": 336.48,
            "east_lon": 336.58,
            "south_lat": -3.37,
            "north_lat": -3.25,
        }

        # 1. Baseline WITHOUT overlap recovery: search window misses shifted features
        res_off = match_images_cfog(
            p1,
            p2,
            output_dir=tmp_path / "out_off",
            explicit_gsd1=2.5,
            explicit_gsd2=2.5,
            recover_overlap_from_content=False,
            grid_size=6,
        )
        inliers_off = res_off.get("metrics", {}).get("inlier_count", 0) if res_off.get("metrics") else 0
        H_off = np.array(res_off["homography"], dtype=np.float64) if res_off.get("homography") else None

        # Baseline either fails or finds a bogus transform diverging by > 30 px from ground truth
        if res_off["status"] == "success" and H_off is not None:
            baseline_tx_err = abs(H_off[0, 2] - gt_dx)
            baseline_ty_err = abs(H_off[1, 2] - gt_dy)
            assert baseline_tx_err > 30.0 or baseline_ty_err > 30.0, (
                f"Baseline unexpectedly recovered true shift without recentering: tx={H_off[0, 2]}, ty={H_off[1, 2]}"
            )

        # 2. WITH overlap recovery: search bounds re-center onto true shift
        res_on = match_images_cfog(
            p1,
            p2,
            output_dir=tmp_path / "out_on",
            explicit_gsd1=2.5,
            explicit_gsd2=2.5,
            recover_overlap_from_content=True,
            initial_bounds=initial_bounds,
            grid_size=6,
        )

        assert res_on["status"] == "success", f"Overlap-recovery matching failed: {res_on.get('message')}"
        rec_info = res_on.get("content_overlap_recovery", {})
        assert rec_info.get("overlap_recovered") is True
        assert "shift_applied" in rec_info
        shift_applied = rec_info["shift_applied"]
        assert abs(shift_applied["shift_work_x"] - gt_dx) < 2.0
        assert abs(shift_applied["shift_work_y"] - gt_dy) < 2.0

        inliers_on = res_on["metrics"]["inlier_count"]
        assert inliers_on >= 20, f"Expected >= 20 inliers with re-centered search bounds, got {inliers_on}"
        assert inliers_on >= 2 * inliers_off, (
            f"Expected inliers_on ({inliers_on}) to double baseline ({inliers_off})"
        )

        # Verify that estimated homography translation strictly matches ground truth within 1.0 px
        H_on = np.array(res_on["homography"], dtype=np.float64)
        assert abs(H_on[0, 2] - gt_dx) < 1.0, f"Homography tx={H_on[0, 2]:.2f} diverges from ground truth {gt_dx}"
        assert abs(H_on[1, 2] - gt_dy) < 1.0, f"Homography ty={H_on[1, 2]:.2f} diverges from ground truth {gt_dy}"
        assert abs(H_on[0, 0] - 1.0) < 0.05, f"Scale x diverged: {H_on[0, 0]}"
        assert abs(H_on[1, 1] - 1.0) < 0.05, f"Scale y diverged: {H_on[1, 1]}"

        # Verify initial_bounds were updated into recovered_bounds
        assert rec_info.get("initial_bounds") == initial_bounds
        rec_bounds = rec_info.get("recovered_bounds")
        assert rec_bounds is not None
        assert rec_bounds != initial_bounds
        assert rec_info.get("bounds_shift_meters") is not None

