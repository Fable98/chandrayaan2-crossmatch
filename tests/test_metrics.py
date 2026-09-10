"""
tests/test_metrics.py — Unit tests for Absolute RMSE (meters) and registration metrics
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from metrics import calculate_absolute_rmse_meters, compute_canonical_metrics


def test_absolute_rmse_flat_dem_scalar():
    """
    Validates that with a flat DEM (zero elevation delta) and a known GSD (2.5 m/px),
    the absolute RMSE in meters equals exactly (pixel RMSE * GSD).
    """
    # 4 points with identical 2.0 px Euclidean error: (dx=1.2, dy=1.6) -> sqrt(1.2^2 + 1.6^2) = 2.0
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    pts2 = np.array([[11.2, 11.6], [21.2, 21.6], [31.2, 31.6], [41.2, 41.6]])
    gsd = 2.5  # meters per pixel

    # Flat DEM (constant 500m elevation)
    flat_dem = np.full((100, 100), 500.0, dtype=np.float32)

    rmse_meters = calculate_absolute_rmse_meters((pts1, pts2), gsd=gsd, dem_data=flat_dem)
    expected_rmse = 2.0 * 2.5  # 5.0 meters

    assert abs(rmse_meters - expected_rmse) < 1e-3, f"Expected {expected_rmse}, got {rmse_meters}"


def test_absolute_rmse_match_dicts_and_flat_dem():
    """
    Validates calculate_absolute_rmse_meters with a list of correspondence dicts.
    """
    # Point 1: dx=3, dy=4 -> dr=5 px -> 5 * 2.5 = 12.5 m
    # Point 2: dx=0, dy=0 -> dr=0 px -> 0 m
    # RMS = sqrt((12.5^2 + 0^2)/2) = sqrt(156.25 / 2) = sqrt(78.125) ≈ 8.8388 m
    match_records = [
        {"source_x": 10.0, "source_y": 20.0, "target_x": 13.0, "target_y": 24.0},
        {"source_x": 50.0, "source_y": 50.0, "target_x": 50.0, "target_y": 50.0},
    ]
    gsd = 2.5
    flat_dem = np.zeros((100, 100), dtype=np.float32)

    rmse_meters = calculate_absolute_rmse_meters(match_records, gsd=gsd, dem_data=flat_dem)
    expected_rmse = float(np.sqrt((12.5**2) / 2.0))

    assert abs(rmse_meters - round(expected_rmse, 4)) < 1e-3, f"Expected {expected_rmse}, got {rmse_meters}"


def test_absolute_rmse_topographic_relief():
    """
    Validates that a non-flat DEM properly incorporates 3D elevation delta (dz).
    """
    # Point from (10, 10) to (13, 14): dx=3 px, dy=4 px -> planar = 5 px * 2.5 = 12.5 m
    # Elevation: dem[10, 10] = 100.0 m, dem[14, 13] = 105.0 m -> dz = 5.0 m
    # 3D distance = sqrt(12.5^2 + 5.0^2) = sqrt(156.25 + 25) = sqrt(181.25) ≈ 13.4629 m
    pts1 = np.array([[10.0, 10.0]])
    pts2 = np.array([[13.0, 14.0]])
    gsd = 2.5

    dem = np.full((100, 100), 100.0, dtype=np.float32)
    dem[14, 13] = 105.0  # target location has +5m elevation

    rmse_meters = calculate_absolute_rmse_meters((pts1, pts2), gsd=gsd, dem_data=dem)
    expected_3d = float(np.sqrt(12.5**2 + 5.0**2))

    assert abs(rmse_meters - round(expected_3d, 4)) < 1e-3, f"Expected {expected_3d}, got {rmse_meters}"


def test_canonical_metrics_includes_absolute_rmse():
    """
    Validates that compute_canonical_metrics returns 'absolute_rmse_m' when gsd_m is provided.
    """
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    pts2 = np.array([[12.0, 10.0], [22.0, 20.0], [32.0, 30.0], [42.0, 40.0]])  # pure dx=2.0 px
    H = np.eye(3, dtype=np.float64)  # Identity homography -> error is 2.0 px
    mask = np.ones(4, dtype=np.uint8)

    metrics = compute_canonical_metrics(
        pts1, pts2, mask, H, image_shape=(100, 100), gsd_m=2.5
    )

    assert "absolute_rmse_m" in metrics
    assert metrics["absolute_rmse_m"] is not None
    assert abs(metrics["absolute_rmse_m"] - 5.0) < 1e-3


def test_honest_metrics_insample_and_subpixel_not_alone():
    """
    Validates that:
    1. 'fit_rmse_insample_px' is present alongside 'fit_rmse_px'.
    2. 'sub_pixel_accurate' is NEVER True on in-sample fit alone when held-out is unavailable.
    3. 'held_out_validation_rmse_px' and 'absolute_rmse_m' are reported side-by-side.
    """
    # 4 points with small in-sample error (0.2 px) -> held-out validation requires >= 8 points
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    pts2 = np.array([[10.2, 10.0], [20.2, 20.0], [30.2, 30.0], [40.2, 40.0]])
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 0.2  # Exact shift
    mask = np.ones(4, dtype=np.uint8)

    metrics = compute_canonical_metrics(
        pts1, pts2, mask, H, image_shape=(100, 100), gsd_m=0.5
    )

    # 1. in-sample naming
    assert "fit_rmse_insample_px" in metrics
    assert metrics["fit_rmse_insample_px"] == pytest.approx(metrics["fit_rmse_px"], abs=1e-4)
    assert metrics["fit_rmse_insample_px"] < 1.0

    # 2. held-out is uncomputable for N=4 -> sub_pixel_accurate must NOT be set on in-sample alone
    assert metrics["held_out_validation_rmse_px"] is None
    assert metrics["sub_pixel_accurate"] is False, "sub_pixel_accurate must not be True without held-out validation"

    # 3. side-by-side exposure
    assert "held_out_validation_rmse_px" in metrics
    assert "absolute_rmse_m" in metrics
    assert metrics["absolute_rmse_m"] is not None


def test_subpixel_accurate_with_heldout():
    """
    Validates that when both in-sample AND held-out validation are < 1.0 px,
    sub_pixel_accurate is True.
    """
    # 20 points with exact small shift
    pts1 = np.array([[float(i * 10 + 10), float(j * 10 + 10)] for i in range(4) for j in range(5)])
    pts2 = pts1 + np.array([0.2, 0.1])
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 0.2
    H[1, 2] = 0.1
    mask = np.ones(len(pts1), dtype=np.uint8)

    metrics = compute_canonical_metrics(
        pts1, pts2, mask, H, image_shape=(200, 200), gsd_m=0.5
    )

    assert metrics["fit_rmse_insample_px"] < 1.0
    assert metrics["held_out_validation_rmse_px"] is not None
    assert metrics["held_out_validation_rmse_px"] < 1.0
    assert metrics["sub_pixel_accurate"] is True


def test_quality_tier_gates():
    """
    Validates tier gates:
    - N < 15 or coverage < 15% -> LOW_CONFIDENCE (tier 'LOW'), never 'HIGH'.
    - N >= 15 + coverage >= 15% + heldout < 2px -> HIGH_CONFIDENCE (tier 'HIGH').
    """
    # Case A: 6 points (like region_001/003/006) -> LOW
    pts1_low = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0], [60.0, 60.0]])
    pts2_low = pts1_low + 0.1
    H_low = np.eye(3, dtype=np.float64)
    mask_low = np.ones(len(pts1_low), dtype=np.uint8)

    metrics_low = compute_canonical_metrics(pts1_low, pts2_low, mask_low, H_low, image_shape=(500, 500))
    assert metrics_low["quality_tier"] == "LOW_CONFIDENCE"
    assert metrics_low["tier"] == "LOW"

    # Case B: 20 points well-distributed across grid -> HIGH
    xs = np.linspace(20, 480, 5)
    ys = np.linspace(20, 480, 4)
    grid_pts = np.array([[x, y] for x in xs for y in ys])
    H_high = np.eye(3, dtype=np.float64)
    metrics_high = compute_canonical_metrics(grid_pts, grid_pts + 0.1, np.ones(len(grid_pts), dtype=np.uint8), H_high, image_shape=(500, 500), grid_size=10)
    assert metrics_high["quality_tier"] == "HIGH_CONFIDENCE"
    assert metrics_high["tier"] == "HIGH"


def test_evaluator_pairwise_records_fit_insample_and_heldout_null():
    """
    Validates that isro_official_evaluator pairwise outputs show fit_insample + heldout=null explicitly,
    and summary reports success_rate and bootstrap CI.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from isro_official_evaluator import _bootstrap_ci

    # 1. Bootstrap CI helper
    ci = _bootstrap_ci([0.25, 0.30, 0.28, 0.32, 0.29, 0.31])
    assert ci is not None
    assert "mean" in ci and "ci_lower" in ci and "ci_upper" in ci
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    # 2. None for small sample
    assert _bootstrap_ci([0.25]) is None

    # 3. Evaluate on simulated pairs to verify pairwise records show fit_insample + heldout=null
    from isro_official_evaluator import evaluate_all
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        s1 = d / "source_01.png"
        r1 = d / "reference_01.png"
        s1.write_bytes(b"\x00" * 50)
        r1.write_bytes(b"\x00" * 50)

        pairs = [{"id": "pair_fail", "source": s1, "reference": r1, "dem": None}]
        summary, pairwise = evaluate_all(pairs, d / "out", use_dem=False)
        assert summary["total_pairs_processed"] == 1
        assert summary["successful_registrations"] == 0
        assert summary["failed_registrations"] == 1
        assert summary["success_rate"] == 0.0
        assert len(pairwise) == 1
        rec = pairwise[0]
        assert "fit_insample" in rec
        assert rec["fit_insample"] is None
        assert "heldout" in rec
        assert rec["heldout"] is None

