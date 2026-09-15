"""
tests/test_relief_azimuth_convention.py — Phase 1.2 synthetic-tilt verification.

Locks the unified relief-shift azimuth convention across all implementations:
compass-style degrees clockwise from north (0=N, 90=E); in y-down pixel
coordinates the unit shift direction is (sin(psi), -cos(psi)).

Covers:
1. Cardinal mapping of geometry.dem_ray_intersection on a tilted-plane DEM.
2. Cross-function directional agreement: dem_ray_intersection vs
   matcher_cfog.compute_dem_ray_shift_correction.
3. Fail-fast on missing azimuth (ValueError, not a hallucinated direction).
4. ransac_dem_aware_fit survives azimuth=None (previously TypeError crash)
   and reports its frame contract honestly.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from geometry import dem_ray_intersection, ransac_dem_aware_fit
from matcher_cfog import compute_dem_ray_shift_correction


def _tilted_plane_dem(h=256, w=256, gx=0.8, gy=0.3, base=1000.0):
    yy, xx = np.indices((h, w), dtype=np.float64)
    return (base + gx * xx + gy * yy).astype(np.float32)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def test_cardinal_azimuth_mapping():
    """E/N/S/W azimuths shift relief along +x/-y/+y/-x respectively."""
    dem = _tilted_plane_dem()
    # Single point far above datum so Delta-z sign is unambiguous.
    dem[0:64, 0:64] += 500.0
    pts = np.array([[32.0, 32.0]])
    expected = {
        0.0: np.array([0.0, -1.0]),    # north -> up in image
        90.0: np.array([1.0, 0.0]),    # east  -> right
        180.0: np.array([0.0, 1.0]),   # south -> down
        270.0: np.array([-1.0, 0.0]),  # west  -> left
    }
    for az, unit in expected.items():
        _, disp = dem_ray_intersection(
            pts, dem, emission_deg=20.0, azimuth_deg=az, gsd_m=5.0
        )
        assert np.linalg.norm(disp[0]) > 1.0, f"az={az}: shift suspiciously small"
        assert np.dot(_unit(disp[0]), unit) > 0.999, (
            f"az={az}: direction {_unit(disp[0])} != expected {unit}"
        )


def test_cross_function_directional_agreement():
    """dem_ray_intersection and compute_dem_ray_shift_correction agree on direction."""
    dem = _tilted_plane_dem()
    rng = np.random.default_rng(7)
    pts = rng.uniform(20, 236, (60, 2))
    for az in (0.0, 30.0, 90.0, 135.0, 200.0, 315.0):
        _, d1 = dem_ray_intersection(
            pts, dem, emission_deg=20.0, azimuth_deg=az, gsd_m=5.0
        )
        c2, info2 = compute_dem_ray_shift_correction(
            pts, dem, 20.0, az, 5.0
        )
        assert info2.get("enabled") is True
        d2 = c2 - pts
        # Per-point cosine similarity; skip near-zero-displacement points.
        for v1, v2 in zip(d1, d2):
            if min(np.linalg.norm(v1), np.linalg.norm(v2)) < 1e-6:
                continue
            assert float(np.dot(_unit(v1), _unit(v2))) > 0.999, (
                f"az={az}: implementations disagree: {_unit(v1)} vs {_unit(v2)}"
            )


def test_missing_azimuth_fails_fast():
    """No hallucinated default direction: ValueError, not a silent 45-degree shift."""
    dem = _tilted_plane_dem()
    pts = np.array([[100.0, 100.0]])
    with pytest.raises(ValueError, match="azimuth_deg is required"):
        dem_ray_intersection(pts, dem, emission_deg=20.0, azimuth_deg=None, gsd_m=5.0)


def test_ransac_none_azimuth_no_crash_and_honest_frame():
    """azimuth=None disables correction (previously: TypeError crash)."""
    rng = np.random.default_rng(3)
    pts1 = rng.uniform(50, 200, (12, 2))
    pts2 = pts1 + rng.normal(0, 0.3, pts1.shape)
    dem = np.zeros((256, 256), dtype=np.float32)

    H, mask, info = ransac_dem_aware_fit(
        pts1, pts2, dem=dem, emission_deg=20.0, azimuth_deg=None, gsd_m=5.0
    )
    assert info["dem_compensated"] is False
    assert info["relief_reason"] == "los_azimuth_unavailable"
    assert info["frame"] == "raw_source_to_dst"

    # With a real azimuth the frame contract is corrected-frame + field stored.
    dem += 200.0  # non-flat so shifts are nonzero somewhere
    H2, mask2, info2 = ransac_dem_aware_fit(
        pts1, pts2, dem=dem, emission_deg=20.0, azimuth_deg=90.0, gsd_m=5.0
    )
    assert info2["frame"] == "relief_corrected_source_to_dst"
    assert np.asarray(info2["relief_correction_field_px"]).shape == (len(pts1), 2)
    assert info2["relief_max_shift_px"] >= 0.0
    assert info2["raw_frame_rmse_px"] is not None
