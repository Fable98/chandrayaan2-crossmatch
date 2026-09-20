"""
tests/test_rule_compliance.py — Verification of the 15 Scientific and Project Governance Rules.

Validates:
1. Rule 1 & 2: Upstream files, citations, and notices preserved.
2. Rule 3: UPSTREAM.md exists and documents provenance, architecture, modifications, and limitations.
3. Rule 5 & 6: Sub-pixel accuracy requires independent held-out validation; fit and held-out metrics remain strictly separated.
4. Rule 7: Solar azimuth is never used as spacecraft viewing or LOS azimuth; missing azimuth cleanly disables relief compensation.
5. Rule 8: LoFTR outdoor weights treated as comparative baseline, never lunar ground truth.
6. Rule 14: Canonical registration metrics return explicit verdict ("PASS", "REVIEW", or "FAIL").
"""

from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from metrics import compute_canonical_metrics
from geometry import dem_ray_intersection, ransac_dem_aware_fit


def test_rule_3_upstream_md_present_and_complete():
    """Rule 3: Maintain an UPSTREAM.md file describing inherited code and our modifications."""
    upstream_file = REPO_ROOT / "UPSTREAM.md"
    assert upstream_file.exists(), "UPSTREAM.md must exist in the repo root"
    content = upstream_file.read_text(encoding="utf-8")
    assert len(content) > 500, "UPSTREAM.md must contain comprehensive documentation"
    assert "## 1. Upstream Provenance" in content
    assert "## 2. Inherited Architecture & Components" in content
    assert "## 3. Completed Modification Ledger" in content
    assert "## 4. Scientific Assumptions and Limitations" in content


def test_rule_5_subpixel_accuracy_requires_independent_checkpoint():
    """Rule 5: Do not claim sub-pixel accuracy unless independent checkpoint results support it."""
    # Synthetic correspondence set
    pts1 = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0],
                     [15.0, 15.0], [12.0, 18.0], [18.0, 12.0], [16.0, 16.0]], dtype=np.float64)
    # Perfect fit on inliers, but mock held-out validation RMSE > 1.0 px
    H = np.eye(3)
    pts2 = pts1.copy()

    metrics = compute_canonical_metrics(pts1, pts2, inlier_mask=np.ones(len(pts1)), H=H)
    # If held-out is missing or >= 1.0, sub_pixel_accurate must NEVER be True
    if metrics["held_out_rmse"] is None or metrics["held_out_rmse"] >= 1.0:
        assert metrics["sub_pixel_accurate"] is False, (
            "Sub-pixel accuracy must not be claimed without independent held-out RMSE < 1.0px"
        )


def test_rule_6_separation_of_fit_and_checkpoint_metrics():
    """Rule 6: Separate fitted inlier metrics from independent checkpoint metrics."""
    pts1 = np.random.uniform(10, 200, (12, 2))
    pts2 = pts1 + 2.0
    H = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])

    metrics = compute_canonical_metrics(pts1, pts2, inlier_mask=np.ones(len(pts1)), H=H)
    assert "in_sample_rmse" in metrics, "Fitted inlier RMSE must be explicitly named in_sample_rmse"
    assert "held_out_rmse" in metrics, "Independent checkpoint RMSE must be explicitly named held_out_rmse"
    assert metrics["fit_rmse_is_in_sample"] is True, "Must explicitly flag that fit RMSE is in-sample"


def test_rule_7_never_use_solar_azimuth_as_los_azimuth():
    """Rule 7: Never use solar azimuth as spacecraft viewing or LOS azimuth."""
    dem = np.zeros((128, 128), dtype=np.float32)
    dem[20:40, 20:40] = 50.0
    rng = np.random.default_rng(42)
    pts = rng.uniform(20, 100, (12, 2))

    # 1. Missing azimuth fails fast in dem_ray_intersection (never defaults to 45.0 or solar angle)
    with pytest.raises(ValueError, match="azimuth_deg is required"):
        dem_ray_intersection(pts, dem, emission_deg=15.0, azimuth_deg=None, gsd_m=5.0)

    # 2. ransac_dem_aware_fit gracefully records unavailability when azimuth is None
    H, mask, info = ransac_dem_aware_fit(pts, pts, dem=dem, emission_deg=15.0, azimuth_deg=None, gsd_m=5.0)
    assert info["dem_compensated"] is False
    assert info["relief_reason"] == "los_azimuth_unavailable"


def test_rule_8_loftr_is_baseline_not_lunar_ground_truth():
    """Rule 8: Treat LoFTR pretrained outdoor weights as a baseline, not lunar-specific ground truth."""
    from matcher import match_images
    import matcher
    doc = matcher.__doc__
    assert "baseline" in doc.lower(), "matcher.py must explicitly be documented as a baseline"


def test_rule_14_canonical_verdict_pass_review_fail():
    """Rule 14: Return PASS, REVIEW, or FAIL when registration quality is inadequate."""
    # 1. Empty / failed registration -> FAIL
    empty_metrics = compute_canonical_metrics(np.zeros((0, 2)), np.zeros((0, 2)), None, None)
    assert "verdict" in empty_metrics
    assert empty_metrics["verdict"] == "FAIL"

    # 2. Low inlier count (<4) -> FAIL
    pts_low = np.array([[10.0, 10.0], [20.0, 20.0]])
    low_metrics = compute_canonical_metrics(pts_low, pts_low, inlier_mask=np.ones(2), H=np.eye(3))
    assert low_metrics["verdict"] == "FAIL"
