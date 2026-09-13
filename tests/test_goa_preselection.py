"""Unit and regression tests for Phase 8: GOA Preselection."""

import inspect
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import (
    compute_goa_search_surface,
    find_best_correspondence_unified,
    compute_phase_congruency,
)


def test_goa_polarity_invariance():
    """Verify that a 180-degree polarity flipped patch produces near-maximum GOA."""
    # Synthetic oriented edge patch
    y, x = np.mgrid[:16, :16].astype(np.float32)
    tmpl = np.sin((x + 2.0 * y) * 0.4)

    # Search region containing:
    # 1. Exact copy at (5, 5)
    # 2. 180-deg flipped copy at (15, 15)
    # 3. Orthogonal edge at (25, 25)
    sr = np.zeros((48, 48), dtype=np.float32)
    sr[5:21, 5:21] = tmpl
    sr[15:31, 15:31] = -tmpl  # Flipped polarity
    ortho = np.sin((-2.0 * x + y) * 0.4)
    sr[25:41, 25:41] = ortho  # Orthogonal

    goa = compute_goa_search_surface(sr, tmpl)

    # GOA at exact copy should be high (near 1.0)
    assert goa[5, 5] > 0.85, f"Expected high GOA at direct copy, got {goa[5, 5]}"

    # GOA at flipped copy MUST ALSO be high (near 1.0) due to polarity invariance
    assert goa[15, 15] > 0.85, f"Expected high GOA at flipped copy, got {goa[15, 15]}"

    # GOA at orthogonal edge should be significantly lower (near 0.0)
    assert goa[25, 25] < 0.30, f"Expected low GOA at orthogonal edge, got {goa[25, 25]}"

    # Standard NCC at (15, 15) should be negative (-1.0), whereas GOA is positive (> 0.85)
    ncc = cv2.matchTemplate(sr, tmpl, cv2.TM_CCOEFF_NORMED)
    assert ncc[15, 15] < -0.80, f"Expected negative NCC at flipped copy, got {ncc[15, 15]}"


def test_same_sensor_path_unchanged():
    """Verify that when multimodal_pair=False, GOA is bypassed and output is 100% identical."""
    rng = np.random.default_rng(42)
    sr = rng.normal(0, 1, (40, 40)).astype(np.float32)
    tmpl = rng.normal(0, 1, (16, 16)).astype(np.float32)

    # Run with use_goa_preselection True vs False when multimodal_pair=False
    score1, loc1 = find_best_correspondence_unified(
        sr, tmpl, multimodal_pair=False, use_goa_preselection=True
    )
    score2, loc2 = find_best_correspondence_unified(
        sr, tmpl, multimodal_pair=False, use_goa_preselection=False
    )

    assert loc1 == loc2, f"Locations differ: {loc1} vs {loc2}"
    assert abs(score1 - score2) < 1e-7, f"Scores differ: {score1} vs {score2}"

    # Also compare with pure matchTemplate
    res = cv2.matchTemplate(sr, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    assert loc1 == max_loc
    assert abs(score1 - float(max_val)) < 1e-7


def test_goa_determinism():
    """Verify that GOA surface computation is strictly deterministic."""
    rng = np.random.default_rng(123)
    sr = rng.normal(0, 1, (35, 35)).astype(np.float32)
    tmpl = rng.normal(0, 1, (16, 16)).astype(np.float32)

    goa1 = compute_goa_search_surface(sr, tmpl)
    goa2 = compute_goa_search_surface(sr, tmpl)

    np.testing.assert_array_equal(goa1, goa2)


def test_no_h_gt_leakage():
    """Verify that find_best_correspondence_unified and compute_goa_search_surface take no H_gt argument."""
    sig1 = inspect.signature(compute_goa_search_surface)
    sig2 = inspect.signature(find_best_correspondence_unified)
    assert "H_gt" not in sig1.parameters
    assert "H_gt" not in sig2.parameters
    assert "gt" not in sig1.parameters
    assert "gt" not in sig2.parameters
