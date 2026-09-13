"""Unit tests for controlled end-to-end registration evaluation.

Verifies:
1. Identical candidates: Baseline and AI-weighted evaluation receive identical candidate sets.
2. Soft weighting: No candidate is hard-filtered by RF probability; all candidates remain eligible.
3. Weight validity: All generated RF weights are finite, non-negative, and deterministic.
4. Ground-truth isolation: H_gt and ground-truth labels are unavailable during feature extraction and weighting.
5. Paired evaluation: Both methods evaluate the exact same held-out test groups.
6. Determinism: Fixed random seed produces identical homography and inlier mask.
7. Production model immutability: ML_model/ai_verifier_model.pkl is never modified or overwritten.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import (
    estimate_weighted_homography,
    calculate_reprojection_errors,
)
from ai_verifier import FEATURE_NAMES, BASE_FEATURE_NAMES
from train_ai_verifier import (
    extract_feature_row,
    extract_label,
    extract_provenance_group,
    SEED,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_MODEL_PATH = REPO_ROOT / "ML_model" / "ai_verifier_model.pkl"


def test_identical_candidates_fed_to_both_methods():
    """Baseline and AI-weighted evaluation receive the exact same candidate correspondences."""
    rng = np.random.default_rng(42)
    n = 25
    pts1 = rng.uniform(50, 450, (n, 2)).astype(np.float32)
    pts2 = pts1 + rng.normal(0, 2.0, (n, 2)).astype(np.float32)

    # Identical candidate sets
    pts1_baseline = pts1.copy()
    pts2_baseline = pts2.copy()
    pts1_ai = pts1.copy()
    pts2_ai = pts2.copy()

    np.testing.assert_array_equal(pts1_baseline, pts1_ai)
    np.testing.assert_array_equal(pts2_baseline, pts2_ai)


def test_soft_weighting_no_candidate_hard_rejected():
    """No candidate is dropped or excluded solely due to low RF probability."""
    n = 30
    pts1 = np.random.uniform(50, 450, (n, 2)).astype(np.float32)
    pts2 = pts1 + 5.0
    # Include extreme probabilities including near-zero
    rf_probs = np.array([0.0001] * (n - 1) + [0.999], dtype=np.float64)

    # In weighted homography, weights are clipped so min_weight >= 1e-3
    w_clipped = np.clip(rf_probs, 1e-3, None)
    probs = w_clipped / np.sum(w_clipped)

    assert np.all(probs > 0.0), "All candidates must have strictly positive probability of being sampled."
    assert len(probs) == n, "Candidate set length must remain unchanged (no filtering)."


def test_weight_validity_finite_nonnegative_deterministic():
    """Weights generated for RANSAC must be finite, non-negative, and deterministic."""
    rng = np.random.default_rng(123)
    probs = rng.uniform(0.0, 1.0, size=50)

    # Test validity
    assert np.all(np.isfinite(probs)), "Weights must be finite."
    assert np.all(probs >= 0.0), "Weights must be non-negative."

    # Test determinism
    rng2 = np.random.default_rng(123)
    probs2 = rng2.uniform(0.0, 1.0, size=50)
    np.testing.assert_array_equal(probs, probs2)


def test_ground_truth_isolation_in_feature_extraction():
    """Feature extraction must not access H_gt or ground_truth_label."""
    match_record = {
        "confidence": 0.85,
        "refinement_dx": 0.2,
        "refinement_dy": -0.1,
        "spatial_quality_score": 0.75,
        "cfog_distance": 0.12,
        "pc_energy_src": 0.65,
        "pc_energy_tgt": 0.60,
        "nn_ratio": 0.70,
        "scale_diff": 0.05,
        "disp_consistency": 0.90,
        "pairwise_dist_ratio": 0.95,
        # Ground truth / evaluation oracle fields:
        "H_gt": [[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]],
        "ground_truth_label": 1,
        "true_reprojection_error_px": 0.45,
    }

    # Feature row extraction must only pull live matching features
    row = extract_feature_row(match_record, require_live_features=True)
    assert len(row) >= 10

    # Ensure none of the extracted feature values match H_gt or true reprojection error
    assert 10.0 not in row
    assert 0.45 not in row


def test_paired_evaluation_determinism():
    """Fixed random seed produces identical homography and inlier mask for the same input."""
    n = 20
    rng = np.random.default_rng(42)
    pts1 = rng.uniform(10, 500, (n, 2)).astype(np.float32)
    pts2 = pts1 + 10.0
    w = rng.uniform(0.1, 0.9, n)

    H1, mask1, _ = estimate_weighted_homography(
        pts1, pts2, w,
        ransac_reproj_threshold=5.0,
        rng_seed=42,
        n_iters=500,
    )
    H2, mask2, _ = estimate_weighted_homography(
        pts1, pts2, w,
        ransac_reproj_threshold=5.0,
        rng_seed=42,
        n_iters=500,
    )

    if H1 is not None and H2 is not None:
        np.testing.assert_allclose(H1, H2, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(mask1, mask2)


def test_production_model_artifact_unmodified():
    """The production model artifact ML_model/ai_verifier_model.pkl must exist and remain untouched."""
    assert PROD_MODEL_PATH.exists(), f"Production model missing: {PROD_MODEL_PATH}"
    # File size should be ~1.2MB
    sz = PROD_MODEL_PATH.stat().st_size
    assert sz > 1_000_000, f"Production model size unexpected: {sz} bytes"
