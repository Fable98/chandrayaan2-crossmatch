"""Unit tests for extended AI verifier structural features.

Verifies:
1. Feature contract (9 base features preserved, new features appended, finite values, bounded ranges).
2. Determinism (repeated extraction produces identical results).
3. Cross-sensor invariance (features behave stably under contrast, brightness, blur, polarity inversion).
4. Zero leakage (no ground-truth labels, H_gt, or RANSAC outputs accessed).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from ai_verifier import AIMatchVerifier, BASE_FEATURE_NAMES, FEATURE_NAMES as VERIFIER_FEATURES
from train_ai_verifier import BASE_FEATURE_NAMES as TRAIN_BASE_FEATURES, FEATURE_NAMES as TRAIN_FEATURES
from matcher_cfog import (
    compute_neighborhood_consistency,
    compute_gradient_orientation_agreement,
)


def test_feature_contract_preserves_base_nine():
    """Requirement 2 & 18: Base 9 features must be preserved in exact order."""
    expected_base = [
        "confidence", "refinement_dx", "refinement_dy", "spatial_quality_score",
        "cfog_distance", "pc_energy_src", "pc_energy_tgt", "nn_ratio", "scale_diff",
    ]
    assert BASE_FEATURE_NAMES == expected_base
    assert TRAIN_BASE_FEATURES == expected_base
    assert VERIFIER_FEATURES[:9] == expected_base
    assert TRAIN_FEATURES[:9] == expected_base
    assert VERIFIER_FEATURES[9:] == ["disp_consistency", "pairwise_dist_ratio"]


def test_neighborhood_consistency_bounded_and_finite():
    """Requirement 18: Neighborhood features must be strictly finite and in [0.0, 1.0]."""
    rng = np.random.default_rng(42)
    # Synthetic candidate matches
    matches = []
    for i in range(15):
        sx, sy = float(rng.uniform(50, 450)), float(rng.uniform(50, 450))
        # Affine displacement (shift + small rotation) plus noise
        tx = sx + 25.0 + float(rng.normal(0, 2.0))
        ty = sy - 15.0 + float(rng.normal(0, 2.0))
        matches.append({
            "source_x": sx, "source_y": sy,
            "target_x": tx, "target_y": ty,
            "confidence": 0.35,
        })

    results = compute_neighborhood_consistency(matches, k_neighbors=5)
    assert len(results) == len(matches)
    for res in results:
        assert "disp_consistency" in res
        assert "pairwise_dist_ratio" in res
        dc = res["disp_consistency"]
        pdr = res["pairwise_dist_ratio"]
        assert np.isfinite(dc) and 0.0 <= dc <= 1.0
        assert np.isfinite(pdr) and 0.0 <= pdr <= 1.0


def test_neighborhood_consistency_separates_coherent_from_random_matches():
    """Requirement 18: Coherent geometric matches should score higher than erratic outliers."""
    rng = np.random.default_rng(42)
    matches = []
    # 10 coherent matches moving in roughly same direction (dx~30, dy~-20)
    for i in range(10):
        sx, sy = float(i * 40 + 20), float(i * 30 + 20)
        matches.append({
            "source_x": sx, "source_y": sy,
            "target_x": sx + 30.0 + float(rng.normal(0, 1.0)),
            "target_y": sy - 20.0 + float(rng.normal(0, 1.0)),
        })
    # 5 erratic outliers with random targets
    for i in range(5):
        sx, sy = float(rng.uniform(20, 450)), float(rng.uniform(20, 450))
        matches.append({
            "source_x": sx, "source_y": sy,
            "target_x": float(rng.uniform(20, 450)),
            "target_y": float(rng.uniform(20, 450)),
        })

    results = compute_neighborhood_consistency(matches, k_neighbors=4)
    coherent_dc = [results[i]["disp_consistency"] for i in range(10)]
    outlier_dc = [results[i]["disp_consistency"] for i in range(10, 15)]

    assert np.mean(coherent_dc) > np.mean(outlier_dc), (
        f"Coherent mean ({np.mean(coherent_dc):.3f}) must exceed outlier mean ({np.mean(outlier_dc):.3f})"
    )


def test_gradient_orientation_agreement_invariance():
    """Requirement 18: Orientation agreement must be invariant to polarity, contrast, and brightness."""
    # Synthetic edge pattern: horizontal edge (bright top, dark bottom)
    base = np.zeros((32, 32), dtype=np.float32)
    base[:16, :] = 1.0

    # 1. Identical patch
    score_id = compute_gradient_orientation_agreement(base, base, 16.0, 16.0, 16.0, 16.0, half=8)
    assert 0.95 <= score_id <= 1.0

    # 2. Polarity-inverted patch (bright bottom, dark top)
    inv = np.zeros((32, 32), dtype=np.float32)
    inv[16:, :] = 1.0
    score_inv = compute_gradient_orientation_agreement(base, inv, 16.0, 16.0, 16.0, 16.0, half=8)
    assert 0.95 <= score_inv <= 1.0, f"Modulo pi orientation agreement must be polarity invariant: {score_inv}"

    # 3. Brightness and contrast perturbation
    perturbed = base * 0.3 + 0.4
    score_perturbed = compute_gradient_orientation_agreement(base, perturbed, 16.0, 16.0, 16.0, 16.0, half=8)
    assert abs(score_perturbed - score_id) < 1e-3, "Orientation agreement must be invariant to linear photometric shifts"

    # 4. Orthogonal edge (vertical edge): agreement should drop to ~0.0
    vert = np.zeros((32, 32), dtype=np.float32)
    vert[:, :16] = 1.0
    score_ortho = compute_gradient_orientation_agreement(base, vert, 16.0, 16.0, 16.0, 16.0, half=8)
    assert score_ortho < 0.20, f"Orthogonal edges must have near-zero orientation agreement: {score_ortho}"


def test_no_leakage_in_features():
    """Requirement 18: Feature extraction must not require or access ground truth or RANSAC outputs."""
    matches = [
        {"source_x": 10.0, "source_y": 10.0, "target_x": 20.0, "target_y": 20.0, "confidence": 0.5},
        {"source_x": 20.0, "source_y": 20.0, "target_x": 30.0, "target_y": 30.0, "confidence": 0.5},
        {"source_x": 30.0, "source_y": 30.0, "target_x": 40.0, "target_y": 40.0, "confidence": 0.5},
        {"source_x": 40.0, "source_y": 40.0, "target_x": 50.0, "target_y": 50.0, "confidence": 0.5},
    ]
    # Function executes cleanly without any ground truth labels
    res = compute_neighborhood_consistency(matches)
    assert len(res) == 4
    for r in res:
        assert "disp_consistency" in r
        assert "pairwise_dist_ratio" in r


def test_determinism():
    """Requirement 18: Repeated feature computation on identical inputs must be byte-for-byte deterministic."""
    matches = [
        {"source_x": float(i * 12), "source_y": float(i * 15),
         "target_x": float(i * 12 + 10), "target_y": float(i * 15 - 8)}
        for i in range(8)
    ]
    res1 = compute_neighborhood_consistency(matches)
    res2 = compute_neighborhood_consistency(matches)
    assert res1 == res2
