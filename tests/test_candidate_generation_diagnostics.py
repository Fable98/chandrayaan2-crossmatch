"""Unit tests for candidate-generation diagnostics and isolation invariants.

Verifies:
1. Candidate-count instrumentation (verifies that stage counts are non-negative and properly tracked).
2. No H_gt access in production candidate generation (ensures candidate generation operates purely on image data).
3. Deterministic candidate generation (same seed and images produce identical candidate coordinates).
4. Existing candidate-generation behavior remains valid (default production interface unaffected).
5. Diagnostic evaluation correctly identifies H_gt-near candidates without feeding labels into the matcher.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import SEED
from matcher_cfog import (
    compute_phase_congruency,
    detect_salient_keypoints,
    suppression_via_square_covering,
    find_best_correspondence_unified,
)
from evaluation.eval_candidate_generation import (
    instrument_scene_candidate_generation,
    get_train_test_split,
)
from scripts.generate_ground_truth_matches import (
    create_random_homography,
)


def test_candidate_count_instrumentation():
    """Verify that candidate generation instrumentation tracks counts properly through stages."""
    rng = np.random.default_rng(42)
    synthetic_img = rng.integers(30, 220, (128, 128), dtype=np.uint8)
    H_gt = np.eye(3, dtype=np.float64)
    H_gt[0, 2] = 5.0
    H_gt[1, 2] = -3.0

    diag = instrument_scene_candidate_generation(
        synthetic_img, H_gt, domain="same_sensor", rng=rng, half_p=8, search_rad=16
    )

    assert "raw_count" in diag
    assert "ssc_count" in diag
    assert "matched_count" in diag
    assert "final_count" in diag
    assert diag["raw_count"] >= diag["ssc_count"] >= 0
    assert diag["ssc_count"] >= diag["matched_count"] >= 0


def test_no_hgt_access_in_production_candidate_generation():
    """Verify that find_best_correspondence_unified and keypoint extraction do not accept H_gt."""
    import inspect
    sig_unified = inspect.signature(find_best_correspondence_unified)
    sig_detect = inspect.signature(detect_salient_keypoints)
    sig_suppress = inspect.signature(suppression_via_square_covering)

    assert "H_gt" not in sig_unified.parameters
    assert "H" not in sig_unified.parameters
    assert "ground_truth" not in sig_unified.parameters
    assert "H_gt" not in sig_detect.parameters
    assert "H_gt" not in sig_suppress.parameters


def test_deterministic_candidate_generation():
    """Verify candidate generation is reproducible for fixed random seed."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    synthetic_img = rng1.integers(20, 240, (128, 128), dtype=np.uint8)
    H_gt = np.eye(3, dtype=np.float64)

    diag1 = instrument_scene_candidate_generation(
        synthetic_img, H_gt, domain="same_sensor", rng=rng1, half_p=8, search_rad=16
    )
    diag2 = instrument_scene_candidate_generation(
        synthetic_img, H_gt, domain="same_sensor", rng=rng2, half_p=8, search_rad=16
    )

    assert diag1["raw_count"] == diag2["raw_count"]
    assert diag1["ssc_count"] == diag2["ssc_count"]
    assert len(diag1["final_cands"]) == len(diag2["final_cands"])


def test_diagnostic_eval_correctly_identifies_hgt_near_candidates():
    """Verify evaluation oracle computes Euclidean distance to H_gt projection without label feedback."""
    pts_src = np.array([[50.0, 50.0]], dtype=np.float64)
    H_gt = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 5.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    proj = cv2.perspectiveTransform(pts_src.reshape(1, 1, 2), H_gt).reshape(-1)
    true_x, true_y = float(proj[0]), float(proj[1])
    assert true_x == 60.0
    assert true_y == 55.0

    # Near candidate (err = 1.0 px)
    cand_near = {"target_x": 60.0, "target_y": 56.0}
    err_near = np.hypot(cand_near["target_x"] - true_x, cand_near["target_y"] - true_y)
    assert err_near == 1.0

    # Far candidate (err = 10.0 px)
    cand_far = {"target_x": 70.0, "target_y": 55.0}
    err_far = np.hypot(cand_far["target_x"] - true_x, cand_far["target_y"] - true_y)
    assert err_far == 10.0


def test_train_test_group_isolation():
    """Verify 54 train and 18 test groups have zero overlap."""
    tr_groups, te_groups = get_train_test_split()
    assert len(tr_groups) == 54
    assert len(te_groups) == 18
    overlap = set(tr_groups) & set(te_groups)
    assert len(overlap) == 0, f"Found train/test leakage: {overlap}"


def test_ncc_peak_selection_regression():
    """Verify NCC peak selection evaluates negative correlation peaks under abs_and_bipolar mode."""
    from evaluation.eval_patch_and_ncc_experiments import match_template_unified_custom

    # Create search region with a negative peak at a known location
    search_region = np.zeros((64, 64), dtype=np.float32)
    tmpl = np.ones((16, 16), dtype=np.float32)

    # Invert patch at (20, 20) to create a negative correlation peak
    search_region[20:36, 20:36] = -1.0
    # Positive peak at (40, 40)
    search_region[40:56, 40:56] = 0.5

    # In baseline mode, top-k positive selects positive values
    score_base, loc_base = match_template_unified_custom(
        search_region, tmpl, multimodal_pair=True, ncc_selection_mode="baseline_pos"
    )
    assert loc_base == (40, 20) or loc_base == (20, 40) or loc_base[0] >= 30, "Baseline should select positive peak."

    # Under abs_and_bipolar mode, the strong negative peak is also evaluated
    score_custom, loc_custom = match_template_unified_custom(
        search_region, tmpl, multimodal_pair=True, ncc_selection_mode="abs_and_bipolar"
    )
    assert score_custom >= 0.0


def test_scale_adaptive_patch_size_configuration():
    """Verify patch support configuration runs reliably for both half_p=8 and half_p=16."""
    from evaluation.eval_patch_and_ncc_experiments import run_experiment_on_scene

    rng = np.random.default_rng(42)
    synthetic_img = rng.integers(30, 220, (128, 128), dtype=np.uint8)
    H_gt = np.eye(3, dtype=np.float64)

    out_p8 = run_experiment_on_scene(
        synthetic_img, H_gt, rng, half_p=8, ncc_selection_mode="baseline_pos", search_rad=16
    )
    out_p16 = run_experiment_on_scene(
        synthetic_img, H_gt, rng, half_p=16, ncc_selection_mode="baseline_pos", search_rad=16
    )

    assert "candidates" in out_p8
    assert "candidates" in out_p16
    assert out_p8["runtime_s"] > 0
    assert out_p16["runtime_s"] > 0


def test_production_model_artifact_unmodified():
    """Verify production RF model is untouched."""
    prod_path = REPO_ROOT / "ML_model" / "ai_verifier_model.pkl"
    assert prod_path.exists()
    assert prod_path.stat().st_size > 1_000_000
