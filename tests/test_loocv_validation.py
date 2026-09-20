"""
tests/test_loocv_validation.py - Tests for Leave-One-Out Cross-Validation (LOOCV) for 4 <= N < 8.

Verifies:
1. Held-out independent checkpoints are kept separate from fitted inliers.
2. LOOCV is invoked only when independent checkpoints are unavailable.
3. Explicit label: validation_mode = "LOOCV".
4. LOOCV is never labeled as independent validation (is_independent_validation is False).
5. Reports loo_rmse, loo_median_error, loo_p95_error, and number_of_folds.
6. If N < 4, return FAIL.
7. If N is 4-7, return REVIEW unless external checkpoints exist.
8. Dedicated tests using exactly 4, 5, 7, and 8 inliers.
"""

import numpy as np
import pytest
import cv2

from ML_model.metrics import (
    compute_canonical_metrics,
    evaluate_loocv_validation,
    evaluate_independent_checkpoints,
)


def _generate_synthetic_inliers(n: int, noise_std: float = 0.05, seed: int = 42):
    """Generates n well-distributed synthetic point correspondences with a rigid shift."""
    rng = np.random.RandomState(seed)
    # Ensure points are non-collinear and well-spread across a 500x500 area
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 150.0
    center_x, center_y = 256.0, 256.0
    src = np.stack([
        center_x + radius * np.cos(angles),
        center_y + radius * np.sin(angles),
    ], axis=1).astype(np.float64)

    # Shift by (3.0, 2.0) with optional Gaussian noise
    shift = np.array([3.0, 2.0], dtype=np.float64)
    dst = src + shift + rng.normal(0.0, noise_std, src.shape)

    H = np.eye(3, dtype=np.float64)
    H[0, 2] = shift[0]
    H[1, 2] = shift[1]

    return src, dst, H


# ===========================================================================
# 1. Exactly 4 Inliers Tests
# ===========================================================================

def test_loocv_exactly_4_inliers():
    """Test LOOCV with exactly 4 inliers without checkpoints."""
    src, dst, H = _generate_synthetic_inliers(4, noise_std=0.05, seed=101)
    mask = np.ones(4, dtype=np.uint8)

    # Direct function test
    loocv_res = evaluate_loocv_validation(src, dst)
    assert loocv_res["validation_status"] == "evaluated"
    assert loocv_res["validation_mode"] == "LOOCV"
    assert loocv_res["is_independent_validation"] is False
    assert loocv_res["number_of_folds"] == 4
    assert loocv_res["loo_rmse"] is not None
    assert loocv_res["loo_rmse"] < 1.0
    assert loocv_res["loo_median_error"] is not None
    assert loocv_res["loo_p95_error"] is not None

    # Canonical metrics test
    metrics = compute_canonical_metrics(src, dst, mask, H, image_shape=(512, 512))
    assert metrics["inlier_count"] == 4
    assert metrics["validation_mode"] == "LOOCV"
    assert metrics["is_independent_validation"] is False
    assert metrics["number_of_folds"] == 4
    assert metrics["loo_rmse"] == loocv_res["loo_rmse"]
    assert metrics["loo_median_error"] == loocv_res["loo_median_error"]
    assert metrics["loo_p95_error"] == loocv_res["loo_p95_error"]
    # Requirement 7: N is 4-7 without external checkpoints -> REVIEW
    assert metrics["verdict"] == "REVIEW"
    assert metrics["has_independent_checkpoints"] is False
    # Requirement 4: Never label LOOCV as independent validation
    assert metrics["sub_pixel_accurate"] is False


# ===========================================================================
# 2. Exactly 5 Inliers Tests
# ===========================================================================

def test_loocv_exactly_5_inliers():
    """Test LOOCV with exactly 5 inliers without checkpoints."""
    src, dst, H = _generate_synthetic_inliers(5, noise_std=0.08, seed=102)
    mask = np.ones(5, dtype=np.uint8)

    metrics = compute_canonical_metrics(src, dst, mask, H, image_shape=(512, 512))
    assert metrics["inlier_count"] == 5
    assert metrics["validation_mode"] == "LOOCV"
    assert metrics["is_independent_validation"] is False
    assert metrics["number_of_folds"] == 5
    assert metrics["loo_rmse"] is not None
    assert metrics["loo_rmse"] < 1.0
    assert metrics["loo_median_error"] is not None
    assert metrics["loo_p95_error"] is not None
    # Requirement 7: N is 4-7 -> REVIEW
    assert metrics["verdict"] == "REVIEW"
    assert metrics["has_independent_checkpoints"] is False
    assert metrics["sub_pixel_accurate"] is False


# ===========================================================================
# 3. Exactly 7 Inliers Tests
# ===========================================================================

def test_loocv_exactly_7_inliers():
    """Test LOOCV with exactly 7 inliers without checkpoints."""
    src, dst, H = _generate_synthetic_inliers(7, noise_std=0.06, seed=103)
    mask = np.ones(7, dtype=np.uint8)

    metrics = compute_canonical_metrics(src, dst, mask, H, image_shape=(512, 512))
    assert metrics["inlier_count"] == 7
    assert metrics["validation_mode"] == "LOOCV"
    assert metrics["is_independent_validation"] is False
    assert metrics["number_of_folds"] == 7
    assert metrics["loo_rmse"] is not None
    assert metrics["loo_rmse"] < 1.0
    assert metrics["loo_median_error"] is not None
    assert metrics["loo_p95_error"] is not None
    # Requirement 7: N is 4-7 -> REVIEW
    assert metrics["verdict"] == "REVIEW"
    assert metrics["has_independent_checkpoints"] is False
    assert metrics["sub_pixel_accurate"] is False


# ===========================================================================
# 4. Exactly 8 Inliers Tests
# ===========================================================================

def test_validation_exactly_8_inliers():
    """Test that exactly 8 inliers transition from LOOCV to 80/20 train/test holdout."""
    src, dst, H = _generate_synthetic_inliers(8, noise_std=0.05, seed=104)
    mask = np.ones(8, dtype=np.uint8)

    metrics = compute_canonical_metrics(src, dst, mask, H, image_shape=(512, 512))
    assert metrics["inlier_count"] == 8
    # 8 points uses standard 80/20 train/test holdout
    assert metrics["validation_mode"] == "HELD_OUT_80_20"
    assert metrics["is_independent_validation"] is False
    # LOOCV fields are None / not active
    assert metrics["loo_rmse"] is None
    assert metrics["number_of_folds"] in (0, None)
    # Held-out RMSE is populated
    assert metrics["validation_rmse_px"] is not None
    assert metrics["held_out_validation_rmse_px"] is not None


# ===========================================================================
# 5. Fewer than 4 Inliers (N < 4) Tests -> FAIL
# ===========================================================================

@pytest.mark.parametrize("n_inliers", [1, 2, 3])
def test_loocv_fewer_than_4_inliers_fails(n_inliers: int):
    """Requirement 6: If N < 4, return FAIL."""
    src, dst, H = _generate_synthetic_inliers(max(4, n_inliers), seed=105)
    src_sub = src[:n_inliers]
    dst_sub = dst[:n_inliers]
    mask = np.ones(n_inliers, dtype=np.uint8)

    # Direct LOOCV function
    loocv_res = evaluate_loocv_validation(src_sub, dst_sub)
    assert loocv_res["validation_status"] == "insufficient_points_for_loocv"
    assert loocv_res["loo_rmse"] is None
    assert loocv_res["number_of_folds"] == n_inliers

    # Canonical metrics
    metrics = compute_canonical_metrics(src_sub, dst_sub, mask, H, image_shape=(512, 512))
    assert metrics["inlier_count"] == n_inliers
    assert metrics["verdict"] == "FAIL"
    assert metrics["loo_rmse"] is None


# ===========================================================================
# 6. Independent Checkpoint Separation & LOOCV Suppression
# ===========================================================================

def test_independent_checkpoints_separation_and_priority():
    """
    Requirement 1 & 2:
    - Keep held-out independent checkpoints separate.
    - Use LOOCV strictly when independent checkpoints are unavailable.
    """
    src, dst, H = _generate_synthetic_inliers(5, noise_std=0.05, seed=106)
    mask = np.ones(5, dtype=np.uint8)

    # Independent checkpoints separate from fitted inliers
    chk_src = np.array([[100.0, 100.0], [400.0, 400.0], [100.0, 400.0], [400.0, 100.0]], dtype=np.float64)
    # True shift is (3.0, 2.0)
    chk_dst = chk_src + np.array([3.0, 2.0])

    metrics = compute_canonical_metrics(
        src, dst, mask, H,
        image_shape=(512, 512),
        independent_checkpoints=(chk_src, chk_dst)
    )

    # Independent checkpoints were available, so LOOCV was NOT used
    assert metrics["validation_mode"] == "INDEPENDENT_CHECKPOINTS"
    assert metrics["is_independent_validation"] is True
    assert metrics["has_independent_checkpoints"] is True
    assert metrics["loo_rmse"] is None
    assert metrics["number_of_folds"] in (0, None)

    # Checkpoint metrics are reported separately
    assert metrics["checkpoint_rmse_px"] is not None
    assert metrics["checkpoint_rmse_px"] < 0.1
    assert metrics["checkpoint_points_count"] == 4
    assert metrics["held_out_rmse"] == metrics["checkpoint_rmse_px"]
    assert metrics["held_out_validation_rmse_px"] == metrics["checkpoint_rmse_px"]


# ===========================================================================
# 7. N in 4-7 Can Return PASS With External Checkpoints
# ===========================================================================

def test_n_between_4_and_7_passes_with_valid_checkpoints():
    """
    Requirement 7: If N is 4-7, return REVIEW unless external checkpoints exist.
    With external checkpoints and excellent registration, verdict can be PASS.
    """
    src, dst, H = _generate_synthetic_inliers(5, noise_std=0.01, seed=107)
    mask = np.ones(5, dtype=np.uint8)

    chk_src = np.array([[120.0, 120.0], [380.0, 380.0], [120.0, 380.0], [380.0, 120.0]], dtype=np.float64)
    chk_dst = chk_src + np.array([3.0, 2.0])

    # Provide synthetic identical images with valid overlap to pass structural gates
    synth_ref = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(synth_ref, (256, 256), 50, 255, -1)
    synth_src = synth_ref.copy()
    synth_warped = synth_ref.copy()

    metrics = compute_canonical_metrics(
        src, dst, mask, H,
        image_shape=(512, 512),
        independent_checkpoints=(chk_src, chk_dst),
        ref_img=synth_ref,
        source_img=synth_src,
        warped_source=synth_warped,
    )

    assert metrics["has_independent_checkpoints"] is True
    assert metrics["validation_mode"] == "INDEPENDENT_CHECKPOINTS"
    assert metrics["is_independent_validation"] is True
    assert metrics["checkpoint_rmse_px"] < 0.1
    # Since external checkpoints exist and all gates pass, verdict is PASS
    assert metrics["verdict"] == "PASS"


# ===========================================================================
# 8. Never Label LOOCV as Independent Validation (Requirement 4)
# ===========================================================================

def test_loocv_never_labeled_as_independent():
    """
    Requirement 4: Never label LOOCV as independent validation.
    Ensure is_independent_validation is strictly False and sub_pixel_accurate is not granted.
    """
    src, dst, H = _generate_synthetic_inliers(6, noise_std=0.01, seed=108)
    mask = np.ones(6, dtype=np.uint8)

    metrics = compute_canonical_metrics(src, dst, mask, H, image_shape=(512, 512))
    assert metrics["validation_mode"] == "LOOCV"
    assert metrics["is_independent_validation"] is False
    assert metrics["held_out_rmse"] is None  # Held-out checkpoints are separate
    assert metrics["held_out_validation_rmse_px"] is None
    assert metrics["sub_pixel_accurate"] is False
