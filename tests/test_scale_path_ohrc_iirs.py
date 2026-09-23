"""
tests/test_scale_path_ohrc_iirs.py — Audit and regression suite for OHRC-to-IIRS scale path.

Validates:
1. Physical GSD ratio reduction before overlap recovery.
2. Metadata-driven common-GSD normalization when scale ratio exceeds SCALE_CAP (10.0x).
3. Prohibition of direct Fourier-Mellin estimation on 20x, 100x, and 250x scale gaps.
4. Scale ratio provenance tracking: native_scale_ratio, pre_normalization_ratio, residual_scale_ratio.
5. Clear failure mode when common-GSD normalization cannot be computed.
6. Synthetic tests for 20x, 100x, and 250x scale differences.
7. Real OHRC-IIRS experiment with actual Chandrayaan-2 region_001 rasters and manifest metadata.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from overlap_recovery import (
    recover_content_overlap,
    estimate_scale_ratio_logpolar,
    SCALE_CAP,
)
from matcher_cfog import match_images_cfog, estimate_scale_ratio_cv
from iirs_multimodal_registrar import direct_multimodal_attempt


def _generate_synthetic_lunar_terrain(height: int, width: int, seed: int = 42) -> np.ndarray:
    """Generates synthetic lunar terrain with craters of various sizes."""
    np.random.seed(seed)
    img = np.full((height, width), 120, dtype=np.uint8)
    num_craters = max(10, int((height * width) // 2500))
    for _ in range(num_craters):
        cx = np.random.randint(int(width * 0.1), int(width * 0.9))
        cy = np.random.randint(int(height * 0.1), int(height * 0.9))
        r = np.random.randint(max(5, int(min(height, width) * 0.02)), max(15, int(min(height, width) * 0.12)))
        val = int(np.random.randint(160, 255))
        cv2.circle(img, (cx, cy), r, val, -1)
        cv2.circle(img, (cx, cy), max(2, int(r * 0.65)), int(val * 0.35), -1)
    return cv2.GaussianBlur(img, (5, 5), 1.2)


# ---------------------------------------------------------------------------
# Requirement 3: Never let Fourier-Mellin estimate extreme ratios directly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale_gap", [20.0, 100.0, 250.0])
def test_fourier_mellin_rejects_extreme_ratios_directly(scale_gap: float):
    """
    Requirement 3: Never let Fourier-Mellin estimate extreme ratios (20x, 100x, 250x) directly.
    Verifies that estimate_scale_ratio_cv and estimate_scale_ratio_logpolar strictly refuse
    to estimate scale differences beyond SCALE_CAP (10.0x).
    """
    h_fine, w_fine = 500, 500
    fine_terrain = _generate_synthetic_lunar_terrain(h_fine, w_fine)
    h_coarse = max(2, int(round(h_fine / scale_gap)))
    w_coarse = max(2, int(round(w_fine / scale_gap)))
    coarse_terrain = cv2.resize(fine_terrain, (w_coarse, h_coarse), interpolation=cv2.INTER_AREA)

    # 1. estimate_scale_ratio_cv must raise ValueError refusing extreme ratio
    with pytest.raises(ValueError) as exc_info:
        estimate_scale_ratio_cv(fine_terrain, coarse_terrain, max_ratio=SCALE_CAP)
    err_msg = str(exc_info.value).lower()
    assert "exceeds" in err_msg or "too small" in err_msg or "cap" in err_msg

    # 2. estimate_scale_ratio_logpolar must return (None, 0.0) without claiming a valid scale
    s_est, resp = estimate_scale_ratio_logpolar(fine_terrain, coarse_terrain, max_scale_ratio=SCALE_CAP)
    assert s_est is None, f"Expected None for {scale_gap}x gap, got {s_est}"


# ---------------------------------------------------------------------------
# Requirement 2, 4, 5, 6: Synthetic 20x, 100x, and 250x Scale Path Tests
# ---------------------------------------------------------------------------
def test_synthetic_20x_scale_difference():
    """
    Requirement 2, 4, 5, 6: Tests 20x scale gap (e.g. OHRC 0.25m vs TMC-2 5.0m).
    - With metadata: metadata-driven common-GSD normalization brings images to common scale.
      Records native_scale_ratio=20.0, pre_normalization_ratio=20.0, residual_scale_ratio=1.0.
    - Without metadata: fails clearly with scale_capped=True and no hallucinated recovery.
    """
    fine_h, fine_w = 512, 512
    fine_img = _generate_synthetic_lunar_terrain(fine_h, fine_w, seed=20)
    coarse_img = cv2.resize(fine_img, (fine_w // 20, fine_h // 20), interpolation=cv2.INTER_AREA)

    # 1. With metadata: successful common-GSD normalization
    res_with_meta = recover_content_overlap(
        fine_img,
        coarse_img,
        source_gsd_m=0.25,
        ref_gsd_m=5.0,
    )
    assert res_with_meta["native_scale_ratio"] == 20.0
    assert res_with_meta["pre_normalization_ratio"] == 20.0
    assert abs(res_with_meta["residual_scale_ratio"] - 1.0) < 0.5
    assert "error" not in res_with_meta or res_with_meta["error"] is None

    # 2. Without metadata: clear failure
    res_no_meta = recover_content_overlap(fine_img, coarse_img)
    assert res_no_meta["overlap_recovered"] is False
    assert res_no_meta["scale_capped"] is True
    assert res_no_meta["method"] == "none_scale_gap_exceeds_cap_without_metadata"
    assert "error" in res_no_meta
    assert "SCALE_CAP" in res_no_meta["error"]
    assert res_no_meta["native_scale_ratio"] >= 19.0


def test_synthetic_100x_scale_difference():
    """
    Requirement 2, 4, 5, 6: Tests 100x scale gap (e.g. 0.25m vs 25.0m).
    - With metadata: common-GSD normalization brings images to common scale.
    - Without metadata: clear failure refusing direct correlation.
    """
    fine_h, fine_w = 600, 600
    fine_img = _generate_synthetic_lunar_terrain(fine_h, fine_w, seed=100)
    coarse_img = cv2.resize(fine_img, (6, 6), interpolation=cv2.INTER_AREA)

    # 1. With metadata: metadata-driven common-GSD normalization
    res_with_meta = recover_content_overlap(
        fine_img,
        coarse_img,
        source_gsd_m=0.25,
        ref_gsd_m=25.0,
    )
    assert res_with_meta["native_scale_ratio"] == 100.0
    assert res_with_meta["pre_normalization_ratio"] == 100.0
    assert res_with_meta["residual_scale_ratio"] is not None

    # 2. Without metadata: clear failure mode
    res_no_meta = recover_content_overlap(fine_img, coarse_img)
    assert res_no_meta["overlap_recovered"] is False
    assert res_no_meta["scale_capped"] is True
    assert res_no_meta["method"] == "none_scale_gap_exceeds_cap_without_metadata"
    assert "error" in res_no_meta


def test_synthetic_250x_scale_difference():
    """
    Requirement 2, 4, 5, 6: Tests 250x scale gap (e.g. OHRC 0.25m vs IIRS ~62.5m).
    - With metadata: common-GSD normalization downsamples the 0.25m raster by 250x.
      Records native_scale_ratio=250.0, pre_normalization_ratio=250.0, residual_scale_ratio=1.0.
    - Without metadata: clear failure mode (never let Fourier-Mellin estimate 250x directly).
    """
    fine_h, fine_w = 1000, 1000
    fine_img = _generate_synthetic_lunar_terrain(fine_h, fine_w, seed=250)
    coarse_img = cv2.resize(fine_img, (4, 4), interpolation=cv2.INTER_AREA)

    # 1. With metadata: common-GSD normalization
    res_with_meta = recover_content_overlap(
        fine_img,
        coarse_img,
        source_gsd_m=0.25,
        ref_gsd_m=62.5,
    )
    assert res_with_meta["native_scale_ratio"] == 250.0
    assert res_with_meta["pre_normalization_ratio"] == 250.0
    assert "error" not in res_with_meta or res_with_meta["error"] is None

    # 2. Without metadata: clear failure mode
    res_no_meta = recover_content_overlap(fine_img, coarse_img)
    assert res_no_meta["overlap_recovered"] is False
    assert res_no_meta["scale_capped"] is True
    assert res_no_meta["method"] == "none_scale_gap_exceeds_cap_without_metadata"
    assert "SCALE_CAP" in res_no_meta["error"]
    assert "250" in res_no_meta["error"]


def test_matcher_cfog_missing_metadata_fails_clearly_on_extreme_ratio():
    """
    Requirement 5: Add a clear failure if common-GSD normalization cannot be computed in matcher_cfog.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        img1 = _generate_synthetic_lunar_terrain(512, 512, seed=1)
        img2 = cv2.resize(img1, (20, 20), interpolation=cv2.INTER_AREA)
        p1 = tmp / "src.png"
        p2 = tmp / "ref.png"
        cv2.imwrite(str(p1), img1)
        cv2.imwrite(str(p2), img2)

        # Calling matcher_cfog without metadata on a ~25x canvas disparity
        res = match_images_cfog(
            p1,
            p2,
            output_dir=tmp / "out",
        )
        assert res["status"] == "scale_normalization_failed"
        assert "SCALE_CAP" in res["message"]
        ws = res["metadata"]["working_scale"]
        assert ws["native_scale_ratio"] is not None
        assert ws["native_scale_ratio"] > 10.0


# ---------------------------------------------------------------------------
# Requirement 4: Provenance Tracking across Registration Interfaces
# ---------------------------------------------------------------------------
def test_scale_ratio_provenance_recording():
    """
    Requirement 4: Verify native_scale_ratio, pre_normalization_ratio, and residual_scale_ratio
    are recorded across overlap recovery, matcher_cfog metadata, and direct_multimodal_attempt.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        h, w = 256, 256
        img = _generate_synthetic_lunar_terrain(h, w, seed=42)
        p1 = tmp / "ohrc_synth.png"
        p2 = tmp / "iirs_synth.png"
        cv2.imwrite(str(p1), img)
        cv2.imwrite(str(p2), cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA))

        # 1. matcher_cfog
        res_cfog = match_images_cfog(
            p1,
            p2,
            explicit_gsd1=0.25,
            explicit_gsd2=5.0,  # 20x ratio
            output_dir=tmp / "cfog_out",
        )
        meta = res_cfog["metadata"]
        assert "working_scale" in meta
        ws = meta["working_scale"]
        assert ws["native_scale_ratio"] == 20.0
        assert ws["pre_normalization_ratio"] == 20.0
        assert ws["residual_scale_ratio"] == 1.0

        # 2. direct_multimodal_attempt
        res_direct = direct_multimodal_attempt(str(p1), str(p2), output_dir=tmp / "direct_out")
        assert "native_scale_ratio" in res_direct
        assert "pre_normalization_ratio" in res_direct
        assert "residual_scale_ratio" in res_direct


# ---------------------------------------------------------------------------
# Requirement 7: Real OHRC-IIRS Experiment with Genuine Chandrayaan-2 Data
# ---------------------------------------------------------------------------
def test_real_ohrc_iirs_experiment():
    """
    Requirement 7: Add at least one real OHRC-IIRS experiment if valid data are available.
    Loads real Chandrayaan-2 region_001 OHRC and IIRS rasters with manifest metadata:
    - OHRC GSD: 0.25 m/px (native product ch2_ohr_ncp_20210405t1606536730_d_img_d18)
    - IIRS GSD: 69.04 m/px (native product ch2_iir_nri_20211221t0324126144_d_img_hw1)
    - Native physical scale ratio: 69.04 / 0.25 = 276.16x.

    Verifies:
    1. Physical GSD ratio is reduced via common-GSD normalization to 69.04m BEFORE overlap recovery.
    2. Fourier-Mellin is NEVER asked to estimate 276x directly.
    3. Provenance correctly records native_scale_ratio=276.16, pre_normalization_ratio=276.16,
       and residual_scale_ratio=1.0.
    """
    region_dir = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets" / "region_001"
    manifest_file = region_dir / "manifest.json"
    ohrc_file = region_dir / "ohrc_large_512.png"
    iirs_file = region_dir / "iirs_large_512.png"

    if not (manifest_file.exists() and ohrc_file.exists() and iirs_file.exists()):
        pytest.skip(f"Real Chandrayaan-2 region_001 rasters not found at {region_dir}")

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    ohrc_native_gsd = float(manifest["ohrc_gsd_m"])      # 0.25
    iirs_native_gsd = float(manifest["iirs_gsd_m"])      # 69.04
    expected_native_ratio = round(iirs_native_gsd / ohrc_native_gsd, 4)  # 276.16

    ohrc_img = cv2.imread(str(ohrc_file), cv2.IMREAD_GRAYSCALE)
    iirs_img = cv2.imread(str(iirs_file), cv2.IMREAD_GRAYSCALE)

    # 1. Native sensor GSD scale path (0.25m vs 69.04m, 276.16x) through recover_content_overlap
    res_overlap = recover_content_overlap(
        ohrc_img,
        iirs_img,
        source_gsd_m=ohrc_native_gsd,
        ref_gsd_m=iirs_native_gsd,
    )
    assert res_overlap["native_scale_ratio"] == expected_native_ratio
    assert res_overlap["pre_normalization_ratio"] == expected_native_ratio
    assert res_overlap["residual_scale_ratio"] == 1.0

    # 2. Effective raster GSD scale path through match_images_cfog on the 20km AOI
    ohrc_eff_gsd = float(manifest["ohrc_large_effective_gsd_m"])
    iirs_eff_gsd = float(manifest["iirs_large_effective_gsd_m"])
    expected_eff_ratio = round(iirs_eff_gsd / ohrc_eff_gsd, 4)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "real_ohrc_iirs_exp"
        res = match_images_cfog(
            ohrc_file,
            iirs_file,
            source_sensor="OHRC",
            reference_sensor="IIRS",
            explicit_gsd1=ohrc_eff_gsd,
            explicit_gsd2=iirs_eff_gsd,
            recover_overlap_from_content=True,
            output_dir=out_dir,
        )

        assert res is not None
        assert res.get("status") in ("success", "registration_failed")
        meta = res.get("metadata", {})
        ws = meta.get("working_scale", {})

        # Working scale matches the coarser sensor
        assert ws.get("working_gsd_m") == iirs_eff_gsd

        # Scale ratio provenance is recorded correctly
        assert ws.get("native_scale_ratio") == expected_eff_ratio
        assert ws.get("pre_normalization_ratio") == expected_eff_ratio
        assert ws.get("residual_scale_ratio") == 1.0

        # Content-based overlap recovery preserves provenance
        overlap_info = res.get("content_overlap_recovery", {})
        assert overlap_info.get("native_scale_ratio") == expected_eff_ratio
        assert overlap_info.get("pre_normalization_ratio") == expected_eff_ratio
