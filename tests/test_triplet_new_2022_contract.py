"""
tests/test_triplet_new_2022_contract.py — Ceiling contract for the hardest pair.

The 162° sun-gap pair (triplet_new_2022) was historically a clean Gate-3
refusal and re-measured 2026-09-11 as a fragile LOW fit. Either outcome is
honest — FAILURE or LOW_CONFIDENCE both pass. What must NEVER happen is the
pipeline claiming MORE than fragile LOW here (ACCEPTED/HIGH tier, or a
sub-pixel verdict), in either direction of future flips.

Bars (deliberately coarse so OpenCV-build RANSAC wiggles don't trip them):
  * quality_tier never above LOW_CONFIDENCE,
  * sub_pixel_accurate never True (requires fit<1 AND held-out<1),
  * inlier_count below 10 (held-out needs >=8; near-threshold counts stay LOW).
"""

import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import match_images_cfog


def test_triplet_new_2022_never_claims_more_than_fragile_low():
    ohrc = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets" / "triplet_new_2022" / "ohrc_512.png"
    tmc = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets" / "triplet_new_2022" / "tmc_512.png"
    if not ohrc.is_file() or not tmc.is_file():
        pytest.skip("triplet_new_2022 tiles unavailable")
    with tempfile.TemporaryDirectory() as out:
        res = match_images_cfog(
            str(ohrc), str(tmc), output_dir=out,
            explicit_gsd1=0.25, explicit_gsd2=5.0,
        )
    metrics = res.get("metrics") or {}
    tier = metrics.get("quality_tier", "FAILED")
    assert tier in ("FAILED", "LOW_CONFIDENCE"), (
        f"hardest pair must never exceed LOW_CONFIDENCE, got {tier}"
    )
    assert metrics.get("sub_pixel_accurate") is not True, (
        "162° pair must never earn a sub-pixel verdict"
    )
    assert int(metrics.get("inlier_count", 0)) < 10, (
        "hardest pair must stay a small-N fragile fit"
    )
