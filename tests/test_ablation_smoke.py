"""
tests/test_ablation_smoke.py — Smoke test for the 4-way ablation benchmark and LoFTR import guards
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from evaluation.run_ablation import run_ablation_study
from evaluation.baselines.loftr_matcher import LOFTR_AVAILABLE, match_loftr


def test_ablation_pipeline_smoke_test(tmp_path):
    """
    Validates that the 4-way ablation pipeline executes end-to-end without crashing,
    even if PyTorch/LoFTR is not installed (testing graceful fallback guards).
    """
    # 1. Create two 128x128 synthetic test images with clear features
    np.random.seed(42)
    img1 = np.random.randint(50, 200, (128, 128), dtype=np.uint8)
    cv2.circle(img1, (64, 64), 20, 255, -1)
    cv2.circle(img1, (32, 90), 12, 0, -1)

    M = np.float32([[1, 0, 3], [0, 1, -2]])
    img2 = cv2.warpAffine(img1, M, (128, 128))

    src_path = tmp_path / "smoke_src.png"
    ref_path = tmp_path / "smoke_ref.png"
    cv2.imwrite(str(src_path), img1)
    cv2.imwrite(str(ref_path), img2)

    # 2. Test LoFTR matcher directly — must never crash whether PyTorch is installed or not
    loftr_res = match_loftr(src_path, ref_path)
    assert isinstance(loftr_res, dict)
    assert "method" in loftr_res
    assert "status" in loftr_res
    if not LOFTR_AVAILABLE:
        assert "NCC Fallback" in loftr_res["method"]

    # 3. Test full run_ablation_study orchestration
    ablation_out = tmp_path / "ablation_out"
    study_results = run_ablation_study(
        source_img=src_path,
        reference_img=ref_path,
        output_dir=ablation_out,
        gsd_m=5.0,
    )

    assert isinstance(study_results, dict)
    assert "results" in study_results
    assert len(study_results["results"]) == 4
    assert (ablation_out / "ablation_results.md").exists()
    assert (ablation_out / "ablation_results.json").exists()
