"""
tests/test_phase1_physics_enforcement.py — Anti-black-box enforcement for Phase 1.

Two hard gates on ``adaptive_illumination_normalization`` (ML_model/matcher_cfog.py):

1. AST gate: the function body must not contain histogram-equalization
   black boxes (``createCLAHE`` / ``equalizeHist``), must not contain a
   shadow-zeroing ``valid_mask``, and must not use a plain ``GaussianBlur``
   as its primary smoother. The mandated physics pipeline (bilateral
   smoothing -> homomorphic log decomposition -> morphological gradient)
   must be structurally present.
2. 180-degree shadow-flip gate: a real lunar sample passed through
   Phase 1 -> Phase 2 (Phase Congruency) must yield keypoints whose spatial
   repeatability against the fully inverted image stays strictly above 85%.
"""

import ast
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import (
    adaptive_illumination_normalization,
    compute_phase_congruency,
)
from spatial_suppression import detect_salient_keypoints

MATCHER_PATH = REPO_ROOT / "ML_model" / "matcher_cfog.py"
TARGET_FUNC = "adaptive_illumination_normalization"

FORBIDDEN_TOKENS = ("createCLAHE", "equalizeHist", "valid_mask", "GaussianBlur")
REQUIRED_TOKENS = ("bilateralFilter", "log1p", "gaussian_filter", "expm1", "MORPH_GRADIENT")

TOP_N = 200
TOL_PX = 3.0
MIN_REPEATABILITY = 0.85


def _function_source() -> str:
    tree = ast.parse(MATCHER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == TARGET_FUNC:
            segment = ast.get_source_segment(MATCHER_PATH.read_text(encoding="utf-8"), node)
            assert segment is not None, f"could not extract source of {TARGET_FUNC}"
            return segment
    raise AssertionError(f"function {TARGET_FUNC} not found in {MATCHER_PATH}")


def test_phase1_has_no_black_box_constructs():
    """AST gate: forbidden histogram/shadow-zeroing constructs must be gone."""
    source = _function_source()
    for token in FORBIDDEN_TOKENS:
        assert token not in source, (
            f"HARD FAIL: forbidden construct {token!r} still present in "
            f"{TARGET_FUNC}; physical signal decomposition is required."
        )
    for token in REQUIRED_TOKENS:
        assert token in source, (
            f"HARD FAIL: required physics-pipeline construct {token!r} missing "
            f"from {TARGET_FUNC}."
        )


def _load_sample() -> np.ndarray:
    sample = REPO_ROOT / "sample_data" / "ohrc_sample.png"
    assert sample.exists(), f"lunar sample missing: {sample}"
    raw = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    assert raw is not None, f"could not read {sample}"
    return (raw.astype(np.float32) / 255.0).astype(np.float32)


def _phase1_phase2_keypoints(img: np.ndarray, top_n: int = TOP_N) -> np.ndarray:
    normed, companion = adaptive_illumination_normalization(img)
    assert normed.shape == img.shape
    assert normed.dtype == np.float32
    assert float(normed.min()) >= 0.0 and float(normed.max()) <= 1.0
    assert np.all(np.isfinite(normed))
    # Shadows must never be zeroed out: companion is all ones.
    assert companion.shape == img.shape
    assert float(companion.min()) == pytest.approx(1.0)
    assert float(companion.max()) == pytest.approx(1.0)
    pc = compute_phase_congruency(normed)
    kps = detect_salient_keypoints(pc, max_corners=500, quality_level=0.01)
    assert len(kps) >= top_n, f"too few keypoints: {len(kps)}"
    kps = sorted(kps, key=lambda t: -t[2])[:top_n]
    return np.array([[x, y] for x, y, _ in kps], dtype=np.float32)


def _repeatability(ref: np.ndarray, test: np.ndarray, tol: float = TOL_PX) -> float:
    if len(ref) == 0 or len(test) == 0:
        return 0.0
    used = np.zeros(len(test), dtype=bool)
    hits = 0
    for p in ref:
        d = np.sqrt(((test - p) ** 2).sum(axis=1))
        d[used] = np.inf
        best = int(np.argmin(d))
        if d[best] <= tol:
            hits += 1
            used[best] = True
    return hits / len(ref)


def test_180_degree_shadow_flip_invariance():
    """Real lunar sample vs its full inversion must repeat keypoints > 85%."""
    img = _load_sample()
    ref_kps = _phase1_phase2_keypoints(img)
    flip_kps = _phase1_phase2_keypoints((1.0 - img).astype(np.float32))
    score = _repeatability(ref_kps, flip_kps, tol=TOL_PX)
    print(f"180-degree flip repeatability={score:.4f} (tol={TOL_PX}px, N={TOP_N})")
    assert score > MIN_REPEATABILITY, (
        f"shadow-flip repeatability {score:.4f} is not strictly above "
        f"{MIN_REPEATABILITY}"
    )
