"""
tests/test_phase1_illumination_invariance.py — "Shadow Flip" invariance test.

Validates that Phase 1 (Wallis / Top-Hat illumination normalization) followed
by Phase 2 (Phase Congruency) yields keypoints that survive extreme
illumination changes simulating sun-angle variation:

  1. Massive diagonal shadow gradient (additive tilt spanning 60% of the
     dynamic range, ~6x the texture std — simulates deep crater shadows).
  2. Full intensity inversion (1 - img, simulating a ~180 deg sun-azimuth
     flip / diametric shadow reversal).
  3. Non-linear gamma distortion (power-law 1.5, compressing shadow SNR).

Assertion: spatial repeatability of the top-N keypoints between the original
and EACH perturbed image must remain strictly above 85%.
"""

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

TOP_N = 200
TOL_PX = 3.0
MIN_REPEATABILITY = 0.85


def _generate_synthetic_lunar(shape=(256, 256), seed=42) -> np.ndarray:
    """Deterministic synthetic lunar-like texture with craters + ridges."""
    rng = np.random.RandomState(seed)
    base = rng.uniform(50.0, 180.0, size=shape).astype(np.float32)
    base = cv2.GaussianBlur(base, (15, 15), 3.0)
    for _ in range(12):
        cx = rng.randint(20, shape[1] - 20)
        cy = rng.randint(20, shape[0] - 20)
        radius = rng.randint(8, 30)
        y, x = np.ogrid[: shape[0], : shape[1]]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        crater = dist <= radius
        rim = (dist > radius - 2) & (dist <= radius + 2)
        base[crater] *= 0.65
        base[rim] = np.clip(base[rim] * 1.35, 0, 255)
    for _ in range(5):  # linear ridges for strong structural edges
        x0, y0 = rng.randint(0, shape[1]), rng.randint(0, shape[0])
        x1, y1 = rng.randint(0, shape[1]), rng.randint(0, shape[0])
        cv2.line(base, (x0, y0), (x1, y1), 200, 1)
    base = np.clip(base, 10.0, 245.0)
    base = (base - base.min()) / (base.max() - base.min())
    return base.astype(np.float32)


def _phase1_phase2_keypoints(img: np.ndarray, top_n: int = TOP_N) -> np.ndarray:
    """Phase 1 -> Phase 2 -> top-N keypoints using the pipeline detector."""
    normed, mask = adaptive_illumination_normalization(img)
    assert normed.shape == img.shape
    assert normed.dtype == np.float32
    assert float(normed.min()) >= 0.0 and float(normed.max()) <= 1.0
    assert np.all(np.isfinite(normed))
    pc = compute_phase_congruency(normed)
    assert pc.shape == img.shape
    kps = detect_salient_keypoints(pc, max_corners=500, quality_level=0.01)
    assert len(kps) >= top_n, f"too few keypoints: {len(kps)}"
    kps = sorted(kps, key=lambda t: -t[2])[:top_n]
    return np.array([[x, y] for x, y, _ in kps], dtype=np.float32)


def _repeatability(ref: np.ndarray, test: np.ndarray, tol: float = TOL_PX) -> float:
    """Greedy one-to-one spatial repeatability within `tol` px."""
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


def test_phase1_output_contract():
    img = _generate_synthetic_lunar()
    normed, mask = adaptive_illumination_normalization(img)
    assert normed.shape == img.shape
    assert mask.shape == img.shape
    # Shadows must be boosted, not zeroed: mask is all-ones by design.
    assert float(mask.mean()) == pytest.approx(1.0)
    assert np.all(np.isfinite(normed))


def test_shadow_flip_illumination_invariance():
    img = _generate_synthetic_lunar()
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    diag = (xx / w + yy / h) / 2.0  # 0..1 diagonal ramp

    perturbed = {
        # Massive diagonal shadow gradient: +/-0.30 tilt (0.60 peak-to-peak,
        # ~6x the texture std of 0.10) while preserving mean contrast.
        "shadow_gradient": np.clip(img + (diag - 0.5) * 0.6, 0, 1).astype(np.float32),
        # 180-degree sun-azimuth flip: shadows become highlights and vice versa.
        "inverted": (1.0 - img).astype(np.float32),
        # Non-linear photometric distortion crushing shadow SNR.
        "gamma_1.5": np.clip(img**1.5, 0, 1).astype(np.float32),
    }

    ref_kps = _phase1_phase2_keypoints(img)
    assert len(ref_kps) == TOP_N

    for name, pert in perturbed.items():
        kps = _phase1_phase2_keypoints(pert)
        score = _repeatability(ref_kps, kps, tol=TOL_PX)
        print(f"perturbation={name} repeatability={score:.4f} (tol={TOL_PX}px, N={TOP_N})")
        assert score > MIN_REPEATABILITY, (
            f"Phase1+Phase2 keypoint repeatability under '{name}' is {score:.4f}, "
            f"required > {MIN_REPEATABILITY}"
        )
