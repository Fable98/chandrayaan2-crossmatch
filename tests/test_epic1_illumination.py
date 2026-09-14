"""
tests/test_epic1_illumination.py — Epic 1: descriptor-level illumination invariance.

Failure definition: Phase Congruency structural edge maps from an image and
its heavily shadow-inverted version (160-degree sun-shift simulation) must be
>85% identical WITHOUT relying on Top-Hat pixel mutation.

Pipeline under test: adaptive_illumination_normalization(enable_tophat=False)
-> compute_phase_congruency (pure 2D Log-Gabor bank).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import (
    adaptive_illumination_normalization,
    compute_phase_congruency,
)

MIN_STRUCTURAL_AGREEMENT = 0.85


def _load_image() -> np.ndarray:
    sample = REPO_ROOT / "sample_data" / "ohrc_sample.png"
    if sample.exists():
        raw = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
        assert raw is not None, f"could not read {sample}"
        return (raw.astype(np.float32) / 255.0).astype(np.float32)
    # Deterministic synthetic fallback (craters + ridges).
    rng = np.random.RandomState(42)
    base = rng.uniform(50.0, 180.0, size=(256, 256)).astype(np.float32)
    base = cv2.GaussianBlur(base, (15, 15), 3.0)
    for _ in range(12):
        cx = rng.randint(20, 236)
        cy = rng.randint(20, 236)
        radius = rng.randint(8, 30)
        y, x = np.ogrid[:256, :256]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        base[dist <= radius] *= 0.65
        base[(dist > radius - 2) & (dist <= radius + 2)] *= 1.35
    base = np.clip(base, 10.0, 245.0)
    return ((base - base.min()) / (base.max() - base.min())).astype(np.float32)


def _structural_agreement(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """Binarized edge-map agreement: fraction of pixels with identical
    above/below-mean labels. Robust to global gain/bias."""
    b1 = (pc1 > float(np.mean(pc1))).astype(np.uint8)
    b2 = (pc2 > float(np.mean(pc2))).astype(np.uint8)
    return float(np.mean(b1 == b2))


def test_epic1_phase_congruency_illumination_invariant_without_tophat():
    img = _load_image()
    # Simulate ~160-degree sun shift: full shadow inversion + gamma crush.
    inverted = (1.0 - img).astype(np.float32)

    normed_ref, _ = adaptive_illumination_normalization(img, enable_tophat=False)
    normed_inv, _ = adaptive_illumination_normalization(inverted, enable_tophat=False)

    assert normed_ref.shape == img.shape
    assert normed_ref.dtype == np.float32
    assert np.all(np.isfinite(normed_ref))

    pc_ref = compute_phase_congruency(normed_ref)
    pc_inv = compute_phase_congruency(normed_inv)

    assert pc_ref.shape == img.shape
    assert float(pc_ref.min()) >= 0.0 and float(pc_ref.max()) <= 1.0

    # High-frequency detail must survive: PC maps must carry texture.
    assert float(np.std(pc_ref)) > 1e-4
    assert float(np.std(pc_inv)) > 1e-4

    agreement = _structural_agreement(pc_ref, pc_inv)
    print(f"Epic1 structural agreement (no TopHat) = {agreement:.4f}")
    assert agreement > MIN_STRUCTURAL_AGREEMENT, (
        f"Phase Congruency edge maps only {agreement:.4f} identical "
        f"under shadow inversion; required > {MIN_STRUCTURAL_AGREEMENT}"
    )


def test_epic1_log_gabor_descriptor_contract():
    """compute_phase_congruency must be a pure Log-Gabor bank (descriptor-level
    invariance), not a pixel-mutating filter."""
    import inspect

    src = inspect.getsource(compute_phase_congruency)
    # Log-Gabor essentials must be present.
    assert "np.log" in src or "log(" in src
    assert "fft2" in src and "ifft2" in src
    # Must not destroy high-frequency rims with morphological pixel mutation.
    assert "MORPH_TOPHAT" not in src
    assert "createCLAHE" not in src
    assert "equalizeHist" not in src
