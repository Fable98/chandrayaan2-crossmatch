"""
tests/test_iirs.py — Validation of IIRS Hyperspectral Feature Engineering and Residual Quantification.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from spectral import enhance_iirs_structural_features, quantify_iirs_residuals
from matcher_cfog import compute_phase_congruency


@pytest.fixture
def dummy_iirs_hypercube():
    """
    Creates a synthetic 10-band IIRS hypercube (256x256x10):
    - Models lunar regolith with distinct mineral absorption features across bands
    - Stamps distinct impact crater structures with rim morphology
    """
    np.random.seed(123)
    h, w, num_bands = 256, 256, 10
    cube = np.zeros((h, w, num_bands), dtype=np.float32)

    # Base spatial albedo pattern (impact crater rim and floor)
    base_morphology = np.zeros((h, w), dtype=np.float32)
    # Primary crater at center
    cv2.circle(base_morphology, (128, 128), 50, 0.8, 4)  # bright rim
    cv2.circle(base_morphology, (128, 128), 45, 0.2, -1) # dark shadowed floor
    # Satellite craters
    cv2.circle(base_morphology, (60, 60), 20, 0.7, 3)
    cv2.circle(base_morphology, (190, 80), 25, 0.75, 3)
    cv2.circle(base_morphology, (80, 190), 18, 0.65, 2)

    # Synthetic wavelengths from 700nm to 1600nm across 10 bands
    wavelengths = np.linspace(700.0, 1600.0, num_bands)

    # Mineral absorption profile (pyroxene absorption centered near band 4 ~1000nm)
    for b in range(num_bands):
        wl = wavelengths[b]
        # Absorption band dip
        absorption = 1.0 - 0.35 * np.exp(-((wl - 1000.0) ** 2) / (2 * 120.0**2))
        band_signal = base_morphology * absorption + 0.3
        # Add sensor noise
        noise = np.random.normal(0, 0.02, (h, w)).astype(np.float32)
        cube[:, :, b] = np.clip(band_signal + noise, 0.0, 1.0)

    return cube, wavelengths


def test_iirs_enhancement_produces_high_contrast_structural_map(dummy_iirs_hypercube):
    """
    Validates that enhance_iirs_structural_features transforms a 10-band hypercube
    into a single, high-contrast, normalized 2D structural map.
    """
    cube, wavelengths = dummy_iirs_hypercube
    structural = enhance_iirs_structural_features(cube, wavelengths=wavelengths)

    assert structural.ndim == 2, "Output must be a 2D structural map"
    assert structural.shape == (256, 256), "Output spatial dimensions must match hypercube"
    assert structural.dtype == np.float32, "Output must be float32"
    assert 0.0 <= structural.min() and structural.max() <= 1.0, "Output must be normalized [0, 1]"

    # High dynamic range & contrast check
    dynamic_range = float(structural.max() - structural.min())
    assert dynamic_range > 0.6, f"Expected high contrast, got dynamic range {dynamic_range}"
    assert float(np.std(structural)) > 0.1, "Standard deviation must indicate substantial structural texture"


def test_subsequent_feature_extractor_pulls_keypoints(dummy_iirs_hypercube):
    """
    CORE VALIDATION: Proves that the structural output from enhance_iirs_structural_features
    allows subsequent feature extractors (SIFT, ORB, Phase Congruency) to successfully pull keypoints.
    """
    cube, wavelengths = dummy_iirs_hypercube
    structural = enhance_iirs_structural_features(cube, wavelengths=wavelengths)

    # 1. Test standard SIFT feature extractor on the enhanced map
    sift = cv2.SIFT_create()
    u8 = (structural * 255.0).astype(np.uint8)
    keypoints_sift, descs_sift = sift.detectAndCompute(u8, None)

    print(f"\n[IIRS Feature Test] Detected {len(keypoints_sift)} SIFT keypoints on PC1 structural map")
    assert len(keypoints_sift) > 15, (
        f"Expected at least 15 SIFT keypoints on crater rims, found {len(keypoints_sift)}"
    )

    # 2. Test Phase Congruency structural feature extractor
    pc = compute_phase_congruency(structural)
    assert pc.shape == (256, 256)
    assert float(np.max(pc)) > 0.1, "Phase congruency must detect sharp structural edges on crater rims"

    # Keypoints from Phase Congruency peaks
    orb = cv2.ORB_create(nfeatures=200)
    pc_u8 = (np.clip(pc, 0.0, 1.0) * 255.0).astype(np.uint8)
    kp_orb, _ = orb.detectAndCompute(pc_u8, None)
    assert len(kp_orb) > 10, f"Expected keypoints from Phase Congruency edges, found {len(kp_orb)}"


def test_quantify_iirs_residuals(dummy_iirs_hypercube):
    """
    Validates that quantify_iirs_residuals separates spatial misalignment from spectral variance.
    """
    cube, _ = dummy_iirs_hypercube
    mock_errors = np.array([0.45, 0.62, 0.38, 0.55], dtype=np.float64)

    residuals = quantify_iirs_residuals(cube, mock_errors)

    assert "spatial_misalignment_px" in residuals
    assert "spectral_variance" in residuals
    assert "spectral_angle_mapper_deg" in residuals

    assert residuals["spatial_misalignment_px"] > 0.0
    assert residuals["spectral_variance"] > 0.0
    assert residuals["spectral_angle_mapper_deg"] >= 0.0
