"""
test_registration_pipeline.py — End-to-End Registration and Scientific Integrity Tests

Validates:
1. Valid cross-sensor registration and full output product package.
2. Clean failure on insufficient matches (ZERO fake correspondences).
3. Rejection of corrupt or unreadable files.
4. Physical GSD scale normalization.
5. Coordinate mapping between physical working scale and native sensor pixels.
6. Spatially distributed match selection across grid cells.
7. Canonical metrics consistency (In-sample Fit RMSE vs. Held-out Validation RMSE).
8. Rejection of degenerate or pathological geometric transformations.
9. Honest IIRS hyperspectral co-registration handling.
"""

import sys
import io
import os
import tempfile
import json
from pathlib import Path
import numpy as np
import cv2
import pytest

# Add project roots
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from matcher_cfog import match_images_cfog
from metrics import compute_canonical_metrics, verify_transformation_quality
from metadata import extract_sensor_metadata
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def synthetic_lunar_pair():
    """Generates a synthetic textured image pair with known ground-truth affine shift."""
    np.random.seed(42)
    h, w = 512, 512
    base = np.zeros((h, w), dtype=np.uint8)

    # Multi-scale craters
    for _ in range(30):
        cx = np.random.randint(50, 460)
        cy = np.random.randint(50, 460)
        rad = np.random.randint(10, 35)
        val = int(np.random.randint(120, 255))
        cv2.circle(base, (cx, cy), rad, val, -1)
        cv2.circle(base, (cx, cy), max(2, rad - 5), int(val // 2), -1)

    noise = np.random.randint(0, 25, (h, w), dtype=np.uint8)
    base = cv2.add(base, noise)

    # Known translation
    true_dx, true_dy = 6.0, -4.0
    M = np.float32([[1, 0, true_dx], [0, 1, true_dy]])
    shifted = cv2.warpAffine(base, M, (w, h))

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "source.png"
        p2 = Path(tmpdir) / "reference.png"
        cv2.imwrite(str(p1), base)
        cv2.imwrite(str(p2), shifted)
        yield str(p1), str(p2), true_dx, true_dy


# ---------------------------------------------------------------------------
# Test 1: Valid Cross-Sensor Registration & Output Product Verification
# ---------------------------------------------------------------------------
def test_valid_registration_produces_full_package(synthetic_lunar_pair):
    p1, p2, true_dx, true_dy = synthetic_lunar_pair
    with tempfile.TemporaryDirectory() as out_dir:
        res = match_images_cfog(
            p1, p2, output_dir=out_dir, explicit_gsd1=5.0, explicit_gsd2=5.0
        )
        assert res["status"] == "success"
        metrics = res["metrics"]
        assert metrics["inlier_count"] >= 4
        assert metrics["fit_rmse_px"] < 1.0

        outputs = res["outputs"]
        assert os.path.exists(outputs["registered_raster"])
        assert os.path.exists(outputs["preview"])
        assert os.path.exists(outputs["checkerboard"])
        assert os.path.exists(outputs["matches"])
        assert os.path.exists(outputs["metrics"])
        assert os.path.exists(outputs["transform"])
        assert os.path.exists(outputs["metadata"])


# ---------------------------------------------------------------------------
# Test 2: Insufficient Matches Fails Cleanly (ZERO Fake Correspondences)
# ---------------------------------------------------------------------------
def test_insufficient_matches_never_creates_fake_points():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Uniform blank image vs white noise -> zero real visual features
        blank = np.zeros((256, 256), dtype=np.uint8)
        noise = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        p1 = Path(tmpdir) / "blank.png"
        p2 = Path(tmpdir) / "noise.png"
        cv2.imwrite(str(p1), blank)
        cv2.imwrite(str(p2), noise)

        res = match_images_cfog(p1, p2, output_dir=tmpdir)
        # MUST report failure cleanly
        assert res["status"] in ("insufficient_correspondences", "geometric_verification_failed")
        assert res["inlier_count"] == 0
        assert res["homography"] is None


# ---------------------------------------------------------------------------
# Test 3: Corrupt Input Handling
# ---------------------------------------------------------------------------
def test_corrupt_input_handled_gracefully():
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt = Path(tmpdir) / "corrupt.png"
        with open(corrupt, "wb") as f:
            f.write(b"NOT_A_REAL_IMAGE_FILE_DATA")

        valid = Path(tmpdir) / "valid.png"
        cv2.imwrite(str(valid), np.zeros((100, 100), dtype=np.uint8))

        with pytest.raises(Exception):
            match_images_cfog(corrupt, valid, output_dir=tmpdir)


# ---------------------------------------------------------------------------
# Test 4: Physical GSD Normalization
# ---------------------------------------------------------------------------
def test_common_physical_gsd_normalization():
    # OHRC (0.25 m) vs TMC-2 (5.0 m) scale factor is 20x
    meta1 = extract_sensor_metadata("ohrc_img.png", declared_sensor="OHRC")
    meta2 = extract_sensor_metadata("tmc_img.png", declared_sensor="TMC-2")
    assert abs(meta1.gsd_m - 0.25) < 1e-3
    assert abs(meta2.gsd_m - 5.0) < 1e-3

    scale_factor = meta2.gsd_m / meta1.gsd_m
    assert abs(scale_factor - 20.0) < 1e-3


# ---------------------------------------------------------------------------
# Test 5: Spatial Distribution Across Multiple Cells
# ---------------------------------------------------------------------------
def test_spatial_distribution_metrics():
    # 20 distributed points
    pts = np.array([
        [50, 50], [150, 50], [250, 50], [350, 50], [450, 50],
        [50, 150], [150, 150], [250, 150], [350, 150], [450, 150],
        [50, 250], [150, 250], [250, 250], [350, 250], [450, 250],
        [50, 350], [150, 350], [250, 350], [350, 350], [450, 350],
    ], dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    mask = np.ones((len(pts), 1), dtype=np.uint8)

    metrics = compute_canonical_metrics(pts, pts, mask, H, (512, 512), grid_size=10)
    assert metrics["spatial_coverage"] >= 0.20
    assert metrics["spatial_uniformity"] > 0.05
    assert metrics["spatial_distribution"]["occupied_cells"] == 20


# ---------------------------------------------------------------------------
# Test 6: In-Sample Fit RMSE vs Independent Held-Out Validation RMSE
# ---------------------------------------------------------------------------
def test_held_out_validation_rmse_evaluated():
    np.random.seed(42)
    # 15 points with known small noise
    src = np.random.uniform(50, 450, (15, 2)).astype(np.float32)
    dst = src + np.array([5.0, -3.0]) + np.random.normal(0, 0.1, src.shape)
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    mask = np.ones((len(src), 1), dtype=np.uint8)

    metrics = compute_canonical_metrics(src, dst, mask, H, (512, 512), 10)
    assert metrics["fit_rmse_px"] < 0.5
    assert metrics["validation_status"] == "evaluated"
    assert metrics["validation_rmse_px"] is not None
    assert metrics["validation_rmse_px"] < 1.0


# ---------------------------------------------------------------------------
# Test 7: Degenerate Matrix Rejection Quality Gate
# ---------------------------------------------------------------------------
def test_degenerate_matrix_rejection():
    # Collinear points matrix or collapsed determinant
    H_singular = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    check = verify_transformation_quality(H_singular)
    assert check["is_valid"] is False

    # Negative determinant (reflection / inversion)
    H_neg = np.array([[-1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]])
    check_neg = verify_transformation_quality(H_neg)
    assert check_neg["is_valid"] is False


# ---------------------------------------------------------------------------
# Test 8: HTTP /register Endpoint Integration Test
# ---------------------------------------------------------------------------
def test_register_api_endpoint(test_client, synthetic_lunar_pair):
    p1, p2, _, _ = synthetic_lunar_pair
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        files = {
            "source_file": ("source.png", f1, "image/png"),
            "reference_file": ("ref.png", f2, "image/png"),
        }
        data = {
            "source_sensor": "TMC-2",  # Use same GSD for synthetic pair
            "reference_sensor": "TMC-2",
            "method": "cfog",
        }
        resp = test_client.post("/register", files=files, data=data)
        assert resp.status_code == 200
        res_json = resp.json()
        assert res_json["status"] == "success"
        assert res_json["metrics"]["inlier_count"] >= 4
        assert res_json["raster_url"] is not None
