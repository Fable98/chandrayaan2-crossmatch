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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from matcher_cfog import match_images_cfog, verify_spatial_quality_gate, load_as_float_and_color
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

        res = match_images_cfog(
            p1, p2, output_dir=tmpdir, explicit_gsd1=5.0, explicit_gsd2=5.0
        )
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
# Test 6: In-Sample Fit RMSE vs Held-Out Inlier Correspondence Validation RMSE
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


# ---------------------------------------------------------------------------
# Test 9: Known-Ground-Truth Synthetic 20x Physical-Scale Integration Test (OHRC vs TMC-2)
# ---------------------------------------------------------------------------
def test_real_20x_scale_disparity_registration():
    """
    Known-ground-truth synthetic 20x physical-scale integration test:
    - OHRC at 0.25 m/px (1024x1024 pixels, covering 256m x 256m physical footprint)
    - TMC-2 at 5.0 m/px (51x51 pixels, covering ~255m x 255m physical footprint)
    - Known physical ground displacement: +10.0 m in X, -5.0 m in Y
      In OHRC native: +40.0 px X, -20.0 px Y (+10m / 0.25m, -5m / 0.25m)
      In TMC-2 native: +2.0 px X, -1.0 px Y (+10m / 5.0m, -5m / 5.0m)
    - Verifies recovery of transformation across the 20x scale bridge.

    Note: This test validates the scale-normalization and coordinate-mapping pipeline
    under controlled known-ground-truth synthetic conditions. It does not constitute
    independent validation on real Chandrayaan-2 imagery.
    """
    np.random.seed(42)
    h_ohrc, w_ohrc = 1024, 1024
    master_terrain = np.zeros((h_ohrc, w_ohrc), dtype=np.float32)

    # Add distinct crater features distributed across the physical terrain
    for _ in range(50):
        cx = np.random.randint(100, 924)
        cy = np.random.randint(100, 924)
        rad = np.random.randint(20, 60)
        val = np.random.uniform(100, 255)
        cv2.circle(master_terrain, (cx, cy), rad, val, -1)
        cv2.circle(master_terrain, (cx, cy), max(3, rad - 8), val * 0.4, -1)

    master_terrain = cv2.GaussianBlur(master_terrain, (0, 0), 2.0)
    noise = np.random.normal(0, 5, (h_ohrc, w_ohrc)).astype(np.float32)
    ohrc_raw = np.clip(master_terrain + noise, 0, 255).astype(np.uint8)

    # Known physical displacement: +10 meters X, -5 meters Y
    # In OHRC native space (0.25 m/px): +40 px X, -20 px Y
    shift_x_ohrc = 40.0
    shift_y_ohrc = -20.0
    M_ohrc = np.float32([[1, 0, shift_x_ohrc], [0, 1, shift_y_ohrc]])
    ohrc_shifted = cv2.warpAffine(ohrc_raw, M_ohrc, (w_ohrc, h_ohrc))

    # Downsample to TMC-2 resolution: 1024 * 0.25 / 5.0 = 51.2 -> 51x51 px @ 5.0 m/px
    w_tmc, h_tmc = 51, 51
    tmc_raw = cv2.resize(ohrc_shifted, (w_tmc, h_tmc), interpolation=cv2.INTER_AREA)

    with tempfile.TemporaryDirectory() as tmpdir:
        p_ohrc = Path(tmpdir) / "test_ohrc.png"
        p_tmc = Path(tmpdir) / "test_tmc.png"
        p_out = Path(tmpdir) / "output_20x"

        cv2.imwrite(str(p_ohrc), ohrc_raw)
        cv2.imwrite(str(p_tmc), tmc_raw)

        # Execute registration with explicit physical scale
        res = match_images_cfog(
            p_ohrc,
            p_tmc,
            output_dir=p_out,
            source_sensor="OHRC",
            reference_sensor="TMC-2",
            explicit_gsd1=0.25,
            explicit_gsd2=5.0,
        )

        # MUST succeed — no fallbacks allowed
        assert res["status"] == "success", f"20x registration failed: {res.get('message')}"
        assert res["homography"] is not None

        # Verify scale factor recovery across the 20x bridge
        # Homography maps OHRC (0.25m) to TMC (5.0m), so scale is ~0.25 / 5.0 = 0.05
        H = np.array(res["homography"])
        scale_x = float(np.sqrt(H[0, 0]**2 + H[1, 0]**2))
        scale_y = float(np.sqrt(H[0, 1]**2 + H[1, 1]**2))
        expected_scale = 0.25 / 5.0  # 0.05
        assert abs(scale_x - expected_scale) < 0.005, f"Scale X deviation: {scale_x} vs {expected_scale}"
        assert abs(scale_y - expected_scale) < 0.005, f"Scale Y deviation: {scale_y} vs {expected_scale}"

        # Verify translation in TMC pixel space: expected +2.0 px X (+10m / 5m), -1.0 px Y (-5m / 5m)
        est_tx_tmc = float(H[0, 2])
        est_ty_tmc = float(H[1, 2])
        assert abs(est_tx_tmc - 2.0) < 0.5, f"Translation X error: {est_tx_tmc} vs 2.0 px"
        assert abs(est_ty_tmc - (-1.0)) < 0.5, f"Translation Y error: {est_ty_tmc} vs -1.0 px"

        # Verify full output package
        outputs = res["outputs"]
        assert os.path.exists(outputs["registered_raster"])
        assert os.path.exists(outputs["preview"])
        assert os.path.exists(outputs["checkerboard"])
        assert os.path.exists(outputs["matches"])
        assert os.path.exists(outputs["metrics"])
        assert os.path.exists(outputs["transform"])
        assert os.path.exists(outputs["metadata"])

        # Verify inlier count
        assert res["metrics"]["inlier_count"] >= 4


# ---------------------------------------------------------------------------
# Test 10: Metadata Safety Rejects Unknown Sensor Without Explicit GSD
# ---------------------------------------------------------------------------
def test_metadata_safety_rejects_unknown_without_gsd():
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_path = Path(tmpdir) / "arbitrary_satellite_img.png"
        cv2.imwrite(str(dummy_path), np.zeros((100, 100), dtype=np.uint8))

        # Must fail with ValueError explaining that GSD could not be determined
        with pytest.raises(ValueError, match="Physical ground sampling distance.*could not be determined"):
            extract_sensor_metadata(dummy_path)


# ---------------------------------------------------------------------------
# Test 11: Spatial Quality Gate Directly Exercises Production Rejection Logic
# ---------------------------------------------------------------------------
def test_spatial_quality_gate_rejects_clustered_matches():
    """
    Directly exercises the production verify_spatial_quality_gate function:
    1. Rejects inliers spanning fewer than 3 distinct spatial cells.
    2. Rejects inliers where a single cell contains > 60% of all points.
    3. Accepts inliers with sufficient spatial dispersion across the grid.
    """
    # Case 1: 6 inliers spanning only 2 distinct cells -> REJECT (< 3 cells)
    few_cells = [(0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1)]
    valid_few, reason_few, details_few = verify_spatial_quality_gate(few_cells)
    assert valid_few is False
    assert details_few["distinct_cells"] == 2
    assert "distinct cells" in reason_few

    # Case 2: 8 inliers across 4 cells, but 5 in cell (0, 0) (62.5% > 60%) -> REJECT
    concentrated = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (2, 2), (3, 3)]
    valid_conc, reason_conc, details_conc = verify_spatial_quality_gate(concentrated)
    assert valid_conc is False
    assert details_conc["concentration_ratio"] == 0.625
    assert "single cell concentration" in reason_conc

    # Case 3: 6 inliers uniformly distributed across 6 distinct cells -> ACCEPT
    distributed = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    valid_dist, reason_dist, details_dist = verify_spatial_quality_gate(distributed)
    assert valid_dist is True
    assert details_dist["distinct_cells"] == 6
    assert details_dist["concentration_ratio"] <= 0.60
    assert "passed" in reason_dist.lower()


# ---------------------------------------------------------------------------
# Test 12: GeoTIFF & Product Package Authenticity & Validity
# ---------------------------------------------------------------------------
def test_geotiff_output_validity(synthetic_lunar_pair):
    """
    Verifies that registered_source.tif is a genuine TIFF/GeoTIFF raster
    (not a renamed PNG/JPEG), matches reference spatial dimensions,
    validates driver == 'GTiff' via rasterio, and confirms all JSON sidecars
    are valid parseable JSON.
    """
    import rasterio
    from PIL import Image
    p1, p2, _, _ = synthetic_lunar_pair
    with tempfile.TemporaryDirectory() as out_dir:
        res = match_images_cfog(
            p1, p2, output_dir=out_dir, explicit_gsd1=5.0, explicit_gsd2=5.0
        )
        assert res["status"] == "success"
        tif_file = Path(res["outputs"]["registered_raster"])
        assert tif_file.exists()

        # 1. Check genuine TIFF format via PIL
        with Image.open(tif_file) as im:
            assert im.format == "TIFF", f"File is not genuine TIFF: {im.format}"
            assert im.size == (512, 512), f"Dimensions mismatch: {im.size} vs (512, 512)"

        # 2. Check GeoTIFF metadata directly via rasterio (mandatory project dependency)
        with rasterio.open(str(tif_file)) as src:
            assert src.driver == "GTiff", f"Rasterio driver is not GTiff: {src.driver}"
            assert src.width == 512
            assert src.height == 512
            assert src.count in (1, 3)
            assert src.dtypes[0] == "uint8"

        # 3. Check JSON sidecars are valid
        for sidecar_key in ("matches", "metrics", "transform", "metadata"):
            sidecar_path = Path(res["outputs"][sidecar_key])
            assert sidecar_path.exists(), f"Sidecar {sidecar_key} missing"
            with open(sidecar_path, "r") as f:
                parsed = json.load(f)
                assert isinstance(parsed, (dict, list))


# ---------------------------------------------------------------------------
# Test 13: Known-Ground-Truth Synthetic IIRS Co-Registration Integration Test
# ---------------------------------------------------------------------------
def test_iirs_hyperspectral_co_registration_integration():
    """
    Known-ground-truth synthetic IIRS co-registration integration test.

    Validates:
    1. Multi-band hyperspectral raster ingestion (> 3 bands, e.g. 8 spectral channels).
    2. Verification that production loader recognizes multi-band cube (count > 3).
    3. Physical-GSD-aware normalization between coarse IIRS (75.0 m/px) and optical reference (37.5 m/px).
    4. End-to-end co-registration via the actual production pipeline (match_images_cfog).
    5. Zero fake correspondences (genuine inliers passing RANSAC and spatial quality gates).
    6. Recovery of known geometric scale factor (75.0 / 37.5 = 2.0).
    7. Full production output package generation with mandatory Rasterio GeoTIFF validation.

    Note: This test validates hyperspectral IIRS ingestion, physical-GSD normalization,
    and end-to-end co-registration under controlled known-ground-truth synthetic
    conditions. It does not constitute independent validation on real Chandrayaan-2
    IIRS imagery and does not establish sub-meter IIRS tie-point accuracy.
    """
    import rasterio
    from PIL import Image

    np.random.seed(101)
    # Reference image: 256x256 at 37.5 m/px -> 9600m x 9600m physical footprint
    h_ref, w_ref = 256, 256
    ref_gsd = 37.5
    iirs_gsd = 75.0

    master_terrain = np.zeros((h_ref, w_ref), dtype=np.float32)
    # Add multiple distinct crater and ridge structures across the terrain
    for _ in range(35):
        cx = np.random.randint(40, w_ref - 40)
        cy = np.random.randint(40, h_ref - 40)
        rad = np.random.randint(15, 38)
        val = np.random.uniform(120, 240)
        cv2.circle(master_terrain, (cx, cy), rad, val, -1)
        cv2.circle(master_terrain, (cx, cy), max(3, rad - 6), val * 0.4, -1)

    master_terrain = cv2.GaussianBlur(master_terrain, (0, 0), 1.5)
    noise_ref = np.random.normal(0, 3, (h_ref, w_ref)).astype(np.float32)
    ref_img = np.clip(master_terrain + noise_ref, 0, 255).astype(np.uint8)

    # IIRS source: 128x128 at 75.0 m/px (covering same 9600m footprint, 2x coarser)
    # Known physical ground shift: +150 m X (+4 px ref, +2 px IIRS), -75 m Y (-2 px ref, -1 px IIRS)
    shift_x_ref = 4.0
    shift_y_ref = -2.0
    M = np.float32([[1, 0, shift_x_ref], [0, 1, shift_y_ref]])
    ref_shifted = cv2.warpAffine(ref_img, M, (w_ref, h_ref))
    iirs_base = cv2.resize(ref_shifted, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Construct 8-band hyperspectral cube with distinct spectral responses per band
    num_bands = 8
    bands = np.zeros((num_bands, 128, 128), dtype=np.uint8)
    for b in range(num_bands):
        spectral_factor = 0.75 + 0.05 * b  # Distinct spectral response per band
        noise_b = np.random.normal(0, 2, (128, 128)).astype(np.float32)
        bands[b] = np.clip(iirs_base * spectral_factor + noise_b, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        iirs_path = Path(tmpdir) / "test_iirs_cube.tif"
        ref_path = Path(tmpdir) / "test_ref_optical.png"
        out_dir = Path(tmpdir) / "output_iirs_reg"

        # 1. Write genuine multi-band GeoTIFF using rasterio
        with rasterio.open(
            str(iirs_path),
            "w",
            driver="GTiff",
            height=128,
            width=128,
            count=num_bands,
            dtype="uint8",
        ) as dst:
            for b in range(num_bands):
                dst.write(bands[b], b + 1)

        cv2.imwrite(str(ref_path), ref_img)

        # 2. Verify production loader directly recognizes multi-band hyperspectral raster
        gray_loaded, color_loaded, raster_meta = load_as_float_and_color(iirs_path)
        assert raster_meta["count"] == num_bands
        assert raster_meta["count"] > 3
        assert raster_meta["driver"] == "GTiff"
        assert gray_loaded.shape == (128, 128)
        assert gray_loaded.dtype == np.float32

        # 3. Verify metadata extraction identifies sensor
        meta_extracted = extract_sensor_metadata(iirs_path, declared_sensor="IIRS", explicit_gsd=iirs_gsd)
        assert meta_extracted.sensor == "IIRS"
        assert meta_extracted.gsd_m == 75.0

        # 4. Call actual production registration pipeline
        res = match_images_cfog(
            iirs_path,
            ref_path,
            output_dir=out_dir,
            source_sensor="IIRS",
            reference_sensor="TMC-2",
            explicit_gsd1=iirs_gsd,
            explicit_gsd2=ref_gsd,
        )

        # Must succeed without fallback
        assert res["status"] == "success", f"IIRS co-registration failed: {res.get('message')}"
        assert res["homography"] is not None

        # Verify physical scale factor recovery:
        # Homography maps IIRS native (75 m/px) to Ref native (37.5 m/px)
        # Expected linear scale factor is 75.0 / 37.5 = 2.0
        H = np.array(res["homography"])
        scale_x = float(np.sqrt(H[0, 0]**2 + H[1, 0]**2))
        scale_y = float(np.sqrt(H[0, 1]**2 + H[1, 1]**2))
        expected_scale = 75.0 / 37.5  # 2.0
        assert abs(scale_x - expected_scale) < 0.15, f"Scale X deviation: {scale_x} vs {expected_scale}"
        assert abs(scale_y - expected_scale) < 0.15, f"Scale Y deviation: {scale_y} vs {expected_scale}"

        # Verify inlier count
        metrics = res["metrics"]
        assert metrics is not None
        assert metrics["inlier_count"] >= 4
        assert metrics["inlier_count"] > 10  # Plentiful genuine correspondences across grid

        # 5. Verify full production output package
        outputs = res["outputs"]
        for key in ("registered_raster", "preview", "checkerboard", "matches", "metrics", "transform", "metadata"):
            p = Path(outputs[key])
            assert p.exists(), f"Output product {key} missing at {p}"

        # 6. Mandatory Rasterio verification of output GeoTIFF
        with rasterio.open(str(outputs["registered_raster"])) as src:
            assert src.driver == "GTiff"
            assert src.width == w_ref
            assert src.height == h_ref
            assert src.count in (1, 3)
            assert src.dtypes[0] == "uint8"

        # 7. Check JSON sidecars are valid parseable JSON
        for sidecar_key in ("matches", "metrics", "transform", "metadata"):
            with open(outputs[sidecar_key], "r") as sf:
                parsed = json.load(sf)
                assert isinstance(parsed, (dict, list))


def test_evaluation_summary_contains_iirs_and_triplet_consistency():
    """
    Regression guard ensuring that evaluation_summary.json contains multi-modal IIRS
    pairwise metrics (OHRC-IIRS, TMC-IIRS) and 3-way circular triplet consistency
    metrics for every region_* and triplet_* dataset.
    """
    summary_path = REPO_ROOT / "evaluation_output" / "evaluation_summary.json"
    assert summary_path.exists(), f"Missing evaluation summary at {summary_path}"

    with open(summary_path) as f:
        summary = json.load(f)

    assert isinstance(summary, list)
    assert len(summary) >= 8, f"Expected at least 8 evaluated datasets, found {len(summary)}"

    found_ids = {entry.get("region_id") or entry.get("dataset_id") for entry in summary}
    required_ids = {
        "region_001", "region_002", "region_003", "region_004", "region_005", "region_006",
        "triplet_01_ch2_ohr_ncp_202", "triplet_new_2022"
    }
    assert required_ids.issubset(found_ids), f"Missing datasets in summary: {required_ids - found_ids}"

    for entry in summary:
        ds_id = entry.get("region_id") or entry.get("dataset_id")

        # 1. Must have pairs breakdown
        assert "pairs" in entry, f"Missing 'pairs' dictionary for {ds_id}"
        pairs = entry["pairs"]
        assert "ohrc_tmc" in pairs, f"Missing 'ohrc_tmc' pair in {ds_id}"
        assert "ohrc_iirs" in pairs, f"Missing 'ohrc_iirs' pair in {ds_id}"
        assert "tmc_iirs" in pairs, f"Missing 'tmc_iirs' pair in {ds_id}"

        # 2. IIRS pairs must be populated dictionaries with status and inlier_count
        assert pairs["ohrc_iirs"] is not None, f"Null ohrc_iirs pair for {ds_id}"
        assert "status" in pairs["ohrc_iirs"], f"Missing status in ohrc_iirs for {ds_id}"
        assert "inlier_count" in pairs["ohrc_iirs"], f"Missing inlier_count in ohrc_iirs for {ds_id}"

        assert pairs["tmc_iirs"] is not None, f"Null tmc_iirs pair for {ds_id}"
        assert "status" in pairs["tmc_iirs"], f"Missing status in tmc_iirs for {ds_id}"
        assert "inlier_count" in pairs["tmc_iirs"], f"Missing inlier_count in tmc_iirs for {ds_id}"

        # 3. Triplet cycle consistency must enforce ZERO synthetic fallbacks
        assert "triplet_consistency" in entry, f"Missing 'triplet_consistency' in {ds_id}"
        tc = entry["triplet_consistency"]
        assert "status" in tc, f"Missing status in triplet_consistency for {ds_id}"
        assert tc["status"] in ("evaluated", "cycle_not_computable"), (
            f"Invalid triplet status '{tc['status']}' in {ds_id}. Must be 'evaluated' or 'cycle_not_computable'."
        )

        failed_legs = tc.get("failed_legs", [])
        if tc["status"] == "cycle_not_computable":
            # Must NOT substitute an identity matrix to produce a fake numeric cycle RMSE
            assert tc["cycle_rmse_px"] is None, (
                f"{ds_id} reports cycle_not_computable but has numeric cycle_rmse_px: {tc['cycle_rmse_px']}"
            )
            assert tc["cycle_closed_successfully"] is False
            assert "reason" in tc and tc["reason"] is not None
            assert len(failed_legs) > 0, f"Expected failed legs list for uncomputable cycle in {ds_id}"
        elif tc["status"] == "evaluated":
            # Evaluated requires all underlying legs to have succeeded with real homographies
            assert isinstance(tc["cycle_rmse_px"], (int, float)), (
                f"Evaluated cycle must have numeric cycle_rmse_px in {ds_id}"
            )
            assert len(failed_legs) == 0, f"Evaluated cycle cannot have failed legs in {ds_id}: {failed_legs}"
            assert pairs.get("ohrc_tmc", {}).get("status") == "success"

        # Hard invariant: NEVER report a numeric cycle RMSE if any underlying leg failed
        if failed_legs:
            assert tc["cycle_rmse_px"] is None, (
                f"Violation of zero-synthetic-fallback principle: {ds_id} reported numeric cycle RMSE "
                f"({tc['cycle_rmse_px']}) despite failed leg(s): {failed_legs}"
            )




