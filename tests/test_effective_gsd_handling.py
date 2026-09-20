"""
tests/test_effective_gsd_handling.py

Tests for effective GSD handling across resized, cropped, tiled, and preprocessed rasters:
1. Distinguish native_sensor_gsd_m from effective_raster_gsd_m.
2. Explicit crop scale and resampling scale tracking.
3. Transform chain and manifest lineage verification.
4. Prevention of blind native GSD attachment to resized rasters.
5. Inconsistent GSD metadata validation.
6. Verification on native, cropped, downsampled, and upsampled products.
7. Real-data manifest verification without fabrication.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "data_preprocessing_pipeline"))

from ML_model.metadata import (
    SensorMetadata,
    extract_sensor_metadata,
    validate_gsd_consistency,
    SENSOR_SPECS,
)
from data_preprocessing_pipeline.lunar_pipeline.scale import (
    compute_effective_gsd,
    compute_resampling_factor,
    resample_to_gsd,
)
import numpy as np


# ---------------------------------------------------------------------------
# Test 1: Native Product
# ---------------------------------------------------------------------------
def test_native_product_gsd():
    """An unscaled, uncropped product has effective_gsd == native_gsd and resampling_factor == 1.0."""
    meta = SensorMetadata(
        sensor="OHRC",
        native_gsd_m=0.25,
        effective_gsd_m=0.25,
        resampling_factor=1.0,
        crop_transform=None,
        parent_product_id="ch2_ohr_ncp_20210405t1606536730_d_img_d18",
    )
    meta.validate_gsd()

    assert meta.native_sensor_gsd_m == 0.25
    assert meta.effective_raster_gsd_m == 0.25
    assert meta.gsd_m == 0.25
    assert meta.resampling_factor == 1.0
    assert meta.crop_transform is None
    assert meta.parent_product_id == "ch2_ohr_ncp_20210405t1606536730_d_img_d18"

    d = meta.to_dict()
    assert d["native_gsd_m"] == 0.25
    assert d["effective_gsd_m"] == 0.25
    assert d["resampling_factor"] == 1.0


# ---------------------------------------------------------------------------
# Test 2: Cropped Product (Pure Crop, No Resampling)
# ---------------------------------------------------------------------------
def test_cropped_product_gsd():
    """A pure crop preserves native resolution with resampling_factor == 1.0 and stores crop_transform."""
    crop_tf = {"col_off": 1024, "row_off": 2048, "width": 512, "height": 512}
    meta = SensorMetadata(
        sensor="OHRC",
        native_gsd_m=0.25,
        effective_gsd_m=0.25,
        resampling_factor=1.0,
        crop_transform=crop_tf,
        parent_product_id="ch2_ohr_ncp_20210405t1606536730_d_img_d18",
    )
    meta.validate_gsd()

    assert meta.native_sensor_gsd_m == 0.25
    assert meta.effective_raster_gsd_m == 0.25
    assert meta.crop_transform == crop_tf
    assert meta.resampling_factor == 1.0
    assert meta.parent_product_id is not None


# ---------------------------------------------------------------------------
# Test 3: Downsampled Product (OHRC 20x Downsample to TMC-2 Grid)
# ---------------------------------------------------------------------------
def test_downsampled_product_gsd():
    """Downsampling 20x coarsen pixel size from 0.25m to 5.0m."""
    native_gsd = 0.25
    working_gsd = 5.0
    # physical_scale_factor = 512 / 10240 = 0.05
    resampling_factor = native_gsd / working_gsd  # 0.05

    effective_gsd = compute_effective_gsd(native_gsd, resampling_factor)
    assert abs(effective_gsd - working_gsd) < 1e-5

    meta = SensorMetadata(
        sensor="OHRC",
        native_gsd_m=native_gsd,
        effective_gsd_m=effective_gsd,
        resampling_factor=resampling_factor,
        crop_transform={"width": 10240, "height": 10240},
        parent_product_id="ch2_ohr_ncp_20210405t1606536730_d_img_d18",
    )
    meta.validate_gsd()

    assert meta.native_sensor_gsd_m == 0.25
    assert meta.effective_raster_gsd_m == 5.0
    assert meta.gsd_m == 5.0
    assert meta.resampling_factor == 0.05


# ---------------------------------------------------------------------------
# Test 4: Upsampled Product (IIRS 2x Upsample)
# ---------------------------------------------------------------------------
def test_upsampled_product_gsd():
    """Upsampling 2x refines raster pixel spacing from 75.0m to 37.5m."""
    native_gsd = 75.0
    resampling_factor = 2.0  # 512px derived from 256px crop
    effective_gsd = compute_effective_gsd(native_gsd, resampling_factor)
    assert abs(effective_gsd - 37.5) < 1e-5

    meta = SensorMetadata(
        sensor="IIRS",
        native_gsd_m=native_gsd,
        effective_gsd_m=effective_gsd,
        resampling_factor=resampling_factor,
        parent_product_id="ch2_iir_nri_20211221t0324126144_d_img_hw1",
    )
    meta.validate_gsd()

    assert meta.native_sensor_gsd_m == 75.0
    assert meta.effective_raster_gsd_m == 37.5
    assert meta.resampling_factor == 2.0


# ---------------------------------------------------------------------------
# Test 5: Validation Rejects Inconsistent GSD Metadata
# ---------------------------------------------------------------------------
def test_validation_rejects_inconsistent_gsd():
    """Rejects invalid GSD configurations (negative, zero, contradictory, or blind attachment)."""
    # 1. Non-positive values
    with pytest.raises(ValueError, match="native_gsd_m must be > 0"):
        validate_gsd_consistency(native_gsd_m=0.0, effective_gsd_m=5.0, resampling_factor=0.05)

    with pytest.raises(ValueError, match="effective_gsd_m must be > 0"):
        validate_gsd_consistency(native_gsd_m=0.25, effective_gsd_m=-1.0, resampling_factor=0.05)

    with pytest.raises(ValueError, match="resampling_factor must be > 0"):
        validate_gsd_consistency(native_gsd_m=0.25, effective_gsd_m=5.0, resampling_factor=0.0)

    # 2. Contradictory effective GSD: claiming 20x downsampled image still has 0.25m effective GSD
    with pytest.raises(ValueError, match="Inconsistent GSD metadata"):
        validate_gsd_consistency(
            native_gsd_m=0.25,
            effective_gsd_m=0.25,  # Blind attachment of native GSD!
            resampling_factor=0.05,  # When downsampled 20x, effective GSD should be 5.0m
        )

    # 3. Contradictory scaling: native 5.0m, unscaled (rf=1.0), but claiming effective 25.0m
    with pytest.raises(ValueError, match="Inconsistent GSD metadata"):
        validate_gsd_consistency(
            native_gsd_m=5.0,
            effective_gsd_m=25.0,
            resampling_factor=1.0,
        )


# ---------------------------------------------------------------------------
# Test 6: Resampling Helper Functions
# ---------------------------------------------------------------------------
def test_scale_helpers():
    """Verifies compute_effective_gsd and compute_resampling_factor."""
    assert compute_resampling_factor(1000, 500) == 0.5
    assert compute_resampling_factor(256, 512) == 2.0
    assert compute_effective_gsd(0.25, 0.05) == pytest.approx(5.0)
    assert compute_effective_gsd(5.0, 1.0) == pytest.approx(5.0)

    with pytest.raises(ValueError):
        compute_effective_gsd(0.25, -1.0)
    with pytest.raises(ValueError):
        compute_resampling_factor(0, 100)


# ---------------------------------------------------------------------------
# Test 7: Real Manifest Verification (region_001 and triplet_new_2022)
# ---------------------------------------------------------------------------
def test_real_manifest_dual_gsd_preservation():
    """Verifies that real flight manifests preserve authoritative native and effective GSDs."""
    repo_root = Path(__file__).resolve().parent.parent
    reg1_manifest = repo_root / "data_preprocessing_pipeline" / "processed_triplets" / "region_001" / "manifest.json"
    assert reg1_manifest.exists(), "region_001 manifest must exist"

    with open(reg1_manifest) as f:
        m1 = json.load(f)

    # Authoritative real flight specs in region_001
    assert m1["ohrc_gsd_m"] == 0.25
    assert m1["tmc2_gsd_m"] == 5.4
    assert m1["iirs_gsd_m"] == 69.04
    assert m1["ohrc_large_effective_gsd_m"] == 15.9883
    assert m1["tmc2_large_effective_gsd_m"] == 39.4316
    assert m1["iirs_large_effective_gsd_m"] == 39.4316

    # Test metadata extraction on region_001 files
    reg1_dir = reg1_manifest.parent
    ohrc_large_path = reg1_dir / "ohrc_large_512.png"
    if ohrc_large_path.exists():
        meta_large = extract_sensor_metadata(ohrc_large_path)
        assert meta_large.native_sensor_gsd_m == 0.25
        assert meta_large.effective_raster_gsd_m == 15.9883
        assert abs(meta_large.resampling_factor - (0.25 / 15.9883)) < 1e-4
        assert meta_large.parent_product_id == "ch2_ohr_ncp_20210405t1606536730_d_img_d18"

    ohrc_512_path = reg1_dir / "ohrc_512.png"
    if ohrc_512_path.exists():
        meta_512 = extract_sensor_metadata(ohrc_512_path)
        assert meta_512.native_sensor_gsd_m == 0.25
        assert meta_512.effective_raster_gsd_m == 5.4
        assert abs(meta_512.resampling_factor - (0.25 / 5.4)) < 1e-4

    # Test triplet_new_2022
    triplet_2022_manifest = repo_root / "data_preprocessing_pipeline" / "processed_triplets" / "triplet_new_2022" / "manifest.json"
    if triplet_2022_manifest.exists():
        with open(triplet_2022_manifest) as f:
            m2022 = json.load(f)
        assert m2022["ohrc_gsd_m"] == 0.32
        assert m2022["tmc2_gsd_m"] == 3.96
        assert m2022["iirs_gsd_m"] == 71.24
        assert m2022["ohrc_large_effective_gsd_m"] == 23.4492


# ---------------------------------------------------------------------------
# Test 8: Resampling Array Execution Integrity
# ---------------------------------------------------------------------------
def test_resample_to_gsd_scaling():
    """Verifies that resample_to_gsd downsizes array dimensions correctly."""
    arr = np.ones((1, 1000, 1000), dtype=np.float32)
    # Downsample from native 0.25 to working 5.0 (factor of 20x -> 50x50)
    out, tf, sf = resample_to_gsd(arr, None, native_gsd=0.25, working_gsd=5.0)
    assert sf == 20.0
    assert out.shape == (1, 50, 50)
