"""
tests/test_anti_aliasing.py — Comprehensive validation of sensor-aware anti-aliasing module.

Verifies:
1. cv2.INTER_AREA is preserved as the reproducible default baseline.
2. Optional prefilter kernels tailored by sensor and downsampling factor.
3. Configurable and provenance-tracked OTF parameters.
4. STRICT SCIENTIFIC HONESTY (Rule 13): Never claim OTF correction when no sensor OTF is provided.
5. Three-way comparison across INTER_AREA, Gaussian prefilter + INTER_AREA, and configured OTF.
6. Quantitative evaluation of edge preservation, aliasing metric, and registration error.
7. Synthetic frequency patterns (Siemens star, Zone plate, high-frequency chirp).
8. Physical disclaimer documentation requiring actual calibration curves.
"""

from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "data_preprocessing_pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "data_preprocessing_pipeline"))

from data_preprocessing_pipeline.lunar_pipeline.anti_aliasing import (
    PHYSICAL_OTF_DISCLAIMER,
    AntiAliasingMethod,
    AntiAliasingResult,
    SensorOTFConfig,
    compare_anti_aliasing_methods,
    compute_aliasing_metric,
    compute_edge_preservation_score,
    compute_registration_error,
    downsample_sensor_aware,
    generate_high_freq_chirp,
    generate_siemens_star,
    generate_zone_plate,
)
from data_preprocessing_pipeline.lunar_pipeline.scale import resample_to_gsd


def test_inter_area_reproducible_default():
    """Requirement 1: Keep cv2.INTER_AREA as the reproducible default baseline."""
    # Test on synthetic lunar surface patch
    np.random.seed(42)
    img = np.random.uniform(0.1, 0.9, (128, 128)).astype(np.float32)

    # 1. Direct call to downsample_sensor_aware with default method
    res = downsample_sensor_aware(img, target_shape_or_factor=2.0)
    assert res.method == AntiAliasingMethod.INTER_AREA
    assert res.otf_correction_claimed is False

    expected = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(
        res.output_raster,
        expected,
        atol=1e-6,
        err_msg="INTER_AREA baseline must match cv2.resize bitwise",
    )

    # 2. Pipeline resample_to_gsd default path
    arr_3d = np.expand_dims(img, axis=0)
    out_arr, _, sf = resample_to_gsd(arr_3d, transform=None, native_gsd=0.25, working_gsd=0.5)
    np.testing.assert_allclose(
        out_arr[0],
        expected,
        atol=1e-6,
        err_msg="resample_to_gsd default must use reproducible INTER_AREA",
    )


def test_optional_prefilter_kernels_by_sensor():
    """Requirement 2: Optional prefilter kernels tailored by sensor and scale factor."""
    zone_plate = generate_zone_plate(size=256, max_freq=0.5)

    sensors = ["OHRC", "TMC-2", "IIRS", "LRO_NAC"]
    downsample_factor = 4.0

    for sensor in sensors:
        # Run Gaussian prefiltering
        res = downsample_sensor_aware(
            zone_plate,
            target_shape_or_factor=downsample_factor,
            method=AntiAliasingMethod.GAUSSIAN_PREFILTER,
            sensor=sensor,
        )

        assert res.method == AntiAliasingMethod.GAUSSIAN_PREFILTER
        assert res.provenance["sensor"] == sensor
        assert "gaussian_sigma" in res.provenance
        assert res.provenance["gaussian_sigma"] > 0.0
        assert res.output_raster.shape == (64, 64)

        # Baseline INTER_AREA without prefilter
        res_baseline = downsample_sensor_aware(
            zone_plate,
            target_shape_or_factor=downsample_factor,
            method=AntiAliasingMethod.INTER_AREA,
            sensor=sensor,
        )

        # Gaussian prefiltering must reduce high-frequency spectral aliasing compared to un-prefiltered INTER_AREA
        alias_prefilter = compute_aliasing_metric(res.output_raster)
        alias_baseline = compute_aliasing_metric(res_baseline.output_raster)
        assert alias_prefilter < alias_baseline, (
            f"Sensor {sensor}: Gaussian prefilter should suppress aliasing ({alias_prefilter:.4f} < {alias_baseline:.4f})"
        )


def test_configurable_and_provenance_tracked_otf():
    """Requirement 3: OTF parameters are configurable and provenance-tracked."""
    custom_cfg = SensorOTFConfig(
        sensor="OHRC",
        cutoff_freq=0.35,
        aperture_shape="circular_diffraction",
        wiener_damping=0.08,
        has_measured_otf=False,
        calibration_source="synthetic_flight_model_v1",
        metadata={"nominal_gsd_m": 0.25, "focal_length_mm": 2000.0},
    )

    img = generate_high_freq_chirp(size=128)
    res = downsample_sensor_aware(
        img,
        target_shape_or_factor=2.0,
        method=AntiAliasingMethod.CONFIGURED_OTF,
        otf_config=custom_cfg,
    )

    prov = res.provenance
    assert "otf_config" in prov
    assert prov["otf_config"]["cutoff_freq"] == 0.35
    assert prov["otf_config"]["aperture_shape"] == "circular_diffraction"
    assert prov["otf_config"]["wiener_damping"] == 0.08
    assert prov["otf_config"]["metadata"]["nominal_gsd_m"] == 0.25
    assert prov["input_dimensions"] == [128, 128]
    assert prov["output_dimensions"] == [64, 64]


def test_never_claim_otf_correction_without_sensor_otf():
    """Requirement 4: STRICT RULE 13 — Never claim OTF correction when no sensor OTF is provided."""
    img = generate_zone_plate(size=128)

    # Case A: Default / synthetic config (has_measured_otf=False)
    unmeasured_cfg = SensorOTFConfig(
        sensor="TMC-2",
        has_measured_otf=False,
    )
    res_unmeasured = downsample_sensor_aware(
        img,
        target_shape_or_factor=2.0,
        method=AntiAliasingMethod.CONFIGURED_OTF,
        otf_config=unmeasured_cfg,
    )
    # Must strictly NOT claim OTF correction!
    assert res_unmeasured.otf_correction_claimed is False
    assert res_unmeasured.provenance["provenance_status"] == "UNMEASURED_HEURISTIC_OTF"
    assert "No actual sensor OTF measurements" in res_unmeasured.provenance["warning"]

    # Case B: Verified calibration measurements provided (has_measured_otf=True)
    measured_cfg = SensorOTFConfig(
        sensor="OHRC",
        has_measured_otf=True,
        calibration_source="ISRO_SAC_CH2_OHRC_CAL_REPORT_2019_DOC402",
    )
    res_measured = downsample_sensor_aware(
        img,
        target_shape_or_factor=2.0,
        method=AntiAliasingMethod.CONFIGURED_OTF,
        otf_config=measured_cfg,
    )
    assert res_measured.otf_correction_claimed is True
    assert res_measured.provenance["provenance_status"] == "MEASURED_SENSOR_OTF"
    assert res_measured.provenance["calibration_source"] == "ISRO_SAC_CH2_OHRC_CAL_REPORT_2019_DOC402"


def test_three_way_comparison_metrics():
    """Requirement 5 & 6: Compare INTER_AREA, Gaussian + INTER_AREA, and Configured OTF."""
    test_pattern = generate_zone_plate(size=256, max_freq=0.4)

    results = compare_anti_aliasing_methods(
        test_pattern,
        downsample_factor=4.0,
        sensor="OHRC",
        true_shift=(1.5, 2.0),
    )

    # 1. All three methods must be present in the comparative analysis
    assert "INTER_AREA" in results
    assert "GAUSSIAN_PREFILTER" in results
    assert "CONFIGURED_OTF" in results

    m_area = results["INTER_AREA"]
    m_gauss = results["GAUSSIAN_PREFILTER"]
    m_otf = results["CONFIGURED_OTF"]

    # 2. Edge preservation score must be positive
    assert m_area.edge_preservation_score > 0.0
    assert m_gauss.edge_preservation_score > 0.0
    assert m_otf.edge_preservation_score > 0.0

    # 3. Aliasing metric must be finite and non-negative
    assert 0.0 <= m_area.aliasing_metric <= 1.0
    assert 0.0 <= m_gauss.aliasing_metric <= 1.0
    assert 0.0 <= m_otf.aliasing_metric <= 1.0

    # 4. Registration error must be finite and sub-pixel (< 1.0 px)
    assert m_area.registration_error_px >= 0.0
    assert m_gauss.registration_error_px >= 0.0
    assert m_otf.registration_error_px >= 0.0
    assert m_area.registration_error_px < 1.0
    assert m_gauss.registration_error_px < 1.0

    # 5. OTF correction claim status must be honest (False for default uncalibrated run)
    assert m_area.otf_correction_claimed is False
    assert m_gauss.otf_correction_claimed is False
    assert m_otf.otf_correction_claimed is False


def test_synthetic_frequency_pattern_generators():
    """Requirement 7: Verify generation and characteristics of synthetic frequency patterns."""
    # 1. Zone plate
    zp = generate_zone_plate(size=128, max_freq=0.5)
    assert zp.shape == (128, 128)
    assert zp.dtype == np.float32
    assert 0.0 <= np.min(zp) <= np.max(zp) <= 1.0

    # 2. Siemens star
    star = generate_siemens_star(size=128, num_spokes=16)
    assert star.shape == (128, 128)
    assert star.dtype == np.float32
    assert 0.0 <= np.min(star) <= np.max(star) <= 1.0

    # 3. High-frequency chirp
    chirp = generate_high_freq_chirp(size=128)
    assert chirp.shape == (128, 128)
    assert chirp.dtype == np.float32
    assert 0.0 <= np.min(chirp) <= np.max(chirp) <= 1.0


def test_physical_otf_disclaimer_documented():
    """Requirement 8: Document that actual sensor OTF values are required for physical interpretation."""
    assert "ACTUAL SENSOR OTF VALUES REQUIRED" in PHYSICAL_OTF_DISCLAIMER
    assert "ISRO SAC" in PHYSICAL_OTF_DISCLAIMER
    assert "must never be claimed as physical ground-truth" in PHYSICAL_OTF_DISCLAIMER

    # Ensure disclaimer is attached to every downsample result
    res = downsample_sensor_aware(generate_high_freq_chirp(64), target_shape_or_factor=2.0)
    assert res.disclaimer == PHYSICAL_OTF_DISCLAIMER
    assert "disclaimer" in res.to_dict()
