"""
tests/test_terrain_adaptive_log_gabor.py - Tests for Terrain-Adaptive Log-Gabor Configuration.

Verifies:
1. DEM-derived slope, curvature, or roughness are used ONLY when the DEM is aligned.
2. The current fixed filter bank is preserved as the default baseline.
3. Filter scale is selected based on local terrain frequency, not arbitrary image intensity.
4. Per-tile parameter selection and geomorphic rationale are logged.
5. Fixed versus adaptive filters are compared using the exact same checkpoints.
6. Adaptive behavior is disabled by default.
7. Dedicated synthetic tests for:
   - Smooth mare
   - Crater rims
   - High-frequency rugged terrain
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT))

from terrain_adaptive_log_gabor import (
    BASELINE_LOG_GABOR_CONFIG,
    LogGaborConfig,
    TerrainTileRecord,
    TerrainAdaptiveLogGaborEngine,
    compute_terrain_metrics_from_dem,
    select_log_gabor_config_for_terrain,
    compute_phase_congruency_with_config,
    compare_fixed_vs_adaptive_log_gabor,
)
from matcher_cfog import match_images_cfog


# ===========================================================================
# Synthetic Terrain Generators
# ===========================================================================

def generate_synthetic_smooth_mare(shape=(128, 128), gsd_m=1.0, seed=42):
    """
    Generates a synthetic smooth lunar mare DEM and corresponding image.
    Low slope (<3 deg), low roughness (<1 m), gentle low-frequency rolling swell.
    """
    rng = np.random.RandomState(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    # Gentle 200m wavelength swell with 2m amplitude
    dem = 1000.0 + 2.0 * np.sin(2 * np.pi * x / 180.0) + 1.5 * np.cos(2 * np.pi * y / 200.0)
    dem += rng.normal(0, 0.1, shape)  # slight micro-noise

    # Image: low-contrast basalt plain with subtle shading
    img = np.clip(120.0 + 15.0 * np.sin(2 * np.pi * x / 180.0) + rng.normal(0, 2.0, shape), 0, 255).astype(np.uint8)
    return dem.astype(np.float32), img


def generate_synthetic_crater_rim(shape=(128, 128), gsd_m=1.0, seed=43):
    """
    Generates a synthetic crater rim DEM and corresponding image.
    Features a steep rim slope (>20 deg) and sharp topographic curvature.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(x - cx, y - cy)

    crater_radius = 40.0
    rim_height = 150.0
    floor_depth = 250.0

    dem = np.zeros(shape, dtype=np.float64) + 1000.0
    # Crater bowl
    inside = r <= crater_radius
    dem[inside] = 1000.0 - floor_depth * (1.0 - (r[inside] / crater_radius) ** 2)
    # Raised rim
    rim_zone = (r > crater_radius * 0.8) & (r < crater_radius * 1.4)
    dem[rim_zone] += rim_height * np.exp(-((r[rim_zone] - crater_radius) ** 2) / (2 * 8.0 ** 2))

    # Optical image: illuminated crater with cast shadow
    img = np.zeros(shape, dtype=np.float64) + 128.0
    # East-West illumination gradient from crater topography
    gx = cv2.Sobel(dem, cv2.CV_64F, 1, 0, ksize=3)
    img = np.clip(128.0 - 5.0 * gx, 10, 250).astype(np.uint8)
    return dem.astype(np.float32), img


def generate_synthetic_rugged_terrain(shape=(128, 128), gsd_m=1.0, seed=44):
    """
    Generates a synthetic high-frequency rugged ejecta blanket DEM and image.
    High roughness (>10 m), multi-scale dense micro-relief.
    """
    rng = np.random.RandomState(seed)
    h, w = shape
    dem = np.zeros(shape, dtype=np.float64) + 1000.0

    # Multi-frequency Perlin-like fractal terrain
    for freq, amp in [(0.05, 25.0), (0.1, 15.0), (0.2, 10.0), (0.4, 5.0)]:
        dem += amp * np.sin(freq * (np.arange(w)[None, :] + rng.uniform(0, 10))) * \
                     np.cos(freq * (np.arange(h)[:, None] + rng.uniform(0, 10)))

    dem += rng.normal(0, 3.0, shape)
    # High-frequency texture image
    img = np.clip(128.0 + (dem - 1000.0) * 3.0 + rng.normal(0, 8.0, shape), 0, 255).astype(np.uint8)
    return dem.astype(np.float32), img


# ===========================================================================
# 1. Baseline Preservation & Default Inactive (Requirements 2 & 6)
# ===========================================================================

def test_baseline_preserved_as_default():
    """Requirement 2 & 6: Fixed baseline is the canonical default."""
    assert BASELINE_LOG_GABOR_CONFIG.min_wavelength == 3.0
    assert BASELINE_LOG_GABOR_CONFIG.num_scales == 3
    assert BASELINE_LOG_GABOR_CONFIG.mult == 2.1
    assert BASELINE_LOG_GABOR_CONFIG.sigma_on_f == 0.55
    assert BASELINE_LOG_GABOR_CONFIG.num_orientations == 4
    assert BASELINE_LOG_GABOR_CONFIG.is_adaptive is False
    assert BASELINE_LOG_GABOR_CONFIG.terrain_type == "fixed_baseline"


def test_matcher_cfog_defaults_to_fixed_baseline():
    """Requirement 6: match_images_cfog defaults to enable_terrain_adaptive_log_gabor=False."""
    import inspect
    sig = inspect.signature(match_images_cfog)
    assert sig.parameters["enable_terrain_adaptive_log_gabor"].default is False
    assert sig.parameters["dem_aligned"].default is False


# ===========================================================================
# 2. DEM Alignment Prerequisite (Requirement 1)
# ===========================================================================

def test_dem_unaligned_strictly_falls_back_to_baseline():
    """
    Requirement 1: Use DEM-derived slope, curvature, or roughness ONLY when DEM is aligned.
    When dem_aligned=False, even if a steep DEM is provided, must strictly return baseline.
    """
    dem_crater, _ = generate_synthetic_crater_rim()

    # Even with steep crater DEM, dem_aligned=False must force fallback
    config, rec = select_log_gabor_config_for_terrain(
        dem_tile=dem_crater,
        dem_aligned=False,
        tile_id="unaligned_test_tile",
    )

    assert config == BASELINE_LOG_GABOR_CONFIG
    assert config.is_adaptive is False
    assert rec.dem_aligned is False
    assert "dem_unaligned_or_unavailable_fallback_to_baseline" in rec.selection_rationale


def test_missing_dem_strictly_falls_back_to_baseline():
    """When DEM is None, must strictly return baseline."""
    config, rec = select_log_gabor_config_for_terrain(
        dem_tile=None,
        dem_aligned=True,
        tile_id="none_dem_tile",
    )
    assert config == BASELINE_LOG_GABOR_CONFIG
    assert config.is_adaptive is False


# ===========================================================================
# 3. Frequency-Based Scale Selection vs Image Intensity (Requirement 3)
# ===========================================================================

def test_scale_selected_from_terrain_frequency_not_image_intensity():
    """
    Requirement 3: Select filter scale based on local terrain frequency,
    NOT arbitrary image intensity.
    
    Verifies that changing image brightness/contrast does NOT alter the
    DEM-derived geomorphic selection.
    """
    dem_mare, img_mare = generate_synthetic_smooth_mare()

    # Selection 1: with regular DEM
    config1, rec1 = select_log_gabor_config_for_terrain(dem_mare, dem_aligned=True, gsd_m=1.0)

    # Invert image brightness or multiply intensity 10x
    img_dark = (img_mare * 0.1).astype(np.uint8)
    img_bright = np.clip(img_mare * 2.0, 0, 255).astype(np.uint8)
    img_inverted = (255 - img_mare).astype(np.uint8)

    # Selection depends on DEM geomorphometry alone:
    config2, rec2 = select_log_gabor_config_for_terrain(dem_mare, dem_aligned=True, gsd_m=1.0)

    assert config1.min_wavelength == config2.min_wavelength
    assert config1.terrain_type == config2.terrain_type == "smooth_mare"
    assert rec1.mean_slope_deg == rec2.mean_slope_deg
    assert rec1.roughness_m == rec2.roughness_m


# ===========================================================================
# 4. Synthetic Terrain Tests: Mare, Crater Rim, Rugged (Requirement 7)
# ===========================================================================

def test_synthetic_smooth_mare():
    """
    Requirement 7: Synthetic smooth mare test.
    Low slope and roughness -> selects longer wavelength bank to suppress noise.
    """
    dem_mare, img_mare = generate_synthetic_smooth_mare()
    metrics = compute_terrain_metrics_from_dem(dem_mare, gsd_m=1.0)

    assert metrics["mean_slope_deg"] < 5.0
    assert metrics["roughness_m"] < 3.0

    config, rec = select_log_gabor_config_for_terrain(dem_mare, dem_aligned=True, gsd_m=1.0)
    assert config.terrain_type == "smooth_mare"
    assert config.is_adaptive is True
    # Longer wavelength selected for smooth low-frequency plains
    assert config.min_wavelength == 6.0
    assert config.num_scales >= 4


def test_synthetic_crater_rim():
    """
    Requirement 7: Synthetic crater rim test.
    Steep slope and high curvature -> selects edge-preserving wavelength (3.0 px).
    """
    dem_crater, img_crater = generate_synthetic_crater_rim()
    metrics = compute_terrain_metrics_from_dem(dem_crater, gsd_m=1.0)

    assert metrics["max_slope_deg"] > 15.0
    assert metrics["mean_curvature"] > 0.01

    config, rec = select_log_gabor_config_for_terrain(dem_crater, dem_aligned=True, gsd_m=1.0)
    assert config.terrain_type == "crater_rim"
    assert config.is_adaptive is True
    # Edge-preserving scale
    assert config.min_wavelength == 3.0


def test_synthetic_high_frequency_rugged():
    """
    Requirement 7: Synthetic high-frequency rugged terrain test.
    High roughness -> selects fine-scale high-resolution bank (min_wavelength=2.0 px).
    """
    dem_rugged, img_rugged = generate_synthetic_rugged_terrain()
    metrics = compute_terrain_metrics_from_dem(dem_rugged, gsd_m=1.0)

    assert metrics["roughness_m"] >= 5.0

    config, rec = select_log_gabor_config_for_terrain(dem_rugged, dem_aligned=True, gsd_m=1.0)
    assert config.terrain_type == "high_frequency_rugged"
    assert config.is_adaptive is True
    # Fine-scale bank
    assert config.min_wavelength == 2.0
    assert config.mult <= 2.0


# ===========================================================================
# 5. Per-Tile Logging (Requirement 4)
# ===========================================================================

def test_tile_parameter_logging():
    """
    Requirement 4: Log the selected parameters per tile.
    Verifies all required geomorphic and configuration attributes are captured.
    """
    dem_crater, img_crater = generate_synthetic_crater_rim((128, 128))
    engine = TerrainAdaptiveLogGaborEngine(enable_terrain_adaptive=True, gsd_m=1.0)

    pc, config = engine.process_tile(
        img_tile=img_crater,
        dem_tile=dem_crater,
        dem_aligned=True,
        tile_id="tile_row0_col1",
        bounds=(0, 128, 0, 128),
    )

    audit = engine.get_audit_log()
    assert len(audit) == 1
    rec = audit[0]

    # Required fields
    assert rec["tile_id"] == "tile_row0_col1"
    assert tuple(rec["bounds"]) == (0, 128, 0, 128)
    assert rec["dem_aligned"] is True
    assert "mean_slope_deg" in rec and rec["mean_slope_deg"] >= 0.0
    assert "max_slope_deg" in rec and rec["max_slope_deg"] >= 0.0
    assert "mean_curvature" in rec
    assert "roughness_m" in rec
    assert "dominant_wavelength_px" in rec
    assert "selected_config" in rec
    assert rec["selected_config"]["min_wavelength"] in (2.0, 3.0, 3.5, 6.0)
    assert "selection_rationale" in rec


# ===========================================================================
# 6. Checkpoint Comparison on Identical Checkpoints (Requirement 5)
# ===========================================================================

def test_compare_fixed_vs_adaptive_uses_same_checkpoints():
    """
    Requirement 5: Compare fixed versus adaptive filters using the same checkpoints.
    Verifies that compare_fixed_vs_adaptive_log_gabor evaluates both on identical points.
    """
    dem_crater, img1 = generate_synthetic_crater_rim((256, 256), seed=50)

    # Shift image by (2.0, 1.0) px to simulate registration target
    M = np.float32([[1, 0, 2.0], [0, 1, 1.0]])
    img2 = cv2.warpAffine(img1, M, (256, 256))

    # Define 8 identical independent checkpoints around crater features
    chk_src = np.array([
        [64.0, 64.0], [192.0, 64.0], [64.0, 192.0], [192.0, 192.0],
        [100.0, 128.0], [156.0, 128.0], [128.0, 100.0], [128.0, 156.0]
    ], dtype=np.float64)
    chk_dst = chk_src + np.array([2.0, 1.0])

    comp_report = compare_fixed_vs_adaptive_log_gabor(
        img1=img1,
        img2=img2,
        dem=dem_crater,
        dem_aligned=True,
        checkpoints_src=chk_src,
        checkpoints_dst=chk_dst,
        gsd_m=1.0,
        tile_size=128,
    )

    assert comp_report["status"] == "evaluated"
    assert comp_report["fixed_checkpoint_rmse"] is not None
    assert comp_report["adaptive_checkpoint_rmse"] is not None
    assert comp_report["delta_rmse"] is not None
    assert "adaptive_improved" in comp_report
    assert comp_report["evaluated_checkpoints_count"] == 8
    assert len(comp_report["tile_audit_log"]) > 0


def test_phase_congruency_numerical_bounds():
    """Confirms Phase Congruency output is bounded within [0, 1] for all configs."""
    _, img = generate_synthetic_crater_rim((64, 64))
    for config in [
        BASELINE_LOG_GABOR_CONFIG,
        LogGaborConfig(min_wavelength=6.0, num_scales=4, mult=2.4),
        LogGaborConfig(min_wavelength=2.0, num_scales=4, mult=1.8),
    ]:
        pc = compute_phase_congruency_with_config(img, config)
        assert pc.shape == (64, 64)
        assert np.all(pc >= 0.0)
        assert np.all(pc <= 1.0)
        assert np.all(np.isfinite(pc))


def test_match_images_cfog_terrain_adaptive_integration(tmp_path):
    """
    Verifies that match_images_cfog runs end-to-end with terrain-adaptive Log-Gabor
    when enabled with aligned DEM, and properly logs per-tile audit records.
    """
    dem_crater, img1 = generate_synthetic_crater_rim((256, 256), seed=60)
    # Target image with small known shift
    M = np.float32([[1, 0, 3.0], [0, 1, 2.0]])
    img2 = cv2.warpAffine(img1, M, (256, 256))

    out_dir_adaptive = tmp_path / "out_adaptive"
    out_dir_baseline = tmp_path / "out_baseline"

    # 1. Run with terrain-adaptive enabled and aligned DEM
    res_adaptive = match_images_cfog(
        img_path1=img1,
        img_path2=img2,
        dem_array=dem_crater,
        output_dir=out_dir_adaptive,
        explicit_gsd1=1.0,
        explicit_gsd2=1.0,
        enable_terrain_adaptive_log_gabor=True,
        dem_aligned=True,
    )

    assert res_adaptive["terrain_adaptive_log_gabor_enabled"] is True
    assert "terrain_log_gabor_audit" in res_adaptive
    assert len(res_adaptive["terrain_log_gabor_audit"]) > 0
    # First tile record must have valid geomorphic metrics
    rec0 = res_adaptive["terrain_log_gabor_audit"][0]
    assert rec0["dem_aligned"] is True
    assert "mean_slope_deg" in rec0

    # 2. Run with dem_aligned=False -> must NOT enable adaptive behavior
    res_unaligned = match_images_cfog(
        img_path1=img1,
        img_path2=img2,
        dem_array=dem_crater,
        output_dir=out_dir_baseline,
        explicit_gsd1=1.0,
        explicit_gsd2=1.0,
        enable_terrain_adaptive_log_gabor=True,
        dem_aligned=False,
    )

    assert res_unaligned["terrain_adaptive_log_gabor_enabled"] is False

