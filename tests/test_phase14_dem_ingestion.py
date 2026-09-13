"""Phase 14 unit tests: SLDEM2015 verified DEM ingestion & read-only hillshade preview.

Scope: isolated ingestion/validation diagnostics only. These tests never touch
production matching, candidate-generation defaults, the RF model, ground truth,
training data, or held-out evaluation groups.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from dem_ingestion import (  # noqa: E402
    DEMCoverageError,
    DEMInvalidMetadataError,
    DEMMissingError,
    DEMChecksumMismatchError,
    DEMNodataError,
    DEMUnsupportedFileError,
    PDS4Metadata,
    REGION_001_HISTORICAL_SOLAR,
    SOLAR_ELEVATION_REQUIRED,
    SolarElevationRequiredError,
    build_provenance_record,
    check_filename_and_type,
    compute_sha256,
    crop_region_window,
    expected_sha256_for_tile,
    find_supplied_file,
    load_region_bounds,
    preview_hillshade,
    resolve_dem_dir,
    validate_crop_statistics,
    validate_raster_metadata,
    verify_checksum,
    write_provenance_record,
)

TILE_ENTRY = {
    "tile_id": "SLDEM2015_512_30S_00S_315_360",
    "expected_filename": "SLDEM2015_512_30S_00S_315_360.jp2",
    "accepted_alternates": ["SLDEM2015_512_30S_00S_315_360.tif"],
    "expected_sha256": None,
    "expected_sha256_env_override": "SLDEM_TILE_315_360_SHA256",
}


def _write_geotiff(path: Path, west: float, south: float, east: float, north: float,
                   width: int = 64, height: int = 64, crs: str = "EPSG:4326",
                   fill=None, nodata: float = -32768.0) -> np.ndarray:
    transform = from_bounds(west, south, east, north, width, height)
    if fill is None:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        fill = 1500.0 + xx * 0.5 - yy * 0.25
    arr = np.asarray(fill, dtype=np.float32)
    with rasterio.open(
        str(path), "w", driver="GTiff", width=width, height=height,
        count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    return arr


# --- 2. checksum verification -------------------------------------------------


def test_checksum_mismatch_rejected(tmp_path: Path):
    target = tmp_path / "tile.tif"
    target.write_bytes(b"authoritative-bytes")
    with pytest.raises(DEMChecksumMismatchError):
        verify_checksum(target, "0" * 64)


def test_checksum_match_accepted(tmp_path: Path):
    target = tmp_path / "tile.tif"
    target.write_bytes(b"authoritative-bytes")
    digest = hashlib.sha256(b"authoritative-bytes").hexdigest()
    assert verify_checksum(target, digest) == digest
    assert compute_sha256(target) == digest


def test_checksum_without_operator_expectation_blocked(tmp_path: Path):
    target = tmp_path / "tile.tif"
    target.write_bytes(b"bytes")
    with pytest.raises(DEMMissingError, match="EXPECTED_CHECKSUM_NOT_SUPPLIED"):
        verify_checksum(target, None)


# --- 1. missing DEM ------------------------------------------------------------


def test_missing_dem_file_reported(tmp_path: Path):
    assert find_supplied_file(tmp_path, TILE_ENTRY) is None
    with pytest.raises(DEMMissingError):
        compute_sha256(tmp_path / "SLDEM2015_512_30S_00S_315_360.tif")


def test_resolve_dem_dir_precedence(tmp_path: Path, monkeypatch):
    cli = tmp_path / "cli"
    env = tmp_path / "envd"
    monkeypatch.setenv("CH2X_DEM_DIR", str(env))
    assert resolve_dem_dir(str(cli), REPO_ROOT) == cli
    assert resolve_dem_dir(None, REPO_ROOT) == env
    monkeypatch.delenv("CH2X_DEM_DIR")
    assert resolve_dem_dir(None, REPO_ROOT) == REPO_ROOT / "data" / "external" / "dem"


def test_expected_sha256_env_override(monkeypatch):
    monkeypatch.setenv("SLDEM_TILE_315_360_SHA256", "AB" * 32)
    assert expected_sha256_for_tile(TILE_ENTRY) == "ab" * 32


# --- unsupported file ----------------------------------------------------------


def test_png_dem512_rejected(tmp_path: Path):
    bad = tmp_path / "dem_512.png"
    bad.write_bytes(b"not elevation")
    with pytest.raises(DEMUnsupportedFileError, match="REJECTED_NON_AUTHORITATIVE"):
        check_filename_and_type(bad, TILE_ENTRY)
    other_png = tmp_path / "hillshade.png"
    other_png.write_bytes(b"not elevation")
    with pytest.raises(DEMUnsupportedFileError):
        check_filename_and_type(other_png, TILE_ENTRY)


def test_unexpected_filename_rejected(tmp_path: Path):
    wrong = tmp_path / "SLDEM2015_512_30S_00S_315_360_FINAL_v2.tif"
    wrong.write_bytes(b"x")
    with pytest.raises(DEMUnsupportedFileError, match="UNEXPECTED_FILENAME"):
        check_filename_and_type(wrong, TILE_ENTRY)


def test_unsupported_suffix_rejected(tmp_path: Path):
    bad = tmp_path / "SLDEM2015_512_30S_00S_315_360.h5"
    bad.write_bytes(b"x")
    with pytest.raises(DEMUnsupportedFileError, match="UNSUPPORTED_FILE_TYPE"):
        check_filename_and_type(bad, TILE_ENTRY)


# --- 3. raster metadata ----------------------------------------------------------


def test_invalid_crs_rejected(tmp_path: Path):
    # Earth-projected CRS (metres) must not pass as lunar cylindrical degrees.
    path = tmp_path / "SLDEM2015_512_30S_00S_315_360.tif"
    _write_geotiff(path, 0.0, 0.0, 1000.0, 1000.0, crs="EPSG:3857")
    with pytest.raises(DEMInvalidMetadataError, match="INVALID_CRS"):
        validate_raster_metadata(path)


def test_missing_crs_rejected(tmp_path: Path):
    path = tmp_path / "SLDEM2015_512_30S_00S_315_360.tif"
    transform = from_bounds(336.0, -4.0, 337.0, -2.0, 32, 32)
    with rasterio.open(
        str(path), "w", driver="GTiff", width=32, height=32,
        count=1, dtype="float32", transform=transform, nodata=-32768.0,
    ) as dst:
        dst.write(np.ones((32, 32), dtype=np.float32), 1)
    with pytest.raises(DEMInvalidMetadataError, match="CRS_MISSING"):
        validate_raster_metadata(path)


def test_valid_geographic_metadata_reported_verbatim(tmp_path: Path):
    path = tmp_path / "SLDEM2015_512_30S_00S_315_360.tif"
    _write_geotiff(path, 336.0, -4.0, 337.0, -2.0, width=512, height=1024)
    report = validate_raster_metadata(path)
    assert report.width == 512 and report.height == 1024
    assert report.bounds_west == pytest.approx(336.0)
    assert report.bounds_north == pytest.approx(-2.0)
    assert report.pixels_per_degree == pytest.approx(512.0)
    assert report.dtype == "float32"
    assert report.crs_string is not None


def test_corrupt_raster_rejected(tmp_path: Path):
    path = tmp_path / "SLDEM2015_512_30S_00S_315_360.tif"
    path.write_bytes(b"this is not a raster")
    with pytest.raises(DEMInvalidMetadataError, match="UNREADABLE_CORRUPT_RASTER"):
        validate_raster_metadata(path)


# --- 4/5. coverage + window cropping ----------------------------------------------


def test_insufficient_coverage_rejected(tmp_path: Path):
    path = tmp_path / "SLDEM2015_512_30S_00S_315_360.tif"
    _write_geotiff(path, 336.0, -4.0, 337.0, -2.0, width=512, height=1024)
    digest = compute_sha256(path)
    # Region window extends beyond the fixture tile to the east.
    outside = {"west_lon": 336.9, "east_lon": 337.5, "south_lat": -3.5, "north_lat": -3.0}
    with pytest.raises(DEMCoverageError, match="INSUFFICIENT_COVERAGE"):
        crop_region_window(path, outside, "region_001",
                           "SLDEM2015_512_30S_00S_315_360", digest)


def test_correct_window_cropping(tmp_path: Path):
    path = tmp_path / "SLDEM2015_512_30S_00S_315_360.tif"
    west, south, east, north = 336.0, -4.0, 337.0, -2.0
    width, height = 512, 1024  # 512 ppd
    source = _write_geotiff(path, west, south, east, north, width=width, height=height)
    digest = compute_sha256(path)
    # 0.1 deg lon x 0.1261284 deg lat window -> deterministic pixel shape.
    box = {"west_lon": 336.484646, "east_lon": 336.589455,
           "south_lat": -3.3748612, "north_lat": -3.2487328}
    crop = crop_region_window(path, box, "region_001",
                              "SLDEM2015_512_30S_00S_315_360", digest)
    exp_w = int(round((box["east_lon"] - box["west_lon"]) * 512))
    exp_h = int(round((box["north_lat"] - box["south_lat"]) * 512))
    assert crop.elevation_m.dtype == np.float32
    assert crop.elevation_m.shape == (exp_h, exp_w)
    assert crop.region_id == "region_001"
    assert crop.source_sha256 == digest
    assert crop.source_product_id == "SLDEM2015_512_30S_00S_315_360"
    # Geospatial correctness: cropped values match the source window.
    col0 = int(round((box["west_lon"] - west) * 512))
    row0 = int(round((north - box["north_lat"]) * 512))
    expected = source[row0:row0 + exp_h, col0:col0 + exp_w]
    np.testing.assert_allclose(crop.elevation_m, expected, rtol=1e-5, atol=1e-3)
    assert list(crop.crop_bounds) == [
        box["west_lon"], box["south_lat"], box["east_lon"], box["north_lat"]]
    # No resampling: crop resolution equals source resolution.
    assert abs(float(crop.transform.a)) == pytest.approx(1.0 / 512.0)


# --- 6. nodata handling ------------------------------------------------------------


def test_excessive_nodata_fails_not_filled():
    arr = np.full((20, 20), -32768.0, dtype=np.float32)
    arr[:5, :] = 1500.0  # 75% nodata
    with pytest.raises(DEMNodataError, match="EXCESSIVE_NODATA"):
        validate_crop_statistics(arr, -32768.0)


def test_all_nodata_crop_fails():
    arr = np.full((10, 10), -32768.0, dtype=np.float32)
    with pytest.raises(DEMNodataError, match="ALL_NODATA"):
        validate_crop_statistics(arr, -32768.0)


def test_healthy_crop_statistics():
    rng = np.random.default_rng(0)
    arr = (1500.0 + rng.normal(0, 5, (40, 30))).astype(np.float32)
    stats = validate_crop_statistics(arr, -32768.0)
    assert stats["cropped_shape"] == [40, 30]
    assert stats["nodata_fraction"] == pytest.approx(0.0)
    assert stats["finite_pixel_fraction"] == pytest.approx(1.0)
    assert stats["min_elevation_m"] < stats["mean_elevation_m"] < stats["max_elevation_m"]
    assert abs(stats["median_elevation_m"] - 1500.0) < 5.0


# --- 7. solar elevation refusal ------------------------------------------------------


def test_missing_solar_elevation_refused():
    dem = np.ones((16, 16), dtype=np.float32) * 1500.0
    with pytest.raises(SolarElevationRequiredError, match="SOLAR_ELEVATION_REQUIRED"):
        preview_hillshade(dem, 269.646098, None)
    with pytest.raises(SolarElevationRequiredError, match="SOLAR_ELEVATION_REQUIRED"):
        preview_hillshade(dem, 269.646098, float("nan"))
    with pytest.raises(SolarElevationRequiredError, match="SOLAR_ELEVATION_REQUIRED"):
        preview_hillshade(dem, None, 45.0)


def test_no_elevation_inferred_from_azimuth():
    # Azimuth alone (even valid) must never produce a rendering.
    dem = np.ones((16, 16), dtype=np.float32) * 1500.0 + np.arange(256).reshape(16, 16)
    with pytest.raises(SolarElevationRequiredError):
        preview_hillshade(dem, 108.866212, None, provenance_label="azimuth_only_probe")


# --- 8. region_001 diagnostic path ------------------------------------------------------


def test_region001_diagnostic_path_with_valid_dem(tmp_path: Path):
    bounds = load_region_bounds(REPO_ROOT)["region_001"]
    path = tmp_path / "SLDEM2015_512_30S_00S_315_360.tif"
    _write_geotiff(path, 336.0, -4.0, 337.0, -2.0, width=512, height=1024)
    digest = compute_sha256(path)
    report = validate_raster_metadata(path)
    crop = crop_region_window(path, bounds, "region_001",
                              "SLDEM2015_512_30S_00S_315_360", digest)
    stats = validate_crop_statistics(crop.elevation_m, crop.nodata)
    assert stats["finite_pixel_fraction"] == pytest.approx(1.0)
    hist = REGION_001_HISTORICAL_SOLAR
    shade = preview_hillshade(
        crop.elevation_m,
        hist["ohrc_sun_azimuth_deg"],
        hist["ohrc_sun_elevation_deg"],
        working_gsd_m=60.0,
        provenance_label=hist["provenance"],
    )
    assert shade.shape == crop.elevation_m.shape
    assert float(np.min(shade)) >= 0.0 and float(np.max(shade)) <= 1.0
    assert hist["provenance"] == "historical_repository_metadata"
    assert report.pixels_per_degree == pytest.approx(512.0)


# --- 9/10. PDS4 interface + provenance ------------------------------------------------------


def test_pds4_interface_elevation_unset_until_authoritative():
    pending = PDS4Metadata(source_image_id="region_002_tmc2")
    assert pending.solar_elevation_deg is None
    assert not pending.has_authoritative_elevation
    supplied = PDS4Metadata(
        source_image_id="region_002_tmc2",
        pds4_label_path="/operator/labels/region_002.xml",
        solar_azimuth_deg=108.866212,
        solar_elevation_deg=48.5,
        provenance="operator_supplied_pds4",
    )
    assert supplied.has_authoritative_elevation
    assert supplied.to_dict()["pds4_label_path"] == "/operator/labels/region_002.xml"


def test_provenance_generation(tmp_path: Path):
    bounds = load_region_bounds(REPO_ROOT)["region_001"]
    path = tmp_path / "SLDEM2015_512_30S_00S_315_360.tif"
    _write_geotiff(path, 336.0, -4.0, 337.0, -2.0, width=512, height=1024)
    digest = compute_sha256(path)
    report = validate_raster_metadata(path)
    crop = crop_region_window(path, bounds, "region_001",
                              "SLDEM2015_512_30S_00S_315_360", digest)
    stats = validate_crop_statistics(crop.elevation_m, crop.nodata)
    record = build_provenance_record(crop, report, stats)
    out = write_provenance_record(record, tmp_path / "prov")
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["region_id"] == "region_001"
    assert saved["source_product_tile"] == "SLDEM2015_512_30S_00S_315_360"
    assert saved["original_filename"] == path.name
    assert saved["sha256"] == digest
    assert saved["source_crs"] is not None
    assert saved["source_bounds"] is not None
    assert saved["crop_bounds"] is not None
    assert saved["crop_transform"] is not None
    assert saved["raster_dimensions"] is not None
    assert saved["elevation_units"] is not None
    assert saved["creation_timestamp_utc"] != ""
    assert saved["validation_status"] == "DEM_CROP_VALIDATED"
    assert float(saved["extra"]["crop_statistics"]["finite_pixel_fraction"]) > 0.9


def test_region_footprints_cover_six_regions():
    bounds = load_region_bounds(REPO_ROOT)
    assert sorted(bounds.keys()) == [
        "region_001", "region_002", "region_003", "region_004", "region_005", "region_006",
    ]
    assert bounds["region_001"]["west_lon"] == pytest.approx(336.484646)
    assert bounds["region_005"]["south_lat"] == pytest.approx(4.86317335)
