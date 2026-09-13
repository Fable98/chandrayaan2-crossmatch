"""Phase 14 — SLDEM2015 verified DEM ingestion & read-only hillshade preview.

ISOLATED DIAGNOSTIC MODULE. This module performs ingestion *validation* only:

* operator-supplied SLDEM2015 tile discovery (never downloads, never fabricates),
* SHA-256 checksum verification against an operator-supplied manifest,
* raster metadata validation (dimensions, CRS, transform, bounds, dtype, nodata),
* region-footprint coverage verification against ``manifest.json`` bounds,
* local window cropping (``rasterio.windows.from_bounds`` equivalent, no resampling),
* cropped elevation diagnostics (shape, min/max/mean/median, nodata fraction),
* read-only hillshade preview via the EXISTING
  :func:`matcher_cfog.render_synthetic_shaded_relief` (imported lazily, never modified),
* PDS4 solar-metadata interface (structure only — no recovery/download),
* immutable-style provenance records.

Hard scope boundaries (enforced by design, not just convention):

* This module never imports or touches the RF model, training code, ground-truth
  matches, candidate-generation defaults, or held-out evaluation groups.
* It never performs correspondence matching and never writes into the
  registration pipeline. Outputs are diagnostic files under ``evaluation*/`` only.
* ``dem_512.png`` (or any ``*.png``) is explicitly rejected as non-authoritative.
* Solar elevation is never guessed: regions without authoritative PDS4-derived
  elevation raise :class:`SolarElevationRequiredError` (``SOLAR_ELEVATION_REQUIRED``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "LRO-L-LOLA-4-GDR-V1.0"
REFERENCE_RADIUS_M = 1737400.0
EXPECTED_PPD = 512  # SLDEM2015 512 ppd, ~60 m posting at equator
DEG_PER_PX_512 = 1.0 / 512.0

SUPPORTED_SUFFIXES = (".jp2", ".tif", ".tiff", ".img")
DEM_DIR_ENV_VAR = "CH2X_DEM_DIR"
DEFAULT_DEM_DIRNAME = "data/external/dem"

# Crop is rejected (not filled/interpolated) when nodata exceeds this fraction.
EXCESSIVE_NODATA_FRACTION = 0.10

# Small tolerance (degrees) for floating-point bound containment checks.
BOUND_EPS_DEG = 1e-9

SOLAR_ELEVATION_REQUIRED = "SOLAR_ELEVATION_REQUIRED"

# Historical repository evidence for region_001 ONLY (from triplets.json entry 2).
# Labeled explicitly as historical_repository_metadata — NOT newly recovered truth.
REGION_001_HISTORICAL_SOLAR = {
    "ohrc_sun_elevation_deg": 14.063051,
    "tmc2_sun_elevation_deg": 48.753985,
    "ohrc_sun_azimuth_deg": 269.646098,
    "tmc2_sun_azimuth_deg": 108.866212,
    "provenance": "historical_repository_metadata",
    "source": "data_preprocessing_pipeline/triplets.json[1] (offline PDS4 label paths, XML not in repo)",
}

ALL_REGION_IDS = [
    "region_001",
    "region_002",
    "region_003",
    "region_004",
    "region_005",
    "region_006",
]

CODE_VERSION = "phase14-v1"


# ---------------------------------------------------------------------------
# Exceptions (each maps to a distinct, reportable rejection reason)
# ---------------------------------------------------------------------------


class DEMIngestionError(Exception):
    """Base class for Phase 14 ingestion validation failures."""


class DEMMissingError(DEMIngestionError):
    """Operator-supplied DEM file (or its expected checksum) is absent."""


class DEMChecksumMismatchError(DEMIngestionError):
    """Computed SHA-256 does not match the operator-supplied expectation."""


class DEMUnsupportedFileError(DEMIngestionError):
    """Unexpected filename or unsupported file type (incl. *.png ban)."""


class DEMInvalidMetadataError(DEMIngestionError):
    """Unreadable/corrupt raster, invalid CRS, or unusable raster metadata."""


class DEMCoverageError(DEMIngestionError):
    """A region footprint is not fully contained in the DEM tile."""


class DEMNodataError(DEMIngestionError):
    """Cropped window contains excessive nodata (never silently filled)."""


class SolarElevationRequiredError(DEMIngestionError):
    """No authoritative solar elevation available — rendering refused."""


# ---------------------------------------------------------------------------
# Small data structures
# ---------------------------------------------------------------------------


@dataclass
class PDS4Metadata:
    """Future authoritative PDS4 solar/observation metadata (interface only).

    Phase 14 does NOT recover or download PDS4 labels. This dataclass exists so
    that future authoritative labels can be supplied without changing DEM
    validation logic. ``solar_elevation_deg`` stays ``None`` until then.
    """

    source_image_id: str
    pds4_label_path: Optional[str] = None
    observation_time_utc: Optional[str] = None
    solar_azimuth_deg: Optional[float] = None
    solar_elevation_deg: Optional[float] = None
    spacecraft_position: Optional[str] = None
    target_body: str = "Moon"
    footprint: Optional[str] = None
    provenance: str = "pending_authoritative_pds4_label"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_authoritative_elevation(self) -> bool:
        try:
            return self.solar_elevation_deg is not None and bool(
                np.isfinite(float(self.solar_elevation_deg))
            )
        except (TypeError, ValueError):
            return False


@dataclass
class RasterMetadataReport:
    """Verbatim raster metadata discovered before any crop."""

    path: str
    driver: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    count: Optional[int] = None
    dtype: Optional[str] = None
    crs_string: Optional[str] = None
    transform: Optional[List[float]] = None
    bounds_west: Optional[float] = None
    bounds_south: Optional[float] = None
    bounds_east: Optional[float] = None
    bounds_north: Optional[float] = None
    res_x_deg: Optional[float] = None
    res_y_deg: Optional[float] = None
    pixels_per_degree: Optional[float] = None
    nodata: Any = None
    unit_conversion_applied: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DEMCropResult:
    """Local float32 elevation window with full geospatial bookkeeping."""

    region_id: str
    elevation_m: np.ndarray
    transform: Any  # affine.Affine
    crs: Any
    nodata: Any
    source_product_id: str
    source_filename: str
    source_sha256: str
    crop_bounds: Tuple[float, float, float, float]  # (west, south, east, north)
    unit_conversion_applied: str = "none (meters per SLDEM2015 reference)"

    def stats(self) -> Dict[str, Any]:
        return validate_crop_statistics(self.elevation_m, self.nodata)


@dataclass
class ProvenanceRecord:
    """Immutable-style provenance record for one validated DEM crop."""

    region_id: str
    source_dataset: str = DATASET_ID
    source_product_tile: str = ""
    original_filename: str = ""
    sha256: str = ""
    source_crs: Optional[str] = None
    source_bounds: Optional[List[float]] = None
    crop_bounds: Optional[List[float]] = None
    crop_transform: Optional[List[float]] = None
    raster_dimensions: Optional[List[int]] = None
    elevation_units: str = "meters relative to SLDEM2015 reference (R=1737400 m)"
    nodata_value: Any = None
    creation_timestamp_utc: str = ""
    validation_status: str = "pending"
    code_version: str = CODE_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Operator-supplied input contract
# ---------------------------------------------------------------------------


def resolve_dem_dir(cli_dir: Optional[str] = None, repo_root: Optional[Path] = None) -> Path:
    """Resolve the operator DEM directory: CLI > $CH2X_DEM_DIR > data/external/dem/."""
    if cli_dir:
        return Path(cli_dir).expanduser()
    env = os.environ.get(DEM_DIR_ENV_VAR)
    if env:
        return Path(env).expanduser()
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    return root / DEFAULT_DEM_DIRNAME


def load_dem_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    """Load the immutable operator manifest (raises DEMMissingError if absent)."""
    path = Path(manifest_path)
    if not path.is_file():
        raise DEMMissingError(f"DEM manifest not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DEMInvalidMetadataError(f"DEM manifest unreadable/corrupt: {path} ({exc})")


def expected_sha256_for_tile(tile_entry: Dict[str, Any]) -> Optional[str]:
    """Operator checksum: manifest value, overridden by the per-tile env var if set."""
    env_name = tile_entry.get("expected_sha256_env_override", "")
    if env_name and os.environ.get(env_name):
        return str(os.environ[env_name]).strip().lower()
    value = tile_entry.get("expected_sha256")
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def accepted_filenames_for_tile(tile_entry: Dict[str, Any]) -> List[str]:
    names = [tile_entry.get("expected_filename", "")]
    names.extend(tile_entry.get("accepted_alternates", []) or [])
    return [n for n in names if n]


def find_supplied_file(dem_dir: str | Path, tile_entry: Dict[str, Any]) -> Optional[Path]:
    """Locate the operator-supplied tile file (expected name first, then alternates)."""
    base = Path(dem_dir)
    for name in accepted_filenames_for_tile(tile_entry):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# 2. Checksum verification
# ---------------------------------------------------------------------------


def compute_sha256(path: str | Path) -> str:
    """Compute SHA-256 of a file (chunked; raises DEMMissingError if absent)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DEMMissingError(f"DEM file not found: {file_path}")
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: str | Path, expected_hex: Optional[str]) -> str:
    """Verify SHA-256; never silently accepts a modified tile.

    Raises:
        DEMMissingError: no expected checksum was supplied by the operator.
        DEMChecksumMismatchError: computed digest differs from expectation.
    """
    actual = compute_sha256(path)
    if expected_hex is None:
        raise DEMMissingError(
            f"EXPECTED_CHECKSUM_NOT_SUPPLIED for {Path(path).name}: operator must record "
            "SHA-256 in dem_manifest.json (or the per-tile env override) before validation."
        )
    if actual.lower() != str(expected_hex).strip().lower():
        raise DEMChecksumMismatchError(
            f"CHECKSUM_MISMATCH for {Path(path).name}: expected {expected_hex}, got {actual}. "
            "Modified tile rejected; re-download from the authoritative source."
        )
    return actual


def check_filename_and_type(path: str | Path, tile_entry: Dict[str, Any]) -> Path:
    """Reject unexpected filenames, unsupported types, and any *.png input."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".png" or "dem_512" in file_path.name:
        raise DEMUnsupportedFileError(
            f"REJECTED_NON_AUTHORITATIVE_INPUT: {file_path.name} — dem_512.png / *.png are "
            "synthesized optical products, never authoritative elevation."
        )
    if suffix not in SUPPORTED_SUFFIXES:
        raise DEMUnsupportedFileError(
            f"UNSUPPORTED_FILE_TYPE: {file_path.name} (suffix '{suffix}'). Supported: "
            f"{sorted(SUPPORTED_SUFFIXES)} (JP2 GIS-ready preferred; TIF/IMG mirrors accepted)."
        )
    if file_path.name not in accepted_filenames_for_tile(tile_entry):
        raise DEMUnsupportedFileError(
            f"UNEXPECTED_FILENAME: {file_path.name} is not the contracted tile "
            f"{tile_entry.get('tile_id')} (expected one of "
            f"{accepted_filenames_for_tile(tile_entry)})."
        )
    return file_path


# ---------------------------------------------------------------------------
# 3. Raster metadata validation
# ---------------------------------------------------------------------------


def _crs_is_acceptable(crs: Any) -> Tuple[bool, str]:
    """Accept geographic (degree) CRS or lunar-body projected CRS; reject Earth-projected.

    Rationale: SLDEM2015 GIS-ready JP2 + AUX.XML is cylindrical with 0–360 E
    degree axes. rasterio/GDAL often reports such Moon rasters with an Earth
    fallback ellipsoid, so the check targets axis units and the lunar radius
    (a ~= 1737400 m) rather than the ellipsoid name.
    """
    if crs is None:
        return False, "CRS_MISSING: raster carries no coordinate reference system."
    try:
        if bool(crs.is_geographic):
            return True, "geographic degree CRS"
    except Exception:
        pass
    try:
        proj4 = str(crs.to_proj4() or "")
        if "1737400" in proj4 or "+a=17374" in proj4:
            return True, "lunar-body CRS (radius ~= 1737400 m)"
    except Exception:
        pass
    try:
        wkt = str(crs.to_wkt() or "")
        if "1737400" in wkt or "17374" in wkt:
            return True, "lunar-body CRS (radius ~= 1737400 m)"
    except Exception:
        pass
    try:
        if bool(crs.is_projected):
            return False, (
                "INVALID_CRS: projected non-lunar CRS "
                f"({crs.to_string() if hasattr(crs, 'to_string') else crs}); expected a "
                "geographic-degree or lunar-body (R=1737400 m) reference."
            )
    except Exception:
        pass
    return False, f"INVALID_CRS: unusable coordinate reference ({crs})."


def validate_raster_metadata(path: str | Path) -> RasterMetadataReport:
    """Open with rasterio and report all metadata verbatim before any crop.

    No reprojection is performed here (or anywhere in Phase 14).
    Raises DEMInvalidMetadataError on corrupt/unreadable rasters or invalid CRS.
    """
    file_path = Path(path)
    try:
        import rasterio
    except ImportError as exc:
        raise DEMInvalidMetadataError(f"rasterio unavailable: {exc}")
    try:
        src = rasterio.open(str(file_path))
    except Exception as exc:
        raise DEMInvalidMetadataError(
            f"UNREADABLE_CORRUPT_RASTER: {file_path.name} could not be opened ({exc})."
        )
    with src:
        try:
            width, height, count = int(src.width), int(src.height), int(src.count)
            dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
            crs_string = str(src.crs) if src.crs else None
            transform = list(src.transform) if src.transform else None
            bounds = src.bounds
            nodata = src.nodata
            driver = src.driver
        except Exception as exc:
            raise DEMInvalidMetadataError(
                f"UNREADABLE_CORRUPT_RASTER: {file_path.name} metadata unreadable ({exc})."
            )
        ok, reason = _crs_is_acceptable(src.crs)
        if not ok:
            raise DEMInvalidMetadataError(f"{reason} File: {file_path.name}.")
        if width <= 0 or height <= 0:
            raise DEMInvalidMetadataError(
                f"INVALID_DIMENSIONS: {file_path.name} has width={width} height={height}."
            )
        res_x, res_y = None, None
        ppd = None
        try:
            res_x = abs(float(src.transform.a))
            res_y = abs(float(src.transform.e))
            if res_x > 0:
                ppd = 1.0 / res_x
        except Exception:
            pass
        report = RasterMetadataReport(
            path=str(file_path),
            driver=driver,
            width=width,
            height=height,
            count=count,
            dtype=dtype,
            crs_string=crs_string,
            transform=transform,
            bounds_west=float(bounds.left),
            bounds_south=float(bounds.bottom),
            bounds_east=float(bounds.right),
            bounds_north=float(bounds.top),
            res_x_deg=res_x,
            res_y_deg=res_y,
            pixels_per_degree=ppd,
            nodata=nodata,
            unit_conversion_applied="none (meters per SLDEM2015 reference; "
            "source elevation values expected in meters rel. R=1737400 m)",
        )
    # Sanity: tile bounds must live in a plausible lunar lon/lat range.
    if not (-1.0 <= (report.bounds_west or -999) <= 361.0 and -1.0 <= (report.bounds_east or 999) <= 361.0):
        raise DEMInvalidMetadataError(
            f"INVALID_BOUNDS: {file_path.name} longitude bounds "
            f"[{report.bounds_west}, {report.bounds_east}] outside 0–360 E convention."
        )
    if not (-90.5 <= (report.bounds_south or -999) <= 90.5 and -90.5 <= (report.bounds_north or 999) <= 90.5):
        raise DEMInvalidMetadataError(
            f"INVALID_BOUNDS: {file_path.name} latitude bounds "
            f"[{report.bounds_south}, {report.bounds_north}] outside planetocentric +-90."
        )
    if report.pixels_per_degree is not None and not (100.0 <= report.pixels_per_degree <= 1024.0):
        raise DEMInvalidMetadataError(
            f"INVALID_RESOLUTION: {file_path.name} resolves to {report.pixels_per_degree:.1f} ppd; "
            f"expected SLDEM2015 {EXPECTED_PPD} ppd (~60 m posting)."
        )
    return report


# ---------------------------------------------------------------------------
# Region footprints (read-only; manifests are never modified)
# ---------------------------------------------------------------------------


def load_region_bounds(repo_root: str | Path) -> Dict[str, Dict[str, float]]:
    """Read requested bounds for all six regions from processed_triplets manifests."""
    root = Path(repo_root)
    bounds: Dict[str, Dict[str, float]] = {}
    for region_id in ALL_REGION_IDS:
        manifest = root / "data_preprocessing_pipeline" / "processed_triplets" / region_id / "manifest.json"
        if not manifest.is_file():
            raise DEMMissingError(f"Region manifest not found: {manifest}")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        box = data.get("bounds") or data.get("bounds_optical")
        if not box:
            raise DEMInvalidMetadataError(f"No bounds in region manifest: {manifest}")
        bounds[region_id] = {
            "west_lon": float(box["west_lon"]),
            "east_lon": float(box["east_lon"]),
            "south_lat": float(box["south_lat"]),
            "north_lat": float(box["north_lat"]),
        }
    return bounds


# ---------------------------------------------------------------------------
# 4. Region coverage verification
# ---------------------------------------------------------------------------


def verify_region_coverage(
    dem_bounds: Tuple[float, float, float, float],
    region_bounds: Dict[str, float],
    res_x_deg: Optional[float] = None,
    res_y_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """Check full containment of a region footprint inside DEM bounds.

    Never expands bounds or substitutes another dataset — failure raises.
    """
    west, south, east, north = dem_bounds
    req = region_bounds
    contained = (
        west - BOUND_EPS_DEG <= req["west_lon"]
        and req["east_lon"] <= east + BOUND_EPS_DEG
        and south - BOUND_EPS_DEG <= req["south_lat"]
        and req["north_lat"] <= north + BOUND_EPS_DEG
    )
    margin_px: Dict[str, Optional[float]] = {
        "west": None,
        "east": None,
        "south": None,
        "north": None,
    }
    if res_x_deg and res_x_deg > 0:
        margin_px["west"] = (req["west_lon"] - west) / res_x_deg
        margin_px["east"] = (east - req["east_lon"]) / res_x_deg
    if res_y_deg and res_y_deg > 0:
        # rasterio bounds: bottom=south, top=north
        margin_px["south"] = (req["south_lat"] - south) / res_y_deg
        margin_px["north"] = (north - req["north_lat"]) / res_y_deg
    return {
        "requested_bounds": dict(req),
        "dem_bounds": [west, south, east, north],
        "contained": bool(contained),
        "margin_pixels": margin_px,
    }


def verify_all_coverage(
    dem_bounds: Tuple[float, float, float, float],
    regions: Dict[str, Dict[str, float]],
    res_x_deg: Optional[float] = None,
    res_y_deg: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    results = {
        region_id: verify_region_coverage(dem_bounds, box, res_x_deg, res_y_deg)
        for region_id, box in regions.items()
    }
    missing = [rid for rid, r in results.items() if not r["contained"]]
    if missing:
        raise DEMCoverageError(
            f"INSUFFICIENT_COVERAGE: region(s) {missing} not fully contained in DEM bounds "
            f"{list(dem_bounds)}. No expansion or substitution performed."
        )
    return results


# ---------------------------------------------------------------------------
# 5. Local window crop (isolated helper; no resampling)
# ---------------------------------------------------------------------------


def crop_region_window(
    path: str | Path,
    region_bounds: Dict[str, float],
    region_id: str,
    source_product_id: str,
    source_sha256: str,
) -> DEMCropResult:
    """Crop a region window with rasterio.windows.from_bounds (no resample)."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise DEMInvalidMetadataError(f"rasterio unavailable: {exc}")
    file_path = Path(path)
    try:
        src = rasterio.open(str(file_path))
    except Exception as exc:
        raise DEMInvalidMetadataError(f"UNREADABLE_CORRUPT_RASTER: {file_path.name} ({exc}).")
    with src:
        dem_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        check = verify_region_coverage(
            dem_bounds,
            region_bounds,
            abs(float(src.transform.a)) if src.transform else None,
            abs(float(src.transform.e)) if src.transform else None,
        )
        if not check["contained"]:
            raise DEMCoverageError(
                f"INSUFFICIENT_COVERAGE: {region_id} bounds {region_bounds} not fully "
                f"contained in {file_path.name} bounds {list(dem_bounds)}."
            )
        window = from_bounds(
            region_bounds["west_lon"],
            region_bounds["south_lat"],
            region_bounds["east_lon"],
            region_bounds["north_lat"],
            src.transform,
        )
        try:
            array = src.read(1, window=window)
        except Exception as exc:
            raise DEMInvalidMetadataError(
                f"CROP_FAILED: could not read window for {region_id} from {file_path.name} ({exc})."
            )
        window_transform = src.window_transform(window)
        crs = src.crs
        nodata = src.nodata
    elevation = np.asarray(array, dtype=np.float32)
    return DEMCropResult(
        region_id=region_id,
        elevation_m=elevation,
        transform=window_transform,
        crs=crs,
        nodata=nodata,
        source_product_id=source_product_id,
        source_filename=file_path.name,
        source_sha256=source_sha256,
        crop_bounds=(
            region_bounds["west_lon"],
            region_bounds["south_lat"],
            region_bounds["east_lon"],
            region_bounds["north_lat"],
        ),
    )


# ---------------------------------------------------------------------------
# 6. Cropped elevation diagnostics
# ---------------------------------------------------------------------------


def validate_crop_statistics(elevation: np.ndarray, nodata: Any) -> Dict[str, Any]:
    """Diagnostics only; excessive nodata fails instead of being filled."""
    arr = np.asarray(elevation, dtype=np.float64)
    total = int(arr.size)
    if total == 0:
        raise DEMNodataError("EMPTY_CROP: cropped elevation window has zero pixels.")
    finite_mask = np.isfinite(arr)
    if nodata is not None:
        try:
            finite_mask = finite_mask & (arr != float(nodata))
        except (TypeError, ValueError):
            pass
    n_finite = int(np.count_nonzero(finite_mask))
    finite_fraction = n_finite / max(total, 1)
    nodata_fraction = 1.0 - finite_fraction
    stats: Dict[str, Any] = {
        "cropped_shape": [int(arr.shape[0]), int(arr.shape[1]) if arr.ndim > 1 else 1],
        "nodata_fraction": float(nodata_fraction),
        "finite_pixel_fraction": float(finite_fraction),
        "nodata_value": nodata,
    }
    if n_finite == 0:
        raise DEMNodataError("ALL_NODATA_CROP: no finite elevation pixels in cropped window.")
    finite = arr[finite_mask]
    stats.update(
        {
            "min_elevation_m": float(np.min(finite)),
            "max_elevation_m": float(np.max(finite)),
            "mean_elevation_m": float(np.mean(finite)),
            "median_elevation_m": float(np.median(finite)),
        }
    )
    if nodata_fraction > EXCESSIVE_NODATA_FRACTION:
        raise DEMNodataError(
            f"EXCESSIVE_NODATA: nodata_fraction={nodata_fraction:.4f} exceeds "
            f"{EXCESSIVE_NODATA_FRACTION:.2f}; validation fails rather than filling/interpolating."
        )
    return stats


# ---------------------------------------------------------------------------
# 7. Read-only hillshade preview (diagnostic wrapper; production untouched)
# ---------------------------------------------------------------------------


def preview_hillshade(
    local_dem: np.ndarray,
    solar_azimuth_deg: Optional[float],
    solar_elevation_deg: Optional[float],
    working_gsd_m: float = 60.0,
    provenance_label: str = "pending_authoritative_pds4_label",
) -> np.ndarray:
    """Render a read-only hillshade preview via matcher_cfog (unmodified).

    Refuses with SOLAR_ELEVATION_REQUIRED when no valid elevation is supplied —
    elevation is never inferred from azimuth, brightness, or terrain.
    """
    if solar_elevation_deg is None:
        raise SolarElevationRequiredError(
            f"{SOLAR_ELEVATION_REQUIRED}: no authoritative solar elevation supplied "
            f"(provenance={provenance_label}); refusing to guess."
        )
    try:
        elev = float(solar_elevation_deg)
    except (TypeError, ValueError):
        raise SolarElevationRequiredError(
            f"{SOLAR_ELEVATION_REQUIRED}: non-numeric solar elevation "
            f"({solar_elevation_deg!r}, provenance={provenance_label})."
        )
    if not np.isfinite(elev):
        raise SolarElevationRequiredError(
            f"{SOLAR_ELEVATION_REQUIRED}: non-finite solar elevation "
            f"(provenance={provenance_label})."
        )
    if solar_azimuth_deg is None:
        raise SolarElevationRequiredError(
            f"{SOLAR_ELEVATION_REQUIRED}: solar azimuth missing "
            f"(provenance={provenance_label})."
        )
    try:
        azim = float(solar_azimuth_deg)
    except (TypeError, ValueError):
        raise SolarElevationRequiredError(
            f"{SOLAR_ELEVATION_REQUIRED}: non-numeric solar azimuth "
            f"(provenance={provenance_label})."
        )
    if not np.isfinite(azim):
        raise SolarElevationRequiredError(
            f"{SOLAR_ELEVATION_REQUIRED}: non-finite solar azimuth "
            f"(provenance={provenance_label})."
        )
    # Lazy import: keeps this diagnostic decoupled; production code unmodified.
    from matcher_cfog import render_synthetic_shaded_relief

    dem = np.asarray(local_dem, dtype=np.float64)
    if dem.ndim != 2 or dem.size == 0:
        raise DEMInvalidMetadataError("HILLSHADE_INPUT_INVALID: local DEM must be a non-empty 2D array.")
    shade = render_synthetic_shaded_relief(
        dem,
        target_azimuth_deg=azim,
        target_elevation_deg=elev,
        working_gsd_m=float(working_gsd_m),
    )
    return np.asarray(shade, dtype=np.float32)


# ---------------------------------------------------------------------------
# 10. Provenance registry
# ---------------------------------------------------------------------------


def build_provenance_record(
    crop: DEMCropResult,
    raster_report: RasterMetadataReport,
    stats: Dict[str, Any],
    validation_status: str = "DEM_CROP_VALIDATED",
) -> ProvenanceRecord:
    try:
        crop_transform = list(crop.transform) if crop.transform is not None else None
    except Exception:
        crop_transform = None
    shape = stats.get("cropped_shape", [0, 0])
    return ProvenanceRecord(
        region_id=crop.region_id,
        source_dataset=DATASET_ID,
        source_product_tile=crop.source_product_id,
        original_filename=crop.source_filename,
        sha256=crop.source_sha256,
        source_crs=str(crop.crs) if crop.crs is not None else None,
        source_bounds=[
            raster_report.bounds_west,
            raster_report.bounds_south,
            raster_report.bounds_east,
            raster_report.bounds_north,
        ],
        crop_bounds=list(crop.crop_bounds),
        crop_transform=crop_transform,
        raster_dimensions=[int(shape[0]), int(shape[1])],
        nodata_value=crop.nodata,
        creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        validation_status=validation_status,
        code_version=CODE_VERSION,
        extra={"crop_statistics": stats},
    )


def write_provenance_record(record: ProvenanceRecord, provenance_dir: str | Path) -> Path:
    out_dir = Path(provenance_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.region_id}_dem_provenance.json"
    out_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return out_path
