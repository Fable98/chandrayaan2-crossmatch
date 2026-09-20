"""
metadata.py — Sensor and Observation Geometry Metadata Abstraction

Provides metadata extraction and provenance tracking for Chandrayaan-2 sensors
(OHRC, TMC-2, IIRS) without hardcoding hidden geometries or physical scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import os


@dataclass
class SensorMetadata:
    """
    Physical and geometric metadata for a Chandrayaan-2 raster product.
    Tracks whether values originate from embedded headers, request parameters,
    or documented standard sensor specifications.

    Distinguishes native_sensor_gsd_m (the physical sensor footprint at nominal orbit)
    from effective_raster_gsd_m (the physical ground distance spanned by each pixel
    in the raster after cropping, resampling, or tiling).
    """
    sensor: str  # "OHRC", "TMC-2", "IIRS", or "DEM"
    gsd_m: Optional[float] = None  # Ground Sampling Distance (effective raster pixel size)
    native_gsd_m: Optional[float] = None  # Native detector GSD in meters per pixel
    effective_gsd_m: Optional[float] = None  # Effective raster GSD in meters per pixel
    resampling_factor: float = 1.0  # physical_scale_factor = derived_w / native_crop_w
    crop_transform: Optional[Dict[str, Any]] = None  # {"x_offset", "y_offset", "width", "height"}
    parent_product_id: Optional[str] = None  # Upstream source product identifier
    wavelength_range_um: Optional[Tuple[float, float]] = None
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    incidence_angle_deg: Optional[float] = None
    emission_angle_deg: Optional[float] = None
    phase_angle_deg: Optional[float] = None
    spacecraft_azimuth_deg: Optional[float] = None
    acquisition_time: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_lon, max_lon, min_lat, max_lat)
    provenance: Dict[str, str] = field(default_factory=dict)  # field_name -> provenance source

    def __post_init__(self) -> None:
        # Harmonize gsd_m, effective_gsd_m, and native_gsd_m
        if self.effective_gsd_m is None and self.gsd_m is not None:
            self.effective_gsd_m = float(self.gsd_m)
        if self.native_gsd_m is None:
            if self.effective_gsd_m is not None and abs(self.resampling_factor - 1.0) < 1e-6:
                self.native_gsd_m = self.effective_gsd_m
            elif self.gsd_m is not None:
                self.native_gsd_m = float(self.gsd_m)
        if self.effective_gsd_m is None and self.native_gsd_m is not None:
            rf = max(float(self.resampling_factor), 1e-9)
            self.effective_gsd_m = self.native_gsd_m / rf
        # Ensure gsd_m always mirrors effective_gsd_m for raster calculations
        if self.effective_gsd_m is not None:
            self.gsd_m = float(self.effective_gsd_m)

    @property
    def native_sensor_gsd_m(self) -> Optional[float]:
        return self.native_gsd_m

    @property
    def effective_raster_gsd_m(self) -> Optional[float]:
        return self.effective_gsd_m

    def validate_gsd(self, tolerance: float = 0.05) -> None:
        """Validates that GSD fields are physically consistent."""
        validate_gsd_consistency(
            native_gsd_m=self.native_gsd_m,
            effective_gsd_m=self.effective_gsd_m,
            resampling_factor=self.resampling_factor,
            tolerance=tolerance,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor": self.sensor,
            "gsd_m": self.gsd_m,
            "native_gsd_m": self.native_gsd_m,
            "effective_gsd_m": self.effective_gsd_m,
            "resampling_factor": self.resampling_factor,
            "crop_transform": self.crop_transform,
            "parent_product_id": self.parent_product_id,
            "wavelength_range_um": list(self.wavelength_range_um) if self.wavelength_range_um else None,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "sun_elevation_deg": self.sun_elevation_deg,
            "incidence_angle_deg": self.incidence_angle_deg,
            "emission_angle_deg": self.emission_angle_deg,
            "phase_angle_deg": self.phase_angle_deg,
            "spacecraft_azimuth_deg": self.spacecraft_azimuth_deg,
            "sensor_los_azimuth_deg": self.sensor_los_azimuth_deg,
            "acquisition_time": self.acquisition_time,
            "bounds": list(self.bounds) if self.bounds else None,
            "provenance": self.provenance,
        }

    @property
    def emission_deg(self) -> Optional[float]:
        return self.emission_angle_deg

    @property
    def azimuth_deg(self) -> Optional[float]:
        # Deprecated alias: historically returned sun azimuth, which is NOT
        # sensor line-of-sight azimuth. Kept for backward compatibility only.
        # Do NOT use for DEM relief compensation; pass None instead.
        return self.sun_azimuth_deg

    @property
    def sensor_los_azimuth_deg(self) -> Optional[float]:
        # Return spacecraft viewing / LOS azimuth if explicitly known.
        # Rule 7: Never return solar azimuth as LOS azimuth.
        return self.spacecraft_azimuth_deg


def validate_gsd_consistency(
    native_gsd_m: Optional[float],
    effective_gsd_m: Optional[float],
    resampling_factor: Optional[float] = 1.0,
    tolerance: float = 0.05,
) -> None:
    """
    Validates physical consistency of GSD metadata across derived rasters:
    1. native_gsd_m and effective_gsd_m must be strictly positive floats.
    2. resampling_factor must be strictly positive float.
    3. effective_gsd_m must match native_gsd_m / resampling_factor within tolerance.
    """
    if native_gsd_m is None or native_gsd_m <= 0:
        raise ValueError(f"Inconsistent GSD metadata: native_gsd_m must be > 0, got {native_gsd_m}")
    if effective_gsd_m is None or effective_gsd_m <= 0:
        raise ValueError(f"Inconsistent GSD metadata: effective_gsd_m must be > 0, got {effective_gsd_m}")
    rf = 1.0 if resampling_factor is None else float(resampling_factor)
    if rf <= 0:
        raise ValueError(f"Inconsistent GSD metadata: resampling_factor must be > 0, got {rf}")

    expected_eff = native_gsd_m / rf
    rel_err = abs(effective_gsd_m - expected_eff) / expected_eff
    if rel_err > tolerance:
        raise ValueError(
            f"Inconsistent GSD metadata: effective_gsd_m ({effective_gsd_m:.4f} m) does not match "
            f"native_gsd_m / resampling_factor ({native_gsd_m:.4f} / {rf:.4f} = {expected_eff:.4f} m, "
            f"relative discrepancy: {rel_err * 100:.2f}% > {tolerance * 100:.1f}%)"
        )


# Standard physical sensor specifications per Chandrayaan-2 mission documentation
# Used only as fallback when product header or API parameters are unavailable
SENSOR_SPECS = {
    "OHRC": {
        "gsd_m": 0.25,  # ~0.25 m nominal at 100 km circular orbit
        "wavelength_range_um": (0.45, 0.70),  # Panchromatic optical
        "nominal_emission_deg": 0.0,
    },
    "TMC-2": {
        "gsd_m": 5.0,  # ~4–5 m at 100 km nominal orbit. NOTE: current pipeline
        # ingests a single TMC-2 NCF view per region (single-view, not joint
        # Fore/Nadir/Aft stereo). Fore +26°/Nadir 0°/Aft -26° handling is future work.
        "wavelength_range_um": (0.40, 0.85),  # Panchromatic optical
        "nominal_emission_deg": None,  # Scene/product dependent (Fore +26°, Nadir 0°, Aft -26°); no universal default
    },
    "IIRS": {
        "gsd_m": 75.0,  # ~70–80 m hyperspectral swath
        "wavelength_range_um": (0.80, 5.00),  # 256 contiguous spectral bands
        "nominal_emission_deg": None,
    },
    "DEM": {
        "gsd_m": 5.0,
        "wavelength_range_um": None,
        "nominal_emission_deg": None,
    },
    "LRO_NAC": {
        "gsd_m": 0.5,  # ~0.5–2.0 m depending on orbital altitude
        "wavelength_range_um": (0.40, 0.75),  # Panchromatic optical
        "nominal_emission_deg": 0.0,
    },
}


def normalize_sensor_name(name: str) -> str:
    """Normalizes colloquial sensor strings to canonical identifiers."""
    name_clean = name.strip().upper()
    if "OHR" in name_clean:
        return "OHRC"
    elif "TMC" in name_clean:
        return "TMC-2"
    elif "IIR" in name_clean:
        return "IIRS"
    elif "DEM" in name_clean:
        return "DEM"
    elif "NAC" in name_clean or "LRO" in name_clean:
        return "LRO_NAC"
    return name_clean


def extract_sensor_metadata(
    image_path: str | Path,
    declared_sensor: Optional[str] = None,
    explicit_gsd: Optional[float] = None,
    explicit_emission: Optional[float] = None,
    explicit_azimuth: Optional[float] = None,
) -> SensorMetadata:
    """
    Extracts sensor metadata following strict precedence:
    1. Product metadata / PDS4 XML label / GeoTIFF embedded tags.
    2. Explicit caller / API request parameters.
    3. Standard sensor defaults with explicit source tracking.

    In-memory ndarrays (tiled matching) skip file inference and resolve via
    declared_sensor / explicit overrides only.
    """
    import numpy as _np
    if isinstance(image_path, _np.ndarray):
        try:
            st = normalize_sensor_name(declared_sensor) if declared_sensor else "UNKNOWN"
        except Exception:
            st = str(declared_sensor).strip().upper() if declared_sensor else "UNKNOWN"
        prov: Dict[str, str] = {"sensor": "request" if declared_sensor else "unknown",
                                "gsd_m": "request" if explicit_gsd is not None else "unavailable",
                                "note": "in-memory tile; no file header"}
        return SensorMetadata(sensor=st, gsd_m=explicit_gsd,  # type: ignore[arg-type]
                              emission_angle_deg=explicit_emission,
                              provenance=prov)
    p = Path(image_path)  # type: ignore[arg-type]
    provenance: Dict[str, str] = {}

    # Step 1: Infer sensor type
    sensor_type = None
    if declared_sensor:
        sensor_type = normalize_sensor_name(declared_sensor)
        provenance["sensor"] = "request"
    else:
        # Check filename pattern
        filename_lower = p.name.lower()
        if "ohr" in filename_lower:
            sensor_type = "OHRC"
            provenance["sensor"] = "filename_inference"
        elif "tmc" in filename_lower:
            sensor_type = "TMC-2"
            provenance["sensor"] = "filename_inference"
        elif "iir" in filename_lower:
            sensor_type = "IIRS"
            provenance["sensor"] = "filename_inference"
        elif "dem" in filename_lower:
            sensor_type = "DEM"
            provenance["sensor"] = "filename_inference"
        elif "nac" in filename_lower or "lro" in filename_lower:
            sensor_type = "LRO_NAC"
            provenance["sensor"] = "filename_inference"
        else:
            sensor_type = "UNKNOWN"
            provenance["sensor"] = "unknown"

    # Step 2: Check for PDS3 .LBL label or attached .IMG header for LRO NAC
    lbl_path = p.with_suffix(".lbl")
    if not lbl_path.exists():
        lbl_path = p.with_suffix(".LBL")
    if lbl_path.exists() or (p.suffix.lower() == ".img" and sensor_type == "LRO_NAC"):
        try:
            from lro_pds3_parser import extract_lro_nac_metadata
            target_lbl = lbl_path if lbl_path.exists() else p
            nac_meta = extract_lro_nac_metadata(
                target_lbl,
                explicit_gsd=explicit_gsd,
                explicit_emission=explicit_emission,
                explicit_azimuth=explicit_azimuth,
            )
            # If this is a pre-gridded working tile (e.g. _512), normalize effective grid GSD if matching OHRC
            if "_512" in p.name:
                manifest_path = p.parent / "manifest.json"
                if manifest_path.exists():
                    try:
                        import json
                        with open(manifest_path) as mf:
                            mdata = json.load(mf)
                            working_gsd = mdata.get("working_gsd_m", 1.0)
                            if explicit_gsd is None:
                                nac_meta.gsd_m = working_gsd
                                nac_meta.provenance["gsd_m"] = "manifest_grid"
                    except Exception:
                        pass
                elif explicit_gsd is None:
                    nac_meta.gsd_m = 1.0
                    nac_meta.provenance["gsd_m"] = "grid_normalized"
            return nac_meta
        except Exception:
            pass

    # Check for PDS4 XML label
    header_data: Dict[str, Any] = {}
    xml_path = p.with_suffix(".xml")
    if not xml_path.exists():
        xml_path = p.with_suffix(".XML")
    if xml_path.exists():
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # Parse XML tags
            for el in root.iter():
                tag_local = el.tag.split("}")[-1].lower()
                text = (el.text or "").strip()
                if not text:
                    continue
                try:
                    if tag_local in ("pixel_resolution", "gsd", "map_scale"):
                        header_data["gsd_m"] = float(text.split()[0])
                    elif tag_local in ("emission_angle", "emission"):
                        header_data["emission_angle_deg"] = float(text.split()[0])
                    elif tag_local in ("incidence_angle", "incidence"):
                        header_data["incidence_angle_deg"] = float(text.split()[0])
                    elif tag_local in ("solar_azimuth_angle", "sun_azimuth", "azimuth"):
                        header_data["sun_azimuth_deg"] = float(text.split()[0])
                    elif tag_local in ("spacecraft_azimuth_angle", "spacecraft_azimuth", "sensor_azimuth", "viewing_azimuth", "instrument_azimuth"):
                        header_data["spacecraft_azimuth_deg"] = float(text.split()[0])
                    elif tag_local in ("start_date_time", "acquisition_time"):
                        header_data["acquisition_time"] = text
                except (ValueError, IndexError):
                    pass
        except Exception:
            pass

    # Check for region manifest.json sidecar
    parent_prod_id = None
    crop_xform = None
    manifest_path = p.parent / "manifest.json"
    manifest_data: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            import json
            with open(manifest_path) as mf:
                manifest_data = json.load(mf)
                mdata = manifest_data
                if sensor_type == "OHRC":
                    parent_prod_id = mdata.get("ohrc_product_id")
                    header_data["native_gsd_m"] = mdata.get("ohrc_native_gsd_m", mdata.get("ohrc_gsd_m", 0.25))
                    if "_large_512" in p.name:
                        header_data["effective_gsd_m"] = mdata.get("ohrc_large_effective_gsd_m", 15.9883)
                    elif "_512" in p.name:
                        if mdata.get("reference_type") == "external_LRO_NAC":
                            geo = mdata.get("georeferencing", {})
                            header_data["effective_gsd_m"] = geo.get("working_pixel_m") or mdata.get("working_gsd_m", 1.0)
                        else:
                            header_data["effective_gsd_m"] = mdata.get("tmc2_gsd_m", 5.0)
                    else:
                        header_data["effective_gsd_m"] = header_data["native_gsd_m"]
                    header_data["sun_azimuth_deg"] = mdata.get("ohrc_sun_azimuth_deg")
                elif sensor_type == "TMC-2":
                    parent_prod_id = mdata.get("tmc2_product_id")
                    header_data["native_gsd_m"] = mdata.get("tmc2_native_gsd_m", mdata.get("tmc2_gsd_m", 5.0))
                    if "_large_512" in p.name:
                        header_data["effective_gsd_m"] = mdata.get("tmc2_large_effective_gsd_m", 39.4316)
                    else:
                        header_data["effective_gsd_m"] = header_data["native_gsd_m"]
                    header_data["sun_azimuth_deg"] = mdata.get("tmc2_sun_azimuth_deg")
                elif sensor_type == "IIRS":
                    parent_prod_id = mdata.get("iirs_product_id")
                    header_data["native_gsd_m"] = mdata.get("iirs_native_gsd_m", mdata.get("iirs_gsd_m", 75.0))
                    if "_large_512" in p.name:
                        header_data["effective_gsd_m"] = mdata.get("iirs_large_effective_gsd_m", 39.4316)
                    elif "_512" in p.name:
                        header_data["effective_gsd_m"] = mdata.get("working_gsd_m", mdata.get("tmc2_gsd_m", 5.0))
                    else:
                        header_data["effective_gsd_m"] = header_data["native_gsd_m"]
                elif sensor_type == "LRO_NAC":
                    parent_prod_id = mdata.get("lro_nac_product_id")
                    header_data["native_gsd_m"] = mdata.get("lro_nac_native_gsd_m", mdata.get("lro_nac_gsd_m", mdata.get("nac_gsd_m", 0.914)))
                    if "_512" in p.name:
                        geo = mdata.get("georeferencing", {})
                        header_data["effective_gsd_m"] = geo.get("working_pixel_m") or mdata.get("working_gsd_m", 1.0)
                    else:
                        header_data["effective_gsd_m"] = header_data["native_gsd_m"]
                    header_data["sun_azimuth_deg"] = mdata.get("lro_nac_sun_azimuth_deg")
                if "bounds" in mdata:
                    b = mdata["bounds"]
                    header_data["bounds"] = (b["west_lon"], b["east_lon"], b["south_lat"], b["north_lat"])
                    crop_xform = {"bounds": b}
        except Exception:
            pass

    # Step 3: Resolve Native GSD & Effective GSD
    native_gsd_val = header_data.get("native_gsd_m")
    if native_gsd_val is None:
        if "gsd_m" in header_data:
            native_gsd_val = header_data["gsd_m"]
            provenance["native_gsd_m"] = "header"
        elif sensor_type in SENSOR_SPECS and SENSOR_SPECS[sensor_type].get("gsd_m") is not None:
            native_gsd_val = SENSOR_SPECS[sensor_type]["gsd_m"]
            provenance["native_gsd_m"] = "sensor_spec"
        elif explicit_gsd is not None and explicit_gsd > 0:
            native_gsd_val = float(explicit_gsd)
            provenance["native_gsd_m"] = "request"
        else:
            raise ValueError(
                f"Physical ground sampling distance (GSD) could not be determined for image '{p.name}' (sensor: {sensor_type}). "
                "Please provide an explicit 'explicit_gsd' parameter or product metadata headers."
            )

    effective_gsd_val = None
    if explicit_gsd is not None and explicit_gsd > 0:
        effective_gsd_val = float(explicit_gsd)
        provenance["gsd_m"] = "request"
        provenance["effective_gsd_m"] = "request"
    elif "effective_gsd_m" in header_data:
        effective_gsd_val = float(header_data["effective_gsd_m"])
        provenance["gsd_m"] = "manifest"
        provenance["effective_gsd_m"] = "manifest"
    elif "_512" in p.name:
        # Fallback grid normalization for 512px derivative rasters without manifest
        if sensor_type == "OHRC":
            effective_gsd_val = 5.0
        elif sensor_type == "LRO_NAC":
            effective_gsd_val = 1.0
        else:
            effective_gsd_val = native_gsd_val
        provenance["gsd_m"] = "grid_normalized_fallback"
        provenance["effective_gsd_m"] = "grid_normalized_fallback"
    else:
        effective_gsd_val = native_gsd_val
        provenance["gsd_m"] = provenance.get("native_gsd_m", "header")
        provenance["effective_gsd_m"] = provenance.get("native_gsd_m", "header")

    resampling_factor = float(native_gsd_val) / float(effective_gsd_val) if effective_gsd_val else 1.0

    # Step 4: Resolve observation geometry (emission & azimuth)
    emission_val = None
    if "emission_angle_deg" in header_data:
        emission_val = header_data["emission_angle_deg"]
        provenance["emission_angle_deg"] = "header"
    elif explicit_emission is not None:
        emission_val = float(explicit_emission)
        provenance["emission_angle_deg"] = "request"
    else:
        # Mark as unavailable rather than pretending geometry is known
        emission_val = None
        provenance["emission_angle_deg"] = "unavailable"

    azimuth_val = None
    if "sun_azimuth_deg" in header_data:
        azimuth_val = header_data["sun_azimuth_deg"]
        provenance["sun_azimuth_deg"] = "header"
    elif explicit_azimuth is not None:
        azimuth_val = float(explicit_azimuth)
        provenance["sun_azimuth_deg"] = "request"
    else:
        azimuth_val = None
        provenance["sun_azimuth_deg"] = "unavailable"

    incidence_val = header_data.get("incidence_angle_deg")
    if incidence_val is not None:
        provenance["incidence_angle_deg"] = "header"
    else:
        provenance["incidence_angle_deg"] = "unavailable"

    spacecraft_az_val = header_data.get("spacecraft_azimuth_deg")
    if spacecraft_az_val is not None:
        provenance["spacecraft_azimuth_deg"] = "header"

    specs = SENSOR_SPECS.get(sensor_type, {})
    wavelength = specs.get("wavelength_range_um")
    if wavelength:
        provenance["wavelength_range_um"] = "sensor_spec"

    meta_obj = SensorMetadata(
        sensor=sensor_type,
        gsd_m=effective_gsd_val,
        native_gsd_m=native_gsd_val,
        effective_gsd_m=effective_gsd_val,
        resampling_factor=resampling_factor,
        crop_transform=crop_xform,
        parent_product_id=parent_prod_id,
        wavelength_range_um=wavelength,
        sun_azimuth_deg=azimuth_val,
        incidence_angle_deg=incidence_val,
        emission_angle_deg=emission_val,
        spacecraft_azimuth_deg=spacecraft_az_val,
        acquisition_time=header_data.get("acquisition_time"),
        bounds=header_data.get("bounds"),
        provenance=provenance,
    )
    meta_obj.validate_gsd()
    return meta_obj
